
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

from moviepy.editor import VideoFileClip, concatenate_videoclips
import functools
import gc
import logging
import math
import os,sys
import random
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import time,einops
import accelerate
import diffusers
import numpy as np
import open_clip
import torch
from utils.pickscore import PickScore
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AnimateDiffPipeline,
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    LCMScheduler,
    MotionAdapter,
    StableDiffusionPipeline,
    TextToVideoSDPipeline,
    UNet2DConditionModel,
    UNet3DConditionModel,
    UNetMotionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, export_to_video, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from tabulate import tabulate
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize, RandomCrop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig

from args import parse_args
from dataset.webvid_dataset_wbd import Text2VideoDataset
from models.discriminator_handcraft import (
    ProjectedDiscriminator,
    get_dino_features,
    preprocess_dino_input,
)

from utils.diffusion_misc import *
from utils.dist import dist_init, dist_init_wo_accelerate, get_deepspeed_config
from utils.misc import *
from utils.wandb import setup_wandb

MAX_SEQ_LENGTH = 77
args = parse_args()
WEBVID_DATA_SIZE = 2467378

dataset = Text2VideoDataset(
        args.dataset_path,
        num_train_examples=args.max_train_samples or int(WEBVID_DATA_SIZE * (args.web_dataset_end - args.web_dataset_begin) / 80),
        per_gpu_batch_size=args.train_batch_size,
        global_batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        duration=args.num_frames,
        frame_interval=args.frame_interval,
        frame_sel=args.frame_sel,
        resolution=args.resolution,
        shuffle_buffer_size=1000,
        pin_memory=True,
        persistent_workers=True,
        pixel_mean=[0.5, 0.5, 0.5],
        pixel_std=[0.5, 0.5, 0.5],
        begin = args.web_dataset_begin,
        end = args.web_dataset_end,
    )
train_dataloader = dataset.train_dataloader

def save_video_tensor(video_tensor, file_path, fps=25):
    """
    将视频 Tensor 保存为 MP4 文件
    :param video_tensor: 视频 Tensor
    :param file_path: 保存路径
    :param fps: 帧率
    """
    # 先将 Tensor 转换为 [T, H, W, C] 形式
    video_tensor = video_tensor.permute(0, 2, 3, 1)
    import torchvision
    # 使用 torchvision.io.write_video 保存视频
    torchvision.io.write_video(file_path, video_tensor, fps)

# 循环处理每个视频
for i, dd in enumerate(train_dataloader):
    video_tensor = dd["video"][0].permute(1,0,2,3)
    video_tensor = ((video_tensor / 2 + 0.5 ) * 255).int()
    print(video_tensor.min(), video_tensor.max(), dd["text"])
    
    # 定义临时视频文件路径
    temp_video_path = f"temp_video_{i}.mp4"
    
    # 保存视频 Tensor 为临时文件
    save_video_tensor(video_tensor, temp_video_path)
    
    # 使用 moviepy 重新编码并保存为最终文件
    clip = VideoFileClip(temp_video_path)
    output_path = f"output_video_{i}.mp4"
    clip.write_videofile(output_path, codec='libx264')
    clip.close()

    # 删除临时文件
    os.remove(temp_video_path)

    print(f"Video {i} saved as {output_path}")