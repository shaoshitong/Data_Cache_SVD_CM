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

if is_wandb_available():
    import wandb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(name)s] - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


def save_to_local(save_dir: str, prompt: str, video):
    if len(prompt) > 256:
        prompt = prompt[:256]
    prompt = prompt.replace(" ", "_")
    logger.info(f"Saving images to {save_dir}")

    export_to_video(video, os.path.join(save_dir, f"{prompt}.mp4"))



def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs


def get_predicted_original_sample(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0

def get_predicted_noise(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == "v_prediction":
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_epsilon


def load_stable_diffusion(device,dtype):
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe = pipe.to(device=device,dtype=dtype)
    del pipe.unet
    
    def compute_embeddings(
        prompt_batch
    ):
        prompt_embeds = pipe.encode_prompt(
            prompt_batch, device,1, True)[0]
        return prompt_embeds

    return compute_embeddings
    
def main(args):
    # torch.multiprocessing.set_sharing_strategy("file_system")
    dist_init()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    setup_wandb()

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
        # deepspeed_plugin=deepspeed_plugin,
    )
    
    local_rank = torch.distributed.get_rank()
    rank_number = torch.cuda.device_count()
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    # Make one log on every process with the configuration for debugging.
    logger.info("Printing accelerate state", main_process_only=False)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_batch_size / 128
        args.disc_learning_rate = (
            args.disc_learning_rate * total_batch_size * args.disc_tsn_num_frames / 128
        )
        logger.info(f"Scaling learning rate to {args.learning_rate}")
        logger.info(f"Scaling discriminator learning rate to {args.disc_learning_rate}")

    sorted_args = sorted(vars(args).items())
    logger.info(
        "\n" + tabulate(sorted_args, headers=["key", "value"], tablefmt="rounded_grid"),
        main_process_only=True,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
                private=True,
            ).repo_id

    try:
        accelerator.wait_for_everyone()
    except Exception as e:
        logger.error(f"Failed to wait for everyone: {e}")
        dist_init_wo_accelerate()
        accelerator.wait_for_everyone()

    # 1. Create the noise scheduler and the desired noise schedule.
    try:
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="scheduler",
            revision=args.teacher_revision,
            rescale_betas_zero_snr=True if args.zero_snr else False,
            beta_schedule=args.beta_schedule,
        )
    except Exception as e:
        logger.error(f"Failed to load the noise scheduler: {e}")
        logger.info("Switching to online pretrained checkpoint")
        args.pretrained_teacher_model = args.online_pretrained_teacher_model
        args.motion_adapter_path = args.online_motion_adapter_path
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="scheduler",
            revision=args.teacher_revision,
            rescale_betas_zero_snr=True if args.zero_snr else False,
            beta_schedule=args.beta_schedule,
        )

    # DDPMScheduler calculates the alpha and sigma noise schedules (based on the alpha bars) for us
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    # Initialize the DDIM ODE solver for distillation.
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
    )

    # 2. Load tokenizers from SD 1.X/2.X checkpoint. TAG-Pretrain
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="tokenizer",
        revision=args.teacher_revision,
        use_fast=False,
    )

    # 3. Load text encoders from SD 1.X/2.X checkpoint. TAG-Pretrain
    # import correct text encoder classes
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="text_encoder",
        revision=args.teacher_revision,
    )

    # 4. Load VAE from SD 1.X/2.X checkpoint. TAG-Pretrain
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="vae",
        revision=args.teacher_revision,
    )

    # 5. Load teacher U-Net from SD 1.X/2.X checkpoint. TAG-Pretrain
    if args.base_model_name == "animatediff":
        teacher_unet = UNet2DConditionModel.from_pretrained( 
            args.pretrained_teacher_model,
            subfolder="unet",
            revision=args.teacher_revision,
        )
        teacher_motion_adapter = MotionAdapter.from_pretrained(args.motion_adapter_path)
        teacher_unet = UNetMotionModel.from_unet2d(teacher_unet, teacher_motion_adapter)
    elif args.base_model_name == "modelscope":
        teacher_unet = UNet3DConditionModel.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="unet",
            revision=args.teacher_revision,
        )

    # 5.2 Load sentence-level CLIP. TAG-Pretrain
    open_clip_model, _, preprocesses = open_clip.create_model_and_transforms(
        "ViT-g-14",
        pretrained="weights/open_clip_pytorch_model.bin",
    )
    open_clip_tokenizer = open_clip.get_tokenizer("ViT-g-14")

    # 6. Freeze teacher vae, text_encoder, and teacher_unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_unet.requires_grad_(False)
    open_clip_model.requires_grad_(False)

    # 7. Create online student U-Net. TAG-Pretrain
    # For whole model fine-tuning, this will be updated by the optimizer (e.g.,
    # via backpropagation.)
    # Add `time_cond_proj_dim` to the student U-Net if `teacher_unet.config.time_cond_proj_dim` is None
    if args.use_lora:
        if args.base_model_name == "animatediff":
            unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_teacher_model,
                subfolder="unet",
                revision=args.teacher_revision,
            )
            motion_adapter = MotionAdapter.from_pretrained(args.motion_adapter_path)
            unet = UNetMotionModel.from_unet2d(unet, motion_adapter)
        elif args.base_model_name == "modelscope":
            unet = UNet3DConditionModel.from_pretrained(
                args.pretrained_teacher_model,
                subfolder="unet",
                revision=args.teacher_revision,
            )
    else:
        assert (
            args.base_model_name == "animatediff"
        ), f"Please use LoRA for {args.base_model_name}"

        time_cond_proj_dim = (
            teacher_unet.config.time_cond_proj_dim
            if "time_cond_proj_dim" in teacher_unet.config
            and teacher_unet.config.time_cond_proj_dim is not None
            else args.unet_time_cond_proj_dim
        )
        if args.base_model_name == "animatediff":
            unet = UNetMotionModel.from_config(
                teacher_unet.config, time_cond_proj_dim=time_cond_proj_dim
            )

            # 8. Create target student U-Net. This will be updated via EMA updates (polyak averaging).
            # Initialize from (online) unet
            target_unet = UNetMotionModel.from_config(
                teacher_unet.config, time_cond_proj_dim=time_cond_proj_dim
            )
        elif args.base_model_name == "modelscope":

            unet = UNet3DConditionModel.from_config(
                teacher_unet.config, time_cond_proj_dim=time_cond_proj_dim
            )

            # 8. Create target student U-Net. This will be updated via EMA updates (polyak averaging).
            # Initialize from (online) unet
            target_unet = UNet3DConditionModel.from_config(
                teacher_unet.config, time_cond_proj_dim=time_cond_proj_dim
            )
        # load teacher_unet weights into unet
        unet.load_state_dict(teacher_unet.state_dict(), strict=False)
        
        target_unet.load_state_dict(unet.state_dict())
        target_unet.train()
        unet.train()
        target_unet.requires_grad_(False)
        unet.requires_grad_(False)

        # count trainable parameters
        trainable_params = 0
        all_param = 0
        for _, param in unet.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    if args.prev_train_unet is not None and args.prev_train_unet != "None" and os.path.exists(args.prev_train_unet):
        lora = UNet3DConditionModel.from_pretrained(
        args.prev_train_unet,
        torch_device="cpu")
        unet = lora
        print(f"Successfully load unet from {args.prev_train_unet}")
    
    if args.prev_teacher_unet is not None and args.prev_teacher_unet != "None" and os.path.exists(args.prev_teacher_unet):
        teacher_unet = UNet3DConditionModel.from_pretrained(
        args.prev_teacher_unet,
        torch_device="cpu")
        print(f"Successfully load teacher unet from {args.prev_teacher_unet}")
    
    
    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )
    
    # 8.1. Create discriminator for the student U-Net.
    c_dim = 1024

    # 9. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device)
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    open_clip_model.to(accelerator.device)

    # Move teacher_unet to device, optionally cast to weight_dtype
    if not args.use_lora:
        target_unet.to(accelerator.device)
    teacher_unet.to(accelerator.device)
    if args.cast_teacher_unet:
        teacher_unet.to(dtype=weight_dtype)
    unet.to(device=accelerator.device,dtype=weight_dtype)
    unet.requires_grad_(False)
    
    # Also move the alpha and sigma noise schedules to accelerator.device.
    alpha_schedule = alpha_schedule.to(accelerator.device)
    sigma_schedule = sigma_schedule.to(accelerator.device)
    # Move the ODE solver to accelerator.device.
    solver = solver.to(accelerator.device)

    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        if args.use_lora:

            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    unet_ = accelerator.unwrap_model(unet)
                    lora_state_dict = get_peft_model_state_dict(
                        unet_, adapter_name="default"
                    )
                    StableDiffusionPipeline.save_lora_weights(
                        os.path.join(output_dir, "unet_lora"), lora_state_dict
                    )
                    # save weights in peft format to be able to load them back
                    unet_.save_pretrained(output_dir)

                    for _, model in enumerate(models):
                        # make sure to pop weight so that corresponding model is not saved again
                        if len(weights) > 0:
                            weights.pop()

        else:
            # only support finetune motion module for AnimateDiff
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    target_unet_ = accelerator.unwrap_model(target_unet)
                    target_unet_.save_motion_modules(
                        os.path.join(output_dir, "target_motion_modules")
                    )

                    unet_ = accelerator.unwrap_model(unet)
                    unet_.save_motion_modules(
                        os.path.join(output_dir, "motion_modules")
                    )

                    for i, model in enumerate(models):
                        # make sure to pop weight so that corresponding model is not saved again
                        if len(weights) > 0:
                            weights.pop()

        if args.use_lora:

            def load_model_hook(models, input_dir):
                # load the LoRA into the model
                unet_ = accelerator.unwrap_model(unet)
                unet_.load_adapter(
                    input_dir, "default", is_trainable=True, torch_device="cpu"
                )

                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    models.pop()

        else:
            # only support finetune motion module for AnimateDiff
            def load_model_hook(models, input_dir):
                target_motion_module = MotionAdapter.from_pretrained(
                    os.path.join(input_dir, "target_motion_modules")
                )
                target_unet.load_motion_modules(target_motion_module)
                del target_motion_module

                student_motion_module = MotionAdapter.from_pretrained(
                    os.path.join(input_dir, "motion_modules")
                )
                unet_ = accelerator.unwrap_model(unet)
                unet_.load_motion_modules(student_motion_module)
                del student_motion_module

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # # load diffusers style into model
                    # load_model = UNet3DConditionModel.from_pretrained(
                    #     input_dir, subfolder="unet"
                    # )
                    # model.register_to_config(**load_model.config)

                    # model.load_state_dict(load_model.state_dict())
                    # del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # 11. Enable optimizations
    if (
        args.enable_xformers_memory_efficient_attention
        and args.base_model_name != "animatediff"
    ):
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            teacher_unet.enable_xformers_memory_efficient_attention()
            if not args.use_lora:
                target_unet.enable_xformers_memory_efficient_attention()
        else:
            logger.warning(
                "xformers is not available. Make sure it is installed correctly"
            )
            # raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # 13. Dataset creation and data processing
    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(
        prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True
    ):
        prompt_embeds = encode_prompt(
            prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train
        )
        return prompt_embeds

    WEBVID_DATA_SIZE = 2467378

    dataset = Text2VideoDataset(
        args.dataset_path,
        num_train_examples=args.max_train_samples or int(WEBVID_DATA_SIZE * (args.web_dataset_end - args.web_dataset_begin) / 80),
        per_gpu_batch_size=args.train_batch_size,
        global_batch_size=args.train_batch_size * accelerator.num_processes,
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
    
    if args.disc_gt_data == "webvid":
        disc_gt_dataloader = None
    elif args.disc_gt_data in ["laion", "disney", "realisticvision", "toonyou"]:
        if args.disc_gt_data == "laion":
            from dataset.laion_dataset_wbd_modified import Text2ImageDataset, cycle

            num_image_train_examples = (
                int(args.max_train_samples * args.disc_tsn_num_frames)
                if args.max_train_samples is not None
                else int(WEBVID_DATA_SIZE * args.disc_tsn_num_frames)
            )
        else:
            from dataset.custom_dataset_wbd import Text2ImageDataset, cycle

            num_image_train_examples = min(
                (
                    int(args.max_train_samples * args.disc_tsn_num_frames)
                    if args.max_train_samples is not None
                    else int(WEBVID_DATA_SIZE * args.disc_tsn_num_frames)
                ),
                478976,
            )

        disc_gt_dataset = Text2ImageDataset(
            args.disc_gt_data_path,
            num_train_examples=num_image_train_examples,
            per_gpu_batch_size=int(args.train_batch_size * args.disc_tsn_num_frames),
            global_batch_size=int(
                args.train_batch_size
                * accelerator.num_processes
                * args.disc_tsn_num_frames
            ),
            num_workers=args.dataloader_num_workers,
            resolution=args.resolution,
            shuffle_buffer_size=1000,
            pin_memory=True,
            persistent_workers=True,
            pixel_mean=[0.5, 0.5, 0.5],
            pixel_std=[0.5, 0.5, 0.5],
            begin = args.web_dataset_begin,
            end = args.web_dataset_end,
        )
        disc_gt_dataloader = disc_gt_dataset.train_dataloader
        disc_gt_dataloader = cycle(disc_gt_dataloader)
    else:
        raise ValueError(
            f"Discriminator ground truth data {args.disc_gt_data} is not supported."
        )

    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    # 14. LR Scheduler creation
    # Scheduler and math around the number of training steps.
    args.gradient_accumulation_steps = 1
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        train_dataloader.num_batches / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        train_dataloader.num_batches / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # remove list objects to avoid bug in tensorboard
        tracker_config = {
            k: v for k, v in vars(args).items() if not isinstance(v, list)
        }
        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs={"wandb": {"name": args.tracker_run_name}},
        )

    uncond_input_ids = tokenizer(
        [""] * args.train_batch_size,
        return_tensors="pt",
        padding="max_length",
        max_length=77,
    ).input_ids.to(accelerator.device)
    uncond_prompt_embeds = text_encoder(uncond_input_ids)[0]
    stable_v15_encode_prompt = load_stable_diffusion(dtype=weight_dtype,device=accelerator.device)
    # Memory 22G per single GPU
    
    # 16. Through Train to Save Data!
    logger.info("***** Running training (saving data for cache)*****")
    logger.info(f"  Num batches each epoch = {train_dataloader.num_batches}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(
        f"  Num learnable parameters = {sum([p.numel() for p in unet.parameters() if p.requires_grad]) / 1e6} M"
    )
    global_step = 0
    first_epoch = 0
    saved_data_idx = 0
    extract_code_dir = args.extract_code_dir
    if not os.path.exists(extract_code_dir):
        os.makedirs(extract_code_dir, exist_ok=True)
    rank_extract_code_dir = os.path.join(extract_code_dir, 'rank{:02d}'.format(local_rank))
    if not os.path.exists(rank_extract_code_dir):
        os.makedirs(rank_extract_code_dir, exist_ok=True)
    gc.collect()
    torch.cuda.empty_cache()
    initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    right_global_step = 0
    
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # if os.path.exists(os.path.join(rank_extract_code_dir,  'cache{:03d}.npz'.format(saved_data_idx))):
            #     saved_data_idx += 1
            #     progress_bar.update(1)
            #     global_step += 1
            #     continue 
            if global_step % 500 == 0:
                print("Pass Rate:",round((global_step-right_global_step)*100/(global_step+1e-3),2),"%")
            store_dict = dict()
            with torch.no_grad():
                # 1. Load and process the image and text conditioning
                video, text = batch["video"], batch["text"]
                video = video.to(accelerator.device, non_blocking=True)
                encoded_text = compute_embeddings_fn(text)
                store_dict["text"] = text
                
                pixel_values = video.to(dtype=weight_dtype)
                if vae.dtype != weight_dtype:
                    vae.to(dtype=weight_dtype)

                # encode pixel values with batch size of at most args.vae_encode_batch_size
                pixel_values = rearrange(pixel_values, "b c t h w -> (b t) c h w")
                latents = []
                for i in range(0, pixel_values.shape[0], args.vae_encode_batch_size):
                    latents.append(
                        vae.encode(
                            pixel_values[i : i + args.vae_encode_batch_size]
                        ).latent_dist.sample()
                    )
                latents = torch.cat(latents, dim=0)
                
                latents = rearrange(
                    latents,
                    "(b t) c h w -> b c t h w",
                    b=args.train_batch_size,
                    t=args.num_frames,
                )

                latents = latents * vae.config.scaling_factor
                latents = latents.to(weight_dtype)
                save_latents = latents
                bsz = latents.shape[0]

                # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
                # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
                topk = (
                    noise_scheduler.config.num_train_timesteps
                    // args.num_ddim_timesteps
                )
                index = torch.randint(
                    0, args.num_ddim_timesteps, (bsz,), device=latents.device
                ).long() ## NEED ##
                
                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - topk
                timesteps = torch.where(
                    timesteps < 0, torch.zeros_like(timesteps), timesteps
                )

                # 3. Get boundary scalings for start_timesteps and (end) timesteps.
                c_skip_start, c_out_start = scalings_for_boundary_conditions(
                    start_timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip_start, c_out_start = [
                    append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]
                ]
                c_skip, c_out = scalings_for_boundary_conditions(
                    timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

                # 4. Sample a random guidance scale w from U[w_min, w_max] and embed it
                # Note that for LCM-LoRA distillation it is not necessary to use a guidance scale embedding
                w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                if not args.use_lora:
                    w_embedding = guidance_scale_embedding(
                        w, embedding_dim=time_cond_proj_dim
                    )
                    w_embedding = w_embedding.to(
                        device=latents.device, dtype=latents.dtype
                    )
                w = w.reshape(bsz, 1, 1, 1, 1)
                # Move to U-Net device and dtype
                w = w.to(device=latents.device, dtype=latents.dtype)

                # if use predicted x_0, use the caption from the disc gt dataset
                # instead of from WebVid
                use_pred_x0 = False
                if global_step >= args.disc_start_step and not args.no_disc:
                    if args.cd_pred_x0_portion >= 0:
                        use_pred_x0 = random.random() < args.cd_pred_x0_portion

                # get CLIP embeddings, which is used for the adversarial loss
                with torch.no_grad():
                    clip_text_token = open_clip_tokenizer(text).to(accelerator.device)
                    clip_emb = open_clip_model.encode_text(clip_text_token)
                    sd_prompt_embeds = stable_v15_encode_prompt(text).to(accelerator.device)
                # 5. Prepare prompt embeds and unet_added_conditions
                prompt_embeds = encoded_text.pop("prompt_embeds")
                store_dict["prompt_embeds"] = prompt_embeds
                store_dict["sd_prompt_embeds"] = sd_prompt_embeds

                # 6. Sample noise from the prior and add it to the latents according to the noise magnitude at each
                # timestep (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
                if use_pred_x0:
                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=weight_dtype):
                            noise = torch.randn_like(latents)
                            last_timestep = solver.ddim_timesteps[-1].unsqueeze(0)
                            last_timestep = last_timestep.repeat(bsz)
                            if args.use_lora:
                                x_0_noise_pred = unet(
                                    noise.to(dtype=weight_dtype),
                                    last_timestep.to(dtype=weight_dtype),
                                    timestep_cond=None,
                                    encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype),
                                ).sample
                            else:
                                x_0_noise_pred = target_unet(
                                    noise.to(dtype=weight_dtype),
                                    last_timestep.to(dtype=weight_dtype),
                                    timestep_cond=w_embedding,
                                    encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype),
                                ).sample
                            latents = get_predicted_original_sample( # Only a single step
                                x_0_noise_pred,
                                last_timestep,
                                noise,
                                noise_scheduler.config.prediction_type,
                                alpha_schedule,
                                sigma_schedule,
                            )
                
                ################### Data-Centric #####################
                pre_image = \
                vae.decode(einops.rearrange(latents.half().cuda(),"b c t h w -> (b t) c h w")[5].unsqueeze(0) / vae.config.scaling_factor).sample
                pre_image = pre_image[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                pre_image = Image.fromarray(pre_image)
                image_emb = open_clip_model.encode_image(preprocesses(pre_image).unsqueeze(0).to(accelerator.device))
                image_features = image_emb / image_emb.norm(dim=-1, keepdim=True) 
                text_features = clip_emb / clip_emb.norm(dim=-1, keepdim=True) 
                text_probs = (100.0 * image_features @ text_features.T)
                score = text_probs * max(start_timesteps.item() / solver.ddim_timesteps[-1], 0.5)
                
                if score < 7.5:
                    progress_bar.update(1)
                    global_step += 1
                    right_global_step += 1
                    continue
                ################### Data-Centric #####################
                
                noise = torch.randn_like(latents).to(dtype=weight_dtype)
                noisy_model_input_list = []
                for b_idx in range(bsz): # Add noise
                    if index[b_idx] != args.num_ddim_timesteps - 1:
                        noisy_model_input = noise_scheduler.add_noise(
                            latents[b_idx, None],
                            noise[b_idx, None],
                            start_timesteps[b_idx, None],
                        )
                    else:
                        # hard swap input to pure noise to ensure zero terminal SNR
                        noisy_model_input = noise[b_idx, None]
                    noisy_model_input_list.append(noisy_model_input)
                noisy_model_input = torch.cat(noisy_model_input_list, dim=0).half()
                
                store_dict["noisy_model_input"] = noisy_model_input
                store_dict["use_pred_x0"] = use_pred_x0
                store_dict["start_timesteps"] = start_timesteps
                store_dict["timesteps"] = timesteps
                store_dict["latents"] = rearrange(
                    save_latents,
                    "b c t h w -> (b t) c h w",
                )
                #######################################################################
                ### NEED noisy_model_input, use_pred_x0, start_timesteps, timesteps ###
                ### noisy_model_input shape: [1, 4, 16, 64, 64]                     ###
                #######################################################################
                
                # shape: [1, 4, 16, 64, 64]
                
                
                # 7. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
                
                start_timesteps = start_timesteps.to(accelerator.device, non_blocking=True)

                # 8. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
                # predicted noise eps_0 and predicted original sample x_0, then run the ODE solver using these
                # estimates to predict the data point in the augmented PF-ODE trajectory corresponding to the next ODE
                # solver timestep.
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        # 1. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
                        cond_teacher_output = teacher_unet(
                            noisy_model_input.to(weight_dtype),
                            start_timesteps,
                            encoder_hidden_states=prompt_embeds.to(weight_dtype),
                        ).sample
                        cond_pred_x0 = get_predicted_original_sample(
                            cond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        cond_pred_noise = get_predicted_noise(
                            cond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )

                        # 2. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and unconditional embedding 0
                        uncond_teacher_output = teacher_unet(
                            noisy_model_input.to(weight_dtype),
                            start_timesteps,
                            encoder_hidden_states=uncond_prompt_embeds.to(weight_dtype),
                        ).sample
                        uncond_pred_x0 = get_predicted_original_sample(
                            uncond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        uncond_pred_noise = get_predicted_noise(
                            uncond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )

                        # 3. Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
                        # Note that this uses the LCM paper's CFG formulation rather than the Imagen CFG formulation
                        # print(f"cond_pred_x0: {cond_pred_x0.shape}; uncond_pred_x0: {uncond_pred_x0.shape}; cond_pred_noise: {cond_pred_noise.shape}; uncond_pred_noise: {uncond_pred_noise.shape}; w: {w.shape}")
                        pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                        pred_noise = cond_pred_noise + w * (
                            cond_pred_noise - uncond_pred_noise
                        )
                        # 4. Run one step of the ODE solver to estimate the next point x_prev on the
                        # augmented PF-ODE trajectory (solving backward in time)
                        # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
                        x_prev = solver.ddim_step(pred_x0, pred_noise, index)

                # 9. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
                # Note that we do not use a separate target network for LCM-LoRA distillation.
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=weight_dtype):
                        if args.use_lora:
                            target_noise_pred = unet(
                                x_prev.float(),
                                timesteps,
                                timestep_cond=None,
                                encoder_hidden_states=prompt_embeds.float(),
                            ).sample
                        else:
                            target_noise_pred = target_unet(
                                x_prev.float(),
                                timesteps,
                                timestep_cond=w_embedding,
                                encoder_hidden_states=prompt_embeds.float(),
                            ).sample
                    pred_x_0 = get_predicted_original_sample(
                        target_noise_pred,
                        timesteps,
                        x_prev,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    target = c_skip * x_prev + c_out * pred_x_0
                store_dict["target"] = target
                
                for key in store_dict.keys():
                    if isinstance(store_dict[key], torch.Tensor):
                        store_dict[key] = store_dict[key].to(dtype=weight_dtype).detach().cpu().numpy()
                output_filename = os.path.join(rank_extract_code_dir,  'cache{:03d}.npz'.format(saved_data_idx))
                np.savez_compressed(output_filename, **store_dict)
                saved_data_idx += 1
                progress_bar.update(1)
                global_step += 1

                
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
