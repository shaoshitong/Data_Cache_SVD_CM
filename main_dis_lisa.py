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
import os
import random
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
import time

from pytorch_memlab import MemReporter
from contextlib import redirect_stdout
import accelerate
import diffusers
import numpy as np
import open_clip
import torch
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
import torch.distributed as torch_dist
from torch.utils.data.distributed import DistributedSampler
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
from dataset.data_cache_dataset import CustomDataset, DataLoaderX
from discriminator.unet_D import (
    Discriminator,
)
from models.spatial_head import IdentitySpatialHead, SpatialHead
from utils.diffusion_misc import *
from utils.dist import dist_init, dist_init_wo_accelerate, get_deepspeed_config
from utils.misc import *
from utils.wandb import setup_wandb
from utils.lisa import LISADiffusion

MAX_SEQ_LENGTH = 77
torch.backends.cudnn.benchmark = True

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
    from accelerate import DistributedDataParallelKwargs, ProfileKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # profile_kwargs = ProfileKwargs(
    #     activities=["cuda"],
    #     profile_memory=True,
    #     record_shapes=True,
    #     output_trace_dir="./output.json"
    # )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
        split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
    )

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
        logger.info(f"Scaling learning rate to {args.learning_rate}")

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

    discriminator = Discriminator()
    if args.dis_output_dir is not None:
        discriminator.load_state_dict(torch.load(args.dis_output_dir, map_location="cpu"))
    logger.info(
        f"\nhLoaded pretrained discriminator from {args.dis_output_dir}\n"
    )

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
        raise NotImplementedError
    
    if args.cd_target in ["learn", "hlearn"]:
        if args.cd_target == "learn":
            spatial_head = SpatialHead(num_channels=4, num_layers=2, kernel_size=1)
            target_spatial_head = SpatialHead(
                num_channels=4, num_layers=2, kernel_size=1
            )
            logger.info("Using SpatialHead for spatial head")
        elif args.cd_target == "hlearn":
            spatial_head = SpatialHead(num_channels=4, num_layers=5, kernel_size=3)
            target_spatial_head = SpatialHead(
                num_channels=4, num_layers=5, kernel_size=3
            )
            logger.info("Using SpatialHead for spatial head")
        else:
            raise ValueError(f"cd_target {args.cd_target} is not supported.")

        spatial_head.train()
        target_spatial_head.load_state_dict(spatial_head.state_dict())
        target_spatial_head.train()
        target_spatial_head.requires_grad_(False)
    else:
        spatial_head = None
        target_spatial_head = None

    if args.prev_train_unet is not None and args.prev_train_unet != "None":
        lora = UNet3DConditionModel.from_pretrained(
        args.prev_train_unet,
        torch_device="cpu")
        unet = lora
        print(f"Successfully load unet from {args.prev_train_unet}")

    unet.train()
    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )
    

    # 9. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    discriminator.to(device=accelerator.device, dtype=weight_dtype)
    discriminator.requires_grad_(False)
    
    if args.cd_target in ["learn", "hlearn"]:
        target_spatial_head.to(accelerator.device)

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
                    # save weights in peft format to be able to load them back
                    unet_.save_pretrained(output_dir)


                    if args.cd_target in ["learn", "hlearn"]:
                        spatial_head_ = accelerator.unwrap_model(spatial_head)
                        spatial_head_.save_pretrained(
                            os.path.join(output_dir, "spatial_head")
                        )
                    
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


                    if args.cd_target in ["learn", "hlearn"]:
                        spatial_head_ = accelerator.unwrap_model(spatial_head)
                        spatial_head_.save_pretrained(
                            os.path.join(output_dir, "spatial_head")
                        )
                        
                    for i, model in enumerate(models):
                        # make sure to pop weight so that corresponding model is not saved again
                        if len(weights) > 0:
                            weights.pop()

        if args.use_lora:

            def load_model_hook(models, input_dir):
                # load the LoRA into the model
                unet_ = accelerator.unwrap_model(unet)
                unet_.from_pretrained(input_dir)

                if args.cd_target in ["learn", "hlearn"]:
                    spatial_head_state_dict = load_file(
                        os.path.join(
                            input_dir,
                            "spatial_head",
                            "diffusion_pytorch_model.safetensors",
                        )
                    )
                    spatial_head_ = accelerator.unwrap_model(spatial_head)
                    spatial_head_.load_state_dict(spatial_head_state_dict)
                    target_spatial_head_ = accelerator.unwrap_model(target_spatial_head)
                    target_spatial_head_.load_state_dict(spatial_head_state_dict)
                    del spatial_head_state_dict

                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    models.pop()

        else:
            # only support finetune motion module for AnimateDiff
            def load_model_hook(models, input_dir):

                student_motion_module = MotionAdapter.from_pretrained(
                    os.path.join(input_dir, "motion_modules")
                )
                unet_ = accelerator.unwrap_model(unet)
                unet_.load_motion_modules(student_motion_module)
                del student_motion_module

                if args.cd_target in ["learn", "hlearn"]:
                    spatial_head_state_dict = load_file(
                        os.path.join(
                            input_dir,
                            "spatial_head",
                            "diffusion_pytorch_model.safetensors",
                        )
                    )
                    spatial_head_ = accelerator.unwrap_model(spatial_head)
                    spatial_head_.load_state_dict(spatial_head_state_dict)
                    target_spatial_head_ = accelerator.unwrap_model(target_spatial_head)
                    target_spatial_head_.load_state_dict(spatial_head_state_dict)
                    del spatial_head_state_dict
                
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
            # teacher_unet.enable_xformers_memory_efficient_attention()
            # if not args.use_lora:
            #     target_unet.enable_xformers_memory_efficient_attention()
        else:
            logger.warning(
                "xformers is not available. Make sure it is installed correctly"
            )
            # raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # if args.allow_tf32:
    #     torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        pass # unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW


    WEBVID_DATA_SIZE = 2467378
    local_rank = torch.distributed.get_rank()
    dataset = CustomDataset(args.extract_code_dir,rank=[0,1,2,3,4,5,6,7])    
    sampler = DistributedSampler(
        dataset,
        num_replicas=torch_dist.get_world_size(),
        rank=local_rank,
        shuffle=True,
    )
    train_dataloader = DataLoaderX(dataset,
                            batch_size=1,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=args.dataloader_num_workers,
                            pin_memory=False,
                            drop_last=True)
    
    # 14. LR Scheduler creation
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(dataset) / (args.gradient_accumulation_steps * accelerator.num_processes)
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lisa_trainer = LISADiffusion(unet, spatial_head, rate=0.2)
    lisa_trainer.insert_hook(optimizer_class=optimizer_class,
                        get_scheduler=get_scheduler,
                        accelerator=accelerator,
                        optim_kwargs=dict(lr=args.learning_rate,
                                          betas=(args.adam_beta1, args.adam_beta2),
                                          weight_decay=args.adam_weight_decay,
                                          eps=args.adam_epsilon),
                        sched_kwargs=dict(name=args.lr_scheduler,
                                          num_warmup_steps=args.lr_warmup_steps,
                                          num_training_steps=args.max_train_steps))
    lisa_trainer.register(optimizer_class=optimizer_class,
                        get_scheduler=get_scheduler,
                        accelerator=accelerator,
                        optim_kwargs=dict(lr=args.learning_rate,
                                          betas=(args.adam_beta1, args.adam_beta2),
                                          weight_decay=args.adam_weight_decay,
                                          eps=args.adam_epsilon),
                        sched_kwargs=dict(name=args.lr_scheduler,
                                          num_warmup_steps=args.lr_warmup_steps,
                                          num_training_steps=args.max_train_steps))

    # 15. Prepare for training
    # Prepare everything with our `accelerator`.
    if args.cd_target in ["learn", "hlearn"]:
        (
            unet,
            spatial_head,
        ) = accelerator.prepare(
            unet,
            spatial_head,
        )
    else:
        (
            unet
        ) = accelerator.prepare(
            unet
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader.dataset) / (args.train_batch_size * args.gradient_accumulation_steps * torch_dist.get_world_size())
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
    
    # 16. Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {len(train_dataloader.dataset) / (args.train_batch_size * torch_dist.get_world_size())}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(
        f"  Num learnable parameters = {sum([p.numel() for p in unet.parameters() if p.requires_grad]) / 1e6} M"
    )
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [
                d
                for d in dirs
                if (d.startswith("checkpoint") and "step" not in d and "final" not in d)
            ]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if os.path.exists(os.path.join(args.output_dir, path)):
                accelerator.load_state(os.path.join(args.output_dir, path))
            else:
                accelerator.load_state(args.resume_from_checkpoint)
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    gc.collect()
    torch.cuda.empty_cache()

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    last_update_r1_step = global_step

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet), accelerator.accumulate(spatial_head):
                
                # 1. Load and process the image and text conditioning
                noisy_model_input, target, sd_prompt_embeds, \
                prompt_embeds, start_timesteps, \
                timesteps, text = batch["noisy_model_input"], batch["target"], \
                    batch["sd_prompt_embeds"], batch["prompt_embeds"], batch["start_timesteps"], \
                    batch["timesteps"], batch["text"]
                
                noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=weight_dtype).squeeze(0)
                target = target.to(device=accelerator.device, dtype=weight_dtype).squeeze(0)
                prompt_embeds = prompt_embeds.to(device=accelerator.device, dtype=weight_dtype).squeeze(0)
                sd_prompt_embeds = sd_prompt_embeds.to(device=accelerator.device, dtype=weight_dtype).squeeze(0)
                start_timesteps, timesteps = start_timesteps.to(device=accelerator.device, dtype=torch.int64).squeeze(0), \
                    timesteps.to(device=accelerator.device, dtype=torch.int64).squeeze(0)                
                # torch.Size([1, 4, 16, 64, 64]) torch.Size([1, 4, 16, 64, 64]) torch.Size([1, 1024]) torch.Size([2, 3, 512, 512]) torch.Size([1, 1024]) torch.Size([1, 77, 768])

                # 3. Get boundary scalings for start_timesteps and (end) timesteps.
            
                c_skip_start, c_out_start = scalings_for_boundary_conditions(
                    start_timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip_start, c_out_start = [
                    append_dims(x, noisy_model_input.ndim) for x in [c_skip_start, c_out_start]
                ]
                c_skip, c_out = scalings_for_boundary_conditions(
                    timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip, c_out = [append_dims(x, noisy_model_input.ndim) for x in [c_skip, c_out]]
                
                # 7. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
                noise_pred = unet(
                    noisy_model_input,
                    start_timesteps,
                    timestep_cond=None,
                    encoder_hidden_states=prompt_embeds,
                    # added_cond_kwargs=encoded_text,
                ).sample 
                pred_x_0_stu = get_predicted_original_sample(
                    noise_pred,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )

                model_pred = (
                    c_skip_start * noisy_model_input + c_out_start * pred_x_0_stu
                ) # Get the predicted video from stu
                
                loss_dict = {}

                # 10.1. Calculate CD loss
                model_pred_cd = prepare_cd_target(model_pred, args.cd_target)
                if args.cd_target in ["learn", "hlearn"]:
                    model_pred_cd = spatial_head(model_pred_cd)
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=weight_dtype):
                        target_cd = prepare_cd_target(target.float(), args.cd_target)
                        if args.cd_target in ["learn", "hlearn"]:
                            target_cd = target_spatial_head(target_cd.float())
                if args.loss_type == "l2":
                    loss_unet_cd = F.mse_loss(
                        model_pred_cd.float(), target_cd.float(), reduction="mean"
                    )
                elif args.loss_type == "huber":
                    loss_unet_cd = torch.mean(
                        torch.sqrt(
                            (model_pred_cd.float() - target_cd.float()) ** 2
                            + args.huber_c**2
                        )
                        - args.huber_c
                    )
                loss_dict["loss_unet_cd"] = loss_unet_cd
                loss_unet_total = loss_unet_cd

                loss_unet_pred_x0 = torch.mean(
                        torch.sqrt(
                            (model_pred.float() - target.float()) ** 2
                            + args.huber_c**2
                        )
                        - args.huber_c
                    )
                loss_dict["loss_unet_pred_x0"] = loss_unet_pred_x0
                loss_unet_total = loss_unet_total + (loss_unet_pred_x0 * 0.1)
                
                if not args.no_disc:
                    gen_latents = rearrange(model_pred, "b c t n m -> (b t) c n m")
                    sd_prompt_embeds = sd_prompt_embeds.expand(gen_latents.shape[0],-1,-1)                 
                    index = torch.randint(
                        0, args.num_ddim_timesteps, (gen_latents.shape[0],), device=gen_latents.device
                    ).long()
                    _start_timesteps = solver.ddim_timesteps[index]
                    _noise = torch.randn_like(gen_latents).to(dtype=weight_dtype)
                
                    gen_noisy_model_input_list = []
                    for b_idx in range(gen_latents.shape[0]): # Add noise
                        if index[b_idx] != args.num_ddim_timesteps - 1:
                            gen_noisy_model_input = noise_scheduler.add_noise(
                                gen_latents[b_idx, None],
                                _noise[b_idx, None],
                                _start_timesteps[b_idx, None],
                            )
                        else:
                            gen_noisy_model_input = _noise[b_idx, None]
                        gen_noisy_model_input_list.append(gen_noisy_model_input)
                    gen_noisy = torch.cat(gen_noisy_model_input_list, dim=0).half()
                    with torch.autocast("cuda", dtype=weight_dtype):
                        disc_pred_gen = discriminator(gen_noisy, _start_timesteps, sd_prompt_embeds)
                        if args.disc_loss_type == "bce":
                            pos_label = torch.ones_like(disc_pred_gen)
                            loss_unet_adv = F.binary_cross_entropy_with_logits(
                                disc_pred_gen, pos_label
                            )
                        elif args.disc_loss_type == "hinge":
                            loss_unet_adv = -disc_pred_gen.mean() + 1
                        elif args.disc_loss_type == "wgan":
                            loss_unet_adv = torch.max(-torch.ones_like(disc_pred_gen), -disc_pred_gen).mean()
                            # loss_unet_adv = -disc_pred_gen.mean()
                        else:
                            raise ValueError(
                                f"Discriminator loss type {args.disc_loss_type} not supported."
                            )

                        loss_dict["loss_unet_adv"] = loss_unet_adv
                    loss_unet_total = (
                        loss_unet_total + args.disc_loss_weight * loss_unet_adv
                    )


                loss_dict["loss_unet_total"] = loss_unet_total
                accelerator.backward(loss_unet_total)
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                
                if args.cd_target in ["learn", "hlearn"]:
                    update_ema(
                        target_spatial_head.parameters(),
                        spatial_head.parameters(),
                        args.ema_decay,
                    )
                progress_bar.update(1)
                if global_step % 10 == 0 and global_step != 0: # you can use other number to replace 6
                    lisa_trainer.lisa_recall()
                    accelerator.clear()
                global_step += 1

                # according to https://github.com/huggingface/diffusers/issues/2606
                # DeepSpeed need to run save for all processes
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d
                                for d in checkpoints
                                if (
                                    d.startswith("checkpoint")
                                    and "step" not in d
                                    and "final" not in d
                                )
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                    accelerator.wait_for_everyone()
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    try:
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    except Exception as e:
                        logger.info(f"Failed to save state: {e}")

            logs = {
                "unet_lr": list(lisa_trainer.optimizer_dict.values())[0].param_groups[0]["lr"],
                "disc_r1_weight": args.disc_lambda_r1,
            }
            for loss_name, loss_value in loss_dict.items():
                if type(loss_value) == torch.Tensor:
                    logs[loss_name] = loss_value.item()
                else:
                    logs[loss_name] = loss_value
            current_time = datetime.now().strftime("%m-%d-%H:%M")
            progress_bar.set_postfix(
                **logs,
                **{"cur time": current_time},
            )
            try:
                accelerator.log(logs, step=global_step)
            except Exception as e:
                logger.info(f"Failed to log metrics at step {global_step}: {e}")
            if global_step >= args.max_train_steps:
                break
            
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_lora:
            unet.save_pretrained(os.path.join(args.output_dir, "checkpoint-final"))
            if args.cd_target in ["learn", "hlearn"]:
                spatial_head_ = accelerator.unwrap_model(spatial_head)
                spatial_head_.save_pretrained(
                    os.path.join(args.output_dir, "spatial_head")
                )
        else:
            # save motion module
            unet_ = accelerator.unwrap_model(unet)
            unet_.save_motion_modules(os.path.join(args.output_dir, "motion_modules"))
            target_unet = accelerator.unwrap_model(target_unet)
            target_unet.save_motion_modules(
                os.path.join(args.output_dir, "target_motion_modules")
            )
            if args.cd_target in ["learn", "hlearn"]:
                spatial_head_ = accelerator.unwrap_model(spatial_head)
                spatial_head_.save_pretrained(
                    os.path.join(args.output_dir, "spatial_head")
                )

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
