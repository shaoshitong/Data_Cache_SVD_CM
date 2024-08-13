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
    DPMSolverMultistepScheduler,
    DiffusionPipeline
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
from dataset.data_cache_dataset import CustomDataset, DataLoaderX

from models.spatial_head import IdentitySpatialHead, SpatialHead
from utils.diffusion_misc import *
from utils.dist import dist_init, dist_init_wo_accelerate, get_deepspeed_config
from utils.misc import *
from utils.wandb import setup_wandb
from discriminator.unet_D import Discriminator

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

def save_image(vae, latents, path):
    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(path)
    
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

def load_stable_diffusion_xl(device):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe = pipe.to(device=device)
    return pipe
    
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
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
        kwargs_handlers=[ddp_kwargs]
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
    # Initialize the DDIM ODE solver for distillation.
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
    ).to(accelerator.device)
    # 3. Load text encoders from SD 1.X/2.X checkpoint.
    # import correct text encoder classes
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_teacher_model, #args.pretrained_teacher_model, # "runwayml/stable-diffusion-v1-5"
        subfolder="vae",
        revision=args.teacher_revision,
    ) 
    vae.requires_grad_(False)
    
    
    video_vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", #, # "runwayml/stable-diffusion-v1-5"
        subfolder="vae",
    ) 
    video_vae.requires_grad_(False)
    
    # 6. Freeze teacher vae, text_encoder, and teacher_unet
    normalize_fn = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    discriminator = Discriminator(is_training=True)

    from torchvision import transforms
    trans = transforms.Compose([transforms.ToTensor(),transforms.Resize(256)]) 
    
    # if args.dis_output_dir is not None and os.path.exists(args.dis_output_dir):
    #     discriminator.from_pretrained(args.dis_output_dir)
    # logger.info(
    #     f"\nhLoaded pretrained discriminator from {args.dis_output_dir}\n"
    # )
    discriminator.train()
    
    inference_sd_model = load_stable_diffusion_xl(accelerator.device)
    inference_sd_model.scheduler = DPMSolverMultistepScheduler.from_config(inference_sd_model.scheduler.config)
    # 9. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    vae.to(accelerator.device)
    video_vae.to(accelerator.device)
    inference_sd_model.to(accelerator.device)
    inference_sd_model.enable_xformers_memory_efficient_attention()
    
    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                discriminator_ = accelerator.unwrap_model(discriminator)
                torch.save(discriminator_.state_dict(),
                    os.path.join(output_dir, "discriminator.pth")
                )
                for _, model in enumerate(models):
                    # make sure to pop weight so that corresponding model is not saved again
                    if len(weights) > 0:
                        weights.pop()
            
        def load_model_hook(models, input_dir):
            disc_state_dict = load_file(
                os.path.join(
                    input_dir,
                    "discriminator.pth",
                )
            )
            disc_ = accelerator.unwrap_model(discriminator)
            disc_.load_state_dict(disc_state_dict)
            del disc_state_dict

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    optimizer_class = torch.optim.AdamW
    discriminator_params = list(filter(lambda x:x.requires_grad, discriminator.parameters()))
    optimizer = optimizer_class(
        discriminator_params,
        lr=args.disc_learning_rate,
        betas=(args.disc_adam_beta1, args.disc_adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 13. Dataset creation and data processing
    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.

    local_rank = torch.distributed.get_rank()
    dataset = CustomDataset(args.extract_code_dir,rank=[0,1,2,3,4,5,6,7])
    train_dataloader = DataLoaderX(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.dataloader_num_workers,
                            pin_memory=False,
                            drop_last=True)

    # 14. LR Scheduler creation
    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps, # TODO
        num_training_steps=args.disc_max_train_steps, # TODO
    )

    # 15. Prepare for training
    # Prepare everything with our `accelerator`.
    (

        discriminator,
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(
        discriminator,
        optimizer,
        lr_scheduler,
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader.dataset) / (args.train_batch_size * args.gradient_accumulation_steps)
    )
    args.num_train_epochs = math.ceil(args.disc_max_train_steps / num_update_steps_per_epoch)

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
    logger.info(f"  Num batches each epoch = {len(train_dataloader.dataset) / (args.train_batch_size)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.disc_max_train_steps}")
    logger.info(
        f"  Num learnable parameters = {sum([p.numel() for p in discriminator.parameters() if p.requires_grad]) / 1e6} M"
    )
    global_step = 0
    first_epoch = 0
    initial_global_step = 0
    gc.collect()
    torch.cuda.empty_cache()

    progress_bar = tqdm(
        range(0, args.disc_max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(discriminator), torch.autograd.set_detect_anomaly(True):
                # 1. Load and process the image and text conditioning
                noisy_model_input, target, sd_prompt_embeds, \
                prompt_embeds, start_timesteps, \
                timesteps, text, real_data = batch["noisy_model_input"], batch["target"], \
                    batch["sd_prompt_embeds"], batch["prompt_embeds"], batch["start_timesteps"], \
                    batch["timesteps"], batch["text"], batch["latents"]
                
                noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=weight_dtype, non_blocking=True).squeeze(0)
                target = target.to(device=accelerator.device, dtype=weight_dtype, non_blocking=True).squeeze(0)
                sd_prompt_embeds = sd_prompt_embeds.to(device=accelerator.device, dtype=weight_dtype, non_blocking=True).squeeze(0)
                start_timesteps, timesteps = start_timesteps.squeeze(0), timesteps.squeeze(0)
                   
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

                loss_dict = {}
                gt_latents = []
                ori_gt_latents = []

                with torch.no_grad():
                    gen_latents = target
                    gen_latents = rearrange(gen_latents, "b c t h w -> (b t) c h w")[:8]
                    gt_latent = inference_sd_model(prompt=text, 
                                                    height=1024, 
                                                    width=1024, 
                                                    num_inference_steps=15, 
                                                    guidance_scale=5).images[0] # latents=add_noise_model_input
                    tmp = (trans(gt_latent).to(accelerator.device) * 2 - 1).unsqueeze(0)
                    gt_latent = vae.config.scaling_factor * vae.encode(tmp).latent_dist.sample()
                    # ori_gt_latent = video_vae.config.scaling_factor * video_vae.encode(tmp.float()).latent_dist.sample().half()
                    
                    # tmp = vae.decode(gen_latents.float() / vae.config.scaling_factor, return_dict=False)[0]
                    # ori_gen_latents = video_vae.config.scaling_factor * video_vae.encode(tmp).latent_dist.sample().half()

                    for i in range(0, 2, 1):
                        gt_latents.append(gt_latent)

                    gt_latent = inference_sd_model(prompt=text, 
                                                    height=1024, 
                                                    width=1024, 
                                                    num_inference_steps=15, 
                                                    guidance_scale=5).images[0] # latents=add_noise_model_input
                    tmp = (trans(gt_latent).to(accelerator.device) * 2 - 1).unsqueeze(0)
                    gt_latent = vae.config.scaling_factor * vae.encode(tmp).latent_dist.sample()
                    # ori_gt_latent = video_vae.config.scaling_factor * video_vae.encode(tmp.float()).latent_dist.sample().half()

                    for i in range(0, 2, 1):
                        gt_latents.append(gt_latent)

                    gt_latents.append(real_data)
                    gt_latents = torch.cat(gt_latents, dim=0)
                    gt_latents = gt_latents.to(weight_dtype)
                    # ori_gt_latents = torch.cat(ori_gt_latents, dim=0)
                    # ori_gt_latents = ori_gt_latents.to(weight_dtype)
                    # ori_gt_latents = gt_latents
                    weight = 0.5
                    # if accelerator.is_main_process:
                    #     print(gt_latents.dtype,gen_latents.dtype)
                    #     save_image(video_vae, gt_latents.float(), "gt_latents.png")
                    #     save_image(video_vae, gen_latents.float(), "gen_latents.png")
                    #     save_image(vae, ori_gt_latents.float(), "ori_gt_latents.png")
                    #     save_image(vae, ori_gen_latents.float(), "ori_gen_latents.png")

                    index = torch.randint(
                        0, args.num_ddim_timesteps, (gt_latents.shape[0],), device=gt_latents.device
                    ).long()
                    start_timesteps = solver.ddim_timesteps[index]
                    
                    noise = torch.randn_like(gt_latents).to(dtype=weight_dtype)
                    gt_noisy_model_input_list = []
                    gen_noisy_model_input_list = []
                    # ori_gt_noisy_model_input_list = []
                    # ori_gen_noisy_model_input_list = []
                    for b_idx in range(gt_latents.shape[0]): # Add noise
                        if index[b_idx] != args.num_ddim_timesteps - 1:
                            gt_noisy_model_input = noise_scheduler.add_noise(
                                gt_latents[b_idx, None],
                                noise[b_idx, None],
                                start_timesteps[b_idx, None],
                            )
                            gen_noisy_model_input = noise_scheduler.add_noise(
                                gen_latents[b_idx, None],
                                noise[b_idx, None],
                                start_timesteps[b_idx, None],
                            )
                            # ori_gt_noisy_model_input = noise_scheduler.add_noise(
                            #     ori_gt_latents[b_idx, None],
                            #     noise[b_idx, None],
                            #     start_timesteps[b_idx, None],
                            # )
                            # ori_gen_noisy_model_input = noise_scheduler.add_noise(
                            #     ori_gen_latents[b_idx, None],
                            #     noise[b_idx, None],
                            #     start_timesteps[b_idx, None],
                            # )
                        else:
                            # hard swap input to pure noise to ensure zero terminal SNR
                            gt_noisy_model_input = noise[b_idx, None]
                            gen_noisy_model_input = noise[b_idx, None]
                            # ori_gt_noisy_model_input = noise[b_idx, None]
                            # ori_gen_noisy_model_input = noise[b_idx, None]
                        gt_noisy_model_input_list.append(gt_noisy_model_input)
                        gen_noisy_model_input_list.append(gen_noisy_model_input)
                        # ori_gt_noisy_model_input_list.append(ori_gt_noisy_model_input)
                        # ori_gen_noisy_model_input_list.append(ori_gen_noisy_model_input)
                                               
                    gt_noisy = torch.cat(gt_noisy_model_input_list, dim=0).half()                       
                    gen_noisy = torch.cat(gen_noisy_model_input_list, dim=0).half()   
                    ori_gt_noisy = gt_noisy # torch.cat(ori_gt_noisy_model_input_list, dim=0).half()                       
                    ori_gen_noisy = gen_noisy # torch.cat(ori_gen_noisy_model_input_list, dim=0).half()   
                                        
                sd_prompt_embeds = sd_prompt_embeds.expand(gt_noisy.shape[0],-1,-1)
                b1, b2 = gt_noisy.shape[0], gen_noisy.shape[0]
                disc_pred, align_loss = discriminator(torch.cat([gt_noisy, gen_noisy],0), 
                                          torch.cat([start_timesteps,start_timesteps],0), 
                                          torch.cat([sd_prompt_embeds, sd_prompt_embeds],0),
                                          ori_sample=torch.cat([ori_gt_noisy, ori_gen_noisy],0))
                disc_pred_gt, disc_pred_gen = torch.split(disc_pred,[b1, b2],dim=0)
            
                if args.disc_loss_type == "bce":
                    pos_label = torch.ones_like(disc_pred_gt)
                    neg_label = torch.zeros_like(disc_pred_gen)
                    loss_disc_gt = F.binary_cross_entropy_with_logits(
                        disc_pred_gt, pos_label
                    ) * (1-weight)
                    loss_disc_gen = F.binary_cross_entropy_with_logits(
                        disc_pred_gen, neg_label
                    ) * weight
                elif args.disc_loss_type == "hinge":
                    loss_disc_gt = (
                        torch.max(torch.zeros_like(disc_pred_gt), 1 - disc_pred_gt)
                    ).mean() * (1-weight)
                    loss_disc_gen = (
                        torch.max(torch.zeros_like(disc_pred_gen), 1 + disc_pred_gen)
                    ).mean() * weight
                elif args.disc_loss_type == "wgan":
                    loss_disc_gt = (
                        torch.max(-torch.ones_like(disc_pred_gt), -disc_pred_gt)
                    ).mean() * (1-weight)
                    loss_disc_gen = (
                        torch.max(-torch.ones_like(disc_pred_gen), disc_pred_gen)
                    ).mean() * weight
                else:
                    raise ValueError(
                        f"Discriminator loss type {args.disc_loss_type} not supported."
                    )
                loss_disc_total = loss_disc_gt + loss_disc_gen + align_loss
                accelerator.backward(loss_disc_total)
                loss_dict["loss_disc_gt"] = loss_disc_gt
                loss_dict["loss_disc_gen"] = loss_disc_gen  
                loss_dict["align_loss"] = align_loss
                loss_dict["loss_disc_total"] = loss_disc_total
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
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
                "unet_lr": lr_scheduler.get_last_lr()[0],
            }
            for loss_name, loss_value in loss_dict.items():
                if type(loss_value) == torch.Tensor:
                    logs[loss_name] = round(loss_value.item(),2)
                else:
                    logs[loss_name] = round(loss_value,2)

            current_time = datetime.now().strftime("%m-%d-%H:%M")
            progress_bar.set_postfix(
                **logs,
                **{"cur time": current_time},
            )
            try:
                accelerator.log(logs, step=global_step)
            except Exception as e:
                logger.info(f"Failed to log metrics at step {global_step}: {e}")

            if global_step >= args.disc_max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    save_path = os.path.join(
        args.output_dir, f"checkpoint-discriminator-final"
    )
    try:
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")
    except Exception as e:
        logger.info(f"Failed to save state: {e}")
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)


