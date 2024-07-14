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

import wandb
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
from diffusers.utils import check_min_version, export_to_video, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from utils.diffusion_misc import *
from utils.misc import *

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

def save_to_local(save_dir: str, prompt: str, video):
    if len(prompt) > 256:
        prompt = prompt[:256]
    prompt = prompt.replace(" ", "_")
    logger.info(f"Saving images to {save_dir}")

    export_to_video(video, os.path.join(save_dir, f"{prompt}.mp4"))
def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid(
        [
            torch.linspace(w_min, w_max, len_w, device=device),
            torch.linspace(h_min, h_max, len_h, device=device),
        ],
    )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2.0, (h - 1) / 2.0]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]


def global_correlation_softmax(
    feature0,
    feature1,
    pred_bidir_flow: bool = False,
):
    # global correlation
    b, c, h, w = feature0.shape
    feature0 = feature0.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
    feature1 = feature1.view(b, c, -1)  # [B, C, H*W]

    correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (
        c**0.5
    )  # [B, H, W, H, W]

    # flow from softmax
    init_grid = coords_grid(b, h, w).to(correlation.device)  # [B, 2, H, W]
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    correlation = correlation.view(b, h * w, h * w)  # [B, H*W, H*W]

    if pred_bidir_flow:
        correlation = torch.cat(
            (correlation, correlation.permute(0, 2, 1)), dim=0
        )  # [2*B, H*W, H*W]
        init_grid = init_grid.repeat(2, 1, 1, 1)  # [2*B, 2, H, W]
        grid = grid.repeat(2, 1, 1)  # [2*B, H*W, 2]
        b = b * 2

    prob = F.softmax(correlation, dim=-1)  # [B, H*W, H*W]

    correspondence = (
        torch.matmul(prob, grid).view(b, h, w, 2).permute(0, 3, 1, 2)
    )  # [B, 2, H, W]

    # when predicting bidirectional flow, flow is the concatenation of forward flow and backward flow
    flow = correspondence - init_grid

    return flow, prob, correlation


def local_correlation_softmax(
    feature0,
    feature1,
    local_radius: int,
    padding_mode: str = "zeros",
):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(
        -local_radius,
        local_radius,
        -local_radius,
        local_radius,
        local_h,
        local_w,
        device=feature0.device,
    )  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1)^2, 2]

    sample_coords_softmax = sample_coords

    # exclude coords that are out of image space
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (
        sample_coords[:, :, :, 0] < w
    )  # [B, H*W, (2R+1)^2]
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (
        sample_coords[:, :, :, 1] < h
    )  # [B, H*W, (2R+1)^2]

    valid = (
        valid_x & valid_y
    )  # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax

    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
    window_feature = F.grid_sample(
        feature1, sample_coords_norm, padding_mode=padding_mode, align_corners=True
    ).permute(
        0, 2, 1, 3
    )  # [B, H*W, C, (2R+1)^2]
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (
        c**0.5
    )  # [B, H*W, (2R+1)^2]

    # mask invalid locations
    corr[~valid] = -1e9

    prob = F.softmax(corr, -1)  # [B, H*W, (2R+1)^2]

    correspondence = (
        torch.matmul(prob.unsqueeze(-2), sample_coords_softmax)
        .squeeze(-2)
        .view(b, h, w, 2)
        .permute(0, 3, 1, 2)
    )  # [B, 2, H, W]

    flow = correspondence - coords_init
    match_prob = prob

    return flow, match_prob, corr



def log_validation(
    vae,
    unet,
    args,
    accelerator,
    weight_dtype,
    step,
    name="target",
    scheduler: str = "lcm",
    num_inference_steps: int = 4,
    add_to_trackers: bool = True,
    use_lora: bool = False,
    disc_gt_images: Optional[List] = None,
    guidance_scale: float = 1.0,
    spatial_head = None,
    logger_prefix: str = "",
):
    global logger
    logger.info("Running validation... ")
    scheduler_additional_kwargs = {}
    if args.base_model_name == "animatediff":
        scheduler_additional_kwargs["beta_schedule"] = "linear"
        scheduler_additional_kwargs["clip_sample"] = False
        scheduler_additional_kwargs["timestep_spacing"] = "linspace"

    if scheduler == "lcm":
        # set beta_schedule="linear" according to https://huggingface.co/wangfuyun/AnimateLCM
        scheduler = LCMScheduler.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="scheduler",
            **scheduler_additional_kwargs,
        )
    elif scheduler == "ddim":
        scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="scheduler",
            **scheduler_additional_kwargs,
        )
    elif scheduler == "euler":
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="scheduler",
            **scheduler_additional_kwargs,
        )
    else:
        raise ValueError(f"Scheduler {scheduler} is not supported.")

    unet = deepcopy(accelerator.unwrap_model(unet))
    if args.base_model_name == "animatediff":
        pipeline_cls = AnimateDiffPipeline
    elif args.base_model_name == "modelscope":
        pipeline_cls = TextToVideoSDPipeline

    if use_lora:
        pipeline = pipeline_cls.from_pretrained(
            args.pretrained_teacher_model,
            vae=vae,
            scheduler=scheduler,
            revision=args.revision,
            torch_dtype=weight_dtype,
            safety_checker=None,
        )
        lora_state_dict = get_module_kohya_state_dict(unet, "lora_unet", weight_dtype)
        pipeline.load_lora_weights(lora_state_dict)
        pipeline.fuse_lora()
    else:
        pipeline = pipeline_cls.from_pretrained(
            args.pretrained_teacher_model,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            revision=args.revision,
            torch_dtype=weight_dtype,
            safety_checker=None,
        )

    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)

    if (
        args.enable_xformers_memory_efficient_attention
        and args.base_model_name != "animatediff"
    ):
        if is_xformers_available():
            pipeline.enable_xformers_memory_efficient_attention()
        else:
            logger.warning(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_prompts = [
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
        "Cute small corgi sitting in a movie theater eating popcorn, unreal engine.",
        "A Pikachu with an angry expression and red eyes, with lightning around it, hyper realistic style.",
        "A dog is reading a thick book.",
        "Three cats having dinner at a table at new years eve, cinematic shot, 8k.",
        "An astronaut riding a pig, highly realistic dslr photo, cinematic shot.",
    ]

    image_logs = []

    for _, prompt in enumerate(validation_prompts):
        output = []
        with torch.autocast("cuda", dtype=weight_dtype):
            output = pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                height=args.resolution,
                width=args.resolution,
                generator=generator,
                guidance_scale=guidance_scale,
                output_type="latent",
            ).frames
            if spatial_head is not None:
                output = spatial_head(output)

            output = pipeline.decode_latents(output)
            video = tensor2vid(output, pipeline.image_processor, output_type="np")
            # video should be a tensor of shape (t, h, w, 3), min 0, max 1
            video = video[0]

        save_dir = os.path.join(args.output_dir, "output", f"{name}-step-{step}")
        if accelerator.is_main_process:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        image_logs.append({"validation_prompt": prompt, "video": video})
        save_to_local(save_dir, prompt, video)

    if add_to_trackers:
        try:
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    for log in image_logs:
                        images = log["video"]
                        validation_prompt = (
                            f"{logger_prefix}{num_inference_steps} steps/"
                            + log["validation_prompt"]
                        )
                        formatted_images = []
                        for image in images:
                            formatted_images.append(np.asarray(image))

                        formatted_images = np.stack(formatted_images)

                        tracker.writer.add_images(
                            validation_prompt,
                            formatted_images,
                            step,
                            dataformats="NHWC",
                        )
                    if disc_gt_images is not None:
                        for i, image in enumerate(disc_gt_images):
                            tracker.writer.add_image(
                                f"discriminator gt image/{i}",
                                image,
                                step,
                                dataformats="HWC",
                            )
                elif tracker.name == "wandb":
                    # log image for comparison
                    formatted_images = []

                    for log in image_logs:
                        images = log["video"]
                        validation_prompt = log["validation_prompt"]
                        image = wandb.Image(images[0], caption=validation_prompt)
                        formatted_images.append(image)

                    if args.use_lora:
                        tracker.log(
                            {
                                f"{logger_prefix}validation image {num_inference_steps} steps": formatted_images
                            },
                            step=step,
                        )
                    else:
                        tracker.log(
                            {
                                f"{logger_prefix}validation image {num_inference_steps} steps/{name}": formatted_images
                            },
                            step=step,
                        )

                    # log video
                    formatted_video = []
                    for log in image_logs:
                        video = (log["video"] * 255).astype(np.uint8)
                        validation_prompt = log[
                            "validation_prompt"
                        ]  # wandb does not support video logging with caption
                        video = wandb.Video(
                            np.transpose(video, (0, 3, 1, 2)), fps=4, format="mp4"
                        )
                        formatted_video.append(video)

                    if args.use_lora:
                        tracker.log(
                            {
                                f"{logger_prefix}validation video {num_inference_steps} steps": formatted_video
                            },
                            step=step,
                        )
                    else:
                        tracker.log(
                            {
                                f"{logger_prefix}validation video {num_inference_steps} steps/{name}": formatted_video
                            },
                            step=step,
                        )
                    # log discriminator ground truth images
                    if disc_gt_images is not None:
                        formatted_disc_gt_images = []
                        for i, image in enumerate(disc_gt_images):
                            image = wandb.Image(
                                image, caption=f"discriminator gt image {i}"
                            )
                            formatted_disc_gt_images.append(image)
                        tracker.log(
                            {"discriminator gt images": formatted_disc_gt_images},
                            step=step,
                        )
                else:
                    logger.warning(f"image logging not implemented for {tracker.name}")
        except Exception as e:
            logger.error(f"Failed to log images: {e}")

    del pipeline
    del unet
    gc.collect()
    torch.cuda.empty_cache()

    return image_logs