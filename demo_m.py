from typing import Optional

import torch
from diffusers import (
    AnimateDiffPipeline,
    DiffusionPipeline,
    UNet3DConditionModel,
    LCMScheduler,
    MotionAdapter,
)
from diffusers.utils import export_to_video
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict

def main():
    # select model_path from ["animatediff-laion", "animatediff-webvid",
    # "modelscopet2v-webvid", "modelscopet2v-laion", "modelscopet2v-anime",
    # "modelscopet2v-real", "modelscopet2v-3d-cartoon"]
    model_path = "modelscopet2v-laion"
    prompts = ["In slow motion, a dessert fork gently presses into the center of the cake. As the fork goes deeper, the outer crust begins to crack."]
    num_inference_steps = 50

    model_id = "yhzhai/mcm"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "animatediff" in model_path:
        pipeline = get_animatediff_pipeline()
    elif "modelscope" in model_path:
        pipeline = get_modelscope_pipeline()
    else:
        raise ValueError(f"Unknown pipeline {model_path}")
    import os

    # lora = UNet3DConditionModel.from_pretrained(
    # "/home/shaoshitong/project/mcm/work_dirs/modelscopet2v_distillation_2/checkpoint-final",
    # torch_device="cpu")
    # unet = lora
    # pipeline.unet = unet
    
    pipeline = pipeline.to(device,dtype=torch.float16)
    output = pipeline(
        prompt=prompts,
        num_frames=16,
        guidance_scale=10.0,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(42),
    ).frames
    
    if not isinstance(output, list):
        output = [output[i] for i in range(output.shape[0])]

    for j in range(len(prompts)):
        export_to_video(
            output[j],
            f"{j}-00.mp4",
            fps=7,
        )


def get_animatediff_pipeline(
    real_variant: Optional[str] = "realvision",
    motion_module_path: str = "guoyww/animatediff-motion-adapter-v1-5-2",
):
    if real_variant is None:
        model_id = "runwayml/stable-diffusion-v1-5"
    elif real_variant == "epicrealism":
        model_id = "emilianJR/epiCRealism"
    elif real_variant == "realvision":
        model_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    else:
        raise ValueError(f"Unknown real_variant {real_variant}")

    adapter = MotionAdapter.from_pretrained(
        motion_module_path, torch_dtype=torch.float16
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        model_id,
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )
    scheduler = LCMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        timestep_scaling=4.0,
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        beta_start=0.00085,
        beta_end=0.012,
        steps_offset=1,
    )
    pipe.scheduler = scheduler
    pipe.enable_vae_slicing()
    return pipe


def get_modelscope_pipeline():
    model_id = "ali-vilab/text-to-video-ms-1.7b"
    pipe = DiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    )
    import diffusers
    scheduler = LCMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        timestep_scaling=4.0,
    )
    pipe.scheduler = scheduler
    pipe.enable_vae_slicing()

    return pipe


if __name__ == "__main__":
    main()