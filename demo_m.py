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
    prompts = [
    "Spiderman is surfing",
    "Yellow and black tropical fish dart through the sea",
    "An epic tornado attacking above aglowing city at night",
    "Slow pan upward of blazing oak fire in an indoor fireplace",
    "a cat wearing sunglasses and working as a lifeguard at pool",
    "A dog in astronaut suit and sunglasses floating in space"
]
    num_inference_steps = 4

    model_id = "yhzhai/mcm"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "animatediff" in model_path:
        pipeline = get_animatediff_pipeline()
    elif "modelscope" in model_path:
        pipeline = get_modelscope_pipeline()
    else:
        raise ValueError(f"Unknown pipeline {model_path}")
    import os

    lora = UNet3DConditionModel.from_pretrained(
    "/home/shaoshitong/project/mcm/work_dirs/modelscopet2v_distillation_14/checkpoint-final",
    torch_device="cpu")
    unet = lora
    pipeline.unet = unet
    
    pipeline = pipeline.to(device,dtype=torch.float16)
    outputs = []
    for prompt in prompts:
        output = pipeline(
            prompt=prompt,
            num_frames=16,
            guidance_scale=1.0,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(42),
        ).frames
        output = output[0]

        outputs.append(output)
    output = outputs

    for j in range(len(prompts)):
        export_to_video(
            output[j],
            f"{j}-step{num_inference_steps}.mp4",
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