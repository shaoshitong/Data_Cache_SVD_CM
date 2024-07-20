from typing import Optional

import torch
from diffusers import (
    AnimateDiffPipeline,
    DiffusionPipeline,
    UNet3DConditionModel,
    LCMScheduler,
    MotionAdapter,
)
model_id = "emilianJR/epiCRealism"
motion_module_path: str = "guoyww/animatediff-motion-adapter-v1-5-2"

adapter = MotionAdapter.from_pretrained(
    motion_module_path, torch_dtype=torch.float16
)
pipe = AnimateDiffPipeline.from_pretrained(
    model_id,
    motion_adapter=adapter,
    torch_dtype=torch.float16,
)
prompt = "Spiderman is surfing"
video_frames = pipe(prompt, num_inference_steps=25).frames
