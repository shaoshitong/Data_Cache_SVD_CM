# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(0)

# def calculate_ema(data, beta):
#     ema = np.ones((data,))
#     ema[0] = (beta ** (data-1))
#     for t in range(1, data):
#         ema[t] = (1-beta) * beta ** (data-t-1)
#     print(np.sum(ema,axis=0))
#     return ema

# beta = 0.9999
# data = 16000
# snapshot = 15
# ema = calculate_ema(data, beta)
# _len = float(data / snapshot)
# results = np.zeros((snapshot,))
# count = 0
# for i in range(data):
#     if (i+1) == int(_len * (count + 1)):
#         results[count] += ema[i]
#         count += 1
#     else:
#         results[count] += ema[i]
# results = results / np.sum(results)
# print(results)
# plt.figure(figsize=(12, 6))
# plt.plot(results, label='Weight of Current Theta')
# plt.title('Weight of Current Theta at Each Time Step')
# plt.xlabel('Time Step')
# plt.ylabel('Weight')
# plt.legend()
# plt.savefig("./test_ema.png")
# [0.22459092 0.02529037 0.02813822 0.03127583 0.03482862 0.03875055
#  0.04307151 0.04796423 0.05336531 0.05931592 0.06605393 0.07349202
#  0.0816869  0.09096616 0.10120952]
import torch
import numpy as np

weights = [0.22459092, 0.02529037, 0.02813822, 0.03127583, 0.03482862, 0.03875055,
 0.04307151, 0.04796423, 0.05336531, 0.05931592, 0.06605393, 0.07349202,
 0.0816869, 0.09096616, 0.10120952]


from typing import Optional
import os,sys
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

def load_lcm_unet(snapshot_index=1, root_path="/data/shaoshitong/mcm_work_dirs_with_dis/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filename = f"modelscopet2v_distillation_{snapshot_index}" + "/checkpoint-final"
    local_snapshot_path = os.path.join(root_path, filename)
    unet = UNet3DConditionModel.from_pretrained(
    local_snapshot_path,
    torch_device="cpu")
    return unet

def merge_unet_for_ema(emas, weights, device="cuda"):
    emas = emas[1:]
    emas[0] = emas[0].to(device)
    state_dict = emas[0].state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key] * weights[0]
        for ema, weight in zip(emas[1:], weights[1:]):
            state_dict[key] += (ema.state_dict()[key].to(device) * weight)
            del ema.state_dict()[key]
    final_ema = emas[0]
    final_ema.load_state_dict(state_dict)
    del emas[1:]
    return final_ema
        
        
        
if __name__ == "__main__":
    pipeline = get_modelscope_pipeline()
    prompts = ["A dog walking on a treadmill","Yellow and black tropical fish dart through the sea."]
    unets = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(1, 17):
        unet = load_lcm_unet(i)
        print(f"Successfully Load UNet-{i}")
        unets.append(unet)
    unet = merge_unet_for_ema(unets, weights, device).to(device)
    if not os.path.exists("./work_dirs/ema/"):
        os.makedirs("./work_dirs/ema/",exist_ok=True)
    unet.save_pretrained("./work_dirs/ema/")
    pipeline.unet = unet
    pipeline = pipeline.to(device,dtype=torch.float16)
    output = pipeline(
        prompt=prompts,
        num_frames=16,
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=torch.Generator("cpu").manual_seed(42),
    ).frames
    
    if not isinstance(output, list):
        output = [output[i] for i in range(output.shape[0])]

    for j in range(len(prompts)):
        export_to_video(
            output[j],
            f"{j}-ema.mp4",
            fps=7,
        )