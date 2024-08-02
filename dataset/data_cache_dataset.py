import torch
import numpy as np
import os, copy
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from prefetch_generator import BackgroundGenerator
from multiprocessing import Pool
import multiprocessing
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

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class CustomDataset(Dataset):
    def __init__(self, feature_dir, rank):
        if isinstance(rank,list):
            self.feature_dir = [os.path.join(feature_dir,'rank{:02d}'.format(i)) for i in rank]
        else:
            self.feature_dir = os.path.join(feature_dir,'rank{:02d}'.format(rank))
        self.rank = rank
        self.cache = None
        self.files = [[os.path.join(feature_dir,j) for j in sorted(os.listdir(feature_dir), reverse=False) if j.endswith(".npz")] for feature_dir in self.feature_dir]
        self.files = [item for subl in self.files for item in subl]
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        try:
            file = np.load(file_name)
            self.cache = file
            noisy_model_input = file["noisy_model_input"]
            target = file["target"]
            sd_prompt_embeds = file["sd_prompt_embeds"]
            prompt_embeds = file["prompt_embeds"]
            start_timesteps = file["start_timesteps"]
            timesteps = file["timesteps"]
            text = str(file["text"][0])
        except:
            file = self.cache if self.cache is not None else np.load(self.files[0])
            noisy_model_input = file["noisy_model_input"]
            target = file["target"]
            sd_prompt_embeds = file["sd_prompt_embeds"]
            prompt_embeds = file["prompt_embeds"]
            start_timesteps = file["start_timesteps"]
            timesteps = file["timesteps"]
            text = str(file["text"][0])
            
        return dict(noisy_model_input=torch.from_numpy(noisy_model_input), 
                    target=torch.from_numpy(target),
                    sd_prompt_embeds=torch.from_numpy(sd_prompt_embeds),
                    prompt_embeds=torch.from_numpy(prompt_embeds),
                    start_timesteps=start_timesteps, 
                    timesteps=timesteps, 
                    text=text)



if __name__ == "__main__":
    
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="vae",
        revision=None,
    ) 
    vae.requires_grad_(False)
    vae = vae.cuda()
    import einops
    dataset = CustomDataset("/home/shaoshitong/extract_code_dir_scope_8/", [0,])
    results = []
    index = 0
    for data in dataset:
        noisy_model_input, target, clip_emb, prompt_embeds, \
            pred_x_0, global_step, local_rank, use_pred_x0, \
        start_timesteps, timesteps = data
        image = vae.decode(einops.rearrange(target.float().cuda(),"b c t h w -> (b t) c h w") / vae.config.scaling_factor).sample
        if index >= 64:
            results.append(image[0])
        if index == 80:
            break
        index +=1
    results = torch.stack(results,0)
    print(results.shape)
    from torchvision.utils import save_image
    save_image(results, "sample1.png", nrow=4, normalize=True, value_range=(-1, 1))