import torch
import numpy as np
import os, copy
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from prefetch_generator import BackgroundGenerator
from multiprocessing import Pool
import multiprocessing

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
        self.files = [[os.path.join(feature_dir,j) for j in sorted(os.listdir(feature_dir), reverse=True) if j.endswith(".npz")] for feature_dir in self.feature_dir]
        self.files = [item for subl in self.files for item in subl]
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        try:
            file = np.load(file_name)
            self.cache = copy.deepcopy(file)
        except:
            file = copy.deepcopy(self.cache) if self.cache is not None else np.load(self.files[0])
        noisy_model_input = file["noisy_model_input"]
        target = file["target"]
        clip_emb = file["clip_emb"]
        gt_sample = file["gt_sample"]
        gt_sample_clip_emb = file["gt_sample_clip_emb"]
        prompt_embeds = file["prompt_embeds"]
        
        global_step = file["global_step"]
        local_rank = file["local_rank"]
        use_pred_x0 = file["use_pred_x0"]
        start_timesteps = file["start_timesteps"]
        timesteps = file["timesteps"]
        return torch.from_numpy(noisy_model_input), torch.from_numpy(target), \
               torch.from_numpy(clip_emb), torch.from_numpy(gt_sample), \
               torch.from_numpy(gt_sample_clip_emb), torch.from_numpy(prompt_embeds), global_step, \
                local_rank, int(use_pred_x0), start_timesteps, timesteps



if __name__ == "__main__":
    dataset = CustomDataset("/data/shaoshitong/extract_code_dir_scope/", [0,])
    for data in dataset:
        noisy_model_input, target, clip_emb, gt_sample, \
        gt_sample_clip_emb, prompt_embeds, global_step, local_rank, use_pred_x0, \
        start_timesteps, timesteps = data
        print(noisy_model_input.shape, target.shape, clip_emb.shape, gt_sample.shape, gt_sample_clip_emb.shape)
    
    # torch.Size([1, 4, 16, 32, 32]) torch.Size([1, 4, 16, 32, 32]) torch.Size([1, 1024]) torch.Size([2, 3, 256, 256]) torch.Size([1, 1024])