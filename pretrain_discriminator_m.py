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
import os
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
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
from models.discriminator_m import (
    Discriminator,
)

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.data import SubsetRandomSampler
import os
import shutil
import argparse
import numpy as np
import models
import torchvision
import torchvision.transforms as transforms
from bisect import bisect_right
import time

scaler=torch.cuda.amp.GradScaler()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data-path', default='/home/shaoshitong/data/adm/imagenet_code_c2i_flip_ten_crop_sdv1_ema/', type=str, help='Dataset directory')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--milestones', default=[30,60,90], type=list, help='milestones for lr-multistep')
parser.add_argument('--warmup-epoch', default=0, type=int, help='warmup epoch')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='directory fot storing checkpoints')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--gpu-id', default="0,1,2,3,4,5,6,7", type=str,
                    help='number of ranks for distributed training')
parser.add_argument('--manual_seed', type=int, default=42)
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')                    

class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = 'flip' in self.feature_dir

        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        # self.feature_files = sorted(os.listdir(feature_dir))
        # self.label_files = sorted(os.listdir(label_dir))
        # TODO: make it configurable
        self.feature_files = [f"{i}.npy" for i in range(1281167)]
        self.label_files = [f"{i}.npy" for i in range(1281167)]

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir
                   
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))
        if self.flip:
            if features.shape[1] > 200:
                aug_idx = torch.randint(low=0, high=features.shape[0], size=(1,)).item()
                features = features[aug_idx]
            else:
                aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
                features = features[:, aug_idx]
        labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)

def build_imagenet_code(data_path):
    feature_dir = f"{data_path}/imagenet256_codes"
    label_dir = f"{data_path}/imagenet256_labels"
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: extract_codes_c2i.sh ..."
    return CustomDataset(feature_dir, label_dir)

def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    port_id = 10000 + np.random.randint(0, 1000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
    if not os.path.exists("./result"):
        os.makedirs('result/',exist_ok=True)
    args.log_txt = 'result/'+ str(os.path.basename(__file__).split('.')[0]) + '_'+\
            'imagenet_code' +'.txt'


    args.log_dir = str(os.path.basename(__file__).split('.')[0]) + '_'+\
            'imagenet_code' + str(args.manual_seed)

    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.set_printoptions(precision=4)

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.log_dir)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    with open(args.log_txt, 'a+') as f:
        f.write("==========\nArgs:{}\n==========".format(args) + '\n')
    print('==> Building model..')
    args.distributed = args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node', ngpus_per_node)
    args.world_size = ngpus_per_node * args.world_size
    print('multiprocessing_distributed')
    torch.multiprocessing.set_start_method('spawn')
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                        world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(args.gpu)

    model = Discriminator(None, False)
    net = model.cuda(args.gpu)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
    
    args.batch_size = int(args.batch_size / ngpus_per_node)
    print(f"local batch size is {args.batch_size}")
    args.workers = 4
    cudnn.benchmark = True
    cudnn.fastest=True
    criterion_cls = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.cuda()

    trainable_list = nn.ModuleList([])
    trainable_list.append(net)
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=0.1, momentum=0.9, weight_decay=args.weight_decay)


    if args.resume:
        print('load intermediate weights from: {}'.format(os.path.join(args.checkpoint_dir, "discriminator" + '.pth.tar')))
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, "discriminator" + '.pth.tar'),
                                map_location='cpu')
        net.module.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    
    train_set = build_imagenet_code(args.data_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    
    def train(epoch, criterion_list, optimizer, iter_number):
        train_loss = 0.
        train_loss_div = 0.
        top1_num = 0
        top5_num = 0
        total = 0
        if epoch >= args.warmup_epoch:
            lr = adjust_lr(optimizer, epoch, args)
        net.train()
        for batch_idx, (input, target) in enumerate(trainloader):
            batch_start_time = time.time()
            input = input.float().cuda()
            b, c, h, w = input.shape[0], 4, 32, 32
            input = input.view(-1, c, h, w) / (0.18215)
            target = target.view(-1)
            target = target.cuda()
            if epoch < args.warmup_epoch:
                lr = adjust_lr(optimizer, epoch, args, batch_idx, len(trainloader))
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                logits = net(input).float()
            loss_div = torch.tensor(0.).cuda()
            loss_div = loss_div + criterion_cls(logits, target)
            loss = loss_div
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 20.)
            scaler.step(optimizer)
            scaler.update()
            iter_number=iter_number+1
            train_loss += loss.item() / len(trainloader)

            top1, top5 = correct_num(logits, target, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += target.size(0)
            if batch_idx%100==0:
                print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
                    epoch, batch_idx, len(trainloader), lr, time.time()-batch_start_time, (top1_num/(total)).item()))


    iter_number=0
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train(epoch, criterion_list, optimizer,iter_number)
        if args.rank == 0:
            state = {
                'net': net.module.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, "discriminator" + '.pth.tar'))



def correct_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k)
    return res


def adjust_lr(optimizer, epoch, args, step=0, all_iters_per_epoch=0, eta_min=0.):
    cur_lr = 0.
    if epoch < args.warmup_epoch:
        cur_lr = args.init_lr * float(1 + step + epoch*all_iters_per_epoch)/(args.warmup_epoch *all_iters_per_epoch)
    else:
        epoch = epoch - args.warmup_epoch
        cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr

if __name__ == '__main__':
    main()
    
        


        

        
