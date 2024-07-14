import logging
import os
from typing import List, Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.utils.spectral_norm import SpectralNorm
from torchvision.transforms import Normalize, RandomCrop
from torchvision.models import resnet18
from .diffaug import DiffAugment
from .shared import ResidualBlock

logger = logging.getLogger(__name__)


class Discriminator(nn.Module):
    def __init__(self, checkpoint=None, discriminator_mode=False, feature_dim=768):
        super(Discriminator,self).__init__()
        self.main = resnet18()
        self.main.conv1 = nn.Conv2d(4, self.main.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.main.bn1 = nn.Identity()
        self.main.relu = nn.Identity()
        self.main.maxpool = nn.Identity()
        self.feature_dim = feature_dim
        if checkpoint is not None:
            state_dict = torch.load(checkpoint,map_location="cpu")
            if 'net' in state_dict.keys():
                state_dict = state_dict['net']
            if not 'cmapper1.0.weight' in state_dict.keys():
                self.load_state_dict(state_dict)
                tag = 0
            else:
                tag = 1
        self.discriminator_mode = discriminator_mode
        if discriminator_mode:
            self.main.fc = nn.Linear(512, 1) # 512
            self.cmapper1 = nn.Sequential(nn.Linear(feature_dim, 64),nn.ReLU(),nn.Linear(64,self.main.layer1[-1].conv2.out_channels*2))
            self.cmapper2 = nn.Sequential(nn.Linear(feature_dim, 64),nn.ReLU(),nn.Linear(64,self.main.layer2[-1].conv2.out_channels*2))
            self.cmapper3 = nn.Sequential(nn.Linear(feature_dim, 64),nn.ReLU(),nn.Linear(64,self.main.layer3[-1].conv2.out_channels*2))
            self.main.layer4.relu = nn.ReLU(inplace=False)
            if tag == 1 and checkpoint is not None:
                self.load_state_dict(state_dict)
            
    def forward(self, x, c=None):
        if not self.discriminator_mode:
            c1 = self.cmapper1(c).unsqueeze(-1).unsqueeze(-1)
            a1, b1 = torch.split(c1,int(c1.shape[1]//2),dim=1)
            c2 = self.cmapper2(c).unsqueeze(-1).unsqueeze(-1)
            a2, b2 = torch.split(c2,int(c2.shape[1]//2),dim=1)
            c3 = self.cmapper3(c).unsqueeze(-1).unsqueeze(-1)
            a3, b3 = torch.split(c3,int(c3.shape[1]//2),dim=1)
            x = self.main.conv1(x)
            x = (self.main.layer1(x)+ a1) * (1 + b1)
            x = (self.main.layer2(x)+ a2) * (1 + b2)
            x = (self.main.layer3(x)+ a3) * (1 + b3)
            x = self.main.layer1(x)
            x = self.main.layer2(x)
            x = self.main.layer3(x)
            x = self.main.layer4(x)
            x = self.main.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.main.fc(x)
            return x
        else:
            return self.main(x).mean(1,keepdim=True)
    
    def save(self,path):
        torch.save(self.state_dict(),path)