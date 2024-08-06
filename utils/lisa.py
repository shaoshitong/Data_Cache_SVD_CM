import torch
import torch.nn as nn
import torch.nn.functional as F
import math,random
import numpy as np
import gc
import accelerate

class LISADiffusion:
    def __init__(self, model, spatial_head=None, rate=None, dtype=torch.float16, 
                 grad_aware=False, accelerator=None):
        self.model = model
        self.spatial_head = spatial_head
        self.dtype = dtype
        self.grad_aware = grad_aware
        if self.spatial_head is not None:
            for param in self.spatial_head.parameters():
                param.requires_grad = True
        self.rate = rate
        self.accelerator = accelerator
        self.last_epoch = 0
        num_sort = []
        name_sort = []
        lisa_target_name = self.get_better_parameters_name
        for i, (name, param) in enumerate(list(model.named_parameters())[1:-1]):
            num_sort.append([i+1, param.numel()])
            name_sort.append(name)
        self.allowed_index = [j[0] for j in sorted(num_sort,key=lambda x:x[1],reverse=False)]
        self.probability = [1. for j in range(len(self.allowed_index))]
        for i, j in enumerate(self.allowed_index):
            self.probability[j-1] = (0.999 ** i)
            if any([(module_key in name_sort[j-1]) for module_key in lisa_target_name]):
                self.probability[j-1] *= 10.
        p_sum = sum(self.probability)
        self.probability = [j/p_sum for j in self.probability]
        self.initialize()

    def freeze_all_layers(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def random_activate_layers(self, model, p):
        activate_number = int((len(list(model.parameters()))-2) * p)
        index = np.random.choice(range(1,len(list(model.parameters()))-1,1), activate_number, replace=False, p=self.probability)
        count = 0
        for param in model.parameters():
            if count == 0 or count == len(list(model.parameters()))-1:
                if torch.distributed.get_rank() == 0:
                    print(f"Record {count}, Mean: {param.mean().item()}, Std: {param.std().item()}")
                param.requires_grad = True
                param.data = param.data.to(dtype=torch.float32)
            elif count in index:
                param.requires_grad = True
                param.data = param.data.to(dtype=torch.float32)        
            else:
                param.requires_grad = False
                param.data = param.data.to(dtype=self.dtype)
            count += 1
        count_number = 0
        pos_count_number = 0
        for param in model.parameters():
            if param.requires_grad:
                pos_count_number += param.numel()
            count_number += param.numel()
        if torch.distributed.get_rank() == 0:
            print("Total Trainable Parameters:", round(pos_count_number*100/count_number,2),"%")


    def lisa(self, model, p=0.25):
        self.freeze_all_layers(model)
        self.random_activate_layers(model, p)

    def lisa_recall(self):
        param_number = len(list(self.model.parameters()))
        lisa_p = 8 / param_number if self.rate is None else self.rate
        self.lisa(model=self.model,p=lisa_p)
        self.last_epoch += 5
        # count = 0
        # for p in self.model.parameters():
        #     if (id(p) in self.optimizer_dict) and (count != 0 or count!=len(list(self.model.parameters()))-1):
        #         self.optimizer_dict[id(p)].state_dict().clear()
        #         self.scheduler_dict[id(p)].state_dict().clear()
        #         del self.optimizer_dict[id(p)]
        #         del self.scheduler_dict[id(p)]
        #     count += 1
        
        self.model.zero_grad()
        if self.accelerator is not None:
            self.accelerator.clear()
        gc.collect()
        torch.cuda.empty_cache()
        
        if self.grad_aware:
            new_probability = []
            for i, param in enumerate(list(self.model.parameters())[1:-1]):
                if self.grad_record.get(param, None) is None or len(self.grad_record[param]) < 200:
                    v = 1. if len(new_probability) == 0 else max(new_probability)
                else:
                    v = torch.stack(self.grad_record[param],0).std().item() + 1e-6
                new_probability.append(v)
            new_probability = [j/sum(new_probability) for j in new_probability]
            for i, new_p in enumerate(new_probability):
                self.probability[i] = self.probability[i] * 0.01 + new_p * 0.99
            self.probability = [j/sum(self.probability) for j in self.probability]

        
    def initialize(self):
        self.optimizer_dict = dict()
        self.scheduler_dict = dict()
        self.grad_record = dict()

    def register(self, optimizer_class=None, get_scheduler=None, accelerator=None, 
                 optim_kwargs={}, sched_kwargs={}):
        
        self.lisa_recall()
        for p in self.model.parameters():
            self.grad_record[id(p)] = []  
            if p.requires_grad:
                self.optimizer_dict[id(p)] = optimizer_class([{"params":p}], **optim_kwargs)      
                if accelerator is not None:
                    self.optimizer_dict[id(p)] = accelerator.prepare_optimizer(self.optimizer_dict[id(p)])

        for p in self.model.parameters():
            if p.requires_grad:
                self.scheduler_dict[id(p)] = get_scheduler(optimizer=self.optimizer_dict[id(p)], **sched_kwargs)
                if accelerator is not None:
                    self.scheduler_dict[id(p)] = accelerator.prepare_scheduler(self.scheduler_dict[id(p)])
        
        if self.spatial_head is not None:
            for p in self.spatial_head.parameters():
                if p.requires_grad:
                    self.optimizer_dict[id(p)] = optimizer_class([{"params":p}], **optim_kwargs)
                    if accelerator is not None:
                        self.optimizer_dict[id(p)] = accelerator.prepare_optimizer(self.optimizer_dict[id(p)])

            for p in self.spatial_head.parameters():
                if p.requires_grad:
                    self.scheduler_dict[id(p)] = get_scheduler(optimizer=self.optimizer_dict[id(p)], **sched_kwargs)
                    if accelerator is not None:
                        self.scheduler_dict[id(p)] = accelerator.prepare_scheduler(self.scheduler_dict[id(p)])
    
    def insert_hook(self, optimizer_class=None, get_scheduler=None, accelerator=None, 
                 optim_kwargs={}, sched_kwargs={}):
        
        for param in self.model.parameters():
            param.requires_grad = True
            
        def optimizer_hook(p):
            if p.grad is None:
                self.optimizer_dict[id(p)].zero_grad(set_to_none=True)
                self.optimizer_dict[id(p)].state_dict().clear()
                self.scheduler_dict[id(p)].state_dict().clear()
                del self.optimizer_dict[id(p)]
                del self.scheduler_dict[id(p)]
                return
            else:
                if self.grad_aware:
                    with torch.no_grad():
                        self.grad_record[id(p)].append(((p ** 2).sum() / p.numel()))
                        while len(self.grad_record[id(p)]) > 200:
                            self.grad_record[id(p)].pop(0)
                        
                if id(p) not in self.optimizer_dict:
                    self.optimizer_dict[id(p)] = optimizer_class([{"params":p}], **optim_kwargs)
                    if accelerator is not None:
                        self.optimizer_dict[id(p)] = accelerator.prepare_optimizer(self.optimizer_dict[id(p)])    
                if id(p) not in self.scheduler_dict:
                    self.scheduler_dict[id(p)] = get_scheduler(optimizer=self.optimizer_dict[id(p)], **sched_kwargs)
                    self.scheduler_dict[id(p)].last_epoch = self.last_epoch
                    if accelerator is not None:
                        self.scheduler_dict[id(p)] = accelerator.prepare_scheduler(self.scheduler_dict[id(p)])

            if accelerator is not None and accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(p, 10.0)
            
            self.optimizer_dict[id(p)].step()
            self.optimizer_dict[id(p)].zero_grad(set_to_none=True)
            self.scheduler_dict[id(p)].step()

        # Register the hook onto every parameter
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
                
        def spatial_optimizer_hook(p):
            self.optimizer_dict[id(p)].step()
            self.optimizer_dict[id(p)].zero_grad(set_to_none=True)
            self.scheduler_dict[id(p)].step()
        
        if self.spatial_head is not None:
            for p in self.spatial_head.parameters():
                if p.requires_grad:
                    p.register_post_accumulate_grad_hook(spatial_optimizer_hook)
            
    @property      
    def get_better_parameters_name(self):
        lisa_target_modules = [
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
                "conv1",
                "conv2",
                "conv_shortcut",
                "downsamplers.0.conv",
                "upsamplers.0.conv",
                "time_emb_proj",
            ]
        return lisa_target_modules