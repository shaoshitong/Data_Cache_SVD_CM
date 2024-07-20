import torch
import torch.nn as nn
import torch.nn.functional as F
import math,random
import numpy as np
import gc
import accelerate


class LISADiffusion:
    def __init__(self, model, spatial_head, rate=None, dtype=torch.float16):
        self.model = model
        self.spatial_head = spatial_head
        self.dtype = dtype
        for param in self.spatial_head.parameters():
            param.requires_grad = True
        self.rate = rate
        self.last_epoch = 0
        num_sort = []
        for i, param in enumerate(list(model.parameters())[1:-1]):
            num_sort.append([i+1,param.numel()])
        self.allowed_index = [j[0] for j in sorted(num_sort,key=lambda x:x[1],reverse=False)]
        self.probability = [1. for j in range(len(self.allowed_index))]
        for i, j in enumerate(self.allowed_index):
            if j > int(0.7 * len(self.probability)):
                self.probability[j-1] = 0
            else:
                self.probability[j-1] = (0.999 ** i)
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
                # if torch.distributed.get_rank() == 0:
                #     print(f"Count {count}, Mean: {param.mean().item()}, Std: {param.std().item()}")
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
                pos_count_number+=param.numel()
            count_number +=param.numel()
        if torch.distributed.get_rank() == 0:
            print("Total Trainable Parameters:", round(pos_count_number*100/count_number,2),"%")
        torch.cuda.empty_cache()
        gc.collect()

    def lisa(self, model, p=0.25):
        self.freeze_all_layers(model)
        self.random_activate_layers(model, p)

    def lisa_recall(self):
        param_number = len(list(self.model.parameters()))
        lisa_p = 8 / param_number if self.rate is None else self.rate
        self.lisa(model=self.model,p=lisa_p)
        self.last_epoch += 1
        count = 0
        for p in self.model.parameters():
            if (p in self.optimizer_dict) and (count != 0 or count!=len(list(self.model.parameters()))-1):
                del self.optimizer_dict[p]
                del self.scheduler_dict[p]
            count += 1

        print("="*60,"begin","="*60)
        
        for k, v in self.optimizer_dict.items():
            print(id(k),id(v))
        
        print("="*60,"end","="*60)

    def initialize(self):
        self.optimizer_dict = dict()
        self.scheduler_dict = dict()

    def register(self, optimizer_class=None, get_scheduler=None, accelerator=None, 
                 optim_kwargs={}, sched_kwargs={}):
        
        self.lisa_recall()
        for p in self.model.parameters():
            if p.requires_grad:
                self.optimizer_dict[p] = optimizer_class([{"params":p}], **optim_kwargs)
                if accelerator is not None:
                    self.optimizer_dict[p] = accelerator.prepare_optimizer(self.optimizer_dict[p])

        for p in self.model.parameters():
            if p.requires_grad:
                self.scheduler_dict[p] = get_scheduler(optimizer=self.optimizer_dict[p], **sched_kwargs)
                if accelerator is not None:
                    self.scheduler_dict[p] = accelerator.prepare_scheduler(self.scheduler_dict[p])
        
        for p in self.spatial_head.parameters():
            if p.requires_grad:
                self.optimizer_dict[p] = optimizer_class([{"params":p}], **optim_kwargs)
                if accelerator is not None:
                    self.optimizer_dict[p] = accelerator.prepare_optimizer(self.optimizer_dict[p])

        for p in self.spatial_head.parameters():
            if p.requires_grad:
                self.scheduler_dict[p] = get_scheduler(optimizer=self.optimizer_dict[p], **sched_kwargs)
                if accelerator is not None:
                    self.scheduler_dict[p] = accelerator.prepare_scheduler(self.scheduler_dict[p])
    
    def insert_hook(self, optimizer_class=None, get_scheduler=None, accelerator=None, 
                 optim_kwargs={}, sched_kwargs={}):
        
        for param in self.model.parameters():
            param.requires_grad = True
            
        def optimizer_hook(p):
            if p.grad is None:
                del self.scheduler_dict[p]
                del self.optimizer_dict[p]
                return
            else:
                if p not in self.optimizer_dict:
                    self.optimizer_dict[p] = optimizer_class([{"params":p}], **optim_kwargs)
                    if accelerator is not None:
                        self.optimizer_dict[p] = accelerator.prepare_optimizer(self.optimizer_dict[p])    
                if p not in self.scheduler_dict:
                    self.scheduler_dict[p] = get_scheduler(optimizer=self.optimizer_dict[p], **sched_kwargs)
                    self.scheduler_dict[p].last_epoch = self.last_epoch
                    if accelerator is not None:
                        self.scheduler_dict[p] = accelerator.prepare_scheduler(self.scheduler_dict[p])

            if accelerator is not None and accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(p, 1.0)
            
            self.optimizer_dict[p].step()
            self.optimizer_dict[p].zero_grad(set_to_none=True)
            self.scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
                
        def spatial_optimizer_hook(p):
            self.optimizer_dict[p].step()
            self.optimizer_dict[p].zero_grad(set_to_none=True)
            self.scheduler_dict[p].step()
        
        for p in self.spatial_head.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(spatial_optimizer_hook)