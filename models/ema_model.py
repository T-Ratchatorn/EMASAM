import torch
import torch.nn as nn

import copy
import math
import random

class MovingAverageModel(nn.Module):
    def __init__(self, model: nn.Module, alpha: float = 0.99, zeros=True):
        """
        model: 元のモデル
        alpha: 移動平均の係数 (0 < alpha < 1)
        """
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)  # EMAモデルの勾配を無効化

        if( zeros ):
            for ema_param in self.ema_model.parameters():
                nn.init.zeros_( ema_param )

    @torch.no_grad()
    def ema_update(self):
        if( self.alpha > 0 ):
            """EMA モデルのパラメータを更新"""
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.alpha).add_(model_param.data, alpha=1 - self.alpha)

            # BatchNorm の running statistics を更新
            for ema_buffer, model_buffer in zip(self.ema_model.buffers(), self.model.buffers()):
                if( ema_buffer.dtype is not torch.float ):
                    ema_buffer = ema_buffer.to(dtype=torch.float)
                ema_buffer.mul_(self.alpha).add_(model_buffer, alpha=1 - self.alpha)

    @torch.no_grad()
    def proc_after_step(self):
        self.ema_update()

    def forward(self, x):
        if( self.training or self.alpha <= 0 ):
            return self.model(x)

        return self.ema_model(x)


class SAMMovingAverageModel(nn.Module):
    def __init__(self, model: nn.Module, alpha: float = 0.99, ema_sch_end = 20, rho: float = 1.5, normalize = False, zeros=True):
        """
        model: 元のモデル
        alpha: 移動平均の係数 (0 < alpha < 1)
        """
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.ema_sch_end = ema_sch_end
        self.normalize = normalize
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)  # EMAモデルの勾配を無効化
        self.org_model = copy.deepcopy(self.model)
        self.org_model.requires_grad_(False)

        self.rho = rho

        if( zeros ):
            for ema_param in self.ema_model.parameters():
                nn.init.zeros_( ema_param )

    @torch.no_grad()
    def proc_before_train(self, current_ep, return_perturb = False):
        total_sq = 0.0
        if( self.alpha > 0 ):
            for model_buffer, org_buffer in zip(self.model.buffers(), self.org_model.buffers()):
                org_buffer = model_buffer.clone()
                
            if self.normalize:
                denom_sq = 0.0
                for ema_param, model_param in zip( self.ema_model.parameters(), self.model.parameters() ):
                    distance = (model_param.data - ema_param.data)
                    denom_sq += distance.pow(2).sum().item()
                norm = (denom_sq ** 0.5) # global L2 norm of the difference
                scale = (self.rho / norm) if norm > 0 else 0.0
            else:
                scale = self.rho

            if current_ep < self.ema_sch_end:
                     scale = 0
            
            for ema_param, org_param, model_param in zip( self.ema_model.parameters(), self.org_model.parameters(), self.model.parameters() ):
                org_param.data = model_param.data.clone()
                distance = (model_param.data - ema_param.data)
                perturb = distance * scale
                model_param.data = perturb + ema_param.data
                if return_perturb:
                    total_sq += perturb.pow(2).sum().item()
                    
        if return_perturb:
            return total_sq**0.5 #sqrt of sum of squares → L2 norm
        else:
            return None

    @torch.no_grad()
    def proc_before_step(self):
        if( self.alpha > 0 ):
            for org_param, model_param in zip(self.org_model.parameters(), self.model.parameters()):
                model_param.data = org_param.data

    @torch.no_grad()
    def proc_after_step(self):
        if( self.alpha > 0 ):
            """EMA モデルのパラメータを更新"""
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.alpha).add_(model_param.data, alpha=1 - self.alpha)

            # BatchNorm の running statistics を更新
            for ema_buffer, org_buffer in zip(self.ema_model.buffers(), self.org_model.buffers()):
                if( ema_buffer.dtype is not torch.float ):
                    ema_buffer = ema_buffer.to(dtype=torch.float)
                ema_buffer.mul_(self.alpha).add_(org_buffer, alpha=1 - self.alpha)

    def forward(self, x):
        if( self.training ):
            with torch.no_grad():
                self.org_model(x)
            return self.model(x)
        else:
            if( self.alpha <= 0 ):
                return self.model(x)
            else:
                return self.ema_model(x)