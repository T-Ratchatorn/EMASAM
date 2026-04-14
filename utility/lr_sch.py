import math

class WarmupCosineLR:
    def __init__(self, optimizer, peak_lr, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.name = "WarmupCosineLR"

    def __call__(self, step):
        self.current_step = step
        if step < self.warmup_steps:
            lr = self.peak_lr * (step / self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = 0.5 * self.peak_lr * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return {
            'peak_lr': self.peak_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'current_step': self.current_step
        }

    def load_state_dict(self, state_dict):
        self.peak_lr = state_dict['peak_lr']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.current_step = state_dict['current_step']
        # Ensure the optimizer has the correct learning rate
        self(self.current_step)


class WarmupLinearLR:
    def __init__(self, optimizer, min_lr, max_lr, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.name = "WarmupLinearLR"

    def __call__(self, step):
        self.current_step = step
        if step < self.warmup_steps:
            lr = self.max_lr * (step / self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.max_lr - (progress * (self.max_lr - self.min_lr))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return {
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'current_step': self.current_step
        }

    def load_state_dict(self, state_dict):
        self.min_lr = state_dict['min_lr']
        self.max_lr = state_dict['max_lr']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.current_step = state_dict['current_step']
        # Ensure the optimizer has the correct learning rate
        self(self.current_step)


class StepLR:
    def __init__(self, optimizer, init_lr: float, milestones: list, gamma: float):
        self.optimizer = optimizer
        self.milestones = milestones
        self.init_lr = init_lr
        self.gamma = gamma
        self.current_epoch = 0
        self.name = "StepLR"

    def __call__(self, epoch):
        self.current_epoch = epoch
        if self.current_epoch < self.milestones[0]:
            self.current_lr = self.init_lr
        elif self.current_epoch < self.milestones[1]:
            self.current_lr = self.init_lr * self.gamma
        elif self.current_epoch < self.milestones[2]:
            self.current_lr = self.init_lr * (self.gamma ** 2)
        else:
            self.current_lr = self.init_lr * (self.gamma ** 3)
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] =  self.current_lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self):
        return {
            'init_lr': self.init_lr,
            'milestones': self.milestones,
            'current_epoch': self.current_epoch,
            'current_lr': self.current_lr
        }

    def load_state_dict(self, state_dict):
        self.init_lr = state_dict['init_lr']
        self.milestones = state_dict['milestones']
        self.current_epoch = state_dict['current_epoch']
        self.current_lr = state_dict['current_lr']
        # Ensure the optimizer has the correct learning rate
        self(self.current_epoch)


def reveres_square_decay_sch( epoch, start_epoch, end_epoch ):
    if( epoch < start_epoch ):
        ratio = 0
    elif( epoch >= end_epoch ):
        ratio = 1
    else:
        tau = (end_epoch - start_epoch)
        pr = (epoch - start_epoch) / tau
        ratio = pr**2 / tau**2

    return ratio
