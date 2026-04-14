import argparse
import os
import yaml
import csv
import torch
import torch.optim
from datetime import datetime
from statistics import mean

from utility.initialize import initialize
# from utility.opt_sch_gen import gen_opt, gen_sch
from utility.checkpoint import load_checkpoint
from utility.lr_sch import WarmupCosineLR, WarmupLinearLR, StepLR, reveres_square_decay_sch
from utility.losses import smooth_crossentropy
from utility.log import Log
from utility.csv_plot import csv_plot
from models.ema_model import MovingAverageModel, SAMMovingAverageModel
from models.utils import get_model
from datasets.utils import get_data

#####################################################################
parser = argparse.ArgumentParser(description='')

parser.add_argument('--config', default=None)
parser.add_argument('--log_dir', default='.results/')
parser.add_argument('--log_name', default='0' )
parser.add_argument('--resume_chkpt', default=None)
parser.add_argument("--chkpt_freq", type=int, default="0")
parser.add_argument("--prev_check_rm", action='store_true', default=False)
parser.add_argument('--num_worker', type=int, default=4)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--tsubame_id", type=str, default=None) #only when using ScienceTokyo's Tsubame supercomputer

parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--method', type=str, default="emasam")
parser.add_argument('--model', type=str, default='ResNet18' )
parser.add_argument('--dataset', type=str, default="cifar100")
parser.add_argument('--num_classes', type=int, default=100)

parser.add_argument('--loss', type=str, default='ce')
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--grad_clip', type=float, default=0.0)

parser.add_argument('--ema_alpha', type=float, default=0.9) # 0
parser.add_argument('--ema_sch_start', type=int, default=0)
parser.add_argument('--ema_sch_end', type=int, default=0)

parser.add_argument('--rho', type=float, default=2.0)

parser.add_argument('--opt', type=str, default="sgd")
parser.add_argument('--opt_momentum', type=float, default=0.0) # 0.9 / 0.0
parser.add_argument('--opt_adamW_beta1', type=float, default=0.0) # 0.9 / 0.0
parser.add_argument('--opt_adamW_beta2', type=float, default=0.999)
parser.add_argument('--opt_weight_decay',  type=float, default=0.0005)
parser.add_argument('--opt_lr', type=float, default=0.1)

parser.add_argument('--sch', type=str, default='step')
parser.add_argument('--sch_gamma', type=float) #0.2
parser.add_argument('--sch_milestones', type=list) #[60, 120, 160]
parser.add_argument('--sch_start', type=int) #0
parser.add_argument('--sch_end', type=int) #0
parser.add_argument('--warmup_steps', type=int)

#####################################################################
cfg = vars( parser.parse_args() ) #create dict from parser

# pop from cfg, they are not saved in yaml
config = cfg.pop('config')
log_dir = cfg.pop('log_dir')
log_name = cfg.pop('log_name')
resume_chkpt = cfg.pop('resume_chkpt')
chkpt_freq = cfg.pop('chkpt_freq')
prev_check_rm = cfg.pop('prev_check_rm')
num_worker = cfg.pop('num_worker')
gpu = cfg.pop('gpu')
tsubame_id = cfg.pop('tsubame_id')

if not os.path.isdir(log_dir):
    os.makedirs(log_dir, exist_ok=True)
print( 'log dir: ', log_dir )

cfg_file = {}
if(config is not None):
    if(os.path.exists(config)):
        with open(config, 'r') as yin:
            cfg_file = yaml.load(yin, Loader=yaml.SafeLoader)
        print(f'read {config}')
else: 
    print("no config file")
        
for k in sorted(cfg.keys()):
    if(k in cfg_file.keys()):
        cfg[k] = cfg_file[k]
        print( 'file:', k, cfg[k] )
    else:
        print( 'args:', k, cfg[k] )
print(f"mils: {cfg['sch_milestones']}")
with open(os.path.join(log_dir, "config.yaml"), "a") as fout:
    yaml.dump(cfg, fout, explicit_end=True)

#####################################################################
initialize(cfg["random_seed"])
device = gpu

model = get_model(cfg['model'], cfg['dataset'], **cfg)
dataset = get_data(cfg['dataset'], cfg['num_classes'], cfg['batch_size'], num_worker, tsubame_id)

#####################################################################
"""model wrapper and optimizer wrapper"""
if cfg["method"].lower() == "emasam":
    print("emasam")
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    print(f"Found {len(bn_layers)} BatchNorm layer(s) in the model.")
    for bn in bn_layers:
        bn.momentum = 1
    model = SAMMovingAverageModel(model, alpha=cfg["ema_alpha"], ema_sch_end = cfg["ema_sch_end"], rho=cfg["rho"], normalize = True)
elif cfg["method"].lower() == "ema_orig":
    print("ema_orig")
    model = MovingAverageModel(model, alpha=cfg["ema_alpha"])
elif cfg["method"].lower() == "vanilla":
    print("vanilla")
else:
    raise NotImplementedError(cfg["method"])


if cfg["opt"].lower() == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['opt_lr'], momentum=cfg['opt_momentum'], weight_decay=cfg['opt_weight_decay'])
elif cfg["opt"].lower() == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['opt_lr'], betas=(cfg['opt_adamW_beta1'], cfg['opt_adamW_beta2']), weight_decay=cfg['opt_weight_decay'])
else:
    raise NotImplementedError(cfg["opt"])

#####################################################################
if cfg["sch"].lower() == "step":
    scheduler = StepLR(optimizer, init_lr = cfg['opt_lr'], milestones = cfg["sch_milestones"], gamma = cfg["sch_gamma"])
elif cfg["sch"].lower() == "cosine":
    total_steps = (len(dataset.train) * cfg['epochs'])
    scheduler = WarmupCosineLR(optimizer, peak_lr = cfg['opt_lr'], warmup_steps = cfg['warmup_steps'], total_steps = total_steps)
else:
    raise NotImplementedError(cfg["sch"])

model = model.to(device)

start_epoch = 1
if resume_chkpt:
    model, optimizer, scheduler, loaded_epoch = load_checkpoint(resume_chkpt, model, optimizer, scheduler)
    start_epoch = loaded_epoch + 1
count_step = (start_epoch - 1) * len(dataset.train)

csv_path = os.path.join(log_dir, "log.csv")

#####################################################################
if start_epoch == 1:
    header = ["time_start", "time_finish", "epoch", "lr", "ema_alpha", "avg_perturbation", "train_loss", "train_acc", "val_loss", "val_acc"]
    with open(csv_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
log = Log(log_each=10, initial_epoch=start_epoch)

for epoch in range(start_epoch, cfg['epochs']+1):
    t_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alpha = 0
    if( cfg["ema_alpha"] > 0 and cfg["ema_sch_start"] >= 0 and cfg["ema_sch_end"] > 0 ):
        ratio = reveres_square_decay_sch(epoch, start_epoch = cfg["ema_sch_start"], end_epoch = cfg["ema_sch_end"])
        alpha = cfg["ema_alpha"] * ratio
        model.alpha = alpha

    model.train()
    
    result_list = []
    log.train(step_per_epoch=len(dataset.train))
    ew_norm_list = []

    for batch in dataset.train:
        count_step += 1
        inputs, targets = (b.to(device) for b in batch)

        if( cfg["rho"] != 0 and cfg["method"].lower() in ["emasam", "norm_emasam"]):
                ew_norm = model.proc_before_train(current_ep = epoch, return_perturb = cfg['verbose'])
                if cfg['verbose']:
                    ew_norm_list.append(ew_norm)
        
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = smooth_crossentropy(predictions, targets, smoothing=cfg['label_smoothing'])
        loss.mean().backward()
        if cfg['grad_clip'] != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])

        if( cfg["rho"] != 0 and cfg["method"].lower() in ["emasam", "norm_emasam"]):
            model.proc_before_step()
            
        optimizer.step()

        if( cfg["ema_alpha"] > 0 and cfg["method"].lower() != "vanilla"):
            model.proc_after_step()

        with torch.no_grad():
            correct = torch.argmax(predictions.data, 1) == targets
            log(model, loss.cpu(), correct.cpu(), scheduler.lr())
            if scheduler.name in ["WarmupCosineLR", "WarmupLinearLR"]:
                scheduler(count_step)
            elif scheduler.name == "StepLR":
                scheduler(epoch)

    if cfg['verbose']:
        if epoch >= cfg["ema_sch_end"]:
            avg_ew_norm = mean(ew_norm_list)
        else:
            avg_ew_norm = 0
    else:
        avg_ew_norm = "NaN"
    
    t_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    model.eval()
    train_loss, train_acc, lr = log.output_0()
    log.eval(step_per_epoch=len(dataset.test))
    with torch.no_grad():
        for batch in dataset.test:
            inputs, targets = (b.to(device) for b in batch)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            correct = torch.argmax(predictions, 1) == targets
            log(model, loss.cpu(), correct.cpu())
    val_loss, val_acc = log.output_1()
    
    result_list.extend([t_start, t_end, epoch, lr, alpha, avg_ew_norm, train_loss, train_acc, val_loss, val_acc])
    with open(csv_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(result_list)
    csv_plot(csv_path, "epoch", "train_loss", "train_loss")
    csv_plot(csv_path, "epoch", "train_acc", "train_acc")
    csv_plot(csv_path, "epoch", "val_loss", "val_loss")
    csv_plot(csv_path, "epoch", "val_acc", "val_acc")

    if chkpt_freq != 0 and epoch % chkpt_freq == 0 and epoch != 0:
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
        except:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
        torch.save(checkpoint, f"{log_dir}/{log_name}_{epoch}_chkpt.pth")
        print(f"Checkpoint Save: {log_dir}/{log_name}_{epoch}_chkpt.pth")
        prev_path = f"{log_dir}/{log_name}_{epoch - chkpt_freq}_chkpt.pth"
        if prev_check_rm and os.path.exists(prev_path):
            try:
                os.remove(prev_path)
                print(f"Deleted previous checkpoint: {prev_path}")
            except Exception as e:
                print(f"Warning: could not delete {prev_path}: {e}")

try:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
except:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
torch.save(checkpoint, f"{log_dir}/{log_name}_Final_chkpt.pth")

log.flush()

    

        
        

