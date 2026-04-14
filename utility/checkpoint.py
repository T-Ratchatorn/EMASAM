import torch

# def load_checkpoint(filepath, model, optimizer, scheduler):
#     checkpoint = torch.load(filepath)
#     model.module.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#     epoch = checkpoint['epoch']
#     return model, optimizer, scheduler, epoch

def load_checkpoint(filepath, model, optimizer, scheduler):
    try:
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
    except:
        checkpoint = torch.load(filepath)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
    return model, optimizer, scheduler, epoch