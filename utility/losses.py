import torch
import torch.nn as nn
import torch.nn.functional as F

def smooth_crossentropy(pred, gold, smoothing=0.0):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

class CrossEntropyNorm(nn.Module):
    def __init__(self, label_smoothing=0):
        super(CrossEntropyNorm, self).__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none' )
    def forward( self, x, t ):
        ce = self.ce( x, t )
        ce = ce.sum() / (ce.detach()>0).to(dtype=torch.float).sum()
        return ce