import torch.optim

from .lr_sch import WarmupCosineLR, WarmupLinearLR, StepLR

#################################################
def _gen_opt( _params_, name, keys, **cfg ):
    args = {}
    for k, v in cfg.items():
        if( k[:4] == 'opt_' ):
            k = k[4:]
        if( k in keys ):
            args[k]=v
    return eval( '{}( _params_, **args )'.format(name) )


def gen_opt( _params_, name=None, **cfg ):
    if( name is None ):
        name = cfg['opt']

    name = name.lower()
    if( name == 'sgd' ):
        keys = ['lr', 'momentum', 'weight_decay', 'nesterov']
        opt = _gen_opt( _params_, 'torch.optim.SGD', keys, **cfg )

    elif( name == 'adam' ):
        keys = ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad']
        opt = _gen_opt( _params_, 'torch.optim.Adam', keys, **cfg )

    else:
        raise NotImplementedError(name)

    return opt

#################################################
def _gen_sch( _opt_, name, keys, **cfg ):
    args = {}
    for k, v in cfg.items():
        if( k[:4] == 'sch_' ):
            k = k[4:]
        if( k in keys ):
            args[k]=v
    return eval( '{}( _opt_, **args )'.format(name) )

def gen_sch( _opt_, **cfg ):
    name = None
    if( name is None ):
        name = cfg['sch']

    name = name.lower()
    if( name == 'coslr' or name == 'cos' or name == "cosineannealinglr" ):
        keys = ['T_max', 'eta_min', 'last_epoch', 'verbose']
        sch = _gen_sch( _opt_, "CosineAnnealingLR", keys, **cfg )

    elif( name == 'steplr' or name == 'step' ):
        keys = ['step_size', 'gamma', 'last_epoch', 'verbose']
        sch = _gen_sch( _opt_, "StepLR", keys, **cfg )

    elif( name == 'multisteplr' or name == 'multistep' ):
        keys = ['milestones', 'gamma', 'last_epoch', 'verbose']
        sch = _gen_sch( _opt_, "MultiStepLR", keys, **cfg )

    elif( name == 'linear' ):
        ratio = LinearDecaySch( start_epoch=cfg["sch_start"], end_epoch=cfg["sch_end"], end_ratio=cfg["sch_gamma"] )
        sch = LambdaLR( _opt_, lr_lambda = ratio )

    else:
        raise NotImplementedError(name)

    return sch
