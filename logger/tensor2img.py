import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

def tensor2img( t, nb_cols ):
    nb_imgs = t.shape[0]
    t = t.permute( (0,1,3,2) )

    nb_rows = nb_imgs // nb_cols + int( nb_imgs % nb_cols > 0 )
    tile = torch.zeros( (t.shape[1], (t.shape[2]+1)*nb_rows-1, (t.shape[3]+1)*nb_cols-1 ), dtype=torch.float32 )

    k = 0
    for row in range(nb_rows):
        for col in range(nb_cols):
            tile[ :,
            row*(t.shape[2]+1):row*(t.shape[2]+1)+t.shape[2],
            col*(t.shape[3]+1):col*(t.shape[3]+1)+t.shape[3] ] = t[k,:,:,:]
            k += 1
    tile = tile.clamp(min=0,max=1)
    return torchvision.transforms.ToPILImage()(tile)

