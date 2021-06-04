import torch
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print("weights_init_normal:", classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1, 0.02)
        init.constant_(m.bias.data,0)

# net.apply(weight_init_normal)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        init.constant(m.bias.data,0)

def weights_init_ortogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.ortogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.orthogonal(m.weight.data, gain=1)
        init.constant(m.bias.data, 0)


