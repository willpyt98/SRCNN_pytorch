import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.001)
        m.bias.data.fill_(0.0)

def calc_psnr(img1, img2):
    return 10. * torch.log10(torch.max(img1) ** 2 / torch.mean((img1 - img2) ** 2))