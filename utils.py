import torch
import torch.nn as nn
import random
import numpy as np
import os
from PIL import Image

def load_image_path():
    all_img = []
    
    path = './2014data/'
    
    folders = [f for f in os.listdir(path)]
    
    for folder in folders:
        path = os.path.join("./2014data/" , folder)
        imgs = [os.path.join("./2014data/" ,os.path.join(folder,f)) for f in os.listdir(path) 
                     if os.path.isfile(os.path.join(path, f)) 
                     and f.endswith(".JPEG")]
        all_img += imgs
    
    all_img = np.array(all_img)

    return all_img

def clean_image(all_img):
    clean_img = []
    for i in range(len(all_img)):
        img_path = all_img[i]
        image = Image.open(img_path)
        width = image.size[0]
        height = image.size[1]
        if width  > 128 and height > 128:
            clean_img.append(img_path)
    clean_img = np.array(clean_img)
    return clean_img

def train_val_split(all_img):
    # Split 95% as train image and 5% as val image
    img_size = len(all_img)
    train = np.ones(img_size, dtype=bool)
    random.seed(1008)
    train[random.sample(range(img_size), img_size // 20)] = False
    train_img = all_img[train]
    val_img = all_img[~train]
    
    return train_img, val_img 


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.001)
        m.bias.data.fill_(0.0)

def calc_psnr(img1, img2):
    return 10. * torch.log10(torch.max(img1) ** 2 / torch.mean((img1 - img2) ** 2))

