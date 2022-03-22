import os
import numpy as np
import random

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

def train_val_split(all_img):
    # Split 95% as train image and 5% as val image
    img_size = len(all_img)
    train = np.ones(img_size, dtype=bool)
    train[random.sample(range(img_size), img_size // 20)] = False
    train_img = all_img[train]
    val_img = all_img[~train]
    
    return train_img, val_img 
    
    
        