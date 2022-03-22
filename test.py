import torch
from torchvision.io import read_image,ImageReadMode
from torchvision.utils import save_image
from sr_cnn import SR_CNN
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-file', type=str, required=True)
    args = parser.parse_args()
    
    net = SR_CNN()
    
    state_dict = net.state_dict()
    
    weight_file = './result/epoch_14.pth'
    net.load_state_dict(torch.load(weight_file))
    net.eval()
    
    image = read_image(args.image_file, ImageReadMode.RGB).float()
    height = image.size()[1]
    width = image.size()[2]
    transforms_train = torch.nn.Sequential(
        transforms.Resize(size = (height // 2, width //2),  interpolation=InterpolationMode.BICUBIC),
        transforms.Resize(size =(height, width),interpolation=InterpolationMode.BICUBIC)
    )
    y = transforms_train(image)

    save_image(y / 255., args.image_file.replace('.', '_bicubic.'))
    
    
    with torch.no_grad():
       pred = net(y)
       
    save_image(pred / 255., args.image_file.replace('.', '_srcnn.'))

