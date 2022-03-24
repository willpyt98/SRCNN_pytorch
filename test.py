import torch
from torchvision.io import read_image,ImageReadMode
from PIL import Image
from torchvision.utils import save_image
from sr_cnn import SR_CNN
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from utils import calc_psnr

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-file', type=str, required=True)
    args = parser.parse_args()
    
    net = SR_CNN()
    
    state_dict = net.state_dict()
    
    weight_file = './result/final_net.pth'
    net.load_state_dict(torch.load(weight_file))
    net.eval()
    
    image = Image.open(args.image_file).convert('RGB')
    trans = transforms.Compose([transforms.ToTensor()])
    image = trans(image)
    height = image.size()[1]
    width = image.size()[2]
    transforms_train = transforms.Compose([
        transforms.Resize(size = (height // 2, width //2),  interpolation=InterpolationMode.BICUBIC),
        transforms.Resize(size =(height, width),interpolation=InterpolationMode.BICUBIC)
    ])
    bicubic = transforms_train(image)

    save_image(bicubic, args.image_file.replace('.', '_bicubic.'))
    
    with torch.no_grad():
       pred = net(bicubic)
       
    save_image(pred, args.image_file.replace('.', '_srcnn.'))
    metric_SRCNN= calc_psnr(image[:, 6:-6, 6:-6], pred)
    metric_bicubic = calc_psnr(image,bicubic)
    print("SRCNN.psnr = {}".format(metric_SRCNN))
    print("bicubic.psnr = {}".format(metric_bicubic))

