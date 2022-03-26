import torch
from PIL import Image
from torchvision.utils import save_image
from sr_cnn import SR_CNN
from torchvision import transforms
from utils import calc_psnr
from torch.nn.functional import interpolate

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--net-file', type=str, required=True)
    args = parser.parse_args()
    
    net = SR_CNN()
    
    state_dict = net.state_dict()
    
    net.load_state_dict(torch.load(args.net_file))
    net.eval()
    
    image = Image.open(args.image_file).convert('RGB')
    trans = transforms.Compose([transforms.ToTensor()])
    image = trans(image)
    height = image.size()[1]
    width = image.size()[2]
    low_res = interpolate(image.unsqueeze(0), scale_factor=0.5)
    bicubic = interpolate(low_res, size=[height, width], mode='bicubic').squeeze(0)

    with torch.no_grad():
       pred = net(bicubic)
       
    save_image(bicubic, args.image_file.replace('.', '_bicubic.'))
    save_image(pred, args.image_file.replace('.', '_srcnn.'))
    metric_SRCNN= calc_psnr(image[:, 6:-6, 6:-6], pred)
    metric_bicubic = calc_psnr(image,bicubic)
    print("SRCNN.psnr = {}".format(metric_SRCNN))
    print("bicubic.psnr = {}".format(metric_bicubic))

