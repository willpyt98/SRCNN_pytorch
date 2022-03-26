from PIL import Image
from torch.utils.data import Dataset
from torch.nn.functional import interpolate

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transforms != None:
            image = self.transforms(image)
        
        label = image        
        label = label[:, 6:-6, 6:-6]
        
        low_res = interpolate(image.unsqueeze(0), scale_factor=0.5)
        train = interpolate(low_res, size=[128, 128], mode='bicubic').squeeze(0)
            
        return train, label
