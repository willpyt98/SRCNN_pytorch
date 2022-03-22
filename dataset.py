from torchvision.io import read_image,ImageReadMode
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transforms_train=None, transforms_label =None):
        self.img_dir = img_dir
        self.transforms_train = transforms_train
        self.transforms_label = transforms_label

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = read_image(img_path, ImageReadMode.RGB).float()
        train = image
        label = image
        
        if self.transforms_label != None:
            label = self.transforms_label(label)
        
        
        if self.transforms_train != None:
            train = self.transforms_train(train)
            
        return train, label
