import torch.nn as nn

class SR_CNN(nn.Module):

    def __init__(self):

        super(SR_CNN, self).__init__()

        # CL1:  filter 3 x 3 , in_chanel = 3, out_chanel = 64
        self.conv1 = nn.Conv2d(3,   64,  kernel_size=9 )
        
        # CL2:  filter 1 x 1 , in_chanel = 64, out_chanel = 32
        self.conv2 = nn.Conv2d(64,  32,  kernel_size=1 )
        
        # CL3: filter 5 x 5, in_chanel = 32, out_chanel = 3
        self.conv3 = nn.Conv2d(32,  3,  kernel_size=5)
        
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):

        # CL1:   3 x 128 x 128 x 3 -> 64 x 128 x 128
        x = self.conv1(x)
        x = self.relu(x)
        
        # CL2: 64 x 128 x 128 -> 32 x 128 x 128
        x = self.conv2(x)
        x = self.relu(x)
        
        # CL3:   32 x 128 x 128  -->  3 x 128 x 128
        x = self.conv3(x)
    
        return x