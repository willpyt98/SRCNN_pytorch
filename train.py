import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from sr_cnn import SR_CNN
from dataset import CustomImageDataset
from image_path import load_image_path, train_val_split

from tqdm import tqdm
import time
import os

def validation(val_loader):
    print("Start calulating validation loss")
    val_loss = 0
    num_batches = 0
    
    for val_data, val_label in val_loader:
        val_data=val_data.to(device)
        val_label=val_label.to(device)
        pred=net( val_data ) 
        loss =  criterion( pred ,val_label) 
        val_loss += loss.detach().item()
        num_batches+=1 
    
    return val_loss / num_batches

if __name__ == '__main__':
    
    all_img = load_image_path()
    train_img, val_img =  train_val_split(all_img)
    
    transforms_label = torch.nn.Sequential(
        transforms.CenterCrop(size = (128,128))
    )

    transforms_train = torch.nn.Sequential(
        transforms.CenterCrop(size = (128,128)),
        transforms.Resize(64, interpolation=InterpolationMode.BICUBIC),
        transforms.Resize(128, interpolation=InterpolationMode.BICUBIC)
    )

    train_dataset = CustomImageDataset(train_img, transforms_train, transforms_label)
    
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=True
    )
    
    val_dataset = CustomImageDataset(val_img, transforms_train, transforms_label)
    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=True, drop_last=True
    )
    
    net=SR_CNN()
    device = torch.device('cuda')
    net = net.to(device)
    
    outputs_dir = './result'
    
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
        
    bs = 128
    criterion = nn.MSELoss()
    lr= torch.cat((1e-6 * torch.ones(10), 1e-7 * torch.ones(10)), dim = 0)
    optimizer=optim.SGD( [{'params':net.conv1.parameters()},
                            {'params':net.conv2.parameters()},
                            {'params':net.conv3.parameters(), 'lr':1e-6}
                        ], weight_decay=0.0005,lr=1e-7, momentum = 0.9)
    
    start=time.time()
    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(15):
            
        epoch_lr = lr[epoch]
        # create a new optimizer at the beginning of each epoch: give the current learning rate.  


            
        # set the running quatities to zero at the beginning of the epoch
        running_loss=0
        num_batches=0
        
        with tqdm(total=(len(train_dataset))- len(train_dataset) % bs) as t:
            t.set_description('epoch: {}/{}'.format(epoch+1, 15))
     
            for minibatch_data,minibatch_label in train_loader:
    
                # Set the gradients to zeros
                optimizer.zero_grad()
    
                # send them to the gpu
                minibatch_data=minibatch_data.to(device)
                minibatch_label=minibatch_label.to(device)
                
                minibatch_data.requires_grad_()    
                
                pred=net( minibatch_data ) 
                loss =  criterion( pred , minibatch_label) 
                loss.backward()
                optimizer.step()
    
    
                # START COMPUTING STATS
                running_loss += loss.detach().item()     
                num_batches+=1     
                t.set_postfix(running_loss='{:.3f}'.format(running_loss/num_batches))
                t.update(bs)
        
        # compute stats for the full training set
        total_loss = running_loss/num_batches
        
        
        val_loss = validation(val_loader)
        train_loss_hist.append(total_loss)
        val_loss_hist.append(val_loss)
        
        elapsed = (time.time()-start)/60
    
        print('epoch=',epoch, '\t time=', elapsed,'min','\t train_loss=', total_loss, '\t val_loss=',val_loss)
        print(' ')
        
        torch.save(net.state_dict(), os.path.join(outputs_dir, 'epoch_{}.pth'.format(epoch)))
    