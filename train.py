import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from sr_cnn import SR_CNN
from dataset import CustomImageDataset
from utils import init_weights, train_val_split, load_image_path, clean_image

from tqdm import tqdm
import time
import os
import argparse

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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default = 128)
    parser.add_argument('--num-epoch', type=int, default = 50)
    parser.add_argument('--optimizer', type=str, default = 'Adam')
    parser.add_argument('--output-dir',type=str, required=True)
    args = parser.parse_args()
    
    all_img = load_image_path()
    all_img = clean_image(all_img)
    train_img, val_img =  train_val_split(all_img)
    
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop((128, 128)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Use the mean and std of ImageNet
        #transforms.Normalize([0.4949,0.4345,0.3825], [0.2843,0.2775,0.2764])
    ])

    train_dataset = CustomImageDataset(train_img, transforms)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    
    val_dataset = CustomImageDataset(val_img, transforms)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    
    net=SR_CNN()
    
    device = torch.device('cuda')
    net = net.to(device)
    
    outputs_dir = args.output_dir
    
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
        
    bs = args.batch_size
    criterion = nn.MSELoss()
    
    if args.optimizer == 'Adam':
        optimizer = optim.Adam([
            {'params': net.conv1.parameters()},
            {'params': net.conv2.parameters()},
            {'params': net.conv3.parameters(), 'lr':1e-5 }
        ], lr=1e-4)
        
    if args.optimizer == 'SGD':
        optimizer = optim.SGD([
            {'params': net.conv1.parameters()},
            {'params': net.conv2.parameters()},
            {'params': net.conv3.parameters(), 'lr':1e-5 }
        ], lr=1e-4, momentum=0.9)
    
    start=time.time()
    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(args.num_epoch):
                        
        # set the running quatities to zero at the beginning of the epoch
        running_loss=0
        num_batches=0
        
        with tqdm(total=(len(train_dataset))- len(train_dataset) % bs) as t:
            t.set_description('epoch: {}/{}'.format(epoch+1, args.num_epoch))
     
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
                t.set_postfix(running_loss='{:.6f}'.format(running_loss/num_batches))
                t.update(bs)
        
        # compute stats for the full training set
        total_loss = running_loss/num_batches
        
        
        val_loss = validation(val_loader)
        train_loss_hist.append(total_loss)
        val_loss_hist.append(val_loss)
        
        elapsed = (time.time()-start)/60
    
        print('epoch=',epoch, '\t time=', elapsed,'min','\t train_loss=', total_loss, '\t val_loss=',val_loss)
        print(' ')
        
        torch.save(net.state_dict(), os.path.join(outputs_dir, 'Adam/epoch_{}.pth'.format(epoch+1)))
    
    torch.save(net.state_dict(), os.path.join(outputs_dir, 'Adam/final_net.pth'))