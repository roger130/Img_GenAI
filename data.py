import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from config import Config

def get_fashion_mnist_dataloader(batch_size=Config.BATCH_SIZE, for_fid=False):
    '''
    Fashion-MNIST Dataloader (training/fid)
    batch_size
    for_fid: for fid or not
    DataLoader
    '''
    #img transform definition
    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.CenterCrop(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  
    ])
    
   
    dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    if not for_fid:
        print("sucessfully load Fashion-MNIST dataset")
    
    
    if for_fid and Config.NUM_FID_IMAGES < len(dataset):
        indices = torch.randperm(len(dataset))[:Config.NUM_FID_IMAGES]
        dataset = Subset(dataset, indices)
    
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True if not for_fid else False
    )
    
    return dataloader