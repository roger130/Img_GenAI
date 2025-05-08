import torch
import torch.nn as nn
from config import Config

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            
            nn.ConvTranspose2d(Config.LATENT_DIM, Config.NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(Config.NGF * 8),
            nn.ReLU(True),
           
            nn.ConvTranspose2d(Config.NGF * 8, Config.NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.NGF * 4),
            nn.ReLU(True),
          
            nn.ConvTranspose2d(Config.NGF * 4, Config.NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.NGF * 2),
            nn.ReLU(True),
           
            nn.ConvTranspose2d(Config.NGF * 2, Config.NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.NGF),
            nn.ReLU(True),
           
            nn.ConvTranspose2d(Config.NGF, Config.CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
            
        )

    def forward(self, input):
    
        return self.main(input)

class Discriminator(nn.Module):
   
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
           
            nn.Conv2d(Config.CHANNELS, Config.NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(Config.NDF, Config.NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
           
            nn.Conv2d(Config.NDF * 2, Config.NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(Config.NDF * 4, Config.NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(Config.NDF * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
     
        return self.main(input).view(-1, 1).squeeze(1)