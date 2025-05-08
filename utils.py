import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from tqdm import tqdm
from config import Config

def save_individual_images(generator, device, epoch, num_images=10):

    epoch_dir = os.path.join(Config.INDIVIDUAL_IMAGES_DIR, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    generator.eval()
    with torch.no_grad():
        for i in range(num_images):
            noise = torch.randn(1, Config.LATENT_DIM, 1, 1, device=device)
            fake = generator(noise).detach().cpu()
            vutils.save_image(fake, os.path.join(epoch_dir, f'image_{i}.png'), normalize=True)
    
   

def generate_fake_images(generator, device, num_images=1000, batch_size=64, output_dir=None):
   
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    generator.eval()
    
    all_images = []
    num_batches = int(np.ceil(num_images / batch_size))
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Generating images"):
           
            current_batch_size = min(batch_size, num_images - i * batch_size)
            if current_batch_size <= 0:
                break
            noise = torch.randn(current_batch_size, Config.LATENT_DIM, 1, 1, device=device)
    
            fake = generator(noise).detach().cpu()
            
            if output_dir is not None:
                for j in range(fake.size(0)):
                    img_idx = i * batch_size + j
                    if img_idx < num_images:
                        vutils.save_image(fake[j], f"{output_dir}/fake_{img_idx}.png", normalize=True)
            
            all_images.append(fake)
    
    all_images = torch.cat(all_images, dim=0)
    if len(all_images) > num_images:
        all_images = all_images[:num_images]
        
    return all_images

def explore_latent_space(netG, device, num_images=10):
    
    print("Exploring latent space...")
    netG.eval()
    
    
    z1 = torch.randn(1, Config.LATENT_DIM, 1, 1, device=device)
    z2 = torch.randn(1, Config.LATENT_DIM, 1, 1, device=device)
    
    alphas = np.linspace(0, 1, num_images)
    interpolated_images = []
    
    for alpha in tqdm(alphas, desc="Generating interpolated images"):
        with torch.no_grad():
            z_interp = alpha * z1 + (1 - alpha) * z2
            
            fake = netG(z_interp)
            interpolated_images.append(fake.detach().cpu())
    
    grid = vutils.make_grid(torch.cat(interpolated_images), nrow=num_images, padding=2, normalize=True)
    
    
    plt.figure(figsize=(15, 3))
    plt.axis("off")
    plt.title("Latent Space Interpolation")
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.savefig(f"{Config.RESULTS_DIR}/latent_interpolation.png")
    plt.close()
    
    print("Latent space exploration completed!")