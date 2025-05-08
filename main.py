import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from config import Config
from data import get_fashion_mnist_dataloader
from models import Generator, Discriminator, weights_init
from utils import save_individual_images, generate_fake_images, explore_latent_space

#create directories
os.makedirs(Config.RESULTS_DIR, exist_ok=True)
os.makedirs(Config.MODELS_DIR, exist_ok=True)
os.makedirs(Config.INDIVIDUAL_IMAGES_DIR, exist_ok=True)
os.makedirs(Config.EVAL_DIR, exist_ok=True)


seed = Config.SEED
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

#select MPS
if torch.backends.mps.is_available():
    device = torch.device("mps") 
elif torch.cuda.is_available():
    device = torch.device("cuda")  
else:
    device = torch.device("cpu")  

print(f"Using device: {device}")

def train_dcgan():
    #data loader for training
    dataloader = get_fashion_mnist_dataloader()
    
    
    #create generator and discriminator instances
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    #loss function and optimizers
    criterion = torch.nn.BCELoss()
    optimizerD = torch.optim.Adam(netD.parameters(), lr=Config.LR, betas=(Config.BETA1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=Config.LR, betas=(Config.BETA1, 0.999))
    
    #create fixed noise for visualization
    fixed_noise = torch.randn(64, Config.LATENT_DIM, 1, 1, device=device)
    
    #define real and fake labels
    real_label = 1.
    fake_label = 0.
    
    #lists to store training metrics
    G_losses = []
    D_losses = []
    fid_scores = []
    
    
    start_time = time.time()
    
    print("Starting Training...")
    
    #loop
    for epoch in range(Config.NUM_EPOCHS):
       
        epoch_start_time = time.time()
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{Config.NUM_EPOCHS}', leave=True)
        
   
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        batch_count = 0
        
        #batch
        for i, (data, _) in enumerate(pbar):
            batch_count += 1
            netD.zero_grad() 
            real_cpu = data.to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()  
            
            #train with fake data
            noise = torch.randn(batch_size, Config.LATENT_DIM, 1, 1, device=device)
            fake = netG(noise)  
            label.fill_(fake_label)
            
            output = netD(fake.detach()) 
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item() 
            
            errD = errD_real + errD_fake 
            optimizerD.step()  
          

            netG.zero_grad() 
            
            label.fill_(real_label) 
            
            output = netD(fake) 
            errG = criterion(output, label)  
            errG.backward()
            D_G_z2 = output.mean().item()  
            #update generator
            optimizerG.step() 
            
            #accumulate losses
            epoch_g_loss += errG.item()
            epoch_d_loss += errD.item()
            
            #store loss values
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
           
            if i % Config.PRINT_EVERY == 0:
                pbar.set_postfix({
                    'Loss_D': f'{errD.item():.4f}',
                    'Loss_G': f'{errG.item():.4f}',
                    'D(x)': f'{D_x:.4f}',
                    'D(G(z))': f'{D_G_z1:.4f}/{D_G_z2:.4f}'
                })
        
        #calculate and display epoch's average losses
        avg_g_loss = epoch_g_loss / batch_count
        avg_d_loss = epoch_d_loss / batch_count
        epoch_time = time.time() - epoch_start_time
        
        print(f'Epoch {epoch}/{Config.NUM_EPOCHS} completed in {epoch_time:.2f}s - '
              f'Avg Loss_D: {avg_d_loss:.4f}, Avg Loss_G: {avg_g_loss:.4f}')
        
        #generate test images 
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        
        #save generated images
        from torchvision.utils import make_grid
        grid = make_grid(fake, padding=2, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Epoch {epoch}")
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.savefig(f"{Config.RESULTS_DIR}/epoch_{epoch}.png")
        plt.close()
        

        
        #save model checkpoints every 5 epochs
        if epoch % 5 == 0 or epoch == Config.NUM_EPOCHS - 1:
            torch.save(netG.state_dict(), f"{Config.MODELS_DIR}/generator_epoch_{epoch}.pth")
            torch.save(netD.state_dict(), f"{Config.MODELS_DIR}/discriminator_epoch_{epoch}.pth")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    
    #plot training loss curves
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{Config.RESULTS_DIR}/loss_curves.png")
    plt.close()
    
    
    # Save final model
    torch.save(netG.state_dict(), f"{Config.MODELS_DIR}/generator_final.pth")
    torch.save(netD.state_dict(), f"{Config.MODELS_DIR}/discriminator_final.pth")
    
    return netG, netD, G_losses, D_losses, fid_scores

if __name__ == "__main__":
   
    netG, netD, G_losses, D_losses, fid_scores = train_dcgan()
    
  
    explore_latent_space(netG, device, num_images=10)
    

    final_eval_dir = f"{Config.EVAL_DIR}/final_evaluation"
    generate_fake_images(netG, device, num_images=1000, output_dir=final_eval_dir)
    
   