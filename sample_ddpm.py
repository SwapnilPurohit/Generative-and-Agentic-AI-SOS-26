import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from unet import SimpleUNet
from diffusion_forward import linear_beta_schedule

def sample_ddpm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load('models/ddpm_mnist.pth', map_location=device))
    model.eval()
    
    timesteps = 200
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Pre-calculate sampling terms
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas # simplified version
    
    # Generate 16 images
    num_images = 16
    x = torch.randn((num_images, 1, 28, 28)).to(device)
    
    # Store history for visualization
    history = []
    
    print("Sampling...")
    with torch.no_grad():
        for t in reversed(range(timesteps)):
            t_tensor = torch.full((num_images,), t, device=device, dtype=torch.long)
            
            predicted_noise = model(x, t_tensor)
            
            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            
            # Equation: x_{t-1} = 1/sqrt(alpha) * (x_t - ((1-alpha)/sqrt(1-alpha_cumprod)) * predicted_noise) + sigma * z
            mean = sqrt_recip_alphas[t] * (x - ((1.0 - alpha) / sqrt_one_minus_alphas_cumprod[t]) * predicted_noise)
            
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(posterior_variance[t])
                x = mean + sigma * noise
            else:
                x = mean
                
            if t % 40 == 0 or t == timesteps - 1:
                history.append(x.cpu().clone())
                print(f"Step {t} done")

    # Save sampling process
    fig, axes = plt.subplots(1, len(history), figsize=(15, 3))
    # Pick the first image from the batch to show the process
    for i, hist_x in enumerate(history):
        img = hist_x[0] # first image
        img = (img + 1.0) / 2.0
        img = torch.clamp(img, 0, 1)
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Step {timesteps - 1 if i==0 else (len(history)-i-1)*40}")
    
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "sampling_process.png"))
    plt.close()
    
    # Save final grid
    final_images = history[-1]
    final_images = (final_images + 1.0) / 2.0 # scale back to [0,1]
    final_images = torch.clamp(final_images, 0, 1)
    
    grid = make_grid(final_images, nrow=4, padding=2, normalize=False)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title("Generated MNIST Digits")
    plt.savefig(os.path.join("outputs", "generated_digits.png"))
    plt.close()
    
    print("Sampling complete. Images saved to 'outputs/'")

if __name__ == "__main__":
    sample_ddpm()
