import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from unet import SimpleUNet
from diffusion_forward import linear_beta_schedule, q_sample

def train_ddpm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
    
    model = SimpleUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    timesteps = 200
    epochs = 5
    
    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for step, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            # Sample random timesteps
            t = torch.randint(0, timesteps, (images.shape[0],), device=device).long()
            
            # Generate random noise
            noise = torch.randn_like(images).to(device)
            
            # Add noise to images (forward process)
            # q_sample needs to be modified slightly to work well with batches and devices
            # So we re-implement it here cleanly for batch processing
            betas = linear_beta_schedule(timesteps).to(device)
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t])[:, None, None, None]
            
            x_noisy = sqrt_alphas_cumprod_t * images + sqrt_one_minus_alphas_cumprod_t * noise
            
            # Predict noise
            predicted_noise = model(x_noisy, t)
            
            # Calculate loss (L_simple)
            loss = nn.functional.mse_loss(predicted_noise, noise)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")
        
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/ddpm_mnist.pth')
    print("Training complete. Model saved to 'models/ddpm_mnist.pth'")

if __name__ == "__main__":
    train_ddpm()
