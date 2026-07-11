import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import math
import os

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

timesteps = 200
betas_linear = linear_beta_schedule(timesteps)
betas_cosine = cosine_beta_schedule(timesteps)

# Plot schedules
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(betas_linear.numpy(), label='Linear Schedule', color='blue')
plt.plot(betas_cosine.numpy(), label='Cosine Schedule', color='orange')
plt.title(r"Variance Schedules ($\beta_t$)")
plt.xlabel(r"Timestep $t$")
plt.ylabel(r"$\beta_t$")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join("outputs", "noise_schedules.png"))
plt.close()

# Forward process visualization
transform = transforms.Compose([
    transforms.ToTensor(),
])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
image, _ = mnist[0] # Get first image
image = image * 2.0 - 1.0 # Scale to [-1, 1]

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    alphas = 1. - betas_linear
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t])
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# Plot forward process at different timesteps
t_vals = [0, 40, 80, 120, 160, 199]
fig, axes = plt.subplots(1, len(t_vals), figsize=(15, 3))

for i, t in enumerate(t_vals):
    if t == 0:
        noisy_image = image
    else:
        # t is 0-indexed, so we pass t-1 conceptually but here t is an index.
        noisy_image = q_sample(image, torch.tensor([t]))
    
    # Scale back to [0, 1] for plotting
    noisy_image = (noisy_image + 1.0) / 2.0
    noisy_image = torch.clamp(noisy_image, 0, 1)
    
    axes[i].imshow(noisy_image.squeeze(), cmap='gray')
    axes[i].set_title(f"$t={t}$")
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join("outputs", "forward_process.png"))
plt.close()

print("Forward process plots generated successfully in 'outputs' directory.")
