import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Create an output directory for plots
os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------
# 1. Hyperparameters & Setup
# ---------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 5      # 5 epochs is enough to see a structure in the latent space
LEARNING_RATE = 1e-3
LATENT_DIM = 2  # We use 2D so we can easily plot it!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------
# 2. Data Loading (MNIST)
# ---------------------------------------------------------
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------
# 3. Model Definition: Variational Autoencoder (VAE)
# ---------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        
        # --- ENCODER ---
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # --- DECODER ---
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ---------------------------------------------------------
# 4. Loss Function (ELBO)
# ---------------------------------------------------------
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ---------------------------------------------------------
# 5. Training Loop
# ---------------------------------------------------------
model = VAE(LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

for epoch in range(1, EPOCHS + 1):
    train(epoch)

print("Training finished.")

# ---------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------
print("Generating latent space visualization...")
model.eval()
latent_z = []
latent_labels = []

with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device)
        mu, logvar = model.encode(data.view(-1, 784))
        latent_z.append(mu.cpu())
        latent_labels.append(labels)

latent_z = torch.cat(latent_z, dim=0).numpy()
latent_labels = torch.cat(latent_labels, dim=0).numpy()

plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_z[:, 0], latent_z[:, 1], c=latent_labels, cmap='tab10', alpha=0.6, s=5)
plt.colorbar(scatter, ticks=range(10))
plt.title("VAE 2D Latent Space on MNIST")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.grid(True, alpha=0.3)
plt.savefig("outputs/latent_space.png", dpi=300, bbox_inches='tight')
plt.close()

print("Generating reconstructions visualization...")
with torch.no_grad():
    sample_data, _ = next(iter(test_loader))
    sample_data = sample_data.to(device)
    recon, _, _ = model(sample_data)
    
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        axes[0, i].imshow(sample_data[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].axis('off')
    
    axes[0, 0].set_title("Original", loc='left')
    axes[1, 0].set_title("Reconstruction", loc='left')
    plt.savefig("outputs/reconstructions.png", dpi=300, bbox_inches='tight')
    plt.close()

print("All done! Plots saved to the 'outputs' directory.")
