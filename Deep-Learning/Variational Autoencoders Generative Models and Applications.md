## Variational Autoencoders Generative Models and Applications
Slide 1: VAE Architecture Fundamentals

A Variational Autoencoder combines probabilistic inference with neural networks, creating a powerful generative model. The architecture consists of an encoder network that maps inputs to a latent distribution and a decoder network that reconstructs inputs from sampled latent vectors.

```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
```

Slide 2: Reparameterization Trick Implementation

The reparameterization trick enables backpropagation through the sampling process by expressing the random sampling as a deterministic function of the mean, variance, and an auxiliary noise variable.

```python
def reparameterize(self, mu, log_var):
    # Training: Sample from distribution
    # Testing: Use mean directly
    if self.training:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    return mu

def forward(self, x):
    # Encode input
    hidden = self.encoder(x)
    mu = self.fc_mu(hidden)
    log_var = self.fc_var(hidden)
    
    # Sample latent vector
    z = self.reparameterize(mu, log_var)
    
    # Decode
    return self.decoder(z), mu, log_var
```

Slide 3: VAE Loss Function

The VAE loss combines reconstruction loss (measuring how well the decoder reconstructs the input) with KL divergence (ensuring the learned latent distribution approaches a standard normal distribution).

```python
def vae_loss(recon_x, x, mu, log_var):
    # Reconstruction loss (Binary Cross Entropy)
    BCE = nn.functional.binary_cross_entropy(
        recon_x, x, reduction='sum'
    )
    
    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return BCE + KLD
```

Slide 4: Training Loop Implementation

A complete training loop implementation showing how to train a VAE model with proper optimization and loss calculation on batches of data.

```python
def train_vae(model, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data)
            loss = vae_loss(recon_batch, data, mu, log_var)
            
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}: Avg Loss = {avg_loss:.4f}')
```

Slide 5: MNIST Example Implementation

Implementation of a VAE for generating MNIST digits, demonstrating real-world application with proper data preprocessing and model setup.

```python
import torchvision
from torchvision import transforms

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.MNIST(
    './data', train=True, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True
)

# Initialize model and optimizer
input_dim = 28 * 28  # MNIST image size
latent_dim = 20
model = VAE(input_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
train_vae(model, train_loader, optimizer, epochs=50)
```

Slide 6: Latent Space Visualization

A crucial aspect of VAEs is understanding the learned latent space. This implementation creates a visualization of the latent space using t-SNE dimensionality reduction to observe clustering patterns.

```python
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def visualize_latent_space(model, loader):
    model.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for data, label in loader:
            data = data.view(data.size(0), -1)
            _, mu, _ = model(data)
            latent_vectors.append(mu)
            labels.append(label)
    
    latent_vectors = torch.cat(latent_vectors).numpy()
    labels = torch.cat(labels).numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    # Plot results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                         c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of VAE latent space')
    plt.show()
```

Slide 7: Advanced Sampling Techniques

Implementation of advanced sampling methods for VAE latent space exploration, including interpolation and conditional generation using learned latent representations.

```python
def latent_space_interpolation(model, start_img, end_img, steps=10):
    model.eval()
    with torch.no_grad():
        # Encode images to latent space
        start_latent = model.encoder(start_img.view(1, -1))
        end_latent = model.encoder(end_img.view(1, -1))
        
        # Create interpolation steps
        alphas = np.linspace(0, 1, steps)
        interpolated_images = []
        
        for alpha in alphas:
            # Linear interpolation in latent space
            interpolated_latent = (1 - alpha) * start_latent + alpha * end_latent
            # Decode interpolated point
            decoded = model.decoder(interpolated_latent)
            interpolated_images.append(decoded.view(28, 28))
            
        return torch.stack(interpolated_images)
```

Slide 8: Conditional VAE Implementation

Extension of the basic VAE to include conditional generation capabilities, allowing control over generated outputs based on specified conditions or labels.

```python
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(ConditionalVAE, self).__init__()
        self.num_classes = num_classes
        
        # Encoder with condition
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        
        # Decoder with condition
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
```

Slide 9: VAE with Custom Prior Distribution

Implementation of a VAE variant that uses a custom prior distribution instead of the standard normal distribution, allowing for more complex latent space structures.

```python
class VAECustomPrior(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAECustomPrior, self).__init__()
        self.prior_mean = nn.Parameter(torch.zeros(latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(latent_dim))
        
    def kl_divergence(self, mu, log_var):
        return -0.5 * torch.sum(
            1 + log_var - self.prior_logvar 
            - (mu - self.prior_mean).pow(2)/torch.exp(self.prior_logvar)
            - torch.exp(log_var - self.prior_logvar)
        )
        
    def forward(self, x):
        # Similar to standard VAE forward pass
        # but uses custom KL divergence
        pass
```

Slide 10: VAE Performance Metrics

Implementation of comprehensive evaluation metrics for VAE models, including reconstruction quality assessment and distribution matching measurements.

```python
def evaluate_vae(model, test_loader):
    model.eval()
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1)
            recon_batch, mu, log_var = model(data)
            
            # Reconstruction loss
            recon_loss = nn.functional.binary_cross_entropy(
                recon_batch, data, reduction='sum'
            )
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(
                1 + log_var - mu.pow(2) - log_var.exp()
            )
            
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    n_samples = len(test_loader.dataset)
    metrics = {
        'avg_recon_loss': total_recon_loss / n_samples,
        'avg_kl_loss': total_kl_loss / n_samples
    }
    return metrics
```

Slide 11: Beta-VAE Implementation

Implementation of Beta-VAE, a variant that introduces a hyperparameter β to control the capacity of the latent bottleneck, allowing better disentanglement of factors.

```python
def beta_vae_loss(recon_x, x, mu, log_var, beta=4.0):
    BCE = nn.functional.binary_cross_entropy(
        recon_x, x, reduction='sum'
    )
    
    # Beta-weighted KL divergence
    KLD = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp()
    )
    
    return BCE + beta * KLD

class BetaVAE(VAE):
    def __init__(self, input_dim, latent_dim, beta=4.0):
        super(BetaVAE, self).__init__(input_dim, latent_dim)
        self.beta = beta
        
    def compute_loss(self, recon_x, x, mu, log_var):
        return beta_vae_loss(recon_x, x, mu, log_var, self.beta)
```

Slide 12: VAE with Convolutional Layers

Implementation of a VAE using convolutional layers for improved performance on image data, demonstrating advanced architectural choices.

```python
class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(128 * 4 * 4, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
```

Slide 13: Advanced VAE Training Techniques

Implementation of advanced training techniques including cyclical annealing of the KL term and progressive training schedule for improved convergence and stability.

```python
class CyclicalKLScheduler:
    def __init__(self, cycle_length=1000, max_beta=1.0):
        self.cycle_length = cycle_length
        self.max_beta = max_beta
        
    def get_beta(self, iteration):
        cycle = (iteration % self.cycle_length) / self.cycle_length
        return self.max_beta * (1 / (1 + np.exp(-12 * cycle + 6)))

def train_with_cyclical_annealing(model, train_loader, optimizer, epochs):
    scheduler = CyclicalKLScheduler()
    iteration = 0
    
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(data.size(0), -1)
            beta = scheduler.get_beta(iteration)
            
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data)
            loss = beta_vae_loss(recon_batch, data, mu, log_var, beta)
            
            loss.backward()
            optimizer.step()
            iteration += 1
```

Slide 14: Practical Real-World Application

Complete implementation of a VAE for generating synthetic tabular data, including preprocessing, training, and evaluation metrics.

```python
class TabularVAE(nn.Module):
    def __init__(self, input_features, hidden_dim=128, latent_dim=20):
        super(TabularVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_features)
        )
        
    def generate_samples(self, n_samples):
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim)
            return self.decoder(z)
```

Slide 15: Additional Resources

*   "Auto-Encoding Variational Bayes" - [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
*   "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" - [https://openreview.net/forum?id=Sy2fzU9gl](https://openreview.net/forum?id=Sy2fzU9gl)
*   "Understanding disentangling in β-VAE" - [https://arxiv.org/abs/1804.03599](https://arxiv.org/abs/1804.03599)
*   Suggested searches:
    *   "Advances in Variational Inference for Deep Learning"
    *   "Deep Generative Models: Survey and Tutorials"
    *   "VAE Applications in Different Domains"

