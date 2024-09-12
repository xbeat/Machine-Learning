## Variational Autoencoders A Python-Powered Generative Modeling Approach
Slide 1: Introduction to Variational Autoencoders (VAEs)

Variational Autoencoders (VAEs) are powerful generative models that combine ideas from deep learning and probabilistic inference. They learn to encode data into a latent space and then decode it back, allowing for both data compression and generation of new samples.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Example usage
vae = VAE(input_dim=784, latent_dim=20)
```

Slide 2: The Encoder Network

The encoder in a VAE transforms input data into a probability distribution in the latent space. It outputs parameters (mean and variance) of this distribution, typically assumed to be Gaussian.

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # Mean
        self.fc22 = nn.Linear(400, latent_dim)  # Log variance
    
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

# Example usage
encoder = Encoder(input_dim=784, latent_dim=20)
x = torch.randn(1, 784)
mu, logvar = encoder(x)
print(f"Mean shape: {mu.shape}, Log variance shape: {logvar.shape}")
```

Slide 3: The Reparameterization Trick

The reparameterization trick allows backpropagation through the random sampling process. It separates the deterministic and stochastic parts of the latent variable generation.

```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# Example usage
mu = torch.zeros(1, 20)
logvar = torch.zeros(1, 20)
z = reparameterize(mu, logvar)
print(f"Sampled z shape: {z.shape}")
```

Slide 4: The Decoder Network

The decoder takes samples from the latent space and reconstructs the original input data. It learns to map the latent representation back to the data space.

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 400)
        self.fc2 = nn.Linear(400, output_dim)
    
    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h1))

# Example usage
decoder = Decoder(latent_dim=20, output_dim=784)
z = torch.randn(1, 20)
reconstructed_x = decoder(z)
print(f"Reconstructed x shape: {reconstructed_x.shape}")
```

Slide 5: The VAE Loss Function

The VAE loss function consists of two terms: the reconstruction loss and the KL divergence. This balances the quality of reconstructions with the regularity of the latent space.

```python
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Example usage
x = torch.randn(1, 784)
recon_x = torch.sigmoid(torch.randn(1, 784))
mu = torch.zeros(1, 20)
logvar = torch.zeros(1, 20)
loss = vae_loss(recon_x, x, mu, logvar)
print(f"VAE loss: {loss.item()}")
```

Slide 6: Training a VAE

Training a VAE involves optimizing both the encoder and decoder networks simultaneously using the combined loss function.

```python
def train_vae(vae, optimizer, data_loader, epochs):
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Example usage (assuming MNIST dataset)
vae = VAE(input_dim=784, latent_dim=20)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
# train_loader would be your DataLoader object
# train_vae(vae, optimizer, train_loader, epochs=10)
```

Slide 7: Generating New Samples

After training, we can use the VAE to generate new samples by sampling from the latent space and passing through the decoder.

```python
def generate_samples(vae, num_samples):
    with torch.no_grad():
        z = torch.randn(num_samples, vae.latent_dim)
        samples = vae.decoder(z)
    return samples

# Example usage
vae = VAE(input_dim=784, latent_dim=20)
# Assuming the VAE has been trained
new_samples = generate_samples(vae, num_samples=10)
print(f"Generated samples shape: {new_samples.shape}")
```

Slide 8: Latent Space Interpolation

VAEs allow for smooth interpolation in the latent space, enabling the generation of semantically meaningful intermediate samples.

```python
def interpolate(vae, start_img, end_img, steps):
    start_z = vae.encoder(start_img)[0]  # Get mean
    end_z = vae.encoder(end_img)[0]  # Get mean
    
    interpolations = []
    for alpha in torch.linspace(0, 1, steps):
        z = start_z * (1-alpha) + end_z * alpha
        interpolation = vae.decoder(z)
        interpolations.append(interpolation)
    
    return torch.stack(interpolations)

# Example usage
start_img = torch.randn(1, 784)
end_img = torch.randn(1, 784)
interpolations = interpolate(vae, start_img, end_img, steps=10)
print(f"Interpolations shape: {interpolations.shape}")
```

Slide 9: Conditional VAEs

Conditional VAEs (CVAEs) extend VAEs by incorporating additional information to guide the generation process.

```python
class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(torch.cat([z, c], dim=1)), mu, logvar

# Example usage
cvae = CVAE(input_dim=784, condition_dim=10, latent_dim=20)
x = torch.randn(1, 784)
c = torch.randn(1, 10)
output, mu, logvar = cvae(x, c)
print(f"CVAE output shape: {output.shape}")
```

Slide 10: Disentangled VAEs

Disentangled VAEs aim to learn interpretable and independent factors in the latent space. One popular approach is the β-VAE, which introduces a hyperparameter β to control the degree of disentanglement.

```python
def beta_vae_loss(recon_x, x, mu, logvar, beta=4.0):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

class BetaVAE(VAE):
    def __init__(self, input_dim, latent_dim, beta=4.0):
        super(BetaVAE, self).__init__(input_dim, latent_dim)
        self.beta = beta
    
    def loss_function(self, recon_x, x, mu, logvar):
        return beta_vae_loss(recon_x, x, mu, logvar, self.beta)

# Example usage
beta_vae = BetaVAE(input_dim=784, latent_dim=20, beta=4.0)
x = torch.randn(1, 784)
recon_x, mu, logvar = beta_vae(x)
loss = beta_vae.loss_function(recon_x, x, mu, logvar)
print(f"β-VAE loss: {loss.item()}")
```

Slide 11: VAEs for Anomaly Detection

VAEs can be used for anomaly detection by comparing the reconstruction error of normal vs. anomalous samples.

```python
def anomaly_score(vae, x):
    recon_x, mu, logvar = vae(x)
    recon_error = F.mse_loss(recon_x, x, reduction='none').sum(dim=1)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return recon_error + kl_div

# Example usage
vae = VAE(input_dim=784, latent_dim=20)
normal_sample = torch.randn(1, 784)
anomalous_sample = torch.randn(1, 784) * 2  # Increased variance

normal_score = anomaly_score(vae, normal_sample)
anomalous_score = anomaly_score(vae, anomalous_sample)

print(f"Normal sample score: {normal_score.item()}")
print(f"Anomalous sample score: {anomalous_score.item()}")
```

Slide 12: VAEs for Image Inpainting

VAEs can be used for image inpainting by encoding the partial image and reconstructing the full image.

```python
def inpaint(vae, partial_img, mask):
    with torch.no_grad():
        # Encode partial image
        _, mu, _ = vae(partial_img)
        # Decode to get full image
        reconstructed = vae.decoder(mu)
        # Combine original and reconstructed parts
        inpainted = partial_img * mask + reconstructed * (1 - mask)
    return inpainted

# Example usage
vae = VAE(input_dim=784, latent_dim=20)
original_img = torch.rand(1, 784)
mask = torch.ones(1, 784)
mask[:, 300:500] = 0  # Create a "hole" in the image
partial_img = original_img * mask

inpainted_img = inpaint(vae, partial_img, mask)
print(f"Inpainted image shape: {inpainted_img.shape}")
```

Slide 13: Real-life Example: Image Generation for Art

VAEs can be used in digital art creation to generate new, unique artworks based on existing styles or themes.

```python
import torchvision.transforms as transforms
from PIL import Image

def generate_artwork(vae, style_image, num_variations=5):
    # Preprocess the style image
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    style_tensor = transform(Image.open(style_image)).unsqueeze(0)
    
    # Encode the style image
    with torch.no_grad():
        _, mu, _ = vae(style_tensor)
    
    # Generate variations
    variations = []
    for _ in range(num_variations):
        z = mu + torch.randn_like(mu) * 0.1  # Add small random variations
        variation = vae.decoder(z)
        variations.append(variation.view(28, 28).cpu().numpy())
    
    return variations

# Example usage (assuming a trained VAE and a style image)
# variations = generate_artwork(vae, 'style_image.jpg')
# for i, var in enumerate(variations):
#     Image.fromarray((var * 255).astype('uint8')).save(f'variation_{i}.png')
```

Slide 14: Real-life Example: Molecular Design

VAEs can be used in drug discovery to generate new molecular structures with desired properties.

```python
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

class MolecularVAE(VAE):
    def __init__(self, input_dim, latent_dim):
        super(MolecularVAE, self).__init__(input_dim, latent_dim)
    
    def mol_to_fingerprint(self, mol):
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.input_dim)
    
    def fingerprint_to_mol(self, fp):
        # This is a simplified placeholder. In practice, this would involve
        # a more complex decoding process to generate valid molecules.
        return None  # Placeholder

def generate_molecules(vae, num_samples=10):
    z = torch.randn(num_samples, vae.latent_dim)
    fingerprints = vae.decoder(z).detach().numpy()
    molecules = [vae.fingerprint_to_mol(fp) for fp in fingerprints]
    return molecules

# Example usage (assuming a trained MolecularVAE)
# mol_vae = MolecularVAE(input_dim=2048, latent_dim=128)
# new_molecules = generate_molecules(mol_vae)
```

Slide 15: Additional Resources

For more in-depth information on Variational Autoencoders, consider exploring these resources:

1. "Auto-Encoding Variational Bayes" by Kingma and Welling (2013). This is the original paper introducing VAEs. Available at: [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
2. "An Introduction to Variational Autoencoders" by Doersch (2016). A comprehensive tutorial on VAEs. Available at: [https://arxiv.org/abs/1606.05908](https://arxiv.org/abs/1606.05908)
3. "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" by Higgins et al. (2017). Introduces the concept of β-VAE for disentangled representation learning. Available at: [https://openreview.net/forum?id=Sy2fzU9gl](https://openreview.net/forum?id=Sy2fzU9gl)
4. "Understanding disentangling in β-VAE" by Burgess et al. (2018). Provides insights into the workings of β-VAE. Available at: [https://arxiv.org/abs/1804.03599](https://arxiv.org/abs/1804.03599)

These papers provide a solid foundation for understanding the theory and applications of Variational Autoencoders.

