## How Autoencoders Work for Dimension Reduction
Slide 1: Autoencoder Architecture Fundamentals

An autoencoder is a neural network architecture that learns to compress data into a lower-dimensional representation and then reconstruct it. The network consists of an encoder that maps input to a latent space and a decoder that attempts to reconstruct the original input from this compressed representation.

```python
import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SimpleAutoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
```

Slide 2: Loss Functions for Autoencoders

The choice of loss function is crucial for training autoencoders. For continuous data, Mean Squared Error (MSE) is commonly used to measure reconstruction quality. For binary data, Binary Cross Entropy (BCE) loss is preferred. The loss function measures how well the decoder reconstructs the input.

```python
def autoencoder_loss(model, criterion, optimizer, input_data):
    # MSE Loss for continuous data
    mse_criterion = nn.MSELoss()
    
    # BCE Loss for binary data
    bce_criterion = nn.BCELoss()
    
    # Forward pass
    output = model(input_data)
    
    # Calculate loss
    mse_loss = mse_criterion(output, input_data)
    bce_loss = bce_criterion(output, input_data)
    
    # Example of optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return mse_loss.item(), bce_loss.item()
```

Slide 3: Training Loop Implementation

A comprehensive training loop for autoencoders requires careful monitoring of both training and validation losses. The implementation includes data loading, model training, and progress tracking through multiple epochs while avoiding overfitting through validation.

```python
def train_autoencoder(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                output = model(data)
                val_loss += criterion(output, data).item()
                
        train_losses.append(epoch_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))
        
        print(f'Epoch {epoch+1}: Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')
```

Slide 4: Implementing the Bottleneck Layer

The bottleneck layer represents the compressed representation of input data. This implementation demonstrates how to create a bottleneck layer with proper dimensionality reduction and optional regularization for better compression.

```python
class BottleneckAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(BottleneckAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim//4),
            nn.ReLU(),
            nn.Linear(input_dim//4, bottleneck_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim//4),
            nn.ReLU(),
            nn.Linear(input_dim//4, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

Slide 5: Data Preprocessing for Autoencoders

Effective data preprocessing is crucial for autoencoder performance. This implementation shows standard preprocessing techniques including normalization, scaling, and handling of missing values specific to autoencoder applications.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_data(data, scaling_method='standard'):
    # Handle missing values
    data = np.nan_to_num(data, nan=0)
    
    # Choose scaling method
    if scaling_method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    # Scale the data
    scaled_data = scaler.fit_transform(data)
    
    # Convert to torch tensor
    tensor_data = torch.FloatTensor(scaled_data)
    
    return tensor_data, scaler
```

Slide 6: Regularization Techniques for Autoencoders

Regularization prevents overfitting and improves the quality of learned representations. Common techniques include L1/L2 regularization on weights and activity regularization on the latent space. This implementation demonstrates how to add various regularization methods to the autoencoder.

```python
class RegularizedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout_rate=0.2):
        super(RegularizedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        # L1 regularization on latent representation
        l1_reg = torch.sum(torch.abs(latent))
        # Store for loss calculation
        self.l1_reg = l1_reg
        return self.decoder(latent)

def compute_loss(model, x, reconstruction, lambda_l1=0.01):
    # MSE reconstruction loss
    mse_loss = F.mse_loss(reconstruction, x)
    # Add L1 regularization term
    total_loss = mse_loss + lambda_l1 * model.l1_reg
    return total_loss
```

Slide 7: Variational Autoencoder Implementation

Variational Autoencoders (VAE) extend traditional autoencoders by learning a probability distribution in the latent space. This implementation shows how to create a VAE with a reparameterization trick and proper loss function including KL divergence.

```python
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU()
        )
        
        # Mean and variance layers
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

def vae_loss(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD
```

Slide 8: Real-world Application: Anomaly Detection

Autoencoders excel at anomaly detection by learning the normal data distribution. This implementation shows how to use reconstruction error as an anomaly score and includes threshold determination for anomaly classification.

```python
class AnomalyDetector:
    def __init__(self, model, threshold_percentile=95):
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.threshold = None
    
    def fit(self, normal_data):
        self.model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch in normal_data:
                reconstructed = self.model(batch)
                error = torch.mean((batch - reconstructed) ** 2, dim=1)
                reconstruction_errors.extend(error.numpy())
        
        # Set threshold based on percentile of reconstruction errors
        self.threshold = np.percentile(reconstruction_errors, 
                                     self.threshold_percentile)
    
    def predict(self, data):
        self.model.eval()
        anomalies = []
        
        with torch.no_grad():
            reconstructed = self.model(data)
            errors = torch.mean((data - reconstructed) ** 2, dim=1)
            anomalies = (errors > self.threshold).numpy()
            
        return anomalies, errors.numpy()
```

Slide 9: Denoising Autoencoder Implementation

Denoising autoencoders learn robust features by reconstructing clean data from corrupted inputs. This implementation demonstrates how to add noise to input data and train the model to recover the original clean signal.

```python
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DenoisingAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, input_dim),
            nn.Sigmoid()
        )
    
    @staticmethod
    def add_noise(x, noise_factor=0.3):
        noise = torch.randn_like(x) * noise_factor
        corrupted_x = x + noise
        return torch.clamp(corrupted_x, 0., 1.)
    
    def forward(self, x, noise_factor=0.3):
        # Add noise to input
        corrupted = self.add_noise(x, noise_factor)
        # Encode and decode
        encoded = self.encoder(corrupted)
        decoded = self.decoder(encoded)
        return decoded, corrupted
```

Slide 10: Real-world Application: Image Compression

Autoencoders can be used for efficient image compression by learning compact representations. This implementation shows how to create and train an autoencoder specifically for image compression tasks.

```python
class ImageCompressor(nn.Module):
    def __init__(self, channels=3):
        super(ImageCompressor, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
    
    def get_compression_ratio(self, image_size):
        original_size = image_size[0] * image_size[1] * 3  # RGB channels
        compressed_size = (image_size[0]//4) * (image_size[1]//4) * 8  # Bottleneck
        return original_size / compressed_size
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
```

Slide 11: Sparse Autoencoder Architecture

Sparse autoencoders impose sparsity constraints on the hidden layer activations, forcing the network to learn more efficient representations. This implementation includes KL divergence sparsity penalty.

```python
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_target=0.05):
        super(SparseAutoencoder, self).__init__()
        self.sparsity_target = sparsity_target
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Calculate average activation of hidden units
        rho_hat = torch.mean(encoded, dim=0)
        return decoded, rho_hat

def sparse_loss(model, decoded, x, rho_hat, beta=3.0):
    # Reconstruction loss
    mse_loss = F.mse_loss(decoded, x)
    # KL divergence for sparsity penalty
    rho = model.sparsity_target
    kl_div = rho * torch.log(rho/rho_hat) + (1-rho) * torch.log((1-rho)/(1-rho_hat))
    kl_div = torch.sum(kl_div)
    
    return mse_loss + beta * kl_div
```

Slide 12: Convolutional Autoencoder

Convolutional autoencoders are specifically designed for processing image data by using convolutional layers instead of fully connected layers. This architecture preserves spatial relationships in the data and is more efficient for image processing tasks.

```python
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

Slide 13: Dimensionality Reduction Application

Autoencoders can be used as a powerful non-linear dimensionality reduction technique. This implementation shows how to create a dimensionality reduction pipeline using autoencoders and compare it with PCA.

```python
from sklearn.decomposition import PCA
import numpy as np

class DimensionalityReducer:
    def __init__(self, input_dim, latent_dim):
        self.autoencoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        
        self.encoder = self.autoencoder[:3]
        self.pca = PCA(n_components=latent_dim)
        
    def fit_transform(self, X, epochs=100, batch_size=32):
        # Train autoencoder
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        
        optimizer = torch.optim.Adam(self.autoencoder.parameters())
        
        for epoch in range(epochs):
            for batch in loader:
                optimizer.zero_grad()
                output = self.autoencoder(batch[0])
                loss = F.mse_loss(output, batch[0])
                loss.backward()
                optimizer.step()
        
        # Get reduced representations
        with torch.no_grad():
            ae_reduced = self.encoder(torch.FloatTensor(X)).numpy()
        
        # Compare with PCA
        pca_reduced = self.pca.fit_transform(X)
        
        return ae_reduced, pca_reduced
```

Slide 14: Real-world Performance Metrics

Performance evaluation is crucial for autoencoder applications. This implementation provides comprehensive metrics including reconstruction error, compression ratio, and specific metrics for different use cases.

```python
class AutoencoderEvaluator:
    def __init__(self, model):
        self.model = model
        
    def compute_metrics(self, original_data, reconstructed_data):
        metrics = {}
        
        # Mean Squared Error
        metrics['mse'] = F.mse_loss(reconstructed_data, original_data).item()
        
        # Peak Signal to Noise Ratio (PSNR)
        mse = metrics['mse']
        metrics['psnr'] = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        
        # Structural Similarity Index (SSIM)
        if len(original_data.shape) == 4:  # For images
            metrics['ssim'] = self.compute_ssim(original_data, reconstructed_data)
        
        # Compression ratio
        original_size = np.prod(original_data.shape)
        compressed_size = np.prod(self.model.encoder(original_data).shape)
        metrics['compression_ratio'] = original_size / compressed_size
        
        return metrics
        
    @staticmethod
    def compute_ssim(img1, img2):
        # Simplified SSIM implementation
        c1 = (0.01 * 255)**2
        c2 = (0.03 * 255)**2
        
        mean1 = torch.mean(img1, dim=[2,3], keepdim=True)
        mean2 = torch.mean(img2, dim=[2,3], keepdim=True)
        std1 = torch.std(img1, dim=[2,3], keepdim=True)
        std2 = torch.std(img2, dim=[2,3], keepdim=True)
        
        covariance = torch.mean((img1 - mean1) * (img2 - mean2), dim=[2,3])
        
        ssim = ((2 * mean1 * mean2 + c1) * (2 * covariance + c2)) / \
               ((mean1**2 + mean2**2 + c1) * (std1**2 + std2**2 + c2))
        
        return torch.mean(ssim).item()
```

Slide 15: Additional Resources

*   "Auto-Encoding Variational Bayes" - [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
*   "Reducing the Dimensionality of Data with Neural Networks" - [https://www.science.org/doi/10.1126/science.1127647](https://www.science.org/doi/10.1126/science.1127647)
*   "Deep Learning for Image and Text Processing" - [https://arxiv.org/abs/1911.03723](https://arxiv.org/abs/1911.03723)
*   "Denoising Autoencoders" - [https://arxiv.org/abs/2003.05991](https://arxiv.org/abs/2003.05991)
*   Search Keywords: "autoencoder architecture", "deep learning autoencoders", "neural network dimensionality reduction"

