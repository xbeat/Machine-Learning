## Variational Autoencoders Generative Models with Deep Learning
Slide 1: Understanding VAE Architecture

A Variational Autoencoder combines probabilistic inference with neural networks, creating a powerful generative model. The architecture consists of an encoder network that maps input data to a latent distribution and a decoder network that reconstructs the input from sampled latent vectors.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
```

Slide 2: The Reparametrization Trick

The reparametrization trick enables backpropagation through random sampling by expressing the random variable as a deterministic transformation of a noise variable and the distribution parameters, making the network end-to-end differentiable.

```python
def reparameterize(self, mu, log_var):
    # Training: Random sampling using reparametrization trick
    if self.training:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    # Evaluation: Return mean
    return mu
```

Slide 3: VAE Loss Function Components

The VAE loss function combines reconstruction loss (measuring how well the model reconstructs input data) with the Kullback-Leibler divergence, which ensures the learned latent distribution approximates a standard normal distribution.

```python
def loss_function(recon_x, x, mu, log_var):
    # Reconstruction loss (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total loss is the sum of both terms
    return BCE + KLD
```

Slide 4: Training Loop Implementation

The training process involves forward propagation through the encoder and decoder networks, computing the loss function, and updating model parameters through backpropagation while maintaining the balance between reconstruction and regularization terms.

```python
def train_epoch(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        mu, log_var = model.encode(data)
        z = model.reparameterize(mu, log_var)
        recon_batch = model.decode(z)
        
        # Compute loss and backpropagate
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    return train_loss / len(train_loader.dataset)
```

Slide 5: Real-world Example: Image Generation

A practical implementation of VAE for generating MNIST handwritten digits demonstrates the model's capability to learn meaningful latent representations and generate new, realistic samples. This example includes data preprocessing and model training.

```python
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
dataset = MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Initialize model and optimizer
model = VAE(input_dim=784, hidden_dim=400, latent_dim=20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Slide 6: Training Results for Image Generation

The implementation demonstrates the VAE's training progression using MNIST data, showing how the model learns to reconstruct digits and generate new ones. The loss metrics indicate both reconstruction quality and latent space regularization performance.

```python
def train_and_evaluate(model, train_loader, num_epochs, device):
    history = {'loss': [], 'recon_loss': [], 'kl_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, log_var = model(data)
            
            # Calculate losses
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kl_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record losses
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
        
        # Store average losses
        avg_loss = epoch_loss / len(train_loader.dataset)
        avg_recon = epoch_recon / len(train_loader.dataset)
        avg_kl = epoch_kl / len(train_loader.dataset)
        
        history['loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)
        
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, Reconstruction = {avg_recon:.4f}, KL = {avg_kl:.4f}')
    
    return history
```

Slide 7: Latent Space Manipulation

Understanding and manipulating the latent space is crucial for generating new samples and performing interpolation between existing data points. This implementation shows how to traverse the latent space to create smooth transitions between different digits.

```python
def interpolate_latent_space(model, start_img, end_img, steps=10):
    model.eval()
    with torch.no_grad():
        # Encode images to get latent representations
        mu_start, _ = model.encode(start_img.unsqueeze(0))
        mu_end, _ = model.encode(end_img.unsqueeze(0))
        
        # Create interpolation points
        alphas = torch.linspace(0, 1, steps)
        latent_interp = torch.zeros(steps, model.latent_dim)
        
        # Generate interpolated images
        for idx, alpha in enumerate(alphas):
            latent_interp[idx] = alpha * mu_end + (1 - alpha) * mu_start
            
        # Decode interpolated points
        interpolated_images = model.decode(latent_interp)
        
    return interpolated_images

# Example usage
start_idx, end_idx = 0, 1  # Interpolate between first two digits
interpolated = interpolate_latent_space(model, 
                                      dataset[start_idx][0], 
                                      dataset[end_idx][0])
```

Slide 8: Conditional VAE Implementation

A Conditional VAE extends the basic VAE architecture by incorporating label information, allowing controlled generation of samples based on specific attributes or classes. This implementation shows how to condition the generation process.

```python
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder with condition
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder with condition
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
```

Slide 9: Real-world Example: Text Generation with VAE

This implementation demonstrates how to use VAEs for text generation, incorporating word embeddings and LSTM layers to handle sequential data. The model learns to generate coherent text while maintaining semantic meaning.

```python
class TextVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim):
        super(TextVAE, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, 
                                  batch_first=True, bidirectional=True)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(latent_dim + embed_dim, hidden_dim, 
                                  batch_first=True)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
```

Slide 10: Text VAE Training Process

The training process for text-based VAE requires special attention to sequence handling and teacher forcing during decoding. This implementation shows how to properly train the model while maintaining coherent text generation capabilities.

```python
def train_text_vae(model, train_loader, optimizer, device, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, lengths) in enumerate(train_loader):
        data = data.to(device)
        batch_size = data.size(0)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Encode
        embedded = model.embedding(data)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        
        # Get latent representation
        encoder_out, _ = model.encoder_lstm(packed)
        encoder_out, _ = nn.utils.rnn.pad_packed_sequence(encoder_out, batch_first=True)
        
        # Get mean and variance
        hidden = encoder_out[:, -1, :]  # Take last hidden state
        mu = model.fc_mu(hidden)
        log_var = model.fc_var(hidden)
        
        # Sample latent vector
        z = model.reparameterize(mu, log_var)
        
        # Decode
        outputs = torch.zeros(batch_size, max(lengths), model.vocab_size).to(device)
        decoder_input = data[:, 0]  # Start token
        
        for t in range(1, max(lengths)):
            # Teacher forcing
            if random.random() < teacher_forcing_ratio:
                decoder_input = data[:, t-1]
            
            decoder_embedded = model.embedding(decoder_input)
            decoder_input = torch.cat([decoder_embedded, z.unsqueeze(1)], dim=2)
            
            decoder_output, hidden = model.decoder_lstm(decoder_input, hidden)
            prediction = model.output_layer(decoder_output.squeeze(1))
            outputs[:, t, :] = prediction
            
            decoder_input = prediction.argmax(1)
        
        # Calculate loss
        recon_loss = F.cross_entropy(
            outputs.view(-1, model.vocab_size),
            data.view(-1),
            ignore_index=PAD_IDX
        )
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)
```

Slide 11: VAE for Anomaly Detection

Implementing VAE for anomaly detection leverages the model's ability to learn the normal data distribution. Anomalies are detected by measuring reconstruction error and KL divergence from the learned distribution.

```python
class AnomalyVAE(VAE):
    def anomaly_score(self, x):
        self.eval()
        with torch.no_grad():
            # Encode and decode
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            x_recon = self.decode(z)
            
            # Calculate reconstruction error
            recon_error = F.mse_loss(x_recon, x, reduction='none').sum(dim=1)
            
            # Calculate KL divergence
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            
            # Combined anomaly score
            anomaly_score = recon_error + kl_div
            
        return anomaly_score

def detect_anomalies(model, data_loader, threshold):
    anomalies = []
    scores = []
    
    for batch_idx, (data, _) in enumerate(data_loader):
        # Calculate anomaly scores
        batch_scores = model.anomaly_score(data)
        
        # Detect anomalies based on threshold
        batch_anomalies = batch_scores > threshold
        
        anomalies.extend(batch_anomalies.cpu().numpy())
        scores.extend(batch_scores.cpu().numpy())
    
    return np.array(anomalies), np.array(scores)
```

Slide 12: Real-world Example: High-dimensional Data Visualization

VAEs can effectively reduce high-dimensional data to a lower-dimensional latent space while preserving important features. This implementation demonstrates dimensionality reduction and visualization of complex datasets.

```python
class VisualizationVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super(VisualizationVAE, self).__init__()
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Latent space parameters (2D for visualization)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    
    def visualize_latent_space(self, data_loader):
        self.eval()
        latent_coords = []
        labels = []
        
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(data_loader):
                mu, _ = self.encode(data)
                latent_coords.append(mu.cpu().numpy())
                labels.extend(label.numpy())
        
        latent_coords = np.concatenate(latent_coords, axis=0)
        return latent_coords, np.array(labels)
```

Slide 13: Hierarchical VAE Implementation

Hierarchical VAEs extend the basic architecture by introducing multiple levels of latent variables, allowing the model to capture complex data distributions at different scales of abstraction. This implementation shows a two-level hierarchical structure.

```python
class HierarchicalVAE(nn.Module):
    def __init__(self, input_dim, latent_dim1=32, latent_dim2=16):
        super(HierarchicalVAE, self).__init__()
        
        # First level encoder
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.fc_mu1 = nn.Linear(256, latent_dim1)
        self.fc_var1 = nn.Linear(256, latent_dim1)
        
        # Second level encoder
        self.encoder2 = nn.Sequential(
            nn.Linear(latent_dim1, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.fc_mu2 = nn.Linear(64, latent_dim2)
        self.fc_var2 = nn.Linear(64, latent_dim2)
        
        # Decoders
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim2, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim1)
        )
        
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim1, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # First level encoding
        h1 = self.encoder1(x)
        mu1 = self.fc_mu1(h1)
        log_var1 = self.fc_var1(h1)
        z1 = self.reparameterize(mu1, log_var1)
        
        # Second level encoding
        h2 = self.encoder2(z1)
        mu2 = self.fc_mu2(h2)
        log_var2 = self.fc_var2(h2)
        z2 = self.reparameterize(mu2, log_var2)
        
        # Decoding
        z1_hat = self.decoder2(z2)
        reconstruction = self.decoder1(z1_hat)
        
        return reconstruction, mu1, log_var1, mu2, log_var2
```

Slide 14: Training Results and Performance Metrics

Comprehensive evaluation of VAE performance includes multiple metrics such as reconstruction quality, KL divergence, and latent space organization. This implementation demonstrates how to track and visualize these metrics during training.

```python
def evaluate_vae_performance(model, test_loader, device):
    model.eval()
    metrics = {
        'reconstruction_loss': 0,
        'kl_divergence': 0,
        'total_loss': 0,
        'latent_variance': 0
    }
    
    latent_representations = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            # Forward pass
            recon_batch, mu, log_var = model(data)
            
            # Calculate losses
            recon_loss = F.binary_cross_entropy(
                recon_batch, data, reduction='sum'
            )
            kl_loss = -0.5 * torch.sum(
                1 + log_var - mu.pow(2) - log_var.exp()
            )
            
            # Update metrics
            metrics['reconstruction_loss'] += recon_loss.item()
            metrics['kl_divergence'] += kl_loss.item()
            metrics['total_loss'] += (recon_loss + kl_loss).item()
            
            # Store latent representations
            latent_representations.append(mu.cpu().numpy())
            
            # Calculate latent space variance
            metrics['latent_variance'] += mu.var(dim=0).mean().item()
    
    # Normalize metrics
    n_samples = len(test_loader.dataset)
    for key in metrics:
        metrics[key] /= n_samples
    
    # Calculate additional metrics
    latent_representations = np.concatenate(latent_representations, axis=0)
    metrics['latent_correlation'] = np.corrcoef(
        latent_representations.T
    ).mean()
    
    return metrics
```

Slide 15: Additional Resources

*   "Auto-Encoding Variational Bayes"
*   [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
*   "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
*   [https://openreview.net/forum?id=Sy2fzU9gl](https://openreview.net/forum?id=Sy2fzU9gl)
*   "Hierarchical Variational Auto-Encoders for Music"
*   [https://arxiv.org/abs/1811.07050](https://arxiv.org/abs/1811.07050)
*   "Understanding disentangling in β-VAE"
*   [https://arxiv.org/abs/1804.03599](https://arxiv.org/abs/1804.03599)
*   "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
*   [https://arxiv.org/abs/1503.03585](https://arxiv.org/abs/1503.03585)

