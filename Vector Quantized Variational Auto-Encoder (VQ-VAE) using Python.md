## Vector Quantized Variational Auto-Encoder (VQ-VAE) using Python
Slide 1: Introduction to VQ-VAE

Vector Quantized Variational Auto-Encoder (VQ-VAE) is a powerful neural network architecture for learning discrete representations of data. It combines the ideas of vector quantization and variational autoencoders to create a model that can compress and reconstruct complex data while maintaining important features. This approach is particularly useful in various applications, including image and audio processing, as well as generative modeling.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VQVAE, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, embedding_dim, kernel_size=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=1)
        )
```

Slide 2: Vector Quantization

Vector quantization is a technique used to map a large set of points (vectors) to a smaller set of points. In VQ-VAE, this process involves mapping the continuous latent representations to a discrete codebook of embeddings. This quantization step allows the model to learn a compressed representation of the input data.

```python
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding.embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.embedding.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        return quantized, encodings, encoding_indices
```

Slide 3: Encoder Architecture

The encoder in VQ-VAE is responsible for transforming the input data into a latent representation. It typically consists of a series of convolutional layers that progressively reduce the spatial dimensions of the input while increasing the number of channels. The final layer of the encoder produces a tensor with the same number of channels as the embedding dimension used in the vector quantizer.

```python
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        return self._residual_stack(x)
```

Slide 4: Decoder Architecture

The decoder in VQ-VAE is responsible for reconstructing the input data from the quantized latent representation. It mirrors the encoder architecture, using transposed convolutional layers to progressively increase the spatial dimensions while decreasing the number of channels. The final layer of the decoder produces an output with the same shape as the original input.

```python
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        return self._conv_trans_2(x)
```

Slide 5: Loss Function

The VQ-VAE loss function consists of three main components: reconstruction loss, codebook loss, and commitment loss. The reconstruction loss measures how well the decoder can reconstruct the input from the quantized representation. The codebook loss encourages the encoder to output latent vectors close to the codebook embeddings. The commitment loss prevents the encoder outputs from growing arbitrarily large.

```python
def vq_vae_loss(x, x_recon, quantized, encoding_indices, beta):
    recon_loss = F.mse_loss(x_recon, x)
    
    e_latent_loss = F.mse_loss(quantized.detach(), encoding_indices)
    q_latent_loss = F.mse_loss(quantized, encoding_indices.detach())
    
    loss = recon_loss + q_latent_loss + beta * e_latent_loss
    
    return loss
```

Slide 6: Training Loop

The training loop for VQ-VAE involves forward-passing the input through the encoder, vector quantizer, and decoder, then computing the loss and updating the model parameters. It's important to use the straight-through estimator to allow gradients to flow through the discrete bottleneck.

```python
def train_vqvae(model, optimizer, data_loader, num_epochs):
    for epoch in range(num_epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            
            x = batch.to(device)
            z_e = model.encoder(x)
            z_q, _, _ = model.vector_quantizer(z_e)
            x_recon = model.decoder(z_q)
            
            loss = vq_vae_loss(x, x_recon, z_q, z_e, beta=0.25)
            loss.backward()
            
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Usage
model = VQVAE(num_embeddings=512, embedding_dim=64).to(device)
optimizer = optim.Adam(model.parameters())
train_vqvae(model, optimizer, train_loader, num_epochs=50)
```

Slide 7: Inference and Generation

After training, VQ-VAE can be used for various tasks, including compression, reconstruction, and generation. For generation, a separate prior model (e.g., PixelCNN) is often trained on the discrete latent space to capture the distribution of the latent codes.

```python
def generate_samples(model, prior_model, num_samples):
    with torch.no_grad():
        # Generate latent codes using the prior model
        latent_codes = prior_model.sample(num_samples)
        
        # Convert latent codes to embeddings
        quantized = model.embedding(latent_codes)
        
        # Decode the embeddings
        generated_samples = model.decoder(quantized)
    
    return generated_samples

# Usage
prior_model = PixelCNN(num_embeddings=512, embedding_dim=64)
samples = generate_samples(vqvae_model, prior_model, num_samples=16)
```

Slide 8: Real-life Example: Image Compression

VQ-VAE can be used for lossy image compression. By training the model on a large dataset of images, it learns to compress images into a discrete latent space while preserving important features. This allows for efficient storage and transmission of images.

```python
def compress_image(model, image):
    with torch.no_grad():
        x = transform(image).unsqueeze(0).to(device)
        z_e = model.encoder(x)
        _, _, encoding_indices = model.vector_quantizer(z_e)
    return encoding_indices.squeeze().cpu().numpy()

def decompress_image(model, encoding_indices):
    with torch.no_grad():
        encoding_indices = torch.from_numpy(encoding_indices).unsqueeze(0).to(device)
        quantized = model.embedding(encoding_indices)
        x_recon = model.decoder(quantized)
    return transforms.ToPILImage()(x_recon.squeeze().cpu())

# Usage
original_image = Image.open("example.jpg")
compressed = compress_image(vqvae_model, original_image)
reconstructed_image = decompress_image(vqvae_model, compressed)
```

Slide 9: Real-life Example: Audio Generation

VQ-VAE can also be applied to audio processing tasks, such as speech synthesis or music generation. By training on a dataset of audio samples, the model learns to encode and decode audio signals efficiently.

```python
import librosa
import soundfile as sf

def process_audio(model, audio_file, sr=22050):
    # Load audio file
    audio, _ = librosa.load(audio_file, sr=sr)
    
    # Convert to spectrogram
    spectrogram = librosa.stft(audio)
    mag, _ = librosa.magphase(spectrogram)
    
    # Encode and quantize
    z_e = model.encoder(torch.from_numpy(mag).unsqueeze(0).to(device))
    z_q, _, _ = model.vector_quantizer(z_e)
    
    # Decode
    recon_mag = model.decoder(z_q).squeeze().cpu().numpy()
    
    # Reconstruct audio
    recon_audio = librosa.griffinlim(recon_mag)
    
    return recon_audio

# Usage
original_audio = "speech.wav"
reconstructed_audio = process_audio(vqvae_model, original_audio)
sf.write("reconstructed_speech.wav", reconstructed_audio, 22050)
```

Slide 10: Handling Discrete Latent Space

One of the key challenges in VQ-VAE is dealing with the discrete latent space. The straight-through estimator is used to allow gradients to flow through the non-differentiable quantization operation during training.

```python
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99):
        super(VectorQuantizerEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embedded_avg', embed.clone())

    def forward(self, inputs):
        flatten = inputs.reshape(-1, self.embedding_dim)
        dist = torch.sum(flatten**2, dim=1, keepdim=True) + \
               torch.sum(self.embed**2, dim=1) - \
               2 * torch.matmul(flatten, self.embed.t())
        _, encoding_indices = (-dist).max(dim=1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        quantized = torch.matmul(encodings, self.embed).view(inputs.shape)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, encodings.sum(0))
            embedded_sum = torch.matmul(encodings.t(), flatten)
            self.embedded_avg.data.mul_(self.decay).add_(
                1 - self.decay, embedded_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + 1e-5)
                / (n + self.num_embeddings * 1e-5) * n)
            embed_normalized = self.embedded_avg / cluster_size.unsqueeze(1)
            self.embed.data._(embed_normalized)

        loss = self.commitment_cost * F.mse_loss(quantized.detach(), inputs)

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity
```

Slide 11: Residual Connections

Residual connections are often used in both the encoder and decoder of VQ-VAE to improve gradient flow and allow the model to learn deeper representations. These connections help mitigate the vanishing gradient problem and enable the training of deeper networks.

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels, num_residual_hiddens, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_residual_hiddens, num_hiddens, 1, 1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([ResidualBlock(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
```

Slide 12: Handling Multi-scale Features

VQ-VAE can be extended to handle multi-scale features, allowing the model to capture both fine-grained and coarse-grained information. This is particularly useful for high-resolution images or complex audio signals.

```python
class MultiScaleVQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_scales=3):
        super(MultiScaleVQVAE, self).__init__()
        self.num_scales = num_scales
        self.encoders = nn.ModuleList([Encoder() for _ in range(num_scales)])
        self.decoders = nn.ModuleList([Decoder() for _ in range(num_scales)])
        self.vector_quantizers = nn.ModuleList([VectorQuantizer(num_embeddings, embedding_dim) for _ in range(num_scales)])

    def forward(self, x):
        encoded = []
        quantized = []
        for i in range(self.num_scales):
            e = self.encoders[i](x if i == 0 else encoded[-1])
            q, _, _ = self.vector_quantizers[i](e)
            encoded.append(e)
            quantized.append(q)

        decoded = []
        for i in range(self.num_scales - 1, -1, -1):
            d = self.decoders[i](quantized[i])
            if i > 0:
                d += F.interpolate(decoded[-1], scale_factor=2, mode='nearest')
            decoded.append(d)

        return decoded[-1], encoded, quantized
```

Slide 13: Conditional VQ-VAE

Conditional VQ-VAE extends the basic model by incorporating conditioning information, allowing for more controlled generation or reconstruction. This can be useful in tasks such as text-to-speech synthesis or style transfer.

```python
class ConditionalVQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, condition_dim):
        super(ConditionalVQVAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.condition_encoder = nn.Linear(condition_dim, embedding_dim)

    def forward(self, x, condition):
        encoded = self.encoder(x)
        quantized, _, _ = self.vector_quantizer(encoded)
        condition_embedding = self.condition_encoder(condition)
        combined = quantized + condition_embedding.unsqueeze(2).unsqueeze(3)
        decoded = self.decoder(combined)
        return decoded, encoded, quantized

# Usage
condition = torch.randn(batch_size, condition_dim)
x_recon, encoded, quantized = conditional_vqvae(x, condition)
```

Slide 14: VQ-VAE for Sequence Data

While VQ-VAE is often used for image and audio data, it can also be adapted for sequence data, such as text or time series. This requires modifications to the encoder and decoder architectures to handle sequential inputs and outputs.

```python
class SequenceVQVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_embeddings):
        super(SequenceVQVAE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.vector_quantizer = VectorQuantizer(num_embeddings, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        encoded, _ = self.encoder(embedded)
        quantized, _, _ = self.vector_quantizer(encoded)
        decoded, _ = self.decoder(quantized)
        output = self.output_layer(decoded)
        return output, encoded, quantized

# Usage
sequence = torch.randint(0, vocab_size, (batch_size, seq_length))
output, encoded, quantized = sequence_vqvae(sequence)
```

Slide 15: Additional Resources

For more information on VQ-VAE and related topics, consider exploring the following resources:

1. Original VQ-VAE paper: "Neural Discrete Representation Learning" by van den Oord et al. (2017) ArXiv link: [https://arxiv.org/abs/1711.00937](https://arxiv.org/abs/1711.00937)
2. VQ-VAE-2 paper: "Generating Diverse High-Fidelity Images with VQ-VAE-2" by Razavi et al. (2019) ArXiv link: [https://arxiv.org/abs/1906.00446](https://arxiv.org/abs/1906.00446)
3. "Understanding Vector Quantized Variational Autoencoders" by Gao et al. (2021) ArXiv link: [https://arxiv.org/abs/2106.08795](https://arxiv.org/abs/2106.08795)

These papers provide in-depth explanations of the VQ-VAE architecture, its variants, and applications in various domains. They offer valuable insights into the theoretical foundations and practical implementations of VQ-VAE models.

