## Diffusion Models for Imaging and Vision in Python
Slide 1: Introduction to Diffusion Models

Diffusion models are a class of generative models that have gained significant attention in the field of computer vision and image processing. These models work by gradually adding noise to data and then learning to reverse this process, allowing them to generate high-quality images from noise. This approach has shown remarkable results in various tasks, including image generation, inpainting, and super-resolution.

```python
import torch
import torch.nn as nn

class SimpleDiffusionModel(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, input_channels, 3, padding=1)

    def forward(self, x, t):
        # t is the timestep, which can be used to condition the model
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x

model = SimpleDiffusionModel()
print(model)
```

Slide 2: The Diffusion Process

The diffusion process involves gradually adding Gaussian noise to an image over a series of steps. This process transforms a complex data distribution into a simple Gaussian distribution. The reverse process, called denoising, learns to reconstruct the original image from the noisy version.

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

def add_noise(image, noise_level):
    return image + noise_level * torch.randn_like(image)

# Load and preprocess an image
image = Image.open("sample_image.jpg")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
image_tensor = transform(image).unsqueeze(0)

# Add noise to the image
noisy_image = add_noise(image_tensor, noise_level=0.1)

print(f"Original image shape: {image_tensor.shape}")
print(f"Noisy image shape: {noisy_image.shape}")
```

Slide 3: Training a Diffusion Model

Training a diffusion model involves learning to predict the noise added to an image at each step of the diffusion process. The model is trained to minimize the difference between the predicted noise and the actual noise added to the image.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DiffusionTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, x_0, t):
        self.optimizer.zero_grad()
        
        # Add noise to the input
        noise = torch.randn_like(x_0)
        x_t = x_0 + t * noise
        
        # Predict the noise
        predicted_noise = self.model(x_t, t)
        
        # Calculate loss
        loss = self.loss_fn(predicted_noise, noise)
        
        # Backpropagate and update weights
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# Usage
model = SimpleDiffusionModel()
trainer = DiffusionTrainer(model)
loss = trainer.train_step(image_tensor, torch.tensor([0.1]))
print(f"Training loss: {loss}")
```

Slide 4: Reverse Diffusion Process

The reverse diffusion process is used to generate new images. Starting from pure noise, the model iteratively denoises the image, gradually revealing a coherent structure. This process is typically done over multiple steps, with each step slightly reducing the noise level.

```python
import torch

def reverse_diffusion(model, steps=100, img_shape=(3, 256, 256)):
    x = torch.randn(img_shape)
    
    for t in reversed(range(steps)):
        t_tensor = torch.tensor([t / steps]).float()
        with torch.no_grad():
            predicted_noise = model(x.unsqueeze(0), t_tensor)
        
        # Update x using the predicted noise
        x = (x - (1 - t / steps) * predicted_noise.squeeze()) / (1 - t / steps)**0.5
    
    return x

# Generate an image
model = SimpleDiffusionModel()
generated_image = reverse_diffusion(model)
print(f"Generated image shape: {generated_image.shape}")
```

Slide 5: Image Inpainting with Diffusion Models

Diffusion models can be used for image inpainting, which involves filling in missing or corrupted parts of an image. By conditioning the diffusion process on the known parts of the image, the model can generate plausible completions for the missing regions.

```python
import torch
import torch.nn.functional as F

def inpaint(model, image, mask, steps=100):
    x = torch.randn_like(image)
    
    for t in reversed(range(steps)):
        t_tensor = torch.tensor([t / steps]).float()
        with torch.no_grad():
            predicted_noise = model(x.unsqueeze(0), t_tensor)
        
        # Update x using the predicted noise
        x = (x - (1 - t / steps) * predicted_noise.squeeze()) / (1 - t / steps)**0.5
        
        # Keep the known parts of the image unchanged
        x = x * mask + image * (1 - mask)
    
    return x

# Create a sample mask (1 for unknown regions, 0 for known regions)
mask = torch.zeros_like(image_tensor)
mask[:, :, 100:150, 100:150] = 1  # Create a 50x50 hole in the image

inpainted_image = inpaint(model, image_tensor, mask)
print(f"Inpainted image shape: {inpainted_image.shape}")
```

Slide 6: Super-Resolution with Diffusion Models

Diffusion models can be applied to super-resolution tasks, where the goal is to increase the resolution of a low-resolution image. The model learns to add realistic high-frequency details to the upsampled low-resolution input.

```python
import torch
import torch.nn.functional as F

def super_resolve(model, low_res_image, scale_factor=2, steps=100):
    # Upsample the low-resolution image
    upsampled = F.interpolate(low_res_image, scale_factor=scale_factor, mode='bilinear')
    
    x = upsampled + torch.randn_like(upsampled) * 0.1  # Add a small amount of noise
    
    for t in reversed(range(steps)):
        t_tensor = torch.tensor([t / steps]).float()
        with torch.no_grad():
            predicted_noise = model(x.unsqueeze(0), t_tensor)
        
        # Update x using the predicted noise
        x = (x - (1 - t / steps) * predicted_noise.squeeze()) / (1 - t / steps)**0.5
        
        # Keep the low-frequency components of the upsampled image
        x = x + (upsampled - F.interpolate(F.interpolate(x, scale_factor=1/scale_factor, mode='bilinear'), 
                                           scale_factor=scale_factor, mode='bilinear'))
    
    return x

# Create a low-resolution image
low_res_image = F.interpolate(image_tensor, scale_factor=0.5, mode='bilinear')

super_res_image = super_resolve(model, low_res_image)
print(f"Super-resolved image shape: {super_res_image.shape}")
```

Slide 7: Real-Life Example: Image Restoration

Diffusion models can be used for image restoration, such as removing noise, artifacts, or compression distortions from images. This has applications in photography, medical imaging, and digital archiving.

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

def add_jpeg_artifacts(image, quality=10):
    # Save the image with low JPEG quality
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    compressed_image = Image.open(buffer)
    return transforms.ToTensor()(compressed_image)

def restore_image(model, corrupted_image, steps=100):
    x = corrupted_image + torch.randn_like(corrupted_image) * 0.1
    
    for t in reversed(range(steps)):
        t_tensor = torch.tensor([t / steps]).float()
        with torch.no_grad():
            predicted_noise = model(x.unsqueeze(0), t_tensor)
        
        x = (x - (1 - t / steps) * predicted_noise.squeeze()) / (1 - t / steps)**0.5
    
    return x

# Load and corrupt an image
image = Image.open("sample_image.jpg")
corrupted_image = add_jpeg_artifacts(image)

# Restore the image
restored_image = restore_image(model, corrupted_image)

print(f"Restored image shape: {restored_image.shape}")
```

Slide 8: Real-Life Example: Text-to-Image Generation

Diffusion models can be combined with text encoders to generate images based on textual descriptions. This has applications in creative tools, content creation, and visual aids for education.

```python
import torch
import torch.nn as nn

class TextConditionedDiffusionModel(nn.Module):
    def __init__(self, image_channels=3, hidden_channels=64, text_embedding_dim=512):
        super().__init__()
        self.text_encoder = nn.Sequential(
            nn.Embedding(10000, text_embedding_dim),
            nn.Linear(text_embedding_dim, hidden_channels)
        )
        self.image_encoder = nn.Conv2d(image_channels, hidden_channels, 3, padding=1)
        self.decoder = nn.Conv2d(hidden_channels * 2, image_channels, 3, padding=1)

    def forward(self, x, t, text_embedding):
        text_features = self.text_encoder(text_embedding)
        image_features = self.image_encoder(x)
        combined_features = torch.cat([image_features, text_features.unsqueeze(-1).unsqueeze(-1).expand_as(image_features)], dim=1)
        return self.decoder(combined_features)

# Usage
model = TextConditionedDiffusionModel()
image = torch.randn(1, 3, 256, 256)
text_embedding = torch.randint(0, 10000, (1, 10))  # Assume 10 words in the description
t = torch.tensor([0.5])

output = model(image, t, text_embedding)
print(f"Output shape: {output.shape}")
```

Slide 9: Conditioning Diffusion Models

Diffusion models can be conditioned on various types of information to control the generation process. This includes class labels, text embeddings, or even other images. Conditioning allows for more targeted and controllable image generation.

```python
import torch
import torch.nn as nn

class ConditionedDiffusionModel(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=64, num_classes=10):
        super().__init__()
        self.class_embedding = nn.Embedding(num_classes, hidden_channels)
        self.conv1 = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, input_channels, 3, padding=1)

    def forward(self, x, t, class_label):
        # Embed the class label
        class_embed = self.class_embedding(class_label).unsqueeze(-1).unsqueeze(-1)
        class_embed = class_embed.expand(-1, -1, x.size(2), x.size(3))
        
        # Concatenate the image and class embedding
        x = torch.cat([x, class_embed], dim=1)
        
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x

# Usage
model = ConditionedDiffusionModel()
image = torch.randn(1, 3, 256, 256)
t = torch.tensor([0.5])
class_label = torch.tensor([5])

output = model(image, t, class_label)
print(f"Output shape: {output.shape}")
```

Slide 10: Sampling Strategies for Diffusion Models

Different sampling strategies can be employed when generating images with diffusion models. These strategies can affect the quality and diversity of the generated images, as well as the sampling speed.

```python
import torch

def ddim_sampling(model, steps=100, img_shape=(3, 256, 256), eta=0.0):
    x = torch.randn(img_shape)
    
    for i in reversed(range(steps)):
        t = i / steps
        t_next = max(0, (i - 1) / steps)
        
        with torch.no_grad():
            predicted_noise = model(x.unsqueeze(0), torch.tensor([t]))
        
        # DDIM update rule
        x0_pred = (x - (1 - t)**0.5 * predicted_noise.squeeze()) / t**0.5
        direction = (1 - t_next)**0.5 * predicted_noise.squeeze()
        noise = eta * torch.randn_like(x) if eta > 0 else 0
        x = t_next**0.5 * x0_pred + direction + (1 - t_next - (1 - t_next)**0.5)**0.5 * noise
    
    return x

# Usage
model = SimpleDiffusionModel()
generated_image = ddim_sampling(model, steps=50, eta=0.2)
print(f"Generated image shape: {generated_image.shape}")
```

Slide 11: Latent Diffusion Models

Latent diffusion models operate in a compressed latent space rather than directly on pixels. This approach can significantly reduce computational requirements while maintaining high-quality generation.

```python
import torch
import torch.nn as nn

class LatentDiffusionModel(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, latent_dim, 3, stride=2, padding=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, 3, 3, stride=2, padding=1, output_padding=1)
        )
        self.diffusion = SimpleDiffusionModel(input_channels=latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def denoise(self, z, t):
        return self.diffusion(z, t)

# Usage
model = LatentDiffusionModel()
image = torch.randn(1, 3, 256, 256)
t = torch.tensor([0.5])

latent = model.encode(image)
denoised_latent = model.denoise(latent, t)
reconstructed = model.decode(denoised_latent)

print(f"Latent shape: {latent.shape}")
print(f"Reconstructed shape: {reconstructed.shape}")
```

Slide 12: Evaluating Diffusion Models

Evaluating the quality of generated images is crucial for assessing and improving diffusion models. Common metrics include Inception Score (IS), Fréchet Inception Distance (FID), and Learned Perceptual Image Patch Similarity (LPIPS). These metrics help quantify the realism and diversity of generated images.

```python
import torch
import torchvision.models as models
from scipy import linalg
import numpy as np

def calculate_fid(real_images, generated_images, batch_size=50):
    inception_model = models.inception_v3(pretrained=True, transform_input=False).eval()
    
    def get_activations(images):
        activations = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            activations.append(inception_model(batch).cpu().detach().numpy())
        return np.concatenate(activations)
    
    real_activations = get_activations(real_images)
    generated_activations = get_activations(generated_images)
    
    mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = generated_activations.mean(axis=0), np.cov(generated_activations, rowvar=False)
    
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# Usage example (assuming we have real_images and generated_images tensors)
fid_score = calculate_fid(real_images, generated_images)
print(f"FID Score: {fid_score}")
```

Slide 13: Challenges and Future Directions

Diffusion models face challenges such as slow sampling speed and high computational requirements. Ongoing research focuses on addressing these issues through techniques like guided sampling, efficient architectures, and improved training strategies. Future directions include combining diffusion models with other generative approaches and exploring applications in 3D generation and video synthesis.

```python
import torch
import torch.nn as nn

class EfficientDiffusionModel(nn.Module):
    def __init__(self, channels=64, num_res_blocks=4):
        super().__init__()
        self.initial_conv = nn.Conv2d(3, channels, 3, padding=1)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1)
            ) for _ in range(num_res_blocks)
        ])
        self.final_conv = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x, t):
        h = self.initial_conv(x)
        for res_block in self.res_blocks:
            h = h + res_block(h)
        return self.final_conv(h)

# Usage
model = EfficientDiffusionModel()
x = torch.randn(1, 3, 64, 64)
t = torch.tensor([0.5])
output = model(x, t)
print(f"Output shape: {output.shape}")
```

Slide 14: Applications in Computer Vision Tasks

Diffusion models have shown promise in various computer vision tasks beyond image generation. These include image segmentation, object detection, and image-to-image translation. The ability of diffusion models to capture complex data distributions makes them versatile tools for tackling a wide range of vision problems.

```python
import torch
import torch.nn as nn

class DiffusionSegmentationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, hidden_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, t):
        features = self.encoder(x)
        return self.decoder(features)

# Usage
model = DiffusionSegmentationModel()
image = torch.randn(1, 3, 256, 256)
t = torch.tensor([0.5])
segmentation_mask = model(image, t)
print(f"Segmentation mask shape: {segmentation_mask.shape}")
```

Slide 15: Additional Resources

For those interested in diving deeper into diffusion models, here are some valuable resources:

1. "Denoising Diffusion Probabilistic Models" by Ho et al. (2020) ArXiv: [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
2. "Diffusion Models Beat GANs on Image Synthesis" by Dhariwal and Nichol (2021) ArXiv: [https://arxiv.org/abs/2105.05233](https://arxiv.org/abs/2105.05233)
3. "High-Resolution Image Synthesis with Latent Diffusion Models" by Rombach et al. (2022) ArXiv: [https://arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)

These papers provide in-depth explanations of the theory behind diffusion models and showcase their applications in various image synthesis tasks.
