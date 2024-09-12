## Denoising Diffusion Probabilistic Models in Python
Slide 1: Introduction to Denoising Diffusion Probabilistic Models

Denoising Diffusion Probabilistic Models (DDPMs) are a class of generative models that have gained significant attention in recent years. They work by gradually adding noise to data and then learning to reverse this process.

```python
import torch
import torch.nn as nn

class DDPM(nn.Module):
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.beta = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
```

Slide 2: The Diffusion Process

The diffusion process involves gradually adding noise to an image over a series of steps. This process transforms a complex data distribution into a simple Gaussian distribution.

```python
def forward_diffusion(self, x, t):
    sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
    epsilon = torch.randn_like(x)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
```

Slide 3: The Reverse Process

The reverse process, also known as denoising, aims to gradually remove noise from a noisy image to recover the original clean image.

```python
def reverse_diffusion(self, model, x, t):
    betas_t = self.beta[t][:, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
    sqrt_recip_alphas_t = torch.sqrt(1. / self.alpha[t])[:, None, None, None]
    
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    
    if t > 0:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(betas_t) * noise
    else:
        return model_mean
```

Slide 4: The U-Net Architecture

DDPMs often use a U-Net architecture as the backbone of the model. U-Net is particularly effective for image-to-image tasks due to its ability to capture both local and global features.

```python
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = self.conv_block(3, 64)
        self.down2 = self.conv_block(64, 128)
        self.down3 = self.conv_block(128, 256)
        
        self.up1 = self.upconv_block(256, 128)
        self.up2 = self.upconv_block(128, 64)
        self.out = nn.Conv2d(64, 3, kernel_size=1)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            self.conv_block(out_ch, out_ch)
        )
    
    def forward(self, x, t):
        # U-Net forward pass implementation
        pass
```

Slide 5: Training the DDPM

Training a DDPM involves minimizing the difference between the predicted noise and the actual noise added during the forward process.

```python
def train_step(model, optimizer, x):
    t = torch.randint(0, noise_steps, (x.shape[0],)).to(device)
    x_t, noise = forward_diffusion(x, t)
    predicted_noise = model(x_t, t)
    loss = nn.MSELoss()(noise, predicted_noise)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(model, optimizer, batch)
        print(f"Epoch {epoch}, Loss: {loss}")
```

Slide 6: Sampling from the DDPM

To generate new samples, we start with random noise and iteratively apply the reverse process.

```python
def sample(model, n_samples, image_size):
    model.eval()
    with torch.no_grad():
        x = torch.randn((n_samples, 3, image_size, image_size)).to(device)
        for i in reversed(range(noise_steps)):
            t = torch.full((n_samples,), i, dtype=torch.long).to(device)
            x = reverse_diffusion(model, x, t)
    model.train()
    return x
```

Slide 7: Real-life Example: Image Denoising

DDPMs can be used for image denoising tasks. Here's an example of how to denoise an image:

```python
def denoise_image(model, noisy_image, noise_level):
    model.eval()
    with torch.no_grad():
        x = noisy_image.unsqueeze(0).to(device)
        t = torch.full((1,), noise_level, dtype=torch.long).to(device)
        for i in reversed(range(noise_level + 1)):
            t[:] = i
            x = reverse_diffusion(model, x, t)
    model.train()
    return x.squeeze(0)

# Usage
noisy_image = add_noise_to_image(original_image, noise_level=50)
denoised_image = denoise_image(model, noisy_image, noise_level=50)
```

Slide 8: Real-life Example: Image Inpainting

DDPMs can also be used for image inpainting, where missing or corrupted parts of an image are reconstructed:

```python
def inpaint_image(model, incomplete_image, mask, noise_steps=1000):
    model.eval()
    with torch.no_grad():
        x = incomplete_image.unsqueeze(0).to(device)
        for i in reversed(range(noise_steps)):
            t = torch.full((1,), i, dtype=torch.long).to(device)
            x_denoised = reverse_diffusion(model, x, t)
            x = x_denoised * mask + incomplete_image * (1 - mask)
    model.train()
    return x.squeeze(0)

# Usage
incomplete_image = create_incomplete_image(original_image, mask)
inpainted_image = inpaint_image(model, incomplete_image, mask)
```

Slide 9: Condition Diffusion Models

Conditional DDPMs allow us to generate samples based on specific conditions or attributes.

```python
class ConditionalUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, 256)
        # ... (rest of the U-Net architecture)
    
    def forward(self, x, t, class_label):
        embed = self.embedding(class_label)
        # Incorporate the embedding into the U-Net forward pass
        # ...

# Usage
class_label = torch.tensor([5]).to(device)  # Example: generate a sample of class 5
sample = sample(model, 1, image_size, class_label)
```

Slide 10: Efficient Sampling with DDIM

Denoising Diffusion Implicit Models (DDIM) allow for faster sampling by skipping steps in the reverse process.

```python
def ddim_sampling(model, n_samples, image_size, ddim_steps=50, ddim_eta=0.0):
    model.eval()
    with torch.no_grad():
        x = torch.randn((n_samples, 3, image_size, image_size)).to(device)
        time_steps = torch.linspace(noise_steps - 1, 0, ddim_steps).long().to(device)
        
        for i in range(ddim_steps):
            t = time_steps[i]
            pred_noise = model(x, t)
            
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            sigma = ddim_eta * torch.sqrt((1 - alpha / alpha_hat) * (1 - alpha_hat) / (1 - alpha))
            
            c1 = torch.sqrt(alpha_hat)
            c2 = torch.sqrt(1 - alpha_hat - sigma ** 2)
            
            x = c1 * x - c2 * pred_noise
            
            if i < ddim_steps - 1:
                noise = torch.randn_like(x)
                x += sigma * noise
    
    model.train()
    return x
```

Slide 11: Latent Diffusion Models

Latent Diffusion Models (LDMs) apply the diffusion process in a compressed latent space, allowing for more efficient training and sampling.

```python
class LatentDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.diffusion = DDPM()
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, t):
        z = self.encode(x)
        noise = self.diffusion(z, t)
        return noise

# Usage
latent_model = LatentDiffusionModel()
z = latent_model.encode(x)
sample = latent_model.diffusion.sample(latent_model, n_samples, latent_size)
generated_image = latent_model.decode(sample)
```

Slide 12: Evaluating DDPMs

Evaluating the quality of generated samples is crucial. Common metrics include Inception Score (IS) and Fréchet Inception Distance (FID).

```python
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

def evaluate_model(model, real_images, n_samples):
    generated_images = sample(model, n_samples, image_size)
    
    fid = FrechetInceptionDistance(feature=2048)
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)
    fid_score = fid.compute()
    
    is_metric = InceptionScore()
    is_metric.update(generated_images)
    is_score = is_metric.compute()
    
    return fid_score, is_score

# Usage
fid, is_score = evaluate_model(model, real_images, n_samples=1000)
print(f"FID: {fid}, Inception Score: {is_score}")
```

Slide 13: Challenges and Future Directions

DDPMs face challenges such as slow sampling and high computational requirements. Current research focuses on addressing these issues and exploring new applications.

```python
def visualize_sampling_time():
    sampling_times = []
    step_sizes = range(10, 1001, 10)
    
    for steps in step_sizes:
        start_time = time.time()
        sample(model, 1, image_size, noise_steps=steps)
        sampling_times.append(time.time() - start_time)
    
    plt.plot(step_sizes, sampling_times)
    plt.xlabel('Number of Sampling Steps')
    plt.ylabel('Sampling Time (s)')
    plt.title('DDPM Sampling Time vs. Number of Steps')
    plt.show()

visualize_sampling_time()
```

Slide 14: Additional Resources

For more information on Denoising Diffusion Probabilistic Models, consider exploring these resources:

1. "Denoising Diffusion Probabilistic Models" by Ho et al. (2020) ArXiv: [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
2. "Improved Denoising Diffusion Probabilistic Models" by Nichol and Dhariwal (2021) ArXiv: [https://arxiv.org/abs/2102.09672](https://arxiv.org/abs/2102.09672)
3. "Diffusion Models Beat GANs on Image Synthesis" by Dhariwal and Nichol (2021) ArXiv: [https://arxiv.org/abs/2105.05233](https://arxiv.org/abs/2105.05233)
4. "High-Resolution Image Synthesis with Latent Diffusion Models" by Rombach et al. (2022) ArXiv: [https://arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)

