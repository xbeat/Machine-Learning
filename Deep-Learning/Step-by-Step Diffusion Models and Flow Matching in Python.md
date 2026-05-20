## Step-by-Step Diffusion Models and Flow Matching in Python
Slide 1: Introduction to Step-by-Step Diffusion Models

Diffusion models are a class of generative models that learn to gradually denoise data. They start with pure noise and iteratively refine it into a desired output. This process mimics the reverse of a diffusion process, where a clean signal is progressively corrupted with noise.

```python
import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleUNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, x):
        x1 = torch.relu(self.conv1(x))
        return self.conv2(x1)

model = SimpleUNet(3, 3)  # For RGB images
```

Slide 2: The Diffusion Process

The diffusion process involves gradually adding Gaussian noise to an image over a series of steps. This process transforms a clear image into pure noise. The goal of a diffusion model is to learn and reverse this process.

```python
def diffusion_process(image, num_steps=1000):
    noisy_images = []
    for t in range(num_steps):
        noise = torch.randn_like(image)
        noisy_image = torch.sqrt(1 - 0.02) * image + torch.sqrt(0.02) * noise
        noisy_images.append(noisy_image)
        image = noisy_image
    return noisy_images

# Example usage
original_image = torch.randn(1, 3, 64, 64)  # Random 64x64 RGB image
noisy_sequence = diffusion_process(original_image)
```

Slide 3: Reverse Diffusion: The Generative Process

The generative process in diffusion models starts with pure noise and gradually denoises it to create an image. This is achieved by learning to predict and remove the noise at each step.

```python
def reverse_diffusion(model, num_steps=1000):
    x = torch.randn(1, 3, 64, 64)  # Start with pure noise
    for t in reversed(range(num_steps)):
        z = torch.randn_like(x) if t > 0 else 0
        predicted_noise = model(x, t)
        x = 1 / torch.sqrt(1 - 0.02) * (x - 0.02 / torch.sqrt(1 - 0.02) * predicted_noise) + torch.sqrt(0.02) * z
    return x

# Example usage
generated_image = reverse_diffusion(model)
```

Slide 4: Training a Diffusion Model

Training a diffusion model involves teaching it to predict the noise added at each step. The model learns to estimate the noise given a noisy image and the timestep.

```python
def train_step(model, optimizer, image):
    noise = torch.randn_like(image)
    t = torch.randint(0, 1000, (1,))
    noisy_image = torch.sqrt(1 - 0.02 ** t) * image + torch.sqrt(0.02 ** t) * noise
    
    predicted_noise = model(noisy_image, t)
    loss = nn.MSELoss()(predicted_noise, noise)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(100):
    loss = train_step(model, optimizer, original_image)
    print(f"Epoch {epoch}, Loss: {loss}")
```

Slide 5: Flow Matching: An Alternative Approach

Flow matching is another technique for generative modeling. Unlike diffusion models, flow matching learns a continuous transformation between noise and data distributions using ordinary differential equations (ODEs).

```python
import torch.nn as nn

class FlowMatchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t.unsqueeze(1)], dim=1))

model = FlowMatchingModel()
```

Slide 6: Flow Matching: Training Process

Training a flow matching model involves learning the vector field that transforms the noise distribution to the data distribution. This is typically done by minimizing the difference between the model's output and the ground truth flow.

```python
def train_flow_matching(model, optimizer, data):
    noise = torch.randn_like(data)
    t = torch.rand(data.shape[0], 1)
    x_t = (1 - t) * data + t * noise
    
    ground_truth_flow = noise - data
    predicted_flow = model(x_t, t)
    
    loss = nn.MSELoss()(predicted_flow, ground_truth_flow)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(100):
    loss = train_flow_matching(model, optimizer, data)
    print(f"Epoch {epoch}, Loss: {loss}")
```

Slide 7: Generating Samples with Flow Matching

To generate samples using a trained flow matching model, we solve the ODE that describes the transformation from noise to data. This is typically done using numerical integration methods.

```python
from torchdiffeq import odeint

def ode_func(t, x):
    return model(x, t.unsqueeze(0).expand(x.shape[0], 1))

def generate_samples(model, num_samples=100):
    z = torch.randn(num_samples, 2)
    t = torch.linspace(1, 0, 100)
    samples = odeint(ode_func, z, t)
    return samples[-1]

# Generate samples
samples = generate_samples(model)
```

Slide 8: Comparing Diffusion Models and Flow Matching

While both diffusion models and flow matching aim to transform noise into data, they differ in their approaches. Diffusion models use a discrete, step-by-step process, while flow matching employs a continuous transformation.

```python
import matplotlib.pyplot as plt

def plot_comparison(diffusion_samples, flow_matching_samples):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(diffusion_samples[:, 0], diffusion_samples[:, 1], alpha=0.5)
    plt.title("Diffusion Model Samples")
    plt.subplot(1, 2, 2)
    plt.scatter(flow_matching_samples[:, 0], flow_matching_samples[:, 1], alpha=0.5)
    plt.title("Flow Matching Samples")
    plt.show()

diffusion_samples = reverse_diffusion(diffusion_model)
flow_matching_samples = generate_samples(flow_matching_model)
plot_comparison(diffusion_samples, flow_matching_samples)
```

Slide 9: Real-life Example: Image Generation

One common application of both diffusion models and flow matching is image generation. Let's consider a simple example of generating handwritten digits using MNIST data.

```python
from torchvision import datasets, transforms

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Train models (simplified for brevity)
diffusion_model = SimpleUNet(1, 1)  # For grayscale images
flow_matching_model = FlowMatchingModel()

# Generate samples
diffusion_samples = reverse_diffusion(diffusion_model)
flow_matching_samples = generate_samples(flow_matching_model)

# Visualize results
plot_comparison(diffusion_samples, flow_matching_samples)
```

Slide 10: Real-life Example: Molecule Generation

Another interesting application is in drug discovery, where these models can be used to generate novel molecular structures. Here's a simplified example using a graph-based representation.

```python
import torch
import torch.nn as nn
import torch_geometric

class MoleculeGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_emb = nn.Embedding(10, 64)  # 10 atom types
        self.edge_emb = nn.Embedding(4, 64)   # 4 bond types
        self.gnn = torch_geometric.nn.GCNConv(64, 64)
        self.output = nn.Linear(64, 10)  # Predict atom types

    def forward(self, x, edge_index, edge_attr):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        x = self.gnn(x, edge_index, edge_attr)
        return self.output(x)

model = MoleculeGenerator()
# Training and generation steps would follow...
```

Slide 11: Challenges and Limitations

While powerful, both diffusion models and flow matching face challenges. Diffusion models can be slow to sample from due to their iterative nature. Flow matching, while faster, can struggle with very complex data distributions.

```python
import time

def benchmark_sampling(diffusion_model, flow_matching_model, num_samples=100):
    start = time.time()
    diffusion_samples = reverse_diffusion(diffusion_model, num_samples)
    diffusion_time = time.time() - start

    start = time.time()
    flow_matching_samples = generate_samples(flow_matching_model, num_samples)
    flow_matching_time = time.time() - start

    print(f"Diffusion sampling time: {diffusion_time:.2f}s")
    print(f"Flow matching sampling time: {flow_matching_time:.2f}s")

benchmark_sampling(diffusion_model, flow_matching_model)
```

Slide 12: Future Directions

Research in these areas is ongoing, with focus on improving sampling speed, handling more complex data types, and combining strengths of different approaches. Hybrid models and improved architectures are active areas of exploration.

```python
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_branch = SimpleUNet(3, 3)
        self.flow_matching_branch = FlowMatchingModel()
        self.combiner = nn.Linear(4, 2)

    def forward(self, x, t):
        diff_out = self.diffusion_branch(x)
        flow_out = self.flow_matching_branch(x, t)
        combined = torch.cat([diff_out, flow_out], dim=1)
        return self.combiner(combined)

hybrid_model = HybridModel()
# Training and inference would combine aspects of both approaches
```

Slide 13: Conclusion

Step-by-step diffusion models and flow matching represent powerful approaches in generative modeling. Each has its strengths: diffusion models excel in image generation tasks, while flow matching offers faster sampling. As research progresses, we can expect these methods to become even more versatile and efficient, opening up new possibilities in various fields from computer vision to drug discovery.

```python
def final_thoughts():
    print("Key takeaways:")
    print("1. Diffusion models: Gradual denoising process")
    print("2. Flow matching: Continuous transformation via ODEs")
    print("3. Both have shown impressive results in various domains")
    print("4. Active research area with ongoing improvements")
    print("5. Hybrid approaches may combine strengths of both methods")

final_thoughts()
```

Slide 14: Additional Resources

For those interested in diving deeper into these topics, here are some valuable resources:

1. "Denoising Diffusion Probabilistic Models" by Ho et al. (2020) ArXiv: [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
2. "Flow Matching for Generative Modeling" by Lipman et al. (2022) ArXiv: [https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)
3. "Diffusion Models Beat GANs on Image Synthesis" by Dhariwal and Nichol (2021) ArXiv: [https://arxiv.org/abs/2105.05233](https://arxiv.org/abs/2105.05233)
4. "Understanding Diffusion Models: A Unified Perspective" by Luo (2022) ArXiv: [https://arxiv.org/abs/2208.11970](https://arxiv.org/abs/2208.11970)

These papers provide in-depth explanations and mathematical foundations for the concepts discussed in this presentation.

