## Next Token Prediction and Diffusion with Python
Slide 1: Introduction to Next Token Prediction and Diffusion

Next token prediction and diffusion are two powerful techniques in natural language processing and image generation. Next token prediction is used in language models to predict the next word or token in a sequence, while diffusion models generate high-quality images by gradually denoising random noise. This presentation will explore both concepts, their applications, and implementation in Python.

```python
import torch
import torch.nn as nn

# Simple next token prediction model
class NextTokenPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)

# Example usage
vocab_size, embedding_dim, hidden_dim = 1000, 128, 256
model = NextTokenPredictor(vocab_size, embedding_dim, hidden_dim)
input_sequence = torch.randint(0, vocab_size, (1, 10))
next_token_logits = model(input_sequence)
```

Slide 2: Next Token Prediction: The Basics

Next token prediction is a fundamental task in natural language processing. Given a sequence of tokens, the model predicts the most likely next token. This technique is used in various applications, including autocomplete systems, language translation, and text generation.

```python
import torch
import torch.nn.functional as F

def predict_next_token(model, input_sequence, temperature=1.0):
    with torch.no_grad():
        logits = model(input_sequence)
        probabilities = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
    return next_token.item()

# Example usage
input_sequence = torch.randint(0, vocab_size, (1, 10))
next_token = predict_next_token(model, input_sequence)
print(f"Predicted next token: {next_token}")
```

Slide 3: Training a Next Token Prediction Model

Training a next token prediction model involves using a large corpus of text data. The model learns to predict the next token given the previous tokens in the sequence. We use cross-entropy loss to measure the difference between predicted and actual next tokens.

```python
import torch.optim as optim

# Training loop
def train_model(model, data_loader, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch in data_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Example usage (assuming data_loader is defined)
num_epochs, learning_rate = 10, 0.001
train_model(model, data_loader, num_epochs, learning_rate)
```

Slide 4: Real-Life Example: Autocomplete System

An autocomplete system is a practical application of next token prediction. It suggests the next word or phrase as users type, improving typing efficiency and accuracy. Let's implement a simple autocomplete system using our trained model.

```python
def autocomplete(model, input_text, max_length=5):
    tokens = tokenize(input_text)  # Assume we have a tokenize function
    input_sequence = torch.tensor([tokens])
    
    for _ in range(max_length):
        next_token = predict_next_token(model, input_sequence)
        input_sequence = torch.cat([input_sequence, torch.tensor([[next_token]])], dim=1)
    
    return detokenize(input_sequence.squeeze().tolist())  # Assume we have a detokenize function

# Example usage
input_text = "The quick brown"
completed_text = autocomplete(model, input_text)
print(f"Input: {input_text}")
print(f"Autocompleted: {completed_text}")
```

Slide 5: Introduction to Diffusion Models

Diffusion models are a class of generative models that learn to gradually denoise a signal. They start with pure noise and iteratively refine it into a high-quality sample. This process is particularly effective for image generation tasks.

```python
import torch
import torch.nn as nn

class SimpleDiffusionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x, t):
        # t is the diffusion step
        return self.net(torch.cat([x, t.float().unsqueeze(1)], dim=1))

# Example usage
input_dim = 784  # 28x28 image
model = SimpleDiffusionModel(input_dim + 1)  # +1 for the time step
x = torch.randn(1, input_dim)
t = torch.tensor([100])
noise_prediction = model(x, t)
```

Slide 6: The Diffusion Process

The diffusion process involves gradually adding noise to data over a series of steps. The reverse process, denoising, is learned by the model. This approach allows for high-quality sample generation by starting from pure noise and iteratively denoising.

```python
import numpy as np

def diffusion_schedule(num_steps):
    beta = np.linspace(0.0001, 0.02, num_steps)
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)
    return torch.from_numpy(alpha_bar).float()

def add_noise(x, t, alpha_bar):
    noise = torch.randn_like(x)
    noisy_x = torch.sqrt(alpha_bar[t]) * x + torch.sqrt(1 - alpha_bar[t]) * noise
    return noisy_x, noise

# Example usage
num_steps = 1000
alpha_bar = diffusion_schedule(num_steps)
x = torch.randn(1, 784)  # Random 28x28 image
t = torch.randint(0, num_steps, (1,))
noisy_x, noise = add_noise(x, t, alpha_bar)
```

Slide 7: Training a Diffusion Model

Training a diffusion model involves predicting the noise added to the input at each step. The model learns to reverse the diffusion process, allowing it to generate high-quality samples from pure noise.

```python
def train_diffusion_model(model, data_loader, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    alpha_bar = diffusion_schedule(1000)

    for epoch in range(num_epochs):
        for batch in data_loader:
            x = batch[0]
            t = torch.randint(0, 1000, (x.size(0),))
            noisy_x, noise = add_noise(x, t, alpha_bar)
            
            predicted_noise = model(noisy_x, t)
            loss = criterion(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Example usage (assuming data_loader is defined)
num_epochs, learning_rate = 100, 0.001
train_diffusion_model(model, data_loader, num_epochs, learning_rate)
```

Slide 8: Generating Samples with Diffusion Models

Once trained, a diffusion model can generate new samples by starting from pure noise and iteratively denoising. This process allows for the creation of high-quality, diverse samples in various domains, such as images or audio.

```python
def generate_sample(model, num_steps):
    alpha_bar = diffusion_schedule(num_steps)
    x = torch.randn(1, 784)  # Start with pure noise

    for t in reversed(range(num_steps)):
        t_tensor = torch.full((1,), t, dtype=torch.long)
        predicted_noise = model(x, t_tensor)
        
        alpha_t = alpha_bar[t]
        alpha_t_minus_1 = alpha_bar[t - 1] if t > 0 else torch.tensor(1.0)
        
        x = (x - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
        
        if t > 0:
            noise = torch.randn_like(x)
            sigma_t = ((1 - alpha_t_minus_1) / (1 - alpha_t) * (1 - alpha_t / alpha_t_minus_1)).sqrt()
            x += sigma_t * noise

    return x

# Example usage
generated_sample = generate_sample(model, num_steps=1000)
```

Slide 9: Real-Life Example: Image Inpainting

Image inpainting is a practical application of diffusion models. It involves filling in missing or corrupted parts of an image. Let's implement a simple image inpainting system using our trained diffusion model.

```python
import torchvision.transforms as T

def inpaint_image(model, image, mask, num_steps):
    # Assume image and mask are PyTorch tensors
    noisy_image = image * (1 - mask) + torch.randn_like(image) * mask
    
    for t in reversed(range(num_steps)):
        t_tensor = torch.full((1,), t, dtype=torch.long)
        predicted_noise = model(noisy_image, t_tensor)
        
        alpha_bar = diffusion_schedule(num_steps)
        alpha_t = alpha_bar[t]
        alpha_t_minus_1 = alpha_bar[t - 1] if t > 0 else torch.tensor(1.0)
        
        noisy_image = (noisy_image - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
        
        if t > 0:
            noise = torch.randn_like(noisy_image)
            sigma_t = ((1 - alpha_t_minus_1) / (1 - alpha_t) * (1 - alpha_t / alpha_t_minus_1)).sqrt()
            noisy_image += sigma_t * noise * mask
        
        noisy_image = image * (1 - mask) + noisy_image * mask

    return noisy_image

# Example usage (assuming we have a trained model and an image with a mask)
inpainted_image = inpaint_image(model, image, mask, num_steps=1000)

# Visualize the result
import matplotlib.pyplot as plt
plt.imshow(T.ToPILImage()(inpainted_image.squeeze()))
plt.axis('off')
plt.show()
```

Slide 10: Combining Next Token Prediction and Diffusion

While next token prediction and diffusion models are typically used separately, there are interesting ways to combine these techniques. For example, we can use a language model to guide the generation process of a diffusion model, creating text-conditional image generation.

```python
def text_guided_image_generation(text_model, diffusion_model, text_prompt, num_steps):
    # Encode the text prompt
    text_encoding = text_model.encode(text_prompt)
    
    # Start with random noise
    image = torch.randn(1, 3, 256, 256)
    
    for t in reversed(range(num_steps)):
        t_tensor = torch.full((1,), t, dtype=torch.long)
        
        # Combine image and text information
        combined_input = torch.cat([image, text_encoding.unsqueeze(2).unsqueeze(3).repeat(1, 1, 256, 256)], dim=1)
        
        # Predict and apply noise
        predicted_noise = diffusion_model(combined_input, t_tensor)
        image = apply_noise(image, predicted_noise, t, num_steps)
    
    return image

# Example usage (assuming we have trained text and diffusion models)
text_prompt = "A beautiful sunset over the ocean"
generated_image = text_guided_image_generation(text_model, diffusion_model, text_prompt, num_steps=1000)

# Visualize the result
plt.imshow(T.ToPILImage()(generated_image.squeeze()))
plt.axis('off')
plt.title(text_prompt)
plt.show()
```

Slide 11: Optimizing Performance: Parallelization and GPU Acceleration

Both next token prediction and diffusion models can benefit from parallelization and GPU acceleration. Here's an example of how to leverage PyTorch's built-in parallelization and move computations to GPU for faster processing.

```python
import torch.multiprocessing as mp

def parallel_text_generation(model, prompts, max_length):
    def generate(prompt):
        return autocomplete(model, prompt, max_length)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(generate, prompts)
    return results

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Example usage
prompts = ["The quick brown", "In a galaxy", "Once upon a"]
generated_texts = parallel_text_generation(model, prompts, max_length=10)

for prompt, generated in zip(prompts, generated_texts):
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}\n")
```

Slide 12: Advanced Techniques: Attention Mechanisms in Next Token Prediction

Attention mechanisms have revolutionized natural language processing tasks, including next token prediction. Let's implement a simple attention-based model for next token prediction.

```python
class AttentionNextTokenPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=4)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        attn_output, _ = self.attention(embedded, embedded, embedded)
        return self.fc(attn_output)

# Example usage
vocab_size, embedding_dim, hidden_dim = 1000, 128, 256
attention_model = AttentionNextTokenPredictor(vocab_size, embedding_dim, hidden_dim)
input_sequence = torch.randint(0, vocab_size, (1, 10))
next_token_logits = attention_model(input_sequence)
```

Slide 13: Advanced Techniques: Conditional Diffusion Models

Conditional diffusion models allow for more controlled generation by incorporating additional information. Let's implement a simple conditional diffusion model that takes a class label as input.

```python
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + num_classes + 1, 128),  # +1 for time step
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        self.class_embedding = nn.Embedding(num_classes, num_classes)

    def forward(self, x, t, class_label):
        t_embedding = t.float().unsqueeze(1)
        class_embedding = self.class_embedding(class_label)
        combined_input = torch.cat([x, t_embedding, class_embedding], dim=1)
        return self.net(combined_input)

# Example usage
input_dim, num_classes = 784, 10
model = ConditionalDiffusionModel(input_dim, num_classes)
x = torch.randn(1, input_dim)
t = torch.tensor([100])
class_label = torch.tensor([3])
noise_prediction = model(x, t, class_label)
```

Slide 14: Evaluating Next Token Prediction Models

Evaluating the performance of next token prediction models is crucial for understanding their effectiveness. Common metrics include perplexity and accuracy. Let's implement functions to calculate these metrics.

```python
import math

def calculate_perplexity(model, data_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += targets.numel()

    perplexity = math.exp(total_loss / total_tokens)
    return perplexity

def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=-1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

    accuracy = correct / total
    return accuracy

# Example usage (assuming data_loader is defined)
perplexity = calculate_perplexity(model, data_loader)
accuracy = calculate_accuracy(model, data_loader)
print(f"Perplexity: {perplexity:.2f}")
print(f"Accuracy: {accuracy:.2%}")
```

Slide 15: Additional Resources

For those interested in diving deeper into next token prediction and diffusion models, here are some valuable resources:

1. "Denoising Diffusion Probabilistic Models" by Ho et al. (2020) ArXiv URL: [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
2. "Attention Is All You Need" by Vaswani et al. (2017) ArXiv URL: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
3. "Language Models are Few-Shot Learners" by Brown et al. (2020) ArXiv URL: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

These papers provide in-depth explanations of the concepts and techniques discussed in this presentation, offering a solid foundation for further exploration in the field of natural language processing and generative models.

