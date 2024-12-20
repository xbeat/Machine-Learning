## Creating a LoRA Adapter with Python
Slide 1: Introduction to LoRA Adapters

LoRA (Low-Rank Adaptation) is a technique used to efficiently fine-tune large language models. It reduces the number of trainable parameters by adding pairs of rank-decomposition matrices to existing weights, enabling faster and more memory-efficient adaptation of pre-trained models.

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / rank)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = 1.0

    def forward(self, x):
        return (self.lora_B @ self.lora_A @ x.T).T * self.scaling
```

Slide 2: LoRA Architecture

LoRA introduces trainable rank decomposition matrices A and B to the original weight matrix W. The adapted weight W' is computed as W' = W + BA, where B and A are low-rank matrices. This approach significantly reduces the number of trainable parameters while maintaining model performance.

```python
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora = LoRALayer(in_features, out_features, rank)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

# Example usage
lora_linear = LoRALinear(768, 512, rank=8)
input_tensor = torch.randn(32, 768)
output = lora_linear(input_tensor)
print(f"Output shape: {output.shape}")  # Output shape: torch.Size([32, 512])
```

Slide 3: Implementing LoRA in a Transformer Layer

LoRA can be applied to specific parts of a transformer model, such as the attention mechanism. Here's an example of how to implement LoRA in a multi-head attention layer:

```python
class LoRAMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, rank=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.lora_q = LoRALayer(embed_dim, embed_dim, rank)
        self.lora_k = LoRALayer(embed_dim, embed_dim, rank)
        self.lora_v = LoRALayer(embed_dim, embed_dim, rank)

    def forward(self, query, key, value, attn_mask=None):
        q = query + self.lora_q(query)
        k = key + self.lora_k(key)
        v = value + self.lora_v(value)
        return self.mha(q, k, v, attn_mask=attn_mask)

# Example usage
lora_mha = LoRAMultiHeadAttention(512, 8, rank=16)
q = k = v = torch.randn(10, 32, 512)
output, _ = lora_mha(q, k, v)
print(f"Output shape: {output.shape}")  # Output shape: torch.Size([10, 32, 512])
```

Slide 4: Training LoRA Adapters

When training a model with LoRA adapters, we freeze the original model parameters and only update the LoRA weights. This approach significantly reduces the number of trainable parameters and memory requirements.

```python
def freeze_base_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_lora_params(model):
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

# Example usage
model = YourPreTrainedModel()
freeze_base_model(model)
add_lora_layers(model)  # Implement this function to add LoRA layers to your model
unfreeze_lora_params(model)

# Training loop
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
```

Slide 5: LoRA Scaling Factor

The scaling factor in LoRA controls the magnitude of the adaptation. It helps balance the contribution of the original weights and the LoRA adaptation. Here's an example of how to implement and use the scaling factor:

```python
class LoRALayerWithScaling(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / rank)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        return (self.lora_B @ self.lora_A @ x.T).T * self.scaling

# Example usage
lora_layer = LoRALayerWithScaling(768, 512, rank=8, alpha=16)
input_tensor = torch.randn(32, 768)
output = lora_layer(input_tensor)
print(f"Output shape: {output.shape}")  # Output shape: torch.Size([32, 512])
```

Slide 6: Merging LoRA Weights

After training, LoRA weights can be merged with the original model weights for efficient inference. This process eliminates the need for separate LoRA computations during inference.

```python
def merge_lora_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.linear.weight.data += (
                    module.lora.lora_B @ module.lora.lora_A
                ) * module.lora.scaling
            delattr(module, 'lora')
    return model

# Example usage
model = YourLoRAModel()
merged_model = merge_lora_weights(model)

# Inference with merged model
input_data = torch.randn(1, 768)
with torch.no_grad():
    output = merged_model(input_data)
print(f"Output shape: {output.shape}")
```

Slide 7: LoRA for Different Layer Types

LoRA can be applied to various types of layers in neural networks, not just linear layers. Here's an example of applying LoRA to a convolutional layer:

```python
class LoRAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.lora_A = nn.Parameter(torch.randn(rank, in_channels * kernel_size ** 2) / rank)
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank))
        self.scaling = 1.0

    def forward(self, x):
        conv_out = self.conv(x)
        b, c, h, w = conv_out.shape
        lora_out = F.conv2d(
            x,
            (self.lora_B @ self.lora_A).view(self.conv.weight.shape),
            stride=self.conv.stride,
            padding=self.conv.padding,
        )
        return conv_out + lora_out * self.scaling

# Example usage
lora_conv = LoRAConv2d(3, 64, kernel_size=3, rank=8)
input_tensor = torch.randn(1, 3, 32, 32)
output = lora_conv(input_tensor)
print(f"Output shape: {output.shape}")  # Output shape: torch.Size([1, 64, 30, 30])
```

Slide 8: LoRA for Cross-lingual Adaptation

LoRA can be used for efficient cross-lingual adaptation of language models. By training LoRA adapters for different languages, we can quickly adapt a base model to multiple languages without fine-tuning the entire model for each language.

```python
class MultilingualLoRAModel(nn.Module):
    def __init__(self, base_model, num_languages, embed_dim, rank=4):
        super().__init__()
        self.base_model = base_model
        self.language_adapters = nn.ModuleList([
            LoRALayer(embed_dim, embed_dim, rank) for _ in range(num_languages)
        ])

    def forward(self, input_ids, language_id):
        base_output = self.base_model(input_ids)
        lora_output = self.language_adapters[language_id](base_output)
        return base_output + lora_output

# Example usage
base_model = YourPreTrainedModel()
multilingual_model = MultilingualLoRAModel(base_model, num_languages=5, embed_dim=768, rank=16)

input_ids = torch.randint(0, 1000, (1, 50))  # Batch size 1, sequence length 50
language_id = 2  # Example: using the adapter for the third language

output = multilingual_model(input_ids, language_id)
print(f"Output shape: {output.shape}")
```

Slide 9: LoRA for Domain Adaptation

LoRA can be employed for efficient domain adaptation of large language models. By training separate LoRA adapters for different domains, we can quickly adapt a base model to multiple domains without fine-tuning the entire model for each domain.

```python
class MultiDomainLoRAModel(nn.Module):
    def __init__(self, base_model, num_domains, embed_dim, rank=4):
        super().__init__()
        self.base_model = base_model
        self.domain_adapters = nn.ModuleList([
            LoRALayer(embed_dim, embed_dim, rank) for _ in range(num_domains)
        ])

    def forward(self, input_ids, domain_id):
        base_output = self.base_model(input_ids)
        lora_output = self.domain_adapters[domain_id](base_output)
        return base_output + lora_output

# Example usage
base_model = YourPreTrainedModel()
multi_domain_model = MultiDomainLoRAModel(base_model, num_domains=3, embed_dim=768, rank=16)

input_ids = torch.randint(0, 1000, (1, 50))  # Batch size 1, sequence length 50
domain_id = 1  # Example: using the adapter for the second domain

output = multi_domain_model(input_ids, domain_id)
print(f"Output shape: {output.shape}")
```

Slide 10: LoRA for Task-specific Adaptation

LoRA can be used for efficient task-specific adaptation of large language models. By training separate LoRA adapters for different tasks, we can quickly adapt a base model to multiple tasks without fine-tuning the entire model for each task.

```python
class MultiTaskLoRAModel(nn.Module):
    def __init__(self, base_model, num_tasks, embed_dim, rank=4):
        super().__init__()
        self.base_model = base_model
        self.task_adapters = nn.ModuleList([
            LoRALayer(embed_dim, embed_dim, rank) for _ in range(num_tasks)
        ])

    def forward(self, input_ids, task_id):
        base_output = self.base_model(input_ids)
        lora_output = self.task_adapters[task_id](base_output)
        return base_output + lora_output

# Example usage
base_model = YourPreTrainedModel()
multi_task_model = MultiTaskLoRAModel(base_model, num_tasks=4, embed_dim=768, rank=16)

input_ids = torch.randint(0, 1000, (1, 50))  # Batch size 1, sequence length 50
task_id = 0  # Example: using the adapter for the first task

output = multi_task_model(input_ids, task_id)
print(f"Output shape: {output.shape}")
```

Slide 11: LoRA for Efficient Fine-tuning of Vision Transformers

LoRA can be applied to efficiently fine-tune Vision Transformers (ViT) for various computer vision tasks. This example demonstrates how to add LoRA adapters to a ViT model:

```python
import torchvision.models as models

class LoRAViT(nn.Module):
    def __init__(self, num_classes, rank=4):
        super().__init__()
        self.vit = models.vit_b_16(pretrained=True)
        embed_dim = self.vit.hidden_dim
        self.lora = LoRALayer(embed_dim, embed_dim, rank)
        self.vit.heads = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.vit.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.vit.class_token.expand(x.shape[0], -1, -1) + x
        x = self.vit.pos_embed + x
        x = self.vit.encoder(x)
        x = self.lora(x)
        x = self.vit.heads(x[:, 0])
        return x

# Example usage
lora_vit = LoRAViT(num_classes=10, rank=16)
input_tensor = torch.randn(1, 3, 224, 224)
output = lora_vit(input_tensor)
print(f"Output shape: {output.shape}")  # Output shape: torch.Size([1, 10])
```

Slide 12: LoRA for Efficient Adaptation of Generative Models

LoRA can be used to efficiently adapt generative models, such as GANs or VAEs, to new domains or styles. This example shows how to apply LoRA to a simple GAN generator:

```python
class LoRAGenerator(nn.Module):
    def __init__(self, latent_dim, img_shape, rank=4):
        super().__init__()
        self.img_shape = img_shape
        self.base_gen = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.lora = LoRALayer(latent_dim, int(np.prod(img_shape)), rank)

    def forward(self, z):
        base_output = self.base_gen(z)
        lora_output = self.lora(z)
        combined_output = base_output + lora_output
        return combined_output.view(combined_output.size(0), *self.img_shape)

# Example usage
latent_dim = 100
img_shape = (1, 28, 28)  # For MNIST-like images
lora_gen = LoRAGenerator(latent_dim, img_shape, rank=8)

z = torch.randn(16, latent_dim)  # Generate 16 images
fake_images = lora_gen(z)
print(f"Generated images shape: {fake_images.shape}")  # Shape: torch.Size([16, 1, 28, 28])
```

Slide 13: LoRA for Reinforcement Learning

LoRA can be applied to efficiently adapt pre-trained reinforcement learning models to new environments or tasks. This example demonstrates how to use LoRA with a simple DQN (Deep Q-Network) agent:

```python
import gym
import numpy as np

class LoRADQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, rank=4):
        super().__init__()
        self.base_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.lora = LoRALayer(state_dim, action_dim, rank)

    def forward(self, state):
        base_q_values = self.base_network(state)
        lora_q_values = self.lora(state)
        return base_q_values + lora_q_values

# Example usage
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

dqn = LoRADQN(state_dim, action_dim, rank=8)
state = env.reset()
state = torch.FloatTensor(state).unsqueeze(0)
q_values = dqn(state)
print(f"Q-values shape: {q_values.shape}")  # Shape: torch.Size([1, 2])
```

Slide 14: LoRA for Efficient Meta-Learning

LoRA can be integrated into meta-learning algorithms to enable quick adaptation to new tasks. This example shows how to use LoRA with a simple MAML (Model-Agnostic Meta-Learning) implementation:

```python
class LoRAMetaLearner(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, rank=4):
        super().__init__()
        self.base_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.lora = LoRALayer(input_dim, output_dim, rank)

    def forward(self, x):
        return self.base_network(x) + self.lora(x)

    def adapt(self, support_x, support_y, steps=5, lr=0.01):
        adapted_model = .deep(self)
        optimizer = torch.optim.SGD(adapted_model.lora.parameters(), lr=lr)
        
        for _ in range(steps):
            pred_y = adapted_model(support_x)
            loss = F.mse_loss(pred_y, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return adapted_model

# Example usage
meta_learner = LoRAMetaLearner(input_dim=10, output_dim=1, rank=8)
support_x = torch.randn(5, 10)
support_y = torch.randn(5, 1)
adapted_model = meta_learner.adapt(support_x, support_y)

query_x = torch.randn(3, 10)
query_y = adapted_model(query_x)
print(f"Query output shape: {query_y.shape}")  # Shape: torch.Size([3, 1])
```

Slide 15: Additional Resources

For more information on LoRA and its applications, consider exploring the following resources:

1. Original LoRA paper: "LoRA: Low-Rank Adaptation of Large Language Models" by Hu et al. (2021) ArXiv: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
2. "QLoRA: Efficient Finetuning of Quantized LLMs" by Dettmers et al. (2023) ArXiv: [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
3. "LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models" by Chen et al. (2023) ArXiv: [https://arxiv.org/abs/2309.12307](https://arxiv.org/abs/2309.12307)

These papers provide in-depth explanations of LoRA and its variations, as well as experimental results and potential applications in various domains of machine learning and natural language processing.

