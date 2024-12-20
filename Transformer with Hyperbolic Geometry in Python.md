## Transformer with Hyperbolic Geometry in Python
Slide 1: Introduction to Transformers with Hyperbolic Geometry

Transformers have revolutionized natural language processing and other sequence-based tasks. By incorporating hyperbolic geometry, we can enhance their ability to capture hierarchical structures in data. This presentation explores the integration of hyperbolic geometry into transformer architectures using Python.

```python
import torch
import torch.nn as nn
import math

class HyperbolicTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(HyperbolicTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.to_hyperbolic = lambda x: self.euclidean_to_poincare(x)
        self.from_hyperbolic = lambda x: self.poincare_to_euclidean(x)

    def forward(self, src, tgt):
        src = self.to_hyperbolic(src)
        tgt = self.to_hyperbolic(tgt)
        output = self.transformer(src, tgt)
        return self.from_hyperbolic(output)

    def euclidean_to_poincare(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x / (1 + torch.sqrt(1 + norm**2))

    def poincare_to_euclidean(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x * 2 / (1 - norm**2)
```

Slide 2: Understanding Hyperbolic Geometry

Hyperbolic geometry is a non-Euclidean geometry that allows for efficient representation of hierarchical structures. In hyperbolic space, the area of a circle grows exponentially with its radius, enabling the embedding of tree-like structures with low distortion.

```python
import numpy as np
import matplotlib.pyplot as plt

def hyperbolic_distance(x, y):
    return np.arccosh(1 + 2 * np.sum((x - y)**2) / ((1 - np.sum(x**2)) * (1 - np.sum(y**2))))

def plot_hyperbolic_circle(center, radius):
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    
    scale = np.tanh(radius) / np.sqrt(center[0]**2 + center[1]**2)
    x = center[0] + scale * x
    y = center[1] + scale * y
    
    plt.plot(x, y)

plt.figure(figsize=(8, 8))
plot_hyperbolic_circle([0, 0], 0.5)
plot_hyperbolic_circle([0, 0], 1)
plot_hyperbolic_circle([0, 0], 2)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Hyperbolic Circles")
plt.show()
```

Slide 3: Poincaré Ball Model

The Poincaré ball model is a popular representation of hyperbolic space. It maps the entire hyperbolic space onto the interior of a unit ball in Euclidean space. This model preserves angles but distorts distances, especially near the boundary of the ball.

```python
import numpy as np
import matplotlib.pyplot as plt

def poincare_distance(x, y):
    num = 2 * np.sum((x - y)**2)
    den = (1 - np.sum(x**2)) * (1 - np.sum(y**2))
    return np.arccosh(1 + num / den)

def plot_poincare_disk():
    circle = plt.Circle((0, 0), 1, fill=False)
    plt.gca().add_artist(circle)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal', adjustable='box')

plt.figure(figsize=(8, 8))
plot_poincare_disk()

# Plot some points in the Poincaré disk
points = np.array([[0, 0], [0.5, 0], [0, 0.7], [-0.6, 0.2]])
plt.scatter(points[:, 0], points[:, 1], c='red')

for i, p in enumerate(points):
    plt.annotate(f'P{i}', (p[0], p[1]), xytext=(5, 5), textcoords='offset points')

plt.title("Poincaré Disk Model")
plt.show()

# Calculate distances between points
for i in range(len(points)):
    for j in range(i+1, len(points)):
        dist = poincare_distance(points[i], points[j])
        print(f"Distance between P{i} and P{j}: {dist:.4f}")
```

Slide 4: Hyperbolic Attention Mechanism

The hyperbolic attention mechanism adapts the standard attention mechanism to work in hyperbolic space. It uses the hyperbolic distance to compute attention weights, allowing the model to capture hierarchical relationships more effectively.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperbolicAttention(nn.Module):
    def __init__(self, d_model):
        super(HyperbolicAttention, self).__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Compute hyperbolic distances
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        dist = torch.arccosh(1 + 2 * (1 - torch.matmul(Q_norm, K_norm.transpose(-2, -1))))

        # Compute attention weights
        attn_weights = F.softmax(-dist, dim=-1)

        # Apply attention weights
        output = torch.matmul(attn_weights, V)
        return output

# Example usage
d_model = 64
seq_len = 10
batch_size = 2

hyp_attention = HyperbolicAttention(d_model)
x = torch.randn(batch_size, seq_len, d_model)
output = hyp_attention(x, x, x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

Slide 5: Hyperbolic Feed-Forward Networks

Hyperbolic feed-forward networks adapt traditional neural network layers to operate in hyperbolic space. This involves mapping inputs to the Poincaré ball, applying transformations, and mapping the results back to Euclidean space.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperbolicFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(HyperbolicFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Map to Poincaré ball
        x_hyp = self.euclidean_to_poincare(x)

        # Apply FFN in hyperbolic space
        h = self.linear1(x_hyp)
        h = F.relu(h)
        h = self.linear2(h)

        # Map back to Euclidean space
        return self.poincare_to_euclidean(h)

    def euclidean_to_poincare(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x / (1 + torch.sqrt(1 + norm**2))

    def poincare_to_euclidean(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x * 2 / (1 - norm**2)

# Example usage
d_model = 64
d_ff = 256
batch_size = 2
seq_len = 10

hyp_ffn = HyperbolicFFN(d_model, d_ff)
x = torch.randn(batch_size, seq_len, d_model)
output = hyp_ffn(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

Slide 6: Hyperbolic Positional Encoding

Positional encoding in hyperbolic space allows the model to capture hierarchical relationships between sequence positions. This is particularly useful for tasks involving nested structures or long-range dependencies.

```python
import torch
import math

def hyperbolic_positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # Map to Poincaré ball
    norm = torch.norm(pe, dim=-1, keepdim=True)
    pe_hyp = pe / (1 + torch.sqrt(1 + norm**2))
    
    return pe_hyp

# Example usage
seq_len = 100
d_model = 64

pe = hyperbolic_positional_encoding(seq_len, d_model)

print(f"Hyperbolic Positional Encoding shape: {pe.shape}")
print(f"Norm of first position: {torch.norm(pe[0])}")
print(f"Norm of last position: {torch.norm(pe[-1])}")

# Visualize the first few dimensions
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(pe[:, :5])
plt.title("First 5 dimensions of Hyperbolic Positional Encoding")
plt.xlabel("Sequence Position")
plt.ylabel("Encoding Value")
plt.legend([f"Dim {i}" for i in range(5)])
plt.show()
```

Slide 7: Hyperbolic Layer Normalization

Layer normalization is an important component in transformer architectures. In hyperbolic space, we need to adapt this operation to maintain the properties of the Poincaré ball model.

```python
import torch
import torch.nn as nn

class HyperbolicLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(HyperbolicLayerNorm, self).__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # Map to tangent space at zero
        x_tang = self.poincare_to_lorentz(x)

        # Perform normalization in tangent space
        mean = x_tang.mean(-1, keepdim=True)
        std = x_tang.std(-1, keepdim=True)
        x_norm = (x_tang - mean) / (std + self.eps)

        # Scale and shift
        x_norm = self.a * x_norm + self.b

        # Map back to Poincaré ball
        return self.lorentz_to_poincare(x_norm)

    def poincare_to_lorentz(self, x):
        x_sq_norm = torch.sum(x**2, dim=-1, keepdim=True)
        return 2 * x / (1 - x_sq_norm)

    def lorentz_to_poincare(self, x):
        return x / (1 + torch.sqrt(1 + torch.sum(x**2, dim=-1, keepdim=True)))

# Example usage
d_model = 64
batch_size = 2
seq_len = 10

hyp_layer_norm = HyperbolicLayerNorm(d_model)
x = torch.randn(batch_size, seq_len, d_model)
x_norm = hyp_layer_norm(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {x_norm.shape}")
print(f"Input norm: {torch.norm(x[0, 0])}")
print(f"Output norm: {torch.norm(x_norm[0, 0])}")
```

Slide 8: Hyperbolic Multi-Head Attention

Multi-head attention is a key component of transformer architectures. In hyperbolic space, we adapt this mechanism to operate on the Poincaré ball, allowing for more effective modeling of hierarchical relationships.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperbolicMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(HyperbolicMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Compute hyperbolic attention
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        dist = torch.arccosh(1 + 2 * (1 - torch.matmul(Q_norm, K_norm.transpose(-2, -1))))
        attn_weights = F.softmax(-dist, dim=-1)

        # Apply attention weights
        output = torch.matmul(attn_weights, V)

        # Concatenate heads and apply final linear transformation
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

# Example usage
d_model = 64
num_heads = 8
batch_size = 2
seq_len = 10

hyp_mha = HyperbolicMultiHeadAttention(d_model, num_heads)
x = torch.randn(batch_size, seq_len, d_model)
output = hyp_mha(x, x, x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

Slide 9: Hyperbolic Transformer Encoder Layer

The hyperbolic transformer encoder layer combines hyperbolic multi-head attention with hyperbolic feed-forward networks. This adaptation allows the encoder to process information in hyperbolic space, preserving hierarchical relationships in the data.

```python
import torch
import torch.nn as nn

class HyperbolicEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(HyperbolicEncoderLayer, self).__init__()
        self.self_attn = HyperbolicMultiHeadAttention(d_model, num_heads)
        self.ffn = HyperbolicFFN(d_model, d_ff)
        self.norm1 = HyperbolicLayerNorm(d_model)
        self.norm2 = HyperbolicLayerNorm(d_model)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

# Example usage
d_model = 64
num_heads = 8
d_ff = 256
batch_size = 2
seq_len = 10

encoder_layer = HyperbolicEncoderLayer(d_model, num_heads, d_ff)
x = torch.randn(batch_size, seq_len, d_model)
output = encoder_layer(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

Slide 10: Hyperbolic Transformer Decoder Layer

The hyperbolic transformer decoder layer extends the encoder layer by adding a second attention mechanism for processing the encoder output. This allows the decoder to generate output sequences while maintaining the benefits of hyperbolic geometry.

```python
import torch
import torch.nn as nn

class HyperbolicDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(HyperbolicDecoderLayer, self).__init__()
        self.self_attn = HyperbolicMultiHeadAttention(d_model, num_heads)
        self.cross_attn = HyperbolicMultiHeadAttention(d_model, num_heads)
        self.ffn = HyperbolicFFN(d_model, d_ff)
        self.norm1 = HyperbolicLayerNorm(d_model)
        self.norm2 = HyperbolicLayerNorm(d_model)
        self.norm3 = HyperbolicLayerNorm(d_model)

    def forward(self, x, encoder_output):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output)
        x = self.norm2(x + cross_attn_output)
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        return x

# Example usage
d_model = 64
num_heads = 8
d_ff = 256
batch_size = 2
seq_len = 10

decoder_layer = HyperbolicDecoderLayer(d_model, num_heads, d_ff)
x = torch.randn(batch_size, seq_len, d_model)
encoder_output = torch.randn(batch_size, seq_len, d_model)
output = decoder_layer(x, encoder_output)

print(f"Input shape: {x.shape}")
print(f"Encoder output shape: {encoder_output.shape}")
print(f"Output shape: {output.shape}")
```

Slide 11: Training Hyperbolic Transformers

Training hyperbolic transformers requires careful consideration of the geometry. We need to use Riemannian optimization techniques and adapt the loss function to work in hyperbolic space.

```python
import torch
import torch.optim as optim

class HyperbolicAdam(optim.Adam):
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Project gradients to tangent space
                p_tang = self.poincare_to_lorentz(p.data)
                grad = p.grad.data
                grad_tang = grad - torch.sum(p_tang * grad, dim=-1, keepdim=True) * p_tang

                # Apply Adam update in tangent space
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad_tang, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_tang, grad_tang, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr'] * math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Project back to Poincaré ball
                p.data = self.lorentz_to_poincare(p.data)

        return loss

    def poincare_to_lorentz(self, x):
        x_sq_norm = torch.sum(x**2, dim=-1, keepdim=True)
        return 2 * x / (1 - x_sq_norm)

    def lorentz_to_poincare(self, x):
        return x / (1 + torch.sqrt(1 + torch.sum(x**2, dim=-1, keepdim=True)))

# Example usage
model = HyperbolicTransformer(d_model=64, nhead=8, num_layers=6)
optimizer = HyperbolicAdam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = hyperbolic_loss(output, target)
        loss.backward()
        optimizer.step()
```

Slide 12: Real-life Example: Hierarchical Text Classification

Hyperbolic transformers can be particularly effective for tasks with inherent hierarchical structure, such as hierarchical text classification. Consider classifying academic papers into a tree-like structure of research fields and subfields.

```python
import torch
import torch.nn as nn

class HyperbolicTextClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes):
        super(HyperbolicTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = HyperbolicTransformer(d_model, num_heads, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)  # Self-attention only
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

# Example usage
vocab_size = 10000
d_model = 128
num_heads = 8
num_layers = 6
num_classes = 50  # Number of academic fields/subfields

model = HyperbolicTextClassifier(vocab_size, d_model, num_heads, num_layers, num_classes)

# Sample input (batch_size=2, seq_len=100)
input_ids = torch.randint(0, vocab_size, (2, 100))
output = model(input_ids)

print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {output.shape}")
```

Slide 13: Real-life Example: Tree-structured Data Processing

Hyperbolic transformers can efficiently process tree-structured data, such as parsing XML documents or processing hierarchical data formats. This example demonstrates how to use a hyperbolic transformer for XML parsing.

```python
import torch
import torch.nn as nn
import xml.etree.ElementTree as ET

class XMLProcessor(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super(XMLProcessor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = HyperbolicTransformer(d_model, num_heads, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        return self.fc(x)

def xml_to_sequence(xml_string, token_to_id):
    root = ET.fromstring(xml_string)
    sequence = []

    def traverse(element, depth):
        sequence.append(token_to_id[f"<{element.tag}>"])
        for child in element:
            traverse(child, depth + 1)
        sequence.append(token_to_id[f"</{element.tag}>"])

    traverse(root, 0)
    return torch.tensor(sequence).unsqueeze(0)  # Add batch dimension

# Example usage
vocab_size = 1000  # Simplified vocabulary size
d_model = 128
num_heads = 8
num_layers = 6

model = XMLProcessor(vocab_size, d_model, num_heads, num_layers)

# Sample XML and token mapping
xml_string = "<root><child1>Text</child1><child2><grandchild>More text</grandchild></child2></root>"
token_to_id = {"<root>": 0, "</root>": 1, "<child1>": 2, "</child1>": 3, "<child2>": 4, "</child2>": 5, "<grandchild>": 6, "</grandchild>": 7}

input_sequence = xml_to_sequence(xml_string, token_to_id)
output = model(input_sequence)

print(f"Input shape: {input_sequence.shape}")
print(f"Output shape: {output.shape}")
```

Slide 14: Additional Resources

For more information on hyperbolic geometry in machine learning and transformers, consider the following resources:

1. "Hyperbolic Neural Networks" by Octavian-Eugen Ganea, Gary Bécigneul, and Thomas Hofmann (2018) ArXiv: [https://arxiv.org/abs/1805.09112](https://arxiv.org/abs/1805.09112)
2. "Poincaré Embeddings for Learning Hierarchical Representations" by Maximilian Nickel and Douwe Kiela (2017) ArXiv: [https://arxiv.org/abs/1705.08039](https://arxiv.org/abs/1705.08039)
3. "Hyperbolic Attention Networks" by Caglar Gulcehre, Misha Denil, Mateusz Malinowski, Ali Razavi, Razvan Pascanu, Karl Moritz Hermann, Peter Battaglia, Victor Bapst, David Raposo, Adam Santoro, and Nando de Freitas (2019) ArXiv: [https://arxiv.org/abs/1805.09786](https://arxiv.org/abs/1805.09786)

These papers provide theoretical foundations and practical implementations of hyperbolic neural networks and their applications in various domains.

