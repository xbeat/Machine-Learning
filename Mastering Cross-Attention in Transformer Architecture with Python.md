## Mastering Cross-Attention in Transformer Architecture with Python
Slide 1: Introduction to Cross-Attention in Transformer Architecture

Cross-attention is a crucial component of the Transformer architecture, enabling models to focus on relevant information from different input sequences. This mechanism allows the model to align and relate different parts of the input, making it particularly useful for tasks like machine translation, text summarization, and question answering.

```python
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads)
    
    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output

# Example usage
d_model, num_heads = 512, 8
cross_attn = CrossAttention(d_model, num_heads)
query = torch.randn(10, 32, d_model)  # (seq_len, batch_size, d_model)
key = value = torch.randn(20, 32, d_model)
output = cross_attn(query, key, value)
print(output.shape)  # torch.Size([10, 32, 512])
```

Slide 2: Understanding Query, Key, and Value in Cross-Attention

In cross-attention, the query comes from one sequence, while the keys and values come from another. This allows the model to attend to different parts of the input based on their relevance to the current position in the output sequence.

```python
import torch
import torch.nn as nn

class QueryKeyValue(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
    
    def forward(self, x, context):
        q = self.query(x)
        k = self.key(context)
        v = self.value(context)
        return q, k, v

# Example usage
d_model = 512
qkv = QueryKeyValue(d_model)
x = torch.randn(10, 32, d_model)  # (seq_len, batch_size, d_model)
context = torch.randn(20, 32, d_model)
q, k, v = qkv(x, context)
print(q.shape, k.shape, v.shape)
# Output: torch.Size([10, 32, 512]) torch.Size([20, 32, 512]) torch.Size([20, 32, 512])
```

Slide 3: Implementing the Scaled Dot-Product Attention

The core of the attention mechanism is the scaled dot-product attention. It computes the compatibility between queries and keys, applies softmax to obtain attention weights, and then aggregates the values based on these weights.

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights

# Example usage
query = torch.randn(32, 10, 64)  # (batch_size, seq_len, d_k)
key = value = torch.randn(32, 20, 64)
output, weights = scaled_dot_product_attention(query, key, value)
print(output.shape, weights.shape)
# Output: torch.Size([32, 10, 64]) torch.Size([32, 10, 20])
```

Slide 4: Multi-Head Attention: Enhancing Cross-Attention

Multi-head attention allows the model to attend to different representation subspaces, enabling it to capture various aspects of the input simultaneously. This is achieved by applying multiple attention operations in parallel and concatenating the results.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        query = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(output)

# Example usage
d_model, num_heads = 512, 8
mha = MultiHeadAttention(d_model, num_heads)
query = torch.randn(32, 10, d_model)
key = value = torch.randn(32, 20, d_model)
output = mha(query, key, value)
print(output.shape)  # torch.Size([32, 10, 512])
```

Slide 5: Position-wise Feed-Forward Networks

After the attention layer, Transformers employ a position-wise feed-forward network. This network applies the same fully connected layer to each position separately and identically, allowing the model to introduce non-linearity and capture complex patterns.

```python
import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Example usage
d_model, d_ff = 512, 2048
ff = PositionWiseFeedForward(d_model, d_ff)
x = torch.randn(32, 10, d_model)  # (batch_size, seq_len, d_model)
output = ff(x)
print(output.shape)  # torch.Size([32, 10, 512])
```

Slide 6: Layer Normalization in Transformer Architecture

Layer normalization is crucial for stabilizing the learning process in deep networks. In Transformers, it's typically applied after the attention and feed-forward layers, helping to reduce training time and improve generalization.

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# Example usage
d_model = 512
ln = LayerNorm(d_model)
x = torch.randn(32, 10, d_model)  # (batch_size, seq_len, d_model)
output = ln(x)
print(output.shape)  # torch.Size([32, 10, 512])
print(output.mean().item(), output.std().item())  # Close to 0 and 1 respectively
```

Slide 7: Residual Connections in Transformer Layers

Residual connections, or skip connections, are used in Transformers to facilitate gradient flow through the network. They help mitigate the vanishing gradient problem in deep networks by providing a direct path for gradients to flow backwards.

```python
import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x

# Example usage
d_model, num_heads, d_ff = 512, 8, 2048
layer = TransformerLayer(d_model, num_heads, d_ff)
x = torch.randn(32, 10, d_model)  # (batch_size, seq_len, d_model)
output = layer(x)
print(output.shape)  # torch.Size([32, 10, 512])
```

Slide 8: Positional Encoding in Transformers

Since Transformers don't have an inherent notion of sequence order, positional encodings are added to the input embeddings. These encodings provide information about the relative or absolute position of tokens in the sequence.

```python
import torch
import math

def positional_encoding(max_seq_len, d_model):
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Example usage
max_seq_len, d_model = 100, 512
pe = positional_encoding(max_seq_len, d_model)
print(pe.shape)  # torch.Size([100, 512])

# Visualize the positional encoding
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.imshow(pe.detach().numpy(), cmap='hot', aspect='auto')
plt.colorbar()
plt.title("Positional Encoding")
plt.xlabel("Embedding Dimension")
plt.ylabel("Sequence Position")
plt.show()
```

Slide 9: Encoder-Decoder Architecture in Transformers

The Transformer architecture consists of an encoder and a decoder. The encoder processes the input sequence, while the decoder generates the output sequence. Cross-attention allows the decoder to attend to relevant parts of the encoded input.

```python
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_len, vocab_size):
        super().__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = positional_encoding(max_seq_len, d_model)
        
        self.encoder_layers = nn.ModuleList([TransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([TransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt):
        src_embedded = self.encoder_embedding(src) + self.positional_encoding[:src.size(1), :].to(src.device)
        tgt_embedded = self.decoder_embedding(tgt) + self.positional_encoding[:tgt.size(1), :].to(tgt.device)
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)
        
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output)
        
        output = self.fc(dec_output)
        return output

# Example usage
d_model, num_heads, num_layers, d_ff, max_seq_len, vocab_size = 512, 8, 6, 2048, 100, 10000
transformer = Transformer(d_model, num_heads, num_layers, d_ff, max_seq_len, vocab_size)
src = torch.randint(0, vocab_size, (32, 20))  # (batch_size, src_seq_len)
tgt = torch.randint(0, vocab_size, (32, 15))  # (batch_size, tgt_seq_len)
output = transformer(src, tgt)
print(output.shape)  # torch.Size([32, 15, 10000])
```

Slide 10: Training a Transformer Model

Training a Transformer involves defining a loss function, typically cross-entropy for sequence generation tasks, and using an optimizer like Adam. The model is trained on paired sequences, such as source and target sentences in machine translation.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assuming we have defined our Transformer model as 'model'
model = Transformer(d_model, num_heads, num_layers, d_ff, max_seq_len, vocab_size)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

def train_step(src, tgt):
    model.train()
    optimizer.zero_grad()
    
    output = model(src, tgt[:, :-1])  # Teacher forcing: use correct tokens as input
    loss = criterion(output.contiguous().view(-1, vocab_size), tgt[:, 1:].contiguous().view(-1))
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Example training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for batch in data_loader:  # Assume we have a data loader
        src, tgt = batch
        loss = train_step(src, tgt)
        total_loss += loss
    
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    # Perform evaluation on a validation set
    # Calculate metrics like BLEU score for translation tasks
```

Slide 11: Implementing Beam Search for Inference

Beam search is a popular decoding strategy for sequence generation tasks. It maintains multiple hypotheses at each step, allowing the model to explore different possibilities and potentially find better overall sequences.

```python
import torch

def beam_search(model, src, max_len, beam_size, device):
    model.eval()
    src = src.to(device)
    
    # Encode the source sequence
    encoder_output = model.encode(src)
    
    # Initialize the beam
    beams = [([model.sos_token], 0)]
    completed_beams = []
    
    for _ in range(max_len):
        candidates = []
        for sequence, score in beams:
            if sequence[-1] == model.eos_token:
                completed_beams.append((sequence, score))
                continue
            
            # Generate next token probabilities
            tgt = torch.tensor(sequence).unsqueeze(0).to(device)
            output = model.decode(tgt, encoder_output)
            probs = torch.nn.functional.log_softmax(output[0, -1], dim=-1)
            
            # Get top k candidates
            top_probs, top_indices = probs.topk(beam_size)
            for prob, idx in zip(top_probs, top_indices):
                candidates.append((sequence + [idx.item()], score + prob.item()))
        
        # Select top beam_size candidates
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        
        if len(completed_beams) == beam_size:
            break
    
    completed_beams.extend(beams)
    return sorted(completed_beams, key=lambda x: x[1], reverse=True)[0][0]

# Usage example
src_sentence = torch.tensor([[1, 2, 3, 4, 5]])  # Example source sentence
result = beam_search(model, src_sentence, max_len=50, beam_size=5, device=torch.device('cuda'))
print("Generated sequence:", result)
```

Slide 12: Attention Visualization

Visualizing attention weights can provide insights into how the model attends to different parts of the input. This is particularly useful for tasks like machine translation, where we can see which source words the model focuses on when generating each target word.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, src_tokens, tgt_tokens):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=src_tokens, yticklabels=tgt_tokens, cmap='YlGnBu')
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    plt.title('Attention Weights Visualization')
    plt.show()

# Example usage (assuming we have attention weights from our model)
src_tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
tgt_tokens = ['Le', 'chat', 'était', 'assis', 'sur', 'le', 'tapis']
attention_weights = torch.rand(len(tgt_tokens), len(src_tokens))  # Example random weights

visualize_attention(attention_weights.numpy(), src_tokens, tgt_tokens)
```

Slide 13: Fine-tuning Pre-trained Transformers

Fine-tuning pre-trained Transformer models like BERT or GPT on specific tasks can lead to impressive results with relatively little training data. This process involves adapting the pre-trained model to a new task by training it on task-specific data.

```python
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from torch.utils.data import DataLoader, TensorDataset

# Load pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare your dataset
texts = ["Example sentence 1", "Example sentence 2", ...]
labels = [0, 1, ...]  # Binary classification labels

# Tokenize and encode the dataset
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
dataset = TensorDataset(torch.tensor(encodings['input_ids']),
                        torch.tensor(encodings['attention_mask']),
                        torch.tensor(labels))
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Fine-tuning loop
optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
```

Slide 14: Real-life Example: Machine Translation

Machine translation is a common application of Transformer models. Here's a simple example of how to use a pre-trained machine translation model for English to French translation.

```python
from transformers import MarianMTModel, MarianTokenizer

# Load pre-trained model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-fr'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

def translate(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    # Generate translation
    translated = model.generate(**inputs)
    
    # Decode the output
    output = tokenizer.decode(translated[0], skip_special_tokens=True)
    return output

# Example usage
english_text = "The quick brown fox jumps over the lazy dog."
french_translation = translate(english_text)
print(f"English: {english_text}")
print(f"French: {french_translation}")
```

Slide 15: Real-life Example: Text Summarization

Text summarization is another practical application of Transformer models. Here's an example using a pre-trained summarization model.

```python
from transformers import pipeline

# Load pre-trained summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=130, min_length=30):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Example usage
long_text = """
Climate change is one of the most pressing issues facing our planet today. It refers to long-term shifts in temperatures and weather patterns, mainly caused by human activities, especially the burning of fossil fuels. These activities release greenhouse gases into the atmosphere, trapping heat and causing the Earth's average temperature to rise. The effects of climate change are far-reaching and include more frequent and severe weather events, rising sea levels, and disruptions to ecosystems. To address this global challenge, countries and organizations worldwide are working on reducing greenhouse gas emissions, developing renewable energy sources, and implementing adaptive strategies to mitigate the impacts of climate change.
"""

summary = summarize_text(long_text)
print("Original text length:", len(long_text))
print("Summary length:", len(summary))
print("Summary:", summary)
```

Slide 16: Additional Resources

For those interested in diving deeper into Transformer architecture and its applications, here are some valuable resources:

1. "Attention Is All You Need" paper (Vaswani et al., 2017): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018): [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Language Models are Few-Shot Learners" (Brown et al., 2020), introducing GPT-3: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
4. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al., 2019), introducing T5: [https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)

These papers provide in-depth explanations of key concepts and advancements in Transformer-based models.

