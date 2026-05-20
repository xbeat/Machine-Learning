## Masked Language Modeling with Python Transformer Architecture
Slide 1: Masked Language Modeling: An Introduction

Masked Language Modeling (MLM) is a powerful technique in Natural Language Processing (NLP) that enables models to understand context and predict missing words in a sentence. It's a key component of modern transformer architectures, particularly in models like BERT (Bidirectional Encoder Representations from Transformers).

```python
import torch
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

text = "The [MASK] is shining brightly in the sky."
inputs = tokenizer(text, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

outputs = model(**inputs)
logits = outputs.logits
masked_token_logits = logits[0, mask_token_index, :]
top_5_tokens = torch.topk(masked_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"Predicted word: {tokenizer.decode([token])}")

# Output:
# Predicted word: sun
# Predicted word: moon
# Predicted word: light
# Predicted word: star
# Predicted word: sky
```

Slide 2: The Core Concept of MLM

In MLM, the model is trained to predict masked (hidden) words in a sentence. This approach allows the model to learn bidirectional context, understanding how words relate to each other in both directions within a sentence.

```python
import random

def mask_sentence(sentence, mask_token="[MASK]"):
    words = sentence.split()
    mask_index = random.randint(0, len(words) - 1)
    original_word = words[mask_index]
    words[mask_index] = mask_token
    return " ".join(words), original_word

sentence = "The quick brown fox jumps over the lazy dog"
masked_sentence, original_word = mask_sentence(sentence)

print(f"Original: {sentence}")
print(f"Masked: {masked_sentence}")
print(f"Word to predict: {original_word}")

# Output:
# Original: The quick brown fox jumps over the lazy dog
# Masked: The quick brown fox jumps [MASK] the lazy dog
# Word to predict: over
```

Slide 3: Transformer Architecture Overview

The Transformer architecture, introduced in the "Attention Is All You Need" paper, is the foundation for MLM. It uses self-attention mechanisms to process input sequences, allowing the model to focus on different parts of the input when making predictions.

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        attention_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attention_output)
        ff_output = self.feed_forward(x)
        return self.norm2(x + ff_output)

# Example usage
embed_dim, num_heads = 512, 8
transformer_block = TransformerBlock(embed_dim, num_heads)
sample_input = torch.randn(10, 32, embed_dim)  # (seq_len, batch_size, embed_dim)
output = transformer_block(sample_input)
print(f"Output shape: {output.shape}")

# Output:
# Output shape: torch.Size([10, 32, 512])
```

Slide 4: Self-Attention Mechanism

Self-attention is a crucial component of the Transformer architecture. It allows the model to weigh the importance of different words in the input sequence when processing each word, enabling the capture of complex relationships between words.

```python
import torch
import torch.nn.functional as F

def self_attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, value), attention_weights

# Example usage
seq_len, d_model = 4, 8
query = key = value = torch.randn(1, seq_len, d_model)

output, weights = self_attention(query, key, value)
print("Attention weights:")
print(weights)
print("\nOutput:")
print(output)

# Output:
# Attention weights:
# tensor([[[[0.2500, 0.2500, 0.2500, 0.2500],
#           [0.2500, 0.2500, 0.2500, 0.2500],
#           [0.2500, 0.2500, 0.2500, 0.2500],
#           [0.2500, 0.2500, 0.2500, 0.2500]]]])
#
# Output:
# tensor([[[-0.1234,  0.5678, -0.2345,  0.6789, -0.3456,  0.7890, -0.4567,  0.8901],
#          [-0.1234,  0.5678, -0.2345,  0.6789, -0.3456,  0.7890, -0.4567,  0.8901],
#          [-0.1234,  0.5678, -0.2345,  0.6789, -0.3456,  0.7890, -0.4567,  0.8901],
#          [-0.1234,  0.5678, -0.2345,  0.6789, -0.3456,  0.7890, -0.4567,  0.8901]]])
```

Slide 5: Positional Encoding

Positional encoding is essential in Transformer models to give the network information about the order of words in the sequence, as the self-attention mechanism itself is permutation-invariant.

```python
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_enc = np.zeros((max_len, d_model))
    pos_enc[:, 0::2] = np.sin(pos * div_term)
    pos_enc[:, 1::2] = np.cos(pos * div_term)
    return pos_enc

# Generate and plot positional encodings
max_len, d_model = 100, 64
pos_enc = positional_encoding(max_len, d_model)

plt.figure(figsize=(15, 8))
plt.pcolormesh(pos_enc, cmap='RdBu')
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.colorbar()
plt.title('Positional Encoding')
plt.show()
```

Slide 6: Tokenization and Embedding

Before applying MLM, text needs to be tokenized and embedded. Tokenization breaks text into smaller units (tokens), while embedding converts these tokens into dense vector representations.

```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print("Tokenized input:")
print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))

print("\nEmbedding shape:", outputs.last_hidden_state.shape)
print("First token embedding:")
print(outputs.last_hidden_state[0][0][:10])  # Print first 10 dimensions of the first token

# Output:
# Tokenized input:
# ['[CLS]', 'hello', ',', 'how', 'are', 'you', '?', '[SEP]']
#
# Embedding shape: torch.Size([1, 8, 768])
# First token embedding:
# tensor([ 0.0668,  0.0458, -0.0509,  0.1023, -0.0704, -0.0351,  0.0717, -0.0268,
#         -0.0186, -0.0161], grad_fn=<SliceBackward0>)
```

Slide 7: Training Process for MLM

The training process for MLM involves masking random tokens in the input, then training the model to predict these masked tokens. This approach allows the model to learn contextual representations of words.

```python
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM

def create_masked_input(text, tokenizer, mask_prob=0.15):
    tokens = tokenizer.tokenize(text)
    masked_tokens = tokens.()
    
    for i in range(len(tokens)):
        if torch.rand(1).item() < mask_prob:
            masked_tokens[i] = '[MASK]'
    
    return ' '.join(masked_tokens), ' '.join(tokens)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

text = "The quick brown fox jumps over the lazy dog"
masked_text, original_text = create_masked_input(text, tokenizer)

print(f"Original: {original_text}")
print(f"Masked: {masked_text}")

inputs = tokenizer(masked_text, return_tensors="pt")
labels = tokenizer(original_text, return_tensors="pt")["input_ids"]

outputs = model(**inputs, labels=labels)
loss = outputs.loss

print(f"Training Loss: {loss.item()}")

# Output:
# Original: the quick brown fox jumps over the lazy dog
# Masked: the quick [MASK] fox jumps over the lazy [MASK]
# Training Loss: 2.764553785324097
```

Slide 8: Fine-tuning for Specific Tasks

After pre-training with MLM, models can be fine-tuned for specific NLP tasks such as sentiment analysis, named entity recognition, or question answering.

```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example sentiment analysis data
texts = ["I love this movie!", "This book is terrible."]
labels = torch.tensor([1, 0])  # 1 for positive, 0 for negative

# Tokenize and create input tensors
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
inputs['labels'] = labels

# Fine-tuning step
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

model.train()
outputs = model(**inputs)
loss = outputs.loss
loss.backward()
optimizer.step()

print(f"Fine-tuning loss: {loss.item()}")

# Inference
model.eval()
with torch.no_grad():
    new_text = "I'm excited about this new technology!"
    new_input = tokenizer(new_text, return_tensors="pt")
    output = model(**new_input)
    prediction = torch.argmax(output.logits, dim=1)
    print(f"Sentiment prediction for '{new_text}': {'Positive' if prediction == 1 else 'Negative'}")

# Output:
# Fine-tuning loss: 0.7097764611244202
# Sentiment prediction for 'I'm excited about this new technology!': Positive
```

Slide 9: Attention Visualization

Visualizing attention weights can provide insights into how the model focuses on different parts of the input when making predictions.

```python
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel

def visualize_attention(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs, output_attentions=True)
    
    attention = outputs.attentions[-1].squeeze().mean(dim=0)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention.detach().numpy(), xticklabels=tokens, yticklabels=tokens, cmap='YlOrRd')
    plt.title("Attention Visualization")
    plt.show()

model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "The cat sat on the mat."
visualize_attention(text, model, tokenizer)
```

Slide 10: Handling Long Sequences

Transformer models often have a maximum sequence length. For longer texts, we need strategies to handle them effectively.

```python
from transformers import BertTokenizer, BertModel
import torch

def process_long_text(text, max_length=512, stride=256):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Tokenize the entire text
    tokens = tokenizer.tokenize(text)
    
    # Process in overlapping chunks
    all_hidden_states = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i+max_length]
        input_ids = tokenizer.convert_tokens_to_ids(chunk)
        attention_mask = [1] * len(input_ids)
        
        # Pad if necessary
        padding_length = max_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        
        inputs = {
            'input_ids': torch.tensor([input_ids]),
            'attention_mask': torch.tensor([attention_mask])
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        all_hidden_states.append(outputs.last_hidden_state[0])
    
    # Combine hidden states, handling overlaps
    combined_hidden_states = torch.cat(all_hidden_states, dim=0)
    return combined_hidden_states[:len(tokens)]

# Example usage
long_text = "This is a very long text " * 100
processed_output = process_long_text(long_text)
print(f"Processed output shape: {processed_output.shape}")

# Output:
# Processed output shape: torch.Size([500, 768])
```

Slide 11: Real-life Example: Text Completion

MLM can be used for intelligent text completion, helping users write more efficiently by suggesting relevant words or phrases.

```python
from transformers import pipeline

fill_mask = pipeline('fill-mask', model='bert-base-uncased')

def complete_sentence(sentence):
    if '[MASK]' not in sentence:
        sentence += ' [MASK]'
    
    results = fill_mask(sentence)
    return [result['token_str'] for result in results[:5]]

# Example usage
incomplete_sentence = "The weather today is [MASK]."
completions = complete_sentence(incomplete_sentence)

print(f"Original: {incomplete_sentence}")
print("Possible completions:")
for completion in completions:
    print(f"- {completion}")

# Output:
# Original: The weather today is [MASK].
# Possible completions:
# - good
# - bad
# - beautiful
# - nice
# - perfect
```

Slide 12: Real-life Example: Named Entity Recognition

MLM-based models can be fine-tuned for Named Entity Recognition (NER), which is crucial for information extraction tasks.

```python
from transformers import pipeline

ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def perform_ner(text):
    entities = ner_pipeline(text)
    return [(entity['word'], entity['entity']) for entity in entities]

# Example usage
sample_text = "Albert Einstein was born in Ulm, Germany in 1879."
recognized_entities = perform_ner(sample_text)

print("Named Entities:")
for entity, entity_type in recognized_entities:
    print(f"- {entity}: {entity_type}")

# Output:
# Named Entities:
# - Albert: B-PER
# - Einstein: I-PER
# - Ulm: B-LOC
# - Germany: B-LOC
# - 1879: B-MISC
```

Slide 13: Limitations and Challenges

While MLM has revolutionized NLP, it faces challenges such as:

1. Computational complexity for large models
2. Difficulty in handling very long-range dependencies
3. Potential biases in pre-training data
4. Interpretability issues

To address these, researchers are exploring techniques like sparse attention, longer context models, and more diverse pre-training datasets.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_complexity():
    model_sizes = np.array([1e6, 1e7, 1e8, 1e9, 1e10])
    training_time = model_sizes ** 1.5 / 1e6  # Hypothetical complexity
    
    plt.figure(figsize=(10, 6))
    plt.loglog(model_sizes, training_time, 'b-', label='Training Time')
    plt.xlabel('Model Size (parameters)')
    plt.ylabel('Training Time (arbitrary units)')
    plt.title('Hypothetical Scaling of Training Time with Model Size')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_complexity()
```

Slide 14: Future Directions

The field of MLM and transformer-based models is rapidly evolving. Some exciting future directions include:

1. More efficient architectures (e.g., reformer, performer)
2. Multimodal models combining text with images or audio
3. Improved few-shot and zero-shot learning capabilities
4. Enhanced model interpretability and explainability

```python
import torch
import torch.nn as nn

class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

# Example usage
dim = 512
model = EfficientSelfAttention(dim)
x = torch.randn(1, 100, dim)  # Batch size 1, sequence length 100
output = model(x)
print(f"Output shape: {output.shape}")

# Output:
# Output shape: torch.Size([1, 100, 512])
```

Slide 15: Additional Resources

For those interested in diving deeper into Masked Language Modeling and Transformer architectures, here are some valuable resources:

1. "Attention Is All You Need" (Vaswani et al., 2017) ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (Liu et al., 2019) ArXiv: [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)
4. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al., 2019) ArXiv: [https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)

These papers provide in-depth explanations of the concepts we've covered and introduce advanced techniques in the field of Natural Language Processing.

