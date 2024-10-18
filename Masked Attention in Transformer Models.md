## Masked Attention in Transformer Models

Slide 1: Introduction to Masked Attention

Masked Attention is a key component of transformer-based models, particularly in tasks involving predicting missing words or filling in blanks in sentences. It allows models to focus on specific parts of an input sequence while ignoring others, enabling bidirectional understanding of context.

```python
import torch.nn as nn

class MaskedAttention(nn.Module):
    def __init__(self, hidden_size):
        super(MaskedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)

    def forward(self, x, mask):
        return self.attention(x, x, x, attn_mask=mask)[0]

# Example usage
hidden_size = 512
seq_length = 10
batch_size = 32

x = torch.randn(seq_length, batch_size, hidden_size)
mask = torch.zeros(seq_length, seq_length).bool()
mask[5:, :5] = True  # Mask future tokens

attention = MaskedAttention(hidden_size)
output = attention(x, mask)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

Slide 2: BERT and Masked Attention

BERT (Bidirectional Encoder Representations from Transformers) is a prominent model utilizing Masked Attention. Developed by Google, BERT's bidirectional reading capability allows it to understand relationships between all words in a sentence, leading to more accurate predictions in various natural language processing tasks.

```python
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

text = "The [MASK] jumped over the lazy dog."
inputs = tokenizer(text, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

outputs = model(**inputs)
logits = outputs.logits
mask_token_logits = logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"Predicted word: {tokenizer.decode([token])}")
```

Slide 3: Masking Process in Masked Attention

The masking process involves replacing some words in the input sequence with a special token, typically \[MASK\]. This technique allows the model to learn contextual relationships by predicting the masked words based on the surrounding context.

```python

def mask_sentence(sentence, mask_prob=0.15):
    words = sentence.split()
    masked_words = []
    
    for word in words:
        if random.random() < mask_prob:
            masked_words.append('[MASK]')
        else:
            masked_words.append(word)
    
    return ' '.join(masked_words)

original_sentence = "The quick brown fox jumps over the lazy dog"
masked_sentence = mask_sentence(original_sentence)

print(f"Original: {original_sentence}")
print(f"Masked:   {masked_sentence}")
```

Slide 4: Attention Mechanism

The attention mechanism helps the model focus on specific words when predicting a masked word. It calculates attention scores for each word in the sequence, determining how much each word should contribute to the prediction of the masked word.

```python
import torch.nn.functional as F

def attention_scores(query, key, mask=None):
    scores = torch.matmul(query, key.transpose(-2, -1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    return F.softmax(scores, dim=-1)

# Example usage
seq_len = 5
d_k = 64
query = torch.randn(1, seq_len, d_k)
key = torch.randn(1, seq_len, d_k)
mask = torch.ones(1, seq_len, seq_len)
mask[:, :, 2:] = 0  # Mask future tokens

scores = attention_scores(query, key, mask)
print("Attention scores:")
print(scores)
```

Slide 5: Calculating Attention Scores

Attention scores are calculated to determine the importance of each word in the context when predicting masked words. Higher scores indicate more attention, while lower scores suggest less attention.

```python
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# Example usage
seq_len = 4
d_k = 64
batch_size = 1

query = torch.randn(batch_size, seq_len, d_k)
key = torch.randn(batch_size, seq_len, d_k)
value = torch.randn(batch_size, seq_len, d_k)
mask = torch.ones(batch_size, seq_len, seq_len)
mask[:, :, 2:] = 0  # Mask future tokens

output, attention_weights = scaled_dot_product_attention(query, key, value, mask)

print("Attention weights:")
print(attention_weights)
print("\nOutput:")
print(output)
```

Slide 6: Making Predictions

Based on the attention scores and the context provided by the unmasked words, the model predicts the most likely words for the masked tokens. This process involves using the learned representations and attention mechanisms to generate probable candidates for the masked positions.

```python
import torch.nn as nn

class SimpleMaskedLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleMaskedLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits

# Example usage
vocab_size = 10000
embed_size = 256
hidden_size = 512
seq_length = 20
batch_size = 32

model = SimpleMaskedLanguageModel(vocab_size, embed_size, hidden_size)
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))


output = model(input_ids)
predicted_tokens = torch.argmax(output, dim=-1)

print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {output.shape}")
print(f"Predicted tokens shape: {predicted_tokens.shape}")
```

Slide 7: Learning by Iteration

The model learns through many iterations during training, gradually improving its ability to focus on relevant words and ignore irrelevant ones for accurate predictions. This iterative process involves adjusting the model's parameters based on the difference between its predictions and the actual masked words.

```python
import torch.nn as nn
import torch.optim as optim

# Simplified model for demonstration
class SimpleMLM(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SimpleMLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        return self.fc(self.embedding(x))

# Training loop
vocab_size = 1000
embed_size = 128
model = SimpleMLM(vocab_size, embed_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

num_epochs = 5
batch_size = 32
seq_length = 10

for epoch in range(num_epochs):
    for _ in range(100):  # 100 batches per epoch
        # Generate random data for demonstration
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        labels = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
```

Slide 8: Real-Life Example: Text Completion

Masked Attention enables models to perform text completion tasks, where they fill in missing words or phrases based on the surrounding context. This capability is useful in applications like autocomplete systems or writing assistants.

```python

fill_mask = pipeline("fill-mask")

text = "The [MASK] is a powerful tool for natural language processing."
results = fill_mask(text)

print("Original text:", text)
print("\nTop 5 predictions:")
for result in results:
    print(f"- {result['token_str']} (score: {result['score']:.4f})")
```

Slide 9: Real-Life Example: Sentiment Analysis

Masked Attention contributes to improved sentiment analysis by helping models understand the context and nuances of language. This enables more accurate classification of sentiment in various texts, such as product reviews or social media posts.

```python

sentiment_analyzer = pipeline("sentiment-analysis")

texts = [
    "I absolutely loved this product! It exceeded all my expectations.",
    "The service was terrible and the staff was rude.",
    "The movie was okay, but I've seen better ones.",
]

for text in texts:
    result = sentiment_analyzer(text)[0]
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}")
    print(f"Confidence: {result['score']:.4f}\n")
```

Slide 10: Masked Attention in Multi-lingual Models

Masked Attention plays a crucial role in multi-lingual models, allowing them to understand and generate text across multiple languages. This capability is particularly useful for tasks like machine translation and cross-lingual information retrieval.

```python

# Load a multi-lingual masked language model
unmasker = pipeline('fill-mask', model='xlm-roberta-base')

# Example sentences in different languages
sentences = [
    "I love to [MASK] in the park.",  # English
    "J'aime [MASK] dans le parc.",    # French
    "Ich liebe es, im Park zu [MASK].",  # German
]

for sentence in sentences:
    results = unmasker(sentence)
    print(f"Original: {sentence}")
    print("Top 3 predictions:")
    for result in results[:3]:
        print(f"- {result['token_str']} (score: {result['score']:.4f})")
    print()
```

Slide 11: Attention Visualization

Visualizing attention weights can provide insights into how the model focuses on different parts of the input when making predictions. This can be useful for interpreting the model's behavior and understanding its decision-making process.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(sentence, attention_weights):
    words = sentence.split()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(attention_weights, annot=True, cmap='YlGnBu', ax=ax)
    ax.set_xticklabels(words, rotation=90)
    ax.set_yticklabels(words, rotation=0)
    ax.set_title("Attention Weights Visualization")
    plt.tight_layout()
    plt.show()

# Example usage
sentence = "The cat sat on the mat"
words = sentence.split()
attention_weights = np.random.rand(len(words), len(words))
attention_weights /= attention_weights.sum(axis=1, keepdims=True)

visualize_attention(sentence, attention_weights)
```

Slide 12: Masked Attention in Transformer Architecture

Masked Attention is a key component of the Transformer architecture, which forms the basis for models like BERT. In Transformers, multiple attention heads work in parallel to capture different aspects of the input sequence.

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

# Example usage
d_model = 512
num_heads = 8
seq_length = 10
batch_size = 32

mha = MultiHeadAttention(d_model, num_heads)
x = torch.randn(batch_size, seq_length, d_model)
mask = torch.ones(batch_size, seq_length, seq_length)

output = mha(x, x, x, mask)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

Slide 13: Masked Attention in Sequence-to-Sequence Tasks

Masked Attention plays a crucial role in sequence-to-sequence tasks like machine translation or text summarization. It enables the model to generate output tokens while attending to relevant parts of the input sequence and previously generated tokens.

```python
import torch.nn as nn

class Seq2SeqAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        outputs = torch.zeros(batch_size, trg_len, self.output_layer.out_features).to(src.device)
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            decoder_output, (hidden, cell) = self.decoder(input.unsqueeze(1), (hidden, cell))
            attention_weights = F.softmax(self.attention(torch.cat((decoder_output, encoder_outputs), dim=2)), dim=1)
            context = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs)
            output = self.output_layer(decoder_output + context)
            outputs[:, t] = output.squeeze(1)
            
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(2).squeeze(1)
        
        return outputs

# Example usage
input_dim = 256
hidden_dim = 512
output_dim = 256
seq_len = 10
batch_size = 32

model = Seq2SeqAttention(input_dim, hidden_dim, output_dim)
src = torch.randn(batch_size, seq_len, input_dim)
trg = torch.randint(0, output_dim, (batch_size, seq_len))

output = model(src, trg)
print(f"Input shape: {src.shape}")
print(f"Output shape: {output.shape}")
```

Slide 14: Masked Attention in Transformers for Computer Vision

While originally designed for natural language processing, Masked Attention has found applications in computer vision tasks. Vision Transformers (ViT) use masked attention to process image patches, enabling the model to capture long-range dependencies in visual data.

```python
import torch.nn as nn

class VisionTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(VisionTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        x = x + self.mlp(x)
        return self.norm2(x)

# Example usage
embed_dim = 768
num_heads = 12
seq_len = 196  # For a 14x14 grid of patches
batch_size = 32

vit_block = VisionTransformerBlock(embed_dim, num_heads)
x = torch.randn(seq_len, batch_size, embed_dim)

output = vit_block(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

Slide 15: Additional Resources

For more in-depth information on Masked Attention and related topics, consider exploring the following resources:

1. "Attention Is All You Need" by Vaswani et al. (2017): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018): [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2020): [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

These papers provide foundational concepts and advanced applications of Masked Attention in various domains of machine learning and artificial intelligence.


