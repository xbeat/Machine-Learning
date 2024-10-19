## Roadmap for Developing a LLM from Harry Potter Books
##interrupted Slide 14

Slide 1: Tokenization of Harry Potter Books

This slide introduces the process of tokenizing text from Harry Potter books. We'll use the NLTK library to tokenize the text into words.

```python
import csv

def tokenize_harry_potter(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        text = ' '.join([row[0] for row in reader])  # Assuming text is in the first column
    
    tokens = nltk.word_tokenize(text)
    return tokens

# Example usage
file_path = 'harry_potter_books.csv'
tokens = tokenize_harry_potter(file_path)
print(f"Number of tokens: {len(tokens)}")
print(f"First 10 tokens: {tokens[:10]}")
```

Slide 2: Token to ID Conversion

Here we create a vocabulary and convert tokens to unique integer IDs.

```python

def create_vocab_and_ids(tokens, vocab_size=10000):
    vocab = Counter(tokens).most_common(vocab_size)
    word_to_id = {word: i for i, (word, _) in enumerate(vocab)}
    
    token_ids = [word_to_id.get(token, word_to_id['<UNK>']) for token in tokens]
    return word_to_id, token_ids

word_to_id, token_ids = create_vocab_and_ids(tokens)
print(f"Vocabulary size: {len(word_to_id)}")
print(f"First 10 token IDs: {token_ids[:10]}")
```

Slide 3: Vector Embeddings

We'll use a simple random initialization for word embeddings. In practice, you'd use pre-trained embeddings or learn them during training.

```python

def create_embeddings(vocab_size, embedding_dim=100):
    return np.random.randn(vocab_size, embedding_dim)

vocab_size = len(word_to_id)
embeddings = create_embeddings(vocab_size)
print(f"Embedding shape: {embeddings.shape}")
print(f"Embedding for 'Harry': {embeddings[word_to_id['Harry']][:5]}")  # First 5 values
```

Slide 4: Positional Embeddings

We add positional information to the word embeddings using sine and cosine functions.

```python
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_enc = np.zeros((seq_len, d_model))
    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term)
    return pos_enc

seq_len, d_model = 100, 100
pos_embeddings = positional_encoding(seq_len, d_model)
print(f"Positional embedding shape: {pos_embeddings.shape}")
print(f"First 5 values of position 10: {pos_embeddings[10][:5]}")
```

Slide 5: Layer Normalization

Implementing layer normalization for better training stability.

```python
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# Example usage
layer_norm = LayerNorm(100)
input_tensor = torch.randn(32, 10, 100)  # (batch_size, seq_len, hidden_dim)
normalized = layer_norm(input_tensor)
print(f"Input mean: {input_tensor.mean():.4f}, std: {input_tensor.std():.4f}")
print(f"Output mean: {normalized.mean():.4f}, std: {normalized.std():.4f}")
```

Slide 6: Multi-Head Attention

Implementing the core of the Transformer: multi-head attention.

```python

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
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attn_output)

# Example usage
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(32, 10, 512)  # (batch_size, seq_len, d_model)
output = mha(x, x, x)
print(f"Multi-head attention output shape: {output.shape}")
```

Slide 7: Feedforward Neural Network

Implementing the feedforward neural network used in the Transformer block.

```python
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

# Example usage
ff = FeedForward(d_model=512, d_ff=2048)
x = torch.randn(32, 10, 512)  # (batch_size, seq_len, d_model)
output = ff(x)
print(f"Feedforward output shape: {output.shape}")
```

Slide 8: GeLU Activation Function

Implementing the Gaussian Error Linear Unit (GeLU) activation function.

```python

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

# Example usage
x = torch.linspace(-5, 5, 100)
y = gelu(x)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), y.numpy())
plt.title('GeLU Activation Function')
plt.xlabel('x')
plt.ylabel('GeLU(x)')
plt.grid(True)
plt.show()
```

Slide 9: Dropout Layer

Implementing dropout for regularization.

```python
    def __init__(self, p=0.1):
        super(Dropout, self).__init__()
        self.p = p
        
    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.full(x.shape, 1 - self.p))
            return x * mask / (1 - self.p)
        return x

# Example usage
dropout = Dropout(p=0.2)
x = torch.ones(5, 5)
print("Input:")
print(x)
print("\nOutput (during training):")
print(dropout(x))
```

Slide 10: Transformer Block

Combining all components to create a complete Transformer block.

```python
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)
        
    def forward(self, x):
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

# Example usage
block = TransformerBlock(d_model=512, num_heads=8, d_ff=2048)
x = torch.randn(32, 10, 512)  # (batch_size, seq_len, d_model)
output = block(x)
print(f"Transformer block output shape: {output.shape}")
```

Slide 11: GPT Model Architecture

Implementing the full GPT model architecture.

```python
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super(GPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.ln_f = LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)
        
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# Example usage
model = GPT(vocab_size=10000, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_len=1024)
x = torch.randint(0, 10000, (32, 50))  # (batch_size, seq_len)
output = model(x)
print(f"GPT model output shape: {output.shape}")
```

Slide 12: Training Loop

Implementing the training loop for the GPT model.

```python
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            inputs, targets = batch[:, :-1], batch[:, 1:]
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader):.4f}")

# Example usage (assuming data_loader is defined)
# train_gpt(model, data_loader, num_epochs=10, lr=0.001)
```

Slide 13: Reducing Overfitting

Implementing techniques to reduce overfitting in the GPT model.

```python

def train_with_regularization(model, data_loader, val_loader, num_epochs, lr, weight_decay=0.01):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in data_loader:
            inputs, targets = batch[:, :-1], batch[:, 1:]
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch[:, :-1], batch[:, 1:]
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss += loss.item()
        
        train_loss /= len(data_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)

# Example usage (assuming data_loader and val_loader are defined)
# train_with_regularization(model, data_loader, val_loader, num_epochs=10, lr=0.001)
```

Slide 14: Text Generation with GPT

This slide demonstrates how to use the trained GPT model for text generation.

```python
    model.eval()
    current_tokens = torch.tensor(start_tokens).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(current_tokens)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
            if next_token.item() == word_to_id['<EOS>']:
                break
    
    return [id_to_word[token.item()] for token in current_tokens[0]]

# Example usage
start_text = "Harry Potter"
start_tokens = [word_to_id.get(word, word_to_id['<UNK>']) for word in start_text.split()]
generated_text = generate_text(model, start_tokens)
print(" ".join(generated_text))
```

Slide 15: Real-life Example: Sentiment Analysis

Using the trained GPT model for sentiment analysis on movie reviews.

```python
    tokens = [word_to_id.get(word, word_to_id['<UNK>']) for word in text.split()]
    input_tensor = torch.tensor(tokens).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_tensor)
    
    word_probabilities = torch.softmax(logits[0, -1, :], dim=-1)
    
    positive_score = sum(word_probabilities[word_to_id[word]] for word in positive_words if word in word_to_id)
    negative_score = sum(word_probabilities[word_to_id[word]] for word in negative_words if word in word_to_id)
    
    sentiment = "Positive" if positive_score > negative_score else "Negative"
    confidence = abs(positive_score - negative_score) / (positive_score + negative_score)
    
    return sentiment, confidence

# Example usage
review = "The movie was fantastic and entertaining."
positive_words = ['good', 'great', 'fantastic', 'excellent']
negative_words = ['bad', 'awful', 'terrible', 'disappointing']

sentiment, confidence = sentiment_analysis(model, review, positive_words, negative_words)
print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}")
```

Slide 16: Additional Resources

For further exploration of Language Models and Transformers:

1. "Attention Is All You Need" by Vaswani et al. (2017) ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "Language Models are Few-Shot Learners" by Brown et al. (2020) ArXiv: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
3. "Improving Language Understanding by Generative Pre-Training" by Radford et al. (2018) OpenAI Blog: [https://openai.com/research/language-unsupervised](https://openai.com/research/language-unsupervised)
4. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

These resources provide in-depth explanations of the concepts and techniques used in modern language models.


