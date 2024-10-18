## Decoding the Transformation! A Slideshow on Input Flow Through a Decoder

Slide 1: Understanding Decoder Input Flow

The process of transforming a simple input like "The cat sat" into a full sentence through a decoder is a fascinating journey. Let's explore this step-by-step, starting with tokenization.

```python
tokens = input_sentence.split()
print(f"Tokenized input: {tokens}")
```

Slide 2: Tokenization

Tokenization breaks down the input sentence into smaller units called tokens. This process is crucial for the model to process and understand the input.

```python
    return sentence.lower().split()

input_sentence = "The cat sat"
tokens = tokenize(input_sentence)
print(f"Tokenized: {tokens}")
```

Slide 3: Embedding

Embedding transforms tokens into vector representations, giving each word a unique digital fingerprint. This allows the model to capture semantic relationships between words.

```python

# Simplified embedding function
def embed(token):
    np.random.seed(hash(token))
    return np.random.rand(5)  # 5-dimensional embedding for simplicity

embedded_tokens = [embed(token) for token in tokens]
print(f"Embedded 'cat': {embedded_tokens[1]}")
```

Slide 4: Positional Encoding

Positional encoding adds information about the order of tokens in the sequence. This helps the model understand the difference between sentences like "The cat sat" and "Sat the cat".

```python
    pos_enc = np.zeros(d_model)
    for i in range(0, d_model, 2):
        pos_enc[i] = np.sin(position / (10000 ** (i / d_model)))
        pos_enc[i + 1] = np.cos(position / (10000 ** ((i + 1) / d_model)))
    return pos_enc

positions = [positional_encoding(i) for i in range(len(tokens))]
print(f"Positional encoding for 'cat': {positions[1]}")
```

Slide 5: Masked Self-Attention

Masked self-attention allows the model to focus on relationships between tokens while preventing it from looking ahead in the sequence. This is crucial for maintaining the autoregressive property of the decoder.

```python

def masked_self_attention(query, key, value, mask):
    scores = np.dot(query, key.T) / np.sqrt(query.shape[-1])
    scores = scores + mask
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return np.dot(attention_weights, value)

# Example usage
seq_len = 3
d_model = 5
query = key = value = np.random.rand(seq_len, d_model)
mask = np.tril(np.ones((seq_len, seq_len)))
mask[mask == 0] = -np.inf

output = masked_self_attention(query, key, value, mask)
print(f"Masked self-attention output shape: {output.shape}")
```

Slide 6: Autoregressive Output Generation

The decoder generates output word by word, using the previously generated words as context for predicting the next word. This process continues until a complete sentence is formed.

```python

def simple_language_model(prefix, vocab, max_length=10):
    sentence = prefix.split()
    while len(sentence) < max_length:
        next_word = random.choice(vocab)
        sentence.append(next_word)
        if next_word == '.':
            break
    return ' '.join(sentence)

vocab = ['the', 'cat', 'sat', 'on', 'mat', 'and', 'slept', '.']
generated = simple_language_model("The cat sat", vocab)
print(f"Generated sentence: {generated}")
```

Slide 7: Real-Life Example: Text Completion

Text completion is a common application of decoders. Here's a simplified example of how it might work in practice.

```python
    words = prefix.split()
    while len(words) < max_length:
        context = ' '.join(words[-3:])  # Use last 3 words as context
        next_word = random.choice([w for w in vocab if w.startswith(context[-1])])
        words.append(next_word)
        if next_word.endswith('.'):
            break
    return ' '.join(words)

vocab = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog.']
completion = text_completion("The quick brown", vocab)
print(f"Completed text: {completion}")
```

Slide 8: Real-Life Example: Language Translation

Language translation is another important application of decoders. Here's a simplified example of English to French translation.

```python
    words = english_sentence.lower().split()
    translated_words = [translation_dict.get(word, word) for word in words]
    return ' '.join(translated_words)

translation_dict = {
    'the': 'le', 'cat': 'chat', 'sat': 's\'est assis', 'on': 'sur', 'mat': 'tapis'
}

english_sentence = "The cat sat on the mat"
french_sentence = simple_translator(english_sentence, translation_dict)
print(f"English: {english_sentence}")
print(f"French: {french_sentence}")
```

Slide 9: Handling Out-of-Vocabulary Words

In real-world applications, decoders need to handle words that are not in their vocabulary. One approach is to use subword tokenization.

```python
    if word in subwords:
        return [word]
    tokens = []
    while word:
        token = max((t for t in subwords if word.startswith(t)), key=len, default='')
        if not token:
            return [word]  # Unknown word
        tokens.append(token)
        word = word[len(token):]
    return tokens

subwords = ['un', 'known', 'word', 'sub', 'token', 'ization']
word = 'unknownword'
tokens = subword_tokenize(word, subwords)
print(f"Subword tokens for '{word}': {tokens}")
```

Slide 10: Beam Search

Beam search is a technique used to improve the quality of generated sequences by considering multiple possible outputs at each step.

```python

def beam_search(initial_state, beam_width, max_steps, next_states_fn, score_fn):
    beam = [(0, initial_state)]
    for _ in range(max_steps):
        candidates = []
        for score, state in beam:
            for next_state in next_states_fn(state):
                new_score = score + score_fn(next_state)
                candidates.append((new_score, next_state))
        beam = heapq.nlargest(beam_width, candidates)
    return beam[0][1]  # Return the best state

# Example usage
initial_state = "The"
beam_width = 3
max_steps = 5
next_states_fn = lambda s: [s + " " + w for w in ["cat", "dog", "bird"]]
score_fn = lambda s: len(s.split())

result = beam_search(initial_state, beam_width, max_steps, next_states_fn, score_fn)
print(f"Beam search result: {result}")
```

Slide 11: Attention Visualization

Visualizing attention weights can help us understand which parts of the input the model focuses on when generating each output token.

```python
import numpy as np

def visualize_attention(input_tokens, output_tokens, attention_weights):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(attention_weights, cmap='viridis')
    
    ax.set_xticks(np.arange(len(input_tokens)))
    ax.set_yticks(np.arange(len(output_tokens)))
    ax.set_xticklabels(input_tokens)
    ax.set_yticklabels(output_tokens)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(len(output_tokens)):
        for j in range(len(input_tokens)):
            ax.text(j, i, f"{attention_weights[i, j]:.2f}", ha="center", va="center", color="w")
    
    ax.set_title("Attention Weights Visualization")
    fig.tight_layout()
    plt.show()

# Example usage
input_tokens = ["The", "cat", "sat", "on", "the", "mat"]
output_tokens = ["Le", "chat", "s'est", "assis", "sur", "le", "tapis"]
attention_weights = np.random.rand(len(output_tokens), len(input_tokens))
attention_weights /= attention_weights.sum(axis=1, keepdims=True)

visualize_attention(input_tokens, output_tokens, attention_weights)
```

Slide 12: Fine-tuning and Transfer Learning

Fine-tuning allows us to adapt pre-trained models to specific tasks or domains, leveraging transfer learning to improve performance with limited data.

```python
import torch.nn as nn

class SimpleDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.fc(output)

# Pretrained model
vocab_size, embedding_dim, hidden_dim = 1000, 128, 256
pretrained_model = SimpleDecoder(vocab_size, embedding_dim, hidden_dim)

# Fine-tuning
def fine_tune(model, new_data, epochs=5):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for batch in new_data:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output.view(-1, vocab_size), batch.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Simulated new data
new_data = [torch.randint(0, vocab_size, (32, 10)) for _ in range(10)]
fine_tune(pretrained_model, new_data)
```

Slide 13: Evaluation Metrics

Evaluating the performance of a decoder is crucial. Common metrics include BLEU score for translation tasks and perplexity for language modeling.

```python
from collections import Counter

def bleu_score(reference, candidate, n=4):
    ref_ngrams = Counter(zip(*[reference[i:] for i in range(n)]))
    cand_ngrams = Counter(zip(*[candidate[i:] for i in range(n)]))
    
    clipped_counts = {ngram: min(count, ref_ngrams[ngram]) 
                      for ngram, count in cand_ngrams.items()}
    
    numerator = sum(clipped_counts.values())
    denominator = sum(cand_ngrams.values())
    
    brevity_penalty = min(1, math.exp(1 - len(reference) / len(candidate)))
    
    return brevity_penalty * (numerator / denominator if denominator > 0 else 0)

reference = "the cat sat on the mat".split()
candidate = "the cat sat on the rug".split()
score = bleu_score(reference, candidate)
print(f"BLEU score: {score:.4f}")

def perplexity(model, data):
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in data:
            output = model(batch[:, :-1])
            loss = criterion(output.view(-1, vocab_size), batch[:, 1:].view(-1))
            total_loss += loss.item() * batch.numel()
            total_tokens += batch.numel()
    
    return math.exp(total_loss / total_tokens)

# Assuming we have a trained model and evaluation data
eval_data = [torch.randint(0, vocab_size, (32, 10)) for _ in range(5)]
ppl = perplexity(pretrained_model, eval_data)
print(f"Perplexity: {ppl:.2f}")
```

Slide 14: Additional Resources

For those interested in diving deeper into the world of decoders and natural language processing, here are some valuable resources:

1. "Attention Is All You Need" by Vaswani et al. (2017): This seminal paper introduced the Transformer architecture, which forms the basis of many modern decoder models. Available at: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018): This paper presents BERT, a powerful language model that has revolutionized natural language processing. Available at: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "The Illustrated Transformer" by Jay Alammar: An excellent visual explanation of the Transformer architecture. While not an academic paper, it provides invaluable insights into how these models work.

These resources provide a solid foundation for understanding the principles and applications of decoders in modern natural language processing.


