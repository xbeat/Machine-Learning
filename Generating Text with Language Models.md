## Generating Text with Language Models
Slide 1: Text Generation with Language Models

Language models are powerful tools for generating text, but the process is not as straightforward as it might seem. These models are trained to predict the probability of the next token in a sequence, but turning these probabilities into coherent text requires careful consideration of various generation strategies.

```python
import torch
import torch.nn as nn

class SimpleLanguageModel(nn.Module):
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
vocab_size = 10000
embedding_dim = 256
hidden_dim = 512
model = SimpleLanguageModel(vocab_size, embedding_dim, hidden_dim)

# Input sequence (batch_size=1, sequence_length=10)
input_seq = torch.randint(0, vocab_size, (1, 10))
output = model(input_seq)

print(f"Output shape: {output.shape}")  # Expected: torch.Size([1, 10, 10000])
```

Slide 2: Greedy Decoding

The simplest approach to text generation is greedy decoding. At each step, we select the token with the highest probability as predicted by the model. While straightforward, this method often leads to repetitive and less diverse output.

```python
import torch

def greedy_decode(model, start_token, max_length):
    current_token = start_token
    output = [current_token]
    
    for _ in range(max_length - 1):
        # Get model predictions
        with torch.no_grad():
            logits = model(torch.tensor([[current_token]]))
        
        # Select the token with highest probability
        current_token = torch.argmax(logits[0, -1]).item()
        output.append(current_token)
        
        # Stop if we've generated an end-of-sequence token
        if current_token == EOS_TOKEN:
            break
    
    return output

# Example usage
start_token = 5  # Assume 5 is the start token
EOS_TOKEN = 1    # Assume 1 is the end-of-sequence token
generated_sequence = greedy_decode(model, start_token, max_length=20)
print(f"Generated sequence: {generated_sequence}")
```

Slide 3: Temperature-based Sampling

To introduce more variety in the generated text, we can use temperature-based sampling. This approach adjusts the probability distribution of the model's output, allowing for more diverse token selection.

```python
import torch
import torch.nn.functional as F

def temperature_sample(logits, temperature=1.0):
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Sample from the distribution
    return torch.multinomial(probs, num_samples=1).item()

def generate_with_temperature(model, start_token, max_length, temperature):
    current_token = start_token
    output = [current_token]
    
    for _ in range(max_length - 1):
        with torch.no_grad():
            logits = model(torch.tensor([[current_token]]))
        
        current_token = temperature_sample(logits[0, -1], temperature)
        output.append(current_token)
        
        if current_token == EOS_TOKEN:
            break
    
    return output

# Example usage
start_token = 5
EOS_TOKEN = 1
temperature = 0.7
generated_sequence = generate_with_temperature(model, start_token, max_length=20, temperature=temperature)
print(f"Generated sequence with temperature {temperature}: {generated_sequence}")
```

Slide 4: Beam Search

Beam search is a more sophisticated generation technique that considers multiple possible sequences simultaneously. It maintains a set of top-k partial sequences at each step, potentially leading to higher-quality output.

```python
import torch
import heapq

def beam_search(model, start_token, beam_width, max_length):
    sequences = [(0, [start_token])]
    
    for _ in range(max_length - 1):
        all_candidates = []
        for score, seq in sequences:
            with torch.no_grad():
                logits = model(torch.tensor([seq]))
            probs = torch.log_softmax(logits[0, -1], dim=-1)
            
            for token, token_prob in enumerate(probs):
                candidate = (score - token_prob.item(), seq + [token])
                all_candidates.append(candidate)
        
        # Select top beam_width candidates
        sequences = heapq.nsmallest(beam_width, all_candidates)
        
        # Stop if all sequences have generated EOS_TOKEN
        if all(seq[-1] == EOS_TOKEN for _, seq in sequences):
            break
    
    return sequences[0][1]  # Return the sequence with the highest score

# Example usage
start_token = 5
EOS_TOKEN = 1
beam_width = 3
generated_sequence = beam_search(model, start_token, beam_width, max_length=20)
print(f"Generated sequence with beam search (width={beam_width}): {generated_sequence}")
```

Slide 5: Top-k Sampling

Top-k sampling is another strategy to balance between diversity and quality in text generation. It restricts the sampling pool to the k most likely tokens, preventing the selection of low-probability tokens that might lead to incoherent text.

```python
import torch

def top_k_sample(logits, k):
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = torch.softmax(top_k_logits, dim=-1)
    chosen_idx = torch.multinomial(probs, num_samples=1).item()
    return top_k_indices[chosen_idx].item()

def generate_with_top_k(model, start_token, max_length, k):
    current_token = start_token
    output = [current_token]
    
    for _ in range(max_length - 1):
        with torch.no_grad():
            logits = model(torch.tensor([[current_token]]))
        
        current_token = top_k_sample(logits[0, -1], k)
        output.append(current_token)
        
        if current_token == EOS_TOKEN:
            break
    
    return output

# Example usage
start_token = 5
EOS_TOKEN = 1
k = 10
generated_sequence = generate_with_top_k(model, start_token, max_length=20, k=k)
print(f"Generated sequence with top-{k} sampling: {generated_sequence}")
```

Slide 6: Nucleus (Top-p) Sampling

Nucleus sampling, also known as top-p sampling, is an adaptive sampling strategy that selects from the smallest possible set of tokens whose cumulative probability exceeds a threshold p. This approach can adapt to the uncertainty of the model's predictions.

```python
import torch

def top_p_sample(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).item()

def generate_with_top_p(model, start_token, max_length, p):
    current_token = start_token
    output = [current_token]
    
    for _ in range(max_length - 1):
        with torch.no_grad():
            logits = model(torch.tensor([[current_token]]))
        
        current_token = top_p_sample(logits[0, -1], p)
        output.append(current_token)
        
        if current_token == EOS_TOKEN:
            break
    
    return output

# Example usage
start_token = 5
EOS_TOKEN = 1
p = 0.9
generated_sequence = generate_with_top_p(model, start_token, max_length=20, p=p)
print(f"Generated sequence with nucleus sampling (p={p}): {generated_sequence}")
```

Slide 7: Contrastive Search

Contrastive search is a more recent technique that aims to balance between quality and diversity. It penalizes the selection of tokens that are too similar to previously generated ones, promoting more varied and interesting output.

```python
import torch
import torch.nn.functional as F

def contrastive_search(model, start_token, max_length, k, alpha):
    current_token = start_token
    output = [current_token]
    past_hidden_states = []
    
    for _ in range(max_length - 1):
        with torch.no_grad():
            logits, hidden_state = model(torch.tensor([[current_token]]), return_hidden=True)
        
        # Get top-k candidates
        top_k_logits, top_k_indices = torch.topk(logits[0, -1], k)
        
        # Compute similarity with past hidden states
        if past_hidden_states:
            similarities = torch.matmul(hidden_state, torch.stack(past_hidden_states).transpose(0, 1))
            max_similarity = similarities.max()
        else:
            max_similarity = 0
        
        # Compute degeneration penalty
        degeneration_penalty = alpha * max_similarity
        
        # Adjust scores
        adjusted_scores = top_k_logits - degeneration_penalty
        
        # Select token
        probs = F.softmax(adjusted_scores, dim=-1)
        current_token = top_k_indices[torch.multinomial(probs, num_samples=1)].item()
        
        output.append(current_token)
        past_hidden_states.append(hidden_state.squeeze())
        
        if current_token == EOS_TOKEN:
            break
    
    return output

# Example usage (assuming model returns hidden states)
start_token = 5
EOS_TOKEN = 1
k = 10
alpha = 0.6
generated_sequence = contrastive_search(model, start_token, max_length=20, k=k, alpha=alpha)
print(f"Generated sequence with contrastive search (k={k}, alpha={alpha}): {generated_sequence}")
```

Slide 8: Comparing Generation Strategies

Different text generation strategies can lead to vastly different outputs. Let's compare the results of various methods on a simple prompt.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simplified language model for demonstration
class ToyLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x):
        return self.fc(self.embedding(x))

# Initialize model and parameters
vocab_size = 1000
embedding_dim = 64
model = ToyLanguageModel(vocab_size, embedding_dim)

# Generate text using different strategies
start_token = 5
max_length = 10

print("Greedy decoding:", greedy_decode(model, start_token, max_length))
print("Temperature sampling (T=0.7):", generate_with_temperature(model, start_token, max_length, temperature=0.7))
print("Beam search (width=3):", beam_search(model, start_token, beam_width=3, max_length=max_length))
print("Top-k sampling (k=10):", generate_with_top_k(model, start_token, max_length, k=10))
print("Nucleus sampling (p=0.9):", generate_with_top_p(model, start_token, max_length, p=0.9))
```

Slide 9: Real-life Example: Chatbot Response Generation

Let's consider a simple chatbot that generates responses to user inputs. We'll use temperature-based sampling to add some variability to the responses.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChatbotModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)

# Initialize model (assuming it's trained)
vocab_size = 10000
embedding_dim = 256
hidden_dim = 512
model = ChatbotModel(vocab_size, embedding_dim, hidden_dim)

# Dummy tokenizer and detokenizer (replace with actual implementations)
def tokenize(text):
    return [hash(word) % vocab_size for word in text.split()]

def detokenize(tokens):
    return " ".join([str(token) for token in tokens])

def generate_response(model, input_text, max_length=20, temperature=0.7):
    input_tokens = tokenize(input_text)
    start_token = input_tokens[-1]
    generated = generate_with_temperature(model, start_token, max_length, temperature)
    return detokenize(generated)

# Example usage
user_input = "How are you today?"
response = generate_response(model, user_input)
print(f"User: {user_input}")
print(f"Chatbot: {response}")
```

Slide 10: Real-life Example: Code Completion

Code completion is a practical application of text generation in software development. Let's implement a basic code completion system using top-k sampling.

```python
import torch
import torch.nn as nn

class CodeCompletionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)

# Initialize model (assuming it's trained on a code corpus)
vocab_size, embedding_dim, hidden_dim = 5000, 128, 256
model = CodeCompletionModel(vocab_size, embedding_dim, hidden_dim)

def tokenize_code(code):
    return [hash(token) % vocab_size for token in code.split()]

def detokenize_code(tokens):
    return " ".join([str(token) for token in tokens])

def complete_code(model, code_snippet, max_length=20, k=5):
    input_tokens = tokenize_code(code_snippet)
    start_token = input_tokens[-1]
    completed = generate_with_top_k(model, start_token, max_length, k)
    return code_snippet + " " + detokenize_code(completed)

# Example usage
code_snippet = "def fibonacci(n):"
completed_code = complete_code(model, code_snippet)
print(f"Original: {code_snippet}")
print(f"Completed: {completed_code}")
```

Slide 11: Handling Long-Range Dependencies

Language models often struggle with maintaining coherence over long sequences. Techniques like attention mechanisms and transformer architectures help address this challenge by allowing the model to focus on relevant parts of the input regardless of distance.

```python
import torch
import torch.nn as nn

class SimpleAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, hidden_states):
        attention_scores = self.attention(hidden_states)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(attention_weights * hidden_states, dim=1)
        return context_vector

class AttentionLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attention = SimpleAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        hidden_states, _ = self.lstm(embedded)
        context = self.attention(hidden_states)
        return self.fc(context)

# Example usage
vocab_size, embedding_dim, hidden_dim = 10000, 256, 512
model = AttentionLanguageModel(vocab_size, embedding_dim, hidden_dim)
input_seq = torch.randint(0, vocab_size, (1, 100))
output = model(input_seq)
print(f"Output shape: {output.shape}")
```

Slide 12: Controlling Generation with Prompts

Prompts play a crucial role in guiding text generation. By carefully crafting the input prompt, we can steer the model towards generating text with specific styles, topics, or formats.

```python
def generate_with_prompt(model, prompt, max_length=50, temperature=0.7):
    prompt_tokens = tokenize(prompt)
    generated = prompt_tokens.()
    
    for _ in range(max_length):
        with torch.no_grad():
            input_tensor = torch.tensor([generated[-100:]])  # Use last 100 tokens as context
            logits = model(input_tensor)
        
        next_token = temperature_sample(logits[0, -1], temperature)
        generated.append(next_token)
        
        if next_token == EOS_TOKEN:
            break
    
    return detokenize(generated)

# Example usage
prompt = "Write a haiku about artificial intelligence:"
generated_text = generate_with_prompt(model, prompt)
print(f"Prompt: {prompt}")
print(f"Generated text: {generated_text}")
```

Slide 13: Ethical Considerations in Text Generation

As we develop and deploy text generation systems, it's crucial to consider the ethical implications. These may include biases in generated content, potential misuse for misinformation, and privacy concerns.

```python
def content_filter(generated_text):
    # Placeholder for a more sophisticated content filtering system
    sensitive_words = ["violence", "hate", "explicit"]
    return not any(word in generated_text.lower() for word in sensitive_words)

def generate_safe_text(model, prompt, max_attempts=5):
    for _ in range(max_attempts):
        generated_text = generate_with_prompt(model, prompt)
        if content_filter(generated_text):
            return generated_text
    return "I apologize, but I couldn't generate an appropriate response."

# Example usage
safe_prompt = "Write a friendly greeting:"
safe_text = generate_safe_text(model, safe_prompt)
print(f"Safe generated text: {safe_text}")
```

Slide 14: Future Directions in Text Generation

The field of text generation is rapidly evolving. Some promising areas for future research and development include:

1. Improved coherence and consistency over long texts
2. Better control over generation style and content
3. More efficient training and inference techniques
4. Enhanced multimodal generation (text + images, audio, etc.)
5. Addressing ethical concerns and biases in generated content

Researchers and practitioners continue to push the boundaries of what's possible with text generation, opening up new applications and possibilities for human-AI interaction.

Slide 15: Additional Resources

For those interested in diving deeper into text generation techniques and language models, here are some valuable resources:

1. "Attention Is All You Need" by Vaswani et al. (2017) ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Language Models are Few-Shot Learners" by Brown et al. (2020) ArXiv: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
4. "The Curious Case of Neural Text Degeneration" by Holtzman et al. (2019) ArXiv: [https://arxiv.org/abs/1904.09751](https://arxiv.org/abs/1904.09751)

These papers provide in-depth discussions of key concepts and techniques in modern text generation and language modeling.

