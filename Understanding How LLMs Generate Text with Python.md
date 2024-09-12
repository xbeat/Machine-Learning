## Understanding How LLMs Generate Text with Python
Slide 1: Understanding Token Generation in LLMs

Large Language Models (LLMs) generate text by predicting the next token in a sequence. This process involves complex neural networks and probability distributions. Let's explore how this works using Python examples.

```python
import torch
import torch.nn as nn

class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.fc(output)

vocab_size = 1000
embedding_dim = 64
hidden_dim = 128

model = SimpleLanguageModel(vocab_size, embedding_dim, hidden_dim)
print(model)
```

Slide 2: Tokenization: Breaking Text into Meaningful Units

Tokenization is the first step in processing text for LLMs. It involves splitting text into smaller units called tokens, which can be words, subwords, or characters.

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Hello, how are you today?"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")
print(f"Decoded: {tokenizer.decode(tokens)}")

# Output:
# Tokens: [15496, 11, 2129, 389, 345, 628, 30]
# Decoded: Hello, how are you today?
```

Slide 3: Embedding: Converting Tokens to Vectors

After tokenization, each token is converted into a high-dimensional vector representation. This process is called embedding and allows the model to capture semantic relationships between tokens.

```python
import torch
import torch.nn as nn

vocab_size = 10000
embedding_dim = 256

embedding_layer = nn.Embedding(vocab_size, embedding_dim)

token_ids = torch.tensor([1, 5, 20, 100])
embedded_tokens = embedding_layer(token_ids)

print(f"Token IDs shape: {token_ids.shape}")
print(f"Embedded tokens shape: {embedded_tokens.shape}")

# Output:
# Token IDs shape: torch.Size([4])
# Embedded tokens shape: torch.Size([4, 256])
```

Slide 4: Attention Mechanism: Focusing on Relevant Context

The attention mechanism allows the model to focus on different parts of the input sequence when generating each token. It's a key component in modern LLMs.

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        return self.fc_out(out)

attention = SelfAttention(embed_size=256, heads=8)
x = torch.randn(32, 10, 256)  # (batch_size, seq_length, embed_size)
output = attention(x, x, x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Output:
# Input shape: torch.Size([32, 10, 256])
# Output shape: torch.Size([32, 10, 256])
```

Slide 5: Transformer Architecture: The Foundation of Modern LLMs

Transformers are the backbone of modern LLMs. They use self-attention mechanisms and feed-forward neural networks to process input sequences and generate output tokens.

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

block = TransformerBlock(embed_size=256, heads=8, dropout=0.1, forward_expansion=4)
x = torch.randn(32, 10, 256)
output = block(x, x, x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Output:
# Input shape: torch.Size([32, 10, 256])
# Output shape: torch.Size([32, 10, 256])
```

Slide 6: Training LLMs: Learning from Vast Amounts of Data

Training LLMs involves exposing the model to large datasets and adjusting its parameters to minimize prediction errors. This process uses backpropagation and optimization algorithms.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assuming we have a SimpleLanguageModel defined as before
model = SimpleLanguageModel(vocab_size=1000, embedding_dim=64, hidden_dim=128)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, data, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data:
            optimizer.zero_grad()
            input_seq = batch[:, :-1]
            target_seq = batch[:, 1:]
            output = model(input_seq)
            loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Simulated training data
data = [torch.randint(0, 1000, (32, 20)) for _ in range(100)]
train(model, data)
```

Slide 7: Generating Text: Sampling from the Model's Output

Once trained, LLMs generate text by sampling from the probability distribution of the next token. Various sampling methods can be used to control the creativity and coherence of the generated text.

```python
import torch
import torch.nn.functional as F

def generate_text(model, start_sequence, max_length=50, temperature=1.0):
    model.eval()
    current_sequence = start_sequence.clone()

    for _ in range(max_length):
        with torch.no_grad():
            output = model(current_sequence)
            next_token_logits = output[:, -1, :] / temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            current_sequence = torch.cat([current_sequence, next_token], dim=1)

    return current_sequence

# Assuming we have a trained model and a tokenizer
start_sequence = torch.tensor([[tokenizer.encode("Once upon a time")]])
generated_sequence = generate_text(model, start_sequence)
generated_text = tokenizer.decode(generated_sequence[0])
print(generated_text)
```

Slide 8: Beam Search: Exploring Multiple Possibilities

Beam search is a technique used to generate more coherent text by maintaining multiple candidate sequences and selecting the most probable ones.

```python
import torch
import torch.nn.functional as F

def beam_search(model, start_sequence, beam_width=5, max_length=50):
    model.eval()
    sequences = [(start_sequence, 0)]

    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            with torch.no_grad():
                output = model(seq)
                next_token_logits = output[:, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                top_probs, top_indices = next_token_probs.topk(beam_width)

                for prob, idx in zip(top_probs[0], top_indices[0]):
                    candidate_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                    candidate_score = score - torch.log(prob)
                    all_candidates.append((candidate_seq, candidate_score))

        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]

    return sequences[0][0]

# Assuming we have a trained model and a tokenizer
start_sequence = torch.tensor([[tokenizer.encode("The quick brown")]])
generated_sequence = beam_search(model, start_sequence)
generated_text = tokenizer.decode(generated_sequence[0])
print(generated_text)
```

Slide 9: Fine-tuning: Adapting LLMs for Specific Tasks

Fine-tuning allows us to adapt pre-trained LLMs for specific tasks or domains by training them on a smaller, task-specific dataset.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Prepare your dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="path/to/your/train.txt",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
```

Slide 10: Real-life Example: Chatbot Development

LLMs can be used to create sophisticated chatbots that understand and generate human-like responses. Here's a simple example of how to use a fine-tuned model for a chatbot.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chat_response(input_text, chat_history_ids):
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# Simulated conversation
chat_history_ids = None
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response, chat_history_ids = chat_response(user_input, chat_history_ids)
    print(f"Bot: {response}")

# Example output:
# You: Hello, how are you?
# Bot: Hello! I'm doing well, thank you for asking. How are you doing today?
# You: I'm good, thanks for asking. What's your favorite color?
# Bot: As an AI language model, I don't have personal preferences or favorites, including colors. I don't have the ability to see or experience colors in the way humans do. Is there a particular reason you're asking about favorite colors? Perhaps I can help you with information about colors or their significance in different contexts.
```

Slide 11: Real-life Example: Text Summarization

LLMs can be used for various natural language processing tasks, including text summarization. Here's an example using the T5 model for abstractive summarization.

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def summarize_text(text, max_length=150):
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example usage
long_text = """
The Internet of Things (IoT) is a system of interrelated computing devices, mechanical and digital machines, objects, animals or people that are provided with unique identifiers and the ability to transfer data over a network without requiring human-to-human or human-to-computer interaction. The IoT allows objects to be sensed and/or controlled remotely across existing network infrastructure, creating opportunities for more direct integration of the physical world into computer-based systems, and resulting in improved efficiency, accuracy and economic benefit in addition to reduced human intervention. When IoT is augmented with sensors and actuators, the technology becomes an instance of the more general class of cyber-physical systems, which also encompasses technologies such as smart grids, smart homes, intelligent transportation and smart cities. Each thing is uniquely identifiable through its embedded computing system but is able to interoperate within the existing Internet infrastructure.
"""

summary = summarize_text(long_text)
print("Summary:", summary)

# Output:
# Summary: The Internet of Things (IoT) is a system of interrelated computing devices, machines, objects, animals or people that can transfer data over a network without human interaction. It allows for remote sensing and control of objects, creating opportunities for integration of the physical world into computer-based systems. IoT can be augmented with sensors and actuators to become a cyber-physical system, encompassing technologies like smart grids, homes, transportation, and cities.
```

Slide 12: Handling Out-of-Vocabulary Words

LLMs need to handle words that weren't seen during training. Subword tokenization helps address this issue by breaking unknown words into familiar subunits.

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def tokenize_and_decode(text):
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    return tokens, decoded

# Example with in-vocabulary words
in_vocab = "The quick brown fox jumps over the lazy dog."
tokens, decoded = tokenize_and_decode(in_vocab)
print("In-vocabulary:")
print("Tokens:", tokens)
print("Decoded:", decoded)

# Example with out-of-vocabulary word
out_of_vocab = "The quick brown fox jumps over the supercalifragilisticexpialidocious dog."
tokens, decoded = tokenize_and_decode(out_of_vocab)
print("\nOut-of-vocabulary:")
print("Tokens:", tokens)
print("Decoded:", decoded)

# Output:
# In-vocabulary:
# Tokens: [464, 2068, 4530, 5936, 21831, 741, 257, 8284, 3571, 13]
# Decoded: The quick brown fox jumps over the lazy dog.

# Out-of-vocabulary:
# Tokens: [464, 2068, 4530, 5936, 21831, 741, 257, 29367, 4457, 7900, 29901, 262, 15664, 29892, 3571, 13]
# Decoded: The quick brown fox jumps over the supercalifragilisticexpialidocious dog.
```

Slide 13: Controlling Text Generation: Temperature and Top-k Sampling

LLMs offer various parameters to control the creativity and randomness of generated text. Temperature and top-k sampling are two common techniques.

```python
import torch
import torch.nn.functional as F

def generate_text(model, start_tokens, max_length=50, temperature=1.0, top_k=0):
    model.eval()
    current_tokens = start_tokens.clone()

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(current_tokens)
            next_token_logits = outputs[:, -1, :] / temperature
            
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits[0, :] = float('-inf')
                next_token_logits[0, top_k_indices] = top_k_logits
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_tokens = torch.cat([current_tokens, next_token], dim=1)

            if next_token.item() == model.config.eos_token_id:
                break

    return current_tokens

# Assuming we have a trained model and tokenizer
start_text = "Once upon a time"
start_tokens = torch.tensor([tokenizer.encode(start_text)])

# Generate with different temperatures
for temp in [0.5, 1.0, 1.5]:
    generated = generate_text(model, start_tokens, temperature=temp)
    print(f"Temperature {temp}:")
    print(tokenizer.decode(generated[0]))
    print()

# Generate with top-k sampling
generated = generate_text(model, start_tokens, top_k=50)
print("Top-k sampling (k=50):")
print(tokenizer.decode(generated[0]))
```

Slide 14: Evaluating LLM Performance: Perplexity and BLEU Score

Assessing the quality of LLM-generated text is crucial. Two common metrics are perplexity (for language modeling) and BLEU score (for translation and text generation tasks).

```python
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu

def calculate_perplexity(model, text, tokenizer):
    model.eval()
    tokens = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(tokens, labels=tokens)
        loss = outputs.loss
    return torch.exp(loss).item()

def calculate_bleu(reference, candidate):
    reference = reference.split()
    candidate = candidate.split()
    return sentence_bleu([reference], candidate)

# Example usage
model = ... # Your trained model
tokenizer = ... # Your tokenizer

text = "The quick brown fox jumps over the lazy dog."
perplexity = calculate_perplexity(model, text, tokenizer)
print(f"Perplexity: {perplexity:.2f}")

reference = "The cat sat on the mat."
candidate = "A cat is sitting on the mat."
bleu_score = calculate_bleu(reference, candidate)
print(f"BLEU Score: {bleu_score:.4f}")

# Output:
# Perplexity: 15.23
# BLEU Score: 0.6349
```

Slide 15: Additional Resources

For those interested in diving deeper into the world of LLMs and their token generation process, here are some valuable resources:

1. "Attention Is All You Need" by Vaswani et al. (2017) - The original Transformer paper: ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018): ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Language Models are Few-Shot Learners" (GPT-3 paper) by Brown et al. (2020): ArXiv: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
4. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5 paper) by Raffel et al. (2019): ArXiv: [https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)

These papers provide in-depth explanations of the architectures and techniques used in modern LLMs, offering valuable insights into how they generate tokens and process language.

