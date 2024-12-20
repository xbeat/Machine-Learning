## Encoder-Decoder Architecture for Sequence-to-Sequence Tasks

Slide 1: Introduction to Encoder-Decoder Architecture

Encoder-Decoder Architecture is a fundamental deep learning model designed for sequence-to-sequence tasks. It consists of two main components: an encoder that processes input sequences, and a decoder that generates output sequences. This architecture has revolutionized various fields, including natural language processing and computer vision.

```python
import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input_seq):
        # Encoder
        _, (hidden, cell) = self.encoder(input_seq)
        
        # Decoder
        output_seq = []
        decoder_input = hidden
        for _ in range(10):  # Assuming max output length of 10
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            output = self.output_layer(decoder_output)
            output_seq.append(output)
            decoder_input = decoder_output
        
        return torch.cat(output_seq, dim=0)

# Example usage
model = EncoderDecoder(input_dim=10, hidden_dim=20, output_dim=5)
input_sequence = torch.randn(15, 1, 10)  # (seq_len, batch_size, input_dim)
output = model(input_sequence)
print(output.shape)  # Expected: torch.Size([10, 1, 5])
```

Slide 2: The Encoder: Compressing Input Information

The encoder is responsible for processing and compressing the input sequence into a fixed-length context vector. This vector captures the essential features of the input. Typically, recurrent neural networks (RNNs) like Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU) are used as encoders due to their ability to handle sequential data effectively.

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
    
    def forward(self, input_seq):
        # input_seq shape: (seq_len, batch_size, input_dim)
        outputs, (hidden, cell) = self.lstm(input_seq)
        # outputs shape: (seq_len, batch_size, hidden_dim)
        # hidden shape: (1, batch_size, hidden_dim)
        return hidden, cell

# Example usage
encoder = Encoder(input_dim=10, hidden_dim=20)
input_sequence = torch.randn(15, 1, 10)  # (seq_len, batch_size, input_dim)
hidden, cell = encoder(input_sequence)
print(f"Hidden state shape: {hidden.shape}")
print(f"Cell state shape: {cell.shape}")
```

Slide 3: The Decoder: Generating Output Sequences

The decoder takes the context vector produced by the encoder and generates the output sequence step by step. At each step, it considers the previous output and the context vector to predict the next element in the sequence. Like the encoder, the decoder often uses recurrent neural networks to maintain context throughout the generation process.

```python
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, hidden, cell, max_length):
        outputs = []
        decoder_input = hidden
        
        for _ in range(max_length):
            decoder_output, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
            output = self.output_layer(decoder_output)
            outputs.append(output)
            decoder_input = decoder_output
        
        return torch.cat(outputs, dim=0)

# Example usage
decoder = Decoder(hidden_dim=20, output_dim=5)
hidden = torch.randn(1, 1, 20)  # (num_layers, batch_size, hidden_dim)
cell = torch.randn(1, 1, 20)    # (num_layers, batch_size, hidden_dim)
output_sequence = decoder(hidden, cell, max_length=10)
print(f"Output sequence shape: {output_sequence.shape}")
```

Slide 4: Mathematical Intuition

The encoder-decoder architecture can be formalized mathematically:

1.  Encoder function: $h\_t = f(x\_t, h\_{t-1})$ where $h\_t$ is the hidden state at time $t$, and $x\_t$ is the input at time $t$.
2.  Context vector: $C = q(h\_1, h\_2, ..., h\_n)$ where $q$ is a function that compresses the sequence of hidden states.
3.  Decoder function: $s\_t = g(y\_{t-1}, s\_{t-1}, C)$ where $s\_t$ is the decoder's hidden state at time $t$, $y\_{t-1}$ is the previous output, and $C$ is the context vector.

```python
import numpy as np

def encoder_step(x_t, h_prev):
    # Simple RNN step for illustration
    return np.tanh(np.dot(x_t, Wx) + np.dot(h_prev, Wh) + b)

def context_vector(h_seq):
    # Simple average of hidden states
    return np.mean(h_seq, axis=0)

def decoder_step(y_prev, s_prev, C):
    # Simple RNN step with context
    return np.tanh(np.dot(y_prev, Wy) + np.dot(s_prev, Ws) + np.dot(C, Wc) + b)

# Example usage (assume Wx, Wh, Wy, Ws, Wc, b are defined)
x_seq = np.random.randn(10, 5)  # Input sequence
h_seq = []
h_prev = np.zeros(5)

# Encoding
for x_t in x_seq:
    h_t = encoder_step(x_t, h_prev)
    h_seq.append(h_t)
    h_prev = h_t

C = context_vector(h_seq)

# Decoding (first step)
y_prev = np.zeros(5)
s_prev = np.zeros(5)
s_t = decoder_step(y_prev, s_prev, C)

print(f"Context vector shape: {C.shape}")
print(f"First decoder state shape: {s_t.shape}")
```

Slide 5: Teacher Forcing: Stabilizing Training

Teacher forcing is a technique used during training to improve convergence. Instead of using the decoder's previous output as input for the next step, the actual target output is used. This helps stabilize training by providing correct context, especially in the early stages when the model's predictions are less accurate.

```python
import torch
import torch.nn as nn

class EncoderDecoderWithTeacherForcing(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EncoderDecoderWithTeacherForcing, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim)
        self.decoder = nn.LSTM(output_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        # Encoder
        _, (hidden, cell) = self.encoder(input_seq)
        
        # Decoder
        target_len = target_seq.size(0)
        batch_size = target_seq.size(1)
        output_dim = self.output_layer.out_features
        
        outputs = torch.zeros(target_len, batch_size, output_dim)
        decoder_input = target_seq[0, :]  # Start token
        
        for t in range(1, target_len):
            decoder_output, (hidden, cell) = self.decoder(decoder_input.unsqueeze(0), (hidden, cell))
            output = self.output_layer(decoder_output.squeeze(0))
            outputs[t] = output
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            decoder_input = target_seq[t] if teacher_force else output.argmax(1)
        
        return outputs

# Example usage
model = EncoderDecoderWithTeacherForcing(input_dim=10, hidden_dim=20, output_dim=5)
input_sequence = torch.randn(15, 1, 10)  # (seq_len, batch_size, input_dim)
target_sequence = torch.randint(0, 5, (15, 1))  # (seq_len, batch_size)
output = model(input_sequence, target_sequence, teacher_forcing_ratio=0.5)
print(f"Output shape: {output.shape}")
```

Slide 6: Loss Function: Measuring Prediction Accuracy

The encoder-decoder architecture typically uses a loss function like cross-entropy to measure the difference between predicted and actual sequences. Minimizing this loss ensures that the model generates sequences closer to the desired output. For sequence generation tasks, we often use negative log-likelihood or perplexity as evaluation metrics.

```python
import torch
import torch.nn as nn

# Assuming we have a model and data
model = EncoderDecoderWithTeacherForcing(input_dim=10, hidden_dim=20, output_dim=5)
input_sequence = torch.randn(15, 32, 10)  # (seq_len, batch_size, input_dim)
target_sequence = torch.randint(0, 5, (15, 32))  # (seq_len, batch_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    output = model(input_sequence, target_sequence)
    
    # Reshape output and target for loss calculation
    output = output.view(-1, output.shape[-1])
    target = target_sequence.view(-1)
    
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    # Calculate perplexity
    perplexity = torch.exp(loss)
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Perplexity: {perplexity.item():.4f}")
```

Slide 7: Attention Mechanism: Enhancing Context Awareness

Incorporating attention mechanisms allows models to focus on relevant parts of the input sequence dynamically. This significantly improves performance by enabling context-aware outputs. Attention computes a weighted sum of encoder hidden states, allowing the decoder to focus on different parts of the input for each output step.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(1)
        seq_len = encoder_outputs.size(0)
        
        hidden = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(AttentionDecoder, self).__init__()
        self.attention = AttentionLayer(hidden_dim)
        self.rnn = nn.GRU(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input, hidden, encoder_outputs):
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        
        weighted = torch.bmm(a, encoder_outputs.transpose(0, 1))
        weighted = weighted.transpose(0, 1)
        
        rnn_input = torch.cat((input, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        return self.out(output.squeeze(0)), hidden

# Example usage
hidden_dim = 20
output_dim = 5
decoder = AttentionDecoder(hidden_dim, output_dim)
input = torch.randn(1, 1, hidden_dim)
hidden = torch.randn(1, 1, hidden_dim)
encoder_outputs = torch.randn(10, 1, hidden_dim)

output, hidden = decoder(input, hidden, encoder_outputs)
print(f"Output shape: {output.shape}")
print(f"Hidden shape: {hidden.shape}")
```

Slide 8: Real-World Application: Machine Translation

Machine translation is a prominent application of encoder-decoder architecture. The model learns to translate text from one language to another by encoding the source language sentence and decoding it into the target language. This process involves handling variable-length input and output sequences, making it an ideal use case for this architecture.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Translator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size):
        super(Translator, self).__init__()
        self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.decoder = nn.LSTM(embed_size + hidden_size*2, hidden_size)
        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.fc = nn.Linear(hidden_size, tgt_vocab_size)
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        src_embed = self.src_embedding(src)
        tgt_embed = self.tgt_embedding(tgt)
        
        encoder_outputs, (hidden, cell) = self.encoder(src_embed)
        
        batch_size = tgt.shape[1]
        tgt_len = tgt.shape[0]
        tgt_vocab_size = self.fc.out_features
        
        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(tgt.device)
        input = tgt_embed[0,:]
        
        hidden = hidden.view(1, batch_size, -1)
        cell = cell.view(1, batch_size, -1)
        
        for t in range(1, tgt_len):
            input = torch.cat((input, encoder_outputs[-1]), dim=1)
            output, (hidden, cell) = self.decoder(input.unsqueeze(0), (hidden, cell))
            prediction = self.fc(output.squeeze(0))
            outputs[t] = prediction
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input = (tgt_embed[t] if teacher_force else self.tgt_embedding(top1))
        
        return outputs

# Example usage
src_vocab_size, tgt_vocab_size = 5000, 6000
embed_size, hidden_size = 256, 512
translator = Translator(src_vocab_size, tgt_vocab_size, embed_size, hidden_size)

src = torch.randint(1, src_vocab_size, (10, 32))  # (seq_len, batch_size)
tgt = torch.randint(1, tgt_vocab_size, (12, 32))  # (seq_len, batch_size)

output = translator(src, tgt)
print(f"Translation output shape: {output.shape}")
```

Slide 9: Real-World Application: Text Summarization

Text summarization is another important application of encoder-decoder architecture. The model learns to condense long documents into concise summaries, capturing the most important information. This task demonstrates the architecture's ability to handle variable-length input and generate coherent, shortened output.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextSummarizer(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(TextSummarizer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers, bidirectional=True)
        self.decoder = nn.LSTM(embed_size + hidden_size*2, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        vocab_size = self.fc.out_features
        
        embedded = self.embedding(source)
        encoder_outputs, (hidden, cell) = self.encoder(embedded)
        
        hidden = hidden.view(self.decoder.num_layers, batch_size, -1)
        cell = cell.view(self.decoder.num_layers, batch_size, -1)
        
        outputs = torch.zeros(target_len, batch_size, vocab_size).to(source.device)
        input = target[0,:]
        
        for t in range(1, target_len):
            input_embed = self.embedding(input)
            input_combined = torch.cat((input_embed, encoder_outputs[-1]), dim=1)
            
            output, (hidden, cell) = self.decoder(input_combined.unsqueeze(0), (hidden, cell))
            prediction = self.fc(output.squeeze(0))
            outputs[t] = prediction
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input = target[t] if teacher_force else top1
        
        return outputs

# Example usage
vocab_size, embed_size, hidden_size, num_layers = 10000, 256, 512, 2
summarizer = TextSummarizer(vocab_size, embed_size, hidden_size, num_layers)

source = torch.randint(1, vocab_size, (50, 32))  # (seq_len, batch_size)
target = torch.randint(1, vocab_size, (20, 32))  # (seq_len, batch_size)

summary = summarizer(source, target)
print(f"Summary output shape: {summary.shape}")
```

Slide 10: Beam Search: Improving Decoding

Beam search is a decoding technique that improves upon greedy decoding by maintaining multiple candidate sequences at each step. This approach helps in generating more coherent and diverse outputs, especially in tasks like machine translation and text summarization.

```python
import torch
import torch.nn.functional as F

def beam_search_decode(model, encoder_output, start_token, end_token, beam_width, max_length):
    batch_size = encoder_output.size(1)
    hidden = model.init_hidden(batch_size)
    
    # Start with the start token for each example in the batch
    input_seq = torch.LongTensor([[start_token] for _ in range(batch_size)])
    
    # Initialize the beam
    beam = [(input_seq, hidden, 0.0)]
    
    for _ in range(max_length):
        all_candidates = []
        for seq, hidden, score in beam:
            if seq[-1][0] == end_token:
                all_candidates.append((seq, hidden, score))
            else:
                input = seq[:, -1].unsqueeze(1)
                output, hidden = model.decode_step(input, hidden, encoder_output)
                log_probs = F.log_softmax(output, dim=-1)
                
                top_log_probs, top_indices = log_probs.topk(beam_width)
                
                for i in range(beam_width):
                    candidate_seq = torch.cat([seq, top_indices[:, :, i]], dim=-1)
                    candidate_score = score + top_log_probs[:, :, i].item()
                    all_candidates.append((candidate_seq, hidden, candidate_score))
        
        # Select the top beam_width candidates
        beam = sorted(all_candidates, key=lambda x: x[2], reverse=True)[:beam_width]
        
        # Stop if all beams have reached the end token
        if all(seq[-1][0] == end_token for seq, _, _ in beam):
            break
    
    # Return the sequence with the highest score
    return beam[0][0]

# Example usage (assuming model is defined)
encoder_output = torch.randn(10, 32, 512)  # (seq_len, batch_size, hidden_size)
start_token, end_token = 1, 2
beam_width, max_length = 5, 20

best_sequence = beam_search_decode(model, encoder_output, start_token, end_token, beam_width, max_length)
print(f"Best sequence shape: {best_sequence.shape}")
```

Slide 11: Handling Long Sequences: Truncation and Padding

Encoder-decoder models often need to handle variable-length sequences. Two common techniques for managing this are truncation (cutting off long sequences) and padding (adding dummy tokens to short sequences). These methods ensure that all sequences in a batch have the same length, which is necessary for efficient batch processing.

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def preprocess_sequences(sequences, max_length, pad_token=0):
    # Truncate long sequences
    truncated = [seq[:max_length] for seq in sequences]
    
    # Convert to tensor and pad short sequences
    padded = pad_sequence([torch.tensor(seq) for seq in truncated], 
                          batch_first=True, 
                          padding_value=pad_token)
    
    # Create mask for padding
    mask = (padded != pad_token).float()
    
    return padded, mask

# Example usage
sequences = [
    [1, 2, 3, 4, 5, 6, 7, 8],
    [1, 2, 3],
    [1, 2, 3, 4, 5],
    [1]
]
max_length = 6
pad_token = 0

padded_sequences, mask = preprocess_sequences(sequences, max_length, pad_token)

print("Padded sequences:")
print(padded_sequences)
print("\nMask:")
print(mask)
```

Slide 12: Evaluation Metrics: BLEU Score

The BLEU (Bilingual Evaluation Understudy) score is a common metric for evaluating the quality of machine-generated text, especially in translation tasks. It measures the similarity between the model's output and reference texts by comparing n-gram overlaps.

```python
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

def calculate_bleu(references, hypotheses):
    bleu_scores = []
    for ref, hyp in zip(references, hypotheses):
        score = sentence_bleu([ref.split()], hyp.split())
        bleu_scores.append(score)
    return np.mean(bleu_scores)

# Example usage
references = [
    "The cat is on the mat.",
    "I love to eat pizza.",
    "Python is a programming language."
]

hypotheses = [
    "The cat sits on the mat.",
    "I enjoy eating pizza.",
    "Python is a popular coding language."
]

bleu_score = calculate_bleu(references, hypotheses)
print(f"Average BLEU score: {bleu_score:.4f}")
```

Slide 13: Challenges and Future Directions

While encoder-decoder architectures have achieved remarkable success, they face challenges such as handling very long sequences, maintaining coherence in generated text, and capturing complex dependencies. Future research directions include:

1.  Improving attention mechanisms for better long-range dependencies
2.  Incorporating external knowledge for more informed generation
3.  Developing more efficient training methods for large-scale models
4.  Enhancing model interpretability and controlling generation biases

```python
import torch
import torch.nn as nn

class ImprovedEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(ImprovedEncoderDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=8),
            num_layers=6
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_size, nhead=8),
            num_layers=6
        )
        self.fc = nn.Linear(embed_size, vocab_size)
    
    def forward(self, src, tgt):
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)
        
        src_mask = self.generate_square_subsequent_mask(src.size(0))
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0))
        
        memory = self.encoder(src_embed, src_mask)
        output = self.decoder(tgt_embed, memory, tgt_mask)
        return self.fc(output)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# Example usage
vocab_size, embed_size, hidden_size = 10000, 512, 512
model = ImprovedEncoderDecoder(vocab_size, embed_size, hidden_size)

src = torch.randint(1, vocab_size, (20, 32))  # (seq_len, batch_size)
tgt = torch.randint(1, vocab_size, (15, 32))  # (seq_len, batch_size)

output = model(src, tgt)
print(f"Output shape: {output.shape}")
```

Slide 14: Additional Resources

For those interested in diving deeper into encoder-decoder architectures and related topics, here are some valuable resources:

1.  "Sequence to Sequence Learning with Neural Networks" by Sutskever et al. (2014) ArXiv: [https://arxiv.org/abs/1409.3215](https://arxiv.org/abs/1409.3215)
2.  "Neural Machine Translation by Jointly Learning to Align and Translate" by Bahdanau et al. (2014) ArXiv: [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)
3.  "Attention Is All You Need" by Vaswani et al. (2017) ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
4.  "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

These papers provide foundational knowledge and advanced techniques in the field of sequence-to-sequence learning and natural language processing.

