## Sequence Modeling Algorithms and Applications
Slide 1: Introduction to Sequence Models

Sequence models are a class of machine learning algorithms designed to process and analyze sequential data. These models are crucial in various fields, including natural language processing, time series analysis, and bioinformatics. They excel at capturing patterns and dependencies in ordered data, making them invaluable for tasks like language translation, speech recognition, and stock price prediction.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a sample sequence
sequence = np.sin(np.linspace(0, 4*np.pi, 100))

# Plot the sequence
plt.figure(figsize=(10, 4))
plt.plot(sequence)
plt.title("Example of a Sequential Data: Sine Wave")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.show()
```

Slide 2: Recurrent Neural Networks (RNNs)

Recurrent Neural Networks are the foundation of sequence modeling. They process sequences by maintaining an internal state or "memory" that allows them to capture temporal dependencies. RNNs can handle variable-length sequences and share parameters across different time steps, making them efficient for sequential data processing.

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hidden = self.rnn(x)
        output = self.fc(hidden.squeeze(0))
        return output

# Example usage
input_size, hidden_size, output_size = 10, 20, 5
model = SimpleRNN(input_size, hidden_size, output_size)
input_sequence = torch.randn(1, 30, input_size)  # Batch size 1, sequence length 30
output = model(input_sequence)
print(f"Output shape: {output.shape}")
```

Slide 3: Long Short-Term Memory (LSTM) Networks

LSTMs are an advanced form of RNNs designed to address the vanishing gradient problem. They introduce a more complex cell structure with gates that control the flow of information. This allows LSTMs to capture long-term dependencies more effectively, making them suitable for tasks requiring memory over extended sequences.

```python
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden.squeeze(0))
        return output

# Example usage
input_size, hidden_size, output_size = 10, 20, 5
model = SimpleLSTM(input_size, hidden_size, output_size)
input_sequence = torch.randn(1, 30, input_size)  # Batch size 1, sequence length 30
output = model(input_sequence)
print(f"Output shape: {output.shape}")
```

Slide 4: Gated Recurrent Units (GRUs)

Gated Recurrent Units are another variant of RNNs that simplify the LSTM architecture while maintaining its ability to capture long-term dependencies. GRUs use two gates (reset and update) instead of the three gates in LSTMs, resulting in a more computationally efficient model that often performs comparably to LSTMs.

```python
import torch
import torch.nn as nn

class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hidden = self.gru(x)
        output = self.fc(hidden.squeeze(0))
        return output

# Example usage
input_size, hidden_size, output_size = 10, 20, 5
model = SimpleGRU(input_size, hidden_size, output_size)
input_sequence = torch.randn(1, 30, input_size)  # Batch size 1, sequence length 30
output = model(input_sequence)
print(f"Output shape: {output.shape}")
```

Slide 5: Bidirectional RNNs

Bidirectional RNNs process sequences in both forward and backward directions, allowing the model to capture context from both past and future states. This bidirectional flow of information is particularly useful in tasks where the entire sequence is available at once, such as machine translation or sentiment analysis.

```python
import torch
import torch.nn as nn

class BidirectionalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        _, hidden = self.rnn(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.fc(hidden)
        return output

# Example usage
input_size, hidden_size, output_size = 10, 20, 5
model = BidirectionalRNN(input_size, hidden_size, output_size)
input_sequence = torch.randn(1, 30, input_size)  # Batch size 1, sequence length 30
output = model(input_sequence)
print(f"Output shape: {output.shape}")
```

Slide 6: Attention Mechanisms

Attention mechanisms allow sequence models to focus on specific parts of the input sequence when generating each output. This technique has revolutionized sequence modeling, enabling models to handle longer sequences and capture complex dependencies more effectively. Attention is a key component in state-of-the-art models like Transformers.

```python
import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states):
        attention_weights = torch.softmax(self.attention(hidden_states), dim=1)
        context_vector = torch.sum(attention_weights * hidden_states, dim=1)
        return context_vector

# Example usage
hidden_size = 20
attention = AttentionLayer(hidden_size)
hidden_states = torch.randn(1, 30, hidden_size)  # Batch size 1, sequence length 30
context = attention(hidden_states)
print(f"Context vector shape: {context.shape}")
```

Slide 7: Transformer Architecture

The Transformer architecture, introduced in the "Attention is All You Need" paper, relies solely on attention mechanisms without using recurrence. It has become the foundation for many state-of-the-art models in natural language processing. Transformers use self-attention to process input sequences in parallel, making them highly efficient and effective for various sequence modeling tasks.

```python
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        x = self.embedding(x)
        output = self.transformer(x)
        return output

# Example usage
input_size, hidden_size, num_heads, num_layers = 10, 64, 4, 2
model = TransformerEncoder(input_size, hidden_size, num_heads, num_layers)
input_sequence = torch.randn(30, 1, input_size)  # Sequence length 30, batch size 1
output = model(input_sequence)
print(f"Output shape: {output.shape}")
```

Slide 8: Sequence-to-Sequence Models

Sequence-to-sequence models, often called encoder-decoder models, are designed to transform one sequence into another. They consist of an encoder that processes the input sequence and a decoder that generates the output sequence. These models are widely used in machine translation, text summarization, and speech recognition tasks.

```python
import torch
import torch.nn as nn

class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        _, (hidden, cell) = self.encoder(src)
        
        batch_size = tgt.size(0)
        max_len = tgt.size(1)
        outputs = torch.zeros(batch_size, max_len, self.fc.out_features).to(tgt.device)
        
        input = tgt[:, 0, :]
        for t in range(1, max_len):
            output, (hidden, cell) = self.decoder(input.unsqueeze(1), (hidden, cell))
            output = self.fc(output.squeeze(1))
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = tgt[:, t] if teacher_force else output
        
        return outputs

# Example usage
input_size, hidden_size, output_size = 10, 20, 5
model = Seq2SeqModel(input_size, hidden_size, output_size)
src_sequence = torch.randn(1, 30, input_size)  # Batch size 1, source sequence length 30
tgt_sequence = torch.randn(1, 25, output_size)  # Batch size 1, target sequence length 25
output = model(src_sequence, tgt_sequence)
print(f"Output shape: {output.shape}")
```

Slide 9: Beam Search Decoding

Beam search is a heuristic search algorithm used in sequence-to-sequence models to improve the quality of generated sequences. Instead of greedily selecting the most probable token at each step, beam search maintains a set of partial hypotheses and explores multiple promising paths simultaneously. This technique often leads to better overall sequence generation.

```python
import torch
import torch.nn.functional as F

def beam_search_decode(model, encoder_output, start_token, end_token, max_length, beam_width):
    batch_size = encoder_output.size(0)
    hidden = encoder_output
    
    # Initialize the beam with start tokens
    beam = [(start_token, hidden, 0)]
    complete_sequences = []
    
    for _ in range(max_length):
        candidates = []
        for seq, h, score in beam:
            if seq[-1] == end_token:
                complete_sequences.append((seq, score))
                continue
            
            # Get the next token probabilities
            output, h = model.decode_step(seq[-1].unsqueeze(0), h)
            probabilities = F.log_softmax(output, dim=-1)
            
            # Get top-k candidates
            top_k_probs, top_k_indices = probabilities.topk(beam_width)
            for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
                candidates.append((seq + [idx], h, score + prob.item()))
        
        # Select top beam_width candidates
        beam = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_width]
        
        # Early stopping if all sequences are complete
        if len(complete_sequences) == beam_width:
            break
    
    # Return the best complete sequence or the best incomplete sequence
    if complete_sequences:
        return max(complete_sequences, key=lambda x: x[1])[0]
    else:
        return beam[0][0]

# Note: This is a simplified implementation. In practice, you would need to implement
# the model's decode_step method and handle batching for efficiency.
```

Slide 10: Conditional Random Fields (CRFs)

Conditional Random Fields are probabilistic models used for structured prediction tasks, particularly in sequence labeling problems like named entity recognition and part-of-speech tagging. CRFs model the conditional probability of a label sequence given an input sequence, taking into account the dependencies between adjacent labels.

```python
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, len(tag_to_ix))
        self.crf = CRF(len(tag_to_ix), batch_first=True)
    
    def forward(self, x, tags=None):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        emissions = self.hidden2tag(lstm_out)
        
        if tags is not None:
            return -self.crf(emissions, tags)  # Negative log-likelihood for training
        else:
            return self.crf.decode(emissions)  # Best tag sequence for inference

# Example usage
vocab_size, tag_to_ix, embedding_dim, hidden_dim = 1000, {"O": 0, "B": 1, "I": 2}, 100, 128
model = BiLSTM_CRF(vocab_size, tag_to_ix, embedding_dim, hidden_dim)
sample_sentence = torch.LongTensor([[1, 2, 3, 4, 5]])  # Example sentence with 5 words
predicted_tags = model(sample_sentence)
print(f"Predicted tags: {predicted_tags}")
```

Slide 11: Hidden Markov Models (HMMs)

Hidden Markov Models are probabilistic models used for modeling sequential data with hidden states. They assume that the sequence of observations is generated by a Markov process with unobserved (hidden) states. HMMs are widely used in speech recognition, part-of-speech tagging, and bioinformatics for tasks like gene prediction.

```python
import numpy as np

class HiddenMarkovModel:
    def __init__(self, n_states, n_observations):
        self.n_states = n_states
        self.n_observations = n_observations
        self.transition_probs = np.random.rand(n_states, n_states)
        self.transition_probs /= self.transition_probs.sum(axis=1, keepdims=True)
        self.emission_probs = np.random.rand(n_states, n_observations)
        self.emission_probs /= self.emission_probs.sum(axis=1, keepdims=True)
        self.initial_probs = np.random.rand(n_states)
        self.initial_probs /= self.initial_probs.sum()

    def viterbi(self, observations):
        T = len(observations)
        viterbi = np.zeros((self.n_states, T))
        backpointer = np.zeros((self.n_states, T), dtype=int)
        
        viterbi[:, 0] = self.initial_probs * self.emission_probs[:, observations[0]]
        
        for t in range(1, T):
            for s in range(self.n_states):
                transition_probs = self.transition_probs[:, s]
                emission_prob = self.emission_probs[s, observations[t]]
                viterbi[s, t] = np.max(viterbi[:, t-1] * transition_probs) * emission_prob
                backpointer[s, t] = np.argmax(viterbi[:, t-1] * transition_probs)
        
        best_path = [np.argmax(viterbi[:, T-1])]
        for t in range(T-1, 0, -1):
            best_path.insert(0, backpointer[best_path[0], t])
        
        return best_path

# Example usage
hmm = HiddenMarkovModel(n_states=2, n_observations=3)
observations = [0, 1, 2, 1, 0]
best_path = hmm.viterbi(observations)
print(f"Best hidden state sequence: {best_path}")
```

Slide 12: N-gram Models

N-gram models are simple yet effective sequence models that predict the probability of a word based on the N-1 preceding words. These models are widely used in natural language processing for tasks such as language modeling, text generation, and speech recognition. Despite their simplicity, N-gram models can be surprisingly effective for many applications.

```python
import numpy as np
from collections import defaultdict

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)

    def train(self, text):
        words = text.split()
        for i in range(len(words) - self.n + 1):
            context = tuple(words[i:i+self.n-1])
            word = words[i+self.n-1]
            self.counts[context][word] += 1
            self.context_counts[context] += 1

    def predict_next_word(self, context):
        context = tuple(context.split()[-self.n+1:])
        if context in self.counts:
            total_count = self.context_counts[context]
            probabilities = {word: count / total_count for word, count in self.counts[context].items()}
            return max(probabilities, key=probabilities.get)
        return None

# Example usage
text = "the quick brown fox jumps over the lazy dog"
model = NGramModel(3)
model.train(text)

context = "the quick"
next_word = model.predict_next_word(context)
print(f"Predicted next word after '{context}': {next_word}")
```

Slide 13: Real-World Application: Named Entity Recognition

Named Entity Recognition (NER) is a crucial task in natural language processing that involves identifying and classifying named entities (e.g., person names, organizations, locations) in text. Sequence models, particularly Bidirectional LSTMs with CRF layers, have proven highly effective for NER tasks.

```python
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF_NER(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF_NER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, len(tag_to_ix))
        self.crf = CRF(len(tag_to_ix), batch_first=True)
    
    def forward(self, x, tags=None):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        emissions = self.hidden2tag(lstm_out)
        
        if tags is not None:
            return -self.crf(emissions, tags)  # Negative log-likelihood for training
        else:
            return self.crf.decode(emissions)  # Best tag sequence for inference

# Example usage
vocab_size, tag_to_ix, embedding_dim, hidden_dim = 10000, {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4}, 100, 128
model = BiLSTM_CRF_NER(vocab_size, tag_to_ix, embedding_dim, hidden_dim)

# Simulating a sentence: "John works at Google"
sample_sentence = torch.LongTensor([[1, 2, 3, 4]])
predicted_tags = model(sample_sentence)
print(f"Predicted NER tags: {predicted_tags}")
```

Slide 14: Real-World Application: Time Series Forecasting

Time series forecasting is a critical application of sequence models in various domains, including weather prediction, stock market analysis, and demand forecasting. Long Short-Term Memory (LSTM) networks have shown remarkable success in capturing complex temporal dependencies in time series data.

```python
import torch
import torch.nn as nn

class LSTM_Forecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Forecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Example usage
input_size, hidden_size, num_layers, output_size = 1, 64, 2, 1
model = LSTM_Forecaster(input_size, hidden_size, num_layers, output_size)

# Simulating a time series with 30 time steps
sample_series = torch.randn(1, 30, 1)  # Batch size 1, 30 time steps, 1 feature
forecast = model(sample_series)
print(f"Forecasted value: {forecast.item():.4f}")
```

Slide 15: Additional Resources

For those interested in delving deeper into sequence models and their applications, here are some valuable resources:

1. "Sequence Models" course by deeplearning.ai on Coursera
2. "Attention Is All You Need" paper (Vaswani et al., 2017): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
3. "Speech and Language Processing" by Jurafsky and Martin
4. "Natural Language Processing with Transformers" by Lewis Tunstall, Leandro von Werra, and Thomas Wolf
5. TensorFlow and PyTorch documentation for sequence modeling tutorials

These resources provide comprehensive coverage of sequence modeling techniques, from foundational concepts to cutting-edge research. They offer both theoretical insights and practical implementations to help you master this critical area of machine learning.

