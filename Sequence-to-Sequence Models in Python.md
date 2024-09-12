## Sequence-to-Sequence Models in Python
Slide 1: Introduction to Sequence-to-Sequence Models

Sequence-to-Sequence (Seq2Seq) models are a type of neural network architecture designed to transform an input sequence into an output sequence. They are particularly useful for tasks like machine translation, text summarization, and speech recognition.

```python
import tensorflow as tf

# Basic Seq2Seq model structure
encoder = tf.keras.layers.LSTM(64, return_state=True)
decoder = tf.keras.layers.LSTM(64, return_sequences=True, return_state=True)
```

Slide 2: Encoder-Decoder Architecture

The Seq2Seq model consists of two main components: an encoder and a decoder. The encoder processes the input sequence and compresses it into a context vector. The decoder then uses this context vector to generate the output sequence.

```python
# Encoder
encoder_inputs = tf.keras.layers.Input(shape=(None, input_dim))
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = tf.keras.layers.Input(shape=(None, output_dim))
decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
```

Slide 3: The Encoder

The encoder reads the input sequence and creates a fixed-length context vector. This vector aims to capture the essence of the input sequence, which will be used by the decoder to generate the output.

```python
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(enc_units, return_state=True)
    
    def call(self, x):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x)
        return output, state_h, state_c

# Usage
encoder = Encoder(vocab_size=10000, embedding_dim=256, enc_units=512)
sample_input = tf.random.uniform((64, 20))  # Batch size: 64, Sequence length: 20
output, h, c = encoder(sample_input)
print(f"Encoder output shape: {output.shape}")
print(f"Hidden state shape: {h.shape}")
print(f"Cell state shape: {c.shape}")
```

Slide 4: The Decoder

The decoder takes the context vector from the encoder and generates the output sequence. It produces one element at a time, using the previously generated elements as additional input.

```python
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(dec_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x, hidden, cell):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x, initial_state=[hidden, cell])
        output = self.dense(output)
        return output, state_h, state_c

# Usage
decoder = Decoder(vocab_size=10000, embedding_dim=256, dec_units=512)
sample_input = tf.random.uniform((64, 1))  # Batch size: 64, Sequence length: 1
sample_hidden = tf.random.normal((64, 512))
sample_cell = tf.random.normal((64, 512))
output, h, c = decoder(sample_input, sample_hidden, sample_cell)
print(f"Decoder output shape: {output.shape}")
print(f"Hidden state shape: {h.shape}")
print(f"Cell state shape: {c.shape}")
```

Slide 5: Training Process

During training, we use teacher forcing, where the actual target outputs are fed as inputs to the decoder at each time step, rather than using the decoder's own predictions.

```python
# Simplified training loop
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(input_seq, target_seq, encoder, decoder):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden, enc_cell = encoder(input_seq)
        dec_hidden, dec_cell = enc_hidden, enc_cell
        
        for t in range(target_seq.shape[1]):
            dec_input = tf.expand_dims(target_seq[:, t], 1)
            predictions, dec_hidden, dec_cell = decoder(dec_input, dec_hidden, dec_cell)
            loss += loss_object(target_seq[:, t], predictions)
    
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss
```

Slide 6: Inference Process

During inference, we use the encoder to process the input sequence and initialize the decoder. Then, we generate the output sequence one element at a time, feeding each generated element back as input for the next time step.

```python
def inference(input_sequence, encoder, decoder, max_length_output):
    # Encode input sequence
    enc_output, enc_hidden, enc_cell = encoder(input_sequence)
    dec_hidden, dec_cell = enc_hidden, enc_cell
    
    # Start with <start> token
    dec_input = tf.expand_dims([start_token], 0)
    result = []
    
    for _ in range(max_length_output):
        predictions, dec_hidden, dec_cell = decoder(dec_input, dec_hidden, dec_cell)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(predicted_id)
        
        if predicted_id == end_token:
            break
        
        # Feed the predicted word id back into the model
        dec_input = tf.expand_dims([predicted_id], 0)
    
    return result
```

Slide 7: Attention Mechanism

Attention allows the model to focus on different parts of the input sequence when generating each output element. This improves performance, especially for long sequences.

```python
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    
    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# Usage in decoder
attention = AttentionLayer(10)
context_vector, attention_weights = attention(dec_hidden, enc_output)
```

Slide 8: Beam Search

Beam search is a decoding strategy that explores multiple possible output sequences simultaneously, keeping track of the most promising ones.

```python
def beam_search(encoder, decoder, input_sequence, beam_width, max_length):
    # Encode input sequence
    enc_output, enc_hidden, enc_cell = encoder(input_sequence)
    
    # Initialize beam
    beam = [([], enc_hidden, enc_cell, 0)]  # (sequence, hidden, cell, score)
    
    for _ in range(max_length):
        candidates = []
        for seq, hidden, cell, score in beam:
            if seq and seq[-1] == end_token:
                candidates.append((seq, hidden, cell, score))
                continue
            
            dec_input = tf.expand_dims([seq[-1] if seq else start_token], 0)
            predictions, new_hidden, new_cell = decoder(dec_input, hidden, cell)
            
            top_k = tf.math.top_k(predictions[0], k=beam_width)
            for i in range(beam_width):
                new_seq = seq + [top_k.indices[i].numpy()]
                new_score = score - tf.math.log(top_k.values[i]).numpy()
                candidates.append((new_seq, new_hidden, new_cell, new_score))
        
        # Select top beam_width candidates
        beam = sorted(candidates, key=lambda x: x[3])[:beam_width]
        
        if all(seq[-1] == end_token for seq, _, _, _ in beam):
            break
    
    return beam[0][0]  # Return the sequence with the highest score
```

Slide 9: Real-life Example: Machine Translation

Machine translation is one of the most common applications of Seq2Seq models. Let's consider an English to French translation system.

```python
# Simplified example of machine translation
en_vocab = {'hello': 1, 'world': 2, '<start>': 3, '<end>': 4}
fr_vocab = {'bonjour': 1, 'monde': 2, '<start>': 3, '<end>': 4}

encoder = Encoder(len(en_vocab), embedding_dim=256, enc_units=512)
decoder = Decoder(len(fr_vocab), embedding_dim=256, dec_units=512)

# Translate "hello world"
input_sequence = tf.constant([[1, 2]])  # [hello, world]
translated = inference(input_sequence, encoder, decoder, max_length_output=10)

# Convert ids back to words
result = [list(fr_vocab.keys())[list(fr_vocab.values()).index(id)] for id in translated]
print("Translation:", ' '.join(result))
```

Slide 10: Real-life Example: Text Summarization

Text summarization is another popular application of Seq2Seq models. Here's a simplified example of how it might work.

```python
# Simplified example of text summarization
vocab = {'the': 1, 'quick': 2, 'brown': 3, 'fox': 4, 'jumps': 5, 'over': 6, 'lazy': 7, 'dog': 8, '<start>': 9, '<end>': 10}

encoder = Encoder(len(vocab), embedding_dim=256, enc_units=512)
decoder = Decoder(len(vocab), embedding_dim=256, dec_units=512)

# Summarize "the quick brown fox jumps over the lazy dog"
input_sequence = tf.constant([[1, 2, 3, 4, 5, 6, 1, 7, 8]])
summary = inference(input_sequence, encoder, decoder, max_length_output=5)

# Convert ids back to words
result = [list(vocab.keys())[list(vocab.values()).index(id)] for id in summary]
print("Summary:", ' '.join(result))
```

Slide 11: Handling Variable-Length Sequences

In practice, input sequences often have different lengths. We use padding and masking to handle this.

```python
# Padding sequences
def pad_sequences(sequences, maxlen, padding='post', value=0):
    return tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=maxlen, padding=padding, value=value)

# Example usage
input_sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
padded_sequences = pad_sequences(input_sequences, maxlen=5)
print("Padded sequences:")
print(padded_sequences)

# Creating a mask
mask = tf.math.not_equal(padded_sequences, 0)
print("\nMask:")
print(mask)

# Using the mask in the model
class MaskedEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(MaskedEncoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(enc_units, return_sequences=True, return_state=True)
    
    def call(self, x, mask):
        x = self.embedding(x)
        return self.lstm(x, mask=mask)

# Usage
encoder = MaskedEncoder(vocab_size=10, embedding_dim=256, enc_units=512)
output, state_h, state_c = encoder(padded_sequences, mask=mask)
print("\nEncoder output shape:", output.shape)
print("Hidden state shape:", state_h.shape)
print("Cell state shape:", state_c.shape)
```

Slide 12: Bidirectional Encoder

A bidirectional encoder processes the input sequence in both forward and backward directions, allowing the model to capture context from both past and future tokens.

```python
class BidirectionalEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(BidirectionalEncoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(enc_units, return_sequences=True, return_state=True)
        )
    
    def call(self, x):
        x = self.embedding(x)
        output, forward_h, forward_c, backward_h, backward_c = self.bi_lstm(x)
        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
        return output, state_h, state_c

# Usage
encoder = BidirectionalEncoder(vocab_size=10000, embedding_dim=256, enc_units=512)
sample_input = tf.random.uniform((64, 20))  # Batch size: 64, Sequence length: 20
output, h, c = encoder(sample_input)
print(f"Encoder output shape: {output.shape}")
print(f"Hidden state shape: {h.shape}")
print(f"Cell state shape: {c.shape}")
```

Slide 13: Evaluation Metrics

To assess the performance of Seq2Seq models, we use various metrics such as BLEU score for translation tasks or ROUGE score for summarization tasks.

```python
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

def calculate_bleu(reference, hypothesis):
    return sentence_bleu([reference.split()], hypothesis.split())

def calculate_rouge(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]['rouge-1']['f']  # Return F1 score for ROUGE-1

# Example usage
reference = "the quick brown fox jumps over the lazy dog"
hypothesis = "a fast fox jumps over a dog"

bleu_score = calculate_bleu(reference, hypothesis)
rouge_score = calculate_rouge(reference, hypothesis)

print(f"BLEU score: {bleu_score}")
print(f"ROUGE-1 F1 score: {rouge_score}")
```

Slide 14: Additional Resources

For more in-depth information on Sequence-to-Sequence models, consider exploring these resources:

1. "Sequence to Sequence Learning with Neural Networks" by Sutskever et al. (2014) ArXiv URL: [https://arxiv.org/abs/1409.3215](https://arxiv.org/abs/1409.3215)
2. "Neural Machine Translation by Jointly Learning to Align and Translate" by Bahdanau et al. (2014) ArXiv URL: [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)
3. "Attention Is All You Need" by Vaswani et al. (2017) ArXiv URL: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
4. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv URL: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

These papers provide foundational concepts and advanced techniques in sequence-to-sequence modeling, attention mechanisms, and transformer architectures, which have significantly influenced the field of natural language processing.

