## Positional Encoding in Transformer Architectures
Slide 1: Understanding Positional Encoding in Transformers

Positional encoding is a crucial component in transformer architectures, enabling these models to understand the order of input sequences. Unlike recurrent neural networks, transformers process all input elements simultaneously, necessitating an explicit way to incorporate position information.

```python
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(position, d_model):
    angle_rads = np.arange(d_model) // 2 * 2 * np.pi / d_model
    angle_rads = position[:, np.newaxis] / np.power(10000, angle_rads[np.newaxis, :])
    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return pos_encoding

# Generate positional encoding for 100 positions and 128 dimensions
pos_encoding = positional_encoding(np.arange(100), 128)

# Plot the positional encoding
plt.figure(figsize=(12, 8))
plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.colorbar()
plt.title('Positional Encoding Visualization')
plt.show()
```

Slide 2: Sine and Cosine Functions in Positional Encoding

Positional encoding typically uses sine and cosine functions of different frequencies. This approach allows the model to easily learn to attend to relative positions, as the absolute position of each token can be expressed as a linear function of these encodings.

```python
import numpy as np
import matplotlib.pyplot as plt

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return pos_encoding

# Plot sine and cosine functions for different frequencies
plt.figure(figsize=(12, 6))
position = 50
d_model = 512
pos_encoding = positional_encoding(position, d_model)
plt.plot(np.arange(d_model), pos_encoding[0, 0, :])
plt.xlabel('Dimension')
plt.ylabel('Value')
plt.title('Sine and Cosine Functions in Positional Encoding')
plt.legend(['sin/cos'])
plt.show()
```

Slide 3: Adding Positional Encoding to Input Embeddings

In practice, positional encodings are added to the input embeddings. This allows the model to use both the semantic information from the embeddings and the position information from the encodings.

```python
import numpy as np
import tensorflow as tf

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

# Example usage
vocab_size = 10000
d_model = 512
max_position = 100

# Create random input embeddings
input_embeddings = tf.random.uniform((1, max_position, d_model))

# Generate positional encodings
pos_encoding = positional_encoding(max_position, d_model)

# Add positional encodings to input embeddings
output = input_embeddings + pos_encoding

print(f"Shape of output: {output.shape}")
print(f"First few values of output[0, 0]: {output[0, 0, :5]}")
```

Slide 4: Learned vs. Fixed Positional Encodings

While the sinusoidal positional encoding is commonly used, some transformer variants use learned positional encodings. Let's compare both approaches.

```python
import tensorflow as tf

class FixedPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_position, d_model):
        super(FixedPositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(max_position, d_model)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return tf.cast(pos, tf.float32) * angle_rates
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.range(position)[:, tf.newaxis],
            tf.range(d_model)[tf.newaxis, :],
            d_model
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return pos_encoding[tf.newaxis, ...]
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class LearnedPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_position, d_model):
        super(LearnedPositionalEncoding, self).__init__()
        self.pos_embedding = tf.keras.layers.Embedding(max_position, d_model)
    
    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        positions = self.pos_embedding(positions)
        return inputs + positions

# Example usage
max_position = 100
d_model = 512
inputs = tf.random.uniform((1, max_position, d_model))

fixed_pe = FixedPositionalEncoding(max_position, d_model)
learned_pe = LearnedPositionalEncoding(max_position, d_model)

fixed_output = fixed_pe(inputs)
learned_output = learned_pe(inputs)

print(f"Fixed PE output shape: {fixed_output.shape}")
print(f"Learned PE output shape: {learned_output.shape}")
```

Slide 5: Visualizing Positional Encoding Patterns

To better understand how positional encoding captures position information, let's visualize the patterns it creates for different sequence lengths.

```python
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(max_position, d_model):
    angle_rads = get_angles(np.arange(max_position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    return angle_rads

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

# Generate positional encodings for different sequence lengths
seq_lengths = [10, 50, 100, 200]
d_model = 512

fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs = axs.ravel()

for i, length in enumerate(seq_lengths):
    pos_encoding = positional_encoding(length, d_model)
    axs[i].pcolormesh(pos_encoding, cmap='RdBu')
    axs[i].set_title(f'Positional Encoding (Length = {length})')
    axs[i].set_xlabel('Dimension')
    axs[i].set_ylabel('Position')

plt.tight_layout()
plt.show()
```

Slide 6: Impact of Positional Encoding on Attention Mechanisms

Positional encoding allows attention mechanisms to consider token positions. Let's implement a simple attention mechanism and observe how positional encoding affects its output.

Slide 7: Impact of Positional Encoding on Attention Mechanisms

```python
import tensorflow as tf

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        return output, attention_weights

# Example usage
d_model = 512
num_heads = 8
seq_length = 10
batch_size = 1

mha = MultiHeadAttention(d_model, num_heads)
input_seq = tf.random.uniform((batch_size, seq_length, d_model))
output, attention_weights = mha(input_seq, input_seq, input_seq, mask=None)

print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
```

Slide 8: Positional Encoding in Transformer Encoder

Let's implement a simple Transformer encoder layer to see how positional encoding integrates with the overall architecture.

Slide 9: Positional Encoding in Transformer Encoder

```python
import tensorflow as tf

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

# Example usage
d_model = 512
num_heads = 8
dff = 2048
seq_length = 10
batch_size = 1

encoder_layer = EncoderLayer(d_model, num_heads, dff)
input_seq = tf.random.uniform((batch_size, seq_length, d_model))
output = encoder_layer(input_seq, training=False, mask=None)

print(f"Encoder layer output shape: {output.shape}")
```

Slide 10: Positional Encoding in Transformer Decoder

The decoder layer in a Transformer incorporates positional encoding to maintain sequence order awareness during the decoding process. Here's a simplified implementation of a decoder layer:

Slide 11: Positional Encoding in Transformer Decoder

```python
import tensorflow as tf

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

# Example usage
d_model = 512
num_heads = 8
dff = 2048
seq_length = 10
batch_size = 1

decoder_layer = DecoderLayer(d_model, num_heads, dff)
input_seq = tf.random.uniform((batch_size, seq_length, d_model))
enc_output = tf.random.uniform((batch_size, seq_length, d_model))
output, _, _ = decoder_layer(input_seq, enc_output, training=False, 
                             look_ahead_mask=None, padding_mask=None)

print(f"Decoder layer output shape: {output.shape}")
```

Slide 12: Positional Encoding in Self-Attention Mechanism

Self-attention is a key component of Transformers, and positional encoding plays a crucial role in its effectiveness. Let's examine how positional information influences self-attention:

Slide 13: Positional Encoding in Self-Attention Mechanism

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

# Example usage
seq_length = 10
d_model = 512
batch_size = 1

# Generate input sequence and positional encoding
input_seq = tf.random.uniform((batch_size, seq_length, d_model))
pos_encoding = positional_encoding(seq_length, d_model)

# Add positional encoding to input
input_with_pos = input_seq + pos_encoding

# Perform self-attention
q = k = v = input_with_pos
output, attention_weights = scaled_dot_product_attention(q, k, v, mask=None)

# Visualize attention weights
plt.figure(figsize=(8, 6))
plt.imshow(attention_weights[0], cmap='viridis')
plt.title('Self-Attention Weights with Positional Encoding')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.colorbar()
plt.show()

print(f"Self-attention output shape: {output.shape}")
```

Slide 14: Positional Encoding in Language Translation

Let's examine how positional encoding contributes to a simple language translation task using a Transformer model:

Slide 15: Positional Encoding in Language Translation

```python
import tensorflow as tf
import numpy as np

class TransformerTranslator(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                 target_vocab_size, max_position_encoding, rate=0.1):
        super(TransformerTranslator, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, max_position_encoding, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                               target_vocab_size, max_position_encoding, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inputs, training):
        inp, tar = inputs
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, _ = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output
    
    def create_masks(self, inp, tar):
        enc_padding_mask = create_padding_mask(inp)
        dec_padding_mask = create_padding_mask(inp)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, look_ahead_mask, dec_padding_mask

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

# Example usage (simplified for demonstration)
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
input_vocab_size = 8500
target_vocab_size = 8000
max_position_encoding = 10000

translator = TransformerTranslator(num_layers, d_model, num_heads, dff,
                                   input_vocab_size, target_vocab_size,
                                   max_position_encoding)

# Dummy input
inp = tf.random.uniform((1, 40), dtype=tf.int64, minval=0, maxval=200)
tar = tf.random.uniform((1, 30), dtype=tf.int64, minval=0, maxval=200)

output = translator((inp, tar), training=False)
print(f"Translation output shape: {output.shape}")
```

Slide 16: Positional Encoding in Text Generation

Positional encoding is crucial for maintaining coherence in text generation tasks. Let's implement a simple text generator using a Transformer decoder:

Slide 17: Positional Encoding in Text Generation

```python
import tensorflow as tf
import numpy as np

class TextGenerator(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, dff, max_length):
        super(TextGenerator, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_length, d_model)
        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff) for _ in range(4)]
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.final_layer = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x, training):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, None, training, look_ahead_mask=None, padding_mask=None)
        output = self.final_layer(x)
        return output

def generate_text(model, start_string, num_generate, temperature=1.0):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()
    
    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    
    return start_string + ''.join(text_generated)

# Example usage (simplified for demonstration)
vocab_size = 10000
d_model = 256
num_heads = 8
dff = 512
max_length = 100

generator = TextGenerator(vocab_size, d_model, num_heads, dff, max_length)

# Dummy input
input_sequence = tf.random.uniform((1, 20), dtype=tf.int64, minval=0, maxval=vocab_size)
output = generator(input_sequence, training=False)
print(f"Generated text logits shape: {output.shape}")

# Note: char2idx and idx2char mappings would be needed for actual text generation
```

Slide 18: Positional Encoding in Image Captioning

Positional encoding can be adapted for image captioning tasks, where it helps maintain spatial awareness in image features:

Slide 19: Positional Encoding in Image Captioning

```python
import tensorflow as tf
import numpy as np

class ImageCaptioner(tf.keras.Model):
    def __init__(self, vocab_size, max_length, d_model, num_heads, dff):
        super(ImageCaptioner, self).__init__()
        self.encoder = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
        self.encoder.trainable = False
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.feature_dense = tf.keras.layers.Dense(d_model)
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_length, d_model)
        self.decoder = DecoderLayer(d_model, num_heads, dff)
        self.final_layer = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs, training):
        image, caption = inputs
        image_features = self.encoder(image)
        image_features = self.pooling(image_features)
        image_features = self.feature_dense(image_features)
        image_features = tf.expand_dims(image_features, 1)
        
        caption = self.embedding(caption)
        seq_len = tf.shape(caption)[1]
        caption += self.pos_encoding[:, :seq_len, :]
        
        x = self.decoder(caption, image_features, training, look_ahead_mask=None, padding_mask=None)
        x = self.final_layer(x)
        return x

# Example usage
vocab_size = 10000
max_length = 50
d_model = 256
num_heads = 8
dff = 512

captioner = ImageCaptioner(vocab_size, max_length, d_model, num_heads, dff)

# Dummy inputs
image = tf.random.uniform((1, 224, 224, 3))
caption = tf.random.uniform((1, 20), dtype=tf.int64, minval=0, maxval=vocab_size)

output = captioner((image, caption), training=False)
print(f"Image captioning output shape: {output.shape}")
```

Slide 20: Positional Encoding in Sentiment Analysis

Even in tasks like sentiment analysis, where the overall sentiment might not be position-dependent, positional encoding can help capture nuanced relationships:

Slide 21: Positional Encoding in Sentiment Analysis

```python
import tensorflow as tf
import numpy as np

class SentimentAnalyzer(tf.keras.Model):
    def __init__(self, vocab_size, max_length, d_model, num_heads, dff):
        super(SentimentAnalyzer, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_length, d_model)
        self.encoder_layer = EncoderLayer(d_model, num_heads, dff)
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.dense = tf.keras.layers.Dense(64, activation='relu')
        self.final_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, x, training):
        x = self.embedding(x)
        seq_len = tf.shape(x)[1]
        x += self.pos_encoding[:, :seq_len, :]
        x = self.encoder_layer(x, training, mask=None)
        x = self.pooling(x)
        x = self.dense(x)
        return self.final_layer(x)

# Example usage
vocab_size = 10000
max_length = 100
d_model = 256
num_heads = 8
dff = 512

analyzer = SentimentAnalyzer(vocab_size, max_length, d_model, num_heads, dff)

# Dummy input
input_sequence = tf.random.uniform((1, 50), dtype=tf.int64, minval=0, maxval=vocab_size)

output = analyzer(input_sequence, training=False)
print(f"Sentiment analysis output shape: {output.shape}")
```

Slide 22: Additional Resources

For those interested in diving deeper into positional encoding and Transformer architectures, here are some valuable resources:

1. "Attention Is All You Need" - The original Transformer paper by Vaswani et al. (2017) ArXiv link: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv link: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Positional Encoding in Transformers" - A detailed blog post on positional encoding URL: [https://kazemnejad.com/blog/transformer\_architecture\_positional\_encoding/](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
4. "The Illustrated Transformer" by Jay Alammar - A visual guide to Transformer architectures URL: [http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)
5. "Transformer: A Novel Neural Network Architecture for Language Understanding" - Google AI Blog post URL: [https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)


