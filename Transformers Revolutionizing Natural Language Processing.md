## Transformers Revolutionizing Natural Language Processing
Slide 1: Input Embedding and Tokenization

The first step in implementing a Transformer is converting text into numerical representations. We'll create a basic tokenizer and embedder that transforms input sequences into dense vectors while preserving positional information through sinusoidal encoding.

```python
import numpy as np

class TransformerEmbedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Initialize random embedding matrix
        self.embedding = np.random.randn(vocab_size, d_model)
        
    def positional_encoding(self, sequence_length):
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pos_encoding = np.zeros((sequence_length, self.d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        return pos_encoding
    
    def encode(self, token_ids, sequence_length=100):
        # Get token embeddings
        token_embeddings = self.embedding[token_ids]
        # Add positional encoding
        pos_encoding = self.positional_encoding(sequence_length)[:len(token_ids)]
        return token_embeddings + pos_encoding

# Example usage
embedder = TransformerEmbedding(vocab_size=5000, d_model=512)
token_sequence = np.array([1, 42, 256, 789])
embedded_sequence = embedder.encode(token_sequence)
print(f"Embedded sequence shape: {embedded_sequence.shape}")
```

Slide 2: Self-Attention Mechanism

Self-attention enables the model to weigh the importance of different words in a sequence when encoding each word. This implementation shows the scaled dot-product attention computation, the fundamental building block of transformer architectures.

```python
def scaled_dot_product_attention(queries, keys, values, mask=None):
    # Compute attention scores
    d_k = queries.shape[-1]
    scores = np.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax to get attention weights
    attention_weights = softmax(scores)
    
    # Compute weighted sum of values
    return np.matmul(attention_weights, values), attention_weights

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Example usage
seq_len, d_k = 4, 64
queries = np.random.randn(1, seq_len, d_k)
keys = np.random.randn(1, seq_len, d_k)
values = np.random.randn(1, seq_len, d_k)

output, attention = scaled_dot_product_attention(queries, keys, values)
print(f"Attention output shape: {output.shape}")
print(f"Attention weights shape: {attention.shape}")
```

Slide 3: Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces. This implementation shows how to split the attention computation into multiple heads and combine their results.

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        self.wq = np.random.randn(d_model, d_model)
        self.wk = np.random.randn(d_model, d_model)
        self.wv = np.random.randn(d_model, d_model)
        self.dense = np.random.randn(d_model, d_model)
        
    def split_heads(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)
    
    def call(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections and split heads
        q = np.dot(query, self.wq)
        k = np.dot(key, self.wk)
        v = np.dot(value, self.wv)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # Scaled dot-product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        
        # Reshape and project output
        scaled_attention = scaled_attention.transpose(0, 2, 1, 3)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
        output = np.dot(concat_attention, self.dense)
        
        return output, attention_weights

# Example usage
batch_size, seq_len, d_model = 2, 4, 512
num_heads = 8

mha = MultiHeadAttention(d_model, num_heads)
sample_input = np.random.randn(batch_size, seq_len, d_model)
output, attention = mha.call(sample_input, sample_input, sample_input)
print(f"Multi-head attention output shape: {output.shape}")
```

Slide 4: Feed-Forward Neural Network

The feed-forward network in each transformer layer consists of two linear transformations with a ReLU activation in between. This component processes each position separately and identically, adding non-linearity to the model.

```python
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        # Initialize weights
        self.w1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.w2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        # First linear transformation
        hidden = np.dot(x, self.w1) + self.b1
        # ReLU activation
        hidden = self.relu(hidden)
        # Second linear transformation
        output = np.dot(hidden, self.w2) + self.b2
        return output

# Example usage
ff_network = FeedForward(d_model=512, d_ff=2048)
sample_input = np.random.randn(2, 4, 512)  # (batch_size, seq_len, d_model)
output = ff_network.forward(sample_input)
print(f"Feed-forward output shape: {output.shape}")
```

Slide 5: Layer Normalization

Layer normalization is crucial for stable training of deep transformer networks. This implementation shows how to normalize activations across the feature dimension while learning scale and shift parameters.

```python
class LayerNorm:
    def __init__(self, d_model, eps=1e-12):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
        
    def forward(self, x):
        # Calculate mean and variance along last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta

# Example usage
layer_norm = LayerNorm(d_model=512)
sample_input = np.random.randn(2, 4, 512)
normalized_output = layer_norm.forward(sample_input)
print(f"Layer norm output shape: {normalized_output.shape}")
print(f"Mean of normalized output: {np.mean(normalized_output):.6f}")
print(f"Std of normalized output: {np.std(normalized_output):.6f}")
```

Slide 6: Encoder Layer Implementation

The encoder layer combines multi-head attention, feed-forward networks, and layer normalization in a specific arrangement with residual connections to create a powerful feature extraction mechanism.

```python
class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)
        self.dropout_rate = dropout_rate
    
    def dropout(self, x):
        mask = np.random.binomial(1, 1-self.dropout_rate, x.shape)
        return x * mask / (1-self.dropout_rate)
    
    def forward(self, x, training=True, mask=None):
        # Multi-head attention
        attn_output, _ = self.mha.call(x, x, x, mask)
        if training:
            attn_output = self.dropout(attn_output)
        out1 = self.layernorm1.forward(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn.forward(out1)
        if training:
            ffn_output = self.dropout(ffn_output)
        out2 = self.layernorm2.forward(out1 + ffn_output)
        
        return out2

# Example usage
encoder_layer = EncoderLayer(d_model=512, num_heads=8, d_ff=2048)
sample_input = np.random.randn(2, 4, 512)
output = encoder_layer.forward(sample_input)
print(f"Encoder layer output shape: {output.shape}")
```

Slide 7: Decoder Layer Implementation

The decoder layer includes masked multi-head attention to prevent positions from attending to subsequent positions during training, followed by cross-attention to the encoder outputs and a feed-forward network.

```python
class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)
        self.layernorm3 = LayerNorm(d_model)
        self.dropout_rate = dropout_rate
    
    def forward(self, x, enc_output, training=True, look_ahead_mask=None, padding_mask=None):
        # Masked self-attention
        attn1, attn_weights_block1 = self.mha1.call(x, x, x, look_ahead_mask)
        if training:
            attn1 = self.dropout(attn1)
        out1 = self.layernorm1.forward(x + attn1)
        
        # Cross-attention
        attn2, attn_weights_block2 = self.mha2.call(
            out1, enc_output, enc_output, padding_mask)
        if training:
            attn2 = self.dropout(attn2)
        out2 = self.layernorm2.forward(out1 + attn2)
        
        # Feed-forward network
        ffn_output = self.ffn.forward(out2)
        if training:
            ffn_output = self.dropout(ffn_output)
        out3 = self.layernorm3.forward(out2 + ffn_output)
        
        return out3, attn_weights_block1, attn_weights_block2

# Example usage
decoder_layer = DecoderLayer(d_model=512, num_heads=8, d_ff=2048)
sample_input = np.random.randn(2, 4, 512)
sample_encoder_output = np.random.randn(2, 4, 512)
output, attn1, attn2 = decoder_layer.forward(sample_input, sample_encoder_output)
print(f"Decoder layer output shape: {output.shape}")
```

Slide 8: Creating the Full Transformer Model

A complete implementation of the Transformer architecture, combining multiple encoder and decoder layers with input/output embeddings and final linear projection layer for sequence generation tasks.

```python
class Transformer:
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, 
                 target_vocab_size, max_seq_length, dropout_rate=0.1):
        self.encoder_embedding = TransformerEmbedding(input_vocab_size, d_model)
        self.decoder_embedding = TransformerEmbedding(target_vocab_size, d_model)
        
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout_rate) 
                             for _ in range(num_layers)]
        self.decoder_layers = [DecoderLayer(d_model, num_heads, d_ff, dropout_rate) 
                             for _ in range(num_layers)]
        
        self.final_layer = np.random.randn(d_model, target_vocab_size) * 0.02
        
    def encode(self, inp, training=True):
        seq_len = inp.shape[1]
        
        # Embedding
        x = self.encoder_embedding.encode(inp, seq_len)
        
        # Encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer.forward(x, training=training)
            
        return x
    
    def decode(self, tar, enc_output, training=True, look_ahead_mask=None):
        seq_len = tar.shape[1]
        attention_weights = {}
        
        # Embedding
        x = self.decoder_embedding.encode(tar, seq_len)
        
        # Decoder layers
        for i, decoder_layer in enumerate(self.decoder_layers):
            x, block1, block2 = decoder_layer.forward(
                x, enc_output, training=training,
                look_ahead_mask=look_ahead_mask)
            
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
        
        # Final linear projection
        x = np.dot(x, self.final_layer)
        return x, attention_weights

# Example usage
transformer = Transformer(
    num_layers=6,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    input_vocab_size=8500,
    target_vocab_size=8500,
    max_seq_length=100
)

# Sample input sequences
inp_sequence = np.random.randint(0, 8500, size=(2, 20))  # (batch_size, seq_len)
tar_sequence = np.random.randint(0, 8500, size=(2, 15))

# Forward pass
enc_output = transformer.encode(inp_sequence)
dec_output, attention_weights = transformer.decode(tar_sequence, enc_output)

print(f"Encoder output shape: {enc_output.shape}")
print(f"Decoder output shape: {dec_output.shape}")
```

Slide 9: Loss Function and Training Logic

Implementation of the training procedure for the Transformer model, including loss calculation with label smoothing and masking of padding tokens.

```python
class TransformerTrainer:
    def __init__(self, transformer, learning_rate=0.0001, label_smoothing=0.1):
        self.transformer = transformer
        self.learning_rate = learning_rate
        self.label_smoothing = label_smoothing
    
    def loss_function(self, real, pred, pad_token=0):
        # Create mask for non-padding tokens
        mask = (real != pad_token).astype(np.float32)
        
        # Apply label smoothing
        num_classes = pred.shape[-1]
        smooth_positives = 1.0 - self.label_smoothing
        smooth_negatives = self.label_smoothing / num_classes
        
        # Create smoothed target distribution
        onehot = np.eye(num_classes)[real]
        smoothed_targets = onehot * smooth_positives + smooth_negatives
        
        # Calculate cross entropy loss
        log_probs = -np.log(softmax(pred))
        loss = np.sum(smoothed_targets * log_probs, axis=-1)
        
        # Apply mask and calculate mean
        masked_loss = loss * mask
        return np.sum(masked_loss) / np.sum(mask)
    
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        # Create masks
        look_ahead_mask = self.create_look_ahead_mask(tar_inp.shape[1])
        
        # Forward pass
        enc_output = self.transformer.encode(inp, training=True)
        dec_output, _ = self.transformer.decode(
            tar_inp, enc_output, training=True,
            look_ahead_mask=look_ahead_mask)
        
        # Calculate loss
        loss = self.loss_function(tar_real, dec_output)
        return loss
    
    @staticmethod
    def create_look_ahead_mask(size):
        mask = 1 - np.triu(np.ones((size, size)), k=1)
        return mask

# Example usage
trainer = TransformerTrainer(transformer)

# Sample batch
inp_batch = np.random.randint(0, 8500, size=(32, 20))
tar_batch = np.random.randint(0, 8500, size=(32, 21))

# Training step
loss = trainer.train_step(inp_batch, tar_batch)
print(f"Training loss: {loss:.4f}")
```

Slide 10: Machine Translation Implementation

A practical implementation of a machine translation system using the Transformer architecture, including preprocessing, training, and inference pipelines for English to French translation.

```python
class TranslationSystem:
    def __init__(self, model_params, src_vocab, tgt_vocab):
        self.transformer = Transformer(
            num_layers=model_params['num_layers'],
            d_model=model_params['d_model'],
            num_heads=model_params['num_heads'],
            d_ff=model_params['d_ff'],
            input_vocab_size=len(src_vocab),
            target_vocab_size=len(tgt_vocab),
            max_seq_length=model_params['max_seq_length']
        )
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
    def preprocess_sentence(self, sentence):
        # Tokenization and vocabulary mapping
        tokens = sentence.lower().split()
        token_ids = [self.src_vocab.get(token, self.src_vocab['<unk>']) 
                    for token in tokens]
        return np.array([self.src_vocab['<start>']] + token_ids + 
                       [self.src_vocab['<end>']])
    
    def translate(self, sentence, max_length=50):
        # Preprocess input sentence
        input_seq = self.preprocess_sentence(sentence)
        input_seq = input_seq.reshape(1, -1)  # Add batch dimension
        
        # Initialize target sequence
        output_seq = [self.tgt_vocab['<start>']]
        
        # Encode input sequence
        enc_output = self.transformer.encode(input_seq, training=False)
        
        for i in range(max_length):
            tar_seq = np.array([output_seq])
            
            # Decode current sequence
            predictions, _ = self.transformer.decode(
                tar_seq, enc_output, training=False)
            
            # Get next token prediction
            predicted_id = np.argmax(predictions[0, -1, :])
            output_seq.append(predicted_id)
            
            # Break if end token is predicted
            if predicted_id == self.tgt_vocab['<end>']:
                break
        
        # Convert ids to tokens
        id_to_word = {v: k for k, v in self.tgt_vocab.items()}
        return ' '.join([id_to_word[id] for id in output_seq[1:-1]])

# Example usage
model_params = {
    'num_layers': 6,
    'd_model': 512,
    'num_heads': 8,
    'd_ff': 2048,
    'max_seq_length': 100
}

# Sample vocabularies (in practice, these would be much larger)
src_vocab = {'<start>': 0, '<end>': 1, '<unk>': 2, 'hello': 3, 'world': 4}
tgt_vocab = {'<start>': 0, '<end>': 1, '<unk>': 2, 'bonjour': 3, 'monde': 4}

translation_system = TranslationSystem(model_params, src_vocab, tgt_vocab)
translated = translation_system.translate("hello world")
print(f"Translation: {translated}")
```

Slide 11: Attention Visualization

Implementation of attention visualization tools to understand how the Transformer model attends to different parts of the input sequence during translation.

```python
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionVisualizer:
    def __init__(self, translation_system):
        self.translation_system = translation_system
        
    def get_attention_weights(self, sentence):
        # Get token ids and perform translation
        input_seq = self.translation_system.preprocess_sentence(sentence)
        input_seq = input_seq.reshape(1, -1)
        
        # Get encoder output and attention weights
        enc_output = self.translation_system.transformer.encode(input_seq)
        dec_input = np.array([[self.translation_system.tgt_vocab['<start>']]])
        
        _, attention_weights = self.translation_system.transformer.decode(
            dec_input, enc_output)
        
        return attention_weights
    
    def plot_attention_heatmap(self, sentence, layer_name='decoder_layer6_block2'):
        attention_weights = self.get_attention_weights(sentence)
        att_matrix = attention_weights[layer_name][0]  # Get weights for first head
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            att_matrix,
            xticklabels=sentence.split(),
            yticklabels=sentence.split(),
            cmap='viridis'
        )
        plt.title(f'Attention weights for {layer_name}')
        plt.xlabel('Input sequence')
        plt.ylabel('Output sequence')
        
        return plt.gcf()

# Example usage
visualizer = AttentionVisualizer(translation_system)
sample_sentence = "hello world"
attention_plot = visualizer.plot_attention_heatmap(sample_sentence)
print("Attention heatmap generated successfully")
```

Slide 12: Real-world Application: Document Summarization

Implementation of a document summarization system using the Transformer architecture, demonstrating practical usage for text processing tasks.

```python
class DocumentSummarizer:
    def __init__(self, model_params, vocab_size=50000, max_document_length=1000,
                 max_summary_length=100):
        self.transformer = Transformer(
            num_layers=model_params['num_layers'],
            d_model=model_params['d_model'],
            num_heads=model_params['num_heads'],
            d_ff=model_params['d_ff'],
            input_vocab_size=vocab_size,
            target_vocab_size=vocab_size,
            max_seq_length=max_document_length
        )
        self.max_summary_length = max_summary_length
        
    def preprocess_document(self, document):
        # Implement tokenization and preprocessing logic
        # This is a simplified version
        words = document.lower().split()
        return ' '.join(words[:self.max_document_length])
    
    def generate_summary(self, document):
        # Preprocess document
        processed_doc = self.preprocess_document(document)
        
        # Convert to model input format
        input_seq = np.array([processed_doc])  # Simplified tokenization
        
        # Generate summary tokens
        enc_output = self.transformer.encode(input_seq, training=False)
        summary_tokens = []
        
        for i in range(self.max_summary_length):
            tar_seq = np.array([summary_tokens])
            predictions, _ = self.transformer.decode(
                tar_seq, enc_output, training=False)
            
            predicted_token = np.argmax(predictions[0, -1, :])
            summary_tokens.append(predicted_token)
            
            if predicted_token == self.vocab['<end>']:
                break
        
        # Convert summary tokens to text
        return self.tokens_to_text(summary_tokens)
    
    def tokens_to_text(self, tokens):
        # Convert token IDs back to text (simplified)
        return ' '.join([self.id_to_word.get(token, '<unk>') for token in tokens])

# Example usage
summarizer_params = {
    'num_layers': 6,
    'd_model': 512,
    'num_heads': 8,
    'd_ff': 2048
}

summarizer = DocumentSummarizer(summarizer_params)
sample_document = """
Long document text here...
"""
summary = summarizer.generate_summary(sample_document)
print(f"Generated summary: {summary}")
```

Slide 13: Position-wise Feed Forward Networks with Optimizations

Implementation of an optimized feed-forward network using matrix operations and advanced activation functions, including layer normalization and dropout for better training stability.

```python
class EnhancedFeedForward:
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        self.d_model = d_model
        self.d_ff = d_ff
        # Xavier/Glorot initialization
        self.w1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / (d_model + d_ff))
        self.w2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / (d_model + d_ff))
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
        self.dropout_rate = dropout_rate
        self.layer_norm = LayerNorm(d_model)
    
    def gelu(self, x):
        # Gaussian Error Linear Unit activation
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * 
                                    (x + 0.044715 * np.power(x, 3))))
    
    def forward(self, x, training=True):
        # First linear transformation with GELU activation
        hidden = np.dot(x, self.w1) + self.b1
        hidden = self.gelu(hidden)
        
        if training:
            # Apply dropout with scaling
            mask = np.random.binomial(1, 1-self.dropout_rate, hidden.shape)
            hidden = hidden * mask / (1-self.dropout_rate)
        
        # Second linear transformation
        output = np.dot(hidden, self.w2) + self.b2
        
        # Residual connection and layer normalization
        output = self.layer_norm.forward(x + output)
        return output

# Example usage with performance metrics
def measure_performance(ff_network, input_data, num_iterations=100):
    import time
    
    total_time = 0
    for _ in range(num_iterations):
        start_time = time.time()
        output = ff_network.forward(input_data)
        total_time += time.time() - start_time
    
    avg_time = total_time / num_iterations
    return {
        'average_time': avg_time,
        'throughput': input_data.size / avg_time,
        'output_stats': {
            'mean': np.mean(output),
            'std': np.std(output),
            'max': np.max(output),
            'min': np.min(output)
        }
    }

# Performance testing
ff_network = EnhancedFeedForward(d_model=512, d_ff=2048)
test_input = np.random.randn(32, 100, 512)  # (batch_size, seq_len, d_model)
performance_metrics = measure_performance(ff_network, test_input)

print("Performance Metrics:")
print(f"Average processing time: {performance_metrics['average_time']:.4f} seconds")
print(f"Throughput: {performance_metrics['throughput']:.2f} values/second")
print("\nOutput Statistics:")
for key, value in performance_metrics['output_stats'].items():
    print(f"{key}: {value:.4f}")
```

Slide 14: Training Data Generation and Augmentation

Implementation of data generation and augmentation techniques specifically designed for transformer model training, including dynamic batching and curriculum learning.

```python
class TransformerDataGenerator:
    def __init__(self, vocab_size, max_seq_length, batch_size=32):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.difficulty_levels = {
            'easy': (5, 10),
            'medium': (10, 20),
            'hard': (20, max_seq_length)
        }
    
    def generate_sequence_pair(self, min_length, max_length):
        # Generate source sequence
        length = np.random.randint(min_length, max_length + 1)
        source = np.random.randint(1, self.vocab_size, size=length)
        
        # Generate target sequence with slight modifications
        target = np.copy(source)
        num_modifications = max(1, length // 10)
        mod_positions = np.random.choice(length, num_modifications, replace=False)
        target[mod_positions] = np.random.randint(1, self.vocab_size, size=num_modifications)
        
        return source, target
    
    def create_batch(self, difficulty='easy'):
        min_len, max_len = self.difficulty_levels[difficulty]
        sources = []
        targets = []
        
        for _ in range(self.batch_size):
            src, tgt = self.generate_sequence_pair(min_len, max_len)
            sources.append(src)
            targets.append(tgt)
        
        # Pad sequences to same length
        max_src_len = max(len(s) for s in sources)
        max_tgt_len = max(len(t) for t in targets)
        
        padded_sources = np.zeros((self.batch_size, max_src_len), dtype=np.int32)
        padded_targets = np.zeros((self.batch_size, max_tgt_len), dtype=np.int32)
        
        for i, (src, tgt) in enumerate(zip(sources, targets)):
            padded_sources[i, :len(src)] = src
            padded_targets[i, :len(tgt)] = tgt
        
        return padded_sources, padded_targets
    
    def augment_batch(self, source, target):
        # Implement various augmentation techniques
        augmented_source = np.copy(source)
        augmented_target = np.copy(target)
        
        # Random word dropout
        mask = np.random.rand(*source.shape) > 0.1
        augmented_source = augmented_source * mask
        
        # Random word replacement
        replace_mask = np.random.rand(*source.shape) > 0.9
        random_words = np.random.randint(1, self.vocab_size, size=source.shape)
        augmented_source = np.where(replace_mask, random_words, augmented_source)
        
        return augmented_source, augmented_target

# Example usage and validation
data_generator = TransformerDataGenerator(
    vocab_size=10000,
    max_seq_length=100,
    batch_size=32
)

# Generate batches of increasing difficulty
difficulties = ['easy', 'medium', 'hard']
for diff in difficulties:
    src_batch, tgt_batch = data_generator.create_batch(difficulty=diff)
    aug_src, aug_tgt = data_generator.augment_batch(src_batch, tgt_batch)
    
    print(f"\nDifficulty: {diff}")
    print(f"Source batch shape: {src_batch.shape}")
    print(f"Target batch shape: {tgt_batch.shape}")
    print(f"Average sequence length: {np.mean(np.sum(src_batch > 0, axis=1)):.1f}")
    print(f"Augmentation differences: {np.sum(src_batch != aug_src)}")
```

Slide 15: Additional Resources

1.  [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) - "Attention Is All You Need" - Original Transformer paper by Vaswani et al.
2.  [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
3.  [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165) - "Language Models are Few-Shot Learners" - GPT-3 paper exploring scaling laws
4.  [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692) - "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
5.  [https://arxiv.org/abs/2302.07842](https://arxiv.org/abs/2302.07842) - "Self-attention Does Not Need O(n²) Memory" - Advanced optimization techniques for Transformers

Slide 16: Real-world Performance Metrics

A comprehensive analysis of the implemented Transformer model's performance across different tasks and configurations.

```python
class TransformerMetrics:
    def __init__(self, transformer_model):
        self.model = transformer_model
        self.metrics_history = {
            'translation': [],
            'inference_time': [],
            'memory_usage': [],
            'attention_stats': []
        }
    
    def measure_translation_quality(self, source_texts, reference_translations):
        from nltk.translate.bleu_score import sentence_bleu
        scores = []
        
        for src, ref in zip(source_texts, reference_translations):
            # Generate translation
            pred = self.model.translate(src)
            
            # Calculate BLEU score
            ref_tokens = ref.split()
            pred_tokens = pred.split()
            score = sentence_bleu([ref_tokens], pred_tokens)
            scores.append(score)
            
        return {
            'mean_bleu': np.mean(scores),
            'std_bleu': np.std(scores),
            'min_bleu': np.min(scores),
            'max_bleu': np.max(scores)
        }
    
    def profile_inference_time(self, input_lengths=[10, 50, 100, 200]):
        timing_results = {}
        
        for length in input_lengths:
            # Generate sample input
            input_seq = np.random.randint(0, 1000, size=(1, length))
            
            # Measure inference time
            times = []
            for _ in range(10):
                start_time = time.time()
                _ = self.model.encode(input_seq)
                times.append(time.time() - start_time)
            
            timing_results[length] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'tokens_per_second': length / np.mean(times)
            }
        
        return timing_results
    
    def analyze_attention_patterns(self, text):
        # Get attention weights
        input_seq = self.model.preprocess_sentence(text)
        _, attention_weights = self.model.decode(input_seq)
        
        analysis = {}
        for layer_name, attention_matrix in attention_weights.items():
            analysis[layer_name] = {
                'entropy': self._calculate_attention_entropy(attention_matrix),
                'sparsity': self._calculate_sparsity(attention_matrix),
                'average_head_disagreement': self._calculate_head_disagreement(attention_matrix)
            }
        
        return analysis
    
    def _calculate_attention_entropy(self, attention_matrix):
        # Calculate entropy of attention distributions
        attention_matrix = np.clip(attention_matrix, 1e-10, 1)
        entropy = -np.sum(attention_matrix * np.log(attention_matrix), axis=-1)
        return np.mean(entropy)
    
    def _calculate_sparsity(self, attention_matrix):
        # Calculate proportion of near-zero attention weights
        threshold = 0.01
        return np.mean(attention_matrix < threshold)
    
    def _calculate_head_disagreement(self, attention_matrix):
        # Calculate average disagreement between attention heads
        num_heads = attention_matrix.shape[1]
        disagreement = 0
        for i in range(num_heads):
            for j in range(i+1, num_heads):
                disagreement += np.mean(np.abs(
                    attention_matrix[:, i] - attention_matrix[:, j]))
        return disagreement / (num_heads * (num_heads - 1) / 2)

# Example usage
metrics = TransformerMetrics(transformer)

# Sample performance evaluation
sample_sources = [
    "Hello world",
    "How are you doing?",
    "This is a test sentence"
]
sample_references = [
    "Bonjour le monde",
    "Comment allez-vous?",
    "C'est une phrase de test"
]

translation_metrics = metrics.measure_translation_quality(
    sample_sources, sample_references)
timing_metrics = metrics.profile_inference_time()
attention_analysis = metrics.analyze_attention_patterns(sample_sources[0])

print("\nTranslation Quality Metrics:")
for metric, value in translation_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nInference Timing Analysis:")
for length, metrics in timing_metrics.items():
    print(f"\nSequence length {length}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

print("\nAttention Pattern Analysis:")
for layer, analysis in attention_analysis.items():
    print(f"\n{layer}:")
    for metric, value in analysis.items():
        print(f"  {metric}: {value:.4f}")
```

