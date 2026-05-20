## Mastering BERT Transforming NLP with Bidirectional Understanding
Slide 1: Understanding BERT Architecture

BERT (Bidirectional Encoder Representations from Transformers) revolutionized natural language processing by introducing true bidirectional understanding through its novel transformer-based architecture. The model processes text sequences using self-attention mechanisms to capture contextual relationships between words in both directions simultaneously.

```python
import torch
import torch.nn as nn

class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(512, hidden_size)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # Generate position indices
        positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
        # Combine token and position embeddings
        x = self.embedding(x) + self.position_embedding(positions)
        
        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        return x
```

Slide 2: Self-Attention Mechanism

The self-attention mechanism is the core component of BERT, allowing it to weigh the importance of different words in relation to each other. This mechanism computes attention scores between all pairs of words in the input sequence through query, key, and value transformations.

```python
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Split heads
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        return self.proj(out)
```

Slide 3: Masked Language Modeling

Masked Language Modeling (MLM) is BERT's primary pre-training objective where random tokens in the input sequence are masked, and the model learns to predict these masked tokens based on their bidirectional context. This training approach enables deep bidirectional representations.

```python
class MLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = torch.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

def create_mlm_predictions(input_ids, tokenizer, mask_prob=0.15):
    masked_inputs = input_ids.clone()
    labels = input_ids.clone()
    
    # Create random mask
    probability_matrix = torch.full(input_ids.shape, mask_prob)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # Replace masked tokens with [MASK]
    masked_inputs[masked_indices] = tokenizer.mask_token_id
    
    # Set labels for non-masked tokens to -100 (ignore)
    labels[~masked_indices] = -100
    
    return masked_inputs, labels
```

Slide 4: Next Sentence Prediction

Next Sentence Prediction (NSP) is BERT's secondary pre-training objective that helps the model understand relationships between sentences. The model learns to predict whether two sentences naturally follow each other or are randomly paired.

```python
class NSPHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(512, 2)
        )
        
    def forward(self, pooled_output):
        return self.classifier(pooled_output)

def create_nsp_examples(text_pairs, tokenizer, max_length=512):
    features = []
    for (text_a, text_b), is_next in text_pairs:
        # Tokenize and combine sentences
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)
        
        # Truncate if necessary
        while len(tokens_a) + len(tokens_b) > max_length - 3:
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        
        # Create input sequence
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        
        # Convert to ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        features.append((input_ids, segment_ids, is_next))
        
    return features
```

Slide 5: Fine-tuning BERT for Text Classification

The true power of BERT lies in its ability to be fine-tuned for specific downstream tasks. Here we implement a text classification model by adding a classification head on top of the pre-trained BERT model.

```python
class BERTForClassification(nn.Module):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]  # Use [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def train_classifier(model, train_dataloader, optimizer, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
```

Slide 6: BERT for Named Entity Recognition

Named Entity Recognition (NER) with BERT involves token-level classification to identify and categorize entities in text. The model processes each token in the sequence and assigns it to predefined entity categories like person, organization, or location.

```python
class BERTForNER(nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]  # Get all token representations
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

def process_ner_batch(batch, model, label_map):
    model.eval()
    with torch.no_grad():
        outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'])
        predictions = torch.argmax(outputs, dim=2)
        
    true_predictions = [
        [label_map[p.item()] for (p, m) in zip(pred, mask) if m.item() == 1]
        for pred, mask in zip(predictions, batch['attention_mask'])
    ]
    return true_predictions
```

Slide 7: Implementing BERT for Question Answering

BERT excels at question answering tasks by predicting start and end positions of answer spans within a given context. This implementation shows how to create a question answering model using BERT's contextual representations.

```python
class BERTForQuestionAnswering(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.qa_outputs = nn.Linear(bert_model.config.hidden_size, 2)  # start/end
        
    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss, start_logits, end_logits
        
        return start_logits, end_logits

def get_answer_span(start_logits, end_logits, tokens, max_answer_length=30):
    # Get the most likely start and end positions
    start_probs = torch.softmax(start_logits, dim=-1)
    end_probs = torch.softmax(end_logits, dim=-1)
    
    # Find the best answer span
    max_prob = -float('inf')
    best_start, best_end = 0, 0
    
    for start_idx in range(len(tokens)):
        for end_idx in range(start_idx, min(start_idx + max_answer_length, len(tokens))):
            prob = start_probs[start_idx] * end_probs[end_idx]
            if prob > max_prob:
                max_prob = prob
                best_start = start_idx
                best_end = end_idx
    
    return ' '.join(tokens[best_start:best_end + 1])
```

Slide 8: Text Summarization with BERT

BERT can be adapted for extractive summarization by scoring and selecting the most important sentences from a document. This implementation demonstrates how to create a BERT-based extractive summarizer.

```python
class BERTForSummarization(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]  # [CLS] token representation
        scores = self.classifier(pooled_output)
        return scores

def extractive_summarize(text, model, tokenizer, num_sentences=3):
    # Split text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Prepare inputs for each sentence
    inputs = [tokenizer(sent, return_tensors='pt', padding=True, truncation=True) 
             for sent in sentences]
    
    # Get importance scores
    scores = []
    model.eval()
    with torch.no_grad():
        for inp in inputs:
            score = model(inp['input_ids'], attention_mask=inp['attention_mask'])
            scores.append(score.item())
    
    # Select top sentences
    ranked_sentences = [sent for _, sent in sorted(
        zip(scores, sentences), reverse=True
    )]
    
    return ' '.join(ranked_sentences[:num_sentences])
```

Slide 9: BERT Position Embeddings Implementation

Position embeddings are crucial for BERT to understand the sequential nature of text input. This implementation shows how BERT combines token embeddings with learned position embeddings to create rich input representations.

```python
class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings=512):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = words_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

# Example usage
def create_embeddings_example():
    vocab_size = 30522  # BERT vocabulary size
    hidden_size = 768
    batch_size = 4
    seq_length = 128
    
    embedder = BERTEmbeddings(vocab_size, hidden_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    embeddings = embedder(input_ids)
    
    return embeddings.shape  # Expected: [batch_size, seq_length, hidden_size]
```

Slide 10: Implementing Multi-head Attention

Multi-head attention allows BERT to jointly attend to information from different representation subspaces. This implementation demonstrates the parallel processing of attention through multiple heads.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=12, dropout=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        
    def transpose_for_scores(self, x):
        batch_size = x.size(0)
        new_shape = (batch_size, -1, self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        output = self.dense(context_layer)
        return output
```

Slide 11: BERT for Sentiment Analysis

This implementation shows how to fine-tune BERT for sentiment analysis tasks, including handling multiple sentiment classes and processing text sequences effectively.

```python
class BERTForSentiment(nn.Module):
    def __init__(self, bert_model, num_labels=3):  # 3 classes: negative, neutral, positive
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(bert_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def train_sentiment_classifier(model, train_dataloader, optimizer, num_epochs=3):
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
            labels = batch['labels']
            
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
```

Slide 12: BERT for Sequence Tagging

BERT's token-level representations make it particularly effective for sequence tagging tasks like part-of-speech tagging and chunking. This implementation shows how to process sequential data and make token-wise predictions.

```python
class BERTForSequenceTagging(nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool())
            return loss, emissions
        return emissions
    
    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)

def train_sequence_tagger(model, train_dataloader, optimizer, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            loss, _ = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
```

Slide 13: Implementing BERT Loss Functions

Understanding and implementing BERT's specialized loss functions is crucial for effective training. This implementation covers both the masked language modeling and next sentence prediction loss calculations.

```python
class BERTLoss(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.nsp_loss = nn.CrossEntropyLoss()
        
    def forward(self, mlm_logits, mlm_labels, nsp_logits, nsp_labels):
        mlm_loss = self.mlm_loss(
            mlm_logits.view(-1, mlm_logits.size(-1)),
            mlm_labels.view(-1)
        )
        
        nsp_loss = self.nsp_loss(nsp_logits, nsp_labels)
        
        # Combined loss with weighting
        total_loss = mlm_loss + 0.5 * nsp_loss
        
        return {
            'total_loss': total_loss,
            'mlm_loss': mlm_loss,
            'nsp_loss': nsp_loss
        }

def calculate_metrics(predictions, labels):
    mlm_preds = torch.argmax(predictions['mlm_logits'], dim=-1)
    mlm_accuracy = (mlm_preds == labels['mlm_labels']).float().mean()
    
    nsp_preds = torch.argmax(predictions['nsp_logits'], dim=-1)
    nsp_accuracy = (nsp_preds == labels['nsp_labels']).float().mean()
    
    return {
        'mlm_accuracy': mlm_accuracy.item(),
        'nsp_accuracy': nsp_accuracy.item()
    }
```

Slide 14: Additional Resources

*   ArXiv:1810.04805 - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
*   ArXiv:2002.12327 - A Primer in BERTology: What We Know About How BERT Works [https://arxiv.org/abs/2002.12327](https://arxiv.org/abs/2002.12327)
*   ArXiv:1907.11692 - RoBERTa: A Robustly Optimized BERT Pretraining Approach [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)
*   For more resources and implementation details:
    *   Google's official BERT repository: [https://github.com/google-research/bert](https://github.com/google-research/bert)
    *   Hugging Face Transformers documentation: [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)
    *   PyTorch documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

