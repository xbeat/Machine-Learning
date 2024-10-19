## Transformer-Based Models for NLP Tasks

Slide 1: Introduction to Transformer-Based Models

Transformer-based models have revolutionized natural language processing (NLP), offering powerful tools for tasks like sentiment analysis. These models use self-attention mechanisms to capture long-range dependencies in text, enabling a deeper understanding of language context and nuances.

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        attended = self.attention(x, x, x)[0]
        x = self.norm1(x + attended)
        fedforward = self.feed_forward(x)
        return self.norm2(x + fedforward)

# Example usage
embed_dim, num_heads = 512, 8
transformer = TransformerBlock(embed_dim, num_heads)
input_tensor = torch.randn(10, 32, embed_dim)  # (seq_len, batch_size, embed_dim)
output = transformer(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
```

Slide 2: BERT (Bidirectional Encoder Representations from Transformers)

BERT introduced bidirectional training, allowing it to capture context from both sides of a word. This bidirectional nature significantly improved performance on various NLP tasks, including sentiment analysis. BERT is pre-trained on massive text data, making it versatile and powerful.

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Example text for sentiment analysis
text = "I love this movie! It's amazing."

# Tokenize and prepare input
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Get prediction
prediction = torch.argmax(outputs.logits, dim=1)
print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
```

Slide 3: RoBERTa (Robustly Optimized BERT Approach)

RoBERTa is a more robust version of BERT, fine-tuned using dynamic masking, larger batch sizes, and longer training times. These modifications led to improved performance, especially on tasks requiring deep language understanding. RoBERTa consistently outperforms BERT on various benchmarks, including sentiment analysis.

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load pre-trained RoBERTa model and tokenizer
model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Example text for sentiment analysis
text = "This product exceeded my expectations. Highly recommended!"

# Tokenize and prepare input
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Get prediction
prediction = torch.argmax(outputs.logits, dim=1)
print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
```

Slide 4: DistilBERT

DistilBERT is a smaller and faster version of BERT obtained through knowledge distillation. It is trained to mimic the behavior of a larger model while being significantly more efficient. DistilBERT's smaller size makes it suitable for applications with limited computational resources.

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load pre-trained DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Example text for sentiment analysis
text = "The customer service was terrible. I'm very disappointed."

# Tokenize and prepare input
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Get prediction
prediction = torch.argmax(outputs.logits, dim=1)
print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
```

Slide 5: ALBERT (A Lite BERT)

ALBERT is another smaller and faster version of BERT that focuses on reducing computational cost. It achieves this by using parameter sharing techniques and factorized embedding parameterization. ALBERT maintains good performance while significantly reducing model size and training time.

```python
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch

# Load pre-trained ALBERT model and tokenizer
model_name = 'albert-base-v2'
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(model_name)

# Example text for sentiment analysis
text = "The new software update is full of bugs. It's frustrating to use."

# Tokenize and prepare input
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Get prediction
prediction = torch.argmax(outputs.logits, dim=1)
print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
```

Slide 6: XLNet

XLNet addresses BERT's limitations by using a permutation language modeling objective. This allows XLNet to capture bidirectional context without relying on masking, leading to improved performance on various tasks. XLNet's unique approach enables it to better handle long-range dependencies and context.

```python
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import torch

# Load pre-trained XLNet model and tokenizer
model_name = 'xlnet-base-cased'
tokenizer = XLNetTokenizer.from_pretrained(model_name)
model = XLNetForSequenceClassification.from_pretrained(model_name)

# Example text for sentiment analysis
text = "The concert was absolutely phenomenal. The band's energy was contagious!"

# Tokenize and prepare input
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Get prediction
prediction = torch.argmax(outputs.logits, dim=1)
print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
```

Slide 7: Comparing Model Sizes

Let's compare the sizes of different transformer-based models to understand their relative computational requirements and potential applications.

```python
from transformers import (BertModel, RobertaModel, DistilBertModel,
                          AlbertModel, XLNetModel)

def get_model_size(model):
    return sum(p.numel() for p in model.parameters()) / 1_000_000

models = {
    "BERT": BertModel.from_pretrained('bert-base-uncased'),
    "RoBERTa": RobertaModel.from_pretrained('roberta-base'),
    "DistilBERT": DistilBertModel.from_pretrained('distilbert-base-uncased'),
    "ALBERT": AlbertModel.from_pretrained('albert-base-v2'),
    "XLNet": XLNetModel.from_pretrained('xlnet-base-cased')
}

for name, model in models.items():
    print(f"{name} size: {get_model_size(model):.2f} million parameters")
```

Slide 8: Fine-tuning for Sentiment Analysis

Fine-tuning a pre-trained model for sentiment analysis allows us to adapt it to specific tasks or domains. Here's an example of fine-tuning BERT for sentiment analysis on a custom dataset.

```python
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Prepare your dataset (example)
texts = ["I love this!", "This is terrible", "Amazing product", "Disappointing experience"]
labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# Tokenize and encode the dataset
encodings = tokenizer(texts, truncation=True, padding=True)
input_ids = torch.tensor(encodings['input_ids'])
attention_mask = torch.tensor(encodings['attention_mask'])
labels = torch.tensor(labels)

# Create DataLoader
dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Fine-tuning setup
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# Fine-tuning loop (example: 3 epochs)
model.train()
for epoch in range(3):
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed")

print("Fine-tuning completed")
```

Slide 9: Attention Visualization

Visualizing attention weights can help us understand how transformer models focus on different parts of the input. Let's create a simple attention visualization for a single-headed attention mechanism.

```python
import numpy as np
import matplotlib.pyplot as plt

def attention_visualization(query, key, value):
    attention_weights = np.dot(query, key.T) / np.sqrt(key.shape[1])
    attention_weights = np.exp(attention_weights) / np.sum(np.exp(attention_weights), axis=1, keepdims=True)
    output = np.dot(attention_weights, value)
    return attention_weights, output

# Example input
query = np.random.randn(1, 64)
key = np.random.randn(5, 64)
value = np.random.randn(5, 64)

attention_weights, _ = attention_visualization(query, key, value)

plt.figure(figsize=(10, 2))
plt.imshow(attention_weights, cmap='viridis')
plt.colorbar()
plt.title('Attention Weights Visualization')
plt.xlabel('Key/Value tokens')
plt.ylabel('Query tokens')
plt.show()
```

Slide 10: Transfer Learning with Transformer Models

Transfer learning allows us to leverage pre-trained models for new tasks. Let's demonstrate how to use a pre-trained BERT model for text classification on a custom dataset.

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Prepare your dataset (example)
texts = ["This is great!", "I'm neutral about this", "Terrible experience", 
         "Absolutely fantastic", "Not sure how I feel", "Worst product ever"]
labels = [2, 1, 0, 2, 1, 0]  # 2: positive, 1: neutral, 0: negative

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# Tokenize and encode the dataset
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Create DataLoaders
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
                              torch.tensor(train_encodings['attention_mask']),
                              torch.tensor(train_labels))
val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']),
                            torch.tensor(val_encodings['attention_mask']),
                            torch.tensor(val_labels))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# Fine-tuning loop (example: 3 epochs)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    print(f"Epoch {epoch+1} - Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {correct/total:.4f}")

print("Transfer learning completed")
```

Slide 11: Real-life Example: Sentiment Analysis for Product Reviews

Let's use a pre-trained BERT model to analyze sentiment in product reviews, a common application in e-commerce platforms.

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Example product reviews
reviews = [
    "This smartphone is amazing! The camera quality is outstanding.",
    "The laptop's performance is average, but it's good for basic tasks.",
    "Terrible headphones. They broke after just a week of use."
]

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)
    return ["Negative", "Neutral", "Positive"][prediction.item()]

# Analyze sentiments
for review in reviews:
    sentiment = predict_sentiment(review)
    print(f"Review: '{review}'")
    print(f"Sentiment: {sentiment}\n")
```

Slide 12: Real-life Example: Language Translation

Transformer models excel in machine translation tasks. Let's demonstrate a simple translation from English to French using a pre-trained model.

```python
from transformers import MarianMTModel, MarianTokenizer

# Load pre-trained model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# English sentences to translate
english_texts = [
    "Hello, how are you?",
    "The weather is beautiful today.",
    "I love learning about artificial intelligence."
]

# Translate function
def translate_to_french(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Perform translations
for text in english_texts:
    french_text = translate_to_french(text)
    print(f"English: {text}")
    print(f"French: {french_text}\n")
```

Slide 13: Transformer Architecture Visualization

Understanding the architecture of transformer models is crucial. Let's create a simple visualization of a transformer block.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_transformer_block():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Input
    ax.text(1, 9.5, "Input", fontweight='bold')
    ax.add_patch(plt.Rectangle((0.5, 8.5), 2, 0.5, fill=False))

    # Multi-Head Attention
    ax.text(3.5, 7.5, "Multi-Head\nAttention", fontweight='bold')
    ax.add_patch(plt.Rectangle((3, 6.5), 2, 1.5, fill=False))

    # Add & Norm
    ax.text(5.5, 7.5, "Add & Norm", fontweight='bold')
    ax.add_patch(plt.Rectangle((5, 7), 2, 0.5, fill=False))

    # Feed Forward
    ax.text(7.5, 5.5, "Feed\nForward", fontweight='bold')
    ax.add_patch(plt.Rectangle((7, 4.5), 2, 1.5, fill=False))

    # Add & Norm
    ax.text(5.5, 3.5, "Add & Norm", fontweight='bold')
    ax.add_patch(plt.Rectangle((5, 3), 2, 0.5, fill=False))

    # Output
    ax.text(1, 1.5, "Output", fontweight='bold')
    ax.add_patch(plt.Rectangle((0.5, 0.5), 2, 0.5, fill=False))

    # Arrows
    ax.arrow(1.5, 8.5, 0, -1, head_width=0.2, head_length=0.2, fc='k', ec='k')
    ax.arrow(4, 6.5, 0, -2, head_width=0.2, head_length=0.2, fc='k', ec='k')
    ax.arrow(6, 7, 0, -2, head_width=0.2, head_length=0.2, fc='k', ec='k')
    ax.arrow(8, 4.5, -2, -1, head_width=0.2, head_length=0.2, fc='k', ec='k')
    ax.arrow(6, 3, -4.5, -2, head_width=0.2, head_length=0.2, fc='k', ec='k')

    plt.title("Transformer Block Architecture")
    plt.tight_layout()
    plt.show()

plot_transformer_block()
```

Slide 14: Comparing Model Performance

Let's compare the performance of different transformer models on a sentiment analysis task using a simple benchmark dataset.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, f1_score
import time

# Sample dataset (you would typically use a larger, real-world dataset)
texts = [
    "I absolutely love this product!",
    "It's okay, nothing special.",
    "Terrible experience, would not recommend.",
    "Exceeded my expectations in every way.",
    "Neutral feelings about this one."
]
labels = [1, 0, -1, 1, 0]  # 1: positive, 0: neutral, -1: negative

# Models to compare
models = ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased']

def evaluate_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Tokenize and prepare input
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.time() - start_time

    # Get predictions
    predictions = torch.argmax(outputs.logits, dim=1).numpy()
    
    # Map predictions to match label format
    pred_mapped = [1 if p == 2 else (-1 if p == 0 else 0) for p in predictions]

    # Calculate metrics
    accuracy = accuracy_score(labels, pred_mapped)
    f1 = f1_score(labels, pred_mapped, average='weighted')

    return accuracy, f1, inference_time

# Evaluate each model
results = {}
for model_name in models:
    accuracy, f1, inference_time = evaluate_model(model_name)
    results[model_name] = {
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Inference Time': inference_time
    }

# Print results
for model, metrics in results.items():
    print(f"\n{model}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
```

Slide 15: Additional Resources

For further exploration of transformer-based models, consider these resources:

1.  "Attention Is All You Need" - The original transformer paper: ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2.  "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding": ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3.  "RoBERTa: A Robustly Optimized BERT Pretraining Approach": ArXiv: [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)
4.  "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter": ArXiv: [https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)
5.  "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations": ArXiv: [https://arxiv.org/abs/1909.11942](https://arxiv.org/abs/1909.11942)
6.  "XLNet: Generalized Autoregressive Pretraining for Language Understanding": ArXiv: [https://arxiv.org/abs/1906.08237](https://arxiv.org/abs/1906.08237)

These papers provide in-depth explanations of the models discussed in this presentation and offer insights into their architectures, training procedures, and performance characteristics.

