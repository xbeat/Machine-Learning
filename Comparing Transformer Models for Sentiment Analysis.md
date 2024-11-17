## Comparing Transformer Models for Sentiment Analysis
Slide 1: Transformer Architecture for Sentiment Analysis

The Transformer architecture revolutionized NLP by introducing self-attention mechanisms that capture contextual relationships between words. This fundamental architecture serves as the backbone for modern sentiment analysis models, enabling them to understand complex emotional nuances in text data through parallel processing of sequences.

```python
import torch
import torch.nn as nn

class TransformerSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        # Global average pooling
        x = torch.mean(x, dim=1)
        return self.classifier(x)

# Example usage
model = TransformerSentiment(
    vocab_size=10000,
    embed_dim=256,
    num_heads=8,
    num_classes=3  # Positive, Negative, Neutral
)
```

Slide 2: BERT Implementation for Sentiment Classification

BERT's bidirectional context understanding makes it particularly effective for sentiment analysis. This implementation demonstrates how to fine-tune a pre-trained BERT model for sentiment classification tasks, incorporating attention masking and special tokens for optimal performance.

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class BERTSentiment:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=3  # Positive, Negative, Neutral
        )
    
    def predict(self, text):
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        outputs = self.model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        return predictions

# Example usage
classifier = BERTSentiment()
text = "This product exceeded my expectations!"
sentiment = classifier.predict(text)
```

Slide 3: Data Preprocessing for Transformer Models

Effective preprocessing is crucial for transformer-based sentiment analysis. This implementation showcases essential preprocessing steps including tokenization, sequence padding, attention masking, and handling of special tokens required by transformer architectures.

```python
import torch
from transformers import BertTokenizer
import pandas as pd

class SentimentPreprocessor:
    def __init__(self, max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
    
    def preprocess(self, texts):
        encoded = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }

# Example usage
preprocessor = SentimentPreprocessor()
texts = ["Great product!", "Terrible service", "Average experience"]
inputs = preprocessor.preprocess(texts)
```

Slide 4: Custom Dataset Implementation

A robust dataset implementation is essential for training transformer models. This code demonstrates how to create a custom PyTorch dataset for sentiment analysis, handling text-label pairs and incorporating proper preprocessing steps.

```python
import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Preprocess single instance
        inputs = self.preprocessor.preprocess([text])
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Example usage
texts = ["Amazing!", "Disappointed", "Not bad"]
labels = [2, 0, 1]  # Positive, Negative, Neutral
dataset = SentimentDataset(texts, labels, preprocessor)
```

Slide 5: Training Loop Implementation

The training loop for transformer-based sentiment analysis requires careful handling of attention masks and gradient optimization. This implementation shows a complete training cycle with learning rate scheduling, gradient clipping, and proper device management.

```python
def train_sentiment_model(model, train_loader, num_epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = criterion(outputs.logits, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
```

Slide 6: Real-world Example - Twitter Sentiment Analysis

This implementation demonstrates how to analyze Twitter data using a fine-tuned BERT model. The code handles the specific challenges of social media text, including emoji processing, hashtag normalization, and handling of user mentions.

```python
import re
from transformers import pipeline

class TwitterSentimentAnalyzer:
    def __init__(self):
        self.analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        
    def preprocess_tweet(self, tweet):
        # Remove URLs
        tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)
        # Remove user mentions
        tweet = re.sub(r'@\w+', '', tweet)
        # Normalize hashtags
        tweet = re.sub(r'#(\w+)', r'\1', tweet)
        # Remove extra whitespace
        tweet = ' '.join(tweet.split())
        return tweet
    
    def analyze_tweets(self, tweets):
        processed_tweets = [self.preprocess_tweet(tweet) for tweet in tweets]
        results = self.analyzer(processed_tweets)
        
        # Convert to user-friendly format
        sentiments = []
        for result in results:
            label = result['label']
            score = result['score']
            sentiment = {
                '1 star': 'Very Negative',
                '2 stars': 'Negative',
                '3 stars': 'Neutral',
                '4 stars': 'Positive',
                '5 stars': 'Very Positive'
            }[label]
            sentiments.append({
                'sentiment': sentiment,
                'confidence': score
            })
        return sentiments

# Example usage
analyzer = TwitterSentimentAnalyzer()
tweets = [
    "Loving the new features! #awesome",
    "Service has been terrible lately @company",
    "Mixed feelings about this update..."
]
results = analyzer.analyze_tweets(tweets)
```

Slide 7: Results Analysis and Visualization

This implementation provides comprehensive visualization and analysis tools for sentiment classification results, including confusion matrices, ROC curves, and confidence distribution plots using seaborn and matplotlib.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

class SentimentAnalysisVisualizer:
    def __init__(self, true_labels, predicted_labels, confidence_scores):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.confidence_scores = confidence_scores
        
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
    def plot_roc_curves(self):
        plt.figure(figsize=(10, 8))
        for i in range(3):  # For each class
            fpr, tpr, _ = roc_curve(
                (self.true_labels == i).astype(int),
                self.confidence_scores[:, i]
            )
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, 
                tpr, 
                label=f'Class {i} (AUC = {roc_auc:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.show()

# Example usage
visualizer = SentimentAnalysisVisualizer(
    true_labels=np.array([0, 1, 2, 1, 0]),
    predicted_labels=np.array([0, 1, 2, 2, 1]),
    confidence_scores=np.random.rand(5, 3)
)
visualizer.plot_confusion_matrix()
visualizer.plot_roc_curves()
```

Slide 8: Self-Attention Implementation

Self-attention mechanisms are fundamental to transformer-based sentiment analysis, allowing models to weigh the importance of different words contextually. This implementation shows a from-scratch approach to computing attention scores and weighted representations.

```python
import torch
import torch.nn.functional as F
import numpy as np

class SelfAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.k_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.v_linear = torch.nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear transformations
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        
        # Reshape and combine heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.embed_dim)
        
        return attention_output, attention_weights

# Example usage
attention = SelfAttention(embed_dim=256, num_heads=8)
x = torch.randn(32, 50, 256)  # batch_size=32, seq_len=50, embed_dim=256
output, weights = attention(x)
```

Slide 9: RoBERTa Fine-tuning Implementation

RoBERTa improves upon BERT through robust optimization techniques. This implementation demonstrates how to fine-tune a RoBERTa model for sentiment analysis with advanced training strategies including gradient accumulation and mixed precision training.

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from torch.cuda.amp import autocast, GradScaler

class RoBERTaSentimentTrainer:
    def __init__(self, model_name='roberta-base', num_labels=3):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.scaler = GradScaler()
        
    def train_step(self, batch, accumulation_steps=4):
        self.model.train()
        
        # Enable automatic mixed precision
        with autocast():
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss / accumulation_steps
        
        # Scale loss and compute gradients
        self.scaler.scale(loss).backward()
        
        if (self.steps + 1) % accumulation_steps == 0:
            # Unscale gradients and clip
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step with scaled gradients
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        return loss.item() * accumulation_steps
    
    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(checkpoint, path)

# Example usage
trainer = RoBERTaSentimentTrainer()
optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=2e-5)
trainer.optimizer = optimizer
trainer.steps = 0

# Training loop would go here
```

Slide 10: Ensemble Model Implementation

Ensemble methods combine multiple transformer models to achieve more robust sentiment predictions. This implementation shows how to create and train an ensemble of different transformer architectures including BERT, RoBERTa, and XLNet.

```python
from transformers import (
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    XLNetForSequenceClassification
)

class TransformerEnsemble(torch.nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.models = torch.nn.ModuleList([
            BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=num_labels
            ),
            RobertaForSequenceClassification.from_pretrained(
                'roberta-base',
                num_labels=num_labels
            ),
            XLNetForSequenceClassification.from_pretrained(
                'xlnet-base-cased',
                num_labels=num_labels
            )
        ])
        
        # Weighted averaging layer
        self.weights = torch.nn.Parameter(torch.ones(len(self.models)))
        
    def forward(self, **kwargs):
        outputs = []
        for model in self.models:
            output = model(**kwargs)
            outputs.append(output.logits)
        
        # Apply softmax to weights
        weights = F.softmax(self.weights, dim=0)
        
        # Weighted average of predictions
        ensemble_output = sum(w * out for w, out in zip(weights, outputs))
        return ensemble_output

class EnsembleTrainer:
    def __init__(self, ensemble, tokenizers):
        self.ensemble = ensemble
        self.tokenizers = tokenizers
        
    def prepare_inputs(self, text):
        # Prepare inputs for each model in the ensemble
        inputs = []
        for tokenizer in self.tokenizers:
            encoded = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            inputs.append(encoded)
        return inputs

# Example usage
ensemble = TransformerEnsemble()
tokenizers = [
    BertTokenizer.from_pretrained('bert-base-uncased'),
    RobertaTokenizer.from_pretrained('roberta-base'),
    XLNetTokenizer.from_pretrained('xlnet-base-cased')
]
trainer = EnsembleTrainer(ensemble, tokenizers)
```

Slide 11: Cross-validation Implementation for Transformer Models

Cross-validation is crucial for evaluating transformer models' generalization capabilities. This implementation shows a stratified k-fold approach specifically designed for transformer-based sentiment analysis, including proper handling of model states and evaluation metrics.

```python
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
import numpy as np

class TransformerCrossValidator:
    def __init__(self, model_class, dataset, n_splits=5):
        self.model_class = model_class
        self.dataset = dataset
        self.n_splits = n_splits
        self.metrics = []
        
    def validate(self, **model_kwargs):
        labels = [self.dataset[i]['labels'].item() for i in range(len(self.dataset))]
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(labels, labels)):
            # Initialize new model for each fold
            model = self.model_class(**model_kwargs)
            
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=16)
            
            # Train and evaluate
            fold_metrics = self.train_and_evaluate(model, train_loader, val_loader)
            self.metrics.append(fold_metrics)
            
            print(f"Fold {fold+1} Metrics:")
            for metric, value in fold_metrics.items():
                print(f"{metric}: {value:.4f}")
    
    def train_and_evaluate(self, model, train_loader, val_loader):
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Training loop
        model.train()
        for epoch in range(3):  # 3 epochs per fold
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        # Evaluation
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'accuracy': np.mean(np.array(predictions) == np.array(true_labels)),
            'f1': f1_score(true_labels, predictions, average='weighted'),
            'precision': precision_score(true_labels, predictions, average='weighted'),
            'recall': recall_score(true_labels, predictions, average='weighted')
        }
        
        return metrics

# Example usage
validator = TransformerCrossValidator(
    BertForSequenceClassification,
    sentiment_dataset,
    n_splits=5
)
validator.validate(
    pretrained_model_name='bert-base-uncased',
    num_labels=3
)
```

Slide 12: Advanced Text Preprocessing Pipeline

A sophisticated preprocessing pipeline is essential for handling real-world text data in sentiment analysis. This implementation demonstrates advanced text cleaning, normalization, and augmentation techniques specifically designed for transformer models.

```python
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textaugment import EDA

class AdvancedPreprocessor:
    def __init__(self, augment=False):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.augmentor = EDA() if augment else None
        
    def preprocess(self, text, augment=False):
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '@user', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Handle emojis and special characters
        text = self._convert_emojis(text)
        text = re.sub(r'[^\w\s@]', ' ', text)
        
        # Tokenization and lemmatization
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        # Text augmentation if requested
        if augment and self.augmentor:
            augmented_texts = [
                self.augmentor.synonym_replacement(text),
                self.augmentor.random_insertion(text),
                self.augmentor.random_swap(text),
                self.augmentor.random_deletion(text)
            ]
            return [' '.join(tokens)] + augmented_texts
        
        return [' '.join(tokens)]
    
    def _convert_emojis(self, text):
        emoji_mapping = {
            '😊': 'happy',
            '😢': 'sad',
            '😡': 'angry',
            # Add more emoji mappings as needed
        }
        for emoji, word in emoji_mapping.items():
            text = text.replace(emoji, f' {word} ')
        return text

# Example usage
preprocessor = AdvancedPreprocessor(augment=True)
text = "Just watched the new movie! 😊 #awesome @friend"
processed_texts = preprocessor.preprocess(text, augment=True)
for i, text in enumerate(processed_texts):
    print(f"Version {i+1}: {text}")
```

Slide 13: Attention Visualization Implementation

This implementation provides tools for visualizing attention patterns in transformer models, helping to interpret how the model weighs different words when making sentiment predictions. The visualization includes head-specific attention maps and aggregated attention patterns.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

class AttentionVisualizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def get_attention_weights(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Get attention weights from all layers and heads
        attention = outputs.attentions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return attention, tokens
    
    def plot_attention_heads(self, text, layer=0):
        attention, tokens = self.get_attention_weights(text)
        attention_layer = attention[layer][0]  # Get specific layer
        
        num_heads = attention_layer.size(0)
        fig, axes = plt.subplots(2, num_heads//2, figsize=(15, 8))
        axes = axes.flatten()
        
        for head in range(num_heads):
            ax = axes[head]
            sns.heatmap(
                attention_layer[head].numpy(),
                xticklabels=tokens,
                yticklabels=tokens,
                ax=ax,
                cmap='viridis'
            )
            ax.set_title(f'Head {head+1}')
            plt.setp(ax.get_xticklabels(), rotation=45)
            plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        return fig
    
    def plot_aggregated_attention(self, text):
        attention, tokens = self.get_attention_weights(text)
        
        # Average attention across all layers and heads
        avg_attention = torch.mean(
            torch.stack([layer[0].mean(dim=0) for layer in attention]),
            dim=0
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            avg_attention.numpy(),
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            annot=True,
            fmt='.2f'
        )
        plt.title('Aggregated Attention Pattern')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        return plt.gcf()

# Example usage
visualizer = AttentionVisualizer(model, tokenizer)
text = "The product quality is excellent but the service was terrible."
head_vis = visualizer.plot_attention_heads(text)
agg_vis = visualizer.plot_aggregated_attention(text)
```

Slide 14: Performance Metrics Calculator

This implementation provides comprehensive performance evaluation metrics for sentiment analysis models, including macro and micro averaging, confusion matrices, and confidence calibration measures.

```python
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, brier_score_loss
)
from scipy.stats import entropy
import numpy as np

class SentimentMetricsCalculator:
    def __init__(self, true_labels, predicted_probs, label_names=None):
        self.true_labels = np.array(true_labels)
        self.predicted_probs = np.array(predicted_probs)
        self.predicted_labels = np.argmax(predicted_probs, axis=1)
        self.label_names = label_names or ['Negative', 'Neutral', 'Positive']
        
    def calculate_all_metrics(self):
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(
            self.true_labels,
            self.predicted_labels
        )
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.true_labels,
            self.predicted_labels,
            average='weighted'
        )
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # ROC AUC
        metrics['roc_auc'] = roc_auc_score(
            np.eye(len(self.label_names))[self.true_labels],
            self.predicted_probs,
            multi_class='ovr'
        )
        
        # Confidence and calibration
        metrics['confidence_avg'] = np.mean(
            np.max(self.predicted_probs, axis=1)
        )
        metrics['brier_score'] = self._calculate_multiclass_brier_score()
        
        # Confusion metrics
        metrics['confusion_matrix'] = confusion_matrix(
            self.true_labels,
            self.predicted_labels
        )
        
        return metrics
    
    def _calculate_multiclass_brier_score(self):
        true_one_hot = np.eye(len(self.label_names))[self.true_labels]
        return np.mean(np.sum((true_one_hot - self.predicted_probs) ** 2, axis=1))
    
    def get_per_class_metrics(self):
        per_class_metrics = {}
        
        for i, label in enumerate(self.label_names):
            true_binary = (self.true_labels == i)
            pred_binary = (self.predicted_labels == i)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_binary,
                pred_binary,
                average='binary'
            )
            
            per_class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': np.sum(true_binary)
            }
            
        return per_class_metrics

# Example usage
calculator = SentimentMetricsCalculator(
    true_labels=[0, 1, 2, 1, 0],
    predicted_probs=np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.7],
        [0.2, 0.3, 0.5],
        [0.6, 0.3, 0.1]
    ])
)

metrics = calculator.calculate_all_metrics()
per_class_metrics = calculator.get_per_class_metrics()
```

Slide 15: Additional Resources

*   Paper: "Attention Is All You Need" - Original Transformer Architecture
    *   [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   Paper: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    *   [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
*   Paper: "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
    *   [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)
*   Paper: "XLNet: Generalized Autoregressive Pretraining for Language Understanding"
    *   [https://arxiv.org/abs/1906.08237](https://arxiv.org/abs/1906.08237)
*   Tutorial: "Fine-tuning Transformers for Sentiment Analysis"
    *   [https://huggingface.co/blog/sentiment-analysis-python](https://huggingface.co/blog/sentiment-analysis-python)
*   Research: "A Survey on Deep Learning for Named Entity Recognition"
    *   [https://arxiv.org/abs/1812.09449](https://arxiv.org/abs/1812.09449)

