## NLP Models for Sentiment Analysis in Python
Slide 1: Introduction to NLP Models for Sentiment Analysis

Natural Language Processing (NLP) models have revolutionized sentiment analysis, enabling machines to understand and interpret human emotions in text. This presentation explores five powerful models: BERT, RoBERTa, DistilBERT, ALBERT, and XLNet. We'll delve into their architectures, use cases, and implementation in Python, providing practical examples for sentiment analysis tasks.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Example usage
model_name = "bert-base-uncased"
tokenizer, model = load_model(model_name)
print(f"Loaded {model_name} model and tokenizer")
```

Slide 2: BERT (Bidirectional Encoder Representations from Transformers)

BERT, developed by Google, is a transformer-based model that learns contextual word embeddings by considering both left and right contexts. It uses masked language modeling and next sentence prediction for pre-training. BERT's bidirectional nature makes it highly effective for various NLP tasks, including sentiment analysis.

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

text = "I love this movie! It's fantastic."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_score = predictions[0][2].item()

print(f"Positive sentiment score: {positive_score:.4f}")
```

Slide 3: RoBERTa (Robustly Optimized BERT Approach)

RoBERTa, introduced by Facebook AI, is an optimized version of BERT. It removes the next sentence prediction task, uses dynamic masking, and is trained on larger datasets with longer sequences. These improvements lead to better performance on various NLP tasks, including sentiment analysis.

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

text = "This product exceeded my expectations. Highly recommended!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_score = predictions[0][2].item()

print(f"Positive sentiment score: {positive_score:.4f}")
```

Slide 4: DistilBERT (Distilled BERT)

DistilBERT is a lighter and faster version of BERT, developed by Hugging Face. It retains 97% of BERT's performance while being 40% smaller and 60% faster. This makes it ideal for resource-constrained environments or real-time sentiment analysis applications.

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

text = "The customer service was terrible. I'm very disappointed."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    negative_score = predictions[0][0].item()

print(f"Negative sentiment score: {negative_score:.4f}")
```

Slide 5: ALBERT (A Lite BERT)

ALBERT, developed by Google Research, is another lightweight version of BERT. It uses parameter-sharing techniques and factorized embedding parameterization to reduce model size while maintaining performance. ALBERT is particularly useful for sentiment analysis tasks requiring large-scale deployment.

```python
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=3)

text = "The restaurant was okay, but nothing special."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    neutral_score = predictions[0][1].item()

print(f"Neutral sentiment score: {neutral_score:.4f}")
```

Slide 6: XLNet (eXtreme Learning NET)

XLNet, developed by Carnegie Mellon University and Google Brain, is an autoregressive language model that overcomes limitations of BERT by using permutation language modeling. This approach allows XLNet to capture bidirectional context without the need for masked inputs, potentially leading to improved performance in sentiment analysis tasks.

```python
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import torch

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=3)

text = "I can't believe how amazing this experience was!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_score = predictions[0][2].item()

print(f"Positive sentiment score: {positive_score:.4f}")
```

Slide 7: Fine-tuning for Sentiment Analysis

Fine-tuning these pre-trained models on a specific sentiment analysis dataset can significantly improve their performance. Here's an example of fine-tuning BERT for sentiment analysis using a custom dataset.

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Example usage (you would need to prepare your own dataset)
texts = ["I love this!", "I hate this!", "It's okay."]
labels = [2, 0, 1]  # 2: positive, 0: negative, 1: neutral

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

dataset = SentimentDataset(texts, labels, tokenizer, max_length=128)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

Slide 8: Data Preprocessing for Sentiment Analysis

Proper data preprocessing is crucial for effective sentiment analysis. This slide demonstrates common preprocessing techniques using Python's NLTK library.

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Example usage
raw_text = "I absolutely loved the movie! It was amazing and thrilling. 10/10 would recommend!"
processed_text = preprocess_text(raw_text)
print(f"Original: {raw_text}")
print(f"Processed: {processed_text}")
```

Slide 9: Ensemble Methods for Sentiment Analysis

Combining multiple models can often lead to improved performance in sentiment analysis. This slide demonstrates how to create an ensemble of different models for more robust predictions.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentEnsemble:
    def __init__(self, model_names):
        self.models = []
        self.tokenizers = []
        for name in model_names:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3)
            self.models.append(model)
            self.tokenizers.append(tokenizer)

    def predict(self, text):
        predictions = []
        for model, tokenizer in zip(self.models, self.tokenizers):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                pred = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        return ensemble_pred

# Example usage
ensemble = SentimentEnsemble(['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased'])
text = "This product is absolutely fantastic! I couldn't be happier with my purchase."
prediction = ensemble.predict(text)
sentiment = ["Negative", "Neutral", "Positive"][prediction.argmax().item()]
confidence = prediction.max().item()

print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence:.4f}")
```

Slide 10: Real-life Example: Social Media Sentiment Analysis

In this example, we'll analyze sentiment from Twitter data using the BERT model. This can be useful for brand monitoring, customer feedback analysis, or trend prediction.

```python
import tweepy
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Twitter API credentials (you need to obtain these from Twitter Developer Portal)
consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"
access_token = "your_access_token"
access_token_secret = "your_access_token_secret"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Load BERT model for sentiment analysis
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = ["Negative", "Neutral", "Positive"][predictions.argmax().item()]
    return sentiment

# Analyze tweets for a specific topic
topic = "artificial intelligence"
tweets = api.search_tweets(q=topic, lang="en", count=100)

sentiments = []
for tweet in tweets:
    sentiment = analyze_sentiment(tweet.text)
    sentiments.append(sentiment)

# Calculate sentiment distribution
sentiment_dist = {
    "Positive": sentiments.count("Positive") / len(sentiments),
    "Neutral": sentiments.count("Neutral") / len(sentiments),
    "Negative": sentiments.count("Negative") / len(sentiments)
}

print(f"Sentiment distribution for '{topic}':")
for sentiment, percentage in sentiment_dist.items():
    print(f"{sentiment}: {percentage:.2%}")
```

Slide 11: Real-life Example: Customer Review Analysis

In this example, we'll use RoBERTa to analyze customer reviews for a product, helping businesses understand customer sentiment and identify areas for improvement.

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Load RoBERTa model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = ["Negative", "Neutral", "Positive"][predictions.argmax().item()]
    confidence = predictions.max().item()
    return sentiment, confidence

# Sample customer reviews
reviews = [
    "This product is amazing! It exceeded all my expectations.",
    "Not bad, but could be better. There's room for improvement.",
    "Terrible experience. I regret buying this product.",
    "It's okay, nothing special but gets the job done.",
    "Absolutely love it! Best purchase I've made in years."
]

# Analyze sentiments
results = [{"review": review, "sentiment": analyze_sentiment(review)[0], 
            "confidence": analyze_sentiment(review)[1]} for review in reviews]

# Create a DataFrame for easy analysis
df = pd.DataFrame(results)

# Calculate and display sentiment distribution
sentiment_dist = df['sentiment'].value_counts(normalize=True)
print("Sentiment Distribution:")
print(sentiment_dist)

# Display top positive and negative reviews
print("\nTop Positive Review:")
print(df[df['sentiment'] == 'Positive'].sort_values('confidence', ascending=False)['review'].iloc[0])
print("\nTop Negative Review:")
print(df[df['sentiment'] == 'Negative'].sort_values('confidence', ascending=False)['review'].iloc[0])
```

Slide 12: Handling Multilingual Sentiment Analysis

As businesses expand globally, the ability to analyze sentiment in multiple languages becomes crucial. This slide demonstrates how to use a multilingual model for sentiment analysis across different languages.

```python
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch

# Load multilingual XLM-RoBERTa model
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=3)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = ["Negative", "Neutral", "Positive"][predictions.argmax().item()]
    confidence = predictions.max().item()
    return sentiment, confidence

# Example reviews in different languages
reviews = {
    "English": "This product is fantastic!",
    "Spanish": "Este producto es fantástico!",
    "French": "Ce produit est fantastique!",
    "German": "Dieses Produkt ist fantastisch!",
    "Chinese": "这个产品太棒了！"
}

# Analyze sentiments
for language, review in reviews.items():
    sentiment, confidence = analyze_sentiment(review)
    print(f"{language}: {review}")
    print(f"Sentiment: {sentiment}, Confidence: {confidence:.4f}\n")
```

Slide 13: Aspect-Based Sentiment Analysis

Aspect-based sentiment analysis allows us to identify sentiments towards specific aspects of a product or service. This slide demonstrates a simple approach using BERT and named entity recognition.

```python
from transformers import pipeline
import spacy

# Load BERT sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Load spaCy for named entity recognition
nlp = spacy.load("en_core_web_sm")

def aspect_based_sentiment(text):
    # Perform named entity recognition
    doc = nlp(text)
    aspects = [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG"]]

    # Analyze sentiment for each aspect
    results = {}
    for aspect in aspects:
        # Find sentences containing the aspect
        sentences = [sent.text for sent in doc.sents if aspect.lower() in sent.text.lower()]
        if sentences:
            # Analyze sentiment for these sentences
            sentiments = sentiment_analyzer(sentences)
            avg_sentiment = sum(s['score'] for s in sentiments) / len(sentiments)
            results[aspect] = "Positive" if avg_sentiment > 0.5 else "Negative"

    return results

# Example usage
review = "The new iPhone camera is amazing, but the battery life is disappointing. Apple's customer service was helpful though."
aspects_sentiment = aspect_based_sentiment(review)

print("Aspect-based sentiments:")
for aspect, sentiment in aspects_sentiment.items():
    print(f"{aspect}: {sentiment}")
```

Slide 14: Sentiment Analysis for Social Media Monitoring

Social media monitoring is crucial for brand management and customer engagement. This slide demonstrates how to use sentiment analysis for real-time social media monitoring.

```python
import tweepy
from transformers import pipeline
import time

# Twitter API credentials (replace with your own)
consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"
access_token = "your_access_token"
access_token_secret = "your_access_token_secret"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

class TweetListener(tweepy.StreamListener):
    def on_status(self, status):
        if hasattr(status, 'retweeted_status'):
            return
        
        tweet = status.text
        sentiment = sentiment_analyzer(tweet)[0]
        
        print(f"Tweet: {tweet}")
        print(f"Sentiment: {sentiment['label']}, Score: {sentiment['score']:.4f}")
        print("-" * 50)

    def on_error(self, status_code):
        if status_code == 420:
            return False

# Set up stream listener
stream_listener = TweetListener()
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)

# Start streaming tweets (replace with your desired keywords)
stream.filter(track=["your_brand_name", "your_product_name"], languages=["en"])
```

Slide 15: Additional Resources

For those interested in diving deeper into NLP models for sentiment analysis, here are some valuable resources:

1. BERT: Bidirectional Encoder Representations from Transformers ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
2. RoBERTa: A Robustly Optimized BERT Pretraining Approach ArXiv: [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)
3. DistilBERT: a distilled version of BERT: smaller, faster, cheaper and lighter ArXiv: [https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)
4. ALBERT: A Lite BERT for Self-supervised Learning of Language Representations ArXiv: [https://arxiv.org/abs/1909.11942](https://arxiv.org/abs/1909.11942)
5. XLNet: Generalized Autoregressive Pretraining for Language Understanding ArXiv: [https://arxiv.org/abs/1906.08237](https://arxiv.org/abs/1906.08237)

These papers provide in-depth explanations of the models we've discussed, including their architectures, training procedures, and performance comparisons. They serve as excellent starting points for understanding the theoretical foundations of these powerful NLP models.

