## Sentiment Analysis on Twitter Data using Transformer Models and Machine Learning
Slide 1: Understanding Sentiment Analysis

Sentiment analysis is the process of identifying and categorizing opinions expressed in text data. It aims to determine the writer's attitude towards a particular topic or the overall contextual polarity of a document. This technique is widely used in social media monitoring, market research, and customer feedback analysis. Sentiment analysis can classify text as positive, negative, or neutral, and sometimes provide more nuanced emotional states like happy, angry, or sad.

Slide 2: Source Code for Understanding Sentiment Analysis

```python
import re

def simple_sentiment_analyzer(text):
    # Define positive and negative word lists
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor']
    
    # Convert text to lowercase and split into words
    words = re.findall(r'\w+', text.lower())
    
    # Count positive and negative words
    positive_count = sum(word in positive_words for word in words)
    negative_count = sum(word in negative_words for word in words)
    
    # Determine sentiment
    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"

# Example usage
sample_text = "The movie was great! The actors were amazing, but the plot was a bit poor."
sentiment = simple_sentiment_analyzer(sample_text)
print(f"Sentiment: {sentiment}")
```

Slide 3: Results for Understanding Sentiment Analysis

```
Sentiment: Positive
```

Slide 4: Preprocessing Twitter Data

Preprocessing is a crucial step in sentiment analysis, especially for Twitter data. It involves cleaning and normalizing the text to make it suitable for analysis. Common preprocessing steps include removing special characters, handling hashtags and mentions, expanding contractions, and tokenizing the text. Proper preprocessing can significantly improve the accuracy of sentiment analysis models.

Slide 5: Source Code for Preprocessing Twitter Data

```python
import re

def preprocess_tweet(tweet):
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    
    # Remove user mentions
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Remove hashtag symbol but keep the text
    tweet = re.sub(r'#(\w+)', r'\1', tweet)
    
    # Remove special characters and numbers
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
    
    # Remove extra whitespace
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    
    return tweet

# Example usage
raw_tweet = "@user This is an #amazing tweet! Check out https://example.com 123"
processed_tweet = preprocess_tweet(raw_tweet)
print(f"Original tweet: {raw_tweet}")
print(f"Processed tweet: {processed_tweet}")
```

Slide 6: Results for Preprocessing Twitter Data

```
Original tweet: @user This is an #amazing tweet! Check out https://example.com 123
Processed tweet: this is an amazing tweet check out
```

Slide 7: BERT for Sentiment Analysis

BERT (Bidirectional Encoder Representations from Transformers) is a powerful transformer-based model that has revolutionized natural language processing tasks, including sentiment analysis. BERT's bidirectional nature allows it to understand context from both left and right sides of each word, making it particularly effective for capturing nuanced sentiments in text. When fine-tuned on sentiment analysis tasks, BERT can achieve state-of-the-art performance.

Slide 8: Source Code for BERT for Sentiment Analysis

```python
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

def bert_sentiment_analysis(text):
    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)

    # Apply softmax to get probabilities
    probs = softmax(outputs.logits, dim=1)

    # Get predicted class (0: Negative, 1: Neutral, 2: Positive)
    predicted_class = torch.argmax(probs).item()

    # Map class to sentiment
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = sentiment_map[predicted_class]

    return sentiment, probs.tolist()[0]

# Example usage
text = "I absolutely loved the new restaurant! The food was delicious and the service was excellent."
sentiment, probabilities = bert_sentiment_analysis(text)
print(f"Sentiment: {sentiment}")
print(f"Probabilities: Negative: {probabilities[0]:.4f}, Neutral: {probabilities[1]:.4f}, Positive: {probabilities[2]:.4f}")
```

Slide 9: Traditional Machine Learning for Sentiment Analysis

While transformer-based models have gained popularity, traditional machine learning techniques still play a crucial role in sentiment analysis. These methods, such as Support Vector Machines (SVM), Naive Bayes, and Logistic Regression, are often faster to train and deploy, making them suitable for large-scale applications or scenarios with limited computational resources. They work by extracting features from the text and using these features to classify the sentiment.

Slide 10: Source Code for Traditional Machine Learning for Sentiment Analysis

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Sample dataset
texts = [
    "I love this product! It's amazing.",
    "This is terrible, don't buy it.",
    "It's okay, nothing special.",
    "Absolutely fantastic experience!",
    "Worst purchase ever, very disappointed."
]
labels = [1, 0, 2, 1, 0]  # 0: Negative, 1: Positive, 2: Neutral

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create feature vectors
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Predict sentiment for a new text
new_text = "This product exceeded my expectations!"
new_text_vectorized = vectorizer.transform([new_text])
prediction = classifier.predict(new_text_vectorized)

sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
print(f"Sentiment: {sentiment_map[prediction[0]]}")
```

Slide 11: Real-Life Example: Social Media Brand Monitoring

Sentiment analysis is widely used for brand monitoring on social media platforms. Companies can track public opinion about their products, services, or overall brand image in real-time. This allows them to quickly identify and address customer concerns, measure the impact of marketing campaigns, and gain insights into consumer preferences.

Slide 12: Source Code for Social Media Brand Monitoring

```python
import random

def simulate_social_media_posts(brand, num_posts=100):
    sentiments = ["positive", "neutral", "negative"]
    posts = []
    for _ in range(num_posts):
        sentiment = random.choice(sentiments)
        if sentiment == "positive":
            post = f"I love {brand}! Their products are amazing."
        elif sentiment == "neutral":
            post = f"{brand} is okay. Nothing special, but not bad either."
        else:
            post = f"I'm disappointed with {brand}. Their customer service is terrible."
        posts.append((post, sentiment))
    return posts

def analyze_brand_sentiment(posts):
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for _, sentiment in posts:
        sentiment_counts[sentiment] += 1
    
    total_posts = len(posts)
    sentiment_percentages = {k: (v / total_posts) * 100 for k, v in sentiment_counts.items()}
    return sentiment_percentages

# Simulate social media posts for a brand
brand_name = "TechGadget"
social_media_posts = simulate_social_media_posts(brand_name)

# Analyze brand sentiment
sentiment_analysis = analyze_brand_sentiment(social_media_posts)

print(f"Brand Sentiment Analysis for {brand_name}:")
for sentiment, percentage in sentiment_analysis.items():
    print(f"{sentiment.capitalize()}: {percentage:.2f}%")
```

Slide 13: Real-Life Example: Customer Feedback Analysis

Sentiment analysis is invaluable for analyzing customer feedback from various sources such as product reviews, support tickets, or survey responses. By automatically categorizing feedback as positive, negative, or neutral, companies can quickly identify areas for improvement, prioritize issues, and track customer satisfaction over time.

Slide 14: Source Code for Customer Feedback Analysis

```python
import random

def generate_customer_feedback(num_feedback=100):
    feedback_templates = {
        "positive": [
            "Great product! Exceeded my expectations.",
            "Excellent customer service. Very helpful.",
            "I'm very satisfied with my purchase. Will buy again.",
        ],
        "neutral": [
            "The product is okay. Nothing special.",
            "It works as expected. No complaints.",
            "Average quality for the price.",
        ],
        "negative": [
            "Disappointed with the quality. Not worth the price.",
            "Poor customer support. Took days to get a response.",
            "The product broke after a few uses. Wouldn't recommend.",
        ]
    }
    
    feedback_list = []
    for _ in range(num_feedback):
        sentiment = random.choice(list(feedback_templates.keys()))
        feedback = random.choice(feedback_templates[sentiment])
        feedback_list.append((feedback, sentiment))
    
    return feedback_list

def analyze_feedback(feedback_list):
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for _, sentiment in feedback_list:
        sentiment_counts[sentiment] += 1
    
    total_feedback = len(feedback_list)
    sentiment_percentages = {k: (v / total_feedback) * 100 for k, v in sentiment_counts.items()}
    return sentiment_percentages

# Generate simulated customer feedback
customer_feedback = generate_customer_feedback()

# Analyze feedback sentiment
sentiment_analysis = analyze_feedback(customer_feedback)

print("Customer Feedback Sentiment Analysis:")
for sentiment, percentage in sentiment_analysis.items():
    print(f"{sentiment.capitalize()}: {percentage:.2f}%")

# Identify areas for improvement
if sentiment_analysis["negative"] > 20:
    print("\nAction Required: High percentage of negative feedback. Investigate common issues and improve product/service quality.")
elif sentiment_analysis["positive"] < 50:
    print("\nAction Required: Low percentage of positive feedback. Focus on enhancing customer satisfaction and product features.")
else:
    print("\nPositive Outcome: Overall sentiment is good. Continue monitoring and maintaining quality.")
```

Slide 15: Additional Resources

1.  Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805. [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
2.  Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692. [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)
3.  Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108. [https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)

