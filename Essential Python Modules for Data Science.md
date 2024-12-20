## Essential Python Modules for Data Science
Slide 1: Web Scraping with BeautifulSoup

BeautifulSoup is a powerful library for parsing HTML and XML documents, making it ideal for web scraping tasks. It provides intuitive methods to navigate, search and extract data from HTML documents while handling malformed markup gracefully.

```python
from bs4 import BeautifulSoup
import requests

# Create a function to scrape article titles from a news website
def scrape_news_titles(url):
    # Send HTTP request and get response
    response = requests.get(url)
    
    # Parse HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all article titles (example using class)
    titles = soup.find_all('h2', class_='article-title')
    
    # Extract and clean text
    titles_list = [title.get_text().strip() for title in titles]
    
    return titles_list

# Example usage
url = 'https://example-news-site.com'
article_titles = scrape_news_titles(url)

# Print first 3 titles
print("Sample titles:")
for title in article_titles[:3]:
    print(f"- {title}")
```

Slide 2: Advanced Pandas Data Manipulation

Pandas provides sophisticated methods for handling complex data transformations and analysis. Understanding advanced grouping operations and window functions enables efficient processing of large datasets while maintaining code readability.

```python
import pandas as pd
import numpy as np

# Create sample data
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10),
    'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'value': np.random.randn(10)
})

# Advanced grouping with multiple aggregations
result = df.groupby('category').agg({
    'value': ['mean', 'std', 'count'],
    'date': ['min', 'max']
}).round(2)

# Rolling window statistics
df['rolling_mean'] = df.groupby('category')['value'].transform(
    lambda x: x.rolling(window=2, min_periods=1).mean()
)

print("Group Statistics:\n", result)
print("\nRolling Mean:\n", df)
```

Slide 3: Neural Network Implementation from Scratch

A fundamental implementation of a neural network using only NumPy demonstrates the core concepts of deep learning, including forward propagation, backpropagation, and gradient descent optimization for better understanding of neural architectures.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.weights = [np.random.randn(y, x) * 0.01 
                       for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.zeros((y, 1)) for y in layers[1:]]
    
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def forward(self, X):
        activation = X
        activations = [X]
        zs = []
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        return activation, activations, zs

# Example usage
nn = NeuralNetwork([2, 3, 1])  # 2 inputs, 3 hidden, 1 output
X = np.array([[0.5], [0.8]])
output, _, _ = nn.forward(X)
print("Network Output:", output)
```

Slide 4: Time Series Analysis with Prophet

Facebook's Prophet library provides robust time series forecasting capabilities with automatic handling of seasonality and holidays. This implementation shows how to prepare data, train a model, and generate future predictions.

```python
from prophet import Prophet
import pandas as pd

# Prepare sample time series data
df = pd.DataFrame({
    'ds': pd.date_range('2023-01-01', '2024-01-01', freq='D'),
    'y': np.random.normal(100, 10, 366) + \
         np.sin(np.linspace(0, 2*np.pi, 366)) * 20
})

# Initialize and train Prophet model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)
model.fit(df)

# Generate future dates for prediction
future_dates = model.make_future_dataframe(periods=30)

# Make predictions
forecast = model.predict(future_dates)

# Print forecast components
print("Forecast components:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
```

Slide 5: Advanced Text Processing with spaCy

SpaCy provides state-of-the-art natural language processing capabilities with pre-trained models for multiple languages. This implementation demonstrates named entity recognition, dependency parsing, and custom text processing pipelines.

```python
import spacy

# Load English language model
nlp = spacy.load('en_core_web_sm')

def analyze_text(text):
    # Process text through spaCy pipeline
    doc = nlp(text)
    
    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Get dependency parsing
    dependencies = [(token.text, token.dep_, token.head.text) 
                   for token in doc]
    
    # Extract noun phrases
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    
    return {
        'entities': entities,
        'dependencies': dependencies,
        'noun_phrases': noun_phrases
    }

# Example usage
text = "Apple Inc. is planning to open new offices in London next year."
results = analyze_text(text)
print("Named Entities:", results['entities'])
print("Noun Phrases:", results['noun_phrases'])
```

Slide 6: Implementing K-Means Clustering from Scratch

Understanding the mechanics of clustering algorithms is crucial for data scientists. This implementation shows how to build a K-means clustering algorithm from first principles, including centroid initialization and iterative optimization.

```python
import numpy as np
from sklearn.datasets import make_blobs

class KMeansClustering:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
    
    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[self.labels == i].mean(axis=0) 
                                    for i in range(self.k)])
            
            # Check convergence
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids
            
        return self

# Generate sample data and test
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60)
kmeans = KMeansClustering(k=3)
kmeans.fit(X)
print("Cluster Centers:\n", kmeans.centroids)
```

Slide 7: Deep Learning with PyTorch - Custom Dataset Implementation

PyTorch's flexibility allows creation of custom datasets for specialized machine learning tasks. This implementation shows how to build a custom dataset class with data loading, transformation, and batching capabilities.

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # Simulate loading data from file
        self.data = np.random.randn(1000, 10)  # Features
        self.labels = np.random.randint(0, 2, 1000)  # Binary labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return torch.FloatTensor(sample), torch.LongTensor([label])

# Example usage
dataset = CustomDataset(data_path="dummy_path")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through batches
for batch_idx, (data, labels) in enumerate(dataloader):
    if batch_idx == 0:
        print("Batch shape:", data.shape)
        print("Labels shape:", labels.shape)
        break
```

Slide 8: Advanced Data Visualization with Seaborn

Seaborn extends matplotlib's capabilities with statistical visualizations and enhanced aesthetics. This implementation demonstrates complex multi-plot visualizations and statistical representations of data distributions.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create sample dataset
np.random.seed(42)
data = pd.DataFrame({
    'group': np.repeat(['A', 'B', 'C'], 100),
    'value1': np.random.normal(0, 1, 300),
    'value2': np.random.normal(2, 1.5, 300),
    'category': np.random.choice(['X', 'Y', 'Z'], 300)
})

# Create complex visualization
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2)

# Violin plot
ax1 = fig.add_subplot(gs[0, 0])
sns.violinplot(data=data, x='group', y='value1', ax=ax1)
ax1.set_title('Distribution by Group')

# Joint plot in custom position
ax2 = fig.add_subplot(gs[0, 1])
sns.scatterplot(data=data, x='value1', y='value2', 
                hue='category', ax=ax2)
ax2.set_title('Value Relationships')

# Box plot
ax3 = fig.add_subplot(gs[1, :])
sns.boxplot(data=data, x='group', y='value2', hue='category')
ax3.set_title('Category Distributions')

plt.tight_layout()
print("Visualization created successfully")
```

Slide 9: Natural Language Understanding with Transformers

This implementation showcases the use of Hugging Face's transformers library for advanced NLP tasks, demonstrating how to perform sentiment analysis, text classification, and token classification using pre-trained models.

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

class TextAnalyzer:
    def __init__(self):
        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Initialize named entity recognition
        self.ner_pipeline = pipeline(
            "ner",
            aggregation_strategy="simple"
        )
    
    def analyze_text(self, text):
        # Get sentiment
        sentiment = self.sentiment_analyzer(text)[0]
        
        # Get named entities
        entities = self.ner_pipeline(text)
        
        return {
            'sentiment': {
                'label': sentiment['label'],
                'score': round(sentiment['score'], 3)
            },
            'entities': [
                {
                    'word': ent['word'],
                    'entity_type': ent['entity_group'],
                    'score': round(ent['score'], 3)
                } for ent in entities
            ]
        }

# Example usage
analyzer = TextAnalyzer()
text = "Amazon CEO Jeff Bezos announced new climate initiatives yesterday."
results = analyzer.analyze_text(text)
print("Analysis Results:", results)
```

Slide 10: Advanced Time Series Forecasting with SARIMA

Implementing Seasonal ARIMA models for complex time series analysis, including automatic parameter selection, residual analysis, and forecast generation with confidence intervals.

```python
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd

class TimeSeriesForecaster:
    def __init__(self):
        self.model = None
        self.order = None
        self.seasonal_order = None
    
    def find_optimal_params(self, data):
        # Grid search for optimal parameters
        p = d = q = range(0, 2)
        pdq = [(x, y, z) for x in p for y in d for z in q]
        seasonal_pdq = [(x, y, z, 12) for x in p for y in d for z in q]
        
        best_aic = float('inf')
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    model = SARIMAX(data,
                                  order=param,
                                  seasonal_order=param_seasonal)
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        self.order = param
                        self.seasonal_order = param_seasonal
                except:
                    continue
    
    def fit(self, data):
        self.model = SARIMAX(data,
                            order=self.order,
                            seasonal_order=self.seasonal_order)
        self.results = self.model.fit()
        
    def forecast(self, steps=30):
        forecast = self.results.get_forecast(steps=steps)
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int()
        return mean_forecast, conf_int

# Example usage
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
data = pd.Series(np.random.normal(0, 1, len(dates)) + \
                 np.sin(np.linspace(0, 8*np.pi, len(dates))), 
                 index=dates)

forecaster = TimeSeriesForecaster()
forecaster.find_optimal_params(data)
forecaster.fit(data)
forecast, conf_int = forecaster.forecast(12)
print("Forecast for next 12 months:\n", forecast)
```

Slide 11: Advanced Data Preprocessing Pipeline

Implementation of a comprehensive data preprocessing pipeline that handles missing values, outliers, feature encoding, and scaling while maintaining the ability to transform new data consistently.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class AdvancedPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features=None, numerical_features=None):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.encoders = {}
        self.scaler = StandardScaler()
        self.numerical_imputer = None
        self.categorical_imputer = None
    
    def fit(self, X, y=None):
        # Initialize imputation values
        self.numerical_imputer = X[self.numerical_features].median()
        self.categorical_imputer = X[self.categorical_features].mode().iloc[0]
        
        # Fit label encoders for categorical features
        for cat_feat in self.categorical_features:
            self.encoders[cat_feat] = LabelEncoder()
            self.encoders[cat_feat].fit(X[cat_feat].fillna(self.categorical_imputer[cat_feat]))
        
        # Fit scaler for numerical features
        numerical_data = X[self.numerical_features].fillna(self.numerical_imputer)
        self.scaler.fit(numerical_data)
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Handle missing values
        X_copy[self.numerical_features] = X_copy[self.numerical_features].fillna(self.numerical_imputer)
        X_copy[self.categorical_features] = X_copy[self.categorical_features].fillna(self.categorical_imputer)
        
        # Encode categorical features
        for cat_feat in self.categorical_features:
            X_copy[cat_feat] = self.encoders[cat_feat].transform(X_copy[cat_feat])
        
        # Scale numerical features
        X_copy[self.numerical_features] = self.scaler.transform(X_copy[self.numerical_features])
        
        return X_copy

# Example usage
data = pd.DataFrame({
    'age': [25, 30, np.nan, 45],
    'income': [50000, 60000, 75000, np.nan],
    'category': ['A', 'B', np.nan, 'A']
})

preprocessor = AdvancedPreprocessor(
    categorical_features=['category'],
    numerical_features=['age', 'income']
)

processed_data = preprocessor.fit_transform(data)
print("Processed Data:\n", processed_data)
```

Slide 12: Deep Learning with PyTorch - Custom Model Architecture

This implementation demonstrates the creation of a flexible neural network architecture using PyTorch, including custom layers, skip connections, and attention mechanisms for complex deep learning tasks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=8)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return self.norm(x + attn_output)

class CustomNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        # Create encoder layers
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            
        self.encoder = nn.Sequential(*layers)
        self.attention = AttentionBlock(hidden_dims[-1])
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        
    def forward(self, x):
        # Encode features
        features = self.encoder(x)
        
        # Apply attention
        features = features.unsqueeze(0)
        attended = self.attention(features)
        attended = attended.squeeze(0)
        
        # Classification
        output = self.classifier(attended)
        return F.log_softmax(output, dim=1)

# Example usage
model = CustomNetwork(
    input_dim=20,
    hidden_dims=[64, 32, 16],
    num_classes=5
)

# Test forward pass
x = torch.randn(10, 20)  # Batch of 10 samples
output = model(x)
print("Output shape:", output.shape)
print("Model architecture:\n", model)
```

Slide 13: Advanced Data Analysis with PySpark

Implementation of distributed data processing using PySpark, demonstrating advanced operations like window functions, custom aggregations, and complex data transformations at scale.

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

class SparkAnalyzer:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("Advanced Analysis") \
            .getOrCreate()
    
    def process_data(self, data_path):
        # Read data
        df = self.spark.read.parquet(data_path)
        
        # Define window specs
        time_window = Window.partitionBy("category") \
            .orderBy("timestamp") \
            .rowsBetween(-2, 0)
        
        # Complex transformations
        result = df.withColumn(
            "moving_avg", 
            F.avg("value").over(time_window)
        ).withColumn(
            "rank", 
            F.dense_rank().over(
                Window.partitionBy("category") \
                    .orderBy(F.desc("value"))
            )
        ).withColumn(
            "pct_diff",
            F.when(F.lag("value").over(time_window).isNotNull(),
                   ((F.col("value") - F.lag("value").over(time_window)) / 
                    F.lag("value").over(time_window)) * 100
            ).otherwise(0)
        )
        
        # Custom aggregation
        summary = result.groupBy("category").agg(
            F.sum("value").alias("total_value"),
            F.expr("percentile_approx(value, 0.5)").alias("median_value"),
            F.collect_list("value").alias("value_history")
        )
        
        return result, summary
    
    def stop(self):
        self.spark.stop()

# Example usage (pseudo-code as Spark needs cluster)
analyzer = SparkAnalyzer()
result, summary = analyzer.process_data("path/to/data")
print("Processing complete")
```

Slide 14: Additional Resources

*   arXiv URL: [https://arxiv.org/abs/2302.14520](https://arxiv.org/abs/2302.14520) - "Modern Deep Learning for Time Series Analysis"
*   arXiv URL: [https://arxiv.org/abs/2103.14030](https://arxiv.org/abs/2103.14030) - "Advances in Natural Language Processing: A Survey of Transformers"
*   arXiv URL: [https://arxiv.org/abs/2006.11287](https://arxiv.org/abs/2006.11287) - "Deep Learning for Automated Data Analysis"
*   Recommended Search: "Google Scholar - Latest Python Data Science Frameworks"
*   Documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
*   Advanced Tutorials: [https://pytorch.org/tutorials/intermediate/torch\_compile\_tutorial.html](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

