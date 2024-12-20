## Data Preprocessing Techniques with Pandas and NumPy
Slide 1: Data Preprocessing with Pandas and NumPy

Modern data analysis requires robust preprocessing to handle missing values, outliers, and inconsistent formats. This implementation demonstrates essential techniques for cleaning and transforming raw data using Pandas and NumPy, including handling missing values and feature scaling.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
def preprocess_dataset(filepath):
    # Read data
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df['numeric_col'] = df['numeric_col'].fillna(df['numeric_col'].mean())
    df['categorical_col'] = df['categorical_col'].fillna(df['categorical_col'].mode()[0])
    
    # Remove outliers using IQR method
    Q1 = df['numeric_col'].quantile(0.25)
    Q3 = df['numeric_col'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['numeric_col'] < (Q1 - 1.5 * IQR)) | 
              (df['numeric_col'] > (Q3 + 1.5 * IQR)))]
    
    # Feature scaling
    scaler = StandardScaler()
    df['numeric_col_scaled'] = scaler.fit_transform(df[['numeric_col']])
    
    return df

# Example usage
data = pd.DataFrame({
    'numeric_col': [1, 2, np.nan, 4, 100, 6],
    'categorical_col': ['A', 'B', None, 'B', 'C', 'A']
})
cleaned_data = preprocess_dataset(data)
print("Processed Dataset:\n", cleaned_data)
```

Slide 2: Time Series Analysis with Prophet

Time series forecasting is crucial for business planning and trend analysis. Facebook's Prophet library excels at handling seasonal patterns and holiday effects while providing robust uncertainty estimates.

```python
from prophet import Prophet
import pandas as pd
import numpy as np

def forecast_timeseries(data, periods=30):
    # Prepare data
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    # Fit model
    model.fit(data)
    
    # Create future dates
    future_dates = model.make_future_dataframe(periods=periods)
    
    # Generate forecast
    forecast = model.predict(future_dates)
    
    return forecast

# Example usage
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.random.normal(loc=100, scale=10, size=len(dates))
values += np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 20  # Add seasonality

df = pd.DataFrame({
    'ds': dates,
    'y': values
})

forecast = forecast_timeseries(df)
print("Forecast Results:\n", forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
```

Slide 3: Deep Learning with PyTorch

PyTorch provides a dynamic computational framework for building and training neural networks. This implementation shows a complete neural network architecture for classification tasks with modern best practices.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DeepNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DeepNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.layer3(x)
        return x

# Training function
def train_model(model, X_train, y_train, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Example usage
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
model = DeepNetwork(10, 64, 2)
train_model(model, X, y)
```

Slide 4: Natural Language Processing with Transformers

Modern NLP leverages transformer architectures for superior text understanding. This implementation demonstrates fine-tuning BERT for text classification, including preprocessing and model training with the Transformers library.

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset

class TextClassifier:
    def __init__(self, num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=num_labels
        )
        
    def preprocess_text(self, texts, labels=None):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        if labels:
            return encodings, torch.tensor(labels)
        return encodings
    
    def train(self, train_texts, train_labels, epochs=3):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        for epoch in range(epochs):
            self.model.train()
            inputs, labels = self.preprocess_text(train_texts, train_labels)
            
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Example usage
texts = [
    "This movie is fantastic!",
    "Terrible waste of time.",
    "Great performance by the actors"
]
labels = [1, 0, 1]  # 1: positive, 0: negative

classifier = TextClassifier()
classifier.train(texts, labels)
```

Slide 5: Advanced Data Visualization with Plotly

Interactive visualizations enhance data exploration and presentation. This implementation creates sophisticated, interactive plots using Plotly, demonstrating multiple chart types and customization options.

```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_advanced_dashboard(data):
    # Create scatter plot with trendline
    scatter_fig = px.scatter(
        data,
        x='x_values',
        y='y_values',
        color='categories',
        trendline="ols",
        title="Interactive Scatter Plot with Trend"
    )
    
    # Create animated bubble chart
    bubble_fig = px.scatter(
        data,
        x='x_values',
        y='y_values',
        size='size_values',
        color='categories',
        animation_frame='time_period',
        title="Animated Bubble Chart"
    )
    
    # Create 3D surface plot
    surface_fig = go.Figure(data=[
        go.Surface(z=data['matrix_values'])
    ])
    surface_fig.update_layout(title="3D Surface Plot")
    
    return scatter_fig, bubble_fig, surface_fig

# Generate sample data
np.random.seed(42)
n_points = 100
data = pd.DataFrame({
    'x_values': np.random.normal(0, 1, n_points),
    'y_values': np.random.normal(0, 1, n_points),
    'size_values': np.random.uniform(10, 50, n_points),
    'categories': np.random.choice(['A', 'B', 'C'], n_points),
    'time_period': np.random.choice(range(5), n_points),
    'matrix_values': np.random.rand(10, 10)
})

scatter, bubble, surface = create_advanced_dashboard(data)
# Figures can be displayed using .show() in a Jupyter notebook
```

Slide 6: Custom Neural Network Architectures

Understanding neural network internals is crucial for deep learning. This implementation builds a neural network from scratch using only NumPy, including forward and backward propagation.

```python
import numpy as np

class CustomNeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers)-1):
            self.weights.append(
                np.random.randn(layers[i], layers[i+1]) * np.sqrt(2/layers[i])
            )
            self.biases.append(
                np.zeros((1, layers[i+1]))
            )
    
    def relu(self, X):
        return np.maximum(0, X)
    
    def relu_derivative(self, X):
        return X > 0
    
    def forward_propagation(self, X):
        activations = [X]
        
        for i in range(len(self.weights)):
            net = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            
            if i != len(self.weights) - 1:
                activation = self.relu(net)
            else:
                activation = net  # Linear activation for last layer
                
            activations.append(activation)
            
        return activations
    
    def backward_propagation(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        activations = self.forward_propagation(X)
        
        dZ = activations[-1] - y
        for i in range(len(self.weights) - 1, -1, -1):
            self.weights[i] -= learning_rate * np.dot(activations[i].T, dZ) / m
            self.biases[i] -= learning_rate * np.sum(dZ, axis=0, keepdims=True) / m
            
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                dZ = dA * self.relu_derivative(activations[i])

# Example usage
X = np.random.randn(100, 10)
y = np.random.randn(100, 1)
model = CustomNeuralNetwork([10, 64, 32, 1])

# Training
for epoch in range(100):
    model.backward_propagation(X, y)
    predictions = model.forward_propagation(X)[-1]
    mse = np.mean((predictions - y) ** 2)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, MSE: {mse:.4f}")
```

Slide 7: Large-Scale Data Processing with PySpark

PySpark enables distributed data processing at scale. This implementation shows how to perform complex aggregations and transformations on large datasets using PySpark's DataFrame API.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, window, avg, count
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType

def process_large_dataset():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("LargeScaleProcessing") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    # Define schema
    schema = StructType([
        StructField("timestamp", TimestampType(), False),
        StructField("user_id", StringType(), False),
        StructField("value", DoubleType(), False),
        StructField("category", StringType(), False)
    ])

    # Read streaming data
    df = spark.readStream \
        .format("json") \
        .schema(schema) \
        .load("path/to/data")

    # Complex transformations
    result = df.groupBy(
        window(col("timestamp"), "1 hour"),
        col("category")
    ).agg(
        avg("value").alias("avg_value"),
        count("user_id").alias("user_count")
    ).filter(col("user_count") > 100)

    # Write results
    query = result.writeStream \
        .outputMode("append") \
        .format("parquet") \
        .option("path", "path/to/output") \
        .option("checkpointLocation", "path/to/checkpoint") \
        .start()

    return query

# Example usage
query = process_large_dataset()
query.awaitTermination()
```

Slide 8: Advanced Time Series Forecasting

Implementing sophisticated time series models requires handling multiple seasonal patterns and external regressors. This code demonstrates a custom implementation combining statistical and machine learning approaches.

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

class AdvancedTimeSeriesForecaster:
    def __init__(self, seasonality_periods=[7, 365]):
        self.seasonality_periods = seasonality_periods
        self.scaler = StandardScaler()
        self.models = []
        
    def decompose_series(self, series):
        # Extract multiple seasonal components
        seasonal_components = []
        residuals = series.copy()
        
        for period in self.seasonality_periods:
            seasonal = series.rolling(window=period, center=True).mean()
            seasonal_components.append(seasonal)
            residuals -= seasonal
            
        return seasonal_components, residuals
    
    def fit(self, series, exog=None):
        # Scale data
        scaled_data = self.scaler.fit_transform(series.reshape(-1, 1)).flatten()
        
        # Decompose series
        seasonal_components, residuals = self.decompose_series(scaled_data)
        
        # Fit SARIMAX model for residuals
        self.residual_model = SARIMAX(
            residuals,
            exog=exog,
            order=(2, 1, 2),
            seasonal_order=(1, 1, 1, self.seasonality_periods[0])
        ).fit()
        
        return self
    
    def predict(self, steps, exog_future=None):
        # Predict residuals
        residual_forecast = self.residual_model.forecast(
            steps=steps,
            exog=exog_future
        )
        
        # Add seasonal components
        final_forecast = residual_forecast.copy()
        for seasonal in self.seasonal_components:
            seasonal_forecast = seasonal[-steps:]
            final_forecast += seasonal_forecast
            
        # Inverse transform
        return self.scaler.inverse_transform(
            final_forecast.reshape(-1, 1)
        ).flatten()

# Example usage
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
values = np.random.normal(100, 10, len(dates))

# Add multiple seasonal patterns
values += np.sin(np.arange(len(dates)) * 2 * np.pi / 7) * 5  # Weekly
values += np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 20  # Yearly

forecaster = AdvancedTimeSeriesForecaster()
forecaster.fit(values)
forecast = forecaster.predict(steps=30)
print("30-day forecast:", forecast)
```

Slide 9: Computer Vision with PyTorch

Modern computer vision tasks require sophisticated neural network architectures. This implementation shows a custom CNN with attention mechanisms for image classification.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels//8, 1)
        self.key = nn.Conv2d(channels, channels//8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        batch, c, h, w = x.size()
        
        # Query, Key, Value projections
        q = self.query(x).view(batch, -1, h*w)
        k = self.key(x).view(batch, -1, h*w)
        v = self.value(x).view(batch, -1, h*w)
        
        # Attention scores
        scores = torch.bmm(q.transpose(1, 2), k)
        attention = F.softmax(scores / (c ** 0.5), dim=2)
        
        # Apply attention to values
        out = torch.bmm(v, attention.transpose(1, 2))
        return out.view(batch, c, h, w)

class VisionNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.attention = AttentionBlock(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.attention(x)
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Example usage
model = VisionNetwork(num_classes=10)
dummy_input = torch.randn(1, 3, 32, 32)
output = model(dummy_input)
print("Output shape:", output.shape)
```

Slide 10: Natural Language Understanding with Custom Attention

This implementation demonstrates a custom attention mechanism for sequence processing, particularly useful for tasks like machine translation and text summarization.

```python
import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Split heads
        Q = Q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        
        return self.out(output), attention

# Example usage
d_model = 512
num_heads = 8
sequence_length = 30
batch_size = 16

attention = MultiHeadAttention(d_model, num_heads)
x = torch.randn(batch_size, sequence_length, d_model)
mask = torch.ones(batch_size, num_heads, sequence_length, sequence_length)

output, attention_weights = attention(x, x, x, mask)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
```

Slide 11: Advanced Data Cleaning Pipeline

Real-world data requires sophisticated cleaning techniques. This implementation shows a comprehensive pipeline for handling complex data quality issues.

```python
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

class AdvancedDataCleaner:
    def __init__(self, categorical_threshold=0.05):
        self.categorical_threshold = categorical_threshold
        self.scaler = RobustScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        
    def detect_outliers(self, series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        return (series < lower) | (series > upper)
    
    def handle_missing_values(self, df):
        # Determine optimal imputation strategy per column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # Handle numeric columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = pd.DataFrame(
                self.imputer.fit_transform(df[numeric_cols]),
                columns=numeric_cols
            )
        
        # Handle categorical columns
        for col in categorical_cols:
            mode_value = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode_value)
            
        return df
    
    def clean_data(self, df):
        # Copy input
        df_cleaned = df.copy()
        
        # Handle missing values
        df_cleaned = self.handle_missing_values(df_cleaned)
        
        # Handle outliers in numeric columns
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outliers = self.detect_outliers(df_cleaned[col])
            if outliers.any():
                df_cleaned.loc[outliers, col] = np.nan
                
        # Final imputation pass
        df_cleaned = self.handle_missing_values(df_cleaned)
        
        # Scale numeric features
        df_cleaned[numeric_cols] = self.scaler.fit_transform(df_cleaned[numeric_cols])
        
        return df_cleaned

# Example usage
np.random.seed(42)
data = pd.DataFrame({
    'numeric1': np.random.normal(0, 1, 1000),
    'numeric2': np.random.normal(100, 15, 1000),
    'categorical': np.random.choice(['A', 'B', 'C', None], 1000)
})

# Add some outliers and missing values
data.loc[0:10, 'numeric1'] = 1000
data.loc[20:30, 'numeric2'] = np.nan

cleaner = AdvancedDataCleaner()
cleaned_data = cleaner.clean_data(data)
print("Cleaned data summary:\n", cleaned_data.describe())
```

Slide 12: Feature Engineering for Machine Learning

Advanced feature engineering techniques can significantly improve model performance. This implementation demonstrates automated feature generation and selection using statistical methods.

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures

class AdvancedFeatureEngineer:
    def __init__(self, max_poly_degree=2, interaction_only=True):
        self.max_poly_degree = max_poly_degree
        self.interaction_only = interaction_only
        self.poly = PolynomialFeatures(
            degree=max_poly_degree,
            interaction_only=interaction_only
        )
        
    def create_time_features(self, date_series):
        features = pd.DataFrame()
        features['hour'] = date_series.dt.hour
        features['day'] = date_series.dt.day
        features['month'] = date_series.dt.month
        features['year'] = date_series.dt.year
        features['day_of_week'] = date_series.dt.dayofweek
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        
        return features
    
    def generate_interactions(self, X):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        return pd.DataFrame(
            self.poly.fit_transform(X[numeric_cols]),
            columns=self.poly.get_feature_names(numeric_cols)
        )
    
    def calculate_feature_importance(self, X, y):
        importances = {}
        
        # Calculate mutual information for numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            mi_scores = mutual_info_regression(X[numeric_cols], y)
            importances.update(dict(zip(numeric_cols, mi_scores)))
        
        # Calculate correlation ratio for categorical features
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            correlation_ratio = self.correlation_ratio(X[col], y)
            importances[col] = correlation_ratio
            
        return pd.Series(importances)
    
    @staticmethod
    def correlation_ratio(categories, y):
        categories = pd.Categorical(categories)
        y_vars = [y[categories == category].var() for category in categories.categories]
        y_means = [y[categories == category].mean() for category in categories.categories]
        
        n = len(y)
        weighted_means = sum(
            len(y[categories == category]) * mean 
            for category, mean in zip(categories.categories, y_means)
        ) / n
        
        numerator = sum(
            len(y[categories == category]) * (mean - weighted_means) ** 2
            for category, mean in zip(categories.categories, y_means)
        )
        
        denominator = sum((y - y.mean()) ** 2)
        
        if denominator == 0:
            return 0
        return numerator / denominator
    
    def transform(self, X, y=None):
        result = X.copy()
        
        # Generate time features for datetime columns
        datetime_cols = X.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            time_features = self.create_time_features(X[col])
            result = pd.concat([result, time_features], axis=1)
            result = result.drop(col, axis=1)
        
        # Generate polynomial features and interactions
        interactions = self.generate_interactions(result)
        result = pd.concat([result, interactions], axis=1)
        
        # Calculate feature importance if target is provided
        if y is not None:
            self.feature_importances_ = self.calculate_feature_importance(result, y)
            
        return result

# Example usage
np.random.seed(42)
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
data = pd.DataFrame({
    'date': dates,
    'numeric1': np.random.normal(0, 1, len(dates)),
    'numeric2': np.random.normal(100, 15, len(dates)),
    'category': np.random.choice(['A', 'B', 'C'], len(dates))
})
target = np.random.normal(0, 1, len(dates))

engineer = AdvancedFeatureEngineer()
transformed_data = engineer.transform(data, target)
print("Original features:", list(data.columns))
print("Transformed features:", list(transformed_data.columns))
print("\nFeature importances:\n", engineer.feature_importances_.sort_values(ascending=False))
```

Slide 13: Additional Resources

*   "Attention Is All You Need" - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   "Deep Residual Learning for Image Recognition" - [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
*   "Efficient Estimation of Word Representations in Vector Space" - [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
*   "XGBoost: A Scalable Tree Boosting System" - [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
*   "Facebook Prophet: Forecasting at Scale" - [https://research.facebook.com/publications/forecasting-at-scale/](https://research.facebook.com/publications/forecasting-at-scale/)

For further research on these topics:

*   Google Scholar: [https://scholar.google.com](https://scholar.google.com)
*   Papers With Code: [https://paperswithcode.com](https://paperswithcode.com)
*   arXiv Machine Learning section: [https://arxiv.org/list/cs.LG/recent](https://arxiv.org/list/cs.LG/recent)

