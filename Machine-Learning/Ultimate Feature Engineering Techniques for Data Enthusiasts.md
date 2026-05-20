## Ultimate Feature Engineering Techniques for Data Enthusiasts
Slide 1: Feature Scaling and Normalization

Feature scaling transforms numerical features to a similar scale, preventing certain features from dominating the learning process due to their larger magnitudes. This fundamental technique ensures that machine learning algorithms converge faster and perform optimally, especially for distance-based methods.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Generate sample data
data = np.array([[100, 2], [200, 4], [300, 6], [400, 8]])

# Standard Scaling (Z-score normalization)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# MinMax Scaling
minmax = MinMaxScaler()
minmax_data = minmax.fit_transform(data)

print("Original Data:\n", data)
print("\nStandard Scaled:\n", scaled_data)
print("\nMinMax Scaled:\n", minmax_data)
```

Slide 2: Handling Missing Values with Advanced Techniques

Missing data can significantly impact model performance. Advanced imputation methods go beyond simple mean/median replacement by considering the underlying data distribution and relationships between features to provide more accurate estimations.

```python
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np

# Create sample data with missing values
np.random.seed(42)
df = pd.DataFrame({
    'A': [1, np.nan, 3, 4, np.nan],
    'B': [10, 20, np.nan, 40, 50],
    'C': [100, 200, 300, np.nan, 500]
})

# Advanced imputation using MICE (Multiple Imputation by Chained Equations)
imputer = IterativeImputer(random_state=42)
imputed_data = imputer.fit_transform(df)

print("Original Data:\n", df)
print("\nImputed Data:\n", pd.DataFrame(imputed_data, columns=df.columns))
```

Slide 3: Time-Based Feature Engineering

Temporal features derived from datetime columns can reveal hidden patterns in time series data. Extracting cyclical components and creating lag features enables models to capture time-dependent relationships and seasonal patterns effectively.

```python
import pandas as pd
import numpy as np

# Create sample datetime data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
df = pd.DataFrame({'date': dates, 'value': np.random.randn(len(dates))})

def create_time_features(df, date_col):
    df = df.copy()
    
    # Extract basic components
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    
    # Create cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Create lag features
    df['lag_1'] = df['value'].shift(1)
    df['lag_7'] = df['value'].shift(7)
    
    return df

transformed_df = create_time_features(df, 'date')
print(transformed_df.head())
```

Slide 4: Advanced Categorical Encoding

Modern categorical encoding techniques extend beyond simple one-hot encoding to capture complex relationships and handle high-cardinality features while maintaining information about category frequencies and relationships.

```python
import pandas as pd
from category_encoders import TargetEncoder, WOEEncoder
import numpy as np

# Create sample data
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B', 'A'] * 100,
    'target': np.random.binomial(1, 0.5, 600)
})

# Target Encoding
target_encoder = TargetEncoder()
target_encoded = target_encoder.fit_transform(df['category'], df['target'])

# Weight of Evidence Encoding
woe_encoder = WOEEncoder()
woe_encoded = woe_encoder.fit_transform(df['category'], df['target'])

print("Original Categories:\n", df['category'].head())
print("\nTarget Encoded:\n", target_encoded.head())
print("\nWOE Encoded:\n", woe_encoded.head())
```

Slide 5: Feature Selection Using Mutual Information

Mutual Information quantifies the statistical dependency between variables, providing a powerful method for selecting relevant features that capture both linear and non-linear relationships with the target variable.

```python
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000
X = pd.DataFrame({
    'feature1': np.random.normal(0, 1, n_samples),
    'feature2': np.random.exponential(1, n_samples),
    'feature3': np.random.uniform(0, 1, n_samples),
    'feature4': np.random.poisson(1, n_samples)
})

# Create target with non-linear relationship
y = (X['feature1']**2 + np.exp(X['feature2']/2) + 
     np.sin(X['feature3']) > 2).astype(int)

# Calculate mutual information scores
mi_scores = mutual_info_classif(X, y)

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'mutual_information': mi_scores
}).sort_values('mutual_information', ascending=False)

print("Feature Importance based on Mutual Information:")
print(feature_importance)
```

Slide 6: Polynomial Feature Generation

Polynomial features capture non-linear relationships between variables by creating interaction terms and higher-order features. This technique expands the feature space to allow linear models to learn non-linear patterns inherent in the data.

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# Create sample data
X = np.array([[1, 2], [3, 4], [5, 6]])
df = pd.DataFrame(X, columns=['x1', 'x2'])

# Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X)

# Create feature names
feature_names = ['x1', 'x2', 'x1^2', 'x1*x2', 'x2^2']
poly_df = pd.DataFrame(poly_features, columns=feature_names)

print("Original features:\n", df)
print("\nPolynomial features:\n", poly_df)
```

Slide 7: Text Feature Engineering

Advanced text feature engineering transforms unstructured text data into meaningful numerical representations using sophisticated techniques beyond simple bag-of-words, incorporating semantic meaning and contextual information.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Sample text data
texts = [
    "Machine learning is fascinating",
    "Deep learning revolutionizes AI",
    "Natural language processing advances"
]

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
tfidf_features = tfidf.fit_transform(texts)

# BERT Embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
bert_embeddings = model.encode(texts)

print("TF-IDF Features Shape:", tfidf_features.shape)
print("TF-IDF Feature Names:", tfidf.get_feature_names_out()[:10])
print("\nBERT Embeddings Shape:", bert_embeddings.shape)
```

Slide 8: Feature Clustering for Dimensionality Reduction

Feature clustering groups similar features together to reduce dimensionality while preserving information content. This technique is particularly useful for high-dimensional datasets where features exhibit strong correlations.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate correlated features
np.random.seed(42)
n_samples = 1000
n_features = 50
X = np.random.randn(n_samples, n_features)
X[:, :25] += np.random.randn(n_samples, 1) * 2  # Create correlation

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster features
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
feature_clusters = kmeans.fit_predict(X_scaled.T)

# Create representative features
clustered_features = np.array([
    X_scaled[:, feature_clusters == i].mean(axis=1)
    for i in range(n_clusters)
]).T

print("Original shape:", X.shape)
print("Clustered shape:", clustered_features.shape)
```

Slide 9: Automated Feature Engineering

Automated feature engineering leverages machine learning to discover and create meaningful features from raw data, reducing manual effort and potentially uncovering complex patterns that might be missed through traditional approaches.

```python
import featuretools as ft
import pandas as pd
import numpy as np

# Create sample entity data
customers = pd.DataFrame({
    'customer_id': range(5),
    'join_date': pd.date_range('2023-01-01', periods=5)
})

transactions = pd.DataFrame({
    'transaction_id': range(20),
    'customer_id': np.random.choice(range(5), 20),
    'amount': np.random.uniform(10, 1000, 20),
    'timestamp': pd.date_range('2023-01-01', periods=20, freq='H')
})

# Create entity set
es = ft.EntitySet(id='customer_transactions')
es = es.add_dataframe(
    dataframe_name='customers',
    dataframe=customers,
    index='customer_id'
)

es = es.add_dataframe(
    dataframe_name='transactions',
    dataframe=transactions,
    index='transaction_id',
    time_index='timestamp'
)

# Add relationship
es = es.add_relationship('customers', 'customer_id', 'transactions', 'customer_id')

# Generate features
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_entity='customers',
    agg_primitives=['mean', 'sum', 'count'],
    trans_primitives=['month', 'hour']
)

print("Generated features:\n", feature_matrix.head())
```

Slide 10: Time Window Feature Engineering

Time window features capture temporal patterns by aggregating data over specific time intervals, enabling models to understand trends, seasonality, and event-driven changes in time series data across multiple granularities.

```python
import pandas as pd
import numpy as np

# Create sample time series data
dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
df = pd.DataFrame({
    'timestamp': dates,
    'value': np.random.normal(100, 10, 1000),
    'event_id': np.random.randint(1, 5, 1000)
})

def create_window_features(df, window_sizes=[24, 168, 720]):
    df = df.set_index('timestamp')
    
    for window in window_sizes:
        # Rolling statistics
        df[f'rolling_mean_{window}h'] = df['value'].rolling(window=window).mean()
        df[f'rolling_std_{window}h'] = df['value'].rolling(window=window).std()
        
        # Event aggregations
        df[f'event_count_{window}h'] = df.groupby('event_id')['value'].rolling(window).count()
        
        # Expanding statistics
        df[f'expanding_max_{window}h'] = df['value'].expanding(min_periods=window).max()
    
    return df.reset_index()

windowed_df = create_window_features(df)
print(windowed_df.head())
```

Slide 11: Feature Interaction Detection

Feature interactions reveal complex relationships between variables that can significantly impact model performance. This advanced technique systematically identifies and quantifies meaningful interaction effects through statistical methods.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations

def detect_interactions(X, y, threshold=0.1):
    # Fit base model
    rf_base = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_base.fit(X, y)
    base_score = rf_base.score(X, y)
    
    # Test feature interactions
    interactions = []
    feature_pairs = list(combinations(X.columns, 2))
    
    for f1, f2 in feature_pairs:
        # Create interaction feature
        X_interaction = X.copy()
        X_interaction[f'{f1}_{f2}_interact'] = X[f1] * X[f2]
        
        # Fit model with interaction
        rf_interact = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_interact.fit(X_interaction, y)
        interact_score = rf_interact.score(X_interaction, y)
        
        # Calculate improvement
        improvement = interact_score - base_score
        if improvement > threshold:
            interactions.append({
                'features': (f1, f2),
                'improvement': improvement
            })
    
    return pd.DataFrame(interactions)

# Generate sample data
np.random.seed(42)
X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 1000),
    'x2': np.random.normal(0, 1, 1000),
    'x3': np.random.normal(0, 1, 1000)
})
y = X['x1'] * X['x2'] + X['x3'] + np.random.normal(0, 0.1, 1000)

# Detect interactions
interactions = detect_interactions(X, y)
print("Detected Interactions:\n", interactions)
```

Slide 12: Frequency Domain Features

Frequency domain transformations convert time series data into frequency components, revealing periodic patterns and cycles that might not be apparent in the time domain representation of the data.

```python
import numpy as np
from scipy.fft import fft, fftfreq
import pandas as pd

def extract_frequency_features(signal, sampling_rate=1.0):
    n = len(signal)
    
    # Compute FFT
    fft_vals = fft(signal)
    freqs = fftfreq(n, d=1/sampling_rate)
    
    # Get positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_vals = np.abs(fft_vals[pos_mask])
    
    # Extract features
    features = {
        'dominant_freq': freqs[np.argmax(fft_vals)],
        'freq_magnitude_mean': np.mean(fft_vals),
        'freq_magnitude_std': np.std(fft_vals),
        'power_spectrum_entropy': -np.sum(fft_vals * np.log2(fft_vals + 1e-10)),
        'spectral_centroid': np.sum(freqs * fft_vals) / np.sum(fft_vals)
    }
    
    return features

# Generate sample signal with multiple frequencies
t = np.linspace(0, 10, 1000)
signal = (np.sin(2*np.pi*2*t) + 
         0.5*np.sin(2*np.pi*5*t) + 
         0.3*np.sin(2*np.pi*10*t))

# Extract frequency features
freq_features = extract_frequency_features(signal, sampling_rate=100)
print("Frequency Domain Features:\n", pd.Series(freq_features))
```

Slide 13: Feature Importance Analysis Using SHAP Values

SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance that shows how each feature contributes to individual predictions while considering feature interactions and maintaining local accuracy.

```python
import shap
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Generate sample dataset
np.random.seed(42)
X = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(0, 1, 1000),
    'feature3': np.random.normal(0, 1, 1000)
})
y = 2*X['feature1'] + X['feature2']**2 + np.random.normal(0, 0.1, 1000)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Get global feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', descending=True)

print("SHAP-based Feature Importance:\n", feature_importance)
```

Slide 14: Advanced Feature Selection Using Recursive Feature Elimination

Recursive Feature Elimination implements a sophisticated feature selection approach that iteratively removes features based on their importance, considering the interdependencies between features and their collective impact on model performance.

```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

class AdvancedRFE:
    def __init__(self, n_features_to_select=None):
        self.base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.selector = RFECV(
            estimator=self.base_estimator,
            step=1,
            cv=self.cv,
            scoring='accuracy',
            n_features_to_select=n_features_to_select,
            min_features_to_select=1
        )
    
    def fit_select(self, X, y):
        # Fit RFE
        self.selector.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[self.selector.support_]
        
        # Get feature ranking
        feature_ranking = pd.DataFrame({
            'feature': X.columns,
            'ranking': self.selector.ranking_,
            'selected': self.selector.support_
        }).sort_values('ranking')
        
        return {
            'selected_features': selected_features,
            'feature_ranking': feature_ranking,
            'cv_scores': self.selector.cv_results_,
            'n_features_selected': self.selector.n_features_
        }

# Example usage
X = pd.DataFrame(np.random.randn(1000, 10), 
                columns=[f'feature_{i}' for i in range(10)])
y = (X['feature_0'] + X['feature_1']**2 > 0).astype(int)

rfe = AdvancedRFE()
results = rfe.fit_select(X, y)

print("Selected Features:\n", results['selected_features'])
print("\nFeature Ranking:\n", results['feature_ranking'])
```

Slide 15: Additional Resources

*   A Survey on Automated Feature Engineering for Deep Learning in Images [https://arxiv.org/abs/2001.02009](https://arxiv.org/abs/2001.02009)
*   Automated Feature Engineering for Predictive Modeling [https://arxiv.org/abs/1912.04977](https://arxiv.org/abs/1912.04977)
*   Deep Feature Synthesis: Towards Automating Data Science Endeavors [https://www.google.com/search?q=deep+feature+synthesis+paper](https://www.google.com/search?q=deep+feature+synthesis+paper)
*   Feature Engineering for Machine Learning: Principles and Techniques [https://www.google.com/search?q=feature+engineering+principles+and+techniques](https://www.google.com/search?q=feature+engineering+principles+and+techniques)
*   Time Series Feature Engineering: A Comprehensive Survey [https://www.google.com/search?q=time+series+feature+engineering+survey](https://www.google.com/search?q=time+series+feature+engineering+survey)

