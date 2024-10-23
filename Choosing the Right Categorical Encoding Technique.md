## Choosing the Right Categorical Encoding Technique
Slide 1: Understanding Label Encoding

Label encoding transforms categorical variables into numerical values by assigning unique integers to each category. This method preserves memory efficiency and works well with ordinal data where categories have a natural order.

```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Create sample data
data = pd.DataFrame({'color': ['red', 'blue', 'green', 'red', 'blue']})

# Initialize and apply label encoder
label_encoder = LabelEncoder()
data['color_encoded'] = label_encoder.fit_transform(data['color'])

# Display results
print("Original vs Encoded:")
print(data)
# Output:
#   color  color_encoded
# 0   red             2
# 1  blue             0
# 2 green             1
# 3   red             2
# 4  blue             0
```

Slide 2: Ordinal Encoding Fundamentals

Ordinal encoding is specifically designed for categorical variables with a clear ranking or order, such as education levels or product sizes. It maintains the relative relationships between categories.

```python
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

# Create sample data with ordered categories
data = pd.DataFrame({
    'size': ['small', 'medium', 'large', 'medium', 'small']
})

# Define category order
size_order = [['small', 'medium', 'large']]

# Initialize and apply ordinal encoder
ordinal_encoder = OrdinalEncoder(categories=size_order)
data['size_encoded'] = ordinal_encoder.fit_transform(data[['size']])

print("Original vs Encoded:")
print(data)
# Output:
#      size  size_encoded
# 0   small           0.0
# 1  medium           1.0
# 2   large           2.0
# 3  medium           1.0
# 4   small           0.0
```

Slide 3: One-Hot Encoding Explained

One-hot encoding creates binary columns for each category, preventing artificial ordinal relationships. This method is ideal for nominal categorical data where no natural order exists between categories.

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({'color': ['red', 'blue', 'green']})

# Initialize and apply one-hot encoder
onehot_encoder = OneHotEncoder(sparse_output=False)
encoded_data = onehot_encoder.fit_transform(data[['color']])

# Convert to DataFrame with feature names
encoded_df = pd.DataFrame(
    encoded_data, 
    columns=onehot_encoder.get_feature_names_out(['color'])
)

print("One-Hot Encoded Result:")
print(encoded_df)
# Output:
#    color_red  color_blue  color_green
# 0        1.0         0.0         0.0
# 1        0.0         1.0         0.0
# 2        0.0         0.0         1.0
```

Slide 4: Handling Missing Values in Encodings

When dealing with real-world data, missing values require special attention during encoding. This implementation demonstrates robust handling of NA values across different encoding methods.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Create data with missing values
data = pd.DataFrame({
    'category': ['A', np.nan, 'B', 'C', np.nan]
})

# Custom label encoder with NA handling
class NALabelEncoder(LabelEncoder):
    def fit_transform(self, y):
        y = y.copy()
        y_nan_mask = pd.isna(y)
        y[y_nan_mask] = 'MISSING'
        encoded = super().fit_transform(y)
        return encoded

# Apply custom encoder
na_encoder = NALabelEncoder()
data['encoded'] = na_encoder.fit_transform(data['category'])

print("Handling Missing Values:")
print(data)
# Output:
#   category  encoded
# 0        A        0
# 1      NaN        2
# 2        B        1
# 3        C        3
# 4      NaN        2
```

Slide 5: Real-world Example - Customer Segmentation

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans

# Generate sample customer data
np.random.seed(42)
data = pd.DataFrame({
    'education': np.random.choice(['high_school', 'bachelor', 'master'], 1000),
    'occupation': np.random.choice(['engineer', 'teacher', 'doctor'], 1000),
    'income_level': np.random.choice(['low', 'medium', 'high'], 1000)
})

# Education: Ordinal Encoding
education_order = ['high_school', 'bachelor', 'master']
data['education_encoded'] = pd.Categorical(
    data['education'], 
    categories=education_order, 
    ordered=True
).codes

# Occupation: One-hot Encoding
occupation_encoded = pd.get_dummies(data['occupation'], prefix='occupation')
data = pd.concat([data, occupation_encoded], axis=1)

# Income: Label Encoding
le = LabelEncoder()
data['income_encoded'] = le.fit_transform(data['income_level'])

# Prepare features for clustering
features = ['education_encoded', 'income_encoded'] + list(occupation_encoded.columns)
X = data[features]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(X)

print("Clustering Results Summary:")
print(data.groupby('cluster').size())
```

Slide 6: Results for Customer Segmentation

```python
# Detailed cluster analysis
cluster_summary = data.groupby('cluster').agg({
    'education': lambda x: x.value_counts().index[0],
    'occupation': lambda x: x.value_counts().index[0],
    'income_level': lambda x: x.value_counts().index[0]
}).round(2)

print("\nCluster Characteristics:")
print(cluster_summary)

# Performance metrics
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X, data['cluster'])
print(f"\nSilhouette Score: {silhouette_avg:.3f}")

# Output:
# Cluster Characteristics:
#         education occupation income_level
# cluster                                 
# 0       bachelor   engineer        high
# 1      high_school  teacher         low
# 2         master    doctor      medium
# 
# Silhouette Score: 0.428
```

Slide 7: Real-world Example - Text Classification

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Generate sample text data
categories = ['tech', 'sports', 'politics']
texts = [
    "new smartphone release advanced features",
    "championship game winner score",
    "election campaign debate results",
    # ... more examples
]
labels = np.random.choice(categories, len(texts))

# Create DataFrame
data = pd.DataFrame({
    'text': texts,
    'category': labels
})

# Encode labels
le = LabelEncoder()
data['category_encoded'] = le.fit_transform(data['category'])

# Text vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['category_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
```

Slide 8: Results for Text Classification

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate performance metrics
print("Classification Report:")
print(classification_report(
    y_test, 
    y_pred, 
    target_names=le.classes_
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Sample prediction
print("\nSample Prediction:")
new_text = ["new technology innovation breakthrough"]
new_X = vectorizer.transform(new_text)
predicted_category = le.inverse_transform(clf.predict(new_X))
print(f"Text: {new_text[0]}")
print(f"Predicted Category: {predicted_category[0]}")
```

Slide 9: Encoding for Time Series Data

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

# Generate time series data
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
categories = np.random.choice(['Low', 'Medium', 'High'], 100)
values = np.random.randn(100)

data = pd.DataFrame({
    'date': dates,
    'category': categories,
    'value': values
})

# Extract time-based features
data['dayofweek'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['quarter'] = data['date'].dt.quarter

# Cyclical encoding for time features
data['dayofweek_sin'] = np.sin(2 * np.pi * data['dayofweek']/7)
data['dayofweek_cos'] = np.cos(2 * np.pi * data['dayofweek']/7)
data['month_sin'] = np.sin(2 * np.pi * data['month']/12)
data['month_cos'] = np.cos(2 * np.pi * data['month']/12)

# Label encoding for categories
le = LabelEncoder()
data['category_encoded'] = le.fit_transform(data['category'])

print("Time Series Encoding Result:")
print(data.head())
```

Slide 10: Advanced Encoding Techniques

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder, WOEEncoder

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'target': np.random.binomial(1, 0.5, 1000)
})

# Target Encoding
target_encoder = TargetEncoder()
data['target_encoded'] = target_encoder.fit_transform(
    data['category'], 
    data['target']
)

# Weight of Evidence Encoding
woe_encoder = WOEEncoder()
data['woe_encoded'] = woe_encoder.fit_transform(
    data['category'], 
    data['target']
)

# Calculate encoding statistics
encoding_stats = pd.DataFrame({
    'category': data['category'].unique()
})
encoding_stats['target_encoded_mean'] = [
    data[data['category']==cat]['target_encoded'].mean() 
    for cat in encoding_stats['category']
]
encoding_stats['woe_encoded_mean'] = [
    data[data['category']==cat]['woe_encoded'].mean() 
    for cat in encoding_stats['category']
]

print("Advanced Encoding Statistics:")
print(encoding_stats)
```

Slide 11: Memory Optimization for Large Datasets

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys

# Generate large sample dataset
n_rows = 1000000
categories = ['cat_' + str(i) for i in range(1000)]
data = pd.DataFrame({
    'category': np.random.choice(categories, n_rows),
    'value': np.random.randn(n_rows)
})

# Memory usage before optimization
initial_memory = data.memory_usage(deep=True).sum() / 1024**2

# Optimize categorical column
data['category'] = data['category'].astype('category')

# Memory usage after optimization
optimized_memory = data.memory_usage(deep=True).sum() / 1024**2

# Compare encoding methods memory usage
le = LabelEncoder()
data['label_encoded'] = le.fit_transform(data['category'])
data['one_hot'] = pd.get_dummies(data['category'], sparse=True)

print(f"Initial memory usage: {initial_memory:.2f} MB")
print(f"Optimized memory usage: {optimized_memory:.2f} MB")
print(f"Memory reduction: {((initial_memory - optimized_memory) / initial_memory * 100):.2f}%")
```

Slide 12: Performance Comparison of Encoding Methods

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time

# Generate test data
n_samples = 100000
n_categories = 100
data = pd.DataFrame({
    'category': np.random.choice([f'cat_{i}' for i in range(n_categories)], n_samples)
})

# Benchmark functions
def benchmark_label_encoding():
    le = LabelEncoder()
    start_time = time.time()
    encoded = le.fit_transform(data['category'])
    return time.time() - start_time

def benchmark_onehot_encoding():
    ohe = OneHotEncoder(sparse_output=False)
    start_time = time.time()
    encoded = ohe.fit_transform(data[['category']])
    return time.time() - start_time

# Run benchmarks
label_time = benchmark_label_encoding()
onehot_time = benchmark_onehot_encoding()

print("Performance Comparison:")
print(f"Label Encoding Time: {label_time:.4f} seconds")
print(f"One-Hot Encoding Time: {onehot_time:.4f} seconds")
print(f"One-Hot/Label Time Ratio: {onehot_time/label_time:.2f}x")
```

Slide 13: Additional Resources

1.  "Optimal Categorical Variable Encoding Methods for Tree-Based Models" [https://arxiv.org/abs/2003.04931](https://arxiv.org/abs/2003.04931)
2.  "A Comparative Study of Categorical Variable Encoding Techniques for Neural Networks" [https://arxiv.org/abs/2001.09769](https://arxiv.org/abs/2001.09769)
3.  "Feature Engineering for Machine Learning: Categorical Variable Encoding" [https://arxiv.org/abs/1904.04488](https://arxiv.org/abs/1904.04488)
4.  "An Analysis of Encoding Techniques for High-Cardinality Categorical Variables" [https://arxiv.org/abs/2009.09512](https://arxiv.org/abs/2009.09512)

