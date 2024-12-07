## Target Encoding for Categorical Variables in Machine Learning
Slide 1: Understanding Target Encoding Fundamentals

Target encoding represents a sophisticated approach to handling categorical variables by replacing category levels with their corresponding target mean values. This technique effectively reduces dimensionality while preserving statistical relationships between features and the target variable.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Sample dataset
data = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B', 'A'],
    'target': [1, 0, 1, 0, 1, 0]
})

# Basic target encoding implementation
target_means = data.groupby('category')['target'].mean()
encoded_values = data['category'].map(target_means)
print(f"Original categories:\n{data['category']}")
print(f"\nEncoded values:\n{encoded_values}")
```

Slide 2: Implementing Basic Target Encoding with Cross-Validation

Cross-validation in target encoding prevents data leakage by encoding each fold using statistics from other folds. This approach maintains the integrity of the validation process and provides more reliable encoded features.

```python
from sklearn.model_selection import KFold

def target_encode_cv(X, y, column, n_splits=5):
    encoded = np.zeros(len(X))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(X):
        # Calculate means from training fold
        means = y[train_idx].groupby(X[column][train_idx]).mean()
        # Apply to validation fold
        encoded[val_idx] = X[column][val_idx].map(means)
    
    return encoded
```

Slide 3: Smoothing in Target Encoding

Smoothing helps mitigate the impact of rare categories and reduces overfitting by incorporating global statistics. The technique uses a weighted average between the category mean and global mean based on category frequency.

```python
def smooth_target_encoding(X, y, column, alpha=10):
    # Calculate global mean
    global_mean = y.mean()
    # Calculate category counts and means
    stats = pd.DataFrame({
        'count': y.groupby(X[column]).count(),
        'mean': y.groupby(X[column]).mean()
    })
    
    # Apply smoothing formula
    smoothed_means = (stats['count'] * stats['mean'] + alpha * global_mean) / (stats['count'] + alpha)
    return smoothed_means

# Example usage
encoded_smooth = X[column].map(smooth_target_encoding(X, y, column))
```

Slide 4: Handling New Categories

Target encoding must address the challenge of encountering new categories during prediction time. This implementation provides a robust solution by maintaining a default encoding value for unseen categories.

```python
class TargetEncoder:
    def __init__(self, default_value='global_mean'):
        self.encoding_dict = {}
        self.global_mean = None
        self.default_value = default_value
    
    def fit(self, X, y):
        self.global_mean = y.mean()
        category_means = y.groupby(X).mean()
        self.encoding_dict = category_means.to_dict()
        return self
    
    def transform(self, X):
        return X.map(lambda x: self.encoding_dict.get(x, self.global_mean))
```

Slide 5: Real-World Application - Customer Churn Prediction

Using target encoding to predict customer churn demonstrates its practical application. This implementation processes categorical features like service plans and customer locations to predict subscription cancellation probability.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder

# Load example customer data
data = pd.DataFrame({
    'service_plan': ['basic', 'premium', 'basic', 'enterprise'],
    'location': ['NY', 'CA', 'TX', 'NY'],
    'churn': [0, 1, 0, 1]
})

# Split data
X = data[['service_plan', 'location']]
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

Slide 6: Source Code for Customer Churn Prediction

```python
# Initialize and fit target encoder
encoder = TargetEncoder()
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

# Train a simple model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_encoded, y_train)

# Evaluate performance
predictions = model.predict_proba(X_test_encoded)[:, 1]
auc_score = roc_auc_score(y_test, predictions)
print(f"AUC-ROC Score: {auc_score:.4f}")
```

Slide 7: Advanced Target Encoding with Leave-One-Out

Leave-one-out target encoding represents a sophisticated variation that excludes the current row when calculating encoding values, providing more robust estimates and reducing overfitting potential in the encoding process.

```python
def leave_one_out_target_encoding(X, y):
    # Calculate global mean for missing values
    global_mean = y.mean()
    
    # Initialize result array
    encoded = np.zeros(len(X))
    
    # Group data
    grouped = pd.DataFrame({'y': y, 'X': X}).groupby('X')
    
    for name, group in grouped:
        group_size = len(group)
        
        if group_size == 1:
            encoded[group.index] = global_mean
        else:
            # Calculate LOO mean for each observation
            sum_total = group['y'].sum()
            for idx in group.index:
                encoded[idx] = (sum_total - group.loc[idx, 'y']) / (group_size - 1)
    
    return encoded
```

Slide 8: Target Encoding with Noise Addition

Adding controlled noise to encoded values helps prevent overfitting and improves model generalization. This implementation demonstrates how to incorporate Gaussian noise proportional to the category frequency.

```python
def noise_target_encoding(X, y, noise_level=0.01):
    # Calculate basic encoding
    means = y.groupby(X).mean()
    counts = y.groupby(X).count()
    
    # Add noise scaled by category frequency
    encoded = X.map(means)
    noise = np.random.normal(0, noise_level, len(X))
    scaled_noise = noise * (1.0 / np.sqrt(X.map(counts)))
    
    return encoded + scaled_noise

# Example usage
X_encoded_with_noise = noise_target_encoding(X['category'], y)
print("Encoded values with noise:", X_encoded_with_noise.head())
```

Slide 9: Real-World Application - Credit Risk Assessment

Target encoding proves particularly valuable in credit risk assessment, where categorical variables like occupation and industry require sophisticated handling to predict default probability accurately.

```python
# Sample credit data
credit_data = pd.DataFrame({
    'occupation': ['engineer', 'teacher', 'doctor', 'engineer'],
    'industry': ['tech', 'education', 'healthcare', 'tech'],
    'default_risk': [0.1, 0.2, 0.05, 0.15]
})

# Initialize multi-column target encoder
te = TargetEncoder()
categorical_features = ['occupation', 'industry']

# Encode multiple features
encoded_features = te.fit_transform(credit_data[categorical_features], 
                                  credit_data['default_risk'])
```

Slide 10: Source Code for Credit Risk Assessment

```python
# Prepare data for modeling
X = encoded_features
y = credit_data['default_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train gradient boosting model
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate performance
from sklearn.metrics import mean_squared_error, r2_score
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
```

Slide 11: Mathematical Foundation of Target Encoding

This slide explores the mathematical principles underlying target encoding. The formula represents the smoothed target encoding calculation, incorporating both global and local statistics for optimal category representation.

```python
# Mathematical representation of smoothed target encoding
"""
The smoothed target encoding formula:

$$\hat{y}_c = \frac{n_c \bar{y}_c + \lambda \bar{y}}{n_c + \lambda}$$

Where:
$$\hat{y}_c$$ = encoded value for category c
$$n_c$$ = number of samples in category c
$$\bar{y}_c$$ = mean target value for category c
$$\bar{y}$$ = global mean target value
$$\lambda$$ = smoothing parameter
"""

def mathematical_target_encoding(X, y, lambda_param=10):
    global_mean = y.mean()
    stats = pd.DataFrame({
        'n': y.groupby(X).count(),
        'mean': y.groupby(X).mean()
    })
    
    encoded_values = (stats['n'] * stats['mean'] + lambda_param * global_mean) / \
                    (stats['n'] + lambda_param)
    return encoded_values
```

Slide 12: Time Series Target Encoding

Target encoding in time series requires special consideration to prevent future data leakage. This implementation ensures temporal coherence by using only historical data for encoding.

```python
def time_series_target_encoding(df, category_col, target_col, time_col):
    df = df.sort_values(time_col)
    encoded_values = np.zeros(len(df))
    
    for i in range(len(df)):
        # Use only past data for encoding
        historical_data = df.iloc[:i]
        if len(historical_data) > 0:
            category_means = historical_data.groupby(category_col)[target_col].mean()
            current_category = df.iloc[i][category_col]
            encoded_values[i] = category_means.get(current_category, 
                                                 historical_data[target_col].mean())
        else:
            encoded_values[i] = df[target_col].mean()
    
    return encoded_values
```

Slide 13: Handling High Cardinality Categories

High cardinality categorical variables present unique challenges in target encoding. This implementation addresses rare categories through hierarchical encoding and fallback strategies.

```python
class HierarchicalTargetEncoder:
    def __init__(self, min_samples_leaf=10):
        self.min_samples_leaf = min_samples_leaf
        self.encoding_map = {}
        self.fallback_value = None
        
    def fit(self, X, y):
        # Calculate global mean for rare categories
        self.fallback_value = y.mean()
        
        # Group categories by frequency
        value_counts = X.value_counts()
        frequent_categories = value_counts[value_counts >= self.min_samples_leaf].index
        
        # Calculate encoding for frequent categories
        for category in frequent_categories:
            mask = X == category
            self.encoding_map[category] = y[mask].mean()
            
        return self
        
    def transform(self, X):
        return X.map(self.encoding_map).fillna(self.fallback_value)
```

Slide 14: Additional Resources

*   Machine Learning with Target Encoding: A Novel Categorical Variable Encoding Method [https://arxiv.org/abs/2011.13161](https://arxiv.org/abs/2011.13161)
*   Robust Target Encoding: A Novel Approach for Handling High-Cardinality Features [https://arxiv.org/abs/2001.07248](https://arxiv.org/abs/2001.07248)
*   Time Series Target Encoding: Temporal Aspects of Categorical Feature Engineering [https://arxiv.org/abs/2103.09605](https://arxiv.org/abs/2103.09605)
*   For more resources, search on Google Scholar using keywords: "target encoding", "categorical encoding", "feature engineering machine learning"
*   Recommended reading: "Feature Engineering for Machine Learning" by Alice Zheng and Amanda Casari

