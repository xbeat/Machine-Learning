## Mastering Feature Discretization for Non-Linear Modeling
Slide 1: Understanding Feature Discretization

Feature discretization transforms continuous variables into categorical ones through binning. This process enables linear models to capture non-linear patterns by creating discrete boundaries that approximate complex relationships. The transformation typically uses techniques like equal-width or equal-frequency binning.

```python
import numpy as np
import pandas as pd

# Sample continuous data
ages = np.random.normal(40, 15, 1000)
df = pd.DataFrame({'age': ages})

# Equal-width binning
df['age_binned'] = pd.cut(df['age'], 
                         bins=[0, 25, 50, 75, 100],
                         labels=['Youth', 'Adult', 'Middle', 'Senior'])

# One-hot encoding
age_dummies = pd.get_dummies(df['age_binned'], prefix='age')
print(df.head())
print("\nOne-hot encoded features:\n", age_dummies.head())
```

Slide 2: Implementing Equal-Width Binning from Scratch

Equal-width binning divides the range of values into k intervals of equal size. This implementation demonstrates how to create bins manually without using pandas, providing more control over the discretization process and better understanding of the underlying mechanics.

```python
def equal_width_binning(data, num_bins):
    min_val, max_val = min(data), max(data)
    bin_width = (max_val - min_val) / num_bins
    
    bins = []
    for i in range(num_bins):
        lower = min_val + i * bin_width
        upper = min_val + (i + 1) * bin_width
        bins.append((lower, upper))
    
    # Assign data points to bins
    binned_data = []
    for value in data:
        for idx, (lower, upper) in enumerate(bins):
            if lower <= value <= upper:
                binned_data.append(idx)
                break
    
    return np.array(binned_data), bins

# Example usage
data = np.random.normal(50, 15, 1000)
binned_values, bin_edges = equal_width_binning(data, 5)
print(f"Original value: {data[0]:.2f}")
print(f"Binned value: {binned_values[0]}")
```

Slide 3: Equal-Frequency Binning Implementation

Equal-frequency binning ensures that each bin contains approximately the same number of samples. This approach is particularly useful when dealing with skewed distributions where equal-width binning might create bins with very different populations.

```python
def equal_frequency_binning(data, num_bins):
    n = len(data)
    samples_per_bin = n // num_bins
    sorted_data = np.sort(data)
    
    bins = []
    binned_data = np.zeros(n)
    
    for i in range(num_bins):
        start_idx = i * samples_per_bin
        end_idx = (i + 1) * samples_per_bin if i < num_bins - 1 else n
        bin_values = sorted_data[start_idx:end_idx]
        bins.append((min(bin_values), max(bin_values)))
    
    # Assign original data to bins
    for i, value in enumerate(data):
        for bin_idx, (lower, upper) in enumerate(bins):
            if lower <= value <= upper:
                binned_data[i] = bin_idx
                break
    
    return binned_data, bins

# Example usage
data = np.random.exponential(scale=2.0, size=1000)
binned_values, bin_edges = equal_frequency_binning(data, 5)
print(f"Bin edges: {bin_edges}")
print(f"Sample distribution: {np.bincount(binned_values.astype(int))}")
```

Slide 4: Adaptive Binning Strategy

Adaptive binning adjusts bin widths based on data distribution, using statistical measures like variance or entropy. This approach optimizes information retention while reducing dimensionality, particularly effective for features with non-uniform distributions.

```python
import numpy as np
from scipy import stats

def adaptive_binning(data, min_bins=3, max_bins=10, threshold=0.05):
    best_bins = min_bins
    max_variance_reduction = 0
    
    for n_bins in range(min_bins, max_bins + 1):
        # Calculate initial variance
        total_variance = np.var(data)
        
        # Try binning with n_bins
        hist, bin_edges = np.histogram(data, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Assign each point to nearest bin center
        digitized = np.digitize(data, bin_edges)
        binned_data = bin_centers[np.clip(digitized - 1, 0, len(bin_centers) - 1)]
        
        # Calculate variance after binning
        binned_variance = np.var(binned_data)
        variance_reduction = (total_variance - binned_variance) / total_variance
        
        if variance_reduction > max_variance_reduction:
            max_variance_reduction = variance_reduction
            best_bins = n_bins
            
        if variance_reduction > (1 - threshold):
            break
    
    return best_bins

# Example usage
np.random.seed(42)
data = np.concatenate([
    np.random.normal(0, 1, 300),
    np.random.normal(4, 0.5, 200)
])

optimal_bins = adaptive_binning(data)
print(f"Optimal number of bins: {optimal_bins}")

# Visualize results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.hist(data, bins=optimal_bins, density=True, alpha=0.7)
plt.title(f"Adaptive Binning Result (n_bins={optimal_bins})")
plt.show()
```

Slide 5: Feature Discretization for Linear Regression

This implementation demonstrates how feature discretization can improve linear regression performance on non-linear data by creating piece-wise linear approximations of complex relationships.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# Generate non-linear data
X = np.linspace(0, 10, 1000).reshape(-1, 1)
y = np.sin(X) + 0.1 * np.random.randn(1000)

def create_discretized_features(X, num_bins):
    bins = np.linspace(X.min(), X.max(), num_bins + 1)
    X_binned = np.digitize(X, bins)
    X_encoded = np.zeros((len(X), num_bins))
    for i in range(len(X)):
        bin_idx = X_binned[i] - 1
        if bin_idx < num_bins:
            X_encoded[i, bin_idx] = 1
    return X_encoded

# Compare regular vs discretized linear regression
X_disc = create_discretized_features(X, num_bins=20)

# Fit models
reg_model = LinearRegression().fit(X, y)
disc_model = LinearRegression().fit(X_disc, y)

# Predictions
y_pred_reg = reg_model.predict(X)
y_pred_disc = disc_model.predict(X_disc)

# Calculate R2 scores
r2_regular = r2_score(y, y_pred_reg)
r2_discretized = r2_score(y, y_pred_disc)

print(f"R2 Score (Regular): {r2_regular:.4f}")
print(f"R2 Score (Discretized): {r2_discretized:.4f}")
```

Slide 6: Entropy-Based Discretization

Entropy-based discretization uses information gain to determine optimal bin boundaries. This method is particularly effective for classification tasks as it creates bins that maximize class separation.

```python
import numpy as np
from scipy.stats import entropy

def calculate_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return entropy(probabilities, base=2)

def entropy_based_split(X, y, threshold):
    initial_entropy = calculate_entropy(y)
    best_gain = 0
    best_threshold = None
    
    left_mask = X <= threshold
    right_mask = ~left_mask
    
    if len(y[left_mask]) > 0 and len(y[right_mask]) > 0:
        left_entropy = calculate_entropy(y[left_mask])
        right_entropy = calculate_entropy(y[right_mask])
        
        # Calculate weighted average entropy
        weighted_entropy = (
            (len(y[left_mask]) * left_entropy + 
             len(y[right_mask]) * right_entropy) / len(y)
        )
        
        information_gain = initial_entropy - weighted_entropy
        return information_gain
    
    return 0

def entropy_based_discretization(X, y, min_samples=50):
    boundaries = []
    
    def recursive_split(X, y, start, end):
        if end - start < min_samples:
            return
        
        possible_thresholds = np.percentile(
            X[start:end], 
            q=range(10, 100, 10)
        )
        
        best_gain = 0
        best_threshold = None
        
        for threshold in possible_thresholds:
            gain = entropy_based_split(X[start:end], y[start:end], threshold)
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        
        if best_threshold is not None:
            boundaries.append(best_threshold)
            split_point = start + np.sum(X[start:end] <= best_threshold)
            recursive_split(X, y, start, split_point)
            recursive_split(X, y, split_point, end)
    
    recursive_split(X, y, 0, len(X))
    return np.sort(boundaries)

# Example usage
np.random.seed(42)
X = np.concatenate([
    np.random.normal(0, 1, 300),
    np.random.normal(4, 0.5, 200)
])
y = (X > 2).astype(int)

boundaries = entropy_based_discretization(X, y)
print(f"Optimal boundaries: {boundaries}")
```

Slide 7: Feature Discretization for Time Series Data

Time series data often benefits from discretization when dealing with cyclical patterns or seasonal effects. This implementation shows how to discretize temporal features while preserving their sequential nature and periodic patterns.

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TimeSeriesDiscretizer:
    def __init__(self, granularity='hour'):
        self.granularity = granularity
        self.mappings = {
            'hour': (24, lambda x: x.hour),
            'dayofweek': (7, lambda x: x.dayofweek),
            'month': (12, lambda x: x.month - 1)
        }
    
    def fit_transform(self, timestamps):
        n_bins, extractor = self.mappings[self.granularity]
        
        # Convert timestamps to datetime if needed
        if isinstance(timestamps[0], str):
            timestamps = pd.to_datetime(timestamps)
        
        # Extract temporal feature and create cyclic encoding
        feature_values = extractor(timestamps)
        transformed = self._create_cyclic_features(feature_values, n_bins)
        
        return transformed
    
    def _create_cyclic_features(self, values, n_bins):
        # Create sine and cosine transformations
        sin_values = np.sin(2 * np.pi * values / n_bins)
        cos_values = np.cos(2 * np.pi * values / n_bins)
        
        return np.column_stack([sin_values, cos_values])

# Example usage
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
values = np.random.normal(0, 1, 1000)

# Add synthetic pattern
values += np.sin(2 * np.pi * pd.to_datetime(dates).hour / 24) * 2

# Create discretizer
discretizer = TimeSeriesDiscretizer(granularity='hour')
transformed = discretizer.fit_transform(dates)

# Demonstrate effectiveness with simple linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Compare regular hours vs transformed features
X_regular = pd.to_datetime(dates).hour.values.reshape(-1, 1)
X_transformed = transformed

model_regular = LinearRegression().fit(X_regular, values)
model_transformed = LinearRegression().fit(X_transformed, values)

print(f"R2 Score (Regular): {r2_score(values, model_regular.predict(X_regular)):.4f}")
print(f"R2 Score (Transformed): {r2_score(values, model_transformed.predict(X_transformed)):.4f}")
```

Slide 8: MDL-Based Discretization

Minimum Description Length (MDL) principle provides a theoretically sound approach to discretization by finding the optimal trade-off between model complexity and data compression efficiency.

```python
import numpy as np
from math import log2
from scipy.stats import entropy

class MDLDiscretizer:
    def __init__(self, min_bins=2, max_bins=20):
        self.min_bins = min_bins
        self.max_bins = max_bins
        
    def _calculate_mdl_cost(self, data, boundaries):
        n = len(data)
        k = len(boundaries) + 1  # number of bins
        
        # Model complexity cost
        model_cost = k * log2(n)
        
        # Data encoding cost
        bins = np.digitize(data, boundaries)
        bin_counts = np.bincount(bins)
        data_cost = 0
        
        for count in bin_counts:
            if count > 0:
                p = count / n
                data_cost -= count * log2(p)
                
        return model_cost + data_cost
    
    def fit_transform(self, data):
        data = np.asarray(data)
        best_boundaries = None
        min_cost = np.inf
        
        for n_bins in range(self.min_bins, self.max_bins + 1):
            # Generate candidate boundaries
            percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
            boundaries = np.percentile(data, percentiles)
            
            # Calculate MDL cost
            cost = self._calculate_mdl_cost(data, boundaries)
            
            if cost < min_cost:
                min_cost = cost
                best_boundaries = boundaries
        
        # Transform data using best boundaries
        return np.digitize(data, best_boundaries)

# Example usage
np.random.seed(42)
# Generate mixture of gaussians
data = np.concatenate([
    np.random.normal(0, 1, 500),
    np.random.normal(4, 0.5, 300),
    np.random.normal(8, 0.8, 200)
])

discretizer = MDLDiscretizer()
discretized = discretizer.fit_transform(data)

# Calculate information content
original_entropy = entropy(np.histogram(data, bins='auto')[0])
discretized_entropy = entropy(np.bincount(discretized))

print(f"Original entropy: {original_entropy:.4f}")
print(f"Discretized entropy: {discretized_entropy:.4f}")
print(f"Number of unique bins: {len(np.unique(discretized))}")
```

Slide 9: Multivariate Feature Discretization

Multivariate discretization considers relationships between features during the binning process. This approach is particularly useful when features have strong correlations or interact in complex ways to influence the target variable.

```python
import numpy as np
from sklearn.mixture import GaussianMixture

class MultivariateDiscretizer:
    def __init__(self, n_components=3, random_state=42):
        self.n_components = n_components
        self.gmm = GaussianMixture(
            n_components=n_components,
            random_state=random_state
        )
        
    def fit_transform(self, X):
        # Fit GMM to identify natural clusters in multivariate space
        self.gmm.fit(X)
        
        # Get cluster assignments
        clusters = self.gmm.predict(X)
        
        # Calculate cluster probabilities for each point
        probs = self.gmm.predict_proba(X)
        
        # Create discretized features based on cluster memberships
        discretized = np.zeros((X.shape[0], self.n_components))
        discretized[np.arange(len(clusters)), clusters] = 1
        
        return discretized, probs
    
    def transform_single_feature(self, X, feature_idx):
        # Project points onto specific feature dimension
        means = self.gmm.means_[:, feature_idx]
        sorted_indices = np.argsort(means)
        
        # Assign points to nearest component
        clusters = self.gmm.predict(X)
        
        # Remap clusters based on feature ordering
        mapping = {old: new for new, old in enumerate(sorted_indices)}
        remapped = np.array([mapping[c] for c in clusters])
        
        return remapped

# Example usage with correlated features
np.random.seed(42)
n_samples = 1000

# Generate correlated features
X1 = np.random.normal(0, 1, n_samples)
X2 = X1 * 0.7 + np.random.normal(0, 0.5, n_samples)
X = np.column_stack([X1, X2])

# Apply multivariate discretization
discretizer = MultivariateDiscretizer(n_components=4)
X_disc, probs = discretizer.fit_transform(X)

# Get feature-specific discretization
X1_disc = discretizer.transform_single_feature(X, 0)
X2_disc = discretizer.transform_single_feature(X, 1)

print("Original shape:", X.shape)
print("Discretized shape:", X_disc.shape)
print("\nFeature 1 unique values:", np.unique(X1_disc))
print("Feature 2 unique values:", np.unique(X2_disc))
```

Slide 10: ChiMerge Discretization Algorithm

ChiMerge is a bottom-up discretization method that uses chi-square statistics to determine when to merge adjacent intervals. It's particularly effective for classification tasks as it maintains class discrimination.

```python
import numpy as np
from scipy.stats import chi2_contingency

class ChiMergeDiscretizer:
    def __init__(self, max_intervals=6, significance_level=0.05):
        self.max_intervals = max_intervals
        self.significance_level = significance_level
        self.boundaries = None
        
    def _calculate_chi_square(self, interval1, interval2, y1, y2):
        # Create contingency table
        classes = np.unique(np.concatenate([y1, y2]))
        cont_table = np.zeros((2, len(classes)))
        
        for i, yi in enumerate([y1, y2]):
            for j, c in enumerate(classes):
                cont_table[i, j] = np.sum(yi == c)
                
        # Calculate chi-square statistic
        if cont_table.sum() == 0:
            return 0
        
        chi2, _, _, _ = chi2_contingency(cont_table)
        return chi2
    
    def fit_transform(self, X, y):
        sorted_indices = np.argsort(X)
        X = X[sorted_indices]
        y = y[sorted_indices]
        
        # Initialize boundaries at unique values
        unique_vals = np.unique(X)
        boundaries = unique_vals[:-1] + np.diff(unique_vals)/2
        
        while len(boundaries) > self.max_intervals - 1:
            chi_squares = []
            
            # Calculate chi-square for adjacent intervals
            for i in range(len(boundaries) - 1):
                mask1 = (X >= boundaries[i]) & (X < boundaries[i+1])
                mask2 = (X >= boundaries[i+1]) & (X < boundaries[i+2] 
                        if i+2 < len(boundaries) else np.inf)
                
                chi2 = self._calculate_chi_square(
                    X[mask1], X[mask2],
                    y[mask1], y[mask2]
                )
                chi_squares.append(chi2)
            
            # Merge intervals with lowest chi-square
            min_chi2_idx = np.argmin(chi_squares)
            boundaries = np.delete(boundaries, min_chi2_idx + 1)
        
        self.boundaries = boundaries
        return np.digitize(X, self.boundaries)

# Example usage
np.random.seed(42)

# Generate synthetic classification data
X = np.concatenate([
    np.random.normal(0, 1, 300),
    np.random.normal(3, 1, 300),
    np.random.normal(6, 1, 400)
])
y = np.concatenate([
    np.zeros(300),
    np.ones(300),
    2 * np.ones(400)
])

discretizer = ChiMergeDiscretizer(max_intervals=5)
X_disc = discretizer.fit_transform(X, y)

print(f"Number of boundaries: {len(discretizer.boundaries)}")
print(f"Boundaries: {discretizer.boundaries}")
print(f"Unique discretized values: {np.unique(X_disc)}")
```

Slide 11: Real-World Application - Customer Segmentation

This implementation demonstrates feature discretization in a customer segmentation scenario, where continuous features like age, income, and purchase frequency are transformed to create meaningful customer segments.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class CustomerSegmentationDiscretizer:
    def __init__(self):
        self.age_bins = [0, 25, 35, 50, 65, 100]
        self.income_bins = [0, 30000, 60000, 100000, 150000, np.inf]
        self.frequency_bins = [0, 2, 5, 10, 20, np.inf]
        
    def transform(self, df):
        df_transformed = pd.DataFrame()
        
        # Age discretization
        df_transformed['age_segment'] = pd.cut(
            df['age'],
            bins=self.age_bins,
            labels=['Gen-Z', 'Young Adult', 'Adult', 'Middle-Age', 'Senior']
        )
        
        # Income discretization
        df_transformed['income_segment'] = pd.cut(
            df['annual_income'],
            bins=self.income_bins,
            labels=['Low', 'Lower-Mid', 'Middle', 'Upper-Mid', 'High']
        )
        
        # Purchase frequency discretization
        df_transformed['frequency_segment'] = pd.cut(
            df['purchase_frequency'],
            bins=self.frequency_bins,
            labels=['Rare', 'Low', 'Medium', 'High', 'VIP']
        )
        
        # One-hot encode all segments
        return pd.get_dummies(df_transformed, prefix_sep='_')

# Generate synthetic customer data
np.random.seed(42)
n_customers = 1000

customer_data = pd.DataFrame({
    'age': np.random.normal(40, 15, n_customers),
    'annual_income': np.random.lognormal(10.5, 0.5, n_customers),
    'purchase_frequency': np.random.gamma(2, 3, n_customers),
    'customer_value': np.zeros(n_customers)  # Target variable
})

# Create target variable based on complex rules
customer_data['customer_value'] = (
    (customer_data['age'] > 35) & 
    (customer_data['annual_income'] > 60000) & 
    (customer_data['purchase_frequency'] > 5)
).astype(int)

# Apply discretization
discretizer = CustomerSegmentationDiscretizer()
X_transformed = discretizer.transform(customer_data)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_transformed, customer_data['customer_value'])

# Analyze feature importance
feature_importance = pd.DataFrame({
    'feature': X_transformed.columns,
    'importance': np.abs(model.coef_[0])
})
print("Top 5 most important segments:")
print(feature_importance.sort_values('importance', ascending=False).head())
```

Slide 12: Real-World Application - Geospatial Analysis

This implementation shows how to discretize geographical coordinates into meaningful regions while preserving spatial relationships, particularly useful for location-based analysis and modeling.

```python
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

class GeospatialDiscretizer:
    def __init__(self, eps=0.1, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.cluster_centers_ = None
        self.cluster_boundaries_ = None
        
    def fit_transform(self, coordinates):
        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='haversine'
        ).fit(np.radians(coordinates))
        
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Calculate cluster centers and boundaries
        self.cluster_centers_ = []
        self.cluster_boundaries_ = []
        
        for i in range(n_clusters):
            mask = labels == i
            cluster_points = coordinates[mask]
            
            # Calculate centroid
            center = np.mean(cluster_points, axis=0)
            self.cluster_centers_.append(center)
            
            # Calculate convex hull for boundary
            if len(cluster_points) >= 3:
                hull = ConvexHull(cluster_points)
                boundary = cluster_points[hull.vertices]
            else:
                boundary = cluster_points
                
            self.cluster_boundaries_.append(boundary)
            
        return labels

# Generate synthetic location data
np.random.seed(42)
n_points = 1000

# Create multiple clusters of points
centers = [
    (40.7128, -74.0060),  # New York
    (34.0522, -118.2437), # Los Angeles
    (41.8781, -87.6298)   # Chicago
]

locations = []
for center in centers:
    cluster = np.random.normal(
        loc=center, 
        scale=[0.1, 0.1], 
        size=(n_points // len(centers), 2)
    )
    locations.append(cluster)

locations = np.vstack(locations)

# Create DataFrame with location data
location_data = pd.DataFrame(
    locations, 
    columns=['latitude', 'longitude']
)

# Apply spatial discretization
discretizer = GeospatialDiscretizer(eps=0.1, min_samples=5)
location_data['region'] = discretizer.fit_transform(
    location_data[['latitude', 'longitude']].values
)

# Print statistics
print("Number of regions:", len(np.unique(location_data['region'])))
print("\nPoints per region:")
print(location_data['region'].value_counts())
```

Slide 13: Advanced Feature Discretization with Dynamic Programming

This implementation uses dynamic programming to find the optimal binning strategy that minimizes information loss while maintaining interpretability. The algorithm considers both the global and local structure of the data.

```python
import numpy as np
from scipy.stats import entropy
import pandas as pd

class DPDiscretizer:
    def __init__(self, max_bins=10, min_samples=30):
        self.max_bins = max_bins
        self.min_samples = min_samples
        self.boundaries = None
        
    def _calculate_bin_cost(self, values, y_values):
        if len(values) < self.min_samples:
            return np.inf
            
        # Calculate class distribution in bin
        class_counts = np.bincount(y_values)
        probabilities = class_counts / len(y_values)
        
        # Calculate entropy of the bin
        bin_entropy = entropy(probabilities)
        
        # Add penalty for bin size
        size_penalty = -np.log(len(values) / self.min_samples)
        
        return bin_entropy + size_penalty
        
    def _find_optimal_bins(self, x, y):
        n = len(x)
        
        # Initialize dynamic programming tables
        dp = np.full((n + 1, self.max_bins + 1), np.inf)
        split_points = np.zeros((n + 1, self.max_bins + 1), dtype=int)
        
        # Base case: zero cost for empty sequence
        dp[0, 0] = 0
        
        # Fill dynamic programming table
        for i in range(1, n + 1):
            for k in range(1, min(i + 1, self.max_bins + 1)):
                for j in range(i):
                    cost = (dp[j, k-1] + 
                           self._calculate_bin_cost(x[j:i], y[j:i]))
                    
                    if cost < dp[i, k]:
                        dp[i, k] = cost
                        split_points[i, k] = j
        
        # Reconstruct optimal boundaries
        boundaries = []
        pos = n
        k = self.max_bins
        
        while k > 0 and pos > 0:
            boundaries.append(x[split_points[pos, k]])
            pos = split_points[pos, k]
            k -= 1
            
        return sorted(boundaries)

    def fit_transform(self, X, y):
        # Sort data by feature value
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        
        # Find optimal bin boundaries
        self.boundaries = self._find_optimal_bins(X_sorted, y_sorted)
        
        # Transform data using boundaries
        return np.digitize(X, self.boundaries)

# Example usage with complex non-linear pattern
np.random.seed(42)

# Generate synthetic data with multiple class regions
def generate_complex_data(n_samples=1000):
    X = np.random.uniform(0, 10, n_samples)
    y = np.zeros(n_samples)
    
    # Create complex class boundaries
    y[(X > 2) & (X < 4)] = 1
    y[(X > 6) & (X < 7)] = 2
    y[X > 8] = 3
    
    # Add noise
    noise_idx = np.random.choice(
        n_samples, 
        size=int(0.1 * n_samples), 
        replace=False
    )
    y[noise_idx] = np.random.randint(0, 4, len(noise_idx))
    
    return X, y

X, y = generate_complex_data()

# Apply discretization
discretizer = DPDiscretizer(max_bins=8)
X_disc = discretizer.fit_transform(X, y)

# Analyze results
results = pd.DataFrame({
    'original_value': X,
    'discretized_value': X_disc,
    'class': y
})

print("Discovered boundaries:", discretizer.boundaries)
print("\nClass distribution in each bin:")
for bin_idx in range(len(discretizer.boundaries) + 1):
    mask = X_disc == bin_idx
    class_dist = np.bincount(y[mask].astype(int))
    print(f"\nBin {bin_idx}:")
    print(f"Samples: {sum(mask)}")
    print(f"Class distribution: {class_dist}")
```

Slide 14: Additional Resources

*   "Optimal Discretization Using Information Theory" - [https://arxiv.org/abs/1401.1914](https://arxiv.org/abs/1401.1914)
*   "MDL-Based Discretization Methods for Classification" - [https://arxiv.org/abs/1509.00922](https://arxiv.org/abs/1509.00922)
*   "A Survey of Discretization Techniques: Taxonomy and Empirical Analysis" - [https://arxiv.org/abs/1405.4534](https://arxiv.org/abs/1405.4534)
*   "Feature Engineering and Selection: A Practical Approach" - Search on Google for "Max Kuhn Feature Engineering Book"
*   "Statistical and Machine Learning Methods for Discretization" - Search for "Dougherty et al. Discretization Survey"
*   "Dynamic Programming Algorithms for Feature Discretization" - Search for "Fayyad & Irani MDL Discretization"

Note: Some recommended search terms are provided since direct URLs may not be available for all resources.

