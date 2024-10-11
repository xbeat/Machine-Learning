## Multivariate Sampling Techniques in Python for Data Science
Slide 1: Multivariate Sampling Techniques in Data Science

Multivariate sampling techniques are essential tools in data science for collecting and analyzing data with multiple variables. These methods help researchers and data scientists gather representative samples from complex datasets, enabling more accurate insights and predictions. In this presentation, we'll explore various multivariate sampling techniques and their implementation using Python.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate a sample multivariate dataset
X, _ = make_blobs(n_samples=1000, n_features=3, centers=5, random_state=42)
df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2', 'Feature 3'])

# Visualize the dataset
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Feature 1'], df['Feature 2'], df['Feature 3'])
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title('Multivariate Dataset Visualization')
plt.show()
```

Slide 2: Simple Random Sampling

Simple random sampling is a basic technique where each data point has an equal probability of being selected. This method is unbiased and easy to implement but may not always capture the full diversity of the dataset, especially in multivariate scenarios.

```python
def simple_random_sampling(df, sample_size):
    return df.sample(n=sample_size, random_state=42)

sample_size = 100
srs_sample = simple_random_sampling(df, sample_size)

print(f"Original dataset shape: {df.shape}")
print(f"Simple random sample shape: {srs_sample.shape}")

# Visualize the sample
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(srs_sample['Feature 1'], srs_sample['Feature 2'], srs_sample['Feature 3'])
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title('Simple Random Sample Visualization')
plt.show()
```

Slide 3: Stratified Sampling

Stratified sampling divides the population into subgroups (strata) based on specific characteristics and then samples from each stratum. This technique ensures representation from all subgroups, making it particularly useful for imbalanced datasets.

```python
from sklearn.preprocessing import KBinsDiscretizer

def stratified_sampling(df, sample_size, strata_column):
    # Create strata based on a specific column
    kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    df['strata'] = kbd.fit_transform(df[[strata_column]])
    
    # Calculate the proportion of each stratum
    strata_counts = df['strata'].value_counts(normalize=True)
    
    # Sample from each stratum
    stratified_sample = pd.DataFrame()
    for stratum, proportion in strata_counts.items():
        stratum_size = int(sample_size * proportion)
        stratum_sample = df[df['strata'] == stratum].sample(n=stratum_size, random_state=42)
        stratified_sample = pd.concat([stratified_sample, stratum_sample])
    
    return stratified_sample.drop('strata', axis=1)

sample_size = 100
stratified_sample = stratified_sampling(df, sample_size, 'Feature 1')

print(f"Original dataset shape: {df.shape}")
print(f"Stratified sample shape: {stratified_sample.shape}")

# Visualize the stratified sample
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(stratified_sample['Feature 1'], stratified_sample['Feature 2'], stratified_sample['Feature 3'])
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title('Stratified Sample Visualization')
plt.show()
```

Slide 4: Cluster Sampling

Cluster sampling involves dividing the population into clusters, randomly selecting clusters, and then sampling all or some units within the chosen clusters. This technique is useful when it's impractical or expensive to sample from the entire population.

```python
from sklearn.cluster import KMeans

def cluster_sampling(df, n_clusters, sample_size):
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df)
    
    # Randomly select clusters
    selected_clusters = np.random.choice(n_clusters, size=sample_size//n_clusters, replace=False)
    
    # Sample from selected clusters
    cluster_sample = df[df['cluster'].isin(selected_clusters)]
    
    return cluster_sample.drop('cluster', axis=1)

n_clusters = 5
sample_size = 100
cluster_sample = cluster_sampling(df, n_clusters, sample_size)

print(f"Original dataset shape: {df.shape}")
print(f"Cluster sample shape: {cluster_sample.shape}")

# Visualize the cluster sample
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cluster_sample['Feature 1'], cluster_sample['Feature 2'], cluster_sample['Feature 3'])
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title('Cluster Sample Visualization')
plt.show()
```

Slide 5: Systematic Sampling

Systematic sampling involves selecting every kth item from the population, where k is the sampling interval. This method is simple to implement and can be effective for ordered data, but it may introduce bias if there are underlying patterns in the data.

```python
def systematic_sampling(df, sample_size):
    # Calculate the sampling interval
    interval = len(df) // sample_size
    
    # Select every kth item
    systematic_sample = df.iloc[::interval]
    
    return systematic_sample.head(sample_size)

sample_size = 100
systematic_sample = systematic_sampling(df, sample_size)

print(f"Original dataset shape: {df.shape}")
print(f"Systematic sample shape: {systematic_sample.shape}")

# Visualize the systematic sample
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(systematic_sample['Feature 1'], systematic_sample['Feature 2'], systematic_sample['Feature 3'])
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title('Systematic Sample Visualization')
plt.show()
```

Slide 6: Weighted Sampling

Weighted sampling assigns different probabilities to each data point based on specific criteria. This technique is useful when certain observations are more important or representative than others.

```python
def weighted_sampling(df, sample_size, weight_column):
    # Normalize weights
    df['normalized_weights'] = df[weight_column] / df[weight_column].sum()
    
    # Perform weighted sampling
    weighted_sample = df.sample(n=sample_size, weights='normalized_weights', random_state=42)
    
    return weighted_sample.drop('normalized_weights', axis=1)

# Create a weight column based on Feature 1
df['weight'] = np.abs(df['Feature 1'])

sample_size = 100
weighted_sample = weighted_sampling(df, sample_size, 'weight')

print(f"Original dataset shape: {df.shape}")
print(f"Weighted sample shape: {weighted_sample.shape}")

# Visualize the weighted sample
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(weighted_sample['Feature 1'], weighted_sample['Feature 2'], weighted_sample['Feature 3'], c=weighted_sample['weight'])
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.colorbar(scatter, label='Weight')
plt.title('Weighted Sample Visualization')
plt.show()
```

Slide 7: Importance Sampling

Importance sampling is a technique used to estimate properties of a particular distribution while sampling from a different distribution. This method is particularly useful in Monte Carlo simulations and rare event sampling.

```python
import scipy.stats as stats

def importance_sampling(target_dist, proposal_dist, num_samples):
    # Generate samples from the proposal distribution
    samples = proposal_dist.rvs(size=num_samples)
    
    # Calculate importance weights
    weights = target_dist.pdf(samples) / proposal_dist.pdf(samples)
    
    # Normalize weights
    normalized_weights = weights / weights.sum()
    
    return samples, normalized_weights

# Define target and proposal distributions
target_dist = stats.norm(loc=0, scale=1)
proposal_dist = stats.norm(loc=1, scale=2)

num_samples = 1000
samples, weights = importance_sampling(target_dist, proposal_dist, num_samples)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, weights=weights, density=True, alpha=0.7, label='Weighted Samples')
x = np.linspace(-5, 5, 100)
plt.plot(x, target_dist.pdf(x), 'r-', lw=2, label='Target Distribution')
plt.plot(x, proposal_dist.pdf(x), 'g--', lw=2, label='Proposal Distribution')
plt.legend()
plt.title('Importance Sampling')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
```

Slide 8: Real-Life Example: Environmental Monitoring

Multivariate sampling techniques are crucial in environmental monitoring to efficiently collect data on various pollutants across different locations. Let's simulate a scenario where we measure air quality parameters in a city.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Generate synthetic air quality data
np.random.seed(42)
n_locations = 1000

data = {
    'Location': range(n_locations),
    'PM2.5': np.random.lognormal(3, 0.5, n_locations),
    'NO2': np.random.normal(40, 10, n_locations),
    'O3': np.random.normal(30, 5, n_locations),
    'Temperature': np.random.normal(25, 5, n_locations),
    'Humidity': np.random.normal(60, 10, n_locations)
}

df = pd.DataFrame(data)

# Normalize the data
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop('Location', axis=1)), columns=df.columns[1:])

# Perform stratified sampling based on PM2.5 levels
def stratified_sampling(df, sample_size, strata_column):
    df['strata'] = pd.qcut(df[strata_column], q=5, labels=False)
    return df.groupby('strata').apply(lambda x: x.sample(n=sample_size//5)).reset_index(drop=True)

sample_size = 100
stratified_sample = stratified_sampling(df_scaled, sample_size, 'PM2.5')

print(f"Original dataset shape: {df_scaled.shape}")
print(f"Stratified sample shape: {stratified_sample.shape}")

# Visualize the sample
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.scatter(df_scaled['PM2.5'], df_scaled['NO2'], alpha=0.3, label='All data')
plt.scatter(stratified_sample['PM2.5'], stratified_sample['NO2'], color='red', label='Stratified sample')
plt.xlabel('PM2.5 (normalized)')
plt.ylabel('NO2 (normalized)')
plt.title('Air Quality Monitoring: Stratified Sampling')
plt.legend()
plt.show()
```

Slide 9: Real-Life Example: Customer Satisfaction Survey

In this example, we'll use multivariate sampling techniques to conduct a customer satisfaction survey for a large e-commerce platform. The goal is to obtain a representative sample of customers across different demographics and purchase histories.

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Generate synthetic customer data
np.random.seed(42)
n_customers = 10000

data = {
    'CustomerID': range(n_customers),
    'Age': np.random.normal(35, 15, n_customers),
    'PurchaseFrequency': np.random.poisson(5, n_customers),
    'TotalSpend': np.random.exponential(100, n_customers),
    'ProductCategories': np.random.randint(1, 6, n_customers)
}

df = pd.DataFrame(data)

# Normalize the data
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop('CustomerID', axis=1)), columns=df.columns[1:])

# Perform cluster sampling
def cluster_sampling(df, n_clusters, sample_size):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df)
    return df.groupby('Cluster').apply(lambda x: x.sample(n=sample_size//n_clusters)).reset_index(drop=True)

n_clusters = 5
sample_size = 500
cluster_sample = cluster_sampling(df_scaled, n_clusters, sample_size)

print(f"Original dataset shape: {df_scaled.shape}")
print(f"Cluster sample shape: {cluster_sample.shape}")

# Visualize the sample
plt.figure(figsize=(12, 8))
plt.scatter(df_scaled['Age'], df_scaled['TotalSpend'], alpha=0.3, label='All customers')
plt.scatter(cluster_sample['Age'], cluster_sample['TotalSpend'], color='red', label='Sampled customers')
plt.xlabel('Age (normalized)')
plt.ylabel('Total Spend (normalized)')
plt.title('Customer Satisfaction Survey: Cluster Sampling')
plt.legend()
plt.show()
```

Slide 10: Reservoir Sampling

Reservoir sampling is a technique used for sampling from a data stream or when the total number of items is unknown. It maintains a fixed-size sample as it processes the data, ensuring that each item has an equal probability of being included in the final sample.

```python
import numpy as np

def reservoir_sampling(stream, k):
    reservoir = []
    for i, item in enumerate(stream):
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            j = np.random.randint(0, i+1)
            if j < k:
                reservoir[j] = item
    return reservoir

# Simulate a data stream
np.random.seed(42)
data_stream = iter(np.random.normal(0, 1, 10000))

# Perform reservoir sampling
sample_size = 100
reservoir_sample = reservoir_sampling(data_stream, sample_size)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.hist(reservoir_sample, bins=20, density=True, alpha=0.7, label='Reservoir Sample')
x = np.linspace(-4, 4, 100)
plt.plot(x, stats.norm.pdf(x), 'r-', lw=2, label='True Distribution')
plt.legend()
plt.title('Reservoir Sampling from a Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
```

Slide 11: Gibbs Sampling

Gibbs sampling is a Markov Chain Monte Carlo (MCMC) algorithm used to obtain a sequence of observations which are approximated from a specified multivariate probability distribution. It's particularly useful when direct sampling is difficult but the conditional distribution of each variable is known.

```python
import numpy as np
import matplotlib.pyplot as plt

def gibbs_sampling(n_samples, mu, sigma):
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    
    x[0], y[0] = 0, 0
    
    for i in range(1, n_samples):
        x[i] = np.random.normal(mu[0] + sigma[0, 1] / sigma[1, 1] * (y[i-1] - mu[1]), 
                                np.sqrt(sigma[0, 0] - sigma[0, 1]**2 / sigma[1, 1]))
        y[i] = np.random.normal(mu[1] + sigma[0, 1] / sigma[0, 0] * (x[i] - mu[0]), 
                                np.sqrt(sigma[1, 1] - sigma[0, 1]**2 / sigma[0, 0]))
    
    return x, y

# Set parameters
n_samples = 5000
mu = np.array([0, 0])
sigma = np.array([[1, 0.8], [0.8, 1]])

# Run Gibbs sampling
x, y = gibbs_sampling(n_samples, mu, sigma)

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(x, y, alpha=0.1)
plt.title('Gibbs Sampling: Bivariate Normal Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

Slide 12: Latin Hypercube Sampling

Latin Hypercube Sampling (LHS) is a statistical method for generating a near-random sample of parameter values from a multidimensional distribution. It's often used in uncertainty quantification and sensitivity analysis.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

def latin_hypercube_sampling(n_samples, n_dimensions):
    sampler = qmc.LatinHypercube(d=n_dimensions)
    sample = sampler.random(n=n_samples)
    return sample

# Generate LHS samples
n_samples = 100
n_dimensions = 2
lhs_samples = latin_hypercube_sampling(n_samples, n_dimensions)

# Plot the samples
plt.figure(figsize=(10, 8))
plt.scatter(lhs_samples[:, 0], lhs_samples[:, 1])
plt.title('Latin Hypercube Sampling')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.show()
```

Slide 13: Importance of Multivariate Sampling in Machine Learning

Multivariate sampling techniques play a crucial role in machine learning, particularly in tasks such as cross-validation, bootstrapping, and imbalanced dataset handling. These methods help ensure that our models are trained and evaluated on representative data subsets.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Generate imbalanced dataset
X, y = make_blobs(n_samples=[900, 100], centers=[[0, 0], [2, 2]], random_state=42)
X = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
y = pd.Series(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model without sampling
clf_original = RandomForestClassifier(random_state=42)
clf_original.fit(X_train, y_train)
y_pred_original = clf_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train model with SMOTE
clf_smote = RandomForestClassifier(random_state=42)
clf_smote.fit(X_train_resampled, y_train_resampled)
y_pred_smote = clf_smote.predict(X_test)
accuracy_smote = accuracy_score(y_test, y_pred_smote)

print(f"Accuracy without sampling: {accuracy_original:.4f}")
print(f"Accuracy with SMOTE: {accuracy_smote:.4f}")

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(X_train['Feature 1'], X_train['Feature 2'], c=y_train, cmap='coolwarm')
ax1.set_title('Original Training Data')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

ax2.scatter(X_train_resampled['Feature 1'], X_train_resampled['Feature 2'], c=y_train_resampled, cmap='coolwarm')
ax2.set_title('SMOTE Resampled Training Data')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

Slide 14: Challenges and Considerations in Multivariate Sampling

When applying multivariate sampling techniques, it's important to consider various challenges:

1. Curse of dimensionality: As the number of dimensions increases, the sample size required to maintain representativeness grows exponentially.
2. Correlation between variables: Some sampling methods may not account for complex relationships between variables.
3. Outliers and extreme values: Certain sampling techniques may be sensitive to outliers, potentially skewing the results.
4. Computational complexity: Some advanced sampling methods can be computationally expensive, especially for large datasets.
5. Sample size determination: Choosing an appropriate sample size that balances representativeness and computational efficiency can be challenging.

To address these challenges, consider using adaptive sampling techniques, dimension reduction methods, or combining multiple sampling strategies based on the specific characteristics of your dataset and analysis goals.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate high-dimensional data
n_samples = 1000
n_features = 100
X = np.random.randn(n_samples, n_features)

# Apply PCA for dimension reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.5)
plt.title('PCA-reduced High-dimensional Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

print(f"Original data shape: {X.shape}")
print(f"Reduced data shape: {X_reduced.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

Slide 15: Additional Resources

For further exploration of multivariate sampling techniques in data science, consider the following resources:

1. ArXiv paper: "A Survey of Multivariate Sampling Techniques for Data Science" (arXiv:2104.12672) URL: [https://arxiv.org/abs/2104.12672](https://arxiv.org/abs/2104.12672)
2. ArXiv paper: "Adaptive Sampling Strategies for Multivariate Data" (arXiv:1903.07689) URL: [https://arxiv.org/abs/1903.07689](https://arxiv.org/abs/1903.07689)
3. ArXiv paper: "Importance Sampling in High Dimensions" (arXiv:1905.04605) URL: [https://arxiv.org/abs/1905.04605](https://arxiv.org/abs/1905.04605)

These papers provide in-depth discussions on various multivariate sampling techniques, their applications, and recent advancements in the field. They offer valuable insights for both beginners and advanced practitioners in data science and machine learning.

