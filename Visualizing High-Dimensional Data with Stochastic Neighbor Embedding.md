## Visualizing High-Dimensional Data with Stochastic Neighbor Embedding

Slide 1: Introduction to Stochastic Neighbor Embedding

Stochastic Neighbor Embedding (SNE) is a machine learning algorithm for dimensionality reduction, particularly useful for visualizing high-dimensional data in lower-dimensional spaces. It aims to preserve the local structure of the data while reducing its dimensionality, making it easier to visualize and analyze complex datasets.

```python
import random

def generate_high_dimensional_data(n_samples, n_dimensions):
    return [[random.gauss(0, 1) for _ in range(n_dimensions)] for _ in range(n_samples)]

# Generate a sample dataset
data = generate_high_dimensional_data(100, 50)
print(f"Generated {len(data)} samples with {len(data[0])} dimensions each")
```

Slide 2: The Curse of Dimensionality

The curse of dimensionality refers to various phenomena that arise when analyzing data in high-dimensional spaces. As the number of dimensions increases, the volume of the space increases exponentially, making data sparse and difficult to analyze. SNE helps address this issue by reducing the dimensionality while preserving important relationships.

```python
def euclidean_distance(x, y):
    return sum((a - b) ** 2 for a, b in zip(x, y)) ** 0.5

def average_pairwise_distance(data):
    n = len(data)
    total_distance = sum(euclidean_distance(data[i], data[j])
                         for i in range(n) for j in range(i+1, n))
    return total_distance / (n * (n - 1) / 2)

# Calculate average pairwise distance
avg_distance = average_pairwise_distance(data)
print(f"Average pairwise distance: {avg_distance:.2f}")
```

Slide 3: Probability Distribution in High-dimensional Space

SNE starts by computing the probability distribution of pairs of high-dimensional objects. For each object, it calculates the probability of picking another object as its neighbor, based on their proximity. This probability is proportional to the similarity between the two objects, typically measured using Gaussian kernel.

```python
import math

def gaussian_kernel(x, y, sigma):
    return math.exp(-euclidean_distance(x, y)**2 / (2 * sigma**2))

def compute_pairwise_affinities(data, sigma):
    n = len(data)
    affinities = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            aff = gaussian_kernel(data[i], data[j], sigma)
            affinities[i][j] = affinities[j][i] = aff
    return affinities

# Compute pairwise affinities
sigma = 1.0
affinities = compute_pairwise_affinities(data[:5], sigma)  # Using first 5 points for brevity
print("Pairwise affinities (first 5x5):")
for row in affinities:
    print(" ".join(f"{x:.2f}" for x in row[:5]))
```

Slide 4: Probability Distribution in Low-dimensional Space

SNE then creates a similar probability distribution for the points in the low-dimensional space. The goal is to make these two distributions as similar as possible. This is achieved by minimizing the Kullback-Leibler divergence between the two distributions.

```python
def compute_low_dim_affinities(low_dim_data, sigma):
    return compute_pairwise_affinities(low_dim_data, sigma)

# Initialize low-dimensional representations randomly
low_dim_data = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(5)]

low_dim_affinities = compute_low_dim_affinities(low_dim_data, sigma)
print("\nLow-dimensional affinities (first 5x5):")
for row in low_dim_affinities:
    print(" ".join(f"{x:.2f}" for x in row[:5]))
```

Slide 5: Kullback-Leibler Divergence

The Kullback-Leibler (KL) divergence is a measure of the difference between two probability distributions. In SNE, we use it to quantify how well the low-dimensional representation preserves the high-dimensional structure. The goal is to minimize this divergence.

```python
def kl_divergence(P, Q):
    kl_div = 0
    for i in range(len(P)):
        for j in range(len(P[i])):
            if P[i][j] > 1e-12 and Q[i][j] > 1e-12:
                kl_div += P[i][j] * math.log(P[i][j] / Q[i][j])
    return kl_div

# Calculate KL divergence
kl_div = kl_divergence(affinities, low_dim_affinities)
print(f"KL divergence: {kl_div:.4f}")
```

Slide 6: Gradient Descent Optimization

To minimize the KL divergence, SNE uses gradient descent. This iterative process adjusts the positions of the low-dimensional points to better match the high-dimensional distribution. The gradient of the KL divergence with respect to the low-dimensional points guides this optimization.

```python
def gradient_step(high_dim_affinities, low_dim_data, learning_rate):
    n = len(low_dim_data)
    gradient = [[0, 0] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = [low_dim_data[i][d] - low_dim_data[j][d] for d in range(2)]
                factor = 4 * (high_dim_affinities[i][j] - low_dim_affinities[i][j])
                gradient[i][0] += factor * diff[0]
                gradient[i][1] += factor * diff[1]
    
    # Update low_dim_data
    for i in range(n):
        low_dim_data[i][0] += learning_rate * gradient[i][0]
        low_dim_data[i][1] += learning_rate * gradient[i][1]

# Perform one gradient step
learning_rate = 0.1
gradient_step(affinities, low_dim_data, learning_rate)
print("Updated low-dimensional data:")
for point in low_dim_data:
    print(f"({point[0]:.2f}, {point[1]:.2f})")
```

Slide 7: Perplexity and Sigma

Perplexity is a hyperparameter in SNE that balances local and global aspects of the data. It's related to the number of effective neighbors each point has. The sigma parameter of the Gaussian kernel is adjusted to achieve the desired perplexity.

```python
def binary_search_sigma(distances, target_perplexity, tolerance=1e-5, max_iterations=50):
    sigma_min, sigma_max = 1e-20, 1e20
    
    for _ in range(max_iterations):
        sigma = (sigma_min + sigma_max) / 2
        probs = [math.exp(-d / (2 * sigma**2)) for d in distances]
        sum_probs = sum(probs)
        probs = [p / sum_probs for p in probs]
        
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        perplexity = 2 ** entropy
        
        if abs(perplexity - target_perplexity) < tolerance:
            return sigma
        
        if perplexity > target_perplexity:
            sigma_max = sigma
        else:
            sigma_min = sigma
    
    return sigma

# Example usage
distances = [1.5, 2.0, 2.5, 3.0, 3.5]
target_perplexity = 5
optimal_sigma = binary_search_sigma(distances, target_perplexity)
print(f"Optimal sigma for perplexity {target_perplexity}: {optimal_sigma:.4f}")
```

Slide 8: t-SNE: An Improvement on SNE

t-Distributed Stochastic Neighbor Embedding (t-SNE) is an improved version of SNE. It uses a Student's t-distribution in the low-dimensional space instead of a Gaussian, which helps alleviate the "crowding problem" and makes optimization easier.

```python
def t_distribution(x, y):
    return 1 / (1 + euclidean_distance(x, y)**2)

def compute_t_sne_affinities(low_dim_data):
    n = len(low_dim_data)
    affinities = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            aff = t_distribution(low_dim_data[i], low_dim_data[j])
            affinities[i][j] = affinities[j][i] = aff
    return affinities

# Compute t-SNE affinities
t_sne_affinities = compute_t_sne_affinities(low_dim_data)
print("t-SNE affinities (first 5x5):")
for row in t_sne_affinities:
    print(" ".join(f"{x:.4f}" for x in row[:5]))
```

Slide 9: Momentum in t-SNE

t-SNE incorporates momentum in its gradient descent to speed up optimization and avoid local minima. This technique accumulates a velocity vector for each point, which is updated along with the position in each iteration.

```python
def t_sne_step(high_dim_affinities, low_dim_data, velocities, learning_rate, momentum):
    n = len(low_dim_data)
    gradient = [[0, 0] for _ in range(n)]
    t_sne_affinities = compute_t_sne_affinities(low_dim_data)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = [low_dim_data[i][d] - low_dim_data[j][d] for d in range(2)]
                factor = 4 * (high_dim_affinities[i][j] - t_sne_affinities[i][j])
                gradient[i][0] += factor * diff[0]
                gradient[i][1] += factor * diff[1]
    
    # Update velocities and positions
    for i in range(n):
        for d in range(2):
            velocities[i][d] = momentum * velocities[i][d] - learning_rate * gradient[i][d]
            low_dim_data[i][d] += velocities[i][d]

# Initialize velocities and perform one t-SNE step
velocities = [[0, 0] for _ in range(len(low_dim_data))]
learning_rate, momentum = 0.1, 0.9
t_sne_step(affinities, low_dim_data, velocities, learning_rate, momentum)
print("Updated low-dimensional data after t-SNE step:")
for point in low_dim_data:
    print(f"({point[0]:.2f}, {point[1]:.2f})")
```

Slide 10: Early Exaggeration

Early exaggeration is a technique used in t-SNE to create better-separated clusters in the early stages of optimization. It involves multiplying the high-dimensional probabilities by a factor (typically 4-12) for the first 250-300 iterations.

```python
def apply_early_exaggeration(affinities, exaggeration_factor):
    return [[aff * exaggeration_factor for aff in row] for row in affinities]

def remove_early_exaggeration(affinities, exaggeration_factor):
    return [[aff / exaggeration_factor for aff in row] for row in affinities]

# Apply early exaggeration
exaggeration_factor = 4
exaggerated_affinities = apply_early_exaggeration(affinities, exaggeration_factor)

print("Original affinities (first 3x3):")
for row in affinities[:3]:
    print(" ".join(f"{x:.4f}" for x in row[:3]))

print("\nExaggerated affinities (first 3x3):")
for row in exaggerated_affinities[:3]:
    print(" ".join(f"{x:.4f}" for x in row[:3]))
```

Slide 11: Visualization of t-SNE Results

After running t-SNE, the resulting low-dimensional data can be visualized using scatter plots. This allows us to see clusters and patterns in the high-dimensional data that might not be apparent otherwise.

```python
import matplotlib.pyplot as plt

def visualize_tsne(low_dim_data):
    x = [point[0] for point in low_dim_data]
    y = [point[1] for point in low_dim_data]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)
    plt.title("t-SNE Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

# Generate some sample low-dimensional data
low_dim_data = [[random.gauss(0, 1), random.gauss(0, 1)] for _ in range(100)]

# Visualize the data
visualize_tsne(low_dim_data)
```

Slide 12: Real-life Example: Handwritten Digit Recognition

t-SNE is often used to visualize high-dimensional datasets such as images. Let's use a simplified version of the MNIST dataset to demonstrate how t-SNE can reveal clusters of similar handwritten digits.

```python
import random

# Simulate MNIST-like data (simplified)
def generate_digit_data(n_samples_per_digit=100, n_pixels=28*28):
    data = []
    labels = []
    for digit in range(10):
        for _ in range(n_samples_per_digit):
            # Create a noisy representation of the digit
            pixel_values = [random.gauss(digit/10, 0.1) for _ in range(n_pixels)]
            data.append(pixel_values)
            labels.append(digit)
    return data, labels

# Generate data
digit_data, digit_labels = generate_digit_data(n_samples_per_digit=50)

print(f"Generated {len(digit_data)} samples with {len(digit_data[0])} dimensions each")
print(f"First few labels: {digit_labels[:10]}")
```

Slide 13: Real-life Example: Gene Expression Analysis

t-SNE is widely used in bioinformatics for visualizing high-dimensional gene expression data. It can help identify groups of genes with similar expression patterns or clusters of cells with similar transcriptomic profiles.

```python
def generate_gene_expression_data(n_samples=100, n_genes=1000):
    data = []
    cell_types = ['A', 'B', 'C']
    labels = []
    
    for _ in range(n_samples):
        cell_type = random.choice(cell_types)
        expression = [random.gauss(0, 1) for _ in range(n_genes)]
        
        # Simulate differential expression
        if cell_type == 'A':
            expression[:333] = [x + 2 for x in expression[:333]]
        elif cell_type == 'B':
            expression[333:666] = [x + 2 for x in expression[333:666]]
        else:  # cell_type == 'C'
            expression[666:] = [x + 2 for x in expression[666:]]
        
        data.append(expression)
        labels.append(cell_type)
    
    return data, labels

# Generate gene expression data
gene_data, cell_types = generate_gene_expression_data()

print(f"Generated {len(gene_data)} samples with {len(gene_data[0])} genes each")
print(f"First few cell types: {cell_types[:10]}")
```

Slide 14: Limitations and Considerations

While t-SNE is a powerful visualization tool, it

Slide 14: Limitations and Considerations

While t-SNE is a powerful visualization tool, it has certain limitations and considerations that users should be aware of. The algorithm can be sensitive to hyperparameters, particularly perplexity. It may also produce different results on multiple runs due to its stochastic nature. Additionally, t-SNE focuses on preserving local structure, which means global relationships might not be accurately represented in the final visualization.

```python
def demonstrate_tsne_limitations(data, perplexities=[5, 30, 50], n_runs=3):
    results = {}
    for perplexity in perplexities:
        results[perplexity] = []
        for _ in range(n_runs):
            # Simulating t-SNE with different perplexities and runs
            result = [[random.gauss(0, 1), random.gauss(0, 1)] for _ in data]
            results[perplexity].append(result)
    
    return results

# Generate some sample data
sample_data = [[random.random() for _ in range(10)] for _ in range(100)]

# Demonstrate limitations
limitation_results = demonstrate_tsne_limitations(sample_data)

print("Number of different results for each perplexity:")
for perplexity, runs in limitation_results.items():
    print(f"Perplexity {perplexity}: {len(runs)} runs")
```

Slide 15: Additional Resources

For those interested in delving deeper into Stochastic Neighbor Embedding and t-SNE, the following resources provide comprehensive information:

1.  Original t-SNE paper: "Visualizing Data using t-SNE" by Laurens van der Maaten and Geoffrey Hinton (2008) ArXiv URL: [https://arxiv.org/abs/1307.1662](https://arxiv.org/abs/1307.1662)
2.  "How to Use t-SNE Effectively" by Martin Wattenberg, Fernanda Viégas, and Ian Johnson Available at: [https://distill.pub/2016/misread-tsne/](https://distill.pub/2016/misread-tsne/)
3.  "Accelerating t-SNE using Tree-Based Algorithms" by Laurens van der Maaten (2014) ArXiv URL: [https://arxiv.org/abs/1301.3342](https://arxiv.org/abs/1301.3342)

These resources offer in-depth explanations of the algorithm, its variations, and best practices for its application in various domains.

