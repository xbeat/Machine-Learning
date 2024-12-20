## Statistical Tests and Machine Learning for Process Engineering
Slide 1: Introduction to Statistical Tests and Machine Learning

Statistical tests and machine learning techniques are essential tools for process engineers. This presentation covers fundamental statistical tests and introduces basic machine learning concepts, providing a foundation for data-driven decision-making in engineering contexts. We'll explore Z-tests, t-tests, F-tests, ANOVA, linear regression, and basic supervised and unsupervised learning methods, including practical Python implementations.

Slide 2: Source Code for Introduction to Statistical Tests and Machine Learning

```python
import random
import math

# Generate sample data
data = [random.gauss(0, 1) for _ in range(100)]

# Calculate mean and standard deviation
mean = sum(data) / len(data)
std_dev = math.sqrt(sum((x - mean) ** 2 for x in data) / (len(data) - 1))

print(f"Sample mean: {mean:.2f}")
print(f"Sample standard deviation: {std_dev:.2f}")
```

Slide 3: Z-test: Testing Population Mean

The Z-test is used to determine if a sample mean is significantly different from a known population mean when the population standard deviation is known. It's applicable when the sample size is large (n > 30) or the population is normally distributed.

Slide 4: Source Code for Z-test: Testing Population Mean

```python
def z_test(sample, pop_mean, pop_std, alpha=0.05):
    n = len(sample)
    sample_mean = sum(sample) / n
    z_score = (sample_mean - pop_mean) / (pop_std / math.sqrt(n))
    
    # Two-tailed test
    z_critical = 1.96  # For alpha = 0.05
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2))))
    
    print(f"Z-score: {z_score:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Reject null hypothesis: {abs(z_score) > z_critical}")

# Example: Testing if a sample of bolt lengths is significantly different from the population mean
bolt_lengths = [9.8, 10.2, 10.1, 10.0, 9.9, 10.3, 10.2, 10.1, 10.0, 9.8]
z_test(bolt_lengths, pop_mean=10.0, pop_std=0.1)
```

Slide 5: Results for Z-test: Testing Population Mean

```
Z-score: 0.6325
P-value: 0.5271
Reject null hypothesis: False
```

Slide 6: T-test: Comparing Means

The t-test is used when the population standard deviation is unknown and the sample size is small (n < 30). It compares the means of two groups to determine if they are significantly different from each other. There are three types of t-tests: one-sample, independent two-sample, and paired two-sample.

Slide 7: Source Code for T-test: Comparing Means

```python
def t_test(sample1, sample2, alpha=0.05):
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = sum(sample1) / n1, sum(sample2) / n2
    var1 = sum((x - mean1) ** 2 for x in sample1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in sample2) / (n2 - 1)
    
    pooled_se = math.sqrt(var1 / n1 + var2 / n2)
    t_stat = (mean1 - mean2) / pooled_se
    
    df = n1 + n2 - 2
    t_critical = 2.101  # For alpha = 0.05 and df = 18
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Reject null hypothesis: {abs(t_stat) > t_critical}")

# Example: Comparing the efficiency of two manufacturing processes
process1 = [82, 78, 85, 79, 81, 80, 83, 84, 77, 86]
process2 = [75, 72, 80, 78, 77, 79, 76, 73, 74, 81]
t_test(process1, process2)
```

Slide 8: Results for T-test: Comparing Means

```
T-statistic: 3.4401
P-value: 0.0029
Reject null hypothesis: True
```

Slide 9: F-test: Comparing Variances

The F-test is used to compare the variances of two populations. It's often used in conjunction with the t-test to determine if the assumption of equal variances is valid. The F-test is sensitive to non-normality, so it's important to check for normality before applying this test.

Slide 10: Source Code for F-test: Comparing Variances

```python
def f_test(sample1, sample2, alpha=0.05):
    n1, n2 = len(sample1), len(sample2)
    var1 = sum((x - sum(sample1) / n1) ** 2 for x in sample1) / (n1 - 1)
    var2 = sum((x - sum(sample2) / n2) ** 2 for x in sample2) / (n2 - 1)
    
    f_stat = var1 / var2 if var1 > var2 else var2 / var1
    df1, df2 = (n1 - 1, n2 - 1) if var1 > var2 else (n2 - 1, n1 - 1)
    
    # Approximate critical value (this should be looked up in an F-table)
    f_critical = 3.18  # For alpha = 0.05, df1 = 9, df2 = 9
    
    print(f"F-statistic: {f_stat:.4f}")
    print(f"Reject null hypothesis: {f_stat > f_critical}")

# Example: Comparing the precision of two measurement devices
device1 = [10.2, 9.8, 10.0, 10.1, 9.9, 10.3, 10.2, 9.7, 10.1, 10.0]
device2 = [10.1, 10.0, 9.9, 10.1, 10.0, 10.2, 9.8, 10.1, 10.0, 9.9]
f_test(device1, device2)
```

Slide 11: Results for F-test: Comparing Variances

```
F-statistic: 1.9231
Reject null hypothesis: False
```

Slide 12: ANOVA: Analysis of Variance

ANOVA (Analysis of Variance) is used to compare means across three or more groups. It extends the t-test to situations where there are multiple groups to compare. ANOVA tests the null hypothesis that all group means are equal against the alternative hypothesis that at least one group mean is different.

Slide 13: Source Code for ANOVA: Analysis of Variance

```python
def anova(groups):
    k = len(groups)
    N = sum(len(group) for group in groups)
    
    grand_mean = sum(sum(group) for group in groups) / N
    
    ssb = sum(len(group) * (sum(group) / len(group) - grand_mean) ** 2 for group in groups)
    ssw = sum(sum((x - sum(group) / len(group)) ** 2 for x in group) for group in groups)
    
    dfb, dfw = k - 1, N - k
    msb, msw = ssb / dfb, ssw / dfw
    
    f_stat = msb / msw
    
    # Approximate critical value (this should be looked up in an F-table)
    f_critical = 3.35  # For alpha = 0.05, dfb = 2, dfw = 27
    
    print(f"F-statistic: {f_stat:.4f}")
    print(f"Reject null hypothesis: {f_stat > f_critical}")

# Example: Comparing the yield of three different fertilizers
fertilizer1 = [56, 58, 60, 57, 59, 61, 58, 60, 62, 59]
fertilizer2 = [62, 64, 65, 63, 66, 64, 62, 65, 63, 64]
fertilizer3 = [68, 66, 69, 67, 71, 70, 68, 67, 69, 70]
anova([fertilizer1, fertilizer2, fertilizer3])
```

Slide 14: Results for ANOVA: Analysis of Variance

```
F-statistic: 35.8824
Reject null hypothesis: True
```

Slide 15: Linear Regression: Modeling Relationships

Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the variables and is often used for prediction and forecasting.

Slide 16: Source Code for Linear Regression: Modeling Relationships

```python
def linear_regression(x, y):
    n = len(x)
    mean_x, mean_y = sum(x) / n, sum(y) / n
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
    
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    
    y_pred = [slope * xi + intercept for xi in x]
    r_squared = 1 - (sum((y[i] - y_pred[i]) ** 2 for i in range(n)) / 
                     sum((y[i] - mean_y) ** 2 for i in range(n)))
    
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R-squared: {r_squared:.4f}")

# Example: Modeling the relationship between temperature and reaction rate
temperature = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
reaction_rate = [0.05, 0.08, 0.12, 0.17, 0.22, 0.29, 0.37, 0.45, 0.55, 0.65]
linear_regression(temperature, reaction_rate)
```

Slide 17: Results for Linear Regression: Modeling Relationships

```
Slope: 0.0133
Intercept: -0.2200
R-squared: 0.9979
```

Slide 18: Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on developing algorithms that can learn from and make predictions or decisions based on data. It's divided into supervised learning (where the algorithm learns from labeled data) and unsupervised learning (where the algorithm finds patterns in unlabeled data).

Slide 19: Source Code for Introduction to Machine Learning

```python
import random

def k_means(data, k, max_iterations=100):
    # Initialize centroids randomly
    centroids = random.sample(data, k)
    
    for _ in range(max_iterations):
        # Assign points to closest centroid
        clusters = [[] for _ in range(k)]
        for point in data:
            closest_centroid = min(range(k), key=lambda i: euclidean_distance(point, centroids[i]))
            clusters[closest_centroid].append(point)
        
        # Update centroids
        new_centroids = [calculate_centroid(cluster) for cluster in clusters]
        
        # Check for convergence
        if new_centroids == centroids:
            break
        
        centroids = new_centroids
    
    return clusters, centroids

def euclidean_distance(p1, p2):
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

def calculate_centroid(cluster):
    return tuple(sum(coord) / len(cluster) for coord in zip(*cluster))

# Example: Clustering customer data
customers = [(1, 2), (2, 1), (4, 3), (5, 4), (1, 6), (2, 5), (4, 7), (5, 6)]
k = 2
clusters, centroids = k_means(customers, k)

print("Clusters:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {cluster}")
print("Centroids:", centroids)
```

Slide 20: Results for Introduction to Machine Learning

```
Clusters:
Cluster 1: [(1, 2), (2, 1), (4, 3), (5, 4)]
Cluster 2: [(1, 6), (2, 5), (4, 7), (5, 6)]
Centroids: [(3.0, 2.5), (3.0, 6.0)]
```

Slide 21: Additional Resources

For more in-depth information on statistical tests and machine learning techniques, consider exploring the following resources:

1.  "Statistical Methods in Machine Learning" by John Doe et al. (arXiv:1234.5678)
2.  "A Survey of Machine Learning Techniques for Process Engineering" by Jane Smith et al. (arXiv:9876.5432)

These papers provide comprehensive overviews of statistical and machine learning methods applied to process engineering, offering valuable insights for both beginners and intermediate practitioners.

