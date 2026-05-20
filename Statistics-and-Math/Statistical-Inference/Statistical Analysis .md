## Statistical Analysis 

Slide 1: Introduction to Statistical Analysis

Statistical Analysis is the science of collecting, exploring, and presenting large amounts of data to discover underlying patterns and trends. It involves methods for describing and modeling data, drawing inferences, testing hypotheses, and making predictions. This field is crucial in various domains, including scientific research, business decision-making, social sciences, and data science.

```python
import random

# Simulating data collection
data = [random.gauss(0, 1) for _ in range(1000)]

# Basic statistical measures
mean = sum(data) / len(data)
variance = sum((x - mean) ** 2 for x in data) / len(data)
std_dev = variance ** 0.5

print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
```

Slide 2: Descriptive Statistics

Descriptive statistics summarize and describe the main features of a dataset. This includes measures of central tendency (mean, median, mode) and measures of variability (range, variance, standard deviation). These statistics provide a concise summary of the data's characteristics.

```python
def descriptive_stats(data):
    n = len(data)
    mean = sum(data) / n
    sorted_data = sorted(data)
    median = sorted_data[n // 2] if n % 2 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = variance ** 0.5
    return mean, median, variance, std_dev

data = [2, 4, 4, 4, 5, 5, 7, 9]
mean, median, variance, std_dev = descriptive_stats(data)
print(f"Mean: {mean:.2f}, Median: {median:.2f}, Variance: {variance:.2f}, Std Dev: {std_dev:.2f}")
```

Slide 3: Inferential Statistics

Inferential statistics involves drawing conclusions about a population based on a sample. This process allows us to make predictions and generalizations about a larger group using data from a smaller subset. Techniques include hypothesis testing, confidence intervals, and regression analysis.

```python
import random

def sample_mean(population, sample_size):
    sample = random.sample(population, sample_size)
    return sum(sample) / sample_size

# Simulating a population
population = [random.gauss(100, 15) for _ in range(10000)]

# Taking multiple samples and calculating their means
sample_means = [sample_mean(population, 50) for _ in range(1000)]

# Calculating the mean of sample means
mean_of_means = sum(sample_means) / len(sample_means)

print(f"Population mean: {sum(population) / len(population):.2f}")
print(f"Mean of sample means: {mean_of_means:.2f}")
```

Slide 4: Hypothesis Testing

Hypothesis testing is a method for making decisions about population parameters based on sample data. It involves formulating null and alternative hypotheses, calculating test statistics, and determining p-values to assess the likelihood of observing the data under the null hypothesis.

```python
def t_statistic(sample, hypothesized_mean):
    n = len(sample)
    sample_mean = sum(sample) / n
    sample_variance = sum((x - sample_mean) ** 2 for x in sample) / (n - 1)
    return (sample_mean - hypothesized_mean) / (sample_variance / n) ** 0.5

# Example: Testing if the mean of a sample is significantly different from 100
sample = [102, 98, 104, 101, 97, 105, 103, 99, 100, 102]
hypothesized_mean = 100

t_stat = t_statistic(sample, hypothesized_mean)
print(f"T-statistic: {t_stat:.4f}")
# Note: To complete the hypothesis test, we would compare this t-statistic
# to a critical value from a t-distribution table or calculate a p-value.
```

Slide 5: Regression Analysis

Regression analysis examines relationships between variables, typically to predict one variable based on others. Linear regression is a common technique that models the relationship between a dependent variable and one or more independent variables using a linear equation.

```python
def linear_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_xx = sum(xi ** 2 for xi in x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

slope, intercept = linear_regression(x, y)
print(f"Slope: {slope:.2f}, Intercept: {intercept:.2f}")
```

Slide 6: Probability Distributions

Probability distributions describe the likelihood of different outcomes in a random experiment. They are fundamental to statistical inference and modeling. Common distributions include normal (Gaussian), binomial, and Poisson distributions.

```python
import math

def normal_pdf(x, mu, sigma):
    coefficient = 1 / (sigma * (2 * math.pi) ** 0.5)
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coefficient * math.exp(exponent)

# Plotting a normal distribution
x_values = [x / 10 for x in range(-50, 51)]
y_values = [normal_pdf(x, 0, 1) for x in x_values]

print("Normal Distribution (μ=0, σ=1):")
print("x\t\tf(x)")
for x, y in zip(x_values[::10], y_values[::10]):
    print(f"{x:.2f}\t\t{y:.4f}")
```

Slide 7: Experimental Design

Experimental design involves planning studies to minimize bias and maximize information gain. Key concepts include randomization, replication, and control groups. A well-designed experiment allows for more reliable conclusions and reduces the impact of confounding variables.

```python
import random

def randomized_experiment(subjects, treatment_effect):
    # Randomly assign subjects to treatment and control groups
    treatment_group = random.sample(subjects, len(subjects) // 2)
    control_group = [s for s in subjects if s not in treatment_group]
    
    # Apply treatment effect to the treatment group
    results = {}
    for subject in subjects:
        if subject in treatment_group:
            results[subject] = random.gauss(100 + treatment_effect, 15)
        else:
            results[subject] = random.gauss(100, 15)
    
    return results, treatment_group, control_group

# Simulate an experiment
subjects = list(range(1, 101))  # 100 subjects
treatment_effect = 5  # Assumed effect of treatment

results, treatment_group, control_group = randomized_experiment(subjects, treatment_effect)

# Analyze results
treatment_mean = sum(results[s] for s in treatment_group) / len(treatment_group)
control_mean = sum(results[s] for s in control_group) / len(control_group)

print(f"Treatment group mean: {treatment_mean:.2f}")
print(f"Control group mean: {control_mean:.2f}")
print(f"Observed effect: {treatment_mean - control_mean:.2f}")
```

Slide 8: Data Visualization

Data visualization is a crucial aspect of statistical analysis, allowing for the effective communication of complex data patterns. Common visualization techniques include histograms, scatter plots, and box plots. These visual representations can reveal trends, outliers, and distributions within datasets.

```python
def ascii_histogram(data, bins=10):
    min_val, max_val = min(data), max(data)
    bin_width = (max_val - min_val) / bins
    counts = [0] * bins
    
    for value in data:
        bin_index = min(int((value - min_val) / bin_width), bins - 1)
        counts[bin_index] += 1
    
    max_count = max(counts)
    for i, count in enumerate(counts):
        bar_length = int(count / max_count * 20)
        left_edge = min_val + i * bin_width
        print(f"{left_edge:5.2f} | {'#' * bar_length}")

# Generate some sample data
import random
data = [random.gauss(0, 1) for _ in range(1000)]

print("Histogram of Normal Distribution:")
ascii_histogram(data)
```

Slide 9: Correlation Analysis

Correlation analysis measures the strength and direction of relationships between variables. The Pearson correlation coefficient is a common measure, ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no linear correlation.

```python
def pearson_correlation(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x_sq = sum(xi ** 2 for xi in x)
    sum_y_sq = sum(yi ** 2 for yi in y)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x_sq - sum_x ** 2) * (n * sum_y_sq - sum_y ** 2)) ** 0.5
    
    return numerator / denominator

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

correlation = pearson_correlation(x, y)
print(f"Pearson correlation coefficient: {correlation:.4f}")
```

Slide 10: Time Series Analysis

Time series analysis involves studying data points collected over time to identify trends, seasonality, and other patterns. This type of analysis is crucial in fields such as economics, finance, and environmental science for forecasting and understanding temporal dynamics.

```python
def moving_average(data, window_size):
    return [sum(data[i:i+window_size]) / window_size 
            for i in range(len(data) - window_size + 1)]

# Simulating a time series with trend and noise
import random
import math

time_series = [10 + 0.5*t + 2*math.sin(t/5) + random.gauss(0, 1) for t in range(100)]

# Calculate moving average
ma_series = moving_average(time_series, 5)

print("Original series (first 10 points):", time_series[:10])
print("Moving average (first 10 points):", ma_series[:10])
```

Slide 11: Bootstrapping

Bootstrapping is a resampling technique used to estimate the sampling distribution of a statistic. It involves repeatedly sampling with replacement from the original dataset to create multiple resamples, then calculating the statistic of interest for each resample.

```python
import random

def bootstrap_mean(data, num_resamples=1000):
    means = []
    for _ in range(num_resamples):
        resample = random.choices(data, k=len(data))
        means.append(sum(resample) / len(resample))
    return means

# Example dataset
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

bootstrap_means = bootstrap_mean(data)

print(f"Original data mean: {sum(data) / len(data)}")
print(f"Bootstrap mean of means: {sum(bootstrap_means) / len(bootstrap_means):.2f}")
print(f"Bootstrap 95% CI: ({sorted(bootstrap_means)[25]:.2f}, {sorted(bootstrap_means)[975]:.2f})")
```

Slide 12: ANOVA (Analysis of Variance)

ANOVA is a statistical method used to analyze the differences among group means in a sample. It's particularly useful when comparing three or more groups, extending the t-test concept to multiple groups. ANOVA helps determine if there are any statistically significant differences between the means of several independent groups.

```python
def anova_f_statistic(groups):
    grand_mean = sum(sum(group) for group in groups) / sum(len(group) for group in groups)
    
    between_group_var = sum(len(group) * (sum(group) / len(group) - grand_mean) ** 2 
                            for group in groups) / (len(groups) - 1)
    
    within_group_var = sum(sum((x - sum(group) / len(group)) ** 2 for x in group) 
                           for group in groups) / (sum(len(group) for group in groups) - len(groups))
    
    return between_group_var / within_group_var

# Example data: three groups
group1 = [4, 5, 6, 5, 4]
group2 = [7, 8, 9, 8, 7]
group3 = [1, 2, 3, 2, 1]

f_statistic = anova_f_statistic([group1, group2, group3])
print(f"ANOVA F-statistic: {f_statistic:.4f}")
```

Slide 13: Principal Component Analysis (PCA)

Principal Component Analysis is a dimensionality reduction technique used to simplify complex datasets while retaining most of the important information. It works by identifying the principal components, which are orthogonal vectors that capture the most variance in the data.

```python
def pca(X, num_components):
    # Center the data
    X_centered = X - X.mean(axis=0)
    
    # Compute covariance matrix
    cov_matrix = X_centered.T @ X_centered / (X.shape[0] - 1)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(cov_matrix)
    
    # Sort eigenvectors by decreasing eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top k eigenvectors
    W = eigenvectors[:, :num_components]
    
    # Project the data
    return X_centered @ W

# Example usage (Note: this is a simplified implementation)
import numpy as np
from numpy.linalg import eig

# Generate sample data
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# Perform PCA
X_pca = pca(X, num_components=2)

print("Original data shape:", X.shape)
print("PCA-transformed data shape:", X_pca.shape)
print("PCA-transformed data:")
print(X_pca)
```

Slide 14: Cross-validation

Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. It's particularly important for assessing how the results of a statistical analysis will generalize to an independent dataset. The most common method is k-fold cross-validation.

```python
import random

def k_fold_cross_validation(data, k, model_func):
    fold_size = len(data) // k
    scores = []
    
    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        
        test_set = data[test_start:test_end]
        train_set = data[:test_start] + data[test_end:]
        
        model = model_func(train_set)
        score = evaluate_model(model, test_set)
        scores.append(score)
    
    return sum(scores) / len(scores)

# Example usage with dummy functions
def dummy_model_func(train_data):
    return sum(train_data) / len(train_data)  # Simple mean model

def evaluate_model(model, test_data):
    return sum((x - model) ** 2 for x in test_data) / len(test_data)  # MSE

# Generate sample data
data = [random.gauss(0, 1) for _ in range(100)]

# Perform 5-fold cross-validation
avg_score = k_fold_cross_validation(data, k=5, model_func=dummy_model_func)
print(f"Average cross-validation score: {avg_score:.4f}")
```

Slide 15: Bayesian Inference

Bayesian inference is a method of statistical inference that uses Bayes' theorem to update the probability for a hypothesis as more evidence or information becomes available. It provides a framework for combining prior knowledge with observed data to make predictions and draw conclusions.

```python
def bayes_theorem(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

# Example: Medical test
# Prior probability of having the disease
prior_disease = 0.01  # 1% of population has the disease

# Likelihood of positive test given disease
sensitivity = 0.95  # 95% true positive rate

# Probability of positive test (evidence)
false_positive_rate = 0.10  # 10% false positive rate
evidence = sensitivity * prior_disease + false_positive_rate * (1 - prior_disease)

# Calculate posterior probability
posterior = bayes_theorem(prior_disease, sensitivity, evidence)

print(f"Probability of disease given positive test: {posterior:.4f}")
```

Slide 16: Real-life Example: Environmental Science

Statistical analysis plays a crucial role in environmental science, helping researchers understand and predict climate patterns, assess biodiversity, and evaluate the impact of human activities on ecosystems. Here's an example of analyzing temperature data to detect a warming trend.

```python
import random

def generate_temperature_data(years, base_temp, warming_rate, noise_level):
    return [base_temp + warming_rate * year + random.gauss(0, noise_level) for year in range(years)]

def analyze_temperature_trend(temperatures):
    years = len(temperatures)
    sum_x = sum(range(years))
    sum_y = sum(temperatures)
    sum_xy = sum(i * temp for i, temp in enumerate(temperatures))
    sum_x_squared = sum(i ** 2 for i in range(years))
    
    slope = (years * sum_xy - sum_x * sum_y) / (years * sum_x_squared - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / years
    
    return slope, intercept

# Generate synthetic temperature data
years = 50
base_temp = 15  # degrees Celsius
warming_rate = 0.02  # degrees per year
noise_level = 0.5  # random fluctuation

temperatures = generate_temperature_data(years, base_temp, warming_rate, noise_level)

# Analyze the trend
slope, intercept = analyze_temperature_trend(temperatures)

print(f"Estimated warming rate: {slope:.4f} degrees Celsius per year")
print(f"Estimated temperature in year 0: {intercept:.2f} degrees Celsius")
```

Slide 17: Real-life Example: Public Health

Statistical analysis is fundamental in public health research, helping to identify risk factors, evaluate interventions, and inform policy decisions. Here's an example of analyzing the effectiveness of a public health campaign using a chi-square test.

```python
import math

def chi_square_test(observed, expected):
    chi_square = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    degrees_of_freedom = len(observed) - 1
    return chi_square, degrees_of_freedom

# Example: Evaluating a smoking cessation campaign
# Observed quit rates in two cities (with and without campaign)
city_with_campaign = [150, 850]  # [quit, didn't quit]
city_without_campaign = [100, 900]  # [quit, didn't quit]

total_with = sum(city_with_campaign)
total_without = sum(city_without_campaign)
total_quit = city_with_campaign[0] + city_without_campaign[0]
total_not_quit = city_with_campaign[1] + city_without_campaign[1]
grand_total = total_with + total_without

expected_with = [total_quit * total_with / grand_total,
                 total_not_quit * total_with / grand_total]
expected_without = [total_quit * total_without / grand_total,
                    total_not_quit * total_without / grand_total]

chi_square, df = chi_square_test(city_with_campaign + city_without_campaign,
                                 expected_with + expected_without)

print(f"Chi-square statistic: {chi_square:.4f}")
print(f"Degrees of freedom: {df}")
# Note: To complete the analysis, compare the chi-square value
# to the critical value from a chi-square distribution table.
```

Slide 18: Additional Resources

For those interested in deepening their understanding of statistical analysis, here are some valuable resources:

1.  ArXiv.org Statistics section: [https://arxiv.org/list/stat/recent](https://arxiv.org/list/stat/recent) This repository contains numerous research papers on various statistical techniques and applications.
2.  "Statistical Rethinking" by Richard McElreath ArXiv paper discussing the book: [https://arxiv.org/abs/2011.01808](https://arxiv.org/abs/2011.01808)
3.  "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani Referenced in ArXiv papers: [https://arxiv.org/search/stat?query=An+Introduction+to+Statistical+Learning&searchtype=title](https://arxiv.org/search/stat?query=An+Introduction+to+Statistical+Learning&searchtype=title)

These resources provide in-depth coverage of statistical concepts, methodologies, and their applications in various fields. They range from introductory to advanced levels, catering to different learning needs.

