## Essential Statistical Concepts for Data Analysis
Slide 1: Correlation Analysis - Pearson vs Spearman

Statistical correlations measure relationships between variables, with Pearson capturing linear relationships and Spearman handling non-linear monotonic relationships. Understanding their differences enables choosing the appropriate method for your data analysis tasks.

```python
import numpy as np
import pandas as pd
from scipy import stats

# Generate sample data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y_linear = x + np.random.normal(0, 1, 100)
y_nonlinear = x**2 + np.random.normal(0, 10, 100)

# Calculate correlations
pearson_linear = stats.pearsonr(x, y_linear)
spearman_linear = stats.spearmanr(x, y_linear)
pearson_nonlinear = stats.pearsonr(x, y_nonlinear)
spearman_nonlinear = stats.spearmanr(x, y_nonlinear)

print(f"Linear relationship:")
print(f"Pearson correlation: {pearson_linear[0]:.3f}")
print(f"Spearman correlation: {spearman_linear[0]:.3f}\n")
print(f"Non-linear relationship:")
print(f"Pearson correlation: {pearson_nonlinear[0]:.3f}")
print(f"Spearman correlation: {spearman_nonlinear[0]:.3f}")
```

Slide 2: P-Value Computation from Scratch

The p-value calculation involves comparing observed test statistics against a null distribution. This implementation demonstrates how to compute p-values for a two-sample t-test without relying on statistical libraries.

```python
def calculate_t_statistic(sample1, sample2):
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    
    pooled_se = np.sqrt(var1/n1 + var2/n2)
    t_stat = (mean1 - mean2) / pooled_se
    return t_stat

def compute_p_value(sample1, sample2, n_permutations=10000):
    observed_t = abs(calculate_t_statistic(sample1, sample2))
    combined = np.concatenate([sample1, sample2])
    n1 = len(sample1)
    
    t_stats = np.zeros(n_permutations)
    for i in range(n_permutations):
        np.random.shuffle(combined)
        perm_sample1 = combined[:n1]
        perm_sample2 = combined[n1:]
        t_stats[i] = abs(calculate_t_statistic(perm_sample1, perm_sample2))
    
    p_value = np.mean(t_stats >= observed_t)
    return p_value

# Example usage
sample1 = np.random.normal(0, 1, 30)
sample2 = np.random.normal(0.5, 1, 30)
p_val = compute_p_value(sample1, sample2)
print(f"Computed p-value: {p_val:.4f}")
```

Slide 3: Survivorship Bias Detection

Survivorship bias can significantly impact analysis results when working with historical data. This implementation shows how to detect and quantify survivorship bias in financial time series data.

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic dataset with survivorship bias
def generate_biased_dataset(n_companies=100, n_years=10):
    dates = pd.date_range(start='2010-01-01', periods=n_years*12, freq='M')
    companies = [f'Company_{i}' for i in range(n_companies)]
    
    # Create performance data with built-in bias
    data = []
    for company in companies:
        start_idx = np.random.randint(0, len(dates)//2)
        performance = np.random.normal(0.005, 0.02, size=len(dates)-start_idx)
        cumulative_perf = np.cumprod(1 + performance)
        
        # Simulate company failure if performance drops below threshold
        if np.min(cumulative_perf) < 0.7:
            end_idx = np.where(cumulative_perf < 0.7)[0][0] + start_idx
        else:
            end_idx = len(dates)
            
        company_data = pd.DataFrame({
            'date': dates[start_idx:end_idx],
            'company': company,
            'returns': performance[:end_idx-start_idx]
        })
        data.append(company_data)
    
    return pd.concat(data, ignore_index=True)

# Detect and quantify survivorship bias
def analyze_survivorship_bias(df):
    # Calculate survival rates over time
    survival_by_period = df.groupby('date')['company'].nunique()
    
    # Calculate returns with and without survivorship bias
    all_returns = df.groupby('date')['returns'].mean()
    surviving_companies = df.groupby('company').date.max()
    survivors = surviving_companies[surviving_companies == df.date.max()].index
    survivor_returns = df[df.company.isin(survivors)].groupby('date')['returns'].mean()
    
    bias_impact = survivor_returns.mean() - all_returns.mean()
    
    return {
        'survival_rate': len(survivors) / df.company.nunique(),
        'bias_impact': bias_impact,
        'survivor_mean_return': survivor_returns.mean(),
        'all_mean_return': all_returns.mean()
    }

# Example usage
df = generate_biased_dataset()
results = analyze_survivorship_bias(df)
print("Survivorship Bias Analysis:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")
```

Slide 4: Simpson's Paradox Implementation

Simpson's Paradox occurs when trends present in different groups reverse when the groups are combined. This implementation demonstrates how to detect and visualize Simpson's Paradox using a practical example.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_simpsons_paradox_data():
    np.random.seed(42)
    
    # Generate data for two groups
    n_samples = 100
    group1_x = np.random.normal(2, 0.5, n_samples)
    group1_y = 0.5 * group1_x + np.random.normal(0, 0.2, n_samples)
    
    group2_x = np.random.normal(4, 0.5, n_samples)
    group2_y = 0.5 * group2_x + np.random.normal(0, 0.2, n_samples) + 2
    
    df = pd.DataFrame({
        'x': np.concatenate([group1_x, group2_x]),
        'y': np.concatenate([group1_y, group2_y]),
        'group': ['A'] * n_samples + ['B'] * n_samples
    })
    
    return df

def detect_simpsons_paradox(df):
    # Calculate correlations
    overall_corr = df['x'].corr(df['y'])
    group_corrs = df.groupby('group').apply(lambda x: x['x'].corr(x['y']))
    
    # Fit regression lines
    from sklearn.linear_model import LinearRegression
    
    def fit_regression(data):
        reg = LinearRegression()
        reg.fit(data['x'].values.reshape(-1, 1), data['y'])
        return reg.coef_[0]
    
    overall_slope = fit_regression(df)
    group_slopes = df.groupby('group').apply(fit_regression)
    
    # Check for paradox
    has_paradox = (
        (all(slope > 0 for slope in group_slopes) and overall_slope < 0) or
        (all(slope < 0 for slope in group_slopes) and overall_slope > 0)
    )
    
    return {
        'overall_correlation': overall_corr,
        'group_correlations': group_corrs,
        'overall_slope': overall_slope,
        'group_slopes': group_slopes,
        'has_paradox': has_paradox
    }

# Example usage
df = generate_simpsons_paradox_data()
results = detect_simpsons_paradox(df)

print("Simpson's Paradox Analysis:")
print(f"Overall correlation: {results['overall_correlation']:.3f}")
print("\nGroup correlations:")
for group, corr in results['group_correlations'].items():
    print(f"Group {group}: {corr:.3f}")
print(f"\nParadox detected: {results['has_paradox']}")
```

Slide 5: Central Limit Theorem Visualization

The Central Limit Theorem demonstrates how sample means approach a normal distribution regardless of the original population distribution. This implementation provides a comprehensive visualization of this fundamental statistical concept.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def demonstrate_clt(population_dist, n_samples, sample_size):
    # Generate samples and calculate means
    sample_means = np.array([
        np.mean(population_dist(size=sample_size))
        for _ in range(n_samples)
    ])
    
    # Calculate theoretical normal distribution
    mean = np.mean(sample_means)
    std = np.std(sample_means)
    x = np.linspace(mean - 4*std, mean + 4*std, 100)
    theoretical = stats.norm.pdf(x, mean, std)
    
    # Plotting
    plt.figure(figsize=(12, 4))
    
    # Original distribution
    plt.subplot(131)
    plt.hist(population_dist(size=10000), bins=50, density=True, alpha=0.7)
    plt.title('Original Distribution')
    
    # Sample means distribution
    plt.subplot(132)
    plt.hist(sample_means, bins=50, density=True, alpha=0.7)
    plt.plot(x, theoretical, 'r-', lw=2)
    plt.title(f'Distribution of Sample Means\n(n={sample_size})')
    
    # Q-Q plot
    plt.subplot(133)
    stats.probplot(sample_means, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Sample Means')
    
    plt.tight_layout()
    return plt.gcf()

# Example with different distributions
distributions = {
    'Uniform': lambda size: np.random.uniform(0, 1, size),
    'Exponential': lambda size: np.random.exponential(1, size),
    'Chi-squared': lambda size: np.random.chisquare(df=2, size=size)
}

for dist_name, dist_func in distributions.items():
    fig = demonstrate_clt(dist_func, n_samples=1000, sample_size=30)
    plt.suptitle(f'CLT Demonstration: {dist_name} Distribution')
    
    # Calculate statistics
    samples = np.array([np.mean(dist_func(size=30)) for _ in range(1000)])
    normality_test = stats.normaltest(samples)
    print(f"\n{dist_name} Distribution:")
    print(f"Normality test p-value: {normality_test.pvalue:.4f}")
    print(f"Sample mean: {np.mean(samples):.4f}")
    print(f"Sample std: {np.std(samples):.4f}")
```

Slide 6: Bayesian Inference Implementation

A practical implementation of Bayesian inference that updates probabilities based on new evidence. This code demonstrates the core concepts of prior, likelihood, and posterior probability calculations.

```python
import numpy as np
from scipy import stats

class BayesianInference:
    def __init__(self, prior_params):
        """
        Initialize with prior distribution parameters
        prior_params: dict with 'mu' and 'sigma' for normal distribution
        """
        self.prior_mu = prior_params['mu']
        self.prior_sigma = prior_params['sigma']
        self.data = []
        self.posterior_mu = self.prior_mu
        self.posterior_sigma = self.prior_sigma
    
    def update_belief(self, new_data, measurement_sigma):
        """
        Update beliefs using Bayesian inference
        new_data: observed value
        measurement_sigma: standard deviation of measurement
        """
        self.data.append(new_data)
        
        # Calculate posterior parameters
        precision_prior = 1 / (self.posterior_sigma ** 2)
        precision_measurement = 1 / (measurement_sigma ** 2)
        
        posterior_precision = precision_prior + precision_measurement
        self.posterior_sigma = np.sqrt(1 / posterior_precision)
        
        self.posterior_mu = (
            (precision_prior * self.posterior_mu + 
             precision_measurement * new_data) / 
            posterior_precision
        )
        
        return self.get_posterior_params()
    
    def get_posterior_params(self):
        return {
            'mu': self.posterior_mu,
            'sigma': self.posterior_sigma
        }
    
    def predict(self, x):
        """
        Make predictions using current posterior
        """
        return stats.norm.pdf(x, self.posterior_mu, self.posterior_sigma)

# Example usage
np.random.seed(42)

# True parameter we're trying to estimate
true_param = 5.0

# Generate synthetic observations
n_observations = 10
observations = np.random.normal(true_param, 1.0, n_observations)

# Initialize Bayesian inference
prior_params = {'mu': 0.0, 'sigma': 2.0}
bayes_model = BayesianInference(prior_params)

# Update beliefs with each observation
print("Bayesian Parameter Estimation:")
print(f"True parameter: {true_param}")
print("\nUpdating beliefs:")

for i, obs in enumerate(observations, 1):
    posterior = bayes_model.update_belief(obs, measurement_sigma=1.0)
    print(f"After observation {i}:")
    print(f"Posterior mean: {posterior['mu']:.3f}")
    print(f"Posterior sigma: {posterior['sigma']:.3f}")

# Final prediction
x = np.linspace(0, 10, 100)
predictions = bayes_model.predict(x)
```

Slide 7: Law of Large Numbers Visualization

This implementation demonstrates the Law of Large Numbers through Monte Carlo simulation, showing how sample means converge to the true population mean as sample size increases.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def demonstrate_law_of_large_numbers(distribution_func, true_mean, n_samples, max_size):
    """
    Visualize the convergence of sample means to true mean
    """
    # Generate samples of increasing size
    sizes = np.logspace(1, np.log10(max_size), n_samples).astype(int)
    means = np.zeros(len(sizes))
    deviations = np.zeros(len(sizes))
    
    for i, size in enumerate(sizes):
        sample = distribution_func(size=size)
        means[i] = np.mean(sample)
        deviations[i] = abs(means[i] - true_mean)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Sample means plot
    ax1.semilogx(sizes, means, 'b-', alpha=0.5, label='Sample Mean')
    ax1.axhline(y=true_mean, color='r', linestyle='--', label='True Mean')
    ax1.fill_between(sizes, 
                     means - 2*np.std(means),
                     means + 2*np.std(means),
                     alpha=0.2)
    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('Sample Mean')
    ax1.set_title('Convergence of Sample Means')
    ax1.legend()
    
    # Deviation plot
    ax2.loglog(sizes, deviations, 'g-', alpha=0.5)
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('Absolute Deviation from True Mean')
    ax2.set_title('Convergence Rate')
    
    return means, deviations

# Example usage with different distributions
distributions = {
    'Normal': {
        'func': lambda size: np.random.normal(loc=0, scale=1, size=size),
        'true_mean': 0
    },
    'Exponential': {
        'func': lambda size: np.random.exponential(scale=2, size=size),
        'true_mean': 2
    },
    'Uniform': {
        'func': lambda size: np.random.uniform(low=0, high=1, size=size),
        'true_mean': 0.5
    }
}

results = {}
for dist_name, dist_info in distributions.items():
    print(f"\nAnalyzing {dist_name} distribution:")
    means, devs = demonstrate_law_of_large_numbers(
        dist_info['func'],
        dist_info['true_mean'],
        n_samples=100,
        max_size=10000
    )
    plt.suptitle(f'Law of Large Numbers: {dist_name} Distribution')
    
    # Calculate convergence statistics
    final_error = devs[-1]
    convergence_rate = -np.polyfit(np.log10(devs), np.log10(range(1, len(devs)+1)), 1)[0]
    
    print(f"Final error: {final_error:.6f}")
    print(f"Convergence rate: {convergence_rate:.3f}")
    
    results[dist_name] = {
        'final_error': final_error,
        'convergence_rate': convergence_rate
    }
```

Slide 8: Selection Bias Detection and Correction

Selection bias occurs when data collection methods create a non-representative sample. This implementation demonstrates techniques for detecting and correcting selection bias in datasets using propensity score matching.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

class SelectionBiasAnalyzer:
    def __init__(self):
        self.propensity_model = LogisticRegression()
        self.scaler = StandardScaler()
        
    def compute_propensity_scores(self, X, selection_indicator):
        """
        Compute propensity scores using logistic regression
        """
        X_scaled = self.scaler.fit_transform(X)
        self.propensity_model.fit(X_scaled, selection_indicator)
        return self.propensity_model.predict_proba(X_scaled)[:, 1]
    
    def assess_selection_bias(self, data, features, selection_var):
        """
        Assess selection bias in the dataset
        """
        # Calculate propensity scores
        propensity_scores = self.compute_propensity_scores(
            data[features], 
            data[selection_var]
        )
        
        # Compute balance metrics
        balance_stats = {}
        for feature in features:
            selected = data[data[selection_var] == 1][feature]
            not_selected = data[data[selection_var] == 0][feature]
            
            t_stat, p_val = stats.ttest_ind(selected, not_selected)
            effect_size = (np.mean(selected) - np.mean(not_selected)) / \
                          np.sqrt((np.var(selected) + np.var(not_selected)) / 2)
            
            balance_stats[feature] = {
                'p_value': p_val,
                'effect_size': effect_size,
                'mean_diff': np.mean(selected) - np.mean(not_selected)
            }
        
        return {
            'propensity_scores': propensity_scores,
            'balance_stats': balance_stats
        }
    
    def correct_selection_bias(self, data, features, selection_var, method='ipw'):
        """
        Correct for selection bias using inverse probability weighting
        or matching
        """
        propensity_scores = self.compute_propensity_scores(
            data[features], 
            data[selection_var]
        )
        
        if method == 'ipw':
            # Inverse probability weighting
            weights = 1 / propensity_scores
            weights[data[selection_var] == 0] = 1 / (1 - propensity_scores[data[selection_var] == 0])
            
            # Normalize weights
            weights = weights / np.sum(weights) * len(weights)
            
            return weights
        
        elif method == 'matching':
            # Propensity score matching
            from scipy.spatial.distance import cdist
            
            selected_idx = data[data[selection_var] == 1].index
            not_selected_idx = data[data[selection_var] == 0].index
            
            distances = cdist(
                propensity_scores[selected_idx].reshape(-1, 1),
                propensity_scores[not_selected_idx].reshape(-1, 1)
            )
            
            matches = not_selected_idx[distances.argmin(axis=1)]
            
            return pd.concat([
                data.loc[selected_idx],
                data.loc[matches]
            ])

# Example usage
np.random.seed(42)

# Generate synthetic dataset with selection bias
n_samples = 1000
X = np.random.normal(size=(n_samples, 3))
selection_prob = 1 / (1 + np.exp(-X[:, 0] - 0.5 * X[:, 1]))
selection = np.random.binomial(1, selection_prob)

data = pd.DataFrame({
    'feature1': X[:, 0],
    'feature2': X[:, 1],
    'feature3': X[:, 2],
    'selected': selection
})

# Analyze and correct selection bias
analyzer = SelectionBiasAnalyzer()
bias_assessment = analyzer.assess_selection_bias(
    data,
    ['feature1', 'feature2', 'feature3'],
    'selected'
)

print("Selection Bias Analysis:")
for feature, stats in bias_assessment['balance_stats'].items():
    print(f"\n{feature}:")
    print(f"Effect size: {stats['effect_size']:.3f}")
    print(f"P-value: {stats['p_value']:.3f}")

# Correct bias using IPW
corrected_weights = analyzer.correct_selection_bias(
    data,
    ['feature1', 'feature2', 'feature3'],
    'selected'
)

# Compare weighted and unweighted statistics
print("\nBias Correction Results:")
for feature in ['feature1', 'feature2', 'feature3']:
    original_mean = np.mean(data[feature])
    corrected_mean = np.average(data[feature], weights=corrected_weights)
    print(f"\n{feature}:")
    print(f"Original mean: {original_mean:.3f}")
    print(f"Corrected mean: {corrected_mean:.3f}")
```

Slide 9: Advanced Outlier Detection

This implementation provides multiple methods for outlier detection, including statistical, distance-based, and density-based approaches, with visualization of results.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

class OutlierDetector:
    def __init__(self, methods=None):
        self.methods = methods or ['zscore', 'iqr', 'isolation_forest', 'lof']
        self.results = {}
        
    def detect_outliers(self, data, threshold=3):
        """
        Apply multiple outlier detection methods and compare results
        """
        for method in self.methods:
            if method == 'zscore':
                z_scores = stats.zscore(data)
                self.results['zscore'] = np.abs(z_scores) > threshold
                
            elif method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                self.results['iqr'] = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
                
            elif method == 'isolation_forest':
                iso_forest = IsolationForest(random_state=42)
                self.results['isolation_forest'] = iso_forest.fit_predict(data.values.reshape(-1, 1)) == -1
                
            elif method == 'lof':
                lof = LocalOutlierFactor()
                self.results['lof'] = lof.fit_predict(data.values.reshape(-1, 1)) == -1
        
        return self.results
    
    def get_consensus_outliers(self, min_methods=2):
        """
        Find outliers detected by multiple methods
        """
        methods_agreement = sum(self.results.values())
        return methods_agreement >= min_methods
    
    def calculate_outlier_stats(self):
        """
        Calculate statistics about detected outliers
        """
        stats_dict = {}
        for method, outliers in self.results.items():
            stats_dict[method] = {
                'num_outliers': sum(outliers),
                'percentage': sum(outliers) / len(outliers) * 100
            }
        
        consensus = self.get_consensus_outliers()
        stats_dict['consensus'] = {
            'num_outliers': sum(consensus),
            'percentage': sum(consensus) / len(consensus) * 100
        }
        
        return stats_dict

# Example usage
np.random.seed(42)

# Generate synthetic dataset with outliers
n_samples = 1000
normal_data = np.random.normal(loc=0, scale=1, size=n_samples)
outliers = np.random.uniform(low=5, high=10, size=int(n_samples * 0.01))
data = pd.Series(np.concatenate([normal_data, outliers]))

# Detect outliers
detector = OutlierDetector()
outliers = detector.detect_outliers(data)
stats = detector.calculate_outlier_stats()

print("Outlier Detection Results:")
for method, method_stats in stats.items():
    print(f"\n{method.upper()}:")
    print(f"Number of outliers: {method_stats['num_outliers']}")
    print(f"Percentage: {method_stats['percentage']:.2f}%")

# Calculate impact of outlier removal
consensus_outliers = detector.get_consensus_outliers()
clean_data = data[~consensus_outliers]

print("\nDataset Statistics:")
print(f"Original mean: {data.mean():.3f}")
print(f"Clean mean: {clean_data.mean():.3f}")
print(f"Original std: {data.std():.3f}")
print(f"Clean std: {clean_data.std():.3f}")
```

Slide 10: Real-World Application - Credit Risk Analysis

This implementation demonstrates a comprehensive statistical analysis pipeline for credit risk assessment, incorporating multiple statistical concepts covered previously.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve

class CreditRiskAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.correlations = None
        self.bias_metrics = None
        self.feature_importance = None
        
    def preprocess_data(self, data):
        """
        Preprocess data with statistical considerations
        """
        # Handle missing values using statistical imputation
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            is_missing = data[col].isnull()
            if is_missing.any():
                # Use correlation-based imputation
                correlations = data[numeric_columns].corr()[col]
                best_predictor = correlations.drop(col).abs().idxmax()
                slope = np.polyfit(data[best_predictor][~is_missing],
                                 data[col][~is_missing], 1)[0]
                data.loc[is_missing, col] = data.loc[is_missing, best_predictor] * slope
        
        # Scale features
        data_scaled = pd.DataFrame(
            self.scaler.fit_transform(data[numeric_columns]),
            columns=numeric_columns,
            index=data.index
        )
        
        return data_scaled
    
    def analyze_feature_distributions(self, data):
        """
        Analyze statistical properties of features
        """
        distribution_stats = {}
        for column in data.columns:
            # Perform normality test
            _, p_value = stats.normaltest(data[column].dropna())
            
            # Calculate moments
            distribution_stats[column] = {
                'mean': np.mean(data[column]),
                'std': np.std(data[column]),
                'skewness': stats.skew(data[column]),
                'kurtosis': stats.kurtosis(data[column]),
                'normal_p_value': p_value
            }
        
        return distribution_stats
    
    def detect_selection_bias(self, data, target):
        """
        Detect potential selection bias in credit applications
        """
        # Compare accepted vs rejected applications
        accepted = data[target == 1]
        rejected = data[target == 0]
        
        bias_metrics = {}
        for column in data.columns:
            t_stat, p_value = stats.ttest_ind(
                accepted[column].dropna(),
                rejected[column].dropna()
            )
            bias_metrics[column] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': (np.mean(accepted[column]) - np.mean(rejected[column])) / \
                              np.std(data[column])
            }
        
        self.bias_metrics = bias_metrics
        return bias_metrics
    
    def calculate_feature_importance(self, data, target):
        """
        Calculate statistical importance of features
        """
        importance_metrics = {}
        for column in data.columns:
            # Calculate mutual information
            mutual_info = mutual_info_classif(
                data[[column]], 
                target,
                random_state=42
            )[0]
            
            # Calculate correlation ratio
            correlation = np.corrcoef(data[column], target)[0, 1]
            
            importance_metrics[column] = {
                'mutual_information': mutual_info,
                'correlation': correlation
            }
        
        self.feature_importance = importance_metrics
        return importance_metrics

# Example usage
np.random.seed(42)

# Generate synthetic credit data
n_samples = 1000
n_features = 5

# Create features
data = pd.DataFrame({
    'income': np.random.lognormal(10, 1, n_samples),
    'debt_ratio': np.random.beta(2, 5, n_samples),
    'credit_history': np.random.normal(650, 50, n_samples),
    'employment_length': np.random.gamma(5, 2, n_samples),
    'loan_amount': np.random.lognormal(9, 0.5, n_samples)
})

# Generate target variable (default probability)
logit = -2 + 0.3 * stats.zscore(data['income']) - \
        0.4 * stats.zscore(data['debt_ratio']) + \
        0.5 * stats.zscore(data['credit_history'])
prob_default = 1 / (1 + np.exp(-logit))
target = (np.random.random(n_samples) < prob_default).astype(int)

# Analyze credit risk
analyzer = CreditRiskAnalyzer()
processed_data = analyzer.preprocess_data(data)
dist_stats = analyzer.analyze_feature_distributions(processed_data)
bias_metrics = analyzer.detect_selection_bias(processed_data, target)
importance_metrics = analyzer.calculate_feature_importance(processed_data, target)

# Print results
print("Credit Risk Analysis Results:")
print("\nFeature Distribution Statistics:")
for feature, stats in dist_stats.items():
    print(f"\n{feature}:")
    for stat, value in stats.items():
        print(f"{stat}: {value:.3f}")

print("\nSelection Bias Analysis:")
for feature, metrics in bias_metrics.items():
    print(f"\n{feature}:")
    print(f"Effect size: {metrics['effect_size']:.3f}")
    print(f"P-value: {metrics['p_value']:.3e}")

print("\nFeature Importance:")
for feature, metrics in importance_metrics.items():
    print(f"\n{feature}:")
    print(f"Mutual Information: {metrics['mutual_information']:.3f}")
    print(f"Correlation: {metrics['correlation']:.3f}")
```

Slide 11: Statistical Process Control (SPC)

This implementation provides tools for statistical process control, including control charts and process capability analysis.

```python
import numpy as np
import pandas as pd
from scipy import stats

class SPCAnalyzer:
    def __init__(self):
        self.control_limits = None
        self.process_capability = None
        
    def calculate_control_limits(self, data, n_sigma=3):
        """
        Calculate control limits for individual measurements
        """
        mean = np.mean(data)
        std = np.std(data)
        
        self.control_limits = {
            'ucl': mean + n_sigma * std,
            'lcl': mean - n_sigma * std,
            'center': mean,
            'std': std
        }
        
        return self.control_limits
    
    def check_violations(self, data):
        """
        Check for control chart rule violations
        """
        if self.control_limits is None:
            self.calculate_control_limits(data)
            
        rules = {
            'beyond_limits': np.any((data > self.control_limits['ucl']) | 
                                  (data < self.control_limits['lcl'])),
            'runs': self._check_runs(data),
            'trends': self._check_trends(data),
            'zone_violations': self._check_zones(data)
        }
        
        return rules
    
    def _check_runs(self, data, run_length=8):
        """
        Check for runs above/below centerline
        """
        above_center = data > self.control_limits['center']
        below_center = data < self.control_limits['center']
        
        runs_above = np.any([sum(1 for _ in group) >= run_length 
                            for _, group in itertools.groupby(above_center) if _])
        runs_below = np.any([sum(1 for _ in group) >= run_length 
                            for _, group in itertools.groupby(below_center) if _])
        
        return runs_above or runs_below
    
    def _check_trends(self, data, trend_length=7):
        """
        Check for trending patterns
        """
        diffs = np.diff(data)
        increasing = diffs > 0
        decreasing = diffs < 0
        
        trend_up = np.any([sum(1 for _ in group) >= trend_length-1 
                          for _, group in itertools.groupby(increasing) if _])
        trend_down = np.any([sum(1 for _ in group) >= trend_length-1 
                           for _, group in itertools.groupby(decreasing) if _])
        
        return trend_up or trend_down
    
    def _check_zones(self, data):
        """
        Check for zone violations (2/3 points in outer thirds)
        """
        sigma = self.control_limits['std']
        center = self.control_limits['center']
        
        outer_zone = (data > center + 2*sigma) | (data < center - 2*sigma)
        rolling_sum = pd.Series(outer_zone).rolling(window=3).sum()
        
        return np.any(rolling_sum >= 2)
    
    def calculate_process_capability(self, data, usl, lsl):
        """
        Calculate process capability indices
        """
        mean = np.mean(data)
        std = np.std(data)
        
        cp = (usl - lsl) / (6 * std)
        cpu = (usl - mean) / (3 * std)
        cpl = (mean - lsl) / (3 * std)
        cpk = min(cpu, cpl)
        
        self.process_capability = {
            'cp': cp,
            'cpu': cpu,
            'cpl': cpl,
            'cpk': cpk
        }
        
        return self.process_capability

# Example usage
np.random.seed(42)

# Generate process data with some abnormalities
n_samples = 100
base_process = np.random.normal(10, 1, n_samples)
trend = np.linspace(0, 2, 20)
abnormal_region = base_process.copy()
abnormal_region[40:60] += 2  # Add shift
abnormal_region[70:90] += trend  # Add trend

# Analyze process
spc = SPCAnalyzer()
control_limits = spc.calculate_control_limits(abnormal_region)
violations = spc.check_violations(abnormal_region)
capability = spc.calculate_process_capability(abnormal_region, usl=13, lsl=7)

print("Statistical Process Control Analysis:")
print("\nControl Limits:")
for limit, value in control_limits.items():
    print(f"{limit}: {value:.3f}")

print("\nRule Violations:")
for rule, violated in violations.items():
    print(f"{rule}: {'Yes' if violated else 'No'}")

print("\nProcess Capability:")
for index, value in capability.items():
    print(f"{index}: {value:.3f}")
```

Slide 12: Time Series Decomposition and Analysis

This implementation provides advanced statistical analysis of time series data, including trend detection, seasonality analysis, and anomaly detection using multiple statistical methods.

```python
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acf

class TimeSeriesAnalyzer:
    def __init__(self, data, frequency=None):
        self.data = pd.Series(data)
        self.frequency = frequency
        self.decomposition = None
        self.trend_stats = None
        self.seasonality_stats = None
        
    def decompose_series(self, model='additive'):
        """
        Decompose time series into trend, seasonal, and residual components
        """
        self.decomposition = seasonal_decompose(
            self.data, 
            model=model, 
            period=self.frequency
        )
        
        return {
            'trend': self.decomposition.trend,
            'seasonal': self.decomposition.seasonal,
            'residual': self.decomposition.resid
        }
    
    def analyze_trend(self):
        """
        Perform statistical analysis of trend component
        """
        if self.decomposition is None:
            self.decompose_series()
            
        trend = self.decomposition.trend.dropna()
        
        # Calculate trend statistics
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.arange(len(trend)),
            trend
        )
        
        # Test for trend stationarity
        adf_stat, adf_pval = adfuller(trend)[:2]
        
        self.trend_stats = {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_error': std_err,
            'adf_statistic': adf_stat,
            'adf_p_value': adf_pval
        }
        
        return self.trend_stats
    
    def analyze_seasonality(self):
        """
        Perform statistical analysis of seasonal component
        """
        if self.decomposition is None:
            self.decompose_series()
            
        seasonal = self.decomposition.seasonal.dropna()
        
        # Calculate seasonal strength
        total_var = np.var(self.data)
        seasonal_var = np.var(seasonal)
        strength = seasonal_var / total_var
        
        # Test for significant seasonality
        f_stat, p_value = self._test_seasonality(seasonal)
        
        self.seasonality_stats = {
            'strength': strength,
            'f_statistic': f_stat,
            'p_value': p_value,
            'peak_period': self._find_peak_period(seasonal)
        }
        
        return self.seasonality_stats
    
    def detect_anomalies(self, threshold=3):
        """
        Detect anomalies using multiple statistical methods
        """
        if self.decomposition is None:
            self.decompose_series()
            
        residuals = self.decomposition.resid.dropna()
        
        # Z-score method
        z_scores = np.abs(stats.zscore(residuals))
        z_score_anomalies = z_scores > threshold
        
        # IQR method
        Q1 = residuals.quantile(0.25)
        Q3 = residuals.quantile(0.75)
        IQR = Q3 - Q1
        iqr_anomalies = (residuals < (Q1 - 1.5 * IQR)) | (residuals > (Q3 + 1.5 * IQR))
        
        # CUSUM method
        cusum = np.cumsum(residuals - np.mean(residuals))
        std_cusum = np.std(cusum)
        cusum_anomalies = np.abs(cusum) > threshold * std_cusum
        
        return {
            'z_score': z_score_anomalies,
            'iqr': iqr_anomalies,
            'cusum': cusum_anomalies
        }
    
    def _test_seasonality(self, seasonal_component):
        """
        Perform F-test for seasonality
        """
        groups = [group for _, group in seasonal_component.groupby(
            seasonal_component.index % self.frequency
        )]
        
        f_stat, p_value = stats.f_oneway(*groups)
        return f_stat, p_value
    
    def _find_peak_period(self, seasonal_component):
        """
        Find dominant seasonal period using spectral analysis
        """
        fft = np.fft.fft(seasonal_component)
        freqs = np.fft.fftfreq(len(seasonal_component))
        peak_freq = freqs[np.argmax(np.abs(fft))]
        
        return abs(1/peak_freq) if peak_freq != 0 else np.inf

# Example usage
np.random.seed(42)

# Generate synthetic time series with trend, seasonality, and noise
n_points = 365
time = np.arange(n_points)

# Components
trend = 0.05 * time
seasonality = 5 * np.sin(2 * np.pi * time / 365) # Annual cycle
noise = np.random.normal(0, 1, n_points)

# Combine components
ts_data = trend + seasonality + noise

# Add some anomalies
ts_data[100:110] += 10  # Level shift
ts_data[200] += 15      # Single point anomaly
ts_data[300:320] *= 1.5 # Variance change

# Analyze time series
analyzer = TimeSeriesAnalyzer(ts_data, frequency=365)
decomposition = analyzer.decompose_series()
trend_analysis = analyzer.analyze_trend()
seasonality_analysis = analyzer.analyze_seasonality()
anomalies = analyzer.detect_anomalies()

print("Time Series Analysis Results:")
print("\nTrend Analysis:")
for metric, value in trend_analysis.items():
    print(f"{metric}: {value:.4f}")

print("\nSeasonality Analysis:")
for metric, value in seasonality_analysis.items():
    print(f"{metric}: {value:.4f}")

print("\nAnomaly Detection:")
for method, anomaly_mask in anomalies.items():
    print(f"{method} anomalies detected: {sum(anomaly_mask)}")
```

Slide 13: Additional Resources

*   "Statistical Learning with Sparsity: The Lasso and Generalizations" [https://arxiv.org/abs/1303.0518](https://arxiv.org/abs/1303.0518)
*   "A Tutorial on Principal Component Analysis" [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
*   "Deep Learning in Statistical Machine Learning" [https://arxiv.org/abs/1603.04467](https://arxiv.org/abs/1603.04467)
*   "Modern Statistics for Modern Biology" [https://arxiv.org/abs/1504.00641](https://arxiv.org/abs/1504.00641)
*   "Causal Inference in Statistics: A Primer" [https://arxiv.org/abs/1505.00269](https://arxiv.org/abs/1505.00269)

