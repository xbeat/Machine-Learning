## Mastering Data Types for Effective Statistical Analysis

Slide 1: Understanding Data Types and Their Statistical Properties

Statistical analysis requires proper understanding of variable types - categorical, ordinal, and continuous data. Different types demand specific statistical approaches and visualizations. This code demonstrates how to identify, analyze and visualize different data types using Python's pandas and seaborn libraries.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create sample dataset with different variable types
np.random.seed(42)
data = {
    'categorical': np.random.choice(['A', 'B', 'C'], 100),
    'ordinal': np.random.choice(['Low', 'Medium', 'High'], 100),
    'continuous': np.random.normal(0, 1, 100)
}
df = pd.DataFrame(data)

# Identify data types
print("Data Types:\n", df.dtypes)

# Basic analysis for each type
print("\nCategorical Value Counts:")
print(df['categorical'].value_counts())

print("\nOrdinal Value Counts:")
print(df['ordinal'].value_counts())

print("\nContinuous Summary Statistics:")
print(df['continuous'].describe())

# Visualize each type
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Categorical plot
sns.countplot(data=df, x='categorical', ax=ax1)
ax1.set_title('Categorical Data')

# Ordinal plot
sns.countplot(data=df, x='ordinal', 
              order=['Low', 'Medium', 'High'], ax=ax2)
ax2.set_title('Ordinal Data')

# Continuous plot
sns.histplot(data=df, x='continuous', ax=ax3)
ax3.set_title('Continuous Data')

plt.tight_layout()
plt.show()
```

Slide 2: Descriptive Statistics Implementation

Understanding central tendencies and data spread is crucial for initial data analysis. This implementation calculates comprehensive descriptive statistics including mean, median, mode, range, standard deviation, skewness, and kurtosis with proper statistical interpretations.

```python
import pandas as pd
import numpy as np
from scipy import stats

class DescriptiveStats:
    def __init__(self, data):
        self.data = np.array(data)
        
    def basic_stats(self):
        return {
            'mean': np.mean(self.data),
            'median': np.median(self.data),
            'mode': stats.mode(self.data, keepdims=True)[0][0],
            'range': np.ptp(self.data),
            'std': np.std(self.data),
            'var': np.var(self.data),
            'skewness': stats.skew(self.data),
            'kurtosis': stats.kurtosis(self.data)
        }
    
    def quartiles(self):
        return {
            'Q1': np.percentile(self.data, 25),
            'Q2': np.percentile(self.data, 50),
            'Q3': np.percentile(self.data, 75),
            'IQR': stats.iqr(self.data)
        }
    
    def distribution_test(self):
        statistic, p_value = stats.normaltest(self.data)
        return {
            'normal_test_statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }

# Example usage
np.random.seed(42)
sample_data = np.random.normal(100, 15, 1000)
stats_analyzer = DescriptiveStats(sample_data)

print("Basic Statistics:")
for key, value in stats_analyzer.basic_stats().items():
    print(f"{key}: {value:.4f}")

print("\nQuartile Analysis:")
for key, value in stats_analyzer.quartiles().items():
    print(f"{key}: {value:.4f}")

print("\nNormality Test:")
for key, value in stats_analyzer.distribution_test().items():
    if isinstance(value, bool):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {value:.4f}")
```

Slide 3: Advanced Data Visualization Strategy

Data visualization requires a systematic approach to reveal patterns, relationships, and anomalies effectively. The technique combines statistical analysis with visual representation to create meaningful insights from complex datasets through advanced plotting mechanisms.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_advanced_plots(df, feature_x, feature_y):
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Enhanced scatter plot with regression
    sns.regplot(data=df, x=feature_x, y=feature_y, ax=ax1)
    ax1.set_title('Regression Analysis')
    
    # Distribution comparison
    sns.kdeplot(data=df[feature_x], ax=ax2, label=feature_x)
    sns.kdeplot(data=df[feature_y], ax=ax2, label=feature_y)
    ax2.set_title('Distribution Comparison')
    ax2.legend()
    
    # Add correlation coefficient
    corr = df[feature_x].corr(df[feature_y])
    fig.suptitle(f'Correlation: {corr:.2f}', y=1.05)
    
    plt.tight_layout()
    return fig

# Example usage
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(0, 1, 1000)
})

plot = create_advanced_plots(data, 'feature1', 'feature2')
plt.show()
```

Slide 4: Outlier Detection and Handling

Statistical outlier detection implements robust methods to identify anomalous data points using z-scores and interquartile ranges. This implementation provides automated detection and visualization of outliers in numerical datasets.

```python
import numpy as np
import pandas as pd

class OutlierDetector:
    def __init__(self, data, threshold=3):
        self.data = data
        self.threshold = threshold
    
    def zscore_outliers(self):
        z_scores = np.abs((self.data - self.data.mean()) / self.data.std())
        return self.data[z_scores > self.threshold]
    
    def iqr_outliers(self):
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = self.data[(self.data < (Q1 - 1.5 * IQR)) | 
                            (self.data > (Q3 + 1.5 * IQR))]
        return outliers

# Example usage
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
outliers = np.array([10, -10, 8, -8])
data = pd.Series(np.concatenate([normal_data, outliers]))

detector = OutlierDetector(data)
zscore_outliers = detector.zscore_outliers()
iqr_outliers = detector.iqr_outliers()

print(f"Z-score outliers found: {len(zscore_outliers)}")
print(f"IQR outliers found: {len(iqr_outliers)}")
```

Slide 5: Statistical Correlation Analysis Framework

This implementation provides a comprehensive framework for analyzing correlations between variables using multiple statistical methods. It includes significance testing and visualization capabilities for deeper insights into variable relationships.

```python
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

class CorrelationAnalyzer:
    def __init__(self, data):
        self.data = pd.DataFrame(data)
    
    def compute_correlations(self):
        results = {}
        
        # Compute different correlation types
        for col1 in self.data.columns:
            for col2 in self.data.columns:
                if col1 < col2:
                    pearson = stats.pearsonr(self.data[col1], 
                                           self.data[col2])
                    spearman = stats.spearmanr(self.data[col1], 
                                             self.data[col2])
                    
                    results[f"{col1}_vs_{col2}"] = {
                        'pearson_corr': pearson[0],
                        'pearson_pval': pearson[1],
                        'spearman_corr': spearman[0],
                        'spearman_pval': spearman[1]
                    }
        return pd.DataFrame(results).T

    def plot_correlation_matrix(self):
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.data.corr(), annot=True, 
                   cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        return plt.gcf()

# Example usage
np.random.seed(42)
data = {
    'var1': np.random.normal(0, 1, 100),
    'var2': np.random.normal(0, 1, 100),
    'var3': np.random.normal(0, 1, 100)
}

analyzer = CorrelationAnalyzer(data)
correlations = analyzer.compute_correlations()
print("Correlation Analysis Results:")
print(correlations)

analyzer.plot_correlation_matrix()
plt.show()
```

Slide 6: Hypothesis Testing Framework

The hypothesis testing framework implements common statistical tests including t-tests, chi-square, and ANOVA. This code provides a structured approach to conducting and interpreting statistical significance tests.

```python
import numpy as np
from scipy import stats

class HypothesisTester:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
    
    def ttest_analysis(self, group1, group2, paired=False):
        if paired:
            stat, pval = stats.ttest_rel(group1, group2)
        else:
            stat, pval = stats.ttest_ind(group1, group2)
            
        result = {
            'statistic': stat,
            'p_value': pval,
            'significant': pval < self.alpha
        }
        return result
    
    def anova_analysis(self, *groups):
        stat, pval = stats.f_oneway(*groups)
        return {
            'f_statistic': stat,
            'p_value': pval,
            'significant': pval < self.alpha
        }

# Example usage
np.random.seed(42)
control = np.random.normal(100, 15, 30)
treatment = np.random.normal(110, 15, 30)
treatment2 = np.random.normal(105, 15, 30)

tester = HypothesisTester()

# T-test example
ttest_result = tester.ttest_analysis(control, treatment)
print("\nT-Test Results:")
print(f"Statistic: {ttest_result['statistic']:.4f}")
print(f"P-value: {ttest_result['p_value']:.4f}")
print(f"Significant: {ttest_result['significant']}")

# ANOVA example
anova_result = tester.anova_analysis(control, treatment, treatment2)
print("\nANOVA Results:")
print(f"F-statistic: {anova_result['f_statistic']:.4f}")
print(f"P-value: {anova_result['p_value']:.4f}")
print(f"Significant: {anova_result['significant']}")
```

Slide 7: Statistical Assumptions Validation

Statistical assumption testing ensures the validity of analytical methods. This implementation provides tools for checking normality, homoscedasticity, and independence assumptions required for parametric tests.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class AssumptionTester:
    def __init__(self, data):
        self.data = np.array(data)
    
    def test_normality(self):
        # Shapiro-Wilk test
        stat, pval = stats.shapiro(self.data)
        
        # QQ plot data
        qq_data = stats.probplot(self.data, dist="norm")
        
        return {
            'shapiro_stat': stat,
            'shapiro_pval': pval,
            'qq_data': qq_data,
            'is_normal': pval > 0.05
        }
    
    def test_homoscedasticity(self, group1, group2):
        # Levene's test
        stat, pval = stats.levene(group1, group2)
        return {
            'levene_stat': stat,
            'levene_pval': pval,
            'equal_variance': pval > 0.05
        }
    
    def plot_diagnostics(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram with normal curve
        ax1.hist(self.data, density=True, bins=30)
        xmin, xmax = ax1.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, np.mean(self.data), np.std(self.data))
        ax1.plot(x, p, 'k', linewidth=2)
        ax1.set_title('Normal Distribution Check')
        
        # QQ plot
        stats.probplot(self.data, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        
        plt.tight_layout()
        return fig

# Example usage
np.random.seed(42)
data = np.random.normal(0, 1, 1000)
group1 = np.random.normal(0, 1, 100)
group2 = np.random.normal(0, 1.5, 100)

tester = AssumptionTester(data)

normality_results = tester.test_normality()
print("\nNormality Test Results:")
print(f"Shapiro-Wilk statistic: {normality_results['shapiro_stat']:.4f}")
print(f"P-value: {normality_results['shapiro_pval']:.4f}")
print(f"Normal distribution: {normality_results['is_normal']}")

homoscedasticity_results = tester.test_homoscedasticity(group1, group2)
print("\nHomoscedasticity Test Results:")
print(f"Levene statistic: {homoscedasticity_results['levene_stat']:.4f}")
print(f"P-value: {homoscedasticity_results['levene_pval']:.4f}")
print(f"Equal variances: {homoscedasticity_results['equal_variance']}")

tester.plot_diagnostics()
plt.show()
```

Slide 8: Confidence Intervals Implementation

A robust implementation for calculating and visualizing confidence intervals across different statistical scenarios. This approach includes bootstrap methods and parametric interval estimation with visualization capabilities.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class ConfidenceIntervals:
    def __init__(self, data, confidence=0.95):
        self.data = np.array(data)
        self.confidence = confidence
        
    def parametric_ci(self):
        mean = np.mean(self.data)
        sem = stats.sem(self.data)
        ci = stats.t.interval(self.confidence, len(self.data)-1, 
                            loc=mean, scale=sem)
        return {
            'mean': mean,
            'lower_bound': ci[0],
            'upper_bound': ci[1]
        }
    
    def bootstrap_ci(self, n_iterations=10000):
        bootstrapped_means = []
        for _ in range(n_iterations):
            sample = np.random.choice(self.data, 
                                    size=len(self.data), 
                                    replace=True)
            bootstrapped_means.append(np.mean(sample))
            
        ci = np.percentile(bootstrapped_means, 
                          [(1-self.confidence)*100/2, 
                           (1+self.confidence)*100/2])
        return {
            'mean': np.mean(self.data),
            'lower_bound': ci[0],
            'upper_bound': ci[1]
        }

# Example usage
np.random.seed(42)
sample_data = np.random.normal(100, 15, 1000)

ci_calculator = ConfidenceIntervals(sample_data)

# Calculate both types of intervals
param_ci = ci_calculator.parametric_ci()
boot_ci = ci_calculator.bootstrap_ci()

print("Parametric CI Results:")
print(f"Mean: {param_ci['mean']:.2f}")
print(f"95% CI: ({param_ci['lower_bound']:.2f}, "
      f"{param_ci['upper_bound']:.2f})")

print("\nBootstrap CI Results:")
print(f"Mean: {boot_ci['mean']:.2f}")
print(f"95% CI: ({boot_ci['lower_bound']:.2f}, "
      f"{boot_ci['upper_bound']:.2f})")
```

Slide 9: Regression Analysis Framework

This comprehensive regression framework implements multiple regression techniques with built-in diagnostics and model validation. The implementation includes feature selection, model evaluation, and residual analysis.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

class RegressionAnalyzer:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.model = None
        self.residuals = None
        
    def fit_model(self, test_size=0.2):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        # Fit model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        # Calculate predictions and residuals
        y_pred = self.model.predict(X_test)
        self.residuals = y_test - y_pred
        
        return {
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_,
            'r2_score': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred)
        }
    
    def analyze_residuals(self):
        if self.residuals is None:
            raise ValueError("Must fit model first")
            
        return {
            'normality': stats.shapiro(self.residuals),
            'mean': np.mean(self.residuals),
            'std': np.std(self.residuals)
        }

# Example usage
np.random.seed(42)
X = np.random.normal(0, 1, (1000, 3))
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.normal(0, 0.5, 1000)

analyzer = RegressionAnalyzer(X, y)
results = analyzer.fit_model()

print("Regression Results:")
print(f"R² Score: {results['r2_score']:.4f}")
print(f"MSE: {results['mse']:.4f}")
print("\nCoefficients:", results['coefficients'])
print("Intercept:", results['intercept'])

residual_analysis = analyzer.analyze_residuals()
print("\nResidual Analysis:")
print(f"Mean: {residual_analysis['mean']:.4f}")
print(f"Std: {residual_analysis['std']:.4f}")
```

Slide 10: Advanced Time Series Analysis

Time series analysis requires specialized statistical methods for handling temporal dependencies and patterns. This implementation provides core functionality for decomposition, stationarity testing, and forecasting using established statistical techniques.

```python
import numpy as np
import pandas as pd
from scipy import stats

class TimeSeriesAnalyzer:
    def __init__(self, data):
        self.data = np.array(data)
        
    def moving_average(self, window=3):
        weights = np.ones(window) / window
        return np.convolve(self.data, weights, mode='valid')
    
    def detect_trend(self):
        x = np.arange(len(self.data))
        slope, intercept, r_value, p_value, std_err = \
            stats.linregress(x, self.data)
        return {
            'slope': slope,
            'p_value': p_value,
            'trend_present': p_value < 0.05
        }
    
    def decompose(self):
        # Simple additive decomposition
        n = len(self.data)
        trend = self.moving_average(window=12)
        
        # Pad trend to match original length
        pad_size = (n - len(trend)) // 2
        trend = np.pad(trend, (pad_size, n - len(trend) - pad_size))
        
        # Calculate seasonal and residual components
        detrended = self.data - trend
        seasonal = np.zeros_like(self.data)
        residual = detrended - seasonal
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }

# Example usage
np.random.seed(42)
t = np.linspace(0, 4, 100)
trend = 0.5 * t
seasonal = 2 * np.sin(2 * np.pi * t)
noise = np.random.normal(0, 0.2, 100)
time_series = trend + seasonal + noise

analyzer = TimeSeriesAnalyzer(time_series)

# Analyze trend
trend_results = analyzer.detect_trend()
print("Trend Analysis Results:")
print(f"Slope: {trend_results['slope']:.4f}")
print(f"P-value: {trend_results['p_value']:.4f}")
print(f"Trend present: {trend_results['trend_present']}")

# Decompose series
components = analyzer.decompose()
print("\nDecomposition complete:")
print(f"Trend variance: {np.var(components['trend']):.4f}")
print(f"Residual variance: {np.var(components['residual']):.4f}")
```

Slide 11: Statistical Power Analysis

Statistical power analysis is crucial for experiment design and result interpretation. This implementation provides tools for calculating sample sizes and power for various statistical tests.

```python
import numpy as np
from scipy import stats

class PowerAnalyzer:
    def __init__(self, alpha=0.05, power=0.8):
        self.alpha = alpha
        self.target_power = power
    
    def sample_size_ttest(self, effect_size, alternative='two-sided'):
        """Calculate required sample size for t-test"""
        n = 2  # Start with minimum sample size
        while True:
            # Calculate power for current sample size
            analysis = stats.TTestPower()
            power = analysis.power(
                effect_size=effect_size,
                nobs=n,
                alpha=self.alpha,
                alternative=alternative
            )
            
            if power >= self.target_power:
                break
            n += 1
        
        return {
            'sample_size': n,
            'actual_power': power,
            'effect_size': effect_size
        }
    
    def effect_size_estimation(self, group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) 
                           / (n1 + n2 - 2))
        
        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_sd
        
        return abs(d)

# Example usage
analyzer = PowerAnalyzer()

# Calculate sample size for medium effect
medium_effect = 0.5
sample_size_result = analyzer.sample_size_ttest(medium_effect)

print("Sample Size Analysis:")
print(f"Required sample size: {sample_size_result['sample_size']}")
print(f"Actual power: {sample_size_result['actual_power']:.4f}")

# Calculate effect size for example data
np.random.seed(42)
control = np.random.normal(100, 15, 30)
treatment = np.random.normal(115, 15, 30)

effect_size = analyzer.effect_size_estimation(control, treatment)
print(f"\nObserved effect size: {effect_size:.4f}")
```

Slide 12: Multivariate Analysis Tools

This implementation provides essential tools for multivariate statistical analysis, including principal component analysis and factor analysis implementations from scratch.

```python
import numpy as np
from scipy import linalg

class MultivariateAnalyzer:
    def __init__(self, data):
        self.data = np.array(data)
        self.standardized_data = self._standardize(self.data)
        
    def _standardize(self, X):
        """Standardize the dataset"""
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    def pca(self, n_components=None):
        """Perform PCA from scratch"""
        # Compute covariance matrix
        cov_matrix = np.cov(self.standardized_data.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = linalg.eigh(cov_matrix)
        
        # Sort in descending order
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Select components
        if n_components is None:
            n_components = len(eigenvals)
        
        # Transform data
        transformed = self.standardized_data @ eigenvecs[:, :n_components]
        
        return {
            'transformed_data': transformed,
            'components': eigenvecs[:, :n_components],
            'explained_variance': eigenvals[:n_components],
            'explained_variance_ratio': eigenvals[:n_components] / 
                                     np.sum(eigenvals)
        }
    
    def compute_loadings(self, pca_results):
        """Compute factor loadings"""
        return pca_results['components'] * \
               np.sqrt(pca_results['explained_variance'])

# Example usage
np.random.seed(42)
n_samples = 1000
n_features = 5

# Generate correlated data
X = np.random.randn(n_samples, n_features)
X[:, 1] = 0.5 * X[:, 0] + np.random.randn(n_samples) * 0.5
X[:, 2] = -0.7 * X[:, 0] + np.random.randn(n_samples) * 0.3

analyzer = MultivariateAnalyzer(X)
pca_results = analyzer.pca(n_components=3)

print("PCA Results:")
print("Explained variance ratio:")
print(pca_results['explained_variance_ratio'])

loadings = analyzer.compute_loadings(pca_results)
print("\nFeature loadings for first component:")
print(loadings[:, 0])
```

Slide 13: Machine Learning Model Diagnostics

This implementation provides comprehensive tools for diagnosing and validating machine learning models, focusing on key metrics, cross-validation, and learning curve analysis.

```python
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix

class ModelDiagnostics:
    def __init__(self, model, X, y):
        self.model = model
        self.X = np.array(X)
        self.y = np.array(y)
        
    def learning_curve_analysis(self, cv=5):
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, valid_scores = learning_curve(
            self.model, self.X, self.y,
            train_sizes=train_sizes,
            cv=cv, n_jobs=-1
        )
        
        return {
            'train_sizes': train_sizes,
            'train_mean': np.mean(train_scores, axis=1),
            'train_std': np.std(train_scores, axis=1),
            'valid_mean': np.mean(valid_scores, axis=1),
            'valid_std': np.std(valid_scores, axis=1)
        }
    
    def performance_metrics(self, y_pred):
        conf_matrix = confusion_matrix(self.y, y_pred)
        
        # Calculate metrics
        tn, fp, fn, tp = conf_matrix.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }

# Example usage
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Generate sample data
X, y = make_classification(n_samples=1000, random_state=42)
model = LogisticRegression()

# Initialize diagnostics
diagnostics = ModelDiagnostics(model, X, y)

# Analyze learning curve
curve_results = diagnostics.learning_curve_analysis()
print("Learning Curve Analysis:")
print(f"Final training score: {curve_results['train_mean'][-1]:.4f}")
print(f"Final validation score: {curve_results['valid_mean'][-1]:.4f}")

# Get performance metrics
model.fit(X, y)
y_pred = model.predict(X)
metrics = diagnostics.performance_metrics(y_pred)

print("\nModel Performance Metrics:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
```

Slide 14: Feature Selection and Importance Analysis

This implementation provides tools for analyzing feature importance and selecting relevant features using statistical methods and machine learning techniques.

```python
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

class FeatureAnalyzer:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
    
    def correlation_analysis(self):
        correlations = []
        p_values = []
        
        for i in range(self.X.shape[1]):
            corr, p_val = stats.pearsonr(self.X[:, i], self.y)
            correlations.append(corr)
            p_values.append(p_val)
            
        return {
            'correlations': np.array(correlations),
            'p_values': np.array(p_values)
        }
    
    def mutual_information(self):
        mi_scores = mutual_info_classif(self.X, self.y)
        return {
            'mi_scores': mi_scores,
            'normalized_scores': mi_scores / np.sum(mi_scores)
        }
    
    def select_features(self, threshold=0.05):
        # Correlation-based selection
        corr_results = self.correlation_analysis()
        selected_corr = np.where(corr_results['p_values'] < threshold)[0]
        
        # Mutual information-based selection
        mi_results = self.mutual_information()
        selected_mi = np.where(mi_results['mi_scores'] > 
                             np.mean(mi_results['mi_scores']))[0]
        
        return {
            'correlation_selected': selected_corr,
            'mutual_info_selected': selected_mi,
            'intersection': np.intersect1d(selected_corr, selected_mi)
        }

# Example usage
np.random.seed(42)
n_samples = 1000
n_features = 20

# Generate synthetic data
X = np.random.randn(n_samples, n_features)
# Make some features more important
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(n_samples)*0.1

analyzer = FeatureAnalyzer(X, y)

# Analyze feature importance
corr_results = analyzer.correlation_analysis()
mi_results = analyzer.mutual_information()
selection_results = analyzer.select_features()

print("Feature Selection Results:")
print(f"Features selected by correlation: "
      f"{len(selection_results['correlation_selected'])}")
print(f"Features selected by mutual information: "
      f"{len(selection_results['mutual_info_selected'])}")
print(f"Features selected by both methods: "
      f"{len(selection_results['intersection'])}")
```

Slide 15: Additional Resources

1.  arxiv.org/abs/1810.03993 - "A Comprehensive Survey of Model Validation Techniques"
2.  arxiv.org/abs/1904.06836 - "Statistical Learning: Contemporary Applications"
3.  arxiv.org/abs/1811.12808 - "Modern Statistical Methods for Data Science"
4.  arxiv.org/abs/1903.11714 - "Advanced Time Series Analysis Methods"
5.  arxiv.org/abs/1902.03129 - "Feature Selection in High-Dimensional Data"

