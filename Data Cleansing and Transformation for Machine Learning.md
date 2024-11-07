## Data Cleansing and Transformation for Machine Learning
Slide 1: Data Loading and Initial Assessment

Data preparation begins with loading and assessing the raw data to understand its structure, identify potential issues, and plan the necessary cleaning steps. This fundamental process establishes the foundation for all subsequent data preprocessing tasks in machine learning workflows.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load sample dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Initial assessment
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)
print("\nBasic Statistics:\n", df.describe())

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Output sample
"""
Dataset Shape: (569, 30)
Missing Values: 
mean radius                0
mean texture               0
mean perimeter            0
...
Data Types:
mean radius      float64
mean texture     float64
...
"""
```

Slide 2: Missing Value Detection and Visualization

Understanding the patterns of missing values is crucial for deciding on appropriate imputation strategies. Visualization helps identify potential relationships between missing values and assists in making informed decisions about handling them.

```python
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_missing_values(df):
    # Create missing value matrix
    missing_matrix = df.isnull()
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(missing_matrix, 
                yticklabels=False,
                cmap='viridis',
                cbar_kws={'label': 'Missing Values'})
    plt.title('Missing Value Patterns')
    
    # Calculate missing percentages
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    missing_info = pd.DataFrame({
        'Column': df.columns,
        'Missing Percentage': missing_percentages
    }).sort_values('Missing Percentage', ascending=False)
    
    return missing_info

# Example usage with synthetic missing values
df_with_missing = df.copy()
df_with_missing.iloc[np.random.randint(0, len(df), 50), 
                    np.random.randint(0, df.shape[1], 50)] = np.nan

missing_info = visualize_missing_values(df_with_missing)
print("\nMissing Value Summary:\n", missing_info)
```

Slide 3: Advanced Missing Value Imputation

Traditional imputation methods like mean or median replacement can be inadequate for complex datasets. This implementation demonstrates sophisticated imputation techniques using iterative imputation, which considers the relationships between features.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

def advanced_imputation(df):
    # Initialize iterative imputer with random forest estimator
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100),
        max_iter=10,
        random_state=42
    )
    
    # Perform imputation
    imputed_data = imputer.fit_transform(df)
    
    # Create new dataframe with imputed values
    df_imputed = pd.DataFrame(imputed_data, 
                            columns=df.columns, 
                            index=df.index)
    
    # Validation metrics
    imputation_quality = {
        'convergence': imputer.n_iter_,
        'feature_importances': dict(zip(
            df.columns,
            imputer.estimator_.feature_importances_
        ))
    }
    
    return df_imputed, imputation_quality

# Example usage
df_imputed, quality_metrics = advanced_imputation(df_with_missing)
print("Imputation Quality Metrics:\n", quality_metrics)
```

Slide 4: Outlier Detection Using Multiple Methods

A comprehensive approach to outlier detection combines statistical methods, density-based approaches, and machine learning techniques to identify anomalous data points with higher confidence and reduced false positives.

```python
from sklearn.ensemble import IsolationForest
from scipy import stats

def multi_method_outlier_detection(df, column):
    # Z-score method
    z_scores = np.abs(stats.zscore(df[column]))
    z_score_outliers = df[z_scores > 3]
    
    # IQR method
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | 
                      (df[column] > (Q3 + 1.5 * IQR))]
    
    # Isolation Forest method
    iso_forest = IsolationForest(contamination=0.1, 
                                random_state=42)
    predictions = iso_forest.fit_predict(df[[column]])
    isolation_outliers = df[predictions == -1]
    
    return {
        'z_score': z_score_outliers,
        'iqr': iqr_outliers,
        'isolation_forest': isolation_outliers,
        'consensus': df[
            df.index.isin(z_score_outliers.index) & 
            df.index.isin(iqr_outliers.index)
        ]
    }

# Example usage
results = multi_method_outlier_detection(df, 'mean radius')
for method, outliers in results.items():
    print(f"\n{method.title()} Outliers Count: {len(outliers)}")
```

Slide 5: Feature Scaling Implementation

Feature scaling is essential for ensuring all variables contribute equally to the model. This implementation provides a comprehensive approach to scaling, including both standardization and normalization with safeguards against data leakage during cross-validation.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

class AdvancedScaler(BaseEstimator, TransformerMixin):
    def __init__(self, method='standard', custom_range=(-1, 1)):
        self.method = method
        self.custom_range = custom_range
        self.scaler = None
        self.feature_stats = {}
    
    def fit(self, X, y=None):
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=self.custom_range)
        
        # Store feature statistics before scaling
        self.feature_stats = {
            'mean': X.mean(),
            'std': X.std(),
            'min': X.min(),
            'max': X.max()
        }
        
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)

# Example usage
scaler = AdvancedScaler(method='standard')
X_scaled = scaler.fit_transform(df)

print("Original Stats:\n", scaler.feature_stats['mean'])
print("\nScaled Stats:\n", X_scaled.mean())
```

Slide 6: Advanced Feature Engineering

Feature engineering transforms raw data into meaningful representations that capture domain knowledge and improve model performance. This implementation demonstrates automated feature generation and selection based on statistical significance.

```python
import scipy.stats as stats
from itertools import combinations

class FeatureEngineer:
    def __init__(self, significance_level=0.05):
        self.significance_level = significance_level
        self.selected_features = []
        
    def generate_polynomial_features(self, X, degree=2):
        feature_names = X.columns
        poly_features = {}
        
        for col1, col2 in combinations(feature_names, 2):
            # Multiplication interaction
            poly_features[f"{col1}_{col2}_mult"] = X[col1] * X[col2]
            # Ratio interaction (with safeguards)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = X[col1] / X[col2]
                poly_features[f"{col1}_{col2}_ratio"] = np.where(
                    np.isfinite(ratio), ratio, 0
                )
        
        # Add polynomial terms
        for col in feature_names:
            for d in range(2, degree + 1):
                poly_features[f"{col}_power_{d}"] = X[col] ** d
                
        return pd.DataFrame(poly_features)
    
    def select_significant_features(self, X, y):
        # Perform statistical tests
        significant_features = []
        
        for column in X.columns:
            correlation = stats.pearsonr(X[column], y)
            if correlation[1] < self.significance_level:
                significant_features.append({
                    'feature': column,
                    'correlation': correlation[0],
                    'p_value': correlation[1]
                })
        
        self.selected_features = pd.DataFrame(significant_features)
        return self.selected_features

# Example usage with breast cancer dataset
X = df
y = data.target

engineer = FeatureEngineer()
poly_features = engineer.generate_polynomial_features(X)
significant_features = engineer.select_significant_features(
    pd.concat([X, poly_features], axis=1), y
)

print("Top 5 Most Significant Features:\n", 
      significant_features.nsmallest(5, 'p_value'))
```

Slide 7: Data Type Transformation and Encoding

Efficient handling of mixed data types is crucial for model performance. This implementation provides a sophisticated approach to automatic data type detection and appropriate encoding strategies for different variable types.

```python
class DataTypeTransformer:
    def __init__(self, max_categories=10):
        self.max_categories = max_categories
        self.encoding_maps = {}
        self.dtypes = {}
        
    def infer_data_types(self, df):
        for column in df.columns:
            # Check if numeric
            if pd.api.types.is_numeric_dtype(df[column]):
                if df[column].nunique() <= 2:
                    self.dtypes[column] = 'binary'
                else:
                    self.dtypes[column] = 'continuous'
            # Check if datetime
            elif pd.to_datetime(df[column], errors='coerce').notnull().all():
                self.dtypes[column] = 'datetime'
            # Check if categorical
            else:
                if df[column].nunique() <= self.max_categories:
                    self.dtypes[column] = 'categorical'
                else:
                    self.dtypes[column] = 'text'
                    
        return self.dtypes
    
    def transform_column(self, series, dtype):
        if dtype == 'binary':
            return pd.get_dummies(series, prefix=series.name)
        elif dtype == 'categorical':
            # Ordinal encoding for ordered categories
            if hasattr(series, 'cat') and series.cat.ordered:
                self.encoding_maps[series.name] = {
                    val: idx for idx, val in enumerate(series.cat.categories)
                }
                return series.map(self.encoding_maps[series.name])
            # One-hot encoding for unordered categories
            else:
                return pd.get_dummies(series, prefix=series.name)
        elif dtype == 'datetime':
            dt = pd.to_datetime(series)
            return pd.DataFrame({
                f'{series.name}_year': dt.dt.year,
                f'{series.name}_month': dt.dt.month,
                f'{series.name}_day': dt.dt.day,
                f'{series.name}_dayofweek': dt.dt.dayofweek
            })
        else:
            return series

# Example usage with synthetic mixed data
mixed_data = pd.DataFrame({
    'numeric': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'binary': np.random.choice([0, 1], 100),
    'date': pd.date_range('2023-01-01', periods=100)
})

transformer = DataTypeTransformer()
dtypes = transformer.infer_data_types(mixed_data)
print("Inferred Data Types:\n", dtypes)

# Transform each column
transformed_data = pd.concat([
    transformer.transform_column(mixed_data[col], dtype)
    for col, dtype in dtypes.items()
], axis=1)

print("\nTransformed Data Shape:", transformed_data.shape)
```

Slide 8: Time Series Data Preprocessing

Time series data requires specialized preprocessing techniques to capture temporal dependencies and handle seasonality. This implementation provides comprehensive tools for time series feature engineering and decomposition.

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

class TimeSeriesPreprocessor:
    def __init__(self, freq='D'):
        self.freq = freq
        self.decomposition = None
        
    def create_temporal_features(self, df, date_column):
        df = df.copy()
        df['date'] = pd.to_datetime(df[date_column])
        
        # Extract temporal components
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        
        # Create cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
        
        return df
    
    def decompose_series(self, series, period=None):
        if period is None:
            # Attempt to automatically detect period
            from statsmodels.tsa.stattools import acf
            acf_values = acf(series, nlags=len(series)//2)
            period = np.argmax(acf_values[1:]) + 1
        
        self.decomposition = seasonal_decompose(
            series, 
            period=period,
            extrapolate_trend='freq'
        )
        
        return pd.DataFrame({
            'trend': self.decomposition.trend,
            'seasonal': self.decomposition.seasonal,
            'residual': self.decomposition.resid
        })

# Example usage with synthetic time series data
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + \
         np.random.normal(0, 0.1, len(dates))

ts_data = pd.DataFrame({
    'date': dates,
    'value': values
})

preprocessor = TimeSeriesPreprocessor()
ts_features = preprocessor.create_temporal_features(ts_data, 'date')
decomposition = preprocessor.decompose_series(ts_data['value'], period=365)

print("Temporal Features:\n", ts_features.head())
print("\nDecomposition Components:\n", decomposition.head())
```

Slide 9: Text Data Preprocessing Pipeline

Text data requires specialized cleaning and normalization techniques. This implementation provides a comprehensive pipeline for text preprocessing, including advanced tokenization and custom cleaning rules.

```python
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation

class TextPreprocessor:
    def __init__(self, language='english'):
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words(language))
        self.custom_patterns = {
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'numbers': r'\b\d+\b',
            'special_chars': f'[{re.escape(punctuation)}]'
        }
        
    def clean_text(self, text, remove_numbers=True):
        text = text.lower()
        
        # Remove URLs and emails
        text = re.sub(self.custom_patterns['url'], ' URL ', text)
        text = re.sub(self.custom_patterns['email'], ' EMAIL ', text)
        
        # Handle numbers
        if remove_numbers:
            text = re.sub(self.custom_patterns['numbers'], ' NUM ', text)
            
        # Remove special characters
        text = re.sub(self.custom_patterns['special_chars'], ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def process(self, text, remove_stopwords=True):
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
            
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return {
            'processed_text': ' '.join(tokens),
            'tokens': tokens,
            'token_count': len(tokens)
        }

# Example usage
text = """
Check out our website at https://example.com! 
Contact us at info@example.com or call 123-456-7890.
The product costs $99.99 and has 4.5/5 stars!!!
"""

preprocessor = TextPreprocessor()
result = preprocessor.process(text)

print("Original Text:\n", text)
print("\nProcessed Text:\n", result['processed_text'])
print("\nToken Count:", result['token_count'])
```

Slide 10: Data Quality Assessment and Reporting

Systematic evaluation of data quality is crucial for maintaining robust machine learning pipelines. This implementation provides comprehensive quality metrics and generates detailed reports for identifying potential issues.

```python
import pandas as pd
import numpy as np
from scipy import stats

class DataQualityAnalyzer:
    def __init__(self, threshold_missing=0.1, threshold_unique=0.95):
        self.threshold_missing = threshold_missing
        self.threshold_unique = threshold_unique
        self.report = {}
        
    def analyze_column_quality(self, series):
        n_values = len(series)
        n_missing = series.isnull().sum()
        n_unique = series.nunique()
        
        # Calculate basic statistics
        stats_dict = {
            'missing_ratio': n_missing / n_values,
            'unique_ratio': n_unique / n_values,
            'zeros_ratio': (series == 0).sum() / n_values,
            'negative_ratio': (series < 0).sum() / n_values if pd.api.types.is_numeric_dtype(series) else 0
        }
        
        # Check for potential constants
        is_constant = n_unique == 1
        
        # Check for potential IDs
        is_potential_id = n_unique / n_values > self.threshold_unique
        
        return {
            'statistics': stats_dict,
            'flags': {
                'high_missing': stats_dict['missing_ratio'] > self.threshold_missing,
                'constant': is_constant,
                'potential_id': is_potential_id
            }
        }
    
    def generate_report(self, df):
        self.report = {
            'global_stats': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'total_missing': df.isnull().sum().sum(),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
            },
            'column_analysis': {}
        }
        
        for column in df.columns:
            self.report['column_analysis'][column] = self.analyze_column_quality(df[column])
            
        return pd.DataFrame({
            col: {
                **analysis['statistics'],
                **analysis['flags']
            }
            for col, analysis in self.report['column_analysis'].items()
        }).T
    
    def suggest_improvements(self):
        suggestions = []
        
        for col, analysis in self.report['column_analysis'].items():
            if analysis['flags']['high_missing']:
                suggestions.append(f"Column '{col}' has high missing values - consider imputation")
            if analysis['flags']['constant']:
                suggestions.append(f"Column '{col}' is constant - consider removing")
            if analysis['flags']['potential_id']:
                suggestions.append(f"Column '{col}' might be an ID column - verify if needed")
                
        return suggestions

# Example usage
np.random.seed(42)
sample_data = pd.DataFrame({
    'id': range(1000),
    'value': np.random.normal(0, 1, 1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'constant': 'same_value',
    'missing_col': np.where(np.random.random(1000) > 0.8, np.nan, 1)
})

analyzer = DataQualityAnalyzer()
quality_report = analyzer.generate_report(sample_data)
suggestions = analyzer.suggest_improvements()

print("Data Quality Report:\n", quality_report)
print("\nSuggestions for Improvement:\n", '\n'.join(suggestions))
```

Slide 11: Automated Feature Selection with Statistical Testing

This implementation combines multiple feature selection techniques with statistical validation to identify the most relevant features while controlling for false discoveries.

```python
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

class StatisticalFeatureSelector:
    def __init__(self, alpha=0.05, method='fdr_bh'):
        self.alpha = alpha
        self.method = method
        self.feature_scores = {}
        
    def calculate_univariate_scores(self, X, y):
        scores = {}
        p_values = {}
        
        for column in X.columns:
            # Calculate multiple metrics
            mi_score = mutual_info_regression(
                X[[column]], y, random_state=42
            )[0]
            
            # Spearman correlation for non-linear relationships
            spearman_corr, spearman_p = spearmanr(X[column], y)
            
            scores[column] = {
                'mutual_info': mi_score,
                'spearman_corr': abs(spearman_corr),
                'spearman_p': spearman_p
            }
            
            p_values[column] = spearman_p
            
        # Correct for multiple testing
        rejected, corrected_p_values, _, _ = multipletests(
            list(p_values.values()),
            alpha=self.alpha,
            method=self.method
        )
        
        # Update scores with corrected p-values
        for (column, score), corrected_p in zip(scores.items(), corrected_p_values):
            score['corrected_p'] = corrected_p
            score['significant'] = corrected_p < self.alpha
            
        self.feature_scores = scores
        return scores
    
    def select_features(self, X, y, min_score=0.01):
        scores = self.calculate_univariate_scores(X, y)
        
        selected_features = [
            column for column, score in scores.items()
            if (score['mutual_info'] > min_score and 
                score['significant'])
        ]
        
        return {
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'feature_scores': pd.DataFrame(scores).T.sort_values(
                'mutual_info', ascending=False
            )
        }

# Example usage with breast cancer dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

selector = StatisticalFeatureSelector()
selection_results = selector.select_features(X, y)

print("Selected Features:", selection_results['n_selected'])
print("\nTop 5 Features by Mutual Information:\n",
      selection_results['feature_scores'].head())
```

Slide 12: Real-World Application: Credit Risk Assessment

This comprehensive example demonstrates a complete data preprocessing pipeline for credit risk assessment, incorporating multiple cleaning and transformation techniques while handling sensitive financial data.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CreditRiskPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoding_maps = {}
        self.feature_stats = {}
        
    def preprocess_financial_data(self, df):
        df = df.copy()
        
        # Handle negative values in monetary columns
        monetary_cols = ['income', 'debt', 'credit_limit']
        for col in monetary_cols:
            df[col] = df[col].abs()
            df[f'{col}_log'] = np.log1p(df[col])
        
        # Create debt-to-income ratio
        df['debt_to_income'] = df['debt'] / df['income'].clip(lower=1)
        
        # Handle categorical variables
        categorical_cols = ['employment_type', 'education']
        for col in categorical_cols:
            dummy_cols = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummy_cols], axis=1)
            df.drop(col, axis=1, inplace=True)
        
        # Create credit utilization feature
        df['credit_utilization'] = df['debt'] / df['credit_limit'].clip(lower=1)
        
        return df
    
    def handle_outliers(self, df, columns, method='iqr'):
        df = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
                
        return df
    
    def fit_transform(self, df, target_col):
        # Store original feature statistics
        self.feature_stats = {
            col: {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'missing_rate': df[col].isnull().mean()
            }
            for col in df.columns
        }
        
        # Preprocess financial features
        df_processed = self.preprocess_financial_data(df)
        
        # Handle outliers in numerical columns
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed = self.handle_outliers(df_processed, numerical_cols)
        
        # Scale numerical features
        self.scalers['standard'] = StandardScaler()
        numerical_cols = [col for col in numerical_cols if col != target_col]
        df_processed[numerical_cols] = self.scalers['standard'].fit_transform(
            df_processed[numerical_cols]
        )
        
        return df_processed
    
    def transform(self, df):
        df_processed = self.preprocess_financial_data(df)
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed = self.handle_outliers(df_processed, numerical_cols)
        df_processed[numerical_cols] = self.scalers['standard'].transform(
            df_processed[numerical_cols]
        )
        return df_processed

# Example usage with synthetic credit data
np.random.seed(42)
n_samples = 1000

synthetic_credit_data = pd.DataFrame({
    'income': np.random.lognormal(10, 1, n_samples),
    'debt': np.random.lognormal(8, 1.5, n_samples),
    'credit_limit': np.random.lognormal(9, 1, n_samples),
    'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed'], n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'default_risk': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
})

# Process the data
preprocessor = CreditRiskPreprocessor()
processed_data = preprocessor.fit_transform(synthetic_credit_data, 'default_risk')

print("Original Data Shape:", synthetic_credit_data.shape)
print("Processed Data Shape:", processed_data.shape)
print("\nFeature Statistics:\n", pd.DataFrame(preprocessor.feature_stats))
```

Slide 13: Results Analysis for Credit Risk Model

A detailed analysis of the preprocessing results, showing the impact of each transformation step and validating the quality of the processed data for the credit risk assessment model.

```python
class PreprocessingAnalyzer:
    def analyze_transformations(self, original_df, processed_df, target_col):
        analysis = {
            'feature_counts': {
                'original': len(original_df.columns),
                'processed': len(processed_df.columns)
            },
            'missing_values': {
                'original': original_df.isnull().sum().sum(),
                'processed': processed_df.isnull().sum().sum()
            },
            'correlation_changes': {}
        }
        
        # Analyze correlation changes
        numerical_cols_original = original_df.select_dtypes(include=[np.number]).columns
        numerical_cols_processed = processed_df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols_original:
            if col in processed_df.columns and col != target_col:
                original_corr = abs(original_df[col].corr(original_df[target_col]))
                processed_corr = abs(processed_df[col].corr(processed_df[target_col]))
                analysis['correlation_changes'][col] = {
                    'original_corr': original_corr,
                    'processed_corr': processed_corr,
                    'correlation_change': processed_corr - original_corr
                }
        
        return pd.DataFrame(analysis['correlation_changes']).T

# Analyze preprocessing results
analyzer = PreprocessingAnalyzer()
correlation_analysis = analyzer.analyze_transformations(
    synthetic_credit_data, 
    processed_data, 
    'default_risk'
)

print("Correlation Analysis:\n", correlation_analysis)

# Visualize key relationships
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.boxplot(x='default_risk', y='credit_utilization', data=processed_data)
plt.title('Credit Utilization by Default Risk')
plt.show()
```

Slide 14: Additional Resources

*   A Survey on Data Preprocessing for Data Mining [https://arxiv.org/abs/2006.01989](https://arxiv.org/abs/2006.01989)
*   Automated Feature Engineering: A Comprehensive Review [https://arxiv.org/abs/2106.15147](https://arxiv.org/abs/2106.15147)
*   Statistical Methods for Handling Missing Data in Large-Scale Machine Learning [https://arxiv.org/abs/2012.08278](https://arxiv.org/abs/2012.08278)
*   Deep Learning Approaches for Data Cleaning and Preprocessing [https://arxiv.org/abs/2102.06409](https://arxiv.org/abs/2102.06409)
*   A Comprehensive Survey on Data Quality Assessment and Validation [https://arxiv.org/abs/2109.14365](https://arxiv.org/abs/2109.14365)

