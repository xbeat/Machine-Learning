## Ultimate EDA CheatCodes
Slide 1: Data Loading and Initial Exploration

The foundation of any EDA begins with efficiently loading data and performing initial checks. This involves reading various file formats, checking basic statistics, and understanding the dataset's structure using pandas and numpy libraries.

```python
import pandas as pd
import numpy as np
from datetime import datetime

# Load dataset with automatic date parsing
df = pd.read_csv('sales_data.csv', parse_dates=['transaction_date'])

# Display comprehensive dataset information
print("Dataset Info:")
print(df.info())

# Generate descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe(include='all'))

# Check missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

# Display first few rows with custom formatting
pd.set_option('display.max_columns', None)
print("\nSample Data:")
print(df.head())
```

Slide 2: Advanced Data Type Analysis

Understanding data types and their distributions is crucial for proper preprocessing. This technique identifies potential data quality issues and helps determine appropriate transformation strategies for different variable types.

```python
def analyze_data_types(df):
    # Dictionary to store analysis results
    analysis = {
        'numerical_cols': [],
        'categorical_cols': [],
        'datetime_cols': [],
        'unique_counts': {},
        'memory_usage': {}
    }
    
    for column in df.columns:
        # Determine data type category
        if pd.api.types.is_numeric_dtype(df[column]):
            analysis['numerical_cols'].append(column)
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            analysis['datetime_cols'].append(column)
        else:
            analysis['categorical_cols'].append(column)
        
        # Count unique values
        analysis['unique_counts'][column] = df[column].nunique()
        
        # Calculate memory usage
        analysis['memory_usage'][column] = df[column].memory_usage(deep=True) / 1024**2  # MB
    
    return analysis

# Example usage
data_analysis = analyze_data_types(df)
for category, columns in data_analysis.items():
    if isinstance(columns, list):
        print(f"\n{category.upper()}:")
        for col in columns:
            print(f"{col}: {data_analysis['unique_counts'][col]} unique values, "
                  f"{data_analysis['memory_usage'][col]:.2f} MB")
```

Slide 3: Statistical Distribution Analysis

A comprehensive analysis of variable distributions helps identify patterns, skewness, and potential outliers. This implementation combines visual and statistical methods to provide insights into data characteristics.

```python
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_distributions(df, numerical_cols):
    plt.style.use('seaborn')
    
    for col in numerical_cols:
        # Calculate statistical measures
        skewness = stats.skew(df[col].dropna())
        kurtosis = stats.kurtosis(df[col].dropna())
        
        # Create distribution plot
        plt.figure(figsize=(12, 6))
        
        # Histogram with KDE
        sns.histplot(data=df, x=col, kde=True)
        
        # Add Q-Q plot as subplot
        stats.probplot(df[col].dropna(), dist="norm", plot=plt)
        
        plt.title(f'Distribution Analysis: {col}\n'
                 f'Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}')
        plt.tight_layout()
        plt.show()
        
        # Print statistical tests
        normality_test = stats.normaltest(df[col].dropna())
        print(f"\nNormality Test for {col}:")
        print(f"p-value: {normality_test.pvalue:.4f}")

# Example usage
numerical_columns = df.select_dtypes(include=[np.number]).columns
analyze_distributions(df, numerical_columns)
```

Slide 4: Advanced Outlier Detection

This implementation combines multiple outlier detection methods including statistical, distance-based, and machine learning approaches to provide a robust identification of anomalous data points.

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_outliers(df, columns, methods=['zscore', 'iqr', 'isolation_forest']):
    outliers_dict = {}
    
    for col in columns:
        outliers_dict[col] = {}
        data = df[col].dropna()
        
        if 'zscore' in methods:
            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            outliers_dict[col]['zscore'] = data[z_scores > 3]
        
        if 'iqr' in methods:
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers_dict[col]['iqr'] = data[(data < (Q1 - 1.5 * IQR)) | 
                                           (data > (Q3 + 1.5 * IQR))]
        
        if 'isolation_forest' in methods:
            # Isolation Forest method
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            yhat = iso_forest.fit_predict(data.values.reshape(-1, 1))
            outliers_dict[col]['isolation_forest'] = data[yhat == -1]
    
    return outliers_dict

# Example usage
numerical_cols = df.select_dtypes(include=[np.number]).columns
outliers = detect_outliers(df, numerical_cols)

# Print summary
for col in outliers:
    print(f"\nOutliers in {col}:")
    for method, values in outliers[col].items():
        print(f"{method}: {len(values)} outliers detected")
```

Slide 5: Correlation Analysis and Feature Relationships

Advanced correlation analysis extends beyond simple Pearson correlations to include non-linear relationships and mutual information metrics, providing deeper insights into feature dependencies and potential predictive power.

```python
import scipy.stats as stats
from sklearn.metrics import mutual_info_score

def advanced_correlation_analysis(df, numerical_cols):
    # Initialize correlation matrices
    n = len(numerical_cols)
    pearson_corr = np.zeros((n, n))
    spearman_corr = np.zeros((n, n))
    mutual_info = np.zeros((n, n))
    
    for i, col1 in enumerate(numerical_cols):
        for j, col2 in enumerate(numerical_cols):
            # Calculate different correlation metrics
            pearson_corr[i,j] = stats.pearsonr(df[col1], df[col2])[0]
            spearman_corr[i,j] = stats.spearmanr(df[col1], df[col2])[0]
            
            # Normalize and bin data for mutual information
            x_norm = pd.qcut(df[col1], q=10, labels=False, duplicates='drop')
            y_norm = pd.qcut(df[col2], q=10, labels=False, duplicates='drop')
            mutual_info[i,j] = mutual_info_score(x_norm, y_norm)
    
    # Create heatmap visualizations
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    sns.heatmap(pearson_corr, annot=True, cmap='RdBu', center=0, 
                xticklabels=numerical_cols, yticklabels=numerical_cols, ax=axes[0])
    axes[0].set_title('Pearson Correlation')
    
    sns.heatmap(spearman_corr, annot=True, cmap='RdBu', center=0,
                xticklabels=numerical_cols, yticklabels=numerical_cols, ax=axes[1])
    axes[1].set_title('Spearman Correlation')
    
    sns.heatmap(mutual_info, annot=True, cmap='viridis',
                xticklabels=numerical_cols, yticklabels=numerical_cols, ax=axes[2])
    axes[2].set_title('Mutual Information')
    
    plt.tight_layout()
    plt.show()

# Example usage
numerical_cols = df.select_dtypes(include=[np.number]).columns
advanced_correlation_analysis(df, numerical_cols)
```

Slide 6: Time Series Decomposition and Analysis

Time series decomposition is crucial for understanding temporal patterns in data. This implementation breaks down time series into trend, seasonal, and residual components while providing statistical measures of each component.

```python
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

def analyze_time_series(df, date_column, value_column, freq='D'):
    # Set datetime index
    df_ts = df.set_index(date_column)
    
    # Perform decomposition
    decomposition = seasonal_decompose(df_ts[value_column], 
                                     period=30 if freq=='D' else 12,
                                     extrapolate_trend='freq')
    
    # Plot components
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
    
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Original Time Series')
    
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    
    plt.tight_layout()
    
    # Perform additional statistical tests
    # Augmented Dickey-Fuller test for stationarity
    adf_test = sm.tsa.stattools.adfuller(df_ts[value_column].dropna())
    
    # Calculate autocorrelation
    acf = sm.tsa.stattools.acf(df_ts[value_column].dropna(), nlags=40)
    
    print(f"ADF Test Statistic: {adf_test[0]}")
    print(f"p-value: {adf_test[1]}")
    print("\nAutocorrelation (first 5 lags):")
    print(acf[:5])

# Example usage
analyze_time_series(df, 'date_column', 'value_column', freq='D')
```

Slide 7: Dimensionality Reduction and Feature Importance

This advanced implementation combines multiple dimensionality reduction techniques with feature importance analysis to provide comprehensive insights into data structure and variable relationships.

```python
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from umap import UMAP

def analyze_feature_importance(df, target_col=None):
    # Prepare data
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_col] if target_col else [])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)
    
    # Calculate explained variance
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Analysis')
    plt.show()
    
    # Feature importance using Random Forest
    if target_col is not None:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, df[target_col])
        
        # Plot feature importance
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        importances.sort_values(ascending=True).plot(kind='barh')
        plt.title('Feature Importance')
        plt.show()
    
    return {
        'pca_components': pca.components_,
        'explained_variance': explained_variance,
        'feature_importance': rf.feature_importances_ if target_col else None
    }

# Example usage
results = analyze_feature_importance(df, target_col='target_variable')
```

Slide 8: Pattern Mining and Association Rules

Advanced pattern mining techniques help discover hidden relationships and frequent patterns in categorical data. This implementation includes support for both categorical and discretized numerical variables.

```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def mine_patterns(df, categorical_cols, min_support=0.01, min_confidence=0.5):
    # Prepare transactions
    transactions = df[categorical_cols].values.tolist()
    
    # Transform data
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Find frequent itemsets
    frequent_itemsets = apriori(df_encoded, 
                               min_support=min_support, 
                               use_colnames=True)
    
    # Generate rules
    rules = association_rules(frequent_itemsets, 
                            metric="confidence", 
                            min_threshold=min_confidence)
    
    # Calculate additional metrics
    rules['lift'] = rules['lift'].round(4)
    rules['conviction'] = np.where(rules['confidence'] == 1, np.inf,
                                 (1 - rules['consequent support']) / 
                                 (1 - rules['confidence']))
    
    # Sort rules by lift ratio
    rules = rules.sort_values('lift', ascending=False)
    
    print("Top 10 Association Rules:")
    print(rules.head(10)[['antecedents', 'consequents', 
                         'support', 'confidence', 'lift']])
    
    return rules, frequent_itemsets

# Example usage
categorical_columns = ['category1', 'category2', 'category3']
rules, itemsets = mine_patterns(df, categorical_columns)
```

Slide 9: Multivariate Anomaly Detection

This implementation combines multiple statistical and machine learning approaches to detect multivariate anomalies in high-dimensional datasets using robust estimation techniques.

```python
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

def detect_multivariate_anomalies(df, numerical_cols):
    # Prepare data
    X = df[numerical_cols]
    X_scaled = StandardScaler().fit_transform(X)
    
    # Initialize detectors
    detectors = {
        'Robust Covariance': EllipticEnvelope(contamination=0.1, 
                                             random_state=42),
        'One-Class SVM': OneClassSVM(kernel='rbf', nu=0.1),
        'Local Outlier Factor': LocalOutlierFactor(n_neighbors=20, 
                                                  contamination=0.1)
    }
    
    results = {}
    for name, detector in detectors.items():
        # Fit and predict
        if name == 'Local Outlier Factor':
            y_pred = detector.fit_predict(X_scaled)
        else:
            y_pred = detector.fit(X_scaled).predict(X_scaled)
        
        # Store results
        results[name] = y_pred == -1  # True for outliers
        
    # Compare results
    agreement_matrix = np.zeros((len(detectors), len(detectors)))
    for i, (name1, res1) in enumerate(results.items()):
        for j, (name2, res2) in enumerate(results.items()):
            agreement = np.mean(res1 == res2)
            agreement_matrix[i, j] = agreement
    
    # Visualize agreement
    plt.figure(figsize=(10, 8))
    sns.heatmap(agreement_matrix, 
                annot=True, 
                xticklabels=detectors.keys(),
                yticklabels=detectors.keys())
    plt.title('Detector Agreement Matrix')
    plt.show()
    
    return results

# Example usage
numerical_cols = df.select_dtypes(include=[np.number]).columns
anomaly_results = detect_multivariate_anomalies(df, numerical_cols)
```

Slide 10: Dynamic Feature Engineering

Advanced feature engineering pipeline that automatically generates and evaluates new features based on domain knowledge and statistical properties of the data.

```python
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures

class DynamicFeatureEngineer:
    def __init__(self, df, target_col=None):
        self.df = df
        self.target_col = target_col
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = df.select_dtypes(include=['object']).columns
        
    def generate_statistical_features(self):
        stats_features = pd.DataFrame()
        
        # Rolling statistics
        for col in self.numerical_cols:
            if col != self.target_col:
                stats_features[f'{col}_rolling_mean'] = self.df[col].rolling(window=3).mean()
                stats_features[f'{col}_rolling_std'] = self.df[col].rolling(window=3).std()
                
        return stats_features
    
    def generate_polynomial_features(self, degree=2):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(self.df[self.numerical_cols])
        
        feature_names = poly.get_feature_names_out(self.numerical_cols)
        return pd.DataFrame(poly_features, columns=feature_names)
    
    def evaluate_features(self, features_df):
        if self.target_col is None:
            return None
            
        mi_scores = mutual_info_regression(features_df, self.df[self.target_col])
        feature_importance = pd.DataFrame({
            'feature': features_df.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def fit_transform(self):
        # Generate features
        statistical_features = self.generate_statistical_features()
        polynomial_features = self.generate_polynomial_features()
        
        # Combine features
        all_features = pd.concat([
            statistical_features,
            polynomial_features
        ], axis=1)
        
        # Evaluate features
        importance = self.evaluate_features(all_features)
        
        if importance is not None:
            print("Top 10 Generated Features:")
            print(importance.head(10))
        
        return all_features

# Example usage
engineer = DynamicFeatureEngineer(df, target_col='target')
new_features = engineer.fit_transform()
```

Slide 11: Temporal Pattern Analysis

Advanced time-based pattern detection that identifies cyclical patterns, seasonality effects, and temporal anomalies using spectral analysis and wavelet transformations.

```python
import pywt
from scipy import signal
from datetime import timedelta

def analyze_temporal_patterns(df, date_column, value_column):
    # Prepare time series data
    ts = df.set_index(date_column)[value_column]
    
    # Spectral Analysis
    frequencies, power_spectrum = signal.welch(ts.dropna(), 
                                             fs=1.0, 
                                             window='hanning',
                                             nperseg=len(ts)//10)
    
    # Wavelet Analysis
    wavelet = 'morl'
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(ts.dropna(), scales, wavelet)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Original Time Series
    ax1.plot(ts.index, ts.values)
    ax1.set_title('Original Time Series')
    
    # Power Spectrum
    ax2.plot(frequencies, power_spectrum)
    ax2.set_title('Power Spectrum')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Power')
    
    # Wavelet Transform
    im = ax3.imshow(abs(coefficients), aspect='auto', cmap='viridis')
    ax3.set_title('Wavelet Transform')
    ax3.set_ylabel('Scale')
    plt.colorbar(im, ax=ax3)
    
    plt.tight_layout()
    
    # Find dominant periods
    peak_frequencies = frequencies[signal.find_peaks(power_spectrum)[0]]
    dominant_periods = 1/peak_frequencies
    
    print("\nDominant Periods Detected:")
    for period in dominant_periods:
        print(f"Period: {timedelta(days=period)}")
    
    return {
        'power_spectrum': (frequencies, power_spectrum),
        'wavelet_coefficients': coefficients,
        'dominant_periods': dominant_periods
    }

# Example usage
temporal_analysis = analyze_temporal_patterns(df, 'date_column', 'value_column')
```

Slide 12: Multi-dimensional Clustering Analysis

Implementation of advanced clustering techniques that automatically determine optimal cluster numbers and handle high-dimensional data with sophisticated validation metrics.

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

class AdvancedClustering:
    def __init__(self, data):
        self.data = StandardScaler().fit_transform(data)
        self.n_samples = data.shape[0]
        
    def determine_optimal_clusters(self, max_clusters=10):
        scores = {
            'silhouette': [],
            'calinski': []
        }
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(self.data)
            
            scores['silhouette'].append(silhouette_score(self.data, labels))
            scores['calinski'].append(calinski_harabasz_score(self.data, labels))
        
        return scores
    
    def perform_hierarchical_clustering(self):
        # Generate linkage matrix
        linkage_matrix = linkage(self.data, method='ward')
        
        # Plot dendrogram
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()
        
        return linkage_matrix
    
    def perform_density_clustering(self):
        # Estimate epsilon using nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=2).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        
        # Plot k-distance graph
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.title('K-distance Graph')
        plt.xlabel('Points')
        plt.ylabel('Distance')
        plt.show()
        
        # Perform DBSCAN with estimated parameters
        epsilon = np.percentile(distances, 90)
        min_samples = int(np.log(self.n_samples))
        
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        labels = dbscan.fit_predict(self.data)
        
        return labels, epsilon, min_samples

# Example usage
clustering = AdvancedClustering(df[numerical_cols])
optimal_scores = clustering.determine_optimal_clusters()
linkage_matrix = clustering.perform_hierarchical_clustering()
dbscan_labels, eps, min_samples = clustering.perform_density_clustering()
```

Slide 13: Interactive Data Profiling

This implementation provides a comprehensive data profiling system that automatically generates statistical summaries, identifies potential data quality issues, and performs distribution analysis for all variable types.

```python
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataProfiler:
    def __init__(self, df):
        self.df = df
        self.profile = {}
        
    def analyze_column(self, column):
        data = self.df[column]
        column_type = data.dtype
        
        analysis = {
            'type': str(column_type),
            'missing': data.isnull().sum(),
            'missing_pct': (data.isnull().sum() / len(data)) * 100,
            'unique_values': data.nunique(),
            'memory_usage': data.memory_usage(deep=True) / 1024**2  # MB
        }
        
        if np.issubdtype(column_type, np.number):
            analysis.update({
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'skew': stats.skew(data.dropna()),
                'kurtosis': stats.kurtosis(data.dropna()),
                'outliers_zscore': len(data[np.abs(stats.zscore(data.dropna())) > 3])
            })
            
        elif column_type == 'object' or column_type == 'category':
            value_counts = data.value_counts()
            analysis.update({
                'mode': data.mode()[0],
                'entropy': stats.entropy(value_counts.values),
                'top_values': value_counts.head().to_dict()
            })
            
        elif np.issubdtype(column_type, np.datetime64):
            analysis.update({
                'min_date': data.min(),
                'max_date': data.max(),
                'date_range': (data.max() - data.min()).days
            })
            
        return analysis
    
    def generate_profile(self):
        for column in self.df.columns:
            self.profile[column] = self.analyze_column(column)
            
        # Generate correlation matrix for numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            self.profile['correlations'] = self.df[numerical_cols].corr()
            
        return self.profile
    
    def print_summary(self):
        print("\nData Profile Summary:")
        print(f"Total Rows: {len(self.df)}")
        print(f"Total Columns: {len(self.df.columns)}")
        print("\nColumn Details:")
        
        for column, analysis in self.profile.items():
            if column != 'correlations':
                print(f"\n{column}:")
                for metric, value in analysis.items():
                    print(f"  {metric}: {value}")

# Example usage
profiler = DataProfiler(df)
profile = profiler.generate_profile()
profiler.print_summary()
```

Slide 14: Automated Insight Generation

This sophisticated implementation automatically discovers and ranks interesting patterns, relationships, and anomalies in the dataset using statistical tests and machine learning techniques.

```python
class InsightGenerator:
    def __init__(self, df):
        self.df = df
        self.insights = []
        
    def test_correlation_significance(self, col1, col2):
        corr, p_value = stats.pearsonr(self.df[col1], self.df[col2])
        return corr, p_value
    
    def find_distribution_changes(self, column, window_size=30):
        rolling_mean = self.df[column].rolling(window=window_size).mean()
        rolling_std = self.df[column].rolling(window=window_size).std()
        
        # Detect significant changes using Kolmogorov-Smirnov test
        changes = []
        for i in range(window_size, len(self.df), window_size):
            window1 = self.df[column].iloc[i-window_size:i]
            window2 = self.df[column].iloc[i:i+window_size]
            if len(window2) >= window_size:
                ks_stat, p_value = stats.ks_2samp(window1, window2)
                if p_value < 0.05:
                    changes.append(i)
        
        return changes
    
    def discover_insights(self):
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Correlation insights
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                corr, p_value = self.test_correlation_significance(col1, col2)
                if abs(corr) > 0.7 and p_value < 0.05:
                    self.insights.append({
                        'type': 'correlation',
                        'description': f'Strong correlation ({corr:.2f}) between {col1} and {col2}',
                        'strength': abs(corr)
                    })
        
        # Distribution changes
        for col in numerical_cols:
            changes = self.find_distribution_changes(col)
            if changes:
                self.insights.append({
                    'type': 'distribution_change',
                    'description': f'Distribution changes detected in {col} at positions: {changes}',
                    'strength': len(changes)
                })
        
        # Sort insights by strength
        self.insights.sort(key=lambda x: x['strength'], reverse=True)
        return self.insights
    
    def print_insights(self, top_n=10):
        print(f"\nTop {top_n} Insights Discovered:")
        for i, insight in enumerate(self.insights[:top_n], 1):
            print(f"\n{i}. {insight['description']}")
            print(f"   Type: {insight['type']}")
            print(f"   Strength: {insight['strength']:.2f}")

# Example usage
insight_generator = InsightGenerator(df)
insights = insight_generator.discover_insights()
insight_generator.print_insights()
```

Slide 15: Additional Resources

*   ArXiv paper: "A Survey of Automated Data Quality Assessment Methods" - [https://arxiv.org/abs/2108.02821](https://arxiv.org/abs/2108.02821)
*   "Modern Statistical Methods for Exploratory Data Analysis" - [https://arxiv.org/abs/2103.05155](https://arxiv.org/abs/2103.05155)
*   "Automated Feature Engineering: A Comprehensive Review" - [https://arxiv.org/abs/2106.15147](https://arxiv.org/abs/2106.15147)
*   "Deep Learning Approaches for Automated EDA" - [https://arxiv.org/abs/2109.01598](https://arxiv.org/abs/2109.01598)
*   Search suggestions:
    *   Google Scholar: "automated exploratory data analysis techniques"
    *   Research Gate: "modern statistical methods in EDA"
    *   ACM Digital Library: "machine learning for data profiling"

