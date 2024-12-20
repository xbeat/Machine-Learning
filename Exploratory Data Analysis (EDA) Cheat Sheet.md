## Exploratory Data Analysis (EDA) Cheat Sheet
Slide 1: Data Loading and Initial Inspection

Understanding your dataset begins with proper loading and initial inspection. This involves reading data from various formats, checking basic properties like shape, data types, and missing values. Pandas provides comprehensive tools for these fundamental EDA tasks.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (using a sample financial dataset)
df = pd.read_csv('financial_data.csv')

# Initial inspection
print("Dataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# Display first few rows and basic statistics
print("\nFirst 5 rows:\n", df.head())
print("\nBasic Statistics:\n", df.describe())
```

Slide 2: Data Cleaning and Preprocessing

Data cleaning involves handling missing values, removing duplicates, and correcting data types. This step ensures data quality and prevents issues in subsequent analysis. Advanced techniques include outlier detection using statistical methods.

```python
# Handle missing values
df_clean = df.copy()
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

# Fill numeric missing values with median
for col in numeric_cols:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

# Remove duplicates
df_clean.drop_duplicates(inplace=True)

# Detect outliers using IQR method
def detect_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# Example for a numeric column
outliers = detect_outliers(df_clean, 'price')
print("Number of outliers:", len(outliers))
```

Slide 3: Univariate Analysis - Numerical Variables

Univariate analysis examines the distribution and statistical properties of individual numerical variables. This includes calculating central tendency measures, dispersion metrics, and visualizing distributions through histograms and box plots.

```python
def analyze_numerical_variable(df, column):
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=column, kde=True)
    plt.title(f'Distribution of {column}')
    
    # Box plot
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, y=column)
    plt.title(f'Box Plot of {column}')
    
    # Statistical metrics
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'skewness': df[column].skew(),
        'kurtosis': df[column].kurtosis()
    }
    print(f"\nStatistics for {column}:")
    for metric, value in stats.items():
        print(f"{metric}: {value:.2f}")
```

Slide 4: Univariate Analysis - Categorical Variables

Analyzing categorical variables requires different approaches than numerical ones. We examine frequency distributions, unique values, and create visualizations that highlight the distribution of categories within the data.

```python
def analyze_categorical_variable(df, column):
    # Calculate value counts and proportions
    value_counts = df[column].value_counts()
    proportions = df[column].value_counts(normalize=True)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Bar plot
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x=column)
    plt.xticks(rotation=45)
    plt.title(f'Count Distribution of {column}')
    
    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(proportions, labels=proportions.index, autopct='%1.1f%%')
    plt.title(f'Percentage Distribution of {column}')
    
    # Print statistics
    print(f"\nValue counts for {column}:")
    print(value_counts)
    print(f"\nCardinality (unique values): {df[column].nunique()}")
    print(f"Mode: {df[column].mode()[0]}")
```

Slide 5: Bivariate Analysis - Numerical Variables

Bivariate analysis explores relationships between pairs of numerical variables. Understanding these relationships helps identify correlations, patterns, and potential feature interactions that could be significant for modeling purposes.

```python
def analyze_numerical_pairs(df, cols=None):
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns
    
    # Correlation matrix
    corr_matrix = df[cols].corr()
    
    # Create visualizations
    plt.figure(figsize=(15, 6))
    
    # Scatter plot with regression line
    plt.subplot(1, 2, 1)
    sns.regplot(data=df, x=cols[0], y=cols[1])
    plt.title(f'Scatter Plot: {cols[0]} vs {cols[1]}')
    
    # Correlation heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    
    # Calculate additional metrics
    print(f"\nPearson Correlation: {df[cols[0]].corr(df[cols[1]]):.3f}")
    print(f"Spearman Correlation: {df[cols[0]].corr(df[cols[1]], method='spearman'):.3f}")
```

Slide 6: Bivariate Analysis - Mixed Variable Types

Analyzing relationships between numerical and categorical variables requires specific visualization techniques and statistical tests. This analysis helps understand how categories affect numerical distributions.

```python
def analyze_numerical_categorical(df, numerical_col, categorical_col):
    plt.figure(figsize=(15, 6))
    
    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x=categorical_col, y=numerical_col)
    plt.xticks(rotation=45)
    plt.title(f'{numerical_col} Distribution by {categorical_col}')
    
    # Violin plot
    plt.subplot(1, 2, 2)
    sns.violinplot(data=df, x=categorical_col, y=numerical_col)
    plt.xticks(rotation=45)
    plt.title(f'Violin Plot of {numerical_col} by {categorical_col}')
    
    # Statistical analysis
    from scipy import stats
    groups = [group for _, group in df.groupby(categorical_col)[numerical_col]]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"\nOne-way ANOVA results:")
    print(f"F-statistic: {f_stat:.3f}")
    print(f"p-value: {p_value:.3f}")
```

Slide 7: Time Series Component Analysis

Time series data requires specialized analysis techniques to understand trends, seasonality, and cyclical patterns. This analysis helps identify temporal patterns and make informed decisions about forecasting approaches.

```python
def analyze_time_series(df, date_col, value_col):
    # Convert to datetime if needed
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    
    # Decomposition
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(df[value_col], period=12)
    
    plt.figure(figsize=(12, 10))
    
    # Plot components
    plt.subplot(4, 1, 1)
    plt.plot(df[value_col])
    plt.title('Original Time Series')
    
    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend)
    plt.title('Trend')
    
    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal)
    plt.title('Seasonal')
    
    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid)
    plt.title('Residuals')
    
    plt.tight_layout()
```

Slide 8: Feature Engineering and Transformation

Feature engineering involves creating new variables and transforming existing ones to better capture underlying patterns. This process includes handling skewed distributions, creating interaction terms, and encoding categorical variables.

```python
def engineer_features(df):
    # Create copy to avoid modifying original data
    df_transformed = df.copy()
    
    # Log transform for skewed numerical variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].skew() > 1:
            df_transformed[f'{col}_log'] = np.log1p(df[col])
    
    # Polynomial features for selected columns
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, include_bias=False)
    
    # One-hot encoding for categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_transformed = pd.get_dummies(df_transformed, columns=categorical_cols)
    
    # Z-score normalization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])
    
    return df_transformed
```

Slide 9: Multivariate Analysis and Dimensionality Reduction

Multivariate analysis examines relationships between multiple variables simultaneously. Principal Component Analysis (PCA) helps reduce dimensionality while preserving important patterns in the data structure.

```python
def perform_multivariate_analysis(df, numeric_cols):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols])
    
    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Analysis')
    
    # Print component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(numeric_cols))],
        index=numeric_cols
    )
    print("\nPrincipal Component Loadings:")
    print(loadings.head())
    
    return pca_result, loadings
```

Slide 10: Advanced Statistical Tests

Statistical tests help validate hypotheses about relationships in the data. This comprehensive suite includes tests for normality, independence, and distribution comparisons across different groups.

```python
def perform_statistical_tests(df, numeric_col, group_col=None):
    from scipy import stats
    
    # Normality test
    stat, p_value = stats.normaltest(df[numeric_col])
    print(f"Normality Test (D'Agostino-Pearson):")
    print(f"Statistic: {stat:.3f}, p-value: {p_value:.3f}")
    
    if group_col is not None:
        # Two-sample tests
        groups = [group for _, group in df.groupby(group_col)[numeric_col]]
        
        # Mann-Whitney U test
        stat, p_value = stats.mannwhitneyu(*groups)
        print(f"\nMann-Whitney U Test:")
        print(f"Statistic: {stat:.3f}, p-value: {p_value:.3f}")
        
        # Kolmogorov-Smirnov test
        stat, p_value = stats.ks_2samp(*groups)
        print(f"\nKolmogorov-Smirnov Test:")
        print(f"Statistic: {stat:.3f}, p-value: {p_value:.3f}")
    
    # Autocorrelation test
    acf = stats.acf(df[numeric_col])
    print(f"\nAutocorrelation (lag-1): {acf[1]:.3f}")
```

Slide 11: Anomaly Detection

Advanced anomaly detection techniques help identify unusual patterns and outliers in multivariate data. This implementation includes statistical and machine learning-based approaches for robust outlier detection.

```python
def detect_anomalies(df, numeric_cols):
    from sklearn.ensemble import IsolationForest
    from sklearn.covariance import EllipticEnvelope
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols])
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest_labels = iso_forest.fit_predict(scaled_data)
    
    # Robust covariance estimation
    robust_cov = EllipticEnvelope(contamination=0.1, random_state=42)
    robust_cov_labels = robust_cov.fit_predict(scaled_data)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(scaled_data[:, 0], scaled_data[:, 1], 
               c=iso_forest_labels, cmap='viridis')
    plt.title('Isolation Forest Anomalies')
    
    plt.subplot(1, 2, 2)
    plt.scatter(scaled_data[:, 0], scaled_data[:, 1], 
               c=robust_cov_labels, cmap='viridis')
    plt.title('Robust Covariance Anomalies')
    
    return iso_forest_labels, robust_cov_labels
```

Slide 12: Real-World Example - Financial Data Analysis

A comprehensive example analyzing financial market data, demonstrating the application of various EDA techniques to understand market trends and relationships between different financial instruments.

```python
def analyze_financial_data(stock_data):
    # Assuming stock_data has columns: Date, Open, High, Low, Close, Volume
    
    # Calculate daily returns
    stock_data['Returns'] = stock_data['Close'].pct_change()
    
    # Calculate volatility (20-day rolling standard deviation)
    stock_data['Volatility'] = stock_data['Returns'].rolling(window=20).std()
    
    # Trading volume analysis
    stock_data['Volume_MA'] = stock_data['Volume'].rolling(window=20).mean()
    
    # Plot technical indicators
    plt.figure(figsize=(15, 10))
    
    # Price and Volume
    plt.subplot(3, 1, 1)
    plt.plot(stock_data.index, stock_data['Close'])
    plt.title('Stock Price')
    
    plt.subplot(3, 1, 2)
    plt.plot(stock_data.index, stock_data['Volatility'])
    plt.title('20-Day Volatility')
    
    plt.subplot(3, 1, 3)
    plt.bar(stock_data.index, stock_data['Volume'])
    plt.plot(stock_data.index, stock_data['Volume_MA'], color='red')
    plt.title('Trading Volume')
    
    # Calculate key statistics
    print("\nKey Statistics:")
    print(f"Average Daily Return: {stock_data['Returns'].mean():.4f}")
    print(f"Annual Volatility: {stock_data['Returns'].std() * np.sqrt(252):.4f}")
    print(f"Sharpe Ratio: {stock_data['Returns'].mean() / stock_data['Returns'].std():.4f}")
```

Slide 13: Real-World Example - Customer Segmentation Analysis

This example demonstrates comprehensive customer segmentation analysis using RFM (Recency, Frequency, Monetary) metrics and clustering techniques to identify distinct customer groups.

```python
def perform_customer_segmentation(transaction_data):
    # Calculate RFM metrics
    today_date = transaction_data['date'].max()
    
    rfm = transaction_data.groupby('customer_id').agg({
        'date': lambda x: (today_date - x.max()).days,  # Recency
        'transaction_id': 'count',                      # Frequency
        'amount': 'sum'                                 # Monetary
    }).rename(columns={
        'date': 'recency',
        'transaction_id': 'frequency',
        'amount': 'monetary'
    })
    
    # Scale the features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    # Perform K-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Segment'] = kmeans.fit_predict(rfm_scaled)
    
    # Visualize segments
    plt.figure(figsize=(15, 5))
    
    # 3D scatter plot
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(121, projection='3d')
    scatter = ax.scatter(rfm['recency'], 
                        rfm['frequency'], 
                        rfm['monetary'],
                        c=rfm['Segment'],
                        cmap='viridis')
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
    plt.colorbar(scatter)
    
    # Segment characteristics
    print("\nSegment Profiles:")
    print(rfm.groupby('Segment').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': ['mean', 'count']
    }).round(2))
    
    return rfm
```

Slide 14: Advanced Visualization Techniques

Complex data relationships often require sophisticated visualization techniques. This implementation showcases advanced plotting methods for multidimensional data analysis and pattern recognition.

```python
def create_advanced_visualizations(df, numeric_cols, categorical_col=None):
    plt.figure(figsize=(15, 10))
    
    # Parallel Coordinates Plot
    from pandas.plotting import parallel_coordinates
    plt.subplot(2, 2, 1)
    normalized_df = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    if categorical_col:
        parallel_coordinates(normalized_df.join(df[categorical_col]), categorical_col)
    plt.title('Parallel Coordinates Plot')
    
    # Andrews Curves
    from pandas.plotting import andrews_curves
    plt.subplot(2, 2, 2)
    if categorical_col:
        andrews_curves(df, categorical_col)
    plt.title('Andrews Curves')
    
    # Radar Chart
    angles = np.linspace(0, 2*np.pi, len(numeric_cols), endpoint=False)
    stats = df[numeric_cols].mean()
    stats = np.concatenate((stats, [stats[0]]))  # complete the circle
    angles = np.concatenate((angles, [angles[0]]))
    
    ax = plt.subplot(2, 2, 3, projection='polar')
    ax.plot(angles, stats)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(numeric_cols)
    plt.title('Radar Chart of Mean Values')
    
    # Joint Plot
    import seaborn as sns
    plt.subplot(2, 2, 4)
    if len(numeric_cols) >= 2:
        sns.jointplot(data=df, x=numeric_cols[0], y=numeric_cols[1], 
                     kind='hex', height=8)
        plt.title('Joint Distribution Plot')
    
    plt.tight_layout()
```

Slide 15: Additional Resources

1.  ArXiv Papers for Advanced EDA Techniques:

*   [https://arxiv.org/abs/2010.09981](https://arxiv.org/abs/2010.09981) - "A Survey on Automated Data Quality Assessment Methods"
*   [https://arxiv.org/abs/1904.02101](https://arxiv.org/abs/1904.02101) - "Automated Machine Learning: Methods, Systems, Challenges"
*   [https://arxiv.org/abs/1803.02352](https://arxiv.org/abs/1803.02352) - "Visualization Techniques for Time Series Data Analysis"
*   [https://arxiv.org/abs/1906.08158](https://arxiv.org/abs/1906.08158) - "Machine Learning Interpretability: A Survey on Methods and Metrics"
*   [https://arxiv.org/abs/2003.02912](https://arxiv.org/abs/2003.02912) - "Interactive Visual Analytics and Visualization for Decision Making"

