## Importance of High-Quality Data for Powerful Neural Networks
Slide 1: Data Quality Assessment

Neural networks require high-quality training data to perform effectively. This implementation demonstrates how to assess dataset quality by checking for missing values, outliers, and basic statistical properties using pandas and numpy for a comprehensive data health check.

```python
import pandas as pd
import numpy as np

def assess_data_quality(dataset_path):
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Basic quality metrics
    quality_report = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'outliers_by_column': {}
    }
    
    # Detect outliers using IQR method
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = len(df[(df[column] < (Q1 - 1.5 * IQR)) | 
                         (df[column] > (Q3 + 1.5 * IQR))])
        quality_report['outliers_by_column'][column] = outliers
    
    return quality_report

# Example usage
sample_data = pd.DataFrame({
    'feature1': [1, 2, 3, 100, None, 4],
    'feature2': [10, 20, 30, 40, 50, 60]
})
sample_data.to_csv('sample_data.csv', index=False)
print(assess_data_quality('sample_data.csv'))
```

Slide 2: Data Preprocessing Pipeline

A robust preprocessing pipeline ensures data consistency and prepares it for neural network training. This implementation showcases essential preprocessing steps including normalization, encoding categorical variables, and handling missing values.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
    
    def preprocess(self, df):
        # Create a copy of the dataframe
        processed_df = df.copy()
        
        # Handle missing values
        numerical_columns = processed_df.select_dtypes(
            include=['float64', 'int64']).columns
        processed_df[numerical_columns] = self.imputer.fit_transform(
            processed_df[numerical_columns])
        
        # Encode categorical variables
        categorical_columns = processed_df.select_dtypes(
            include=['object']).columns
        for column in categorical_columns:
            le = LabelEncoder()
            processed_df[column] = le.fit_transform(
                processed_df[column].astype(str))
            self.label_encoders[column] = le
        
        # Scale numerical features
        processed_df[numerical_columns] = self.scaler.fit_transform(
            processed_df[numerical_columns])
        
        return processed_df

# Example usage
data = pd.DataFrame({
    'numeric_feat': [1, 2, None, 4, 5],
    'category': ['A', 'B', 'A', 'C', 'B']
})
preprocessor = DataPreprocessor()
processed_data = preprocessor.preprocess(data)
print("Original data:\n", data)
print("\nProcessed data:\n", processed_data)
```

Slide 3: Dataset Class Implementation

Custom dataset implementation is crucial for efficient data handling in neural networks. This class manages data loading, batching, and provides iteration capabilities while maintaining memory efficiency for large datasets.

```python
import numpy as np
from torch.utils.data import Dataset, DataLoader

class NeuralDataset(Dataset):
    def __init__(self, features, labels, batch_size=32):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Get feature and label for given index
        feature = self.features[idx]
        label = self.labels[idx]
        
        return {'feature': feature, 'label': label}
    
    def get_batches(self):
        # Generate batches for training
        indices = np.random.permutation(len(self))
        for start_idx in range(0, len(self), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            batch_features = self.features[batch_indices]
            batch_labels = self.labels[batch_indices]
            yield batch_features, batch_labels

# Example usage
X = np.random.randn(1000, 10)  # 1000 samples, 10 features
y = np.random.randint(0, 2, 1000)  # Binary labels
dataset = NeuralDataset(X, y)

# Create data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Print first batch
for batch in dataloader:
    print("Batch shape:", batch['feature'].shape)
    print("Labels shape:", batch['label'].shape)
    break
```

Slide 4: Data Augmentation Techniques

Data augmentation is essential for improving model generalization by artificially expanding the training dataset. This implementation demonstrates various augmentation techniques for different data types, including numerical and categorical features.

```python
import numpy as np
from scipy.interpolate import interp1d

class DataAugmenter:
    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level
    
    def add_gaussian_noise(self, data):
        """Add Gaussian noise to numerical features"""
        noise = np.random.normal(0, self.noise_level, data.shape)
        return data + noise
    
    def smote_like_synthesis(self, data, num_synthetic=1):
        """Generate synthetic samples using SMOTE-like approach"""
        synthetic_samples = []
        for _ in range(num_synthetic):
            idx = np.random.randint(0, len(data))
            neighbor_idx = np.random.randint(0, len(data))
            
            # Generate synthetic sample
            diff = data[neighbor_idx] - data[idx]
            gap = np.random.random()
            synthetic = data[idx] + gap * diff
            synthetic_samples.append(synthetic)
            
        return np.array(synthetic_samples)
    
    def time_warp(self, sequence, num_warps=1):
        """Apply time warping to sequential data"""
        warped_sequences = []
        for _ in range(num_warps):
            time_steps = np.arange(len(sequence))
            distorted_steps = time_steps + np.random.normal(
                0, self.noise_level, len(time_steps))
            
            # Interpolate sequence
            warper = interp1d(time_steps, sequence, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
            warped = warper(distorted_steps)
            warped_sequences.append(warped)
            
        return np.array(warped_sequences)

# Example usage
data = np.random.randn(100, 5)  # Sample dataset
augmenter = DataAugmenter(noise_level=0.1)

# Apply augmentations
noisy_data = augmenter.add_gaussian_noise(data)
synthetic_samples = augmenter.smote_like_synthesis(data, num_synthetic=10)
sequence = np.sin(np.linspace(0, 10, 100))  # Sample sequence
warped_sequences = augmenter.time_warp(sequence, num_warps=3)

print("Original data shape:", data.shape)
print("Synthetic samples shape:", synthetic_samples.shape)
print("Warped sequences shape:", warped_sequences.shape)
```

Slide 5: Data Validation and Cross-Validation

Implementing robust validation techniques ensures model reliability and helps prevent overfitting. This code demonstrates stratified k-fold cross-validation and custom validation metrics for neural network performance assessment.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve
import numpy as np

class DataValidator:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state)
        
    def cross_validate(self, X, y, model):
        """Perform stratified k-fold cross-validation"""
        metrics = {
            'auc_scores': [],
            'precision_scores': [],
            'recall_scores': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            auc = roc_auc_score(y_val, y_pred)
            precision, recall, _ = precision_recall_curve(y_val, y_pred)
            
            metrics['auc_scores'].append(auc)
            metrics['precision_scores'].append(np.mean(precision))
            metrics['recall_scores'].append(np.mean(recall))
            
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def validate_distribution(self, train_data, val_data):
        """Check for distribution shifts between train and validation sets"""
        distribution_metrics = {}
        
        for column in train_data.columns:
            if train_data[column].dtype in ['int64', 'float64']:
                # KS test for numerical features
                from scipy.stats import ks_2samp
                ks_stat, p_value = ks_2samp(
                    train_data[column], val_data[column])
                distribution_metrics[column] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value
                }
        
        return distribution_metrics

# Example usage
from sklearn.ensemble import RandomForestClassifier
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

validator = DataValidator(n_splits=5)
model = RandomForestClassifier(random_state=42)
cv_results = validator.cross_validate(X, y, model)

print("Cross-validation results:", cv_results)
```

Slide 6: Data Streaming for Large Datasets

Processing large datasets requires efficient streaming mechanisms to handle memory constraints. This implementation creates a generator-based streaming system for loading and preprocessing data in chunks while maintaining computational efficiency.

```python
import pandas as pd
import numpy as np
from typing import Generator

class DataStreamer:
    def __init__(self, file_path: str, chunk_size: int = 1000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self._validate_file()
        
    def _validate_file(self):
        try:
            pd.read_csv(self.file_path, nrows=1)
        except Exception as e:
            raise ValueError(f"Invalid file: {str(e)}")
    
    def stream_data(self) -> Generator:
        """Stream data in chunks with preprocessing"""
        chunks = pd.read_csv(
            self.file_path, 
            chunksize=self.chunk_size,
            iterator=True
        )
        
        for chunk in chunks:
            # Preprocess chunk
            processed_chunk = self._preprocess_chunk(chunk)
            yield processed_chunk
            
    def _preprocess_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps to each chunk"""
        # Remove duplicates
        chunk = chunk.drop_duplicates()
        
        # Handle missing values
        numeric_cols = chunk.select_dtypes(include=[np.number]).columns
        chunk[numeric_cols] = chunk[numeric_cols].fillna(
            chunk[numeric_cols].mean())
        
        # Normalize numerical features
        for col in numeric_cols:
            chunk[col] = (chunk[col] - chunk[col].mean()) / chunk[col].std()
            
        return chunk

    def get_statistics(self) -> dict:
        """Calculate running statistics for the dataset"""
        stats = {
            'total_rows': 0,
            'processed_chunks': 0,
            'memory_usage': []
        }
        
        for chunk in self.stream_data():
            stats['total_rows'] += len(chunk)
            stats['processed_chunks'] += 1
            stats['memory_usage'].append(chunk.memory_usage().sum() / 1024**2)
            
        return stats

# Example usage
import tempfile

# Create sample data file
temp_file = tempfile.NamedTemporaryFile(delete=False)
sample_data = pd.DataFrame({
    'feature1': np.random.randn(10000),
    'feature2': np.random.randn(10000)
})
sample_data.to_csv(temp_file.name, index=False)

# Initialize streamer
streamer = DataStreamer(temp_file.name, chunk_size=1000)

# Process data in streams
for i, chunk in enumerate(streamer.stream_data()):
    print(f"Processing chunk {i+1}")
    print(f"Chunk shape: {chunk.shape}")
    print(f"Memory usage: {chunk.memory_usage().sum() / 1024**2:.2f} MB\n")
    if i >= 2:  # Show only first 3 chunks
        break

# Get overall statistics
stats = streamer.get_statistics()
print("Dataset Statistics:", stats)
```

Slide 7: Data Distribution Analysis

Understanding data distributions is crucial for model performance. This implementation provides tools for analyzing and visualizing feature distributions, identifying skewness, and detecting distribution shifts in the dataset.

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple

class DistributionAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.numerical_cols = data.select_dtypes(
            include=[np.number]).columns
    
    def analyze_distributions(self) -> Dict:
        """Analyze distributions of numerical features"""
        distribution_stats = {}
        
        for col in self.numerical_cols:
            # Calculate basic statistics
            basic_stats = self._calculate_basic_stats(self.data[col])
            
            # Test for normality
            normality_test = self._test_normality(self.data[col])
            
            # Detect outliers
            outliers = self._detect_outliers(self.data[col])
            
            distribution_stats[col] = {
                'basic_stats': basic_stats,
                'normality_test': normality_test,
                'outliers': outliers
            }
            
        return distribution_stats
    
    def _calculate_basic_stats(self, series: pd.Series) -> Dict:
        """Calculate basic statistical measures"""
        return {
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'skewness': stats.skew(series.dropna()),
            'kurtosis': stats.kurtosis(series.dropna())
        }
    
    def _test_normality(self, series: pd.Series) -> Dict:
        """Perform normality tests"""
        statistic, p_value = stats.normaltest(series.dropna())
        return {
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
    
    def _detect_outliers(self, series: pd.Series) -> Dict:
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            'count': len(outliers),
            'percentage': len(outliers) / len(series) * 100,
            'bounds': (lower_bound, upper_bound)
        }
    
    def compare_distributions(self, other_data: pd.DataFrame) -> Dict:
        """Compare distributions between two datasets"""
        comparison_results = {}
        
        for col in self.numerical_cols:
            if col in other_data.columns:
                # Perform Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(
                    self.data[col].dropna(),
                    other_data[col].dropna()
                )
                
                comparison_results[col] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'distributions_different': p_value < 0.05
                }
                
        return comparison_results

# Example usage
np.random.seed(42)
data1 = pd.DataFrame({
    'normal_dist': np.random.normal(0, 1, 1000),
    'skewed_dist': np.random.exponential(2, 1000),
    'uniform_dist': np.random.uniform(0, 1, 1000)
})

data2 = pd.DataFrame({
    'normal_dist': np.random.normal(0.5, 1, 1000),
    'skewed_dist': np.random.exponential(2.5, 1000),
    'uniform_dist': np.random.uniform(0.2, 1.2, 1000)
})

analyzer = DistributionAnalyzer(data1)
distribution_stats = analyzer.analyze_distributions()
comparison_results = analyzer.compare_distributions(data2)

print("Distribution Statistics:")
print(distribution_stats)
print("\nDistribution Comparison Results:")
print(comparison_results)
```

Slide 8: Batch Generator with Memory Management

Efficient batch generation is crucial for training neural networks on large datasets. This implementation provides a memory-efficient batch generator with advanced shuffling and prefetching capabilities.

```python
import numpy as np
from typing import Generator, Tuple, Optional
import threading
import queue

class AdvancedBatchGenerator:
    def __init__(self, data: np.ndarray, labels: np.ndarray, 
                 batch_size: int = 32, prefetch_size: int = 2):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.prefetch_queue = queue.Queue(maxsize=prefetch_size)
        self._validate_inputs()
    
    def _validate_inputs(self):
        assert len(self.data) == len(self.labels), "Data and labels must have same length"
        assert self.batch_size > 0, "Batch size must be positive"
        
    def _create_batch(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create a single batch from indices"""
        batch_data = self.data[indices]
        batch_labels = self.labels[indices]
        return batch_data, batch_labels
    
    def _prefetch_worker(self, indices: np.ndarray):
        """Worker function for prefetching batches"""
        start_idx = 0
        while start_idx < len(indices):
            end_idx = min(start_idx + self.batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            batch = self._create_batch(batch_indices)
            self.prefetch_queue.put(batch)
            start_idx = end_idx
    
    def generate_batches(self, shuffle: bool = True) -> Generator:
        """Generate batches with optional shuffling and prefetching"""
        indices = np.arange(len(self.data))
        if shuffle:
            np.random.shuffle(indices)
        
        # Start prefetching thread
        prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(indices,)
        )
        prefetch_thread.daemon = True
        prefetch_thread.start()
        
        # Yield batches from queue
        num_batches = int(np.ceil(len(indices) / self.batch_size))
        for _ in range(num_batches):
            try:
                batch = self.prefetch_queue.get(timeout=10)
                yield batch
            except queue.Empty:
                break
                
    def get_memory_usage(self) -> float:
        """Calculate approximate memory usage in MB"""
        data_memory = self.data.nbytes / (1024 * 1024)
        labels_memory = self.labels.nbytes / (1024 * 1024)
        batch_memory = (data_memory + labels_memory) * self.prefetch_size / len(self.data)
        return {
            'total_data_memory': data_memory,
            'total_labels_memory': labels_memory,
            'batch_memory': batch_memory
        }

# Example usage
# Generate sample data
X = np.random.randn(10000, 100)  # 10000 samples with 100 features
y = np.random.randint(0, 2, 10000)  # Binary labels

# Initialize batch generator
batch_gen = AdvancedBatchGenerator(
    data=X,
    labels=y,
    batch_size=64,
    prefetch_size=2
)

# Print memory usage
print("Memory Usage (MB):", batch_gen.get_memory_usage())

# Generate and process batches
for batch_idx, (batch_data, batch_labels) in enumerate(
    batch_gen.generate_batches(shuffle=True)):
    print(f"Batch {batch_idx + 1}:")
    print(f"  Data shape: {batch_data.shape}")
    print(f"  Labels shape: {batch_labels.shape}")
    if batch_idx >= 2:  # Show only first 3 batches
        break
```

Slide 9: Data Cleaning and Standardization

A comprehensive data cleaning pipeline that handles various types of data issues while maintaining data integrity. This implementation includes advanced techniques for outlier detection, missing value imputation, and feature scaling.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional, List

class AdvancedDataCleaner:
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.scalers = {}
        self.statistics = {}
        
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main cleaning pipeline"""
        df_cleaned = df.copy()
        
        # Remove duplicate rows
        df_cleaned = self._remove_duplicates(df_cleaned)
        
        # Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned)
        
        # Remove outliers
        df_cleaned = self._remove_outliers(df_cleaned)
        
        # Standardize numerical features
        df_cleaned = self._standardize_features(df_cleaned)
        
        return df_cleaned
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows with advanced handling"""
        initial_rows = len(df)
        df = df.drop_duplicates(keep='first')
        
        self.statistics['duplicates_removed'] = initial_rows - len(df)
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using multiple strategies"""
        # For numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                # Use inter-quartile range for imputation
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                median = df[col].median()
                df[col] = df[col].fillna(median)
                
                # Cap outliers
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        # For categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)
            
        self.statistics['missing_values'] = {
            col: df[col].isnull().sum() for col in df.columns
        }
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using Isolation Forest"""
        from sklearn.ensemble import IsolationForest
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            iso_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            outliers = iso_forest.fit_predict(df[numerical_cols])
            df = df[outliers == 1]
            
        self.statistics['outliers_removed'] = len(outliers[outliers == -1])
        return df
    
    def _standardize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize numerical features using RobustScaler"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            scaler = RobustScaler()
            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler
            
        return df
    
    def get_cleaning_statistics(self) -> Dict:
        """Return statistics about the cleaning process"""
        return self.statistics

# Example usage
# Create sample dirty dataset
np.random.seed(42)
sample_data = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.concatenate([
        np.random.randn(950),
        np.random.randn(50) * 10  # Outliers
    ]),
    'category': np.random.choice(['A', 'B', 'C', None], 1000)
})

# Add some missing values
sample_data.loc[np.random.choice(len(sample_data), 100), 'feature1'] = np.nan

# Initialize and run cleaner
cleaner = AdvancedDataCleaner(contamination=0.1)
cleaned_data = cleaner.clean_dataset(sample_data)

print("Original shape:", sample_data.shape)
print("Cleaned shape:", cleaned_data.shape)
print("\nCleaning statistics:")
print(cleaner.get_cleaning_statistics())
```

Slide 10: Feature Importance Analysis

Analyzing feature importance helps identify the most relevant data attributes for neural network training. This implementation provides multiple methods for feature importance calculation and visualization using statistical and model-based approaches.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from typing import Dict, List, Tuple

class FeatureImportanceAnalyzer:
    def __init__(self, data: pd.DataFrame, target: str):
        self.data = data
        self.target = target
        self.feature_cols = [col for col in data.columns if col != target]
        self.importance_scores = {}
        
    def analyze_feature_importance(self) -> Dict[str, Dict]:
        """Analyze feature importance using multiple methods"""
        # Calculate importance using different methods
        self.importance_scores['mutual_information'] = self._mutual_information()
        self.importance_scores['random_forest'] = self._random_forest_importance()
        self.importance_scores['correlation'] = self._correlation_analysis()
        
        return self.get_consolidated_importance()
    
    def _mutual_information(self) -> Dict[str, float]:
        """Calculate mutual information scores"""
        X = self.data[self.feature_cols]
        y = self.data[self.target]
        
        mi_scores = mutual_info_classif(X, y)
        return dict(zip(self.feature_cols, mi_scores))
    
    def _random_forest_importance(self) -> Dict[str, float]:
        """Calculate feature importance using Random Forest"""
        X = self.data[self.feature_cols]
        y = self.data[self.target]
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        return dict(zip(self.feature_cols, rf.feature_importances_))
    
    def _correlation_analysis(self) -> Dict[str, Dict[str, float]]:
        """Analyze feature correlations"""
        correlation_matrix = self.data.corr()
        target_correlations = correlation_matrix[self.target].abs()
        
        return dict(target_correlations[self.feature_cols])
    
    def get_consolidated_importance(self) -> Dict[str, Dict]:
        """Consolidate importance scores from all methods"""
        consolidated = {}
        
        for feature in self.feature_cols:
            consolidated[feature] = {
                'mutual_information': self.importance_scores['mutual_information'][feature],
                'random_forest': self.importance_scores['random_forest'][feature],
                'correlation': self.importance_scores['correlation'][feature],
                'average_score': np.mean([
                    self.importance_scores['mutual_information'][feature],
                    self.importance_scores['random_forest'][feature],
                    self.importance_scores['correlation'][feature]
                ])
            }
            
        return consolidated
    
    def get_top_features(self, n_features: int = 10) -> List[str]:
        """Get top n most important features"""
        consolidated = self.get_consolidated_importance()
        
        # Sort features by average importance score
        sorted_features = sorted(
            consolidated.items(),
            key=lambda x: x[1]['average_score'],
            reverse=True
        )
        
        return [feature for feature, _ in sorted_features[:n_features]]

# Example usage
# Create sample dataset
np.random.seed(42)
n_samples = 1000
n_features = 20

# Generate synthetic features
X = np.random.randn(n_samples, n_features)
# Generate target variable with dependence on first 5 features
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2 + 
     X[:, 3] * 0.1 + X[:, 4] * 0.1 + np.random.randn(n_samples) * 0.1)
y = (y > y.mean()).astype(int)

# Create DataFrame
feature_names = [f'feature_{i}' for i in range(n_features)]
data = pd.DataFrame(X, columns=feature_names)
data['target'] = y

# Initialize and run analyzer
analyzer = FeatureImportanceAnalyzer(data, 'target')
importance_results = analyzer.analyze_feature_importance()
top_features = analyzer.get_top_features(n_features=5)

print("Top 5 Most Important Features:")
for feature in top_features:
    scores = importance_results[feature]
    print(f"\n{feature}:")
    for method, score in scores.items():
        print(f"  {method}: {score:.4f}")
```

Slide 11: Data Drift Detection

Monitoring and detecting data drift is essential for maintaining model performance over time. This implementation provides methods for detecting and quantifying various types of data drift between training and production datasets.

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import warnings

class DataDriftDetector:
    def __init__(self, reference_data: pd.DataFrame, 
                 current_data: pd.DataFrame,
                 drift_threshold: float = 0.05):
        self.reference_data = reference_data
        self.current_data = current_data
        self.drift_threshold = drift_threshold
        self.drift_metrics = {}
        
    def detect_drift(self) -> Dict:
        """Detect various types of drift in the dataset"""
        self.drift_metrics['statistical_drift'] = self._detect_statistical_drift()
        self.drift_metrics['distribution_drift'] = self._detect_distribution_drift()
        self.drift_metrics['correlation_drift'] = self._detect_correlation_drift()
        
        return self.get_drift_summary()
    
    def _detect_statistical_drift(self) -> Dict[str, Dict]:
        """Detect drift in basic statistical measures"""
        stats_drift = {}
        
        for column in self.reference_data.columns:
            if pd.api.types.is_numeric_dtype(self.reference_data[column]):
                ref_stats = self._calculate_statistics(self.reference_data[column])
                curr_stats = self._calculate_statistics(self.current_data[column])
                
                # Calculate relative changes
                drift_metrics = {
                    metric: abs(curr_stats[metric] - ref_stats[metric]) / 
                            (abs(ref_stats[metric]) + 1e-10)
                    for metric in ref_stats.keys()
                }
                
                stats_drift[column] = {
                    'metrics': drift_metrics,
                    'has_drift': any(v > self.drift_threshold for v in drift_metrics.values())
                }
                
        return stats_drift
    
    def _detect_distribution_drift(self) -> Dict[str, Dict]:
        """Detect drift in feature distributions"""
        dist_drift = {}
        
        for column in self.reference_data.columns:
            if pd.api.types.is_numeric_dtype(self.reference_data[column]):
                # Perform Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(
                    self.reference_data[column].dropna(),
                    self.current_data[column].dropna()
                )
                
                dist_drift[column] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'has_drift': p_value < self.drift_threshold
                }
                
        return dist_drift
    
    def _detect_correlation_drift(self) -> Dict[str, float]:
        """Detect drift in feature correlations"""
        ref_corr = self.reference_data.corr()
        curr_corr = self.current_data.corr()
        
        correlation_drift = {}
        for col1 in ref_corr.columns:
            for col2 in ref_corr.columns:
                if col1 < col2:  # Only consider upper triangle
                    key = f"{col1}_{col2}"
                    ref_value = ref_corr.loc[col1, col2]
                    curr_value = curr_corr.loc[col1, col2]
                    
                    correlation_drift[key] = {
                        'reference_correlation': ref_value,
                        'current_correlation': curr_value,
                        'absolute_difference': abs(ref_value - curr_value),
                        'has_drift': abs(ref_value - curr_value) > self.drift_threshold
                    }
                    
        return correlation_drift
    
    def _calculate_statistics(self, series: pd.Series) -> Dict[str, float]:
        """Calculate basic statistical measures"""
        return {
            'mean': series.mean(),
            'std': series.std(),
            'median': series.median(),
            'skewness': stats.skew(series.dropna()),
            'kurtosis': stats.kurtosis(series.dropna())
        }
    
    def get_drift_summary(self) -> Dict:
        """Summarize drift detection results"""
        summary = {
            'overall_drift_detected': False,
            'statistical_drift_count': 0,
            'distribution_drift_count': 0,
            'correlation_drift_count': 0,
            'detailed_metrics': self.drift_metrics
        }
        
        # Count drifting features
        for feature in self.drift_metrics['statistical_drift']:
            if self.drift_metrics['statistical_drift'][feature]['has_drift']:
                summary['statistical_drift_count'] += 1
                
        for feature in self.drift_metrics['distribution_drift']:
            if self.drift_metrics['distribution_drift'][feature]['has_drift']:
                summary['distribution_drift_count'] += 1
                
        correlation_drifts = sum(
            1 for v in self.drift_metrics['correlation_drift'].values() 
            if v['has_drift']
        )
        summary['correlation_drift_count'] = correlation_drifts
        
        # Determine overall drift
        summary['overall_drift_detected'] = any([
            summary['statistical_drift_count'] > 0,
            summary['distribution_drift_count'] > 0,
            summary['correlation_drift_count'] > 0
        ])
        
        return summary

# Example usage
# Generate reference and current datasets with drift
np.random.seed(42)
n_samples = 1000
n_features = 5

# Reference data
X_ref = np.random.randn(n_samples, n_features)
df_ref = pd.DataFrame(
    X_ref, 
    columns=[f'feature_{i}' for i in range(n_features)]
)

# Current data with drift
X_curr = np.random.randn(n_samples, n_features) * 1.2 + 0.5
df_curr = pd.DataFrame(
    X_curr, 
    columns=[f'feature_{i}' for i in range(n_features)]
)

# Initialize and run drift detector
detector = DataDriftDetector(df_ref, df_curr)
drift_results = detector.detect_drift()

print("Drift Detection Summary:")
print(f"Overall drift detected: {drift_results['overall_drift_detected']}")
print(f"Statistical drift count: {drift_results['statistical_drift_count']}")
print(f"Distribution drift count: {drift_results['distribution_drift_count']}")
print(f"Correlation drift count: {drift_results['correlation_drift_count']}")
```

Slide 12: Advanced Data Sampling Techniques

This implementation provides sophisticated sampling methods for handling imbalanced datasets and creating representative subsets for neural network training, including stratified and adaptive sampling approaches.

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Dict, Tuple, Optional, List

class AdvancedSampler:
    def __init__(self, data: pd.DataFrame, target_column: str):
        self.data = data
        self.target_column = target_column
        self.sampling_stats = {}
        
    def adaptive_sampling(self, target_size: int,
                         method: str = 'stratified_kmeans') -> pd.DataFrame:
        """Perform adaptive sampling using specified method"""
        if method == 'stratified_kmeans':
            return self._stratified_kmeans_sampling(target_size)
        elif method == 'density_based':
            return self._density_based_sampling(target_size)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def _stratified_kmeans_sampling(self, target_size: int) -> pd.DataFrame:
        """Combine stratification and k-means for intelligent sampling"""
        features = self.data.drop(columns=[self.target_column])
        labels = self.data[self.target_column]
        
        sampled_indices = []
        unique_labels = labels.unique()
        
        for label in unique_labels:
            # Get indices for current class
            class_indices = labels[labels == label].index
            class_data = features.loc[class_indices]
            
            # Calculate proportional sample size for this class
            proportion = len(class_indices) / len(self.data)
            class_target_size = int(target_size * proportion)
            
            if len(class_indices) <= class_target_size:
                sampled_indices.extend(class_indices)
            else:
                # Use k-means to find representatives
                kmeans = KMeans(
                    n_clusters=class_target_size,
                    random_state=42
                )
                clusters = kmeans.fit_predict(class_data)
                
                # Select samples closest to centroids
                selected = []
                for cluster_id in range(class_target_size):
                    cluster_points = class_data[clusters == cluster_id]
                    centroid = kmeans.cluster_centers_[cluster_id]
                    
                    # Find point closest to centroid
                    distances = np.linalg.norm(
                        cluster_points - centroid, axis=1
                    )
                    closest_idx = cluster_points.index[np.argmin(distances)]
                    selected.append(closest_idx)
                
                sampled_indices.extend(selected)
        
        self.sampling_stats['stratified_kmeans'] = {
            'original_size': len(self.data),
            'sampled_size': len(sampled_indices),
            'class_distribution': self.data.loc[sampled_indices][self.target_column].value_counts().to_dict()
        }
        
        return self.data.loc[sampled_indices]
    
    def _density_based_sampling(self, target_size: int) -> pd.DataFrame:
        """Sample based on data density estimation"""
        from sklearn.neighbors import KernelDensity
        
        features = self.data.drop(columns=[self.target_column])
        
        # Estimate density for each point
        kde = KernelDensity(kernel='gaussian')
        kde.fit(features)
        log_density = kde.score_samples(features)
        density = np.exp(log_density)
        
        # Calculate sampling probabilities
        sampling_probs = 1 / (density + 1e-10)
        sampling_probs = sampling_probs / sampling_probs.sum()
        
        # Sample points
        sampled_indices = np.random.choice(
            len(self.data),
            size=target_size,
            p=sampling_probs,
            replace=False
        )
        
        self.sampling_stats['density_based'] = {
            'original_size': len(self.data),
            'sampled_size': len(sampled_indices),
            'density_stats': {
                'min_density': density.min(),
                'max_density': density.max(),
                'mean_density': density.mean()
            }
        }
        
        return self.data.loc[sampled_indices]
    
    def get_sampling_stats(self) -> Dict:
        """Return statistics about the sampling process"""
        return self.sampling_stats

# Example usage
# Generate imbalanced dataset
np.random.seed(42)
n_samples = 1000

# Create imbalanced synthetic data
X1 = np.random.normal(0, 1, (800, 2))  # Majority class
X2 = np.random.normal(3, 1, (200, 2))  # Minority class

# Combine data and create labels
X = np.vstack([X1, X2])
y = np.hstack([np.zeros(800), np.ones(200)])

# Create DataFrame
df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
df['target'] = y

# Initialize sampler
sampler = AdvancedSampler(df, 'target')

# Perform sampling
target_size = 500
stratified_sample = sampler.adaptive_sampling(
    target_size, method='stratified_kmeans'
)
density_sample = sampler.adaptive_sampling(
    target_size, method='density_based'
)

# Print sampling statistics
print("Original class distribution:")
print(df['target'].value_counts())

print("\nStratified K-means sampling results:")
print(stratified_sample['target'].value_counts())

print("\nDensity-based sampling results:")
print(density_sample['target'].value_counts())

print("\nSampling statistics:")
print(sampler.get_sampling_stats())
```

Slide 13: Additional Resources

*   "A Survey on Data Collection and Management Techniques for Neural Networks" [https://arxiv.org/abs/2201.00494](https://arxiv.org/abs/2201.00494)
*   "Data Quality Assessment Methods for Deep Learning" [https://arxiv.org/abs/2108.02497](https://arxiv.org/abs/2108.02497)
*   "Efficient Data Sampling Strategies for Deep Neural Networks" [https://arxiv.org/abs/2105.05542](https://arxiv.org/abs/2105.05542)
*   "Detection and Mitigation of Data Drift in Production ML Systems" Search on Google Scholar for recent papers on data drift detection
*   "Best Practices for Feature Engineering in Neural Networks" Visit [https://paperswithcode.com](https://paperswithcode.com) for latest research on feature engineering
*   "Handling Imbalanced Datasets in Deep Learning" Search IEEE Xplore Digital Library for comprehensive reviews

