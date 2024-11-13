## Leveraging Big Data Analytics
Slide 1: Understanding Sampling Distributions

Statistical sampling theory forms the foundation of data science, enabling us to make inferences about populations through representative subsets. The central limit theorem demonstrates that sample means approximate normal distribution regardless of the underlying population distribution.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate population data with non-normal distribution
population = np.concatenate([
    np.random.exponential(size=10000),
    np.random.gamma(2, 2, size=10000)
])

# Function to calculate sampling distribution
def sampling_distribution(data, sample_size, n_samples):
    sample_means = []
    for _ in range(n_samples):
        sample = np.random.choice(data, size=sample_size, replace=True)
        sample_means.append(np.mean(sample))
    return np.array(sample_means)

# Calculate sampling distribution
sample_means = sampling_distribution(population, sample_size=30, n_samples=1000)

# Visualization
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.hist(population, bins=50, density=True)
plt.title('Population Distribution')
plt.subplot(122)
plt.hist(sample_means, bins=50, density=True)
plt.title('Sampling Distribution of Mean')
plt.show()
```

Slide 2: Simple Random Sampling Implementation

Simple random sampling provides an unbiased method for dataset reduction where each element has an equal probability of selection. This implementation demonstrates both with and without replacement sampling techniques.

```python
import numpy as np
import pandas as pd

class SimpleRandomSampler:
    def __init__(self, data):
        self.data = data
        
    def sample(self, size, replace=False):
        """
        Parameters:
        size (int): Sample size
        replace (bool): Sampling with/without replacement
        """
        indices = np.random.choice(
            len(self.data), 
            size=size, 
            replace=replace
        )
        return self.data.iloc[indices]

# Example usage
df = pd.DataFrame({
    'value': np.random.normal(0, 1, 10000),
    'category': np.random.choice(['A', 'B', 'C'], 10000)
})

sampler = SimpleRandomSampler(df)
sample = sampler.sample(size=1000, replace=False)
print(f"Original size: {len(df)}, Sample size: {len(sample)}")
print("\nSample statistics vs Population statistics:")
print(df['value'].describe().round(2))
print(sample['value'].describe().round(2))
```

Slide 3: Stratified Sampling

Stratified sampling ensures proportional representation of different subgroups within the data, maintaining the original distribution of key variables while reducing sample size. This is crucial for handling imbalanced datasets.

```python
class StratifiedSampler:
    def __init__(self, data, strata_column):
        self.data = data
        self.strata_column = strata_column
        
    def sample(self, sample_size):
        """
        Performs proportional stratified sampling
        """
        strata = self.data[self.strata_column].value_counts(normalize=True)
        sampled_data = []
        
        for stratum, proportion in strata.items():
            stratum_size = int(sample_size * proportion)
            stratum_data = self.data[
                self.data[self.strata_column] == stratum
            ]
            sampled_stratum = stratum_data.sample(
                n=stratum_size, 
                random_state=42
            )
            sampled_data.append(sampled_stratum)
            
        return pd.concat(sampled_data, axis=0)

# Example usage
data = pd.DataFrame({
    'value': np.random.normal(0, 1, 10000),
    'category': np.random.choice(['A', 'B', 'C'], 10000, 
                                p=[0.6, 0.3, 0.1])
})

stratified = StratifiedSampler(data, 'category')
sample = stratified.sample(1000)

print("Original distribution:")
print(data['category'].value_counts(normalize=True))
print("\nSampled distribution:")
print(sample['category'].value_counts(normalize=True))
```

Slide 4: Systematic Sampling for Time Series

Systematic sampling is particularly useful for time series data, where we want to maintain temporal patterns while reducing data volume. This implementation includes handling of seasonal variations and trend components.

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class SystematicTimeSampler:
    def __init__(self, time_series, interval):
        self.time_series = time_series
        self.interval = interval
        
    def sample(self):
        """
        Performs systematic sampling on time series data
        """
        indices = np.arange(0, len(self.time_series), self.interval)
        return self.time_series.iloc[indices]

# Generate example time series data
dates = pd.date_range(
    start='2023-01-01', 
    end='2023-12-31', 
    freq='H'
)
values = np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + \
         np.random.normal(0, 0.1, len(dates))

ts_data = pd.Series(values, index=dates)

# Sample every 6 hours
sampler = SystematicTimeSampler(ts_data, interval=6)
sampled_ts = sampler.sample()

print(f"Original observations: {len(ts_data)}")
print(f"Sampled observations: {len(sampled_ts)}")
print("\nFirst 5 original timestamps:")
print(ts_data.head().index)
print("\nFirst 5 sampled timestamps:")
print(sampled_ts.head().index)
```

Slide 5: Statistical Validation of Sampling Methods

Implementing statistical tests to validate sampling quality ensures that our reduced dataset maintains the essential characteristics of the original population. This approach uses Kolmogorov-Smirnov and Chi-square tests for distribution comparison.

```python
from scipy import stats
import numpy as np
import pandas as pd

class SamplingValidator:
    def __init__(self, population, sample):
        self.population = population
        self.sample = sample
        
    def ks_test(self, column):
        """
        Performs Kolmogorov-Smirnov test for continuous variables
        """
        statistic, p_value = stats.ks_2samp(
            self.population[column],
            self.sample[column]
        )
        return {'statistic': statistic, 'p_value': p_value}
    
    def chi_square_test(self, column):
        """
        Performs Chi-square test for categorical variables
        """
        pop_freq = pd.value_counts(self.population[column])
        sample_freq = pd.value_counts(self.sample[column])
        
        # Normalize sample frequencies
        scale_factor = len(self.population) / len(self.sample)
        sample_freq = sample_freq * scale_factor
        
        statistic, p_value = stats.chisquare(
            sample_freq, 
            pop_freq[sample_freq.index]
        )
        return {'statistic': statistic, 'p_value': p_value}

# Example usage
np.random.seed(42)
population = pd.DataFrame({
    'continuous': np.random.normal(0, 1, 10000),
    'categorical': np.random.choice(['A', 'B', 'C'], 10000)
})

sample = population.sample(n=1000)
validator = SamplingValidator(population, sample)

# Validate continuous variable
ks_results = validator.ks_test('continuous')
print("KS Test Results:", ks_results)

# Validate categorical variable
chi_results = validator.chi_square_test('categorical')
print("Chi-square Test Results:", chi_results)
```

Slide 6: Reservoir Sampling for Streaming Data

Reservoir sampling provides a method to maintain a fixed-size representative sample when processing streaming data of unknown size. This implementation ensures each element has an equal probability of being selected.

```python
import numpy as np
from typing import Iterator, List

class ReservoirSampler:
    def __init__(self, reservoir_size: int):
        self.reservoir_size = reservoir_size
        self.reservoir: List = []
        self.items_seen = 0
        
    def update(self, item: any) -> None:
        """
        Update reservoir with streaming item
        """
        self.items_seen += 1
        
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(item)
        else:
            j = np.random.randint(0, self.items_seen)
            if j < self.reservoir_size:
                self.reservoir[j] = item
    
    def process_stream(self, stream: Iterator) -> List:
        """
        Process entire data stream
        """
        for item in stream:
            self.update(item)
        return self.reservoir

# Example usage with simulated stream
def data_stream(n: int):
    for i in range(n):
        yield np.random.normal()

# Initialize sampler and process stream
sampler = ReservoirSampler(reservoir_size=1000)
stream_size = 1000000
sample = sampler.process_stream(data_stream(stream_size))

print(f"Stream size: {stream_size}")
print(f"Sample size: {len(sample)}")
print(f"Sample mean: {np.mean(sample):.3f}")
print(f"Sample std: {np.std(sample):.3f}")
```

Slide 7: Adaptive Sampling Based on Data Distribution

Adaptive sampling adjusts the sampling rate based on data characteristics, increasing efficiency by sampling more frequently in regions of high variability and less frequently in stable regions.

```python
import numpy as np
import pandas as pd
from scipy.stats import entropy

class AdaptiveSampler:
    def __init__(self, base_rate: float = 0.1, window_size: int = 100):
        self.base_rate = base_rate
        self.window_size = window_size
        
    def calculate_entropy(self, data: np.ndarray) -> float:
        """
        Calculate normalized entropy of data window
        """
        hist, _ = np.histogram(data, bins='auto', density=True)
        return entropy(hist) / np.log(len(hist))
    
    def sample(self, data: np.ndarray) -> np.ndarray:
        """
        Perform adaptive sampling based on local entropy
        """
        sampled_indices = []
        current_pos = 0
        
        while current_pos < len(data):
            window = data[current_pos:current_pos + self.window_size]
            if len(window) < self.window_size:
                break
                
            local_entropy = self.calculate_entropy(window)
            sample_rate = self.base_rate * (1 + local_entropy)
            
            if np.random.random() < sample_rate:
                sampled_indices.append(current_pos)
            
            current_pos += 1
            
        return data[sampled_indices]

# Example usage with synthetic data
np.random.seed(42)
# Generate data with varying complexity
x = np.linspace(0, 10, 1000)
data = np.sin(x) + np.random.normal(0, 0.1 + 0.2 * np.sin(x), len(x))

sampler = AdaptiveSampler(base_rate=0.1, window_size=50)
sampled_data = sampler.sample(data)

print(f"Original data size: {len(data)}")
print(f"Sampled data size: {len(sampled_data)}")
print(f"Effective sampling rate: {len(sampled_data)/len(data):.3f}")
```

Slide 8: Importance Sampling for Rare Events

Importance sampling focuses on capturing rare but significant events in the dataset by adjusting sampling probabilities based on event importance. This technique is crucial for imbalanced datasets and anomaly detection.

```python
import numpy as np
from scipy.stats import norm

class ImportanceSampler:
    def __init__(self, threshold: float):
        self.threshold = threshold
        
    def importance_weight(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate importance weights based on value magnitude
        """
        return np.exp(np.abs(x) - self.threshold)
    
    def sample(self, data: np.ndarray, sample_size: int) -> tuple:
        """
        Perform importance sampling
        Returns: (samples, weights)
        """
        weights = self.importance_weight(data)
        probs = weights / np.sum(weights)
        indices = np.random.choice(
            len(data), 
            size=sample_size, 
            p=probs
        )
        
        return data[indices], weights[indices]

# Generate synthetic data with rare events
np.random.seed(42)
normal_data = np.random.normal(0, 1, 10000)
rare_events = np.random.normal(4, 0.5, 100)
data = np.concatenate([normal_data, rare_events])

# Apply importance sampling
sampler = ImportanceSampler(threshold=2.0)
samples, weights = sampler.sample(data, sample_size=1000)

print(f"Original rare event ratio: {np.mean(np.abs(data) > 3):.4f}")
print(f"Sampled rare event ratio: {np.mean(np.abs(samples) > 3):.4f}")
print(f"Mean importance weight: {np.mean(weights):.4f}")
```

Slide 9: Cross-Validation with Stratified Time Series Sampling

This implementation combines stratified sampling with time series cross-validation to ensure temporal dependencies are preserved while maintaining representative distributions across different time periods.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

class StratifiedTimeSeriesSampler:
    def __init__(self, n_splits: int, test_size: float):
        self.n_splits = n_splits
        self.test_size = test_size
        
    def split(self, data: pd.DataFrame, date_column: str, 
             strata_column: str) -> list:
        """
        Generate stratified time series splits
        """
        data = data.sort_values(date_column)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        splits = []
        
        for train_idx, test_idx in tscv.split(data):
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]
            
            # Stratified sampling within each fold
            strata = train[strata_column].value_counts(normalize=True)
            sampled_train = []
            
            for stratum, prop in strata.items():
                stratum_data = train[train[strata_column] == stratum]
                sample_size = int(len(stratum_data) * (1 - self.test_size))
                sampled_stratum = stratum_data.sample(n=sample_size)
                sampled_train.append(sampled_stratum)
                
            splits.append((
                pd.concat(sampled_train),
                test
            ))
            
        return splits

# Example usage
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
categories = np.random.choice(['A', 'B', 'C'], size=len(dates))
values = np.random.normal(0, 1, size=len(dates))

data = pd.DataFrame({
    'date': dates,
    'category': categories,
    'value': values
})

sampler = StratifiedTimeSeriesSampler(n_splits=5, test_size=0.2)
splits = sampler.split(data, 'date', 'category')

for i, (train, test) in enumerate(splits):
    print(f"\nFold {i+1}:")
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    print("Category distribution in train:")
    print(train['category'].value_counts(normalize=True))
```

Slide 10: Sequential Pattern Sampling

Sequential pattern sampling preserves temporal patterns and sequential dependencies while reducing data volume. This approach is particularly useful for time series analysis and sequence modeling.

```python
import numpy as np
import pandas as pd
from collections import defaultdict

class SequentialPatternSampler:
    def __init__(self, sequence_length: int, overlap: float = 0.5):
        self.sequence_length = sequence_length
        self.overlap = overlap
        
    def extract_patterns(self, data: np.ndarray) -> dict:
        """
        Extract sequential patterns with overlap
        """
        step_size = int(self.sequence_length * (1 - self.overlap))
        patterns = defaultdict(int)
        
        for i in range(0, len(data) - self.sequence_length + 1, step_size):
            pattern = tuple(data[i:i + self.sequence_length])
            patterns[pattern] += 1
            
        return patterns
    
    def sample(self, data: np.ndarray, sample_size: int) -> np.ndarray:
        """
        Sample sequences based on pattern frequency
        """
        patterns = self.extract_patterns(data)
        pattern_probs = np.array(list(patterns.values()))
        pattern_probs = pattern_probs / np.sum(pattern_probs)
        
        selected_patterns = np.random.choice(
            list(patterns.keys()),
            size=sample_size,
            p=pattern_probs
        )
        
        return np.array(list(selected_patterns))

# Generate example sequence data
np.random.seed(42)
n_samples = 1000
t = np.linspace(0, 10, n_samples)
sequence = np.sin(t) + np.random.normal(0, 0.1, n_samples)

# Sample sequences
sampler = SequentialPatternSampler(sequence_length=50, overlap=0.3)
sampled_sequences = sampler.sample(sequence, sample_size=20)

print(f"Original sequence length: {len(sequence)}")
print(f"Number of sampled sequences: {len(sampled_sequences)}")
print(f"Each sequence length: {sampled_sequences.shape[1]}")
```

Slide 11: Computational Performance Analysis of Sampling Methods

This implementation provides a comprehensive benchmark framework for comparing different sampling methods in terms of execution time, memory usage, and statistical accuracy across varying dataset sizes.

```python
import numpy as np
import pandas as pd
import time
import psutil
from memory_profiler import profile
from dataclasses import dataclass

@dataclass
class BenchmarkResults:
    execution_time: float
    memory_usage: float
    statistical_error: float

class SamplingBenchmark:
    def __init__(self, sampling_methods: dict):
        self.sampling_methods = sampling_methods
        
    def measure_performance(self, data: np.ndarray, 
                          sample_size: int) -> dict:
        results = {}
        
        for name, sampler in self.sampling_methods.items():
            start_time = time.time()
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Perform sampling
            sample = sampler(data, sample_size)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            memory_usage = (process.memory_info().rss - initial_memory) / 1024 / 1024
            statistical_error = np.abs(np.mean(sample) - np.mean(data))
            
            results[name] = BenchmarkResults(
                execution_time=execution_time,
                memory_usage=memory_usage,
                statistical_error=statistical_error
            )
            
        return results

# Define sampling methods
def simple_random_sample(data, size):
    return np.random.choice(data, size=size)

def systematic_sample(data, size):
    step = len(data) // size
    return data[::step]

def reservoir_sample(data, size):
    sample = np.zeros(size)
    for i, item in enumerate(data):
        if i < size:
            sample[i] = item
        else:
            j = np.random.randint(0, i)
            if j < size:
                sample[j] = item
    return sample

# Run benchmark
methods = {
    'simple_random': simple_random_sample,
    'systematic': systematic_sample,
    'reservoir': reservoir_sample
}

benchmark = SamplingBenchmark(methods)

# Test with different dataset sizes
sizes = [10000, 100000, 1000000]
sample_ratio = 0.1

for size in sizes:
    data = np.random.normal(0, 1, size)
    sample_size = int(size * sample_ratio)
    results = benchmark.measure_performance(data, sample_size)
    
    print(f"\nDataset size: {size}")
    for method, metrics in results.items():
        print(f"\n{method.upper()}:")
        print(f"Time: {metrics.execution_time:.4f}s")
        print(f"Memory: {metrics.memory_usage:.2f}MB")
        print(f"Error: {metrics.statistical_error:.6f}")
```

Slide 12: Bootstrapped Sampling for Uncertainty Estimation

Implementation of bootstrap sampling techniques to estimate uncertainty in sample statistics and model parameters, crucial for robust statistical inference in reduced datasets.

```python
import numpy as np
from scipy import stats
from typing import Callable, Tuple
from dataclasses import dataclass

@dataclass
class BootstrapEstimates:
    point_estimate: float
    confidence_interval: Tuple[float, float]
    standard_error: float

class BootstrapSampler:
    def __init__(self, n_iterations: int = 1000, 
                 confidence_level: float = 0.95):
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level
    
    def bootstrap_statistic(self, data: np.ndarray, 
                          statistic_fn: Callable) -> BootstrapEstimates:
        """
        Compute bootstrap estimates for given statistic
        """
        bootstrap_estimates = np.zeros(self.n_iterations)
        
        for i in range(self.n_iterations):
            bootstrap_sample = np.random.choice(
                data, 
                size=len(data), 
                replace=True
            )
            bootstrap_estimates[i] = statistic_fn(bootstrap_sample)
            
        # Calculate estimates
        point_estimate = np.mean(bootstrap_estimates)
        standard_error = np.std(bootstrap_estimates)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower, ci_upper = np.percentile(
            bootstrap_estimates, 
            [alpha/2 * 100, (1-alpha/2) * 100]
        )
        
        return BootstrapEstimates(
            point_estimate=point_estimate,
            confidence_interval=(ci_lower, ci_upper),
            standard_error=standard_error
        )

# Example usage with different statistics
np.random.seed(42)
data = np.random.lognormal(0, 0.5, 1000)
sampler = BootstrapSampler(n_iterations=10000)

# Define statistics to estimate
statistics = {
    'mean': np.mean,
    'median': np.median,
    'std': np.std,
    'skewness': stats.skew
}

for name, statistic in statistics.items():
    estimates = sampler.bootstrap_statistic(data, statistic)
    print(f"\n{name.upper()} Estimates:")
    print(f"Point estimate: {estimates.point_estimate:.4f}")
    print(f"95% CI: ({estimates.confidence_interval[0]:.4f}, "
          f"{estimates.confidence_interval[1]:.4f})")
    print(f"Standard error: {estimates.standard_error:.4f}")
```

Slide 13: Neural Network-Based Adaptive Sampling

This advanced implementation uses a neural network to learn optimal sampling strategies based on data characteristics and downstream task performance, automatically adjusting sampling rates for different data regions.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SamplingNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class AdaptiveNeuralSampler:
    def __init__(self, window_size: int = 10, hidden_dim: int = 64):
        self.window_size = window_size
        self.model = SamplingNetwork(window_size, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.BCELoss()
        
    def create_windows(self, data: np.ndarray) -> np.ndarray:
        """
        Create sliding windows from time series data
        """
        windows = []
        for i in range(len(data) - self.window_size + 1):
            windows.append(data[i:i + self.window_size])
        return np.array(windows)
    
    def train(self, data: np.ndarray, epochs: int = 100):
        """
        Train the sampling network
        """
        windows = self.create_windows(data)
        window_std = np.std(windows, axis=1)
        # Higher variance regions should be sampled more frequently
        labels = (window_std > np.median(window_std)).astype(np.float32)
        
        X = torch.FloatTensor(windows)
        y = torch.FloatTensor(labels)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs.squeeze(), y)
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    @torch.no_grad()
    def sample(self, data: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Perform adaptive sampling using trained network
        """
        windows = self.create_windows(data)
        X = torch.FloatTensor(windows)
        sampling_probs = self.model(X).numpy()
        
        # Select samples based on network output
        samples = []
        for i, prob in enumerate(sampling_probs):
            if prob > threshold:
                samples.append(data[i:i + self.window_size])
                
        return np.array(samples)

# Example usage
np.random.seed(42)
# Generate synthetic data with varying complexity
t = np.linspace(0, 10, 1000)
base_signal = np.sin(t) + 0.5 * np.sin(5 * t)
noise = np.random.normal(0, 0.1 + 0.2 * np.abs(np.sin(t)), len(t))
data = base_signal + noise

# Train and apply neural sampler
sampler = AdaptiveNeuralSampler(window_size=20)
sampler.train(data, epochs=50)
sampled_sequences = sampler.sample(data)

print(f"\nOriginal data length: {len(data)}")
print(f"Number of sampled sequences: {len(sampled_sequences)}")
print(f"Sampling ratio: {len(sampled_sequences)/len(data):.3f}")
```

Slide 14: Multi-objective Sampling Optimization

This implementation balances multiple objectives in the sampling process, including representativeness, diversity, and computational efficiency through Pareto optimization.

```python
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from sklearn.metrics import pairwise_distances

@dataclass
class SamplingObjectives:
    representativeness: float
    diversity: float
    efficiency: float

class MultiObjectiveSampler:
    def __init__(self, n_solutions: int = 100):
        self.n_solutions = n_solutions
        
    def calculate_objectives(self, data: np.ndarray, 
                           sample_indices: np.ndarray) -> SamplingObjectives:
        """
        Calculate multiple objective values for a sample
        """
        sample = data[sample_indices]
        
        # Representativeness: KL divergence approximation
        rep_score = -np.mean(
            np.abs(np.mean(sample, axis=0) - np.mean(data, axis=0))
        )
        
        # Diversity: average pairwise distance
        div_score = np.mean(pairwise_distances(sample))
        
        # Efficiency: inverse of sample size
        eff_score = 1.0 - (len(sample) / len(data))
        
        return SamplingObjectives(
            representativeness=rep_score,
            diversity=div_score,
            efficiency=eff_score
        )
    
    def dominates(self, obj1: SamplingObjectives, 
                 obj2: SamplingObjectives) -> bool:
        """
        Check if obj1 dominates obj2 in Pareto sense
        """
        return (obj1.representativeness >= obj2.representativeness and
                obj1.diversity >= obj2.diversity and
                obj1.efficiency >= obj2.efficiency and
                (obj1.representativeness > obj2.representativeness or
                 obj1.diversity > obj2.diversity or
                 obj1.efficiency > obj2.efficiency))
    
    def sample(self, data: np.ndarray, 
               sample_size_range: Tuple[int, int]) -> np.ndarray:
        """
        Find Pareto-optimal sampling solutions
        """
        solutions = []
        objectives = []
        
        for _ in range(self.n_solutions):
            # Generate random sample size within range
            size = np.random.randint(*sample_size_range)
            indices = np.random.choice(len(data), size=size, replace=False)
            obj = self.calculate_objectives(data, indices)
            
            # Check for Pareto dominance
            dominated = False
            i = 0
            while i < len(objectives):
                if self.dominates(objectives[i], obj):
                    dominated = True
                    break
                elif self.dominates(obj, objectives[i]):
                    del objectives[i]
                    del solutions[i]
                else:
                    i += 1
                    
            if not dominated:
                solutions.append(indices)
                objectives.append(obj)
        
        # Return best solution based on weighted sum
        weights = [0.4, 0.4, 0.2]  # Customize based on preferences
        scores = [
            weights[0] * obj.representativeness +
            weights[1] * obj.diversity +
            weights[2] * obj.efficiency
            for obj in objectives
        ]
        best_idx = np.argmax(scores)
        
        return data[solutions[best_idx]]

# Example usage
np.random.seed(42)
# Generate synthetic data with clusters
n_samples = 1000
data = np.concatenate([
    np.random.normal(0, 1, (n_samples//2, 2)),
    np.random.normal(3, 0.5, (n_samples//2, 2))
])

sampler = MultiObjectiveSampler(n_solutions=200)
sample = sampler.sample(data, sample_size_range=(100, 300))

print(f"Original data shape: {data.shape}")
print(f"Sampled data shape: {sample.shape}")
print(f"Sampling ratio: {len(sample)/len(data):.3f}")
```

Slide 15: Additional Resources

*   Modern Sampling Methods: An Overview - arXiv:2208.12386 [https://arxiv.org/abs/2208.12386](https://arxiv.org/abs/2208.12386)
*   Neural Adaptive Sampling Strategies - arXiv:2105.09734 [https://arxiv.org/abs/2105.09734](https://arxiv.org/abs/2105.09734)
*   Statistical Guarantees in Data Sampling - arXiv:2203.15560 [https://arxiv.org/abs/2203.15560](https://arxiv.org/abs/2203.15560)
*   Practical guide to sampling techniques in machine learning: [https://machinelearningmastery.com/statistical-sampling-and-resampling/](https://machinelearningmastery.com/statistical-sampling-and-resampling/)
*   Survey of Sampling Methods for Big Data Analytics: [https://www.sciencedirect.com/topics/computer-science/sampling-method](https://www.sciencedirect.com/topics/computer-science/sampling-method)

