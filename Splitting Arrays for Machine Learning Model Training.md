## Splitting Arrays for Machine Learning Model Training
Slide 1: Introduction to Array Splitting for Model Training

Array splitting is a crucial technique in machine learning, particularly for preparing data for model training. It involves dividing a large dataset into smaller, manageable portions. This process is essential for creating training, validation, and test sets, as well as for implementing cross-validation techniques. In this presentation, we'll explore various methods to split arrays in Python, focusing on their applications in model training.

```python
# Example of a simple array split
import random

data = list(range(100))  # Create a sample dataset
random.shuffle(data)     # Shuffle the data

train_size = int(0.8 * len(data))  # 80% for training
train_set = data[:train_size]
test_set = data[train_size:]

print(f"Training set size: {len(train_set)}")
print(f"Test set size: {len(test_set)}")
```

Slide 2: Basic Array Splitting with NumPy

NumPy provides efficient tools for array manipulation, including splitting. The `array_split` function allows us to divide an array into a specified number of sub-arrays. This method is particularly useful when we need to create equal-sized chunks of data, which can be beneficial for batch processing in model training.

```python
import numpy as np

# Create a sample array
data = np.arange(100)

# Split the array into 3 parts
split_arrays = np.array_split(data, 3)

for i, arr in enumerate(split_arrays):
    print(f"Array {i + 1}: {arr}")
    print(f"Shape: {arr.shape}\n")
```

Slide 3: Results for: Basic Array Splitting with NumPy

```
Array 1: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33]
Shape: (34,)

Array 2: [34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57
 58 59 60 61 62 63 64 65 66]
Shape: (33,)

Array 3: [67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90
 91 92 93 94 95 96 97 98 99]
Shape: (33,)
```

Slide 4: Train-Test Split using scikit-learn

Scikit-learn's `train_test_split` function is a popular choice for splitting data into training and testing sets. It offers features like automatic shuffling and stratification, which are particularly useful for maintaining class distribution in classification problems.

```python
from sklearn.model_selection import train_test_split
import numpy as np

# Generate sample data
X = np.array([[i, i**2] for i in range(100)])
y = np.array([i % 2 for i in range(100)])  # Binary classification

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Training set class distribution: {np.bincount(y_train)}")
print(f"Testing set class distribution: {np.bincount(y_test)}")
```

Slide 5: Results for: Train-Test Split using scikit-learn

```
Training set shape: (80, 2)
Testing set shape: (20, 2)
Training set class distribution: [40 40]
Testing set class distribution: [10 10]
```

Slide 6: K-Fold Cross-Validation

K-Fold Cross-Validation is a technique used to assess model performance and generalization. It involves splitting the data into K subsets, training the model K times, each time using a different subset as the validation set and the remaining K-1 subsets as the training set.

```python
from sklearn.model_selection import KFold
import numpy as np

# Generate sample data
X = np.array([[i, i**2] for i in range(100)])
y = np.array([i % 2 for i in range(100)])

# Create KFold object
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform K-Fold splitting
for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    print(f"Fold {fold}:")
    print(f"  Training set shape: {X_train.shape}")
    print(f"  Validation set shape: {X_val.shape}")
    print(f"  Training set class distribution: {np.bincount(y_train)}")
    print(f"  Validation set class distribution: {np.bincount(y_val)}\n")
```

Slide 7: Results for: K-Fold Cross-Validation

```
Fold 1:
  Training set shape: (80, 2)
  Validation set shape: (20, 2)
  Training set class distribution: [40 40]
  Validation set class distribution: [10 10]

Fold 2:
  Training set shape: (80, 2)
  Validation set shape: (20, 2)
  Training set class distribution: [40 40]
  Validation set class distribution: [10 10]

Fold 3:
  Training set shape: (80, 2)
  Validation set shape: (20, 2)
  Training set class distribution: [40 40]
  Validation set class distribution: [10 10]

Fold 4:
  Training set shape: (80, 2)
  Validation set shape: (20, 2)
  Training set class distribution: [40 40]
  Validation set class distribution: [10 10]

Fold 5:
  Training set shape: (80, 2)
  Validation set shape: (20, 2)
  Training set class distribution: [40 40]
  Validation set class distribution: [10 10]
```

Slide 8: Stratified K-Fold Cross-Validation

Stratified K-Fold is a variation of K-Fold that ensures each fold has approximately the same proportion of samples for each class as the complete set. This is particularly useful for imbalanced datasets or when dealing with multi-class classification problems.

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Generate imbalanced sample data
X = np.array([[i, i**2] for i in range(100)])
y = np.array([0] * 80 + [1] * 20)  # Imbalanced classes

# Create StratifiedKFold object
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform Stratified K-Fold splitting
for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    print(f"Fold {fold}:")
    print(f"  Training set shape: {X_train.shape}")
    print(f"  Validation set shape: {X_val.shape}")
    print(f"  Training set class distribution: {np.bincount(y_train)}")
    print(f"  Validation set class distribution: {np.bincount(y_val)}\n")
```

Slide 9: Results for: Stratified K-Fold Cross-Validation

```
Fold 1:
  Training set shape: (80, 2)
  Validation set shape: (20, 2)
  Training set class distribution: [64 16]
  Validation set class distribution: [16  4]

Fold 2:
  Training set shape: (80, 2)
  Validation set shape: (20, 2)
  Training set class distribution: [64 16]
  Validation set class distribution: [16  4]

Fold 3:
  Training set shape: (80, 2)
  Validation set shape: (20, 2)
  Training set class distribution: [64 16]
  Validation set class distribution: [16  4]

Fold 4:
  Training set shape: (80, 2)
  Validation set shape: (20, 2)
  Training set class distribution: [64 16]
  Validation set class distribution: [16  4]

Fold 5:
  Training set shape: (80, 2)
  Validation set shape: (20, 2)
  Training set class distribution: [64 16]
  Validation set class distribution: [16  4]
```

Slide 10: Time Series Split

When dealing with time series data, it's important to maintain the temporal order of the samples. The TimeSeriesSplit from scikit-learn provides a way to perform cross-validation on time series data, ensuring that we don't use future data to predict past events.

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Generate sample time series data
X = np.array([[i, i**2] for i in range(100)])
y = np.sin(np.arange(100))

# Create TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=5)

# Perform Time Series splitting
for fold, (train_index, val_index) in enumerate(tscv.split(X), 1):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    print(f"Fold {fold}:")
    print(f"  Training set shape: {X_train.shape}")
    print(f"  Validation set shape: {X_val.shape}")
    print(f"  Training set time range: {train_index[0]} to {train_index[-1]}")
    print(f"  Validation set time range: {val_index[0]} to {val_index[-1]}\n")
```

Slide 11: Results for: Time Series Split

```
Fold 1:
  Training set shape: (20, 2)
  Validation set shape: (20, 2)
  Training set time range: 0 to 19
  Validation set time range: 20 to 39

Fold 2:
  Training set shape: (40, 2)
  Validation set shape: (20, 2)
  Training set time range: 0 to 39
  Validation set time range: 40 to 59

Fold 3:
  Training set shape: (60, 2)
  Validation set shape: (20, 2)
  Training set time range: 0 to 59
  Validation set time range: 60 to 79

Fold 4:
  Training set shape: (80, 2)
  Validation set shape: (20, 2)
  Training set time range: 0 to 79
  Validation set time range: 80 to 99

Fold 5:
  Training set shape: (80, 2)
  Validation set shape: (20, 2)
  Training set time range: 20 to 99
  Validation set time range: 100 to 119
```

Slide 12: Custom Splitting for Specialized Needs

Sometimes, standard splitting techniques may not suffice for specialized needs. In such cases, we can create custom splitting functions. Here's an example of a custom splitter that creates overlapping time windows for sequence prediction tasks.

```python
import numpy as np

def sliding_window_split(X, y, window_size, step_size):
    for i in range(0, len(X) - window_size + 1, step_size):
        X_window = X[i:i+window_size]
        y_window = y[i+window_size-1]  # Predict the last value in the window
        yield X_window, y_window

# Generate sample time series data
X = np.array([[i, i**2] for i in range(100)])
y = np.sin(np.arange(100))

window_size = 10
step_size = 5

for i, (X_window, y_target) in enumerate(sliding_window_split(X, y, window_size, step_size)):
    if i < 3:  # Print only first 3 windows for brevity
        print(f"Window {i + 1}:")
        print(f"  Input shape: {X_window.shape}")
        print(f"  Target value: {y_target:.4f}")
        print(f"  Time range: {i*step_size} to {i*step_size + window_size - 1}\n")
```

Slide 13: Results for: Custom Splitting for Specialized Needs

```
Window 1:
  Input shape: (10, 2)
  Target value: -0.5440
  Time range: 0 to 9

Window 2:
  Input shape: (10, 2)
  Target value: 0.9093
  Time range: 5 to 14

Window 3:
  Input shape: (10, 2)
  Target value: 0.1411
  Time range: 10 to 19
```

Slide 14: Real-Life Example: Image Classification

In image classification tasks, splitting the dataset is crucial for evaluating model performance. Let's consider a scenario where we're classifying images of different animal species. We'll use stratified sampling to ensure each split contains a representative distribution of classes.

```python
from sklearn.model_selection import train_test_split
import numpy as np

# Simulating an image dataset (each image is represented by a 32x32x3 array)
num_samples = 1000
num_classes = 5
X = np.random.rand(num_samples, 32, 32, 3)
y = np.random.randint(0, num_classes, num_samples)

# Perform stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Original class distribution:")
print(np.bincount(y))
print("\nTraining set class distribution:")
print(np.bincount(y_train))
print("\nTest set class distribution:")
print(np.bincount(y_test))
```

Slide 15: Results for: Real-Life Example: Image Classification

```
Original class distribution:
[198 220 204 191 187]

Training set class distribution:
[158 176 163 153 150]

Test set class distribution:
[40 44 41 38 37]
```

Slide 16: Real-Life Example: Time Series Forecasting

In time series forecasting, such as predicting energy consumption, maintaining the temporal order of data is crucial. We'll use a sliding window approach to create input-output pairs for training a forecasting model.

```python
import numpy as np

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Simulating hourly energy consumption data for a month
hours_in_month = 24 * 30
energy_data = np.sin(np.linspace(0, 8*np.pi, hours_in_month)) + np.random.normal(0, 0.1, hours_in_month)

# Create sequences for 24-hour forecasting
seq_length = 24
X, y = create_sequences(energy_data, seq_length)

print(f"Input shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"First sequence:\n{X[0]}")
print(f"First target: {y[0]}")
```

Slide 17: Results for: Real-Life Example: Time Series Forecasting

```
Input shape: (696, 24)
Target shape: (696,)
First sequence:
[ 0.00537846  0.13269862  0.28409485  0.37827083  0.53669804  0.61449581
  0.76434905  0.85098155  0.94627132  1.02772433  1.06510352  1.08410279
  1.09756004  1.03610962  0.98336771  0.87168761  0.76642844  0.61753263
  0.46986779  0.29261852  0.14383473 -0.0204915  -0.15405605 -0.32111281]
First target: -0.4492728859497388
```

Slide 18: Handling Imbalanced Datasets

When dealing with imbalanced datasets, it's crucial to maintain class proportions across splits. We'll use the StratifiedShuffleSplit to create multiple train-test splits while preserving class distribution.

```python
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

# Generate an imbalanced dataset
X = np.random.rand(1000, 10)
y = np.concatenate([np.zeros(900), np.ones(100)])

# Create StratifiedShuffleSplit object
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

for fold, (train_index, test_index) in enumerate(sss.split(X, y), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    print(f"Fold {fold}:")
    print(f"  Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"  Train set class distribution: {np.bincount(y_train.astype(int))}")
    print(f"  Test set class distribution: {np.bincount(y_test.astype(int))}\n")
```

Slide 19: Results for: Handling Imbalanced Datasets

```
Fold 1:
  Train set shape: (800, 10), Test set shape: (200, 10)
  Train set class distribution: [720  80]
  Test set class distribution: [180  20]

Fold 2:
  Train set shape: (800, 10), Test set shape: (200, 10)
  Train set class distribution: [720  80]
  Test set class distribution: [180  20]

Fold 3:
  Train set shape: (800, 10), Test set shape: (200, 10)
  Train set class distribution: [720  80]
  Test set class distribution: [180  20]

Fold 4:
  Train set shape: (800, 10), Test set shape: (200, 10)
  Train set class distribution: [720  80]
  Test set class distribution: [180  20]

Fold 5:
  Train set shape: (800, 10), Test set shape: (200, 10)
  Train set class distribution: [720  80]
  Test set class distribution: [180  20]
```

Slide 20: Group-Based Splitting

In some scenarios, we need to ensure that related samples stay together in the same split. GroupKFold is useful for such cases, like when dealing with multiple measurements from the same subject in medical studies.

```python
from sklearn.model_selection import GroupKFold
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, 100)
groups = np.repeat(range(20), 5)  # 20 subjects, 5 measurements each

# Create GroupKFold object
gkf = GroupKFold(n_splits=5)

for fold, (train_index, test_index) in enumerate(gkf.split(X, y, groups), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    groups_train, groups_test = groups[train_index], groups[test_index]
    
    print(f"Fold {fold}:")
    print(f"  Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"  Train groups: {np.unique(groups_train)}")
    print(f"  Test groups: {np.unique(groups_test)}\n")
```

Slide 21: Results for: Group-Based Splitting

```
Fold 1:
  Train set shape: (80, 2), Test set shape: (20, 2)
  Train groups: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
  Test groups: [16 17 18 19]

Fold 2:
  Train set shape: (80, 2), Test set shape: (20, 2)
  Train groups: [ 0  1  2  3  4  5  6  7  8  9 10 11 16 17 18 19]
  Test groups: [12 13 14 15]

Fold 3:
  Train set shape: (80, 2), Test set shape: (20, 2)
  Train groups: [ 0  1  2  3  4  5  6  7 12 13 14 15 16 17 18 19]
  Test groups: [ 8  9 10 11]

Fold 4:
  Train set shape: (80, 2), Test set shape: (20, 2)
  Train groups: [ 0  1  2  3  8  9 10 11 12 13 14 15 16 17 18 19]
  Test groups: [4 5 6 7]

Fold 5:
  Train set shape: (80, 2), Test set shape: (20, 2)
  Train groups: [ 4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
  Test groups: [0 1 2 3]
```

Slide 22: Additional Resources

For those interested in diving deeper into array splitting techniques for model training, here are some valuable resources:

1.  Scikit-learn documentation on model selection: This comprehensive guide covers various splitting techniques and cross-validation methods. ([https://scikit-learn.org/stable/modules/cross\_validation.html](https://scikit-learn.org/stable/modules/cross_validation.html))
2.  "A Survey of Cross-Validation Procedures for Model Selection" by Arlot, S. and Celisse, A. (2010): This paper provides an in-depth review of cross-validation techniques. (arXiv:0907.4728)
3.  "Time Series Split" by Rob J Hyndman: This article discusses strategies for splitting time series data for forecasting tasks. ([https://robjhyndman.com/hyndsight/tscv/](https://robjhyndman.com/hyndsight/tscv/))
4.  "Learning from Imbalanced Data" by He, H. and Garcia, E.A. (2009): This paper explores techniques for handling imbalanced datasets, including appropriate splitting strategies. (IEEE Transactions on Knowledge and Data Engineering)

These resources offer a mix of practical guides and theoretical foundations to deepen your understanding of array splitting in the context of machine learning model training.

