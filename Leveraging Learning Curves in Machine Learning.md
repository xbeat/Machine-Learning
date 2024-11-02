## Leveraging Learning Curves in Machine Learning
Slide 1: Understanding Learning Curves

Learning curves are essential diagnostic tools in machine learning that plot model performance against training dataset size. They help identify underfitting, overfitting, and determine if additional training data would benefit model performance by showing the relationship between training and validation metrics.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression

def plot_learning_curve(estimator, X, y, title='Learning Curve'):
    # Generate learning curve data
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='accuracy', n_jobs=-1)
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, val_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
```

Slide 2: Implementing Cross-Validation for Learning Curves

Cross-validation provides more robust learning curve estimates by evaluating model performance across multiple data splits. This implementation demonstrates how to create learning curves using k-fold cross-validation to assess model stability and generalization.

```python
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def custom_learning_curve(model, X, y, train_sizes):
    n_samples = len(X)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_scores = []
    val_scores = []
    
    for size in train_sizes:
        size_scores_train = []
        size_scores_val = []
        
        # Perform k-fold CV for each training size
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Use only a subset of training data
            n_train = int(size * len(X_train_fold))
            X_train_subset = X_train_fold[:n_train]
            y_train_subset = y_train_fold[:n_train]
            
            # Train and evaluate
            model.fit(X_train_subset, y_train_subset)
            train_score = accuracy_score(y_train_subset, 
                                      model.predict(X_train_subset))
            val_score = accuracy_score(y_val_fold, 
                                     model.predict(X_val_fold))
            
            size_scores_train.append(train_score)
            size_scores_val.append(val_score)
        
        train_scores.append(np.mean(size_scores_train))
        val_scores.append(np.mean(size_scores_val))
    
    return np.array(train_scores), np.array(val_scores)
```

Slide 3: Detecting Overfitting and Underfitting

Learning curves provide visual indicators of model bias and variance problems. This implementation focuses on analyzing the gap between training and validation curves to identify overfitting (high variance) and underfitting (high bias) scenarios.

```python
def analyze_learning_curve(train_scores, val_scores, threshold=0.1):
    gap = np.abs(train_scores - val_scores)
    final_gap = gap[-1]
    final_val_score = val_scores[-1]
    
    # Calculate score trends
    train_trend = np.gradient(train_scores)
    val_trend = np.gradient(val_scores)
    
    analysis = {
        'overfitting': final_gap > threshold and train_scores[-1] > val_scores[-1],
        'underfitting': final_val_score < 0.6 and final_gap < threshold,
        'good_fit': final_gap < threshold and final_val_score > 0.8,
        'needs_more_data': val_trend[-1] > 0.01  # Still improving
    }
    
    return analysis, {
        'final_gap': final_gap,
        'final_val_score': final_val_score,
        'train_trend': train_trend[-1],
        'val_trend': val_trend[-1]
    }
```

Slide 4: Dynamic Learning Rate Analysis

Understanding how learning rate affects model convergence through learning curves helps optimize training. This implementation shows how to analyze learning rate impact on model performance using dynamic learning rate schedules.

```python
def analyze_learning_rates(X, y, learning_rates=[0.1, 0.01, 0.001]):
    results = {}
    
    for lr in learning_rates:
        model = LogisticRegression(learning_rate_init=lr, max_iter=1000)
        
        # Generate learning curves for each learning rate
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores, val_scores = custom_learning_curve(
            model, X, y, train_sizes)
        
        # Calculate convergence metrics
        convergence_speed = np.mean(np.gradient(val_scores))
        stability = np.std(val_scores)
        
        results[lr] = {
            'convergence_speed': convergence_speed,
            'stability': stability,
            'final_score': val_scores[-1],
            'train_scores': train_scores,
            'val_scores': val_scores
        }
    
    return results
```

Slide 5: Mathematical Foundations of Learning Curves

Learning curves are grounded in statistical learning theory, particularly the bias-variance tradeoff. The theoretical error can be decomposed into bias and variance components, helping understand model behavior as training size increases.

```python
def theoretical_learning_curve():
    # Example showing theoretical components of learning curve
    n_points = 100
    training_sizes = np.linspace(10, 1000, n_points)
    
    # Theoretical components
    bias_term = 0.3  # Constant bias
    variance = 2.0 / training_sizes  # Variance decreases with more data
    noise = 0.1 * np.ones(n_points)  # Irreducible error
    
    # Total error calculation
    total_error = bias_term + variance + noise
    
    # Code showing mathematical formulas (not rendered)
    """
    $$E_{total} = E_{bias}^2 + E_{variance} + \sigma^2$$
    $$E_{variance} \propto \frac{1}{n}$$
    $$E_{bias} = constant$$
    """
    
    return {
        'training_sizes': training_sizes,
        'bias': bias_term * np.ones(n_points),
        'variance': variance,
        'noise': noise,
        'total_error': total_error
    }
```

Slide 6: Real-world Implementation: Credit Risk Assessment

This implementation demonstrates learning curve analysis for a credit risk prediction model, including data preprocessing, model training, and comprehensive performance evaluation across different training dataset sizes.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def credit_risk_analysis(data_path):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    X = df.drop('default', axis=1)
    y = df['default']
    
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    
    # Initialize model
    model = LogisticRegression(random_state=42)
    
    # Generate learning curves
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores, val_scores = custom_learning_curve(
        model, X_train, y_train, train_sizes)
    
    # Calculate performance metrics
    metrics = {
        'convergence_rate': np.mean(np.gradient(val_scores)),
        'final_accuracy': val_scores[-1],
        'score_stability': np.std(val_scores),
        'training_efficiency': len(X_train) / np.where(
            np.gradient(val_scores) < 0.01)[0][0] 
            if len(np.where(np.gradient(val_scores) < 0.01)[0]) > 0 else None
    }
    
    return train_scores, val_scores, metrics
```

Slide 7: Advanced Visualization Techniques

Advanced visualization techniques help extract deeper insights from learning curves by incorporating confidence intervals, trend analysis, and comparative metrics across different model architectures.

```python
def advanced_learning_curve_visualization(models_results):
    plt.figure(figsize=(15, 10))
    colors = ['blue', 'red', 'green', 'purple']
    
    for (model_name, results), color in zip(models_results.items(), colors):
        train_sizes = results['train_sizes']
        train_mean = results['train_scores'].mean(axis=1)
        train_std = results['train_scores'].std(axis=1)
        val_mean = results['val_scores'].mean(axis=1)
        val_std = results['val_scores'].std(axis=1)
        
        # Plot means
        plt.plot(train_sizes, train_mean, f'{color}--', 
                label=f'{model_name} (train)')
        plt.plot(train_sizes, val_mean, f'{color}-', 
                label=f'{model_name} (val)')
        
        # Plot confidence intervals
        plt.fill_between(train_sizes, 
                        train_mean - 2*train_std,
                        train_mean + 2*train_std, 
                        color=color, alpha=0.1)
        plt.fill_between(train_sizes, 
                        val_mean - 2*val_std,
                        val_mean + 2*val_std, 
                        color=color, alpha=0.1)
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Performance Score')
    plt.title('Comparative Learning Curves Analysis')
    plt.legend(loc='best')
    plt.grid(True)
    
    return plt.gcf()
```

Slide 8: Statistical Significance Testing

Statistical analysis of learning curves helps determine if performance differences are significant and not due to random variation. This implementation includes methods for confidence interval calculation and hypothesis testing.

```python
from scipy import stats
import numpy as np

def analyze_curve_significance(curve1, curve2, alpha=0.05):
    # Calculate confidence intervals and perform statistical tests
    def bootstrap_curve(curve, n_bootstrap=1000):
        bootstrap_samples = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(curve), size=len(curve))
            bootstrap_samples.append(np.mean(curve[indices]))
        return np.array(bootstrap_samples)
    
    # Perform t-test between curves
    t_stat, p_value = stats.ttest_ind(curve1, curve2)
    
    # Bootstrap confidence intervals
    boot1 = bootstrap_curve(curve1)
    boot2 = bootstrap_curve(curve2)
    
    ci1 = np.percentile(boot1, [2.5, 97.5])
    ci2 = np.percentile(boot2, [2.5, 97.5])
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'confidence_intervals': {
            'curve1': ci1,
            'curve2': ci2
        },
        'effect_size': (np.mean(curve1) - np.mean(curve2)) / np.sqrt(
            (np.var(curve1) + np.var(curve2)) / 2)
    }
```

Slide 9: Data Efficiency Analysis

This implementation focuses on analyzing how efficiently models learn from data, helping determine optimal dataset sizes and identify diminishing returns in model performance improvement.

```python
def analyze_data_efficiency(train_sizes, val_scores):
    # Calculate incremental improvements
    score_improvements = np.diff(val_scores)
    data_increases = np.diff(train_sizes)
    efficiency_ratio = score_improvements / data_increases
    
    # Find point of diminishing returns
    threshold = 0.001  # Minimum acceptable improvement per data point
    diminishing_returns_idx = np.where(efficiency_ratio < threshold)[0]
    optimal_size = train_sizes[diminishing_returns_idx[0]] if len(
        diminishing_returns_idx) > 0 else train_sizes[-1]
    
    # Calculate data utilization metrics
    auc = np.trapz(val_scores, train_sizes) / (
        train_sizes[-1] * val_scores[-1])
    learning_speed = np.mean(efficiency_ratio[:5])  # Early learning rate
    
    return {
        'optimal_dataset_size': optimal_size,
        'data_utilization_efficiency': auc,
        'early_learning_speed': learning_speed,
        'efficiency_curve': efficiency_ratio,
        'diminishing_returns_point': diminishing_returns_idx[0] if len(
            diminishing_returns_idx) > 0 else None
    }
```

Slide 10: Real-world Implementation: Image Classification

A comprehensive implementation showing learning curve analysis for deep learning image classification, including data augmentation effects and model complexity considerations.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def analyze_cnn_learning_curves(X_train, y_train, input_shape, n_classes):
    # Define model architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    # Training with different dataset sizes
    train_sizes = np.linspace(0.1, 1.0, 5)
    results = []
    
    for size in train_sizes:
        n_samples = int(len(X_train) * size)
        X_subset = X_train[:n_samples]
        y_subset = y_train[:n_samples]
        
        history = model.fit(
            datagen.flow(X_subset, y_subset, batch_size=32),
            epochs=10,
            validation_split=0.2,
            verbose=0
        )
        
        results.append({
            'size': n_samples,
            'train_acc': history.history['accuracy'][-1],
            'val_acc': history.history['val_accuracy'][-1]
        })
    
    return pd.DataFrame(results)
```

Slide 11: Early Stopping Analysis with Learning Curves

Early stopping analysis using learning curves helps prevent overfitting by monitoring validation performance trends. This implementation provides sophisticated stopping criteria based on curve derivatives and statistical tests.

```python
def analyze_early_stopping(val_scores, patience=5, min_delta=0.001):
    def calculate_trend(scores, window=3):
        return np.convolve(np.gradient(scores), 
                          np.ones(window)/window, 
                          mode='valid')
    
    # Calculate moving derivatives
    score_trend = calculate_trend(val_scores)
    acceleration = calculate_trend(score_trend)
    
    # Detect plateau and overfitting signals
    plateau_mask = np.abs(score_trend) < min_delta
    overfitting_mask = score_trend < -min_delta
    
    # Find optimal stopping point
    counter = 0
    stopping_epoch = len(val_scores)
    
    for i, (is_plateau, is_overfitting) in enumerate(
            zip(plateau_mask, overfitting_mask)):
        if is_plateau or is_overfitting:
            counter += 1
        else:
            counter = 0
            
        if counter >= patience:
            stopping_epoch = i - patience + 1
            break
    
    return {
        'optimal_epoch': stopping_epoch,
        'final_score': val_scores[stopping_epoch],
        'improvement_rate': score_trend,
        'acceleration': acceleration,
        'stopped_early': stopping_epoch < len(val_scores)
    }
```

Slide 12: Comparative Model Analysis

Implementation for comparing learning curves across different model architectures, helping identify the most efficient model for a given dataset size and computational budget.

```python
def compare_model_efficiency(models, X, y, cv_splits=5):
    from sklearn.model_selection import cross_validate
    
    def calculate_efficiency_metrics(scores, train_times):
        return {
            'score_mean': np.mean(scores),
            'score_std': np.std(scores),
            'time_per_score': np.mean(train_times) / np.mean(scores),
            'efficiency_index': np.mean(scores) / np.log(
                np.mean(train_times) + 1)
        }
    
    results = {}
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    for name, model in models.items():
        size_results = []
        
        for size in train_sizes:
            n_samples = int(len(X) * size)
            X_subset = X[:n_samples]
            y_subset = y[:n_samples]
            
            cv_results = cross_validate(
                model, X_subset, y_subset,
                cv=cv_splits,
                return_train_score=True,
                n_jobs=-1
            )
            
            metrics = calculate_efficiency_metrics(
                cv_results['test_score'],
                cv_results['fit_time']
            )
            metrics['train_size'] = n_samples
            size_results.append(metrics)
            
        results[name] = pd.DataFrame(size_results)
    
    return results
```

Slide 13: Additional Resources

*   [https://arxiv.org/abs/1712.06559](https://arxiv.org/abs/1712.06559) - "Deep Learning Scaling is Predictable, Empirically" 
*   [https://arxiv.org/abs/1606.04838](https://arxiv.org/abs/1606.04838) - "Learning Curves for Deep Neural Networks" 
*   [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361) - "Reconciling Modern Machine Learning Practice and the Bias-Variance Trade-Off" [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946) - "Learning Rate Annealing Methods for Neural Networks: A Practical Study"
*   [https://arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709) - "A Statistical Theory of Learning Curves"


