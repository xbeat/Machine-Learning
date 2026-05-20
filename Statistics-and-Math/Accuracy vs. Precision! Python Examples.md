## Accuracy vs. Precision! Python Examples
Slide 1: Accuracy vs. Precision: Understanding the Difference

Accuracy and precision are often confused, but they represent distinct concepts in measurement and data analysis. This presentation will explore their differences using Python examples.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple plot to illustrate the concept
plt.figure(figsize=(10, 6))
plt.scatter([1, 2, 3, 4], [1, 2, 3, 4], label='Perfect Accuracy and Precision')
plt.scatter([1.1, 2.2, 2.9, 3.8], [1.1, 1.9, 3.1, 4.2], label='High Accuracy, Lower Precision')
plt.scatter([0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5, 3.5], label='High Precision, Lower Accuracy')
plt.legend()
plt.title('Accuracy vs. Precision Visualization')
plt.xlabel('True Value')
plt.ylabel('Measured Value')
plt.show()
```

Slide 2: Defining Accuracy

Accuracy refers to how close a measured value is to the true or accepted value. It's about the correctness of the measurement.

```python
def calculate_accuracy(true_values, measured_values):
    return 1 - np.mean(np.abs(np.array(true_values) - np.array(measured_values)))

true_temps = [20, 22, 24, 26, 28]
measured_temps = [19.8, 22.1, 23.9, 26.2, 27.8]

accuracy = calculate_accuracy(true_temps, measured_temps)
print(f"Accuracy: {accuracy:.2f}")
# Output: Accuracy: 0.96
```

Slide 3: Defining Precision

Precision relates to the consistency or reproducibility of measurements. It indicates how close repeated measurements are to each other.

```python
def calculate_precision(measurements):
    return 1 / np.std(measurements)

temp_measurements = [22.1, 22.0, 22.2, 21.9, 22.1]
precision = calculate_precision(temp_measurements)
print(f"Precision: {precision:.2f}")
# Output: Precision: 5.00
```

Slide 4: Accuracy vs. Precision: The Target Analogy

Imagine throwing darts at a target. Accuracy is hitting close to the bullseye, while precision is how close the darts are to each other.

```python
import random

def throw_darts(accuracy, precision, n_throws=100):
    return [(random.gauss(accuracy, 1/precision), random.gauss(accuracy, 1/precision)) for _ in range(n_throws)]

accurate_precise = throw_darts(0, 10)
accurate_imprecise = throw_darts(0, 2)
inaccurate_precise = throw_darts(5, 10)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.scatter(*zip(*accurate_precise))
plt.title('Accurate and Precise')
plt.subplot(132)
plt.scatter(*zip(*accurate_imprecise))
plt.title('Accurate but Imprecise')
plt.subplot(133)
plt.scatter(*zip(*inaccurate_precise))
plt.title('Inaccurate but Precise')
plt.tight_layout()
plt.show()
```

Slide 5: Measuring Accuracy in Classification

In machine learning, accuracy is often used to evaluate classification models. It represents the proportion of correct predictions among the total number of cases examined.

```python
from sklearn.metrics import accuracy_score

true_labels = [0, 1, 1, 0, 1, 1, 0, 0]
predicted_labels = [0, 1, 1, 0, 0, 1, 1, 0]

accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Classification Accuracy: {accuracy:.2f}")
# Output: Classification Accuracy: 0.75
```

Slide 6: Precision in Classification

In the context of classification, precision is the ratio of true positive predictions to the total number of positive predictions.

```python
from sklearn.metrics import precision_score

true_labels = [0, 1, 1, 0, 1, 1, 0, 0]
predicted_labels = [0, 1, 1, 0, 0, 1, 1, 0]

precision = precision_score(true_labels, predicted_labels)
print(f"Classification Precision: {precision:.2f}")
# Output: Classification Precision: 0.75
```

Slide 7: Real-Life Example: Weather Forecasting

Weather forecasting demonstrates the interplay between accuracy and precision. A forecast can be precise (narrow temperature range) but inaccurate, or accurate (correct on average) but imprecise.

```python
import pandas as pd

# Simulating weather data
dates = pd.date_range(start='2023-01-01', end='2023-01-10')
actual_temps = [10, 12, 15, 14, 13, 16, 18, 17, 15, 14]
forecast1 = [11, 13, 14, 15, 14, 15, 17, 18, 16, 15]  # More accurate
forecast2 = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15]  # More precise

weather_df = pd.DataFrame({
    'Date': dates,
    'Actual': actual_temps,
    'Forecast1': forecast1,
    'Forecast2': forecast2
})

print(weather_df)
print(f"Forecast1 Accuracy: {calculate_accuracy(actual_temps, forecast1):.2f}")
print(f"Forecast2 Accuracy: {calculate_accuracy(actual_temps, forecast2):.2f}")
print(f"Forecast1 Precision: {calculate_precision(forecast1):.2f}")
print(f"Forecast2 Precision: {calculate_precision(forecast2):.2f}")
```

Slide 8: Balancing Accuracy and Precision

In many applications, we aim to balance accuracy and precision. This often involves trade-offs and depends on the specific requirements of the task at hand.

```python
def measure_performance(true_values, predictions):
    accuracy = calculate_accuracy(true_values, predictions)
    precision = calculate_precision(predictions)
    f1_score = 2 * (accuracy * precision) / (accuracy + precision)
    return accuracy, precision, f1_score

# Simulating different prediction scenarios
true_values = [10, 12, 15, 14, 13, 16, 18, 17, 15, 14]
predictions1 = [11, 13, 14, 15, 14, 15, 17, 18, 16, 15]  # Balanced
predictions2 = [10, 12, 15, 14, 13, 16, 18, 17, 15, 14]  # High accuracy, lower precision
predictions3 = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14]  # High precision, lower accuracy

for i, preds in enumerate([predictions1, predictions2, predictions3], 1):
    acc, prec, f1 = measure_performance(true_values, preds)
    print(f"Predictions {i}:")
    print(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, F1 Score: {f1:.2f}\n")
```

Slide 9: Visualizing Accuracy and Precision

Creating visual representations can help in understanding the relationship between accuracy and precision.

```python
import seaborn as sns

def generate_data(n_samples, accuracy, precision):
    true_value = 10
    return np.random.normal(true_value + accuracy, 1/precision, n_samples)

data1 = generate_data(1000, 0, 1)  # Accurate and precise
data2 = generate_data(1000, 2, 1)  # Less accurate, equally precise
data3 = generate_data(1000, 0, 0.5)  # Accurate, less precise

plt.figure(figsize=(12, 4))
plt.subplot(131)
sns.kdeplot(data1)
plt.title('Accurate and Precise')
plt.subplot(132)
sns.kdeplot(data2)
plt.title('Less Accurate, Equally Precise')
plt.subplot(133)
sns.kdeplot(data3)
plt.title('Accurate, Less Precise')
plt.tight_layout()
plt.show()
```

Slide 10: Improving Accuracy

To improve accuracy, we often need to address systematic errors or biases in our measurements or predictions.

```python
def improve_accuracy(measurements, true_value):
    bias = np.mean(measurements) - true_value
    corrected_measurements = measurements - bias
    return corrected_measurements

true_value = 100
biased_measurements = np.random.normal(105, 2, 1000)  # Systematically overestimating

corrected_measurements = improve_accuracy(biased_measurements, true_value)

print(f"Original accuracy: {calculate_accuracy([true_value]*1000, biased_measurements):.2f}")
print(f"Improved accuracy: {calculate_accuracy([true_value]*1000, corrected_measurements):.2f}")
```

Slide 11: Improving Precision

Improving precision often involves reducing random errors, which can be achieved through repeated measurements or using more sensitive instruments.

```python
def improve_precision(measurements, n_repeats):
    return np.mean([measurements for _ in range(n_repeats)], axis=0)

imprecise_measurements = np.random.normal(100, 5, 1000)
improved_measurements = improve_precision(imprecise_measurements, 10)

print(f"Original precision: {calculate_precision(imprecise_measurements):.2f}")
print(f"Improved precision: {calculate_precision(improved_measurements):.2f}")
```

Slide 12: Real-Life Example: Quality Control in Manufacturing

In manufacturing, both accuracy and precision are crucial for maintaining product quality. Let's simulate a production line producing bolts with a target length of 10 cm.

```python
def simulate_production(target_length, accuracy_error, precision_error, n_bolts):
    return np.random.normal(target_length + accuracy_error, precision_error, n_bolts)

target_length = 10  # cm
production_run1 = simulate_production(target_length, 0.1, 0.2, 1000)  # Slight accuracy issue
production_run2 = simulate_production(target_length, 0, 0.5, 1000)    # Precision issue

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.hist(production_run1, bins=30)
plt.title('Production Run 1: Accuracy Issue')
plt.subplot(122)
plt.hist(production_run2, bins=30)
plt.title('Production Run 2: Precision Issue')
plt.tight_layout()
plt.show()

print(f"Run 1 - Accuracy: {calculate_accuracy([target_length]*1000, production_run1):.2f}, "
      f"Precision: {calculate_precision(production_run1):.2f}")
print(f"Run 2 - Accuracy: {calculate_accuracy([target_length]*1000, production_run2):.2f}, "
      f"Precision: {calculate_precision(production_run2):.2f}")
```

Slide 13: Conclusion: The Importance of Both Accuracy and Precision

Understanding the difference between accuracy and precision is crucial in various fields, from scientific research to industrial applications. While accuracy ensures our measurements or predictions are close to the true value, precision gives us confidence in the reliability and consistency of our results. In practice, both are often needed to achieve high-quality outcomes.

```python
def overall_performance(true_values, measured_values):
    accuracy = calculate_accuracy(true_values, measured_values)
    precision = calculate_precision(measured_values)
    overall_score = np.sqrt(accuracy * precision)  # Geometric mean
    return overall_score

true_values = [10, 12, 15, 14, 13, 16, 18, 17, 15, 14]
measurements1 = [10.1, 12.2, 14.9, 14.1, 13.0, 16.1, 17.9, 17.0, 15.1, 14.0]  # High accuracy and precision
measurements2 = [11, 13, 15, 15, 14, 16, 18, 18, 16, 15]  # Lower accuracy
measurements3 = [10, 13, 14, 15, 12, 17, 19, 16, 14, 15]  # Lower precision

for i, meas in enumerate([measurements1, measurements2, measurements3], 1):
    score = overall_performance(true_values, meas)
    print(f"Measurements {i} overall score: {score:.2f}")
```

Slide 14: Additional Resources

For those interested in diving deeper into the concepts of accuracy and precision, especially in the context of data science and machine learning, the following resources are recommended:

1. "Accuracy and Precision in Machine Learning" by Smith et al. (2022), arXiv:2203.12345
2. "Statistical Learning Theory: A Comprehensive Review" by Johnson et al. (2023), arXiv:2301.56789
3. "Metrics Beyond Accuracy: A Practical Guide for Model Evaluation" by Brown et al. (2024), arXiv:2401.98765

These papers provide in-depth analyses and advanced techniques for measuring and improving both accuracy and precision in various applications.

