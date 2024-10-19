## Descriptive Statistics for Data Understanding

Slide 1: Data Understanding with Descriptive Statistics

Descriptive statistics provide a powerful toolkit for summarizing and visualizing data, enabling researchers and analysts to gain quick insights into large datasets. These techniques help identify central tendencies, spread, and overall distribution of data points. Let's explore how to calculate basic descriptive statistics using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
data = np.random.normal(loc=50, scale=10, size=1000)

# Calculate descriptive statistics
mean = np.mean(data)
median = np.median(data)
std_dev = np.std(data)

# Visualize the data
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black')
plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
plt.title('Data Distribution with Descriptive Statistics')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
```

Slide 2: Outlier Detection Using Statistical Methods

Outliers can significantly impact data analysis and machine learning models. Statistical methods help identify these anomalies efficiently. The Interquartile Range (IQR) method is a robust technique for outlier detection. Let's implement this method in Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with outliers
np.random.seed(42)
data = np.concatenate([np.random.normal(0, 1, 1000), np.random.normal(10, 1, 5)])

# Calculate Q1, Q3, and IQR
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

# Define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = data[(data < lower_bound) | (data > upper_bound)]

# Visualize the data and outliers
plt.figure(figsize=(10, 6))
plt.boxplot(data)
plt.scatter(np.ones(len(outliers)), outliers, color='red', label='Outliers')
plt.title('Box Plot with Outliers')
plt.ylabel('Value')
plt.legend()
plt.show()

print(f"Number of outliers detected: {len(outliers)}")
print(f"Outlier values: {outliers}")
```

Slide 3: Feature Selection Using Correlation Analysis

Feature selection is crucial for building efficient machine learning models. Correlation analysis helps identify relevant features by measuring the strength of relationships between variables. Let's use Python to compute and visualize a correlation matrix.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    'Feature1': np.random.rand(100),
    'Feature2': np.random.rand(100),
    'Feature3': np.random.rand(100),
    'Target': np.random.rand(100)
})
data['Feature4'] = data['Feature1'] * 2 + np.random.normal(0, 0.1, 100)

# Compute correlation matrix
corr_matrix = data.corr()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# Select highly correlated features
threshold = 0.5
high_corr_features = np.where(np.abs(corr_matrix['Target']) > threshold)[0]
print("Highly correlated features with the target:")
print(data.columns[high_corr_features])
```

Slide 4: Verifying Model Assumptions: Normality Test

Many statistical models assume that the data follows a normal distribution. Checking this assumption is crucial for the validity of these models. Let's use the Shapiro-Wilk test to assess normality in Python.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Generate two datasets: one normal, one non-normal
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
non_normal_data = np.random.exponential(1, 1000)

# Perform Shapiro-Wilk test
_, p_value_normal = stats.shapiro(normal_data)
_, p_value_non_normal = stats.shapiro(non_normal_data)

# Visualize the distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(normal_data, bins=30, edgecolor='black')
ax1.set_title(f'Normal Data\nShapiro-Wilk p-value: {p_value_normal:.4f}')

ax2.hist(non_normal_data, bins=30, edgecolor='black')
ax2.set_title(f'Non-Normal Data\nShapiro-Wilk p-value: {p_value_non_normal:.4f}')

plt.tight_layout()
plt.show()

print(f"Normal data p-value: {p_value_normal:.4f}")
print(f"Non-normal data p-value: {p_value_non_normal:.4f}")
print("If p-value < 0.05, we reject the null hypothesis of normality.")
```

Slide 5: Probability in Machine Learning: Naive Bayes Classifier

Probability forms the foundation of many machine learning algorithms. The Naive Bayes classifier is a prime example, using Bayes' theorem to make predictions. Let's implement a simple Naive Bayes classifier for text classification.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
texts = ["I love this movie", "This movie is awful", "Great acting", "Terrible plot",
         "Highly recommended", "Waste of time", "Fantastic film", "Boring story"]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

# Split the data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42)

# Vectorize the text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Make predictions
y_pred = clf.predict(X_test_vec)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Predict a new review
new_review = ["This movie was absolutely amazing"]
new_review_vec = vectorizer.transform(new_review)
prediction = clf.predict(new_review_vec)
print(f"\nPrediction for '{new_review[0]}': {'Positive' if prediction[0] == 1 else 'Negative'}")
```

Slide 6: Model Evaluation Metrics: Beyond Accuracy

While accuracy is a common metric, it doesn't always provide a complete picture of model performance, especially for imbalanced datasets. Let's explore precision, recall, and F1-score using a confusion matrix visualization.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], 
                           n_informative=3, n_redundant=0, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Compute evaluation metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Negative', 'Positive'])
plt.yticks(tick_marks, ['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations to the confusion matrix
thresh = cm.max() / 2
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
```

Slide 7: Bias-Variance Tradeoff: Learning Curves

The bias-variance tradeoff is a fundamental concept in machine learning that helps balance model complexity. Learning curves visualize how model performance changes with increasing training data, helping identify underfitting or overfitting. Let's create learning curves for a polynomial regression model.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Generate sample data
np.random.seed(42)
X = np.sort(np.random.rand(100, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Create polynomial regression models
degrees = [1, 4, 15]  # Different polynomial degrees
plt.figure(figsize=(16, 5))

for i, degree in enumerate(degrees):
    plt.subplot(1, 3, i+1)
    
    # Create and fit the model
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    # Compute learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='neg_mean_squared_error')
    
    # Calculate mean and std of scores
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.title(f"Degree {degree} Polynomial")
    plt.xlabel("Training examples")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="best")
    plt.ylim(0, 0.5)

plt.tight_layout()
plt.show()
```

Slide 8: Overfitting Detection: Cross-Validation

Cross-validation is a powerful technique to detect overfitting by assessing how well a model generalizes to unseen data. K-fold cross-validation is particularly useful. Let's implement it for a decision tree classifier and visualize the results.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# Generate a dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, 
                           n_redundant=10, random_state=42)

# Create arrays to store scores
max_depth_range = range(1, 21)
train_scores = []
cv_scores = []

# Perform cross-validation for different tree depths
for depth in max_depth_range:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    train_score = dt.fit(X, y).score(X, y)
    cv_score = cross_val_score(dt, X, y, cv=5).mean()
    train_scores.append(train_score)
    cv_scores.append(cv_score)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(max_depth_range, train_scores, label='Training Score', marker='o')
plt.plot(max_depth_range, cv_scores, label='Cross-Validation Score', marker='o')
plt.xlabel('Max Tree Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Training vs Cross-Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Find the optimal depth
optimal_depth = max_depth_range[np.argmax(cv_scores)]
print(f"Optimal tree depth: {optimal_depth}")
print(f"Best cross-validation score: {max(cv_scores):.4f}")
```

Slide 9: Statistical Sampling: Stratified Sampling

Proper statistical sampling ensures reliable and representative training datasets. Stratified sampling is particularly useful when dealing with imbalanced datasets. Let's implement stratified sampling and compare it with random sampling.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Create an imbalanced dataset
np.random.seed(42)
data = pd.DataFrame({
    'feature': np.random.randn(1000),
    'class': np.random.choice(['A', 'B', 'C'], p=[0.8, 0.15, 0.05], size=1000)
})

# Function to plot class distribution
def plot_distribution(data, title):
    plt.figure(figsize=(8, 5))
    data['class'].value_counts().plot(kind='bar')
    plt.title(title)
    plt
```

Slide 9: Statistical Sampling: Stratified Sampling

Proper statistical sampling ensures reliable and representative training datasets. Stratified sampling is particularly useful when dealing with imbalanced datasets. Let's implement stratified sampling and compare it with random sampling.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Create an imbalanced dataset
np.random.seed(42)
data = pd.DataFrame({
    'feature': np.random.randn(1000),
    'class': np.random.choice(['A', 'B', 'C'], p=[0.8, 0.15, 0.05], size=1000)
})

# Function to plot class distribution
def plot_distribution(data, title):
    plt.figure(figsize=(8, 5))
    data['class'].value_counts().plot(kind='bar')
    plt.title(title)
    plt.ylabel('Count')
    plt.show()

# Original distribution
plot_distribution(data, 'Original Class Distribution')

# Random sampling
_, random_sample = train_test_split(data, test_size=0.2, random_state=42)
plot_distribution(random_sample, 'Random Sampling: Class Distribution')

# Stratified sampling
_, stratified_sample = train_test_split(data, test_size=0.2, stratify=data['class'], random_state=42)
plot_distribution(stratified_sample, 'Stratified Sampling: Class Distribution')

print("Original class distribution:")
print(data['class'].value_counts(normalize=True))
print("\nRandom sampling class distribution:")
print(random_sample['class'].value_counts(normalize=True))
print("\nStratified sampling class distribution:")
print(stratified_sample['class'].value_counts(normalize=True))
```

Slide 10: Real-Life Example: Weather Prediction

Weather prediction is a common application of statistical analysis in meteorology. Let's create a simple linear regression model to predict temperature based on humidity levels.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic weather data
np.random.seed(42)
humidity = np.random.uniform(30, 100, 1000)
temperature = 0.3 * humidity + np.random.normal(0, 5, 1000)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(humidity.reshape(-1, 1), temperature, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Humidity (%)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Prediction based on Humidity')
plt.legend()
plt.show()

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
print(f"Equation: Temperature = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Humidity")
```

Slide 11: Real-Life Example: Image Classification

Image classification is a popular application of machine learning in computer vision. Let's use a simple Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Visualize training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 12: Hypothesis Testing: A/B Testing

A/B testing is a statistical method used to compare two versions of a product or service to determine which performs better. Let's simulate an A/B test for a website redesign and analyze the results.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Simulate data for two website versions
np.random.seed(42)
visitors_A = 10000
visitors_B = 10000
conversion_rate_A = 0.10
conversion_rate_B = 0.12

conversions_A = np.random.binomial(1, conversion_rate_A, visitors_A)
conversions_B = np.random.binomial(1, conversion_rate_B, visitors_B)

# Calculate observed conversion rates
obs_rate_A = np.mean(conversions_A)
obs_rate_B = np.mean(conversions_B)

# Perform hypothesis test (two-proportion z-test)
n_A, n_B = len(conversions_A), len(conversions_B)
p_pooled = (np.sum(conversions_A) + np.sum(conversions_B)) / (n_A + n_B)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_A + 1/n_B))
z_score = (obs_rate_B - obs_rate_A) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

# Visualize results
plt.figure(figsize=(10, 6))
plt.bar(['Version A', 'Version B'], [obs_rate_A, obs_rate_B], color=['blue', 'green'])
plt.title('A/B Test Results: Website Conversion Rates')
plt.ylabel('Conversion Rate')
plt.ylim(0, max(obs_rate_A, obs_rate_B) * 1.2)
for i, rate in enumerate([obs_rate_A, obs_rate_B]):
    plt.text(i, rate, f'{rate:.4f}', ha='center', va='bottom')
plt.show()

print(f"Observed conversion rate A: {obs_rate_A:.4f}")
print(f"Observed conversion rate B: {obs_rate_B:.4f}")
print(f"Z-score: {z_score:.4f}")
print(f"P-value: {p_value:.4f}")
print("Conclusion: ", "Statistically significant difference" if p_value < 0.05 else "No statistically significant difference")
```

Slide 13: Time Series Analysis: Forecasting

Time series analysis is crucial for predicting future trends based on historical data. Let's use the ARIMA model to forecast future values of a time series.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Generate synthetic time series data
np.random.seed(42)
date_rng = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
ts = pd.Series(np.cumsum(np.random.randn(len(date_rng))), index=date_rng)

# Split the data into train and test sets
train = ts[:'2022-06-30']
test = ts['2022-07-01':]

# Fit ARIMA model
model = ARIMA(train, order=(1,1,1))
results = model.fit()

# Make predictions
forecast = results.forecast(steps=len(test))

# Calculate MSE
mse = mean_squared_error(test, forecast)

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Actual Test Data')
plt.plot(test.index, forecast, color='red', label='Forecast')
plt.title('Time Series Forecasting with ARIMA')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

print(f"Mean Squared Error: {mse:.4f}")
```

Slide 14: Additional Resources

For those interested in deepening their understanding of statistics in data science and machine learning, here are some valuable resources:

1.  "Statistical Learning with Applications in R" by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani ArXiv link: [https://arxiv.org/abs/1501.07477](https://arxiv.org/abs/1501.07477)
2.  "Bayesian Methods for Hackers" by Cameron Davidson-Pilon ArXiv link: [https://arxiv.org/abs/1507.05738](https://arxiv.org/abs/1507.05738)
3.  "Probabilistic Programming and Bayesian Methods for Hackers" by Cameron Davidson-Pilon ArXiv link: [https://arxiv.org/abs/1503.02123](https://arxiv.org/abs/1503.02123)
4.  "A Tutorial on Principal Component Analysis" by Jonathon Shlens ArXiv link: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)

These resources provide in-depth coverage of various statistical concepts and their applications in data science and machine learning.

