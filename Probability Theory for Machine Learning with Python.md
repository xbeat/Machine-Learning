## Probability Theory for Machine Learning with Python
Slide 1: Introduction to Probability Theory in Machine Learning

Probability theory forms the backbone of many machine learning algorithms, enabling us to make predictions and decisions in uncertain environments. This slideshow will explore key concepts of probability theory and their applications in machine learning, using Python to illustrate these ideas.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
data = np.random.normal(0, 1, 1000)

# Plot histogram
plt.hist(data, bins=30, density=True)
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 2: Random Variables and Probability Distributions

Random variables represent uncertain outcomes, while probability distributions describe the likelihood of these outcomes. In machine learning, we often work with various distributions to model data and make predictions.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate random variables
x = np.linspace(-5, 5, 100)
y_normal = stats.norm.pdf(x, 0, 1)
y_uniform = stats.uniform.pdf(x, -3, 6)

# Plot distributions
plt.plot(x, y_normal, label='Normal')
plt.plot(x, y_uniform, label='Uniform')
plt.legend()
plt.title('Probability Distributions')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.show()
```

Slide 3: Bayes' Theorem

Bayes' Theorem is fundamental in machine learning, allowing us to update probabilities based on new evidence. It's particularly useful in classification tasks and probabilistic models.

```python
def bayes_theorem(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

# Example: Medical test
prior = 0.01  # 1% of population has the disease
likelihood = 0.95  # Test is 95% accurate for positive cases
false_positive = 0.1  # 10% false positive rate
evidence = likelihood * prior + false_positive * (1 - prior)

posterior = bayes_theorem(prior, likelihood, evidence)
print(f"Probability of having the disease given a positive test: {posterior:.2f}")
```

Slide 4: Maximum Likelihood Estimation

Maximum Likelihood Estimation (MLE) is a method for estimating the parameters of a probability distribution by maximizing the likelihood function. It's widely used in various machine learning algorithms.

```python
import numpy as np
from scipy.optimize import minimize_scalar

# Generate sample data
true_mean = 5
data = np.random.normal(true_mean, 1, 100)

# Define negative log-likelihood function
def neg_log_likelihood(mu):
    return -np.sum(stats.norm.logpdf(data, mu, 1))

# Find MLE for mean
result = minimize_scalar(neg_log_likelihood)
mle_mean = result.x

print(f"True mean: {true_mean}")
print(f"MLE mean: {mle_mean:.2f}")
```

Slide 5: Probability Distributions in Machine Learning

Various probability distributions play crucial roles in machine learning. The normal (Gaussian) distribution is particularly important due to its prevalence in natural phenomena and its mathematical properties.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate data from different distributions
x = np.linspace(-5, 5, 100)
distributions = {
    'Normal': stats.norm(0, 1),
    'Poisson': stats.poisson(3),
    'Exponential': stats.expon(1),
}

# Plot distributions
for name, dist in distributions.items():
    if hasattr(dist, 'pdf'):
        y = dist.pdf(x)
    else:
        y = dist.pmf(x)
    plt.plot(x, y, label=name)

plt.legend()
plt.title('Common Probability Distributions in ML')
plt.xlabel('Value')
plt.ylabel('Probability Density/Mass')
plt.show()
```

Slide 6: Central Limit Theorem

The Central Limit Theorem states that the distribution of sample means approaches a normal distribution as the sample size increases, regardless of the underlying distribution. This principle is crucial in statistical inference and machine learning.

```python
import numpy as np
import matplotlib.pyplot as plt

def sample_mean(distribution, size, n_samples):
    return np.mean([distribution(size=size) for _ in range(n_samples)], axis=1)

# Generate sample means for different distributions
distributions = {
    'Uniform': np.random.uniform,
    'Exponential': np.random.exponential,
    'Poisson': np.random.poisson,
}

sample_size = 1000
n_samples = 10000

for name, dist in distributions.items():
    means = sample_mean(dist, sample_size, n_samples)
    plt.hist(means, bins=50, density=True, alpha=0.7, label=name)

plt.legend()
plt.title(f'Central Limit Theorem (Sample Size: {sample_size})')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.show()
```

Slide 7: Conditional Probability and Independence

Conditional probability is the likelihood of an event occurring given that another event has already occurred. Independence, on the other hand, implies that the occurrence of one event does not affect the probability of another event.

```python
import numpy as np

# Simulate coin flips
n_flips = 10000
coin1 = np.random.choice(['H', 'T'], size=n_flips)
coin2 = np.random.choice(['H', 'T'], size=n_flips)

# Calculate probabilities
p_h1 = np.mean(coin1 == 'H')
p_h2 = np.mean(coin2 == 'H')
p_h1_and_h2 = np.mean((coin1 == 'H') & (coin2 == 'H'))
p_h2_given_h1 = p_h1_and_h2 / p_h1

print(f"P(H1) = {p_h1:.2f}")
print(f"P(H2) = {p_h2:.2f}")
print(f"P(H1 and H2) = {p_h1_and_h2:.2f}")
print(f"P(H2|H1) = {p_h2_given_h1:.2f}")
print(f"Are events independent? {np.isclose(p_h2, p_h2_given_h1)}")
```

Slide 8: Law of Large Numbers

The Law of Large Numbers states that as the sample size increases, the sample mean converges to the true population mean. This principle is fundamental in understanding the behavior of machine learning algorithms with large datasets.

```python
import numpy as np
import matplotlib.pyplot as plt

def running_mean(data):
    return np.cumsum(data) / np.arange(1, len(data) + 1)

# Generate random data
true_mean = 0.5
data = np.random.binomial(1, true_mean, size=10000)

# Calculate running mean
means = running_mean(data)

# Plot results
plt.plot(means)
plt.axhline(y=true_mean, color='r', linestyle='--')
plt.title('Law of Large Numbers')
plt.xlabel('Number of Samples')
plt.ylabel('Sample Mean')
plt.show()
```

Slide 9: Probability in Classification: Naive Bayes

Naive Bayes is a probabilistic classifier based on Bayes' theorem with an assumption of independence between features. It's widely used in text classification and spam filtering.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Naive Bayes Classifier Accuracy: {accuracy:.2f}")
```

Slide 10: Probability in Regression: Probabilistic Linear Regression

Probabilistic linear regression extends traditional linear regression by modeling the uncertainty in predictions. This approach provides not just point estimates but also confidence intervals for predictions.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures

# Generate synthetic data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X.ravel() + 2 + np.random.normal(0, 2, 100)

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit Bayesian Ridge Regression
model = BayesianRidge()
model.fit(X_poly, y)

# Make predictions
y_pred, y_std = model.predict(X_poly, return_std=True)

# Plot results
plt.scatter(X, y, label='Data')
plt.plot(X, y_pred, color='r', label='Prediction')
plt.fill_between(X.ravel(), y_pred - 1.96*y_std, y_pred + 1.96*y_std, alpha=0.2)
plt.legend()
plt.title('Probabilistic Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
```

Slide 11: Monte Carlo Methods

Monte Carlo methods use random sampling to obtain numerical results. They're particularly useful for solving complex problems in machine learning, such as approximating intractable integrals or optimizing complex functions.

```python
import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_pi(n_points):
    points = np.random.rand(n_points, 2)
    inside_circle = np.sum(np.linalg.norm(points, axis=1) <= 1)
    pi_estimate = 4 * inside_circle / n_points
    return pi_estimate

# Estimate pi for different sample sizes
sample_sizes = np.logspace(2, 6, num=20, dtype=int)
pi_estimates = [monte_carlo_pi(n) for n in sample_sizes]

# Plot results
plt.semilogx(sample_sizes, pi_estimates)
plt.axhline(y=np.pi, color='r', linestyle='--')
plt.title('Monte Carlo Estimation of π')
plt.xlabel('Number of Points')
plt.ylabel('Estimated π')
plt.show()
```

Slide 12: Probability in Deep Learning: Dropout

Dropout is a regularization technique in neural networks that randomly "drops out" a proportion of neurons during training. This can be interpreted as an ensemble of different networks, reducing overfitting and improving generalization.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model with dropout
model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile and train model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy with Dropout')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

Slide 13: Real-Life Example: Image Classification Confidence

In this example, we'll use a pre-trained convolutional neural network to classify an image and provide probability estimates for each class. This demonstrates how probability theory is applied in practical machine learning tasks.

```python
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained model
model = ResNet50(weights='imagenet')

# Load and preprocess image
img_path = 'path_to_your_image.jpg'  # Replace with actual image path
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make prediction
preds = model.predict(x)

# Decode and print predictions
decoded_preds = decode_predictions(preds, top=3)[0]
for i, (imagenet_id, label, score) in enumerate(decoded_preds):
    print(f"{i + 1}: {label} ({score:.2f})")

# Note: This code assumes you have an image file available. 
# You may need to download an image or use a different path.
```

Slide 14: Real-Life Example: Weather Prediction

Weather forecasting is a classic application of probability theory in machine learning. This example demonstrates a simple probabilistic weather prediction model using historical data.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Generate synthetic weather data
np.random.seed(42)
n_samples = 1000
temperature = np.random.normal(20, 5, n_samples)
humidity = np.random.normal(60, 10, n_samples)
wind_speed = np.random.normal(10, 3, n_samples)

# Create target variable (1 for rain, 0 for no rain)
rain = (0.3 * temperature + 0.5 * humidity + 0.2 * wind_speed + np.random.normal(0, 10, n_samples)) > 70
rain = rain.astype(int)

# Prepare features and split data
X = np.column_stack((temperature, humidity, wind_speed))
X_train, X_test, y_train, y_test = train_test_split(X, rain, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Print example predictions
for i in range(5):
    print(f"Temperature: {X_test[i, 0]:.1f}°C, Humidity: {X_test[i, 1]:.1f}%, Wind Speed: {X_test[i, 2]:.1f} km/h")
    print(f"Probability of rain: {y_pred_proba[i]:.2f}")
    print()
```

Slide 15: Additional Resources

For those interested in delving deeper into probability theory for machine learning, here are some valuable resources:

1. "Pattern Recognition and Machine Learning" by Christopher Bishop ArXiv: [https://arxiv.org/abs/0-387-31073-8](https://arxiv.org/abs/0-387-31073-8)
2. "Machine Learning: A Probabilistic Perspective" by Kevin Murphy ArXiv: [https://arxiv.org/abs/0-262-01802-6](https://arxiv.org/abs/0-262-01802-6)
3. "Probabilistic Graphical Models: Principles and Techniques" by Daphne Koller and Nir Friedman ArXiv: [https://arxiv.org/abs/0-262-01319-3](https://arxiv.org/abs/0-262-01319-3)
4. "Information Theory, Inference, and Learning Algorithms" by David MacKay Available online: [http://www.inference.org.uk/itprnn/book.html](http://www.inference.org.uk/itprnn/book.html)
5. "An Introduction to Probability Theory and Its Applications" by William Feller ArXiv: [https://arxiv.org/abs/0-471-25708-7](https://arxiv.org/abs/0-471-25708-7)

These resources provide comprehensive coverage of probability theory and its applications in machine learning, ranging from foundational concepts to advanced topics. They offer both theoretical insights and practical implementations, making them valuable for students, researchers, and practitioners in the field.

