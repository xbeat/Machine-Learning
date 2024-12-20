## Probability in Machine Learning with Python
Slide 1: Introduction to Probability in Machine Learning

Probability theory forms the backbone of many machine learning algorithms. It provides a framework for dealing with uncertainty and making predictions based on incomplete information. In this presentation, we'll explore key concepts of probability and their applications in machine learning using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
data = np.random.normal(0, 1, 1000)

# Plot histogram
plt.hist(data, bins=30, density=True)
plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

Slide 2: Random Variables and Distributions

Random variables are fundamental in probability theory. They represent quantities whose values are determined by chance. In machine learning, we often work with probability distributions, which describe the likelihood of different outcomes for a random variable.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Define a random variable
x = np.linspace(-5, 5, 100)

# Create normal distribution
normal_dist = stats.norm(0, 1)

# Plot PDF
plt.plot(x, normal_dist.pdf(x))
plt.title("Probability Density Function of Normal Distribution")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.show()
```

Slide 3: Bayes' Theorem

Bayes' theorem is a fundamental principle in probability theory that allows us to update our beliefs based on new evidence. It's widely used in machine learning for tasks like classification and inference.

```python
def bayes_theorem(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

# Example: Medical test
prior = 0.01  # 1% of population has the disease
likelihood = 0.95  # 95% accuracy for positive test
false_positive = 0.1  # 10% false positive rate

evidence = likelihood * prior + false_positive * (1 - prior)
posterior = bayes_theorem(prior, likelihood, evidence)

print(f"Probability of having the disease given a positive test: {posterior:.2f}")
```

Slide 4: Maximum Likelihood Estimation

Maximum Likelihood Estimation (MLE) is a method for estimating the parameters of a statistical model. It's widely used in machine learning for parameter estimation in various models.

```python
import numpy as np
from scipy.optimize import minimize

# Generate some data
true_mean, true_std = 5, 2
data = np.random.normal(true_mean, true_std, 1000)

# Define negative log-likelihood function
def neg_log_likelihood(params, data):
    mean, std = params
    return -np.sum(np.log(stats.norm.pdf(data, mean, std)))

# Perform MLE
initial_guess = [0, 1]
result = minimize(neg_log_likelihood, initial_guess, args=(data,))

print(f"Estimated mean: {result.x[0]:.2f}, Estimated std: {result.x[1]:.2f}")
print(f"True mean: {true_mean}, True std: {true_std}")
```

Slide 5: Probability Distributions in Machine Learning

Various probability distributions play crucial roles in machine learning. The normal (Gaussian) distribution is particularly important due to its properties and frequent occurrence in real-world phenomena.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate data from different distributions
normal = np.random.normal(0, 1, 1000)
uniform = np.random.uniform(-3, 3, 1000)
exponential = np.random.exponential(1, 1000)

# Plot histograms
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.hist(normal, bins=30, density=True)
plt.title("Normal")
plt.subplot(132)
plt.hist(uniform, bins=30, density=True)
plt.title("Uniform")
plt.subplot(133)
plt.hist(exponential, bins=30, density=True)
plt.title("Exponential")
plt.tight_layout()
plt.show()
```

Slide 6: Probabilistic Classifiers: Naive Bayes

Naive Bayes is a probabilistic classifier based on Bayes' theorem. It's simple yet effective for many classification tasks, especially in natural language processing.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Make predictions
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

Slide 7: Probabilistic Graphical Models

Probabilistic graphical models represent complex probability distributions using graphs. They're useful for modeling relationships between variables and performing inference.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple Bayesian network
G = nx.DiGraph()
G.add_edges_from([('Rain', 'Wet Grass'), ('Sprinkler', 'Wet Grass')])

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=12, arrows=True)
plt.title("Simple Bayesian Network")
plt.axis('off')
plt.show()
```

Slide 8: Monte Carlo Methods

Monte Carlo methods use random sampling to solve problems that might be deterministic in principle. They're widely used in machine learning for approximating complex distributions and integrals.

```python
import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(n_points):
    points = np.random.rand(n_points, 2)
    inside_circle = np.sum(np.sum(points**2, axis=1) <= 1)
    pi_estimate = 4 * inside_circle / n_points
    return pi_estimate

n_simulations = 1000
estimates = [estimate_pi(1000) for _ in range(n_simulations)]

plt.hist(estimates, bins=30)
plt.axvline(np.pi, color='r', linestyle='dashed', linewidth=2)
plt.title(f"Monte Carlo Estimation of π\nMean estimate: {np.mean(estimates):.4f}")
plt.xlabel("Estimated π")
plt.ylabel("Frequency")
plt.show()
```

Slide 9: Uncertainty in Neural Networks

Incorporating uncertainty in neural networks is crucial for reliable predictions. Techniques like dropout and Bayesian neural networks help quantify uncertainty in deep learning models.

```python
import tensorflow as tf
import tensorflow_probability as tfp

# Define a simple Bayesian neural network
def bayesian_neural_network(input_shape):
    model = tf.keras.Sequential([
        tfp.layers.DenseVariational(10, activation='relu', input_shape=input_shape),
        tfp.layers.DenseVariational(1)
    ])
    return model

# Create and compile the model
model = bayesian_neural_network((1,))
model.compile(optimizer='adam', loss='mse')

# Generate synthetic data
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

# Train the model
model.fit(X, y, epochs=1000, verbose=0)

# Make predictions with uncertainty
predictions = [model(X) for _ in range(100)]
mean_pred = np.mean(predictions, axis=0)
std_pred = np.std(predictions, axis=0)

plt.scatter(X, y, label='Data')
plt.plot(X, mean_pred, 'r-', label='Mean prediction')
plt.fill_between(X.ravel(), mean_pred.ravel() - 2*std_pred.ravel(), 
                 mean_pred.ravel() + 2*std_pred.ravel(), alpha=0.3)
plt.legend()
plt.title("Bayesian Neural Network: Predictions with Uncertainty")
plt.show()
```

Slide 10: Probabilistic Topic Models

Probabilistic topic models, such as Latent Dirichlet Allocation (LDA), are used to discover abstract topics in a collection of documents. They're widely used in natural language processing and text mining.

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load data
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Preprocess text data
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(newsgroups.data)

# Apply LDA
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(doc_term_matrix)

# Print top words for each topic
feature_names = vectorizer.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")
```

Slide 11: Real-Life Example: Spam Detection

Spam detection is a common application of probabilistic machine learning. We'll use a Naive Bayes classifier to categorize emails as spam or not spam.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample data (replace with real email dataset)
emails = [
    "Get rich quick! Buy now!",
    "Meeting scheduled for tomorrow",
    "Claim your prize! Limited time offer!",
    "Project update: new deadlines",
    "Free money! Click here!",
    "Reminder: submit your report"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for not spam

# Preprocess and split data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train and evaluate model
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))
```

Slide 12: Real-Life Example: Weather Prediction

Weather prediction is another area where probabilistic machine learning is widely used. We'll create a simple model to predict the likelihood of rain based on temperature and humidity.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Generate synthetic weather data
np.random.seed(42)
n_samples = 1000
temperature = np.random.normal(25, 5, n_samples)
humidity = np.random.normal(60, 10, n_samples)
rain = (0.3 * temperature + 0.7 * humidity + np.random.normal(0, 10, n_samples) > 80).astype(int)

# Train logistic regression model
X = np.column_stack((temperature, humidity))
model = LogisticRegression()
model.fit(X, rain)

# Create a grid for visualization
temp_range = np.linspace(10, 40, 100)
humid_range = np.linspace(30, 90, 100)
temp_grid, humid_grid = np.meshgrid(temp_range, humid_range)
X_grid = np.column_stack((temp_grid.ravel(), humid_grid.ravel()))

# Predict probabilities
rain_prob = model.predict_proba(X_grid)[:, 1].reshape(temp_grid.shape)

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(temp_grid, humid_grid, rain_prob, levels=20, cmap='RdYlBu_r')
plt.colorbar(label='Probability of Rain')
plt.scatter(temperature[rain==0], humidity[rain==0], c='blue', alpha=0.5, label='No Rain')
plt.scatter(temperature[rain==1], humidity[rain==1], c='red', alpha=0.5, label='Rain')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.title('Weather Prediction: Probability of Rain')
plt.legend()
plt.show()
```

Slide 13: Future Directions and Challenges

As machine learning continues to evolve, probabilistic methods are becoming increasingly important. Some key areas of development include:

1. Bayesian deep learning for improved uncertainty quantification
2. Causal inference in machine learning
3. Probabilistic programming languages for more flexible model specification
4. Scalable inference algorithms for large-scale probabilistic models

These advancements will enable more robust and interpretable machine learning systems, capable of handling complex real-world scenarios with greater accuracy and reliability.

Slide 14: Additional Resources

For those interested in diving deeper into probability in machine learning, here are some valuable resources:

1. "Probabilistic Machine Learning: An Introduction" by Kevin Murphy (2022) ArXiv: [https://arxiv.org/abs/2006.12563](https://arxiv.org/abs/2006.12563)
2. "Bayesian Reasoning and Machine Learning" by David Barber Available online: [http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.Online](http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.Online)
3. "Pattern Recognition and Machine Learning" by Christopher Bishop
4. "Machine Learning: A Probabilistic Perspective" by Kevin Murphy

These resources provide comprehensive coverage of probabilistic methods in machine learning, from foundational concepts to advanced techniques.

