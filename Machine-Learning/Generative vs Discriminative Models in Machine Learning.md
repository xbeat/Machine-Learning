## Generative vs Discriminative Models in Machine Learning
Slide 1: Understanding Generative Models

Generative models aim to learn the joint probability distribution P(X,Y) of input features X and labels Y, enabling them to generate new samples by sampling from the learned distribution. They model how the data was generated to make predictions, making them particularly useful for data synthesis and density estimation.

```python
import numpy as np
from scipy.stats import multivariate_normal

class GaussianGenerativeModel:
    def __init__(self):
        self.priors = {}  # P(Y)
        self.means = {}   # Mean vectors for P(X|Y)
        self.covs = {}    # Covariance matrices for P(X|Y)
    
    def fit(self, X, y):
        # Calculate class priors P(Y)
        unique_classes = np.unique(y)
        n_samples = len(y)
        
        for c in unique_classes:
            # Get samples for current class
            X_c = X[y == c]
            
            # Calculate prior P(Y)
            self.priors[c] = len(X_c) / n_samples
            
            # Calculate mean vector and covariance matrix for P(X|Y)
            self.means[c] = np.mean(X_c, axis=0)
            self.covs[c] = np.cov(X_c.T)
    
    def generate_samples(self, class_label, n_samples=1):
        # Generate new samples from learned distribution
        return multivariate_normal.rvs(
            mean=self.means[class_label],
            cov=self.covs[class_label],
            size=n_samples
        )
```

Slide 2: Understanding Discriminative Models

Discriminative models directly learn the conditional probability P(Y|X), focusing on finding decision boundaries between classes. These models are optimized specifically for classification tasks and don't learn the underlying data distribution, making them more efficient for prediction tasks.

```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.max_iter):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Update weights and bias
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return (self.sigmoid(linear_pred) >= 0.5).astype(int)
```

Slide 3: Mathematical Foundations of Generative Models

The fundamental principle of generative models lies in Bayes' theorem and joint probability distribution. These models learn both the likelihood P(X|Y) and prior P(Y) to compute the posterior P(Y|X), providing a complete probabilistic framework for inference and generation.

```python
# Mathematical formulas for generative models
"""
$$P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}$$

$$P(X) = \sum_{y} P(X|Y=y)P(Y=y)$$

$$\log P(X,Y) = \log P(X|Y) + \log P(Y)$$
"""

class BayesianGenerativeModel:
    def compute_posterior(self, x, likelihood, prior):
        # P(Y|X) ∝ P(X|Y)P(Y)
        numerator = likelihood * prior
        evidence = np.sum(numerator)
        return numerator / evidence if evidence != 0 else 0
```

Slide 4: Mathematical Foundations of Discriminative Models

Discriminative models optimize the conditional probability directly through various loss functions and gradient-based methods. They focus on learning decision boundaries without modeling the underlying data distribution, making them more efficient for classification tasks.

```python
# Mathematical formulas for discriminative models
"""
$$P(Y|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + ... + \beta_nX_n)}}$$

$$\text{Loss} = -\frac{1}{N}\sum_{i=1}^N [y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

$$\frac{\partial \text{Loss}}{\partial \beta} = -\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)X_i$$
"""

def compute_loss(y_true, y_pred, epsilon=1e-15):
    # Binary cross-entropy loss
    return -np.mean(
        y_true * np.log(y_pred + epsilon) + 
        (1 - y_true) * np.log(1 - y_pred + epsilon)
    )
```

Slide 5: Implementing a Gaussian Mixture Model (GMM)

A Gaussian Mixture Model is a powerful generative model that represents complex probability distributions as a weighted sum of Gaussian components. This implementation demonstrates how GMMs can model multi-modal data distributions and generate new samples.

```python
import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, n_components=3, max_iter=100):
        self.n_components = n_components
        self.max_iter = max_iter
        
    def initialize_parameters(self, X):
        n_samples, n_features = X.shape
        # Initialize means randomly
        self.means = X[np.random.choice(n_samples, self.n_components)]
        # Initialize covariances as identity matrices
        self.covs = [np.eye(n_features) for _ in range(self.n_components)]
        # Initialize weights uniformly
        self.weights = np.ones(self.n_components) / self.n_components
        
    def expectation_step(self, X):
        responsibilities = np.zeros((len(X), self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                X, self.means[k], self.covs[k]
            )
        # Normalize responsibilities
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities
    
    def maximization_step(self, X, responsibilities):
        N = responsibilities.sum(axis=0)
        # Update means
        self.means = np.dot(responsibilities.T, X) / N[:, np.newaxis]
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covs[k] = np.dot(responsibilities[:, k] * diff.T, diff) / N[k]
        # Update weights
        self.weights = N / len(X)
```

Slide 6: Naive Bayes as a Simple Generative Model

Naive Bayes represents a simple yet effective generative model that makes strong independence assumptions between features. Despite its simplicity, it performs well in text classification and other high-dimensional problems where feature independence approximation is reasonable.

```python
import numpy as np
from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_likelihood = defaultdict(lambda: defaultdict(dict))
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        
        # Calculate class priors P(Y)
        for c in classes:
            self.class_priors[c] = np.sum(y == c) / n_samples
            
        # Calculate feature likelihoods P(X|Y)
        for c in classes:
            X_c = X[y == c]
            for feature in range(n_features):
                mean = np.mean(X_c[:, feature])
                std = np.std(X_c[:, feature])
                self.feature_likelihood[c][feature] = {'mean': mean, 'std': std}
    
    def _calculate_likelihood(self, x, mean, std):
        return (1 / (np.sqrt(2 * np.pi) * std)) * \
               np.exp(-np.power(x - mean, 2) / (2 * np.power(std, 2)))
    
    def predict(self, X):
        predictions = []
        for x in X:
            class_scores = {}
            for c in self.class_priors:
                score = np.log(self.class_priors[c])
                for feature, value in enumerate(x):
                    params = self.feature_likelihood[c][feature]
                    likelihood = self._calculate_likelihood(
                        value, params['mean'], params['std']
                    )
                    score += np.log(likelihood + 1e-10)
                class_scores[c] = score
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)
```

Slide 7: Implementing a Variational Autoencoder (VAE)

Variational Autoencoders are sophisticated generative models that learn a continuous latent representation of data through variational inference. This implementation shows how VAEs combine neural networks with probabilistic modeling to generate new samples.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
```

Slide 8: Real-World Application - Text Classification

This example demonstrates how discriminative and generative models perform differently on a text classification task. We'll implement both approaches for spam detection using a bag-of-words representation of email content.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class TextClassificationComparison:
    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=1000)
        self.generative_model = NaiveBayes()  # From previous slide
        self.discriminative_model = LogisticRegression()  # From previous slide
        
    def preprocess_data(self, texts, labels):
        # Convert text to bag-of-words representation
        X = self.vectorizer.fit_transform(texts).toarray()
        return train_test_split(X, labels, test_size=0.2, random_state=42)
    
    def compare_models(self, texts, labels):
        # Prepare data
        X_train, X_test, y_train, y_test = self.preprocess_data(texts, labels)
        
        # Train and evaluate generative model
        self.generative_model.fit(X_train, y_train)
        gen_predictions = self.generative_model.predict(X_test)
        gen_accuracy = np.mean(gen_predictions == y_test)
        
        # Train and evaluate discriminative model
        self.discriminative_model.fit(X_train, y_train)
        disc_predictions = self.discriminative_model.predict(X_test)
        disc_accuracy = np.mean(disc_predictions == y_test)
        
        return {
            'generative_accuracy': gen_accuracy,
            'discriminative_accuracy': disc_accuracy
        }
```

Slide 9: Real-World Application - Image Generation

This implementation shows how a generative model can be used for image synthesis, demonstrating the unique capability of generative models to create new, realistic data samples.

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class ImageGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super(ImageGenerator, self).__init__()
        
        self.generator = nn.Sequential(
            # Initial projection and reshape
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True),
            
            # Reshape to start convolutions
            lambda x: x.view(-1, 256, 4, 4),
            
            # Upsampling layers
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.generator(z)
    
    def generate_images(self, num_images=1, device='cuda'):
        with torch.no_grad():
            z = torch.randn(num_images, 100).to(device)
            return self.forward(z)
```

Slide 10: Results Analysis - Performance Metrics

A comprehensive comparison of generative and discriminative models across different metrics reveals their strengths and weaknesses in various scenarios.

```python
class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, y_true, y_pred_gen, y_pred_disc):
        def calculate_accuracy(y_true, y_pred):
            return np.mean(y_true == y_pred)
        
        def calculate_precision(y_true, y_pred):
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            return tp / (tp + fp) if (tp + fp) > 0 else 0
        
        self.metrics = {
            'generative': {
                'accuracy': calculate_accuracy(y_true, y_pred_gen),
                'precision': calculate_precision(y_true, y_pred_gen)
            },
            'discriminative': {
                'accuracy': calculate_accuracy(y_true, y_pred_disc),
                'precision': calculate_precision(y_true, y_pred_disc)
            }
        }
        return self.metrics
    
    def print_comparison(self):
        for model_type, metrics in self.metrics.items():
            print(f"\n{model_type.capitalize()} Model Metrics:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
```

Slide 11: Advanced Discriminative Model - Neural Network Implementation

This implementation showcases a deep neural network as a sophisticated discriminative model, incorporating modern architecture elements like dropout and batch normalization for improved performance.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class AdvancedDiscriminativeNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super(AdvancedDiscriminativeNet, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers dynamically
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
    def train_model(self, train_loader, epochs=10, lr=0.001):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
```

Slide 12: Implementation of a Conditional Generative Model

Conditional generative models extend basic generative models by incorporating additional control over the generation process through conditioning variables. This implementation demonstrates how to generate samples with specific desired attributes.

```python
class ConditionalGenerativeModel:
    def __init__(self, feature_dim, condition_dim):
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.feature_distributions = {}
        
    def fit(self, X, conditions):
        # For each unique condition, learn feature distributions
        unique_conditions = np.unique(conditions, axis=0)
        
        for condition in unique_conditions:
            condition_key = tuple(condition)
            mask = np.all(conditions == condition, axis=1)
            X_subset = X[mask]
            
            # Store mean and covariance for each condition
            self.feature_distributions[condition_key] = {
                'mean': np.mean(X_subset, axis=0),
                'cov': np.cov(X_subset.T)
            }
    
    def generate(self, condition, n_samples=1):
        condition_key = tuple(condition)
        if condition_key not in self.feature_distributions:
            raise ValueError("Unknown condition")
            
        dist = self.feature_distributions[condition_key]
        return np.random.multivariate_normal(
            dist['mean'], 
            dist['cov'], 
            size=n_samples
        )
    
    def generate_interpolated(self, condition1, condition2, steps=5):
        samples = []
        for alpha in np.linspace(0, 1, steps):
            interpolated_condition = tuple(
                alpha * np.array(condition1) + 
                (1 - alpha) * np.array(condition2)
            )
            samples.append(self.generate(interpolated_condition))
        return np.array(samples)
```

Slide 13: Additional Resources

*   "Deep Generative Models: A Complete Survey" - [https://arxiv.org/abs/2006.10863](https://arxiv.org/abs/2006.10863)
*   "A Comparative Study of Discriminative and Generative Models" - [https://arxiv.org/abs/1901.08358](https://arxiv.org/abs/1901.08358)
*   "Understanding the Effectiveness of Deep Generative Models" - [https://arxiv.org/abs/1912.09791](https://arxiv.org/abs/1912.09791)
*   Google Scholar search suggestion: "comparison discriminative generative models deep learning"
*   Recommended search: "recent advances in generative models neural networks"

