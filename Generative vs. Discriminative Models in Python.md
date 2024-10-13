## Generative vs. Discriminative Models in Python
Slide 1: Introduction to Generative and Discriminative Models

Generative and discriminative models are two fundamental approaches in machine learning. Generative models learn the joint probability distribution of inputs and outputs, while discriminative models focus on learning the conditional probability distribution of outputs given inputs. This slideshow will explore their differences, characteristics, and applications using Python examples.

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualizing the difference
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title("Generative Model")
plt.text(0.5, 0.5, "P(X, Y)", fontsize=20, ha='center')
plt.axis('off')
plt.subplot(122)
plt.title("Discriminative Model")
plt.text(0.5, 0.5, "P(Y|X)", fontsize=20, ha='center')
plt.axis('off')
plt.tight_layout()
plt.show()
```

Slide 2: Generative Models: Overview

Generative models aim to learn the joint probability distribution P(X, Y) of input features X and output labels Y. They can generate new data points and are often used for tasks like data synthesis and anomaly detection. Examples include Naive Bayes, Hidden Markov Models, and Generative Adversarial Networks (GANs).

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Simulating data
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Training a Naive Bayes model (generative)
nb_model = GaussianNB()
nb_model.fit(X, y)

# Generating new data points
new_points = nb_model.sample(n_samples=5)
print("Generated points:", new_points)
```

Slide 3: Discriminative Models: Overview

Discriminative models focus on learning the conditional probability distribution P(Y|X), directly modeling the decision boundary between classes. They are often used for classification and regression tasks. Examples include Logistic Regression, Support Vector Machines (SVMs), and Neural Networks.

```python
from sklearn.linear_model import LogisticRegression

# Training a Logistic Regression model (discriminative)
lr_model = LogisticRegression()
lr_model.fit(X, y)

# Predicting probabilities for new data points
new_points = np.random.randn(5, 2)
probabilities = lr_model.predict_proba(new_points)
print("Predicted probabilities:", probabilities)
```

Slide 4: Key Differences: Model Focus

Generative models learn the full joint distribution P(X, Y), allowing them to generate new data points. Discriminative models focus solely on the conditional distribution P(Y|X), optimizing the decision boundary between classes.

```python
import seaborn as sns

# Visualizing the difference in focus
plt.figure(figsize=(12, 5))

plt.subplot(121)
sns.kdeplot(x=X[:, 0], y=X[:, 1], cmap="Blues", shade=True)
plt.title("Generative: Joint Distribution P(X, Y)")

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
plt.title("Discriminative: Decision Boundary")

plt.tight_layout()
plt.show()
```

Slide 5: Training Objective

Generative models maximize the likelihood of the observed data under the joint distribution. Discriminative models maximize the conditional likelihood of the labels given the input features.

```python
import torch
import torch.nn as nn

# Generative model (simplified)
class GenerativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)
    
    def forward(self, x):
        return torch.softmax(self.layer(x), dim=1)

# Discriminative model
class DiscriminativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.layer(x))

# Loss functions
gen_loss = nn.NLLLoss()  # Negative log-likelihood for generative
disc_loss = nn.BCELoss()  # Binary cross-entropy for discriminative
```

Slide 6: Data Efficiency

Generative models typically require more data to learn the full joint distribution effectively. Discriminative models can often achieve good performance with less data, as they focus solely on the decision boundary.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def compare_efficiency(n_samples):
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    nb_model = GaussianNB()
    lr_model = LogisticRegression()
    
    nb_model.fit(X, y)
    lr_model.fit(X, y)
    
    X_test = np.random.randn(1000, 2)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
    
    return accuracy_score(y_test, nb_model.predict(X_test)), accuracy_score(y_test, lr_model.predict(X_test))

samples = [10, 50, 100, 500, 1000]
nb_scores, lr_scores = zip(*[compare_efficiency(n) for n in samples])

plt.plot(samples, nb_scores, label='Naive Bayes (Generative)')
plt.plot(samples, lr_scores, label='Logistic Regression (Discriminative)')
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Performance vs. Training Data Size')
plt.show()
```

Slide 7: Handling Missing Data

Generative models often handle missing data more naturally, as they model the full joint distribution. Discriminative models may require additional techniques or preprocessing to handle missing features.

```python
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Create data with missing values
X = np.random.randn(100, 3)
X[np.random.choice(100, 20), 1] = np.nan  # Introduce missing values
y = (X[:, 0] + X[:, 2] > 0).astype(int)

# Generative model (Naive Bayes)
nb_model = GaussianNB()
nb_model.fit(X, y)  # Naive Bayes handles missing data internally

# Discriminative model (Logistic Regression) with imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
lr_model = LogisticRegression()
lr_model.fit(X_imputed, y)

print("Naive Bayes prediction:", nb_model.predict(X[:5]))
print("Logistic Regression prediction:", lr_model.predict(X_imputed[:5]))
```

Slide 8: Interpretability

Generative models often provide more intuitive interpretations of the data distribution. Discriminative models focus on decision boundaries, which may be less interpretable in high-dimensional spaces.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Generate data
X = np.random.randn(1000, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train models
nb_model = GaussianNB()
lr_model = LogisticRegression()
nb_model.fit(X, y)
lr_model.fit(X, y)

# Visualize decision boundaries
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z_nb = nb_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_lr = lr_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z_nb, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.title("Naive Bayes (Generative)")

plt.subplot(122)
plt.contourf(xx, yy, Z_lr, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.title("Logistic Regression (Discriminative)")

plt.tight_layout()
plt.show()
```

Slide 9: Flexibility and Model Complexity

Discriminative models are often more flexible and can model complex decision boundaries. Generative models may struggle with complex distributions but can be more robust to overfitting with limited data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Generate non-linear data
X = np.random.randn(1000, 2)
y = ((X[:, 0]**2 + X[:, 1]**2) > 1).astype(int)

# Train models
nb_model = GaussianNB()
mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
nb_model.fit(X, y)
mlp_model.fit(X, y)

# Visualize decision boundaries
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z_nb = nb_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_mlp = mlp_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z_nb, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.title("Naive Bayes (Generative)")

plt.subplot(122)
plt.contourf(xx, yy, Z_mlp, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.title("Neural Network (Discriminative)")

plt.tight_layout()
plt.show()
```

Slide 10: Real-Life Example: Text Classification

Let's compare a generative model (Naive Bayes) with a discriminative model (Logistic Regression) for text classification.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample text data
texts = [
    "I love this product", "Great service", "Terrible experience",
    "Awful customer support", "Amazing features", "Waste of money",
    "Highly recommended", "Do not buy", "Best purchase ever",
    "Disappointing quality"
]
labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]

# Preprocess text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train and evaluate models
nb_model = MultinomialNB()
lr_model = LogisticRegression()

nb_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

nb_accuracy = accuracy_score(y_test, nb_model.predict(X_test))
lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))

print(f"Naive Bayes accuracy: {nb_accuracy:.2f}")
print(f"Logistic Regression accuracy: {lr_accuracy:.2f}")
```

Slide 11: Real-Life Example: Image Generation vs. Classification

Comparing a generative model (Variational Autoencoder) for image generation with a discriminative model (Convolutional Neural Network) for image classification using the MNIST dataset.

Slide 12: Real-Life Example: Image Generation vs. Classification

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Simplified Variational Autoencoder (VAE) for image generation
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 40)
        )
        self.decoder = nn.Sequential(
            nn.Linear(20, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = x.view(-1, 784)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Simplified CNN for image classification
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist = MNIST('./data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=64, shuffle=True)

# Initialize models
vae = VAE()
cnn = CNN()

print("VAE can generate new images")
print("CNN can classify existing images")
```

Slide 13: Choosing Between Generative and Discriminative Models

The choice between generative and discriminative models depends on the specific task and available data. Generative models are preferred for tasks like data generation, anomaly detection, and handling missing data. Discriminative models excel in classification and regression tasks with sufficient labeled data.

Slide 14: Choosing Between Generative and Discriminative Models

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

def plot_learning_curves(X, y):
    train_sizes = np.linspace(0.1, 1.0, 5)
    
    _, train_scores_nb, test_scores_nb = learning_curve(
        GaussianNB(), X, y, train_sizes=train_sizes, cv=5)
    
    _, train_scores_lr, test_scores_lr = learning_curve(
        LogisticRegression(), X, y, train_sizes=train_sizes, cv=5)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, np.mean(train_scores_nb, axis=1), label='NB Train')
    plt.plot(train_sizes, np.mean(test_scores_nb, axis=1), label='NB Test')
    plt.plot(train_sizes, np.mean(train_scores_lr, axis=1), label='LR Train')
    plt.plot(train_sizes, np.mean(test_scores_lr, axis=1), label='LR Test')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves: Generative (NB) vs Discriminative (LR)')
    plt.legend()
    plt.show()

# Generate sample data
X = np.random.randn(1000, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

plot_learning_curves(X, y)
```

Slide 15: Hybrid Approaches: Combining Generative and Discriminative Models

Researchers have explored hybrid approaches that combine the strengths of both generative and discriminative models. These methods aim to leverage the complementary advantages of each approach for improved performance and robustness.

```python
import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HybridModel, self).__init__()
        self.generative = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.discriminative = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        gen_output = self.generative(x)
        disc_output = self.discriminative(x)
        return gen_output, disc_output

# Example usage
input_dim = 10
hidden_dim = 20
output_dim = 2

model = HybridModel(input_dim, hidden_dim, output_dim)
x = torch.randn(5, input_dim)
gen_out, disc_out = model(x)

print("Generative output shape:", gen_out.shape)
print("Discriminative output shape:", disc_out.shape)
```

Slide 16: Conclusion and Future Directions

Generative and discriminative models each have their strengths and weaknesses. Generative models excel in tasks requiring data generation and handling missing information, while discriminative models are often preferred for classification and regression tasks. Future research may focus on developing more sophisticated hybrid approaches and exploring the synergies between these two paradigms.

```python
import matplotlib.pyplot as plt
import numpy as np

# Visualizing the complementary nature of generative and discriminative models
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], 'r--', label='Ideal performance')
plt.plot([0, 0.7, 1], [0, 0.8, 0.9], 'b-', label='Discriminative')
plt.plot([0, 0.3, 1], [0, 0.5, 0.95], 'g-', label='Generative')
plt.plot([0, 0.5, 1], [0, 0.85, 0.98], 'm-', label='Hybrid')

plt.xlabel('Model Complexity')
plt.ylabel('Performance')
plt.title('Conceptual Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 17: Additional Resources

For those interested in diving deeper into the topic of generative and discriminative models, here are some valuable resources:

1. "On Discriminative vs. Generative Classifiers: A comparison of logistic regression and naive Bayes" by A. Ng and M. Jordan (2002). Available at: [https://arxiv.org/abs/cs/0409058](https://arxiv.org/abs/cs/0409058)
2. "Generative Adversarial Nets" by I. Goodfellow et al. (2014). Available at: [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)
3. "An Introduction to Variational Autoencoders" by D. Kingma and M. Welling (2019). Available at: [https://arxiv.org/abs/1906.02691](https://arxiv.org/abs/1906.02691)
4. "A Tutorial on Energy-Based Learning" by Y. LeCun et al. (2006). Available at: [https://cs.nyu.edu/~yann/research/ebm/](https://cs.nyu.edu/~yann/research/ebm/)

These papers provide in-depth discussions on various aspects of generative and discriminative models, including theoretical foundations and practical applications.

