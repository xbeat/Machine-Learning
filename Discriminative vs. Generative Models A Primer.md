## Discriminative vs. Generative Models A Primer
Slide 1: Understanding Discriminative Models Fundamentals

Discriminative models directly learn the mapping between input features and output labels by modeling conditional probability P(Y|X). They focus on decision boundaries between classes rather than understanding the underlying data distribution, making them efficient for classification tasks.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate synthetic binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X[:5])
probabilities = model.predict_proba(X[:5])

print("Predictions:", predictions)
print("Probabilities:", probabilities)
```

Slide 2: Mathematical Foundation of Discriminative Models

The core principle of discriminative models revolves around learning P(Y|X) directly. For binary classification using logistic regression, the model learns the decision boundary through the logistic function, transforming linear combinations of features into probabilities.

```python
# Mathematical representation in code form
def logistic_function(z):
    """
    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
    """
    return 1 / (1 + np.exp(-z))

def decision_boundary(X, weights, bias):
    """
    $$z = w^T x + b$$
    """
    return np.dot(X, weights) + bias

# Example usage
X = np.array([[1, 2], [2, 3], [3, 4]])
weights = np.array([0.5, -0.5])
bias = 0.1

z = decision_boundary(X, weights, bias)
probabilities = logistic_function(z)
print("Decision boundary values:", z)
print("Probabilities:", probabilities)
```

Slide 3: Custom Implementation of Linear Discriminative Model

A from-scratch implementation of a simple linear discriminative model demonstrates the core concepts of gradient descent optimization and decision boundary learning for binary classification problems in a clear, mathematical way.

```python
class LinearDiscriminativeModel:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            # Forward pass
            linear_output = np.dot(X, self.weights) + self.bias
            predictions = 1 / (1 + np.exp(-linear_output))
            
            # Gradient computation
            dz = predictions - y
            dw = (1/n_samples) * np.dot(X.T, dz)
            db = (1/n_samples) * np.sum(dz)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return (linear_output >= 0).astype(int)

# Example usage
X_train = np.random.randn(100, 2)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

model = LinearDiscriminativeModel()
model.fit(X_train, y_train)
predictions = model.predict(X_train[:5])
print("Predictions:", predictions)
```

Slide 4: Understanding Generative Models Architecture

Generative models learn the joint probability distribution P(X,Y) of inputs and outputs, enabling them to generate new samples from the learned distribution. This fundamental difference from discriminative models allows for both classification and data generation.

```python
import torch
import torch.nn as nn

class SimpleGenerativeModel(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.generator(z)

# Initialize model
latent_dim = 10
output_dim = 784  # 28x28 image
model = SimpleGenerativeModel(latent_dim, output_dim)

# Generate samples
z = torch.randn(5, latent_dim)
generated_samples = model(z)
print("Generated samples shape:", generated_samples.shape)
```

Slide 5: Implementing a Basic Generative Adversarial Network (GAN)

A GAN consists of two competing networks: a generator that creates synthetic data and a discriminator that distinguishes between real and fake samples. This architecture enables the model to learn complex data distributions through adversarial training.

```python
import torch.optim as optim

class SimpleGAN:
    def __init__(self, data_dim, latent_dim):
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, data_dim),
            nn.Tanh()
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        self.loss_fn = nn.BCELoss()
        
    def train_step(self, real_data, batch_size):
        # Train Discriminator
        z = torch.randn(batch_size, latent_dim)
        fake_data = self.generator(z)
        
        d_real = self.discriminator(real_data)
        d_fake = self.discriminator(fake_data.detach())
        
        d_loss_real = self.loss_fn(d_real, torch.ones_like(d_real))
        d_loss_fake = self.loss_fn(d_fake, torch.zeros_like(d_fake))
        d_loss = d_loss_real + d_loss_fake
        
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        fake_data = self.generator(z)
        g_loss = self.loss_fn(self.discriminator(fake_data), torch.ones(batch_size, 1))
        
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()

# Usage example
data_dim = 28 * 28
latent_dim = 100
gan = SimpleGAN(data_dim, latent_dim)
```

Slide 6: Comparing Model Performance Metrics

Understanding the performance differences between discriminative and generative models requires specific evaluation metrics for each type. Discriminative models focus on classification metrics, while generative models need quality and diversity measures.

```python
def evaluate_models(discriminative_model, generative_model, test_data, test_labels):
    # Discriminative model metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    disc_predictions = discriminative_model.predict(test_data)
    accuracy = accuracy_score(test_labels, disc_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, disc_predictions, average='weighted'
    )
    
    # Generative model evaluation (using FID score for image data)
    def calculate_fid(real_features, generated_features):
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
        
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
        return np.real(fid)
    
    # Generate samples for evaluation
    z = torch.randn(len(test_data), latent_dim)
    generated_samples = generative_model(z).detach().numpy()
    
    fid_score = calculate_fid(test_data, generated_samples)
    
    return {
        'discriminative': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'generative': {
            'fid_score': fid_score
        }
    }

# Example usage
results = evaluate_models(discriminative_model, generative_model, X_test, y_test)
print("Evaluation Results:", results)
```

Slide 7: Real-world Application - Credit Card Fraud Detection

Implementing a discriminative model for fraud detection demonstrates the practical application of decision boundaries in financial transaction classification, with emphasis on handling imbalanced datasets and feature engineering.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class FraudDetectionSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(class_weight='balanced')
        
    def preprocess_data(self, X, y=None, training=True):
        # Feature engineering
        amount_stats = np.concatenate([
            X['amount'].values.reshape(-1, 1),
            X['amount'].rolling(window=3).mean().values.reshape(-1, 1),
            X['amount'].rolling(window=3).std().values.reshape(-1, 1)
        ], axis=1)
        
        if training:
            self.scaler.fit(amount_stats)
        
        scaled_features = self.scaler.transform(amount_stats)
        
        if training:
            # Handle imbalanced dataset
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(scaled_features, y)
            return X_resampled, y_resampled
        
        return scaled_features
    
    def train(self, X, y):
        X_processed, y_processed = self.preprocess_data(X, y, training=True)
        self.model.fit(X_processed, y_processed)
    
    def predict(self, X):
        X_processed = self.preprocess_data(X, training=False)
        return self.model.predict(X_processed)

# Example usage with synthetic data
import pandas as pd
np.random.seed(42)

# Create synthetic transaction data
n_samples = 10000
data = pd.DataFrame({
    'amount': np.random.exponential(100, n_samples),
    'is_fraud': np.random.binomial(1, 0.001, n_samples)
})

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('is_fraud', axis=1),
    data['is_fraud'],
    test_size=0.2
)

fraud_detector = FraudDetectionSystem()
fraud_detector.train(X_train, y_train)
predictions = fraud_detector.predict(X_test)
```

Slide 8: Implementing Naive Bayes as a Generative Model

Naive Bayes is a simple yet effective generative model that learns the joint probability distribution P(X,Y) by applying Bayes' theorem with the "naive" assumption of feature independence, making it particularly useful for text classification tasks.

```python
class NaiveBayesFromScratch:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        
        # Calculate class probabilities
        for c in classes:
            self.class_probs[c] = np.mean(y == c)
            
            # Calculate feature probabilities for each class
            class_samples = X[y == c]
            self.feature_probs[c] = {}
            
            for feature in range(n_features):
                # Using Gaussian distribution for continuous features
                mean = np.mean(class_samples[:, feature])
                std = np.std(class_samples[:, feature]) + 1e-6
                self.feature_probs[c][feature] = (mean, std)
    
    def _calculate_likelihood(self, x, mean, std):
        """
        $$P(x|\mu,\sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
        """
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent
    
    def predict(self, X):
        predictions = []
        for x in X:
            class_scores = {}
            for c in self.class_probs:
                # Calculate log probability to avoid numerical underflow
                log_prob = np.log(self.class_probs[c])
                for feature, value in enumerate(x):
                    mean, std = self.feature_probs[c][feature]
                    likelihood = self._calculate_likelihood(value, mean, std)
                    log_prob += np.log(likelihood + 1e-10)
                class_scores[c] = log_prob
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)

# Example usage
X = np.random.randn(1000, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

model = NaiveBayesFromScratch()
model.fit(X, y)
predictions = model.predict(X[:5])
print("Sample predictions:", predictions)
```

Slide 9: Real-world Application - Text Generation with LSTM

Implementing a character-level LSTM as a generative model showcases how sequence generation works in practice, demonstrating the model's ability to learn and reproduce text patterns.

```python
class TextGeneratorLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden
    
    def generate_text(self, start_char_idx, char_to_idx, idx_to_char, length=100):
        self.eval()
        with torch.no_grad():
            current_idx = torch.tensor([[start_char_idx]])
            text = [idx_to_char[start_char_idx]]
            hidden = None
            
            for _ in range(length):
                output, hidden = self(current_idx, hidden)
                probs = torch.softmax(output[0, -1], dim=0)
                next_idx = torch.multinomial(probs, 1)
                text.append(idx_to_char[next_idx.item()])
                current_idx = next_idx.view(1, 1)
            
        return ''.join(text)

# Example usage
sample_text = """This is a sample text for training our generative model.
The model will learn to generate similar text patterns."""

# Create character mappings
chars = sorted(list(set(sample_text)))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# Initialize and train model
model = TextGeneratorLSTM(
    vocab_size=len(chars),
    embedding_dim=32,
    hidden_dim=64
)

# Generate sample text
generated_text = model.generate_text(
    start_char_idx=char_to_idx['T'],
    char_to_idx=char_to_idx,
    idx_to_char=idx_to_char
)
print("Generated text:", generated_text)
```

Slide 10: Performance Visualization for Model Comparison

Implementing comprehensive visualization tools to compare discriminative and generative models helps understand their different strengths and trade-offs across various metrics and scenarios.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

class ModelPerformanceVisualizer:
    def __init__(self, discriminative_model, generative_model):
        self.disc_model = discriminative_model
        self.gen_model = generative_model
        
    def plot_comparison(self, X_test, y_test, generated_samples):
        plt.figure(figsize=(15, 5))
        
        # Discriminative model performance
        plt.subplot(131)
        y_pred = self.disc_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Discriminative Model\nConfusion Matrix')
        
        # ROC curve
        plt.subplot(132)
        y_prob = self.disc_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # Generated samples distribution
        plt.subplot(133)
        sns.kdeplot(data=generated_samples[:, 0], label='Generated')
        sns.kdeplot(data=X_test[:, 0], label='Real')
        plt.title('Data Distribution\nComparison')
        plt.legend()
        
        plt.tight_layout()
        return plt.gcf()

# Example usage
import numpy as np

# Generate synthetic data
np.random.seed(42)
X_test = np.random.randn(1000, 2)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
generated_samples = np.random.randn(1000, 2) * 1.2  # Simulated generated samples

visualizer = ModelPerformanceVisualizer(
    discriminative_model=LogisticRegression().fit(X_test, y_test),
    generative_model=None  # Placeholder for generative model
)

fig = visualizer.plot_comparison(X_test, y_test, generated_samples)
plt.close()  # Close the figure to prevent display
```

Slide 11: Advanced Feature Engineering for Both Model Types

Feature engineering techniques differ significantly between discriminative and generative models, with discriminative models focusing on decision-relevant features and generative models needing complete distribution information.

```python
class AdvancedFeatureProcessor:
    def __init__(self, model_type='discriminative'):
        self.model_type = model_type
        self.scalers = {}
        self.feature_stats = {}
        
    def engineer_features(self, X, training=True):
        if self.model_type == 'discriminative':
            return self._engineer_discriminative(X, training)
        return self._engineer_generative(X, training)
    
    def _engineer_discriminative(self, X, training):
        transformed_features = {}
        
        # Create interaction terms
        for i in range(X.shape[1]):
            for j in range(i+1, X.shape[1]):
                feature_name = f'interaction_{i}_{j}'
                transformed_features[feature_name] = X[:, i] * X[:, j]
        
        # Add polynomial features
        for i in range(X.shape[1]):
            feature_name = f'poly_{i}'
            transformed_features[feature_name] = X[:, i] ** 2
        
        # Normalize all features
        if training:
            for name, feature in transformed_features.items():
                self.scalers[name] = StandardScaler()
                transformed_features[name] = self.scalers[name].fit_transform(
                    feature.reshape(-1, 1)
                )
        else:
            for name, feature in transformed_features.items():
                transformed_features[name] = self.scalers[name].transform(
                    feature.reshape(-1, 1)
                )
                
        return np.hstack([X] + [v for v in transformed_features.values()])
    
    def _engineer_generative(self, X, training):
        if training:
            # Calculate distribution parameters
            self.feature_stats['means'] = np.mean(X, axis=0)
            self.feature_stats['stds'] = np.std(X, axis=0)
            self.feature_stats['correlations'] = np.corrcoef(X.T)
        
        # Standardize features while preserving correlations
        X_centered = X - self.feature_stats['means']
        X_scaled = X_centered / self.feature_stats['stds']
        
        # Add higher-order moments
        skewness = np.mean(X_scaled ** 3, axis=0)
        kurtosis = np.mean(X_scaled ** 4, axis=0) - 3
        
        return np.hstack([
            X_scaled,
            skewness.reshape(1, -1),
            kurtosis.reshape(1, -1)
        ])

# Example usage
X = np.random.randn(1000, 4)
processor = AdvancedFeatureProcessor(model_type='discriminative')
X_disc = processor.engineer_features(X, training=True)

processor_gen = AdvancedFeatureProcessor(model_type='generative')
X_gen = processor_gen.engineer_features(X, training=True)

print("Discriminative features shape:", X_disc.shape)
print("Generative features shape:", X_gen.shape)
```

Slide 12: Model Validation and Cross-Validation Strategies

Different validation strategies are required for discriminative and generative models, with discriminative models focusing on prediction accuracy and generative models requiring distribution-based validation metrics.

```python
class ModelValidator:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        
    def validate_discriminative(self, model, X, y):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, f1_score, precision_score
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
        metrics = {
            'accuracy': [],
            'f1': [],
            'precision': []
        }
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['f1'].append(f1_score(y_val, y_pred, average='weighted'))
            metrics['precision'].append(precision_score(y_val, y_pred, average='weighted'))
        
        return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}
    
    def validate_generative(self, model, X_real):
        # Split data for validation
        indices = np.arange(len(X_real))
        np.random.shuffle(indices)
        split_size = len(indices) // self.n_splits
        
        metrics = {
            'kl_divergence': [],
            'wasserstein_distance': []
        }
        
        for i in range(self.n_splits):
            val_indices = indices[i*split_size:(i+1)*split_size]
            X_val = X_real[val_indices]
            
            # Generate samples
            X_gen = model.generate(len(X_val))
            
            # Calculate KL divergence
            kl_div = self._calculate_kl_divergence(X_val, X_gen)
            metrics['kl_divergence'].append(kl_div)
            
            # Calculate Wasserstein distance
            w_dist = self._calculate_wasserstein(X_val, X_gen)
            metrics['wasserstein_distance'].append(w_dist)
        
        return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}
    
    def _calculate_kl_divergence(self, P, Q):
        """
        $$D_{KL}(P||Q) = \sum_i P(i) \log(\frac{P(i)}{Q(i)})$$
        """
        from scipy.stats import gaussian_kde
        
        kde_P = gaussian_kde(P.T)
        kde_Q = gaussian_kde(Q.T)
        
        x = np.linspace(P.min(), P.max(), 1000)
        P_dist = kde_P(x)
        Q_dist = kde_Q(x)
        
        return np.sum(P_dist * np.log(P_dist / (Q_dist + 1e-10)))
    
    def _calculate_wasserstein(self, P, Q):
        """
        Approximates Wasserstein distance using sorted samples
        """
        P_sorted = np.sort(P, axis=0)
        Q_sorted = np.sort(Q, axis=0)
        return np.mean(np.abs(P_sorted - Q_sorted))

# Example usage
X = np.random.randn(1000, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

validator = ModelValidator(n_splits=5)

# Validate discriminative model
disc_model = LogisticRegression()
disc_results = validator.validate_discriminative(disc_model, X, y)
print("Discriminative model validation results:", disc_results)

# Mock generative model for example
class MockGenerativeModel:
    def generate(self, n_samples):
        return np.random.randn(n_samples, 4)

gen_model = MockGenerativeModel()
gen_results = validator.validate_generative(gen_model, X)
print("Generative model validation results:", gen_results)
```

Slide 13: Model Deployment and Production Considerations

When deploying discriminative and generative models to production, different optimization strategies and monitoring approaches are needed to ensure robust performance and efficient inference.

```python
class ModelDeployment:
    def __init__(self, discriminative_model=None, generative_model=None):
        self.disc_model = discriminative_model
        self.gen_model = generative_model
        self.monitoring_stats = {'disc': {}, 'gen': {}}
        
    def optimize_for_production(self):
        if self.disc_model:
            self._optimize_discriminative()
        if self.gen_model:
            self._optimize_generative()
    
    def _optimize_discriminative(self):
        """
        Optimize discriminative model for production
        """
        import joblib
        from sklearn.tree import DecisionTreeClassifier
        
        # Convert complex model to decision tree for production
        if hasattr(self.disc_model, 'predict_proba'):
            X_sample = np.random.randn(1000, self.disc_model.n_features_in_)
            y_pred = self.disc_model.predict(X_sample)
            
            prod_model = DecisionTreeClassifier(max_depth=5)
            prod_model.fit(X_sample, y_pred)
            
            # Save model
            joblib.dump(prod_model, 'prod_discriminative_model.joblib')
            
            # Update monitoring stats
            self.monitoring_stats['disc']['model_size'] = os.path.getsize('prod_discriminative_model.joblib')
            self.monitoring_stats['disc']['inference_time'] = self._measure_inference_time(prod_model, X_sample)
    
    def _optimize_generative(self):
        """
        Optimize generative model for production
        """
        import torch.jit
        
        # Convert to TorchScript for production
        if isinstance(self.gen_model, nn.Module):
            example_input = torch.randn(1, self.gen_model.input_size)
            traced_model = torch.jit.trace(self.gen_model, example_input)
            
            # Save model
            traced_model.save('prod_generative_model.pt')
            
            # Update monitoring stats
            self.monitoring_stats['gen']['model_size'] = os.path.getsize('prod_generative_model.pt')
            self.monitoring_stats['gen']['inference_time'] = self._measure_inference_time(traced_model, example_input)
    
    def _measure_inference_time(self, model, sample_input, n_runs=100):
        import time
        
        times = []
        for _ in range(n_runs):
            start = time.time()
            _ = model(sample_input)
            times.append(time.time() - start)
            
        return np.mean(times)
    
    def monitor_performance(self, X_test, y_test=None):
        metrics = {}
        
        if self.disc_model and y_test is not None:
            y_pred = self.disc_model.predict(X_test)
            metrics['discriminative'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'latency': self._measure_inference_time(self.disc_model, X_test[:1])
            }
        
        if self.gen_model:
            gen_samples = self.gen_model.generate(len(X_test))
            metrics['generative'] = {
                'sample_quality': self._calculate_sample_quality(X_test, gen_samples),
                'latency': self._measure_inference_time(self.gen_model, X_test[:1])
            }
        
        return metrics

# Example usage
deployment = ModelDeployment(
    discriminative_model=LogisticRegression().fit(X, y),
    generative_model=MockGenerativeModel()
)

deployment.optimize_for_production()
metrics = deployment.monitor_performance(X, y)
print("Production metrics:", metrics)
```

Slide 14: Hybrid Model Architecture Implementation

A hybrid approach combining discriminative and generative models can leverage the strengths of both paradigms, using the generative model to augment training data and the discriminative model for final predictions.

```python
class HybridModel:
    def __init__(self, feature_dim, latent_dim=32):
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        
        # Generative component
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.Tanh()
        )
        
        # Discriminative component
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        self.gen_optimizer = optim.Adam(self.generator.parameters())
        self.disc_optimizer = optim.Adam(self.classifier.parameters())
        
    def augment_data(self, X, y, augmentation_factor=0.5):
        """
        $$P(x_{aug}) = P(x|G(z))P(G(z))$$
        """
        n_augment = int(len(X) * augmentation_factor)
        z = torch.randn(n_augment, self.latent_dim)
        
        with torch.no_grad():
            X_gen = self.generator(z).numpy()
        
        # Combine real and generated data
        X_augmented = np.vstack([X, X_gen])
        y_augmented = np.concatenate([
            y,
            self.classifier(torch.tensor(X_gen, dtype=torch.float32))
            .argmax(dim=1).numpy()
        ])
        
        return X_augmented, y_augmented
    
    def fit(self, X, y, epochs=100, batch_size=32):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        for epoch in range(epochs):
            # Train generator
            z = torch.randn(batch_size, self.latent_dim)
            gen_samples = self.generator(z)
            
            gen_validity = self.classifier(gen_samples)
            g_loss = F.cross_entropy(gen_validity, 
                                   torch.randint(0, 2, (batch_size,)))
            
            self.gen_optimizer.zero_grad()
            g_loss.backward()
            self.gen_optimizer.step()
            
            # Train classifier
            X_batch = X[torch.randint(0, len(X), (batch_size,))]
            y_batch = y[torch.randint(0, len(y), (batch_size,))]
            
            pred = self.classifier(X_batch)
            d_loss = F.cross_entropy(pred, y_batch)
            
            self.disc_optimizer.zero_grad()
            d_loss.backward()
            self.disc_optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: G_loss = {g_loss.item():.4f}, "
                      f"D_loss = {d_loss.item():.4f}")
    
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.classifier(X)
            return logits.argmax(dim=1).numpy()

# Example usage
X = np.random.randn(1000, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

model = HybridModel(feature_dim=10)
model.fit(X, y)

# Test prediction
X_test = np.random.randn(100, 10)
predictions = model.predict(X_test)
print("Sample predictions:", predictions[:10])
```

Slide 15: Additional Resources

*   ArXiv: "A Survey of Deep Generative Models" - [https://arxiv.org/abs/2006.10863](https://arxiv.org/abs/2006.10863)
*   ArXiv: "On Discriminative vs. Generative Classifiers: A Comparison of Logistic Regression and Naive Bayes" - [https://arxiv.org/abs/2002.12062](https://arxiv.org/abs/2002.12062)
*   ArXiv: "Recent Advances in Deep Learning: A Comprehensive Survey" - [https://arxiv.org/abs/2012.01254](https://arxiv.org/abs/2012.01254)
*   Google Scholar search suggestions:
    *   "Hybrid discriminative-generative models"
    *   "Deep generative modeling advances"
    *   "Comparative analysis of discriminative and generative approaches"
*   Recommended textbooks:
    *   "Pattern Recognition and Machine Learning" by Christopher Bishop
    *   "Deep Learning" by Goodfellow, Bengio, and Courville

