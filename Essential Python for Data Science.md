## Essential Python for Data Science
Slide 1: Statistical Fundamentals in Python

The foundation of data science rests on statistical measures that help us understand data distribution, central tendency, and dispersion. These metrics form the basis for more complex analyses and machine learning algorithms, making them crucial for any data scientist.

```python
import numpy as np

# Function to calculate comprehensive statistical measures
def statistical_analysis(data):
    # Basic measures of central tendency
    mean = np.mean(data)
    median = np.median(data)
    mode = np.bincount(data).argmax()
    
    # Measures of dispersion
    variance = np.var(data)
    std_dev = np.std(data)
    
    # Calculate quartiles and IQR
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    
    return {
        'mean': mean,
        'median': median,
        'mode': mode,
        'variance': variance,
        'std_dev': std_dev,
        'iqr': iqr
    }

# Example usage
data = np.random.normal(100, 15, 1000).astype(int)
results = statistical_analysis(data)
print("Statistical Analysis Results:")
for metric, value in results.items():
    print(f"{metric}: {value:.2f}")
```

Slide 2: Matrix Operations and Linear Algebra

Matrix operations underpin many machine learning algorithms, from simple linear regression to neural networks. Understanding these operations is crucial for implementing algorithms from scratch and optimizing computational efficiency.

```python
# Matrix operations implementation from scratch
def matrix_multiply(A, B):
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions incompatible")
    
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

# Example matrices
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

# Perform multiplication
result = matrix_multiply(A, B)
print("Matrix Multiplication Result:")
for row in result:
    print(row)

# Using NumPy for comparison
import numpy as np
np_result = np.dot(np.array(A), np.array(B))
print("\nNumPy Result:")
print(np_result)
```

Slide 3: Gradient Descent Implementation

Gradient descent is the cornerstone optimization algorithm in machine learning, used to minimize loss functions. This implementation demonstrates the mathematics behind the algorithm and its practical application in finding optimal parameters.

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m = len(y)
    theta = np.zeros(X.shape[1])
    cost_history = []
    
    for i in range(iterations):
        # Forward propagation
        prediction = np.dot(X, theta)
        
        # Calculate error
        error = prediction - y
        
        # Calculate gradients
        gradients = (1/m) * np.dot(X.T, error)
        
        # Update parameters
        theta -= learning_rate * gradients
        
        # Calculate cost
        cost = (1/(2*m)) * np.sum(error**2)
        cost_history.append(cost)
    
    return theta, cost_history

# Generate sample data
X = np.random.rand(100, 3)
y = 2*X[:,0] + 3*X[:,1] + 4*X[:,2] + np.random.randn(100)*0.1

# Add bias term
X = np.column_stack([np.ones(len(X)), X])

# Run gradient descent
theta, cost_history = gradient_descent(X, y)
print("Optimized parameters:", theta)
print("Final cost:", cost_history[-1])
```

Slide 4: Principal Component Analysis (PCA)

Principal Component Analysis is a fundamental dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving maximum variance. This implementation shows the mathematical foundation behind PCA.

```python
def pca_from_scratch(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Calculate covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components eigenvectors
    components = eigenvectors[:, :n_components]
    
    # Project data onto new space
    transformed_data = np.dot(X_centered, components)
    
    # Calculate explained variance ratio
    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return transformed_data, explained_variance_ratio

# Example usage
X = np.random.rand(100, 5)
transformed_data, explained_variance = pca_from_scratch(X, n_components=2)
print("Transformed data shape:", transformed_data.shape)
print("Explained variance ratio:", explained_variance)
```

Slide 5: K-Means Clustering Implementation

K-means clustering is an unsupervised learning algorithm that partitions data into k distinct clusters. This implementation demonstrates the iterative process of centroid calculation and cluster assignment.

```python
def kmeans_clustering(X, k, max_iters=100):
    # Randomly initialize centroids
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Store old centroids
        old_centroids = centroids.copy()
        
        # Update centroids
        for i in range(k):
            centroids[i] = X[labels == i].mean(axis=0)
        
        # Check convergence
        if np.all(old_centroids == centroids):
            break
    
    return labels, centroids

# Generate sample data
X = np.concatenate([
    np.random.normal(0, 1, (100, 2)),
    np.random.normal(5, 1, (100, 2)),
    np.random.normal(-5, 1, (100, 2))
])

# Apply k-means
labels, centroids = kmeans_clustering(X, k=3)
print("Cluster centroids:", centroids)
```

Slide 6: Neural Network Architecture

Neural networks form the backbone of deep learning, consisting of interconnected layers of neurons that transform input data through non-linear activations. This implementation demonstrates a basic feedforward neural network with backpropagation.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers)-1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2/layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            net = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activations.append(self.sigmoid(net))
        return activations
    
    def backward(self, X, y, learning_rate=0.1):
        m = X.shape[0]
        activations = self.forward(X)
        
        # Backpropagation
        delta = activations[-1] - y
        for i in range(len(self.weights)-1, -1, -1):
            self.weights[i] -= learning_rate * np.dot(activations[i].T, delta)/m
            self.biases[i] -= learning_rate * np.sum(delta, axis=0, keepdims=True)/m
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])

# Example usage
nn = NeuralNetwork([2, 4, 1])
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Training
for _ in range(1000):
    nn.backward(X, y)

# Predictions
print("Predictions:", nn.forward(X)[-1])
```

Slide 7: Support Vector Machine Implementation

Support Vector Machines find the optimal hyperplane that maximizes the margin between classes. This implementation shows the mathematical principles behind SVM optimization using the Sequential Minimal Optimization (SMO) algorithm.

```python
def svm_train(X, y, C=1.0, tol=0.001, max_passes=5):
    m, n = X.shape
    alphas = np.zeros(m)
    b = 0
    passes = 0
    
    def kernel(x1, x2):
        return np.dot(x1, x2)  # Linear kernel
    
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            Ei = np.sum(alphas * y * kernel(X, X[i])) + b - y[i]
            
            if ((y[i]*Ei < -tol and alphas[i] < C) or 
                (y[i]*Ei > tol and alphas[i] > 0)):
                
                # Select random j != i
                j = i
                while j == i:
                    j = np.random.randint(m)
                
                Ej = np.sum(alphas * y * kernel(X, X[j])) + b - y[j]
                
                # Save old alphas
                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]
                
                # Compute bounds
                if y[i] != y[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                
                if L == H:
                    continue
                
                # Compute eta
                eta = 2*kernel(X[i], X[j]) - kernel(X[i], X[i]) - kernel(X[j], X[j])
                if eta >= 0:
                    continue
                
                # Update alpha j
                alphas[j] -= y[j]*(Ei - Ej)/eta
                alphas[j] = min(H, max(L, alphas[j]))
                
                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue
                
                # Update alpha i
                alphas[i] += y[i]*y[j]*(alpha_j_old - alphas[j])
                
                # Update threshold b
                b1 = b - Ei - y[i]*(alphas[i] - alpha_i_old)*kernel(X[i], X[i]) - \
                     y[j]*(alphas[j] - alpha_j_old)*kernel(X[i], X[j])
                b2 = b - Ej - y[i]*(alphas[i] - alpha_i_old)*kernel(X[i], X[j]) - \
                     y[j]*(alphas[j] - alpha_j_old)*kernel(X[j], X[j])
                
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2)/2
                
                num_changed_alphas += 1
                
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
            
    return alphas, b

# Example usage
X = np.random.randn(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
alphas, b = svm_train(X, y)
print("Number of support vectors:", np.sum((alphas > 0) & (alphas < 1.0)))
```

Slide 8: Time Series Analysis

Time series analysis is crucial for understanding temporal patterns and making predictions based on historical data. This implementation shows various time series components including trend, seasonality, and residuals.

```python
def time_series_decomposition(data, period=7):
    # Calculate trend using moving average
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    
    trend = moving_average(data, period)
    
    # Pad trend to match original data length
    pad_start = (len(data) - len(trend)) // 2
    pad_end = len(data) - len(trend) - pad_start
    trend = np.pad(trend, (pad_start, pad_end), mode='edge')
    
    # Calculate seasonal component
    detrended = data - trend
    seasonal = np.zeros_like(data)
    for i in range(period):
        seasonal[i::period] = np.mean(detrended[i::period])
    
    # Calculate residuals
    residuals = data - trend - seasonal
    
    return {
        'original': data,
        'trend': trend,
        'seasonal': seasonal,
        'residuals': residuals
    }

# Generate sample time series data
np.random.seed(42)
t = np.linspace(0, 4*np.pi, 200)
trend = 0.1 * t
seasonal = 2 * np.sin(t)
noise = np.random.normal(0, 0.5, len(t))
data = trend + seasonal + noise

# Perform decomposition
components = time_series_decomposition(data)
for name, component in components.items():
    print(f"{name} mean: {np.mean(component):.3f}")
```

Slide 9: Ensemble Methods Implementation

Ensemble methods combine multiple models to create more robust predictions. This implementation demonstrates bagging and boosting techniques, showing how they aggregate individual model predictions to improve overall performance.

```python
import numpy as np
from collections import Counter

class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        
    def predict(self, X):
        predictions = np.ones(len(X))
        if self.polarity == 1:
            predictions[X[:, self.feature_idx] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_idx] < self.threshold] = 1
        return predictions

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.stump_weights = []
        
    def fit(self, X, y):
        n_samples = len(X)
        w = np.full(n_samples, (1 / n_samples))
        
        for _ in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float('inf')
            
            # Find best threshold and feature
            for feature_idx in range(X.shape[1]):
                thresholds = np.unique(X[:, feature_idx])
                for threshold in thresholds:
                    for polarity in [-1, 1]:
                        stump.polarity = polarity
                        stump.threshold = threshold
                        stump.feature_idx = feature_idx
                        
                        predictions = stump.predict(X)
                        error = sum(w[y != predictions])
                        
                        if error < min_error:
                            min_error = error
                            best_stump = stump.predict(X)
                            best_feature_idx = feature_idx
                            best_threshold = threshold
                            best_polarity = polarity
            
            # Calculate stump weight
            eps = 1e-10
            stump_weight = 0.5 * np.log((1.0 - min_error + eps) / (min_error + eps))
            
            # Update sample weights
            w *= np.exp(-stump_weight * y * best_stump)
            w /= np.sum(w)
            
            # Save stump and its weight
            new_stump = DecisionStump()
            new_stump.feature_idx = best_feature_idx
            new_stump.threshold = best_threshold
            new_stump.polarity = best_polarity
            
            self.stumps.append(new_stump)
            self.stump_weights.append(stump_weight)
    
    def predict(self, X):
        stump_predictions = np.array([stump.predict(X) for stump in self.stumps])
        return np.sign(np.dot(self.stump_weights, stump_predictions))

# Example usage
X = np.random.randn(100, 2)
y = np.sign(X[:, 0] + X[:, 1])

# Train AdaBoost
model = AdaBoost(n_estimators=10)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 10: Natural Language Processing Fundamentals

Natural Language Processing combines statistical and linguistic approaches to analyze text data. This implementation shows basic NLP techniques including tokenization, TF-IDF calculation, and text classification.

```python
import re
from collections import defaultdict
import numpy as np

class NLPProcessor:
    def __init__(self):
        self.vocabulary = set()
        self.idf = {}
        self.label_map = {}
        
    def tokenize(self, text):
        """Convert text to lowercase and split into tokens"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def compute_tf(self, tokens):
        """Compute term frequency"""
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        # Normalize
        total_terms = len(tokens)
        return {term: freq/total_terms for term, freq in tf.items()}
    
    def compute_idf(self, documents):
        """Compute inverse document frequency"""
        doc_count = len(documents)
        term_doc_count = defaultdict(int)
        
        for doc in documents:
            tokens = set(self.tokenize(doc))
            for token in tokens:
                term_doc_count[token] += 1
                self.vocabulary.add(token)
        
        self.idf = {term: np.log(doc_count/(count + 1)) 
                   for term, count in term_doc_count.items()}
    
    def compute_tfidf(self, text):
        """Compute TF-IDF vector"""
        tokens = self.tokenize(text)
        tf = self.compute_tf(tokens)
        
        tfidf_vector = np.zeros(len(self.vocabulary))
        vocab_list = sorted(list(self.vocabulary))
        
        for i, term in enumerate(vocab_list):
            if term in tf:
                tfidf_vector[i] = tf[term] * self.idf.get(term, 0)
        
        return tfidf_vector
    
    def train_classifier(self, texts, labels):
        """Train a simple Naive Bayes classifier"""
        # Compute IDF for feature extraction
        self.compute_idf(texts)
        
        # Convert labels to numerical values
        unique_labels = set(labels)
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
        
        # Convert texts to TF-IDF vectors
        X = np.array([self.compute_tfidf(text) for text in texts])
        y = np.array([self.label_map[label] for label in labels])
        
        # Compute class probabilities and feature likelihoods
        self.class_probs = np.bincount(y) / len(y)
        self.feature_probs = []
        
        for c in range(len(unique_labels)):
            class_docs = X[y == c]
            # Add smoothing
            class_prob = (class_docs.sum(axis=0) + 1) / (len(class_docs) + 2)
            self.feature_probs.append(class_prob)
    
    def predict(self, text):
        """Predict class for new text"""
        x = self.compute_tfidf(text)
        
        # Compute log probabilities for each class
        log_probs = []
        for c in range(len(self.class_probs)):
            log_prob = np.log(self.class_probs[c])
            log_prob += np.sum(x * np.log(self.feature_probs[c]))
            log_probs.append(log_prob)
        
        # Return predicted class
        return max(enumerate(log_probs), key=lambda x: x[1])[0]

# Example usage
texts = [
    "machine learning is fascinating",
    "deep neural networks are powerful",
    "natural language processing with python",
    "statistical analysis and data science"
]
labels = ["ML", "DL", "NLP", "Stats"]

processor = NLPProcessor()
processor.train_classifier(texts, labels)

# Test prediction
test_text = "learning with neural networks"
prediction = processor.predict(test_text)
print(f"Predicted class: {list(processor.label_map.keys())[prediction]}")
```

Slide 11: Real-world Application - Credit Risk Analysis

This implementation demonstrates a complete machine learning pipeline for credit risk assessment, including data preprocessing, feature engineering, model training, and evaluation using multiple algorithms.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CreditRiskAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_importances = None
        
    def preprocess_data(self, data):
        """Preprocess credit data with feature engineering"""
        # Calculate financial ratios
        data['debt_to_income'] = data['total_debt'] / (data['annual_income'] + 1e-6)
        data['payment_to_income'] = data['monthly_payment'] / (data['annual_income']/12 + 1e-6)
        data['credit_utilization'] = data['current_balance'] / (data['credit_limit'] + 1e-6)
        
        # Create credit history features
        data['late_payment_ratio'] = data['late_payments'] / (data['credit_age_months'] + 1e-6)
        data['avg_monthly_transactions'] = data['total_transactions'] / (data['credit_age_months'] + 1e-6)
        
        # Handle missing values
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())
        
        return data
    
    def train_model(self, X, y):
        """Train ensemble model for credit risk prediction"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize base models
        models = {
            'logistic': LogisticRegression(random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gbm': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        # Train models
        predictions = {}
        for name, model in models.items():
            model.fit(X_scaled, y)
            predictions[name] = model.predict_proba(X_scaled)[:, 1]
            
        # Combine predictions using weighted average
        weights = {
            'logistic': 0.2,
            'rf': 0.4,
            'gbm': 0.4
        }
        
        final_predictions = np.zeros(len(y))
        for name, pred in predictions.items():
            final_predictions += weights[name] * pred
            
        # Calculate feature importance
        self.feature_importances = models['rf'].feature_importances_
        
        return final_predictions
    
    def evaluate_model(self, y_true, y_pred, threshold=0.5):
        """Evaluate model performance with multiple metrics"""
        y_binary = (y_pred >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_binary),
            'precision': precision_score(y_true, y_binary),
            'recall': recall_score(y_true, y_binary),
            'f1': f1_score(y_true, y_binary),
            'auc_roc': roc_auc_score(y_true, y_pred)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_binary)
        
        # Calculate cost matrix (example values)
        cost_matrix = {
            'fn': 500,  # Cost of false negative (failing to identify default)
            'fp': 100   # Cost of false positive (wrongly identifying default)
        }
        
        total_cost = (cm[1,0] * cost_matrix['fn'] + 
                     cm[0,1] * cost_matrix['fp'])
        
        metrics['total_cost'] = total_cost
        
        return metrics

# Example usage with synthetic data
np.random.seed(42)
n_samples = 1000

# Generate synthetic credit data
data = pd.DataFrame({
    'annual_income': np.random.normal(50000, 20000, n_samples),
    'total_debt': np.random.normal(20000, 10000, n_samples),
    'credit_score': np.random.normal(700, 50, n_samples),
    'credit_age_months': np.random.uniform(12, 240, n_samples),
    'late_payments': np.random.poisson(1, n_samples),
    'credit_limit': np.random.normal(15000, 5000, n_samples),
    'current_balance': np.random.normal(5000, 2000, n_samples),
    'monthly_payment': np.random.normal(500, 200, n_samples),
    'total_transactions': np.random.normal(100, 30, n_samples)
})

# Generate target variable (default probability)
risk_score = (0.3 * (data['total_debt'] / data['annual_income']) +
              0.2 * (data['late_payments'] / data['credit_age_months']) +
              0.5 * (1 - (data['credit_score'] - 300) / 550))

y = (risk_score > np.percentile(risk_score, 80)).astype(int)

# Initialize and run analysis
analyzer = CreditRiskAnalyzer()
processed_data = analyzer.preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(processed_data, y, test_size=0.2)

predictions = analyzer.train_model(X_train, y_train)
metrics = analyzer.evaluate_model(y_test, predictions)

print("Model Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

Slide 12: Deep Learning for Time Series Forecasting

This implementation showcases a deep learning approach to time series forecasting using LSTM networks, including data preprocessing, model architecture, and evaluation metrics specific to time series data.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

class TimeSeriesForecaster:
    def __init__(self, sequence_length=10, n_features=1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model()
        
    def _build_model(self):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(64, return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def prepare_sequences(self, data):
        """Convert time series data into sequences"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def train(self, data, validation_split=0.2, epochs=50, batch_size=32):
        """Train the model"""
        # Prepare sequences
        X, y = self.prepare_sequences(data)
        
        # Reshape for LSTM [samples, timesteps, features]
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        
        # Train model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def predict(self, data):
        """Generate predictions"""
        # Prepare input sequence
        X = data[-self.sequence_length:].reshape(1, self.sequence_length, self.n_features)
        return self.model.predict(X)[0]
    
    def evaluate(self, true_values, predictions):
        """Calculate forecast accuracy metrics"""
        metrics = {
            'mse': np.mean((true_values - predictions) ** 2),
            'rmse': np.sqrt(np.mean((true_values - predictions) ** 2)),
            'mae': np.mean(np.abs(true_values - predictions)),
            'mape': np.mean(np.abs((true_values - predictions) / true_values)) * 100
        }
        return metrics

# Example usage with synthetic data
np.random.seed(42)
t = np.linspace(0, 8*np.pi, 1000)
data = np.sin(t) + np.random.normal(0, 0.1, len(t))

# Initialize forecaster
forecaster = TimeSeriesForecaster(sequence_length=50, n_features=1)

# Split data into train and test
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Train model
history = forecaster.train(train_data, epochs=20)

# Make predictions
predictions = []
for i in range(len(test_data)):
    pred = forecaster.predict(data[train_size+i-50:train_size+i])
    predictions.append(pred)

# Evaluate performance
metrics = forecaster.evaluate(test_data, np.array(predictions))
print("\nForecast Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

Slide 13: Real-world Application - Image Classification with CNN

This implementation demonstrates a complete Convolutional Neural Network pipeline for image classification, including data augmentation, model architecture, and performance visualization techniques.

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self):
        """Build CNN architecture"""
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', padding='same', 
                  input_shape=self.input_shape),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def create_data_generators(self):
        """Create data generators with augmentation"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        return train_datagen, test_datagen
    
    def train(self, train_data, validation_data, epochs=50, batch_size=32):
        """Train the model"""
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
        return history
    
    def evaluate_model(self, test_data):
        """Evaluate model performance"""
        results = self.model.evaluate(test_data)
        metrics = dict(zip(self.model.metrics_names, results))
        
        # Calculate additional metrics
        predictions = self.model.predict(test_data)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_data.classes
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)
        
        class_metrics = {}
        for i in range(self.num_classes):
            class_metrics[f'class_{i}'] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i]
            }
        
        return {
            'overall_metrics': metrics,
            'class_metrics': class_metrics,
            'confusion_matrix': cm
        }
    
    def predict(self, image):
        """Make prediction for single image"""
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        return np.argmax(prediction[0])

# Example usage with synthetic data
input_shape = (64, 64, 3)
num_classes = 5

# Create synthetic dataset
X_train = np.random.rand(1000, 64, 64, 3)
y_train = np.random.randint(0, num_classes, 1000)
X_test = np.random.rand(200, 64, 64, 3)
y_test = np.random.randint(0, num_classes, 200)

# Initialize classifier
classifier = ImageClassifier(input_shape, num_classes)

# Create data generators
train_datagen, test_datagen = classifier.create_data_generators()

# Prepare data generators
train_generator = train_datagen.flow(
    X_train, keras.utils.to_categorical(y_train, num_classes),
    batch_size=32
)

test_generator = test_datagen.flow(
    X_test, keras.utils.to_categorical(y_test, num_classes),
    batch_size=32
)

# Train model
history = classifier.train(
    train_generator,
    validation_data=test_generator,
    epochs=10
)

# Evaluate model
evaluation = classifier.evaluate_model(test_generator)
print("\nModel Evaluation Results:")
print("\nOverall Metrics:")
for metric, value in evaluation['overall_metrics'].items():
    print(f"{metric}: {value:.4f}")

print("\nPer-class Metrics:")
for class_name, metrics in evaluation['class_metrics'].items():
    print(f"\n{class_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
```

Slide 14: Additional Resources

*   ArXiv papers for deep learning mathematics:
    *   "Mathematics of Deep Learning": [https://arxiv.org/abs/1712.04741](https://arxiv.org/abs/1712.04741)
    *   "Understanding Deep Learning requires Rethinking Generalization": [https://arxiv.org/abs/1611.03530](https://arxiv.org/abs/1611.03530)
    *   "A Theoretical Framework for Deep Learning": [https://arxiv.org/abs/1711.00501](https://arxiv.org/abs/1711.00501)
*   For implementing algorithms from scratch:
    *   "Numerical Optimization for Deep Learning": [https://arxiv.org/abs/1905.11692](https://arxiv.org/abs/1905.11692)
    *   Search "Implementation of Machine Learning Algorithms from Scratch" on Google Scholar
    *   Visit [https://paperswithcode.com](https://paperswithcode.com) for state-of-the-art implementations
*   Recommended online resources:
    *   Deep Learning Book by Ian Goodfellow et al.
    *   Machine Learning Mastery Blog
    *   Towards Data Science on Medium

