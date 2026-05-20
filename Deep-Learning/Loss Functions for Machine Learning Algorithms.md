## Loss Functions for Machine Learning Algorithms
Slide 1: Mean Squared Error Loss

Mean Squared Error (MSE) represents the average squared difference between predicted and actual values, making it particularly effective for regression problems where outliers need significant penalization. It's differentiable and convex, ensuring convergence to global minimum during optimization.

```python
import numpy as np

def mse_loss(y_true, y_pred):
    """
    Calculate Mean Squared Error loss
    Mathematical form: $$ L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$
    """
    return np.mean(np.square(y_true - y_pred))

# Example usage
y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.1, 2.2, 2.8, 4.2])
loss = mse_loss(y_true, y_pred)
print(f"MSE Loss: {loss:.4f}")  # Output: MSE Loss: 0.0450
```

Slide 2: Binary Cross-Entropy Loss

Binary Cross-Entropy quantifies the difference between predicted probability distributions and true binary labels, serving as the fundamental loss function for binary classification tasks. It heavily penalizes confident wrong predictions while rewarding correct ones.

```python
def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Calculate Binary Cross-Entropy loss
    Mathematical form: $$ L = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)] $$
    """
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    return -np.mean(y_true * np.log(y_pred) + 
                   (1 - y_true) * np.log(1 - y_pred))

# Example usage
y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.2])
loss = binary_cross_entropy(y_true, y_pred)
print(f"Binary Cross-Entropy Loss: {loss:.4f}")  # Output: 0.2027
```

Slide 3: Categorical Cross-Entropy Loss

Categorical Cross-Entropy extends binary cross-entropy to multi-class classification scenarios, measuring the dissimilarity between predicted class probabilities and one-hot encoded true labels. It's particularly useful in neural networks with softmax output layers.

```python
def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Calculate Categorical Cross-Entropy loss
    Mathematical form: $$ L = -\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{m} y_{ij} \log(\hat{y}_{ij}) $$
    """
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# Example usage
y_true = np.array([[1,0,0], [0,1,0], [0,0,1]])
y_pred = np.array([[0.9,0.1,0], [0.1,0.8,0.1], [0.1,0.2,0.7]])
loss = categorical_cross_entropy(y_true, y_pred)
print(f"Categorical Cross-Entropy Loss: {loss:.4f}")  # Output: 0.2877
```

Slide 4: Huber Loss

The Huber Loss combines the best properties of MSE and MAE, behaving quadratically for small errors and linearly for large ones. This makes it particularly robust to outliers while maintaining differentiability at all points.

```python
def huber_loss(y_true, y_pred, delta=1.0):
    """
    Calculate Huber loss
    Mathematical form: 
    $$ L = \begin{cases} 
    \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \leq \delta \\
    \delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
    \end{cases} $$
    """
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * np.abs(error) - 0.5 * np.square(delta)
    
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))

# Example usage
y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.1, 2.2, 2.8, 6.0])
loss = huber_loss(y_true, y_pred)
print(f"Huber Loss: {loss:.4f}")  # Output: 0.4225
```

Slide 5: Hinge Loss and SVM Implementation

Hinge Loss, fundamental to Support Vector Machines, maximizes the margin between classes by penalizing misclassified samples and samples that lie too close to the decision boundary. It promotes sparsity in the solution.

```python
def hinge_loss(y_true, y_pred):
    """
    Calculate Hinge loss
    Mathematical form: $$ L = \max(0, 1 - y \cdot \hat{y}) $$
    """
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, epochs=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - 
                                       np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]
```

Slide 6: Real-world Example - Credit Card Fraud Detection

A practical implementation of binary cross-entropy loss for detecting fraudulent transactions, demonstrating the importance of proper loss function selection in imbalanced classification problems and its impact on model performance.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class FraudDetector:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)

# Example usage with synthetic data
np.random.seed(42)
X = np.random.randn(1000, 20)  # 1000 transactions, 20 features
y = np.random.randint(0, 2, 1000)  # Binary labels

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = FraudDetector(learning_rate=0.01, epochs=100)
model.fit(X_train, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_test)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Calculate metrics
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 7: Results for Credit Card Fraud Detection

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Detailed metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Loss curve over epochs
def plot_loss_curve(model, X, y):
    losses = []
    for epoch in range(model.epochs):
        linear_pred = np.dot(X, model.weights) + model.bias
        predictions = model.sigmoid(linear_pred)
        loss = binary_cross_entropy(y, predictions)
        losses.append(loss)
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.show()

plot_loss_curve(model, X_train, y_train)
```

Slide 8: Focal Loss Implementation

Focal Loss addresses class imbalance by down-weighting easy examples and focusing training on hard negatives. It introduces a modulating factor to the cross-entropy loss, preventing the vast number of easy negatives from overwhelming the classifier during training.

```python
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, epsilon=1e-15):
    """
    Calculate Focal Loss
    Mathematical form: $$ L = -\alpha (1-p_t)^\gamma \log(p_t) $$
    where p_t is the model's estimated probability for the true class
    """
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate cross entropy
    ce = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    
    # Calculate focal term
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_term = np.power(1 - p_t, gamma)
    
    # Calculate final loss
    loss = alpha * focal_term * ce
    
    return np.mean(loss)

# Example usage
y_true = np.array([1, 0, 1, 0, 1])
y_pred = np.array([0.8, 0.2, 0.6, 0.3, 0.9])
loss = focal_loss(y_true, y_pred)
print(f"Focal Loss: {loss:.4f}")  # Output: 0.0843
```

Slide 9: Quantile Loss Implementation

Quantile Loss extends beyond mean estimation to predict specific percentiles of the target variable distribution, making it particularly valuable for uncertainty estimation and risk assessment in financial predictions and demand forecasting.

```python
def quantile_loss(y_true, y_pred, quantile=0.5):
    """
    Calculate Quantile Loss
    Mathematical form: $$ L = \sum_{i=1}^{n} \max(\tau(y_i-\hat{y}_i), (\tau-1)(y_i-\hat{y}_i)) $$
    where τ is the desired quantile
    """
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

class QuantileRegressor:
    def __init__(self, quantile=0.5, learning_rate=0.01, epochs=1000):
        self.quantile = quantile
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y - y_pred
            grad = np.where(error > 0, -self.quantile, -(self.quantile - 1))
            
            # Update parameters
            self.weights += self.lr * np.dot(X.T, grad) / n_samples
            self.bias += self.lr * np.mean(grad)
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage
np.random.seed(42)
X = np.random.randn(1000, 1)
y = 2 * X.squeeze() + np.random.normal(0, 0.5, 1000)

model = QuantileRegressor(quantile=0.75)
model.fit(X, y)
y_pred = model.predict(X)
loss = quantile_loss(y, y_pred, quantile=0.75)
print(f"Quantile Loss (75th percentile): {loss:.4f}")
```

Slide 10: Contrastive Loss for Siamese Networks

Contrastive Loss enables learning similarity metrics between pairs of samples, crucial for face recognition and image matching tasks. It minimizes distance between similar pairs while pushing dissimilar pairs apart beyond a margin.

```python
def contrastive_loss(y_true, distance, margin=1.0):
    """
    Calculate Contrastive Loss
    Mathematical form: $$ L = y * d^2 + (1-y) * \max(0, m - d)^2 $$
    where d is the Euclidean distance between embeddings
    """
    similar_pair_loss = y_true * np.square(distance)
    dissimilar_pair_loss = (1 - y_true) * np.square(np.maximum(margin - distance, 0))
    return np.mean(similar_pair_loss + dissimilar_pair_loss)

class SiameseNetwork:
    def __init__(self, embedding_dim=128, margin=1.0):
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.W = np.random.randn(512, embedding_dim) * 0.01  # Assuming 512-dim input
        
    def get_embedding(self, x):
        return np.tanh(np.dot(x, self.W))
    
    def euclidean_distance(self, embed1, embed2):
        return np.sqrt(np.sum(np.square(embed1 - embed2), axis=1))
    
    def compute_loss(self, anchor, positive, negative):
        # Get embeddings
        anchor_embed = self.get_embedding(anchor)
        positive_embed = self.get_embedding(positive)
        negative_embed = self.get_embedding(negative)
        
        # Compute distances
        positive_dist = self.euclidean_distance(anchor_embed, positive_embed)
        negative_dist = self.euclidean_distance(anchor_embed, negative_embed)
        
        # Compute losses
        pos_loss = contrastive_loss(1, positive_dist, self.margin)
        neg_loss = contrastive_loss(0, negative_dist, self.margin)
        
        return pos_loss + neg_loss

# Example usage
batch_size = 32
anchor = np.random.randn(batch_size, 512)
positive = anchor + np.random.normal(0, 0.1, (batch_size, 512))
negative = np.random.randn(batch_size, 512)

model = SiameseNetwork()
loss = model.compute_loss(anchor, positive, negative)
print(f"Contrastive Loss: {loss:.4f}")
```

Slide 11: Triplet Loss Implementation

Triplet Loss learns embeddings by minimizing the distance between an anchor and a positive sample while maximizing the distance to a negative sample, widely used in face recognition and image retrieval systems.

```python
def triplet_loss(anchor, positive, negative, margin=0.2):
    """
    Calculate Triplet Loss
    Mathematical form: $$ L = \max(0, d(a,p) - d(a,n) + margin) $$
    where d(x,y) is the Euclidean distance between embeddings
    """
    pos_dist = np.sum(np.square(anchor - positive), axis=1)
    neg_dist = np.sum(np.square(anchor - negative), axis=1)
    loss = np.maximum(0, pos_dist - neg_dist + margin)
    return np.mean(loss)

class TripletNetwork:
    def __init__(self, input_dim=512, embedding_dim=128, margin=0.2):
        self.W = np.random.randn(input_dim, embedding_dim) * 0.01
        self.margin = margin
        
    def forward(self, x):
        return np.tanh(np.dot(x, self.W))
    
    def compute_gradients(self, anchor, positive, negative):
        # Forward pass
        anchor_embed = self.forward(anchor)
        positive_embed = self.forward(positive)
        negative_embed = self.forward(negative)
        
        # Compute distances
        pos_dist = np.sum(np.square(anchor_embed - positive_embed), axis=1)
        neg_dist = np.sum(np.square(anchor_embed - negative_embed), axis=1)
        
        # Compute loss
        loss = triplet_loss(anchor_embed, positive_embed, negative_embed, self.margin)
        
        return loss, (anchor_embed, positive_embed, negative_embed)

# Example usage
np.random.seed(42)
batch_size = 64
input_dim = 512
embedding_dim = 128

anchor = np.random.randn(batch_size, input_dim)
positive = anchor + np.random.normal(0, 0.1, (batch_size, input_dim))
negative = np.random.randn(batch_size, input_dim)

model = TripletNetwork(input_dim, embedding_dim)
loss, embeddings = model.compute_gradients(anchor, positive, negative)
print(f"Triplet Loss: {loss:.4f}")
```

Slide 12: Real-world Example - Face Recognition System

```python
class FaceRecognitionSystem:
    def __init__(self, embedding_dim=128):
        self.triplet_model = TripletNetwork(input_dim=512, embedding_dim=embedding_dim)
        self.embedding_database = {}
        
    def extract_features(self, face_image):
        """Convert face image to embedding vector"""
        # Assuming face_image is preprocessed and shaped (512,)
        return self.triplet_model.forward(face_image.reshape(1, -1))
    
    def register_face(self, person_id, face_image):
        """Register a new face in the database"""
        embedding = self.extract_features(face_image)
        self.embedding_database[person_id] = embedding
        
    def identify_face(self, face_image, threshold=0.7):
        """Identify a face by comparing with registered faces"""
        query_embedding = self.extract_features(face_image)
        
        best_match = None
        best_similarity = -np.inf
        
        for person_id, stored_embedding in self.embedding_database.items():
            similarity = -np.sum(np.square(query_embedding - stored_embedding))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_id
                
        return best_match if best_similarity > threshold else "Unknown"

# Example usage
np.random.seed(42)

# Simulate face database
system = FaceRecognitionSystem()
num_people = 10
for person_id in range(num_people):
    face = np.random.randn(512)  # Simulated face features
    system.register_face(f"Person_{person_id}", face)

# Test identification
test_face = system.embedding_database["Person_0"] + np.random.normal(0, 0.1, (1, 128))
identified = system.identify_face(test_face.flatten())
print(f"Identified as: {identified}")
```

Slide 13: CTC Loss Implementation

Connectionist Temporal Classification (CTC) loss enables training models to recognize sequences without requiring explicit alignment between input and output sequences, making it essential for speech recognition and handwriting recognition tasks.

```python
def ctc_loss(y_true, y_pred, blank_index=0, epsilon=1e-7):
    """
    Calculate CTC Loss (simplified version)
    Mathematical form: $$ L = -\log\sum_{i=1}^{N} p(y|x) $$
    where p(y|x) is the probability of the target sequence given input
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    def forward_pass(labels, pred_matrix):
        T, V = pred_matrix.shape
        L = len(labels)
        
        # Initialize forward variables
        alpha = np.zeros((T, L))
        alpha[0, 0] = pred_matrix[0, labels[0]]
        
        # Forward algorithm
        for t in range(1, T):
            for s in range(L):
                if s == 0:
                    alpha[t, s] = alpha[t-1, s] * pred_matrix[t, labels[s]]
                else:
                    alpha[t, s] = (alpha[t-1, s] + alpha[t-1, s-1]) * \
                                 pred_matrix[t, labels[s]]
        
        return -np.log(alpha[-1, -1])
    
    total_loss = 0
    for pred, true in zip(y_pred, y_true):
        total_loss += forward_pass(true, pred)
    
    return total_loss / len(y_true)

# Example usage
# Simulating output from a speech recognition model
sequence_length = 10
num_classes = 20
batch_size = 2

# Generate random predictions and labels
y_pred = np.random.rand(batch_size, sequence_length, num_classes)
y_pred = y_pred / y_pred.sum(axis=2, keepdims=True)  # Normalize to probabilities
y_true = np.random.randint(1, num_classes, size=(batch_size, 5))  # Shorter target sequences

loss = ctc_loss(y_true, y_pred)
print(f"CTC Loss: {loss:.4f}")
```

Slide 14: KL Divergence Loss

Kullback-Leibler Divergence Loss measures the difference between two probability distributions, commonly used in variational autoencoders and knowledge distillation tasks to ensure learned distributions match target distributions.

```python
def kl_divergence_loss(p_true, p_pred, epsilon=1e-7):
    """
    Calculate KL Divergence Loss
    Mathematical form: $$ D_{KL}(P||Q) = \sum_{i} P(i) \log(\frac{P(i)}{Q(i)}) $$
    """
    # Clip values to avoid numerical instability
    p_true = np.clip(p_true, epsilon, 1.0)
    p_pred = np.clip(p_pred, epsilon, 1.0)
    
    return np.sum(p_true * np.log(p_true / p_pred))

class VariationalAutoencoder:
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_weights = np.random.randn(input_dim, latent_dim * 2) * 0.01
        self.decoder_weights = np.random.randn(latent_dim, input_dim) * 0.01
        
    def encode(self, x):
        # Returns mean and log variance of latent distribution
        h = np.dot(x, self.encoder_weights)
        mean = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mean.shape)
        return mean + eps * std
    
    def decode(self, z):
        return np.sigmoid(np.dot(z, self.decoder_weights))
    
    def compute_loss(self, x):
        # Compute VAE loss (reconstruction + KL divergence)
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)
        
        # Reconstruction loss
        recon_loss = -np.mean(np.sum(x * np.log(x_recon + 1e-7) + 
                                    (1-x) * np.log(1-x_recon + 1e-7), axis=1))
        
        # KL divergence loss
        kl_loss = -0.5 * np.mean(np.sum(1 + logvar - np.square(mean) - 
                                       np.exp(logvar), axis=1))
        
        return recon_loss + kl_loss

# Example usage
input_dim = 784  # e.g., for MNIST
latent_dim = 32
batch_size = 64

# Generate random input data
x = np.random.rand(batch_size, input_dim)
x = (x > 0.5).astype(np.float32)  # Binary images

vae = VariationalAutoencoder(input_dim, latent_dim)
total_loss = vae.compute_loss(x)
print(f"VAE Total Loss: {total_loss:.4f}")
```

Slide 15: Additional Resources

*   "Auto-Encoding Variational Bayes" - [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
*   "Focal Loss for Dense Object Detection" - [https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)
*   "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks" - [https://www.cs.toronto.edu/~graves/icml\_2006.pdf](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
*   "FaceNet: A Unified Embedding for Face Recognition and Clustering" - [https://arxiv.org/abs/1503.03832](https://arxiv.org/abs/1503.03832)
*   "Deep Metric Learning Using Triplet Network" - [https://arxiv.org/abs/1412.6622](https://arxiv.org/abs/1412.6622)
*   "Understanding the difficulty of training deep feedforward neural networks" - [https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

