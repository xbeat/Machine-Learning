## Understanding and Avoiding Overfitting in Machine Learning
Slide 1: Understanding Overfitting Fundamentals

Overfitting occurs when a machine learning model learns the training data too precisely, including its noise and fluctuations. This results in poor generalization to new, unseen data. The model essentially memorizes the training examples rather than learning the underlying patterns that would enable it to make accurate predictions on new data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate synthetic data with noise
np.random.seed(42)
X = 2 * np.random.rand(15, 1)
y = 3 * X + np.random.randn(15, 1) * 0.2

# Create and fit models with different polynomial degrees
degrees = [1, 15]  # Linear vs High-degree polynomial
X_test = np.linspace(0, 2, 100).reshape(100, 1)

for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    X_test_poly = poly_features.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_test_poly)
    
    plt.scatter(X, y, color='blue', label='Training data')
    plt.plot(X_test, y_pred, color='red' if degree > 1 else 'green',
             label=f'Polynomial degree {degree}')
    
plt.legend()
plt.show()
```

Slide 2: Cross-Validation for Overfitting Detection

Cross-validation provides a robust method to detect overfitting by evaluating model performance on different subsets of the data. By comparing training and validation scores across multiple folds, we can identify when a model starts to overfit and optimize its complexity accordingly.

```python
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
import numpy as np

# Generate more complex synthetic data
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Create pipelines with different model complexities
architectures = [(10,), (100, 100), (500, 500, 500)]

for hidden_layers in architectures:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=hidden_layers, 
                            max_iter=2000, random_state=42))
    ])
    
    # Perform cross-validation
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Architecture {hidden_layers}:")
    print(f"Mean MSE: {-scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

Slide 3: Learning Curves Analysis

Learning curves provide visual insights into model overfitting by plotting training and validation performance against training set size. A widening gap between training and validation scores as data increases indicates overfitting, while converging scores suggest good generalization.

```python
from sklearn.model_selection import learning_curve
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curves(X, y, estimator, title):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='neg_mean_squared_error')
    
    train_scores_mean = -train_scores.mean(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)
    
    plt.plot(train_sizes, train_scores_mean, label='Training score')
    plt.plot(train_sizes, val_scores_mean, label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('Mean Squared Error')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid()

# Generate data
np.random.seed(42)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Create and plot learning curves for different kernel complexities
svr = SVR(kernel='rbf', C=100)
plot_learning_curves(X, y, svr, 'Learning Curves (RBF Kernel)')
plt.show()
```

Slide 4: Regularization Techniques

Regularization helps prevent overfitting by adding penalties to the model's complexity. This implementation demonstrates L1 (Lasso) and L2 (Ridge) regularization effects on a linear model, showing how different penalty strengths affect model coefficients and prediction accuracy.

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Generate high-dimensional data
np.random.seed(42)
n_samples, n_features = 100, 20
X = np.random.randn(n_samples, n_features)
true_coefficients = np.array([1 if i < 5 else 0 for i in range(n_features)])
y = X @ true_coefficients + np.random.randn(n_samples) * 0.1

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models with different regularization strengths
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
ridge_coefs = []
lasso_coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha)
    
    ridge.fit(X_scaled, y)
    lasso.fit(X_scaled, y)
    
    ridge_coefs.append(ridge.coef_)
    lasso_coefs.append(lasso.coef_)

# Plot coefficients paths
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title('Ridge coefficients paths')
for coef in zip(*ridge_coefs):
    plt.plot(alphas, coef)
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Coefficient value')

plt.subplot(1, 2, 2)
plt.title('Lasso coefficients paths')
for coef in zip(*lasso_coefs):
    plt.plot(alphas, coef)
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Coefficient value')
plt.tight_layout()
plt.show()
```

Slide 5: Early Stopping Implementation

Early stopping prevents overfitting by monitoring validation performance during training and stopping when performance begins to degrade. This implementation shows a custom early stopping mechanism that tracks validation loss and stops training when the model starts to overfit.

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class EarlyStoppingTrainer:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.early_stop = False
        
    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Example usage with a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Generate synthetic data
X = torch.randn(1000, 10)
y = torch.sum(X[:, :3], dim=1, keepdim=True) + torch.randn(1000, 1) * 0.1

# Train/validation split
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Training with early stopping
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
early_stopping = EarlyStoppingTrainer()

train_losses = []
val_losses = []

for epoch in range(100):
    model.train()
    train_loss = criterion(model(X_train), y_train)
    
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val)
    
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    
    early_stopping(val_loss.item())
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        break
```

Slide 6: Dropout Regularization

Dropout is a powerful regularization technique that randomly deactivates neurons during training to prevent co-adaptation. This implementation demonstrates how dropout layers affect model performance and help prevent overfitting in deep neural networks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class DropoutNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x, training=True):
        x = F.relu(self.fc1(x))
        if training:
            x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        if training:
            x = self.dropout2(x)
        return self.fc3(x)

# Generate synthetic image data
batch_size = 64
n_samples = 1000
X = torch.randn(n_samples, 784)  # Simulated MNIST-like data
y = torch.randint(0, 10, (n_samples,))

# Create data loaders
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Training loop
model = DropoutNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs, training=False)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

for epoch in range(10):
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)
    print(f'Epoch {epoch+1}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')
```

Slide 7: K-Fold Cross-Validation Implementation

K-fold cross-validation provides a robust method for model evaluation and hyperparameter tuning while preventing overfitting. This implementation shows a custom k-fold validation framework with performance tracking across folds.

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

class CustomKFoldValidator:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
    def validate(self, X, y, model):
        mse_scores = []
        r2_scores = []
        fold_predictions = []
        
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            mse_scores.append(mse)
            r2_scores.append(r2)
            fold_predictions.append((val_idx, y_pred))
            
            print(f'Fold {fold}: MSE = {mse:.4f}, R2 = {r2:.4f}')
        
        # Sort and concatenate predictions
        fold_predictions.sort(key=lambda x: x[0][0])
        all_predictions = np.concatenate([pred for _, pred in fold_predictions])
        
        return {
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'predictions': all_predictions
        }

# Example usage
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.sum(X[:, :3], axis=1) + np.random.randn(1000) * 0.1

# Create and validate model
model = RandomForestRegressor(n_estimators=100, random_state=42)
validator = CustomKFoldValidator()
results = validator.validate(X, y, model)

print("\nOverall Results:")
print(f"Mean MSE: {results['mse_mean']:.4f} (+/- {results['mse_std']:.4f})")
print(f"Mean R2: {results['r2_mean']:.4f} (+/- {results['r2_std']:.4f})")
```

Slide 8: Model Complexity Analysis

This implementation demonstrates how to analyze model complexity through the lens of bias-variance tradeoff. We'll create a framework to measure how different model complexities affect overfitting by computing training and validation errors across multiple complexity levels.

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class ComplexityAnalyzer:
    def __init__(self, max_degree=15):
        self.max_degree = max_degree
        self.training_errors = []
        self.validation_errors = []
        self.degrees = range(1, max_degree + 1)
        
    def analyze(self, X, y, test_size=0.2):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        for degree in self.degrees:
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
            X_val_poly = poly.transform(X_val.reshape(-1, 1))
            
            # Fit model
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            
            # Calculate errors
            train_pred = model.predict(X_train_poly)
            val_pred = model.predict(X_val_poly)
            
            train_error = mean_squared_error(y_train, train_pred)
            val_error = mean_squared_error(y_val, val_pred)
            
            self.training_errors.append(train_error)
            self.validation_errors.append(val_error)
            
        return {
            'degrees': self.degrees,
            'training_errors': self.training_errors,
            'validation_errors': self.validation_errors
        }

# Generate synthetic data with noise
np.random.seed(42)
X = np.linspace(0, 1, 100)
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.1, X.shape[0])

# Analyze complexity
analyzer = ComplexityAnalyzer()
results = analyzer.analyze(X, y)

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(results['degrees'], results['training_errors'], 
         label='Training Error', marker='o')
plt.plot(results['degrees'], results['validation_errors'], 
         label='Validation Error', marker='s')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Model Complexity Analysis')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 9: Real-world Example: Credit Card Fraud Detection

This implementation demonstrates overfitting prevention in a real-world credit card fraud detection scenario, utilizing multiple techniques including data preprocessing, feature scaling, and balanced sampling.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

class FraudDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced'
        )
        self.smote = SMOTE(random_state=42)
        
    def preprocess(self, X, y=None, training=True):
        if training:
            X_scaled = self.scaler.fit_transform(X)
            if y is not None:
                X_resampled, y_resampled = self.smote.fit_resample(X_scaled, y)
                return X_resampled, y_resampled
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
        
    def train_evaluate(self, X, y):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Preprocess training data
        X_train_processed, y_train_processed = self.preprocess(X_train, y_train)
        
        # Train model
        self.model.fit(X_train_processed, y_train_processed)
        
        # Evaluate on test set
        X_test_processed = self.preprocess(X_test, training=False)
        y_pred = self.model.predict(X_test_processed)
        y_pred_proba = self.model.predict_proba(X_test_processed)[:, 1]
        
        return {
            'classification_report': classification_report(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

# Generate synthetic credit card transaction data
np.random.seed(42)
n_samples = 10000
n_features = 30

# Create synthetic features
X = np.random.randn(n_samples, n_features)
# Create imbalanced classes (0.3% fraud rate)
y = np.zeros(n_samples)
fraud_indices = np.random.choice(n_samples, size=int(0.003 * n_samples), replace=False)
y[fraud_indices] = 1

# Train and evaluate model
detector = FraudDetector()
results = detector.train_evaluate(X, y)

print("Model Performance:")
print(results['classification_report'])
print(f"\nROC AUC Score: {results['roc_auc']:.4f}")
```

Slide 10: Real-world Example: Stock Price Prediction

This implementation showcases overfitting prevention in time series prediction using LSTM networks with various regularization techniques. The model includes proper sequence handling and time-based cross-validation.

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx):
        return (
            self.data[idx:idx+self.sequence_length],
            self.data[idx+self.sequence_length]
        )

class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

def train_model(model, train_loader, val_loader, epochs=50):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            train_loss += loss.item()
            
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch.unsqueeze(1)).item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        print(f'Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, '
              f'Val Loss = {val_losses[-1]:.4f}')
    
    return train_losses, val_losses

# Generate synthetic stock price data
np.random.seed(42)
days = 1000
price = 100
prices = [price]

for _ in range(days-1):
    change = np.random.normal(0, 1)
    price *= (1 + change/100)
    prices.append(price)

# Prepare data
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(np.array(prices).reshape(-1, 1))

# Create datasets
sequence_length = 10
dataset = TimeSeriesDataset(scaled_prices, sequence_length)
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, len(dataset) - train_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize and train model
model = StockPredictor(
    input_size=1,
    hidden_size=50,
    num_layers=2,
    dropout=0.2
)

train_losses, val_losses = train_model(model, train_loader, val_loader)
```

Slide 11: Ensemble Methods for Overfitting Prevention

This implementation demonstrates how ensemble methods can help prevent overfitting by combining multiple models. It includes bagging, boosting, and stacking approaches with cross-validation.

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import numpy as np

class EnsembleRegressor:
    def __init__(self):
        self.base_models = [
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=10)),
            ('gbm', GradientBoostingRegressor(n_estimators=100, max_depth=5)),
            ('svr', SVR(kernel='rbf', C=1.0))
        ]
        self.meta_model = LassoCV()
        self.trained_models = []
        
    def create_meta_features(self, X, y=None, training=True):
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models):
            if training:
                # Use cross-validation predictions for training
                cv_predictions = np.zeros(X.shape[0])
                for train_idx, val_idx in KFold(5).split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train = y[train_idx]
                    
                    model.fit(X_train, y_train)
                    cv_predictions[val_idx] = model.predict(X_val)
                
                meta_features[:, i] = cv_predictions
                
                # Retrain on full dataset
                model.fit(X, y)
                self.trained_models.append(model)
            else:
                # For test data, use predictions from models trained on full training data
                meta_features[:, i] = self.trained_models[i].predict(X)
                
        return meta_features
    
    def fit(self, X, y):
        meta_features = self.create_meta_features(X, y, training=True)
        self.meta_model.fit(meta_features, y)
        return self
    
    def predict(self, X):
        meta_features = self.create_meta_features(X, training=False)
        return self.meta_model.predict(meta_features)

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.sin(X[:, 0]) + 0.1 * np.random.randn(1000)

# Split data
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Train and evaluate ensemble
ensemble = EnsembleRegressor()
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

# Evaluate individual models and ensemble
for name, model in ensemble.base_models:
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"{name} MSE: {-scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

ensemble_score = mean_squared_error(y_test, y_pred)
print(f"Ensemble MSE: {ensemble_score:.4f}")
```

Slide 12: Bayesian Approach to Overfitting

Bayesian methods provide a principled approach to preventing overfitting by incorporating prior knowledge and uncertainty estimation. This implementation demonstrates Bayesian Neural Networks with variational inference for robust predictions.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(torch.zeros(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu)
        nn.init.constant_(self.weight_sigma, -3)
        nn.init.constant_(self.bias_mu, 0)
        nn.init.constant_(self.bias_sigma, -3)
        
    def forward(self, x):
        weight = Normal(self.weight_mu, F.softplus(self.weight_sigma))
        bias = Normal(self.bias_mu, F.softplus(self.bias_sigma))
        
        w = weight.rsample()
        b = bias.rsample()
        
        return F.linear(x, w, b)
    
    def kl_loss(self):
        weight = Normal(self.weight_mu, F.softplus(self.weight_sigma))
        bias = Normal(self.bias_mu, F.softplus(self.bias_sigma))
        
        weight_prior = Normal(0, 1)
        bias_prior = Normal(0, 1)
        
        return (kl_divergence(weight, weight_prior).sum() + 
                kl_divergence(bias, bias_prior).sum())

class BayesianNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = BayesianLinear(input_size, hidden_size)
        self.layer2 = BayesianLinear(hidden_size, output_size)
        
    def forward(self, x, num_samples=1):
        outputs = []
        for _ in range(num_samples):
            h = F.relu(self.layer1(x))
            output = self.layer2(h)
            outputs.append(output)
        return torch.stack(outputs)
    
    def kl_loss(self):
        return self.layer1.kl_loss() + self.layer2.kl_loss()

# Training function
def train_bayesian_model(model, X_train, y_train, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass with multiple samples
        outputs = model(X_train, num_samples=5)
        
        # Negative log likelihood loss
        likelihood = Normal(outputs.mean(0), outputs.std(0))
        nll_loss = -likelihood.log_prob(y_train).mean()
        
        # Add KL divergence
        kl_loss = model.kl_loss() / len(X_train)
        
        # Total loss
        loss = nll_loss + kl_loss
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Generate synthetic data
torch.manual_seed(42)
X = torch.randn(100, 1)
y = X.pow(2) + 0.1 * torch.randn(100, 1)

# Create and train model
model = BayesianNN(1, 10, 1)
train_bayesian_model(model, X, y)

# Make predictions with uncertainty
with torch.no_grad():
    X_test = torch.linspace(-2, 2, 100).reshape(-1, 1)
    predictions = model(X_test, num_samples=100)
    mean_pred = predictions.mean(0)
    std_pred = predictions.std(0)
```

Slide 13: Additional Resources

*   Bayesian Neural Networks for Uncertainty Estimation
    *   [https://arxiv.org/abs/1505.05424](https://arxiv.org/abs/1505.05424)
    *   Search: "Weight Uncertainty in Neural Networks"
*   Comprehensive Survey on Overfitting Prevention
    *   [https://arxiv.org/abs/1901.06566](https://arxiv.org/abs/1901.06566)
    *   Search: "A Survey of Regularization Methods in Deep Learning"
*   Advanced Ensemble Methods for Model Regularization
    *   [https://arxiv.org/abs/2004.13302](https://arxiv.org/abs/2004.13302)
    *   Search: "Modern Ensemble Methods in Deep Learning"
*   Time Series Prediction with Regularization
    *   Search: "Deep Learning for Time Series Forecasting"
    *   Recommended reading on time series specific overfitting prevention
*   Real-world Applications of Overfitting Prevention
    *   Search: "Applications of Regularization in Production ML Systems"
    *   Case studies from industry implementations

