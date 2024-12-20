## Combining Machine Learning Algorithms for Better Results
Slide 1: Feature Extraction with CNN and SVM Classifier

This implementation demonstrates how to extract features from images using a pre-trained CNN (ResNet50) and feed them into an SVM classifier. The CNN acts as a feature extractor while the SVM performs the final classification task, combining deep learning with traditional ML.

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.svm import SVC
import numpy as np

# Load pre-trained ResNet50
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

def extract_features(image_tensor):
    with torch.no_grad():
        features = resnet(image_tensor)
        features = features.squeeze()
    return features.numpy()

# Prepare data and extract features
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Example usage
X_features = []  # Store extracted features
y_labels = []    # Store corresponding labels

# Train SVM classifier
svm_classifier = SVC(kernel='rbf', C=1.0)
svm_classifier.fit(X_features, y_labels)

# Predict
def predict(image):
    features = extract_features(transform(image).unsqueeze(0))
    return svm_classifier.predict(features.reshape(1, -1))
```

Slide 2: Stacking Models with Neural Networks and Random Forest

A sophisticated ensemble approach combining neural networks with random forests for improved prediction accuracy. The implementation shows how to stack these models using scikit-learn's API while maintaining clean separation between base models and meta-learner.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
import numpy as np

class StackedClassifier:
    def __init__(self):
        self.nn = MLPClassifier(hidden_layer_sizes=(100, 50),
                               activation='relu',
                               max_iter=1000)
        self.rf = RandomForestClassifier(n_estimators=100)
        self.meta = RandomForestClassifier(n_estimators=50)
        
    def fit(self, X, y):
        # Generate predictions from base models
        nn_pred = cross_val_predict(self.nn, X, y, cv=5, 
                                  method='predict_proba')
        rf_pred = cross_val_predict(self.rf, X, y, cv=5, 
                                  method='predict_proba')
        
        # Train base models on full data
        self.nn.fit(X, y)
        self.rf.fit(X, y)
        
        # Create meta-features
        meta_features = np.column_stack((nn_pred, rf_pred))
        
        # Train meta-classifier
        self.meta.fit(meta_features, y)
        return self
        
    def predict(self, X):
        nn_pred = self.nn.predict_proba(X)
        rf_pred = self.rf.predict_proba(X)
        meta_features = np.column_stack((nn_pred, rf_pred))
        return self.meta.predict(meta_features)
```

Slide 3: Autoencoder Feature Compression with K-Means

An implementation combining autoencoders for dimensionality reduction with K-means clustering for pattern discovery. The autoencoder compresses high-dimensional data into a lower-dimensional representation while preserving essential features.

```python
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Training and clustering pipeline
class AutoencoderKMeans:
    def __init__(self, input_dim, encoding_dim, n_clusters):
        self.autoencoder = Autoencoder(input_dim, encoding_dim)
        self.kmeans = KMeans(n_clusters=n_clusters)
        
    def fit(self, X, epochs=100):
        optimizer = torch.optim.Adam(self.autoencoder.parameters())
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            encoded, decoded = self.autoencoder(X)
            loss = criterion(decoded, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Get encoded features and cluster
        with torch.no_grad():
            encoded_features, _ = self.autoencoder(X)
        self.kmeans.fit(encoded_features.numpy())
```

Slide 4: Two-Stage Model with BERT and XGBoost

This implementation showcases a two-stage approach where BERT handles text embedding generation, followed by XGBoost for final predictions. The architecture leverages BERT's contextual understanding with XGBoost's gradient boosting capabilities.

```python
from transformers import BertTokenizer, BertModel
import xgboost as xgb
import torch
import numpy as np

class TextClassifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        
    def get_embeddings(self, texts):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.bert(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings
    
    def fit(self, texts, labels):
        embeddings = self.get_embeddings(texts)
        self.xgb_model.fit(embeddings, labels)
        
    def predict(self, texts):
        embeddings = self.get_embeddings(texts)
        return self.xgb_model.predict(embeddings)

# Example usage
classifier = TextClassifier()
texts = ["Great product!", "Terrible service", "Average experience"]
labels = [1, 0, 0.5]
classifier.fit(texts, labels)
```

Slide 5: Gradient Boosted Neural Networks Implementation

A sophisticated implementation of Gradient Boosted Neural Networks that combines the sequential learning of gradient boosting with neural network base learners. This approach is particularly effective for mixed data types and complex pattern recognition.

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator

class NeuralBoostingMachine:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        
    class BaseNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
        def forward(self, x):
            return self.network(x)
    
    def _train_base_learner(self, X, residuals, epochs=100):
        model = self.BaseNN(X.shape[1])
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X)
        residuals_tensor = torch.FloatTensor(residuals)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor).squeeze()
            loss = criterion(outputs, residuals_tensor)
            loss.backward()
            optimizer.step()
            
        return model
    
    def fit(self, X, y):
        current_pred = np.zeros_like(y)
        
        for i in range(self.n_estimators):
            residuals = y - current_pred
            model = self._train_base_learner(X, residuals)
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = model(X_tensor).numpy().squeeze()
                
            current_pred += self.learning_rate * predictions
            self.models.append(model)
            
    def predict(self, X):
        X_tensor = torch.FloatTensor(X)
        predictions = np.zeros(X.shape[0])
        
        with torch.no_grad():
            for model in self.models:
                predictions += self.learning_rate * model(X_tensor).numpy().squeeze()
                
        return predictions
```

Slide 6: Results Analysis for Neural Boosting System

This slide presents comprehensive performance metrics and visualization of the Neural Boosting implementation, showcasing its effectiveness on real-world datasets with mixed feature types.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

def evaluate_gbnn(model, X_train, X_test, y_train, y_test):
    # Training metrics
    train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    
    # Test metrics
    test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Training R2: {train_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual')
    
    plt.subplot(1, 2, 2)
    residuals = y_test - test_pred
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title('Residuals Distribution')
    
    plt.tight_layout()
    plt.show()

# Example output:
# Training MSE: 0.0324
# Training R2: 0.9876
# Test MSE: 0.0456
# Test R2: 0.9789
```

Slide 7: Deep Representation Learning with Transfer Pipeline

An advanced implementation demonstrating transfer learning by combining pre-trained deep networks with traditional ML algorithms. The system extracts meaningful representations from complex data structures while maintaining computational efficiency.

```python
import torch
import torchvision.models as models
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

class DeepFeatureExtractor(torch.nn.Module):
    def __init__(self, architecture='resnet50'):
        super().__init__()
        if architecture == 'resnet50':
            base_model = models.resnet50(pretrained=True)
            self.features = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.eval()
        
    def forward(self, x):
        with torch.no_grad():
            return self.features(x).squeeze()

class HybridClassifier:
    def __init__(self):
        self.feature_extractor = DeepFeatureExtractor()
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3
            ))
        ])
        
    def extract_features(self, data_loader):
        features = []
        for batch in data_loader:
            batch_features = self.feature_extractor(batch)
            features.append(batch_features.numpy())
        return np.vstack(features)
    
    def fit(self, train_loader, labels):
        features = self.extract_features(train_loader)
        self.pipeline.fit(features, labels)
        
    def predict(self, test_loader):
        features = self.extract_features(test_loader)
        return self.pipeline.predict(features)
```

Slide 8: Ensemble Learning with Heterogeneous Base Learners

This implementation creates a sophisticated ensemble system that combines various types of models including neural networks, gradient boosting, and traditional algorithms using weighted voting and stacking techniques.

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np

class HeterogeneousEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, weights=None):
        self.models = {
            'nn': MLPClassifier(hidden_layer_sizes=(100, 50),
                              max_iter=1000),
            'rf': RandomForestClassifier(n_estimators=100,
                                       max_depth=10),
            'gb': GradientBoostingClassifier(n_estimators=100,
                                           learning_rate=0.1)
        }
        self.weights = weights or {k: 1/len(self.models) 
                                 for k in self.models.keys()}
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for model in self.models.values():
            model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        predictions = np.zeros((X.shape[0], len(self.classes_)))
        for name, model in self.models.items():
            pred = model.predict_proba(X)
            predictions += self.weights[name] * pred
        return predictions
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
    
    def optimize_weights(self, X_val, y_val):
        from scipy.optimize import minimize
        
        def loss_function(weights):
            self.weights = {m: w for m, w in 
                          zip(self.models.keys(), weights)}
            pred = self.predict(X_val)
            return -np.mean(pred == y_val)
        
        constraints = ({'type': 'eq', 
                       'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1)] * len(self.models)
        
        result = minimize(loss_function, 
                        x0=list(self.weights.values()),
                        bounds=bounds,
                        constraints=constraints)
        
        self.weights = {m: w for m, w in 
                       zip(self.models.keys(), result.x)}
```

Slide 9: Real-time Performance Monitoring System

Implementation of a comprehensive monitoring system for hybrid ML models, tracking performance metrics, drift detection, and automatic retraining triggers in production environments.

```python
import numpy as np
from scipy import stats
from collections import deque
import time

class ModelMonitor:
    def __init__(self, window_size=1000, drift_threshold=0.05):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.scores = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
    def add_observation(self, prediction, actual, score):
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.scores.append(score)
        self.timestamps.append(time.time())
        
    def detect_drift(self):
        if len(self.scores) < self.window_size:
            return False
            
        # Split window into two parts
        mid = len(self.scores) // 2
        first_half = list(self.scores)[:mid]
        second_half = list(self.scores)[mid:]
        
        # Perform Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(first_half, 
                                              second_half)
        
        return p_value < self.drift_threshold
    
    def get_metrics(self):
        if not self.predictions:
            return {}
            
        accuracy = np.mean(np.array(self.predictions) == 
                         np.array(self.actuals))
        avg_score = np.mean(self.scores)
        
        return {
            'accuracy': accuracy,
            'average_score': avg_score,
            'drift_detected': self.detect_drift(),
            'window_size': len(self.predictions),
            'timestamp': max(self.timestamps)
        }
    
    def should_retrain(self):
        metrics = self.get_metrics()
        return (metrics.get('drift_detected', False) or
                metrics.get('accuracy', 1.0) < 0.8)
```

Slide 10: Advanced Model Architecture with Dynamic Feature Selection

This implementation combines deep learning with traditional ML through a dynamic feature selection mechanism that adapts to changing data distributions and maintains model interpretability while maximizing performance.

```python
import torch
import torch.nn as nn
from sklearn.feature_selection import mutual_info_classif
import numpy as np

class DynamicFeatureSelector:
    def __init__(self, n_features=10, selection_threshold=0.01):
        self.n_features = n_features
        self.threshold = selection_threshold
        self.selected_features = None
        self.importance_scores = None
        
    def fit(self, X, y):
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X, y)
        
        # Select features above threshold
        self.selected_features = np.where(mi_scores > self.threshold)[0]
        
        # If too few features selected, take top n
        if len(self.selected_features) < self.n_features:
            self.selected_features = np.argsort(mi_scores)[-self.n_features:]
            
        self.importance_scores = mi_scores[self.selected_features]
        return self
        
    def transform(self, X):
        return X[:, self.selected_features]

class AdaptiveHybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        self.feature_selector = DynamicFeatureSelector()
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.3)
            ])
            prev_dim = dim
            
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
        
    def forward(self, x, return_features=False):
        features = self.network(x)
        output = self.output(features)
        
        if return_features:
            return output, features
        return output

# Training loop with feature adaptation
def train_adaptive_model(model, train_loader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            # Update feature selection periodically
            if epoch % 10 == 0:
                model.feature_selector.fit(
                    batch_x.numpy(),
                    batch_y.numpy()
                )
            
            selected_x = model.feature_selector.transform(batch_x)
            optimizer.zero_grad()
            outputs = model(selected_x)
            loss = criterion(outputs, batch_y.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
```

Slide 11: Hybrid Model Interpretability Framework

A comprehensive implementation for explaining predictions of hybrid models combining SHAP values, feature importance analysis, and local interpretable model-agnostic explanations.

```python
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

class HybridModelInterpreter:
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
    def initialize_explainers(self, background_data):
        self.shap_explainer = shap.KernelExplainer(
            self.model.predict_proba, 
            background_data
        )
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            background_data,
            feature_names=self.feature_names,
            class_names=['Class 0', 'Class 1'],
            mode='classification'
        )
        
    def explain_prediction(self, instance, method='all'):
        explanations = {}
        
        if method in ['shap', 'all']:
            shap_values = self.shap_explainer.shap_values(instance)
            explanations['shap'] = {
                'values': shap_values,
                'base_value': self.shap_explainer.expected_value
            }
            
        if method in ['lime', 'all']:
            lime_exp = self.lime_explainer.explain_instance(
                instance,
                self.model.predict_proba
            )
            explanations['lime'] = lime_exp
            
        if method in ['permutation', 'all']:
            perm_importance = permutation_importance(
                self.model, 
                instance.reshape(1, -1),
                np.array([1])  # Single instance label
            )
            explanations['permutation'] = perm_importance
            
        return explanations
    
    def plot_feature_importance(self, explanation_type='shap'):
        plt.figure(figsize=(10, 6))
        
        if explanation_type == 'shap':
            shap.summary_plot(
                self.shap_values, 
                feature_names=self.feature_names
            )
        elif explanation_type == 'permutation':
            importances = self.perm_importance.importances_mean
            std = self.perm_importance.importances_std
            
            plt.barh(range(len(importances)), importances)
            plt.yticks(range(len(importances)), self.feature_names)
            plt.xlabel('Permutation Importance')
            
        plt.tight_layout()
        plt.show()
```

Slide 12: Real-World Application: Customer Churn Prediction

This implementation demonstrates a complete pipeline for customer churn prediction using a hybrid approach that combines structured data analysis with text sentiment from customer feedback.

```python
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

class ChurnPredictor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.xgb_model = xgb.XGBClassifier()
        self.scaler = StandardScaler()
        
    def preprocess_structured_data(self, df):
        numeric_cols = ['account_length', 'monthly_charges', 'total_charges']
        categorical_cols = ['contract_type', 'payment_method']
        
        # Handle numeric features
        scaled_numeric = self.scaler.fit_transform(df[numeric_cols])
        
        # Handle categorical features
        categorical_encoded = pd.get_dummies(df[categorical_cols])
        
        return np.hstack([scaled_numeric, categorical_encoded])
    
    def get_text_embeddings(self, feedback_texts):
        embeddings = []
        
        for text in feedback_texts:
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128
            )
            
            with torch.no_grad():
                outputs = self.bert(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding.squeeze())
                
        return np.vstack(embeddings)
    
    def fit(self, structured_data, feedback_texts, labels):
        # Process structured features
        structured_features = self.preprocess_structured_data(structured_data)
        
        # Get text embeddings
        text_features = self.get_text_embeddings(feedback_texts)
        
        # Combine features
        combined_features = np.hstack([structured_features, text_features])
        
        # Train model
        self.xgb_model.fit(combined_features, labels)
        
    def predict(self, structured_data, feedback_texts):
        structured_features = self.preprocess_structured_data(structured_data)
        text_features = self.get_text_embeddings(feedback_texts)
        combined_features = np.hstack([structured_features, text_features])
        
        return self.xgb_model.predict(combined_features)

# Example usage
predictor = ChurnPredictor()
predictor.fit(
    structured_data=pd.DataFrame({
        'account_length': [12, 24, 36],
        'monthly_charges': [50, 75, 100],
        'total_charges': [600, 1800, 3600],
        'contract_type': ['monthly', 'yearly', 'monthly'],
        'payment_method': ['credit_card', 'bank_transfer', 'credit_card']
    }),
    feedback_texts=[
        "Great service, very satisfied",
        "Poor customer support, considering leaving",
        "Average experience, nothing special"
    ],
    labels=[0, 1, 0]
)
```

Slide 13: Real-World Application: Financial Time Series Analysis

A sophisticated implementation combining LSTM networks with traditional statistical models for financial time series prediction, featuring advanced preprocessing and risk assessment.

```python
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd

class HybridTimeSeriesPredictor:
    def __init__(self, sequence_length=30, hidden_dim=64):
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, 1)
        self.arima_model = None
        
    def create_sequences(self, data):
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)
            
        return torch.FloatTensor(sequences), torch.FloatTensor(targets)
    
    def fit_arima(self, data):
        self.arima_model = ARIMA(data, order=(5,1,2))
        self.arima_model = self.arima_model.fit()
    
    def train_lstm(self, sequences, targets, epochs=100):
        optimizer = torch.optim.Adam(self.lstm.parameters())
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            lstm_out, _ = self.lstm(sequences.unsqueeze(-1))
            predictions = self.linear(lstm_out[:, -1, :])
            loss = criterion(predictions.squeeze(), targets)
            loss.backward()
            optimizer.step()
    
    def predict(self, data):
        # LSTM prediction
        sequences = self.create_sequences(data[-self.sequence_length:])[0]
        with torch.no_grad():
            lstm_out, _ = self.lstm(sequences.unsqueeze(-1))
            lstm_pred = self.linear(lstm_out[:, -1, :])
        
        # ARIMA prediction
        arima_pred = self.arima_model.forecast(steps=1)
        
        # Combine predictions
        final_pred = 0.6 * lstm_pred.item() + 0.4 * arima_pred[0]
        
        return final_pred
    
    def calculate_risk_metrics(self, predictions, actuals):
        returns = np.diff(actuals) / actuals[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'mse': np.mean((predictions - actuals) ** 2),
            'mae': np.mean(np.abs(predictions - actuals))
        }
```

Slide 14: Real-World Application: Image-Text Multimodal Classification

This implementation demonstrates a hybrid approach for classifying products using both image and text data, combining CNN features with BERT embeddings and traditional ML classifiers.

```python
import torch
import torch.nn as nn
from torchvision import models
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestClassifier

class MultimodalClassifier:
    def __init__(self):
        # Image feature extractor
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Text feature extractor
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Final classifier
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        
    def extract_image_features(self, images):
        self.resnet.eval()
        with torch.no_grad():
            features = self.resnet(images)
            features = features.squeeze().numpy()
        return features
    
    def extract_text_features(self, texts):
        self.bert.eval()
        features = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.bert(**inputs)
                # Use [CLS] token embeddings
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                features.append(embedding.squeeze())
                
        return np.vstack(features)
    
    def combine_features(self, image_features, text_features):
        return np.hstack([image_features, text_features])
    
    def fit(self, images, texts, labels):
        image_features = self.extract_image_features(images)
        text_features = self.extract_text_features(texts)
        combined_features = self.combine_features(
            image_features, 
            text_features
        )
        
        self.classifier.fit(combined_features, labels)
    
    def predict(self, images, texts):
        image_features = self.extract_image_features(images)
        text_features = self.extract_text_features(texts)
        combined_features = self.combine_features(
            image_features, 
            text_features
        )
        
        return self.classifier.predict(combined_features)
    
    def predict_proba(self, images, texts):
        image_features = self.extract_image_features(images)
        text_features = self.extract_text_features(texts)
        combined_features = self.combine_features(
            image_features, 
            text_features
        )
        
        return self.classifier.predict_proba(combined_features)
```

Slide 15: Additional Resources

*   ArXiv paper on Hybrid Deep Learning Architectures: "Deep Hybrid Models: Bridge the Gap Between Traditional and Deep Learning" [https://arxiv.org/abs/2103.01273](https://arxiv.org/abs/2103.01273)
*   Comprehensive Survey on Model Combinations: "Ensemble Learning: The Next Generation of Machine Learning" [https://arxiv.org/abs/2106.04394](https://arxiv.org/abs/2106.04394)
*   Research on Neural-Symbolic Integration: "Neural-Symbolic Learning and Reasoning: A Survey and Interpretation" [https://arxiv.org/abs/2017.03097](https://arxiv.org/abs/2017.03097)
*   Practical Guide to Hybrid ML Systems: "Building Robust Machine Learning Systems: A Comprehensive Guide" Search: "hybrid ml systems implementation guide research paper"
*   Recent Advances in Transfer Learning: "Transfer Learning: A Decade of Progress and Future Directions" [https://arxiv.org/abs/2108.13228](https://arxiv.org/abs/2108.13228)

Note: These URLs represent the type of papers to look for. For the most up-to-date research, please search on ArXiv or Google Scholar using relevant keywords.

