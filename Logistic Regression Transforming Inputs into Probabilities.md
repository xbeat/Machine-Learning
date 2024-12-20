## Logistic Regression Transforming Inputs into Probabilities
Slide 1: Understanding Logistic Regression Fundamentals

Logistic regression models the probability of binary outcomes through a sigmoid function that maps any real-valued input to a probability between 0 and 1. This fundamental algorithm serves as the foundation for binary classification tasks in machine learning.

```python
import numpy as np

def sigmoid(z):
    # Sigmoid activation function
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    
    for _ in range(epochs):
        # Forward propagation
        z = np.dot(X, weights) + bias
        predictions = sigmoid(z)
        
        # Compute gradients
        dw = (1/m) * np.dot(X.T, (predictions - y))
        db = (1/m) * np.sum(predictions - y)
        
        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
    return weights, bias
```

Slide 2: Mathematics Behind Logistic Regression

The logistic regression model combines linear regression with the sigmoid function to transform continuous inputs into probability outputs. The mathematical foundation involves the logistic function and maximum likelihood estimation.

```python
# Mathematical formulation of Logistic Regression
'''
Logistic Function (Sigmoid):
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Cost Function:
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$$

Gradient Descent Update:
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$
'''

def cost_function(X, y, weights, bias):
    m = len(y)
    z = np.dot(X, weights) + bias
    predictions = sigmoid(z)
    cost = (-1/m) * np.sum(y * np.log(predictions) + (1-y) * np.log(1-predictions))
    return cost
```

Slide 3: Feature Engineering and Data Preprocessing

Effective feature engineering and preprocessing are crucial for logistic regression performance. This includes handling missing values, scaling features, and encoding categorical variables to prepare data for model training.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_features(data):
    # Handle missing values
    data = data.fillna(data.mean())
    
    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data, scaler
```

Slide 4: Implementation of Credit Risk Assessment Model

Real-world application demonstrating logistic regression for credit risk assessment. This implementation processes loan application data to predict the probability of default, incorporating multiple features and risk factors.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Sample credit risk assessment implementation
def credit_risk_model():
    # Generate synthetic credit data
    np.random.seed(42)
    n_samples = 1000
    
    # Create features
    income = np.random.normal(50000, 20000, n_samples)
    debt_ratio = np.random.uniform(0.1, 0.6, n_samples)
    credit_history = np.random.uniform(300, 850, n_samples)
    
    # Create target variable (default probability)
    X = np.column_stack((income, debt_ratio, credit_history))
    z = -2 + 0.00003*income - 3*debt_ratio + 0.01*credit_history
    prob_default = sigmoid(z)
    y = (np.random.random(n_samples) < prob_default).astype(int)
    
    return X, y
```

Slide 5: Training and Validation Pipeline

Implementing a robust training and validation pipeline is essential for model reliability. This includes cross-validation, hyperparameter tuning, and performance monitoring to ensure the model generalizes well to unseen data.

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve

def train_validate_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    weights, bias = logistic_regression(X_train, y_train, 
                                      learning_rate=0.01, 
                                      epochs=1000)
    
    # Make predictions
    z = np.dot(X_test, weights) + bias
    y_pred_proba = sigmoid(z)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    return weights, bias, auc_score, precision, recall
```

Slide 6: Regularization Techniques

Regularization prevents overfitting by adding penalty terms to the cost function. L1 (Lasso) and L2 (Ridge) regularization constrain model complexity and improve generalization performance on unseen data.

```python
def regularized_logistic_regression(X, y, lambda_param=0.1, reg_type='l2'):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    learning_rate = 0.01
    
    for _ in range(1000):
        z = np.dot(X, weights) + bias
        predictions = sigmoid(z)
        
        # Compute gradients with regularization
        if reg_type == 'l2':
            reg_term = lambda_param * weights
        else:  # l1
            reg_term = lambda_param * np.sign(weights)
            
        dw = (1/m) * np.dot(X.T, (predictions - y)) + reg_term
        db = (1/m) * np.sum(predictions - y)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
    return weights, bias
```

Slide 7: Customer Churn Prediction Implementation

Real-world application of logistic regression for predicting customer churn in a telecommunications company. This implementation processes customer behavior data to identify likely churners.

```python
def churn_prediction_model():
    # Generate synthetic customer data
    n_customers = 1000
    
    # Customer features
    usage_minutes = np.random.normal(600, 200, n_customers)
    contract_length = np.random.choice([1, 12, 24], n_customers)
    support_calls = np.random.poisson(2, n_customers)
    bill_amount = np.random.normal(70, 25, n_customers)
    
    # Create feature matrix
    X = np.column_stack((
        usage_minutes,
        contract_length,
        support_calls,
        bill_amount
    ))
    
    # Generate churn labels
    z = -2 + 0.001*usage_minutes - 0.1*contract_length + \
        0.5*support_calls + 0.02*bill_amount
    prob_churn = sigmoid(z)
    y = (np.random.random(n_customers) < prob_churn).astype(int)
    
    return X, y, ['usage', 'contract', 'support', 'bill']
```

Slide 8: Model Evaluation and Metrics

Comprehensive model evaluation requires multiple metrics to assess different aspects of performance. This implementation calculates and visualizes key classification metrics for model assessment.

```python
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred, y_pred_proba):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate various metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Visualization
    plt.figure(figsize=(12, 4))
    
    # Plot confusion matrix
    plt.subplot(121)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    
    # Plot ROC curve
    plt.subplot(122)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    return report
```

Slide 9: Handling Imbalanced Datasets

Imbalanced datasets require special handling techniques to prevent model bias towards the majority class. This implementation demonstrates various resampling methods and weighted loss functions to address class imbalance.

```python
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

def handle_imbalanced_data(X, y):
    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    
    # Implement weighted logistic regression
    def weighted_logistic_regression(X, y, weights, class_weights):
        m, n = X.shape
        model_weights = np.zeros(n)
        bias = 0
        learning_rate = 0.01
        
        for _ in range(1000):
            z = np.dot(X, model_weights) + bias
            predictions = sigmoid(z)
            
            # Apply class weights to gradient calculation
            sample_weights = np.array([class_weights[int(label)] for label in y])
            weighted_error = (predictions - y) * sample_weights
            
            dw = (1/m) * np.dot(X.T, weighted_error)
            db = (1/m) * np.sum(weighted_error)
            
            model_weights -= learning_rate * dw
            bias -= learning_rate * db
            
        return model_weights, bias
    
    # Perform SMOTE-like oversampling
    minority_class = X[y == 1]
    majority_class = X[y == 0]
    
    # Oversample minority class
    minority_upsampled = resample(
        minority_class,
        replace=True,
        n_samples=len(majority_class),
        random_state=42
    )
    
    # Combine balanced dataset
    X_balanced = np.vstack([majority_class, minority_upsampled])
    y_balanced = np.hstack([
        np.zeros(len(majority_class)),
        np.ones(len(minority_upsampled))
    ])
    
    return X_balanced, y_balanced, class_weights
```

Slide 10: Gradient Descent Optimization Variants

Advanced optimization techniques improve convergence speed and model performance. This implementation showcases different gradient descent variants including mini-batch and adaptive learning rates.

```python
def advanced_optimization(X, y, batch_size=32, learning_rate=0.01):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    
    # Adam optimizer parameters
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    v_dw = np.zeros(n)
    v_db = 0
    s_dw = np.zeros(n)
    s_db = 0
    t = 0
    
    for epoch in range(100):
        # Mini-batch gradient descent
        indices = np.random.permutation(m)
        
        for i in range(0, m, batch_size):
            t += 1
            batch_indices = indices[i:min(i + batch_size, m)]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Forward propagation
            z = np.dot(X_batch, weights) + bias
            predictions = sigmoid(z)
            
            # Compute gradients
            dw = (1/len(batch_indices)) * np.dot(X_batch.T, (predictions - y_batch))
            db = (1/len(batch_indices)) * np.sum(predictions - y_batch)
            
            # Adam optimization
            v_dw = beta1 * v_dw + (1 - beta1) * dw
            v_db = beta1 * v_db + (1 - beta1) * db
            s_dw = beta2 * s_dw + (1 - beta2) * np.square(dw)
            s_db = beta2 * s_db + (1 - beta2) * np.square(db)
            
            v_dw_corrected = v_dw / (1 - beta1**t)
            v_db_corrected = v_db / (1 - beta1**t)
            s_dw_corrected = s_dw / (1 - beta2**t)
            s_db_corrected = s_db / (1 - beta2**t)
            
            # Update parameters
            weights -= learning_rate * v_dw_corrected / (np.sqrt(s_dw_corrected) + epsilon)
            bias -= learning_rate * v_db_corrected / (np.sqrt(s_db_corrected) + epsilon)
            
    return weights, bias
```

Slide 11: Feature Selection and Dimensionality Reduction

Feature selection optimizes model performance by identifying the most relevant predictors. This implementation combines statistical tests and regularization techniques to select optimal features for logistic regression.

```python
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA

def feature_selection_pipeline(X, y):
    # Statistical feature selection
    selector = SelectKBest(score_func=chi2, k='all')
    X_normalized = X - X.min() + 0.1  # Ensure non-negative values for chi2
    selector.fit(X_normalized, y)
    
    # Get feature importance scores
    feature_scores = pd.DataFrame({
        'Feature': range(X.shape[1]),
        'Score': selector.scores_
    })
    
    # L1-based feature selection
    def l1_feature_selection(X, y, alpha=0.01):
        # Train logistic regression with L1 penalty
        weights, _ = regularized_logistic_regression(
            X, y, lambda_param=alpha, reg_type='l1'
        )
        
        # Get feature importance based on coefficient magnitude
        feature_importance = np.abs(weights)
        return feature_importance
    
    l1_importance = l1_feature_selection(X, y)
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=0.95)  # Preserve 95% variance
    X_pca = pca.fit_transform(X)
    
    return {
        'statistical_scores': feature_scores,
        'l1_importance': l1_importance,
        'pca_components': X_pca,
        'explained_variance': pca.explained_variance_ratio_
    }
```

Slide 12: Cross-Validation and Model Selection

Robust model validation ensures reliable performance estimates. This implementation demonstrates k-fold cross-validation with stratification and hyperparameter tuning.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, f1_score

def cross_validate_model(X, y, n_splits=5):
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Hyperparameter grid
    learning_rates = [0.001, 0.01, 0.1]
    regularization_strengths = [0.0, 0.1, 1.0]
    
    best_score = -np.inf
    best_params = {}
    cv_results = []
    
    for lr in learning_rates:
        for reg_strength in regularization_strengths:
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model with current parameters
                weights, bias = regularized_logistic_regression(
                    X_train, y_train,
                    learning_rate=lr,
                    lambda_param=reg_strength
                )
                
                # Evaluate on validation set
                val_pred = predict(X_val, weights, bias)
                fold_score = f1_score(y_val, val_pred)
                fold_scores.append(fold_score)
            
            # Average score across folds
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            cv_results.append({
                'learning_rate': lr,
                'reg_strength': reg_strength,
                'mean_score': mean_score,
                'std_score': std_score
            })
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = {
                    'learning_rate': lr,
                    'reg_strength': reg_strength
                }
    
    return best_params, cv_results
```

Slide 13: Model Interpretability and Explainability

Understanding model decisions is crucial for real-world applications. This implementation provides tools for interpreting feature importance, decision boundaries, and individual predictions.

```python
import shap
from lime import lime_tabular

def interpret_model(model_weights, X, feature_names):
    # Calculate feature importance
    feature_importance = np.abs(model_weights)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    def predict_proba(X):
        z = np.dot(X, model_weights)
        return sigmoid(z)
    
    # SHAP values calculation
    explainer = shap.KernelExplainer(predict_proba, X)
    shap_values = explainer.shap_values(X[:100])  # Sample for efficiency
    
    # LIME explanation for individual prediction
    explainer_lime = lime_tabular.LimeTabularExplainer(
        X,
        feature_names=feature_names,
        mode='classification'
    )
    
    def get_local_explanation(instance):
        exp = explainer_lime.explain_instance(
            instance, 
            predict_proba,
            num_features=len(feature_names)
        )
        return exp.as_list()
    
    return {
        'global_importance': importance_df,
        'shap_values': shap_values,
        'local_explanation': get_local_explanation
    }
```

Slide 14: Production Deployment Pipeline

Implementing a production-ready logistic regression pipeline requires robust preprocessing, model versioning, and monitoring capabilities.

```python
import joblib
import json
from datetime import datetime

class ProductionLogisticRegression:
    def __init__(self, feature_names, scaler=None):
        self.feature_names = feature_names
        self.scaler = scaler
        self.weights = None
        self.bias = None
        self.metadata = {}
        
    def preprocess_input(self, X):
        # Validate input features
        if isinstance(X, pd.DataFrame):
            if not all(col in X.columns for col in self.feature_names):
                raise ValueError("Missing required features")
            X = X[self.feature_names].values
        
        # Apply scaling if available
        if self.scaler:
            X = self.scaler.transform(X)
        return X
    
    def predict_proba(self, X):
        X = self.preprocess_input(X)
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)
    
    def save_model(self, path):
        model_data = {
            'weights': self.weights.tolist(),
            'bias': float(self.bias),
            'feature_names': self.feature_names,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'metrics': self.metadata.get('metrics', {})
            }
        }
        
        if self.scaler:
            joblib.dump(self.scaler, f"{path}_scaler.pkl")
        
        with open(f"{path}_model.json", 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load_model(cls, path):
        with open(f"{path}_model.json", 'r') as f:
            model_data = json.load(f)
        
        model = cls(model_data['feature_names'])
        model.weights = np.array(model_data['weights'])
        model.bias = model_data['bias']
        model.metadata = model_data['metadata']
        
        try:
            model.scaler = joblib.load(f"{path}_scaler.pkl")
        except:
            model.scaler = None
            
        return model
```

Slide 15: Additional Resources

*   "Deep Understanding of Logistic Regression for Machine Learning"
    *   [https://arxiv.org/abs/1407.1419](https://arxiv.org/abs/1407.1419)
*   "On the Convergence of Logistic Regression with Regularization"
    *   [https://arxiv.org/abs/1802.06384](https://arxiv.org/abs/1802.06384)
*   "Feature Selection Methods in Logistic Regression: A Comparative Study"
    *   [https://www.sciencedirect.com/science/article/pii/S0169743X18303190](https://www.sciencedirect.com/science/article/pii/S0169743X18303190)
*   "Interpretable Machine Learning with Logistic Regression"
    *   [https://www.nature.com/articles/s42256-019-0138-9](https://www.nature.com/articles/s42256-019-0138-9)
*   Suggested searches:
    *   "Logistic Regression implementation from scratch Python"
    *   "Advanced optimization techniques for Logistic Regression"
    *   "Production deployment best practices for ML models"

