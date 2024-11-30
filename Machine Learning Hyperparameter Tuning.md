## Machine Learning Hyperparameter Tuning
Slide 1: Linear Regression Hyperparameters - Regularization Impact

Linear regression hyperparameters significantly influence model performance through regularization strength (alpha) and penalty type (l1/l2). Understanding these parameters enables proper control of model complexity and prevents overfitting while maintaining predictive accuracy on unseen data.

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression
import numpy as np

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1)

# Compare L1 (Lasso) and L2 (Ridge) regularization
alphas = [0.1, 1.0, 10.0]
for alpha in alphas:
    # L2 regularization
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, y)
    ridge_coef = np.sum(np.abs(ridge.coef_))
    
    # L1 regularization
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    lasso_coef = np.sum(np.abs(lasso.coef_))
    
    print(f"Alpha: {alpha}")
    print(f"Ridge coefficients sum: {ridge_coef:.4f}")
    print(f"Lasso coefficients sum: {lasso_coef:.4f}\n")
```

Slide 2: Decision Tree Depth and Splitting Criteria

Decision trees require careful tuning of maximum depth and splitting criteria to balance model complexity. These hyperparameters directly impact tree structure, prediction accuracy, and generalization capabilities through controlled growth patterns.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# Generate classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)

# Test different max_depths and criteria
depths = [3, 5, 10]
criteria = ['gini', 'entropy']

for criterion in criteria:
    for depth in depths:
        dt = DecisionTreeClassifier(
            max_depth=depth,
            criterion=criterion,
            random_state=42
        )
        scores = cross_val_score(dt, X, y, cv=5)
        print(f"Criterion: {criterion}, Max Depth: {depth}")
        print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std()*2:.4f})\n")
```

Slide 3: Random Forest Ensemble Parameters

Random Forest hyperparameters extend beyond individual tree settings to include ensemble-specific parameters. The number of estimators and maximum features per tree significantly impact model robustness and computational requirements while maintaining prediction accuracy.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Test different n_estimators and max_features
n_estimators_list = [50, 100, 200]
max_features_list = ['sqrt', 'log2', None]

for n_est in n_estimators_list:
    for max_feat in max_features_list:
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_features=max_feat,
            random_state=42
        )
        rf.fit(X_train, y_train)
        score = rf.score(X_test, y_test)
        print(f"n_estimators: {n_est}, max_features: {max_feat}")
        print(f"Test accuracy: {score:.4f}\n")
```

Slide 4: Gradient Boosting Learning Rate Impact

The learning rate in gradient boosting algorithms controls the contribution of each tree to the final prediction. This crucial hyperparameter balances training speed with model accuracy and requires careful optimization to prevent overfitting or underfitting.

```python
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import numpy as np

# Generate dataset
X, y = make_classification(n_samples=500, n_features=10)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Test different learning rates
learning_rates = [0.01, 0.1, 1.0]
training_scores = []
testing_scores = []

for lr in learning_rates:
    gb = GradientBoostingClassifier(
        learning_rate=lr,
        n_estimators=100,
        random_state=42
    )
    gb.fit(X_train, y_train)
    train_score = gb.score(X_train, y_train)
    test_score = gb.score(X_test, y_test)
    
    print(f"Learning rate: {lr}")
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}\n")
```

Slide 5: PCA Component Selection and Variance Ratio

Principal Component Analysis requires careful selection of the number of components to retain meaningful variance while reducing dimensionality. The n\_components parameter directly influences the trade-off between data compression and information preservation.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import numpy as np

# Load digits dataset
digits = load_digits()
X = digits.data

# Test different numbers of components
n_components_list = [10, 20, 30, 40]

for n_comp in n_components_list:
    pca = PCA(n_components=n_comp)
    X_transformed = pca.fit_transform(X)
    
    # Calculate explained variance ratio
    explained_variance = np.sum(pca.explained_variance_ratio_)
    
    print(f"Components: {n_comp}")
    print(f"Explained variance ratio: {explained_variance:.4f}")
    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_transformed.shape}\n")
```

Slide 6: KNN Distance Metrics and Neighbor Weights

K-Nearest Neighbors algorithm performance heavily depends on the distance metric and neighbor weighting scheme. These hyperparameters determine how similarity is measured and how neighbor votes are weighted in classification tasks.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Generate and scale dataset
X, y = make_classification(n_samples=1000, n_features=20)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test different metrics and weights
metrics = ['euclidean', 'manhattan', 'minkowski']
weights = ['uniform', 'distance']

for metric in metrics:
    for weight in weights:
        knn = KNeighborsClassifier(
            n_neighbors=5,
            metric=metric,
            weights=weight
        )
        scores = cross_val_score(knn, X_scaled, y, cv=5)
        print(f"Metric: {metric}, Weights: {weight}")
        print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std()*2:.4f})\n")
```

Slide 7: K-Means Clustering Initialization Methods

K-means clustering results are significantly influenced by initialization method and number of clusters. The initialization strategy affects convergence speed and the quality of final cluster assignments.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Generate clusterable data
X = np.concatenate([
    np.random.normal(0, 1, (100, 2)),
    np.random.normal(4, 1, (100, 2)),
    np.random.normal(8, 1, (100, 2))
])

# Test different initialization methods
init_methods = ['k-means++', 'random']
n_clusters_list = [2, 3, 4]

for init in init_methods:
    for n_clusters in n_clusters_list:
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=10,
            random_state=42
        )
        labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        
        print(f"Init method: {init}, n_clusters: {n_clusters}")
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        print(f"Inertia: {kmeans.inertia_:.4f}\n")
```

Slide 8: Neural Network Architecture - Hidden Layer Configuration

Neural network architecture design requires careful consideration of hidden layer sizes and activation functions. These fundamental hyperparameters determine the network's capacity to learn complex patterns and representations.

```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Prepare dataset
X, y = make_classification(n_samples=1000, n_features=20)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test different hidden layer configurations
hidden_layer_sizes = [(50,), (100,), (50, 25), (100, 50)]
activations = ['relu', 'tanh']

for hidden_size in hidden_layer_sizes:
    for activation in activations:
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_size,
            activation=activation,
            max_iter=1000,
            random_state=42
        )
        scores = cross_val_score(mlp, X_scaled, y, cv=5)
        print(f"Hidden layers: {hidden_size}, Activation: {activation}")
        print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std()*2:.4f})\n")
```

Slide 9: Neural Network Regularization - Dropout and L2

Neural network regularization techniques like dropout and L2 penalty prevent overfitting by adding controlled noise during training and constraining weight magnitudes. These hyperparameters balance model complexity with generalization capability.

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np

# Generate dataset
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

# Test different dropout rates and L2 penalties
dropout_rates = [0.2, 0.5]
l2_penalties = [0.01, 0.001]

for dropout_rate in dropout_rates:
    for l2_penalty in l2_penalties:
        model = tf.keras.Sequential([
            Dense(64, activation='relu', kernel_regularizer=l2(l2_penalty)),
            Dropout(dropout_rate),
            Dense(32, activation='relu', kernel_regularizer=l2(l2_penalty)),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X, y, epochs=10, validation_split=0.2, verbose=0)
        
        print(f"Dropout: {dropout_rate}, L2 penalty: {l2_penalty}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
```

Slide 10: Real-world Example - Credit Card Fraud Detection

Implementation of a comprehensive fraud detection system using multiple models with optimized hyperparameters. This example demonstrates hyperparameter impact on imbalanced classification problems common in financial applications.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Simulate credit card transaction data
np.random.seed(42)
n_samples = 10000
n_features = 30

# Create imbalanced dataset (1% fraud)
X = np.random.randn(n_samples, n_features)
y = np.zeros(n_samples)
fraud_indices = np.random.choice(n_samples, int(0.01 * n_samples), replace=False)
y[fraud_indices] = 1

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define hyperparameter search space
param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Random search with cross-validation
rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    rf, param_distributions, n_iter=10, cv=3, scoring='f1',
    random_state=42, n_jobs=-1
)

# Fit and print results
random_search.fit(X_scaled, y)
print("Best parameters:", random_search.best_params_)
print("Best F1 score:", random_search.best_score_)
```

Slide 11: Results Analysis for Credit Card Fraud Detection

```python
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Get best model predictions
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_scaled)

# Print detailed metrics
print("Classification Report:")
print(classification_report(y, y_pred))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': [f'feature_{i}' for i in range(n_features)],
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 5 Most Important Features:")
print(feature_importance.head())
```

Slide 12: Real-world Example - Customer Churn Prediction

Customer churn prediction requires careful hyperparameter optimization across multiple algorithms to achieve optimal predictive performance. This example demonstrates a practical approach to hyperparameter tuning in a business context.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd

# Simulate customer data
np.random.seed(42)
n_customers = 5000

# Generate synthetic customer features
data = {
    'tenure': np.random.randint(1, 72, n_customers),
    'monthly_charges': np.random.uniform(20, 200, n_customers),
    'contract_type': np.random.choice(['Month-to-month', '1 year', '2 year'], n_customers),
    'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check'], n_customers),
    'total_charges': np.random.uniform(100, 8000, n_customers)
}

df = pd.DataFrame(data)
y = np.random.binomial(1, 0.2, n_customers)  # 20% churn rate

# Create preprocessing pipeline
numeric_features = ['tenure', 'monthly_charges', 'total_charges']
categorical_features = ['contract_type', 'payment_method']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Create model pipeline with hyperparameter search
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Define hyperparameter grid
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__max_depth': [3, 5],
    'classifier__min_samples_split': [2, 5]
}

# Perform grid search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(df, y)
```

Slide 13: Results for Customer Churn Prediction

```python
# Print best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best ROC-AUC score:", grid_search.best_score_)

# Feature importance analysis
best_model = grid_search.best_estimator_
feature_names = (numeric_features + 
                [f"{feat}_{val}" for feat, vals in 
                 zip(categorical_features, 
                     best_model.named_steps['preprocessor']
                     .named_transformers_['cat'].categories_) 
                 for val in vals[1:]])

importances = best_model.named_steps['classifier'].feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# Cross-validation performance metrics
from sklearn.metrics import make_scorer, precision_score, recall_score
scoring = {
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'roc_auc': 'roc_auc'
}

from sklearn.model_selection import cross_validate
cv_results = cross_validate(grid_search.best_estimator_, df, y, 
                          scoring=scoring, cv=5)

print("\nCross-validation metrics:")
for metric, scores in cv_results.items():
    if metric.startswith('test_'):
        print(f"{metric}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

Slide 14: Additional Resources

*   Hyperparameter Optimization for Neural Networks
    *   [https://arxiv.org/abs/2006.15745](https://arxiv.org/abs/2006.15745)
*   Automated Machine Learning: Methods, Systems, Challenges
    *   [https://arxiv.org/abs/1904.12054](https://arxiv.org/abs/1904.12054)
*   A Systematic Review of Hyperparameter Optimization in Machine Learning
    *   [https://arxiv.org/abs/2003.05689](https://arxiv.org/abs/2003.05689)
*   Practical Guidelines for Hyperparameter Optimization
    *   Search "Hyperparameter Optimization Best Practices" on Google Scholar
*   Survey of Hyperparameter Optimization Methods
    *   Search "Survey Hyperparameter Optimization Machine Learning" on Google Scholar

