## Mastering Hyperparameter Tuning in Machine Learning with Python
Slide 1: Introduction to Hyperparameter Tuning

Hyperparameter tuning is a crucial step in optimizing machine learning models. It involves adjusting the configuration settings that control the learning process. These settings are not learned from the data but are set before training begins. Effective tuning can significantly improve model performance.

```python
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Define the model and parameter grid
model = SVC()
param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}

# Perform grid search
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

Slide 2: Common Hyperparameters

Different machine learning algorithms have various hyperparameters. For example, in neural networks, we have learning rate, batch size, and number of hidden layers. In random forests, we have the number of trees and maximum depth. Understanding these parameters is key to effective tuning.

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=10,      # Maximum depth of trees
    min_samples_split=2,  # Minimum samples required to split an internal node
    min_samples_leaf=1,   # Minimum samples required to be at a leaf node
    random_state=42
)

rf_model.fit(X, y)
print(f"Random Forest Accuracy: {rf_model.score(X, y):.4f}")
```

Slide 3: Grid Search

Grid Search is an exhaustive search through a manually specified subset of the hyperparameter space. It tries all possible combinations of parameter values and is guaranteed to find the best combination in the specified grid.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
```

Slide 4: Random Search

Random Search samples random combinations of hyperparameters. It's often more efficient than Grid Search, especially when only a few hyperparameters actually matter.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(rf, param_distributions, n_iter=100, cv=5, random_state=42)
random_search.fit(X, y)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")
```

Slide 5: Bayesian Optimization

Bayesian Optimization uses probabilistic models to guide the search for the best hyperparameters. It's particularly useful for expensive-to-evaluate functions and can often find better solutions in fewer iterations than random or grid search.

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer

search_spaces = {
    'n_estimators': Integer(100, 500),
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10)
}

rf = RandomForestClassifier(random_state=42)
bayes_search = BayesSearchCV(rf, search_spaces, n_iter=50, cv=5, random_state=42)
bayes_search.fit(X, y)

print(f"Best parameters: {bayes_search.best_params_}")
print(f"Best cross-validation score: {bayes_search.best_score_:.4f}")
```

Slide 6: Cross-Validation in Hyperparameter Tuning

Cross-validation is crucial in hyperparameter tuning to prevent overfitting to the validation set. K-Fold cross-validation is a common technique where the data is split into K subsets, and the model is trained and evaluated K times.

```python
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")
print(f"Standard deviation of CV score: {cv_scores.std():.4f}")
```

Slide 7: Automated Machine Learning (AutoML)

AutoML tools automate the process of selecting models and tuning hyperparameters. They can be particularly useful for beginners or in situations where manual tuning is impractical due to time constraints.

```python
from autosklearn.classification import AutoSklearnClassifier

automl = AutoSklearnClassifier(time_left_for_this_task=300, per_run_time_limit=30)
automl.fit(X, y)

print("Statistics:")
print(automl.sprint_statistics())
print("\nBest model:")
print(automl.show_models())
```

Slide 8: Learning Curves

Learning curves help visualize how model performance changes with increasing amounts of training data. They can indicate whether a model would benefit from more data or different hyperparameters.

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(random_state=42), X, y, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, test_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('Number of training examples')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend()
plt.show()
```

Slide 9: Hyperparameter Importance

Not all hyperparameters are equally important. Identifying the most influential parameters can help focus tuning efforts and improve efficiency.

```python
from sklearn.inspection import PartialDependenceDisplay

# Assuming 'grid_search' is a fitted GridSearchCV object
results = pd.DataFrame(grid_search.cv_results_)
params = ['param_n_estimators', 'param_max_depth', 'param_min_samples_split']

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for i, param in enumerate(params):
    PartialDependenceDisplay.from_estimator(grid_search.best_estimator_, X, [param], ax=axes[i])
    axes[i].set_title(f'Partial Dependence of {param}')

plt.tight_layout()
plt.show()
```

Slide 10: Real-Life Example: Image Classification

In this example, we'll tune a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset.

```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from kerastuner import RandomSearch

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_model(hp):
    model = Sequential()
    model.add(Conv2D(hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
                     kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(hp.Int('dense_units', min_value=32, max_value=128, step=16), activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='cifar10_cnn')

tuner.search(x_train, y_train, epochs=10, validation_split=0.2)

best_model = tuner.get_best_models(num_models=1)[0]
print(best_model.summary())
```

Slide 11: Real-Life Example: Natural Language Processing

In this example, we'll tune a text classification model using the IMDb movie review dataset.

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from kerastuner import Hyperband

max_features = 10000
maxlen = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

def build_model(hp):
    model = Sequential()
    model.add(Embedding(max_features, hp.Int('embed_dim', 32, 256, step=32),
                        input_length=maxlen))
    model.add(LSTM(hp.Int('lstm_units', 32, 256, step=32)))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='my_dir',
    project_name='imdb_lstm')

tuner.search(x_train, y_train, epochs=10, validation_split=0.2)

best_model = tuner.get_best_models(num_models=1)[0]
print(best_model.summary())
```

Slide 12: Overfitting and Regularization

Overfitting occurs when a model performs well on training data but poorly on unseen data. Regularization techniques can help prevent overfitting during hyperparameter tuning.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# No regularization
model_no_reg = LogisticRegression(C=1e6, random_state=42)
model_no_reg.fit(X_train, y_train)

# L2 regularization
model_l2 = LogisticRegression(C=1.0, random_state=42)
model_l2.fit(X_train, y_train)

print(f"No regularization - Train: {model_no_reg.score(X_train, y_train):.4f}, Test: {model_no_reg.score(X_test, y_test):.4f}")
print(f"L2 regularization - Train: {model_l2.score(X_train, y_train):.4f}, Test: {model_l2.score(X_test, y_test):.4f}")
```

Slide 13: Hyperparameter Tuning Best Practices

Effective hyperparameter tuning requires a systematic approach. Some best practices include: starting with a broad search and then refining, using domain knowledge to set reasonable ranges, monitoring for overfitting, and considering the computational cost of tuning.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Broad initial search
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.8, 0.2),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

gb = GradientBoostingClassifier(random_state=42)
random_search = RandomizedSearchCV(gb, param_distributions=param_dist, n_iter=100, cv=5, random_state=42)
random_search.fit(X, y)

print("Best parameters from broad search:", random_search.best_params_)

# Refined search
refined_param_dist = {
    'n_estimators': randint(random_search.best_params_['n_estimators'] - 50, random_search.best_params_['n_estimators'] + 50),
    'max_depth': randint(random_search.best_params_['max_depth'] - 1, random_search.best_params_['max_depth'] + 1),
    'learning_rate': uniform(random_search.best_params_['learning_rate'] - 0.05, random_search.best_params_['learning_rate'] + 0.05),
    'subsample': uniform(random_search.best_params_['subsample'] - 0.1, random_search.best_params_['subsample'] + 0.1),
    'min_samples_split': randint(random_search.best_params_['min_samples_split'] - 2, random_search.best_params_['min_samples_split'] + 2),
    'min_samples_leaf': randint(random_search.best_params_['min_samples_leaf'] - 1, random_search.best_params_['min_samples_leaf'] + 1)
}

refined_random_search = RandomizedSearchCV(gb, param_distributions=refined_param_dist, n_iter=50, cv=5, random_state=42)
refined_random_search.fit(X, y)

print("Best parameters from refined search:", refined_random_search.best_params_)
```

Slide 14: Additional Resources

For those interested in diving deeper into hyperparameter tuning, here are some valuable resources:

1. "Automatic Machine Learning: Methods, Systems, Challenges" (Hutter et al., 2019) - Available on arXiv: [https://arxiv.org/abs/1908.00709](https://arxiv.org/abs/1908.00709)
2. "Neural Architecture Search: A Survey" (Elsken et al., 2019) - Available on arXiv: [https://arxiv.org/abs/1808.05377](https://arxiv.org/abs/1808.05377)
3. "Hyperparameter Optimization: A Spectral Approach" (Hazan et al., 2017) - Available on arXiv: [https://arxiv.org/abs/1706.00764](https://arxiv.org/abs/1706.00764)
4. "Taking the Human Out of the Loop: A Review of Bayesian Optimization" (Shahriari et al., 2016) - Available on arXiv: [https://arxiv.org/abs/1507.05853](https://arxiv.org/abs/1507.05853)

These papers provide in-depth discussions on various aspects of hyperparameter tuning and automated machine learning.

Slide 15: Conclusion

Mastering hyperparameter tuning is crucial for optimizing machine learning models. We've covered various techniques from simple grid search to more advanced methods like Bayesian optimization. Remember that effective tuning requires a balance between exploration of the parameter space and exploitation of promising regions. Always consider the computational cost and use domain knowledge when possible. With practice, you'll develop intuition for tuning different types of models and datasets.

```python
# Pseudocode for a general hyperparameter tuning workflow
def tune_hyperparameters(model, param_space, X, y):
    # Choose a search method (e.g., RandomizedSearchCV, BayesianOptimization)
    search_method = select_search_method()
    
    # Set up cross-validation
    cv = set_up_cross_validation()
    
    # Perform the search
    best_params, best_score = search_method.search(model, param_space, X, y, cv)
    
    # Fine-tune around the best parameters if needed
    refined_params = refine_search(best_params)
    
    # Train final model with best parameters
    final_model = train_model(model, refined_params, X, y)
    
    return final_model, best_score

# Usage
best_model, score = tune_hyperparameters(RandomForestClassifier(), param_space, X, y)
print(f"Best model score: {score:.4f}")
```

This concludes our slideshow on Mastering Hyperparameter Tuning in Machine Learning using Python. Remember that hyperparameter tuning is as much an art as it is a science, and experience will help you develop intuition for different problems and datasets.

