## Machine Learning with Hyperparameter Tuning in Python
Slide 1: Introduction to Machine Learning and Hyperparameter Tuning

Machine Learning (ML) is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Hyperparameter tuning is the process of optimizing the configuration parameters of ML algorithms to improve their performance. This slideshow will explore the concepts of ML and hyperparameter tuning using Python, providing practical examples and code snippets.

```python
# Simple example of a machine learning model
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a random dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train a Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Evaluate the model
accuracy = rf_model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")
```

Slide 2: Types of Machine Learning Algorithms

Machine Learning algorithms can be broadly categorized into three types: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning deals with labeled data, unsupervised learning works with unlabeled data, and reinforcement learning learns through interaction with an environment. This slide focuses on supervised learning, which is commonly used for classification and regression tasks.

```python
# Examples of supervised learning algorithms
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Linear Regression (for regression tasks)
lr_model = LinearRegression()

# Support Vector Machine (for classification tasks)
svm_model = SVC()

# Decision Tree (for both regression and classification tasks)
dt_model = DecisionTreeClassifier()

# Note: These models need to be trained with appropriate data
# lr_model.fit(X_train, y_train)
# svm_model.fit(X_train, y_train)
# dt_model.fit(X_train, y_train)
```

Slide 3: Hyperparameters in Machine Learning

Hyperparameters are configuration settings for ML algorithms that are set before the learning process begins. Unlike model parameters, which are learned during training, hyperparameters are manually set and can significantly impact model performance. Examples include learning rate, number of hidden layers in neural networks, and maximum depth in decision trees.

```python
from sklearn.ensemble import RandomForestClassifier

# Creating a Random Forest Classifier with specific hyperparameters
rf_model = RandomForestClassifier(
    n_estimators=100,  # Number of trees in the forest
    max_depth=10,      # Maximum depth of each tree
    min_samples_split=5,  # Minimum number of samples required to split an internal node
    random_state=42    # Seed for random number generator
)

# The hyperparameters above can be tuned to optimize model performance
```

Slide 4: Importance of Hyperparameter Tuning

Hyperparameter tuning is crucial for optimizing ML model performance. It helps in finding the best configuration for a given problem, reducing overfitting, and improving generalization. Proper tuning can lead to significant improvements in model accuracy, efficiency, and robustness.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Function to evaluate model with different hyperparameters
def evaluate_model(n_estimators, max_depth):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
    return scores.mean()

# Example: Comparing two different hyperparameter settings
score1 = evaluate_model(n_estimators=50, max_depth=5)
score2 = evaluate_model(n_estimators=100, max_depth=10)

print(f"Model 1 score: {score1:.3f}")
print(f"Model 2 score: {score2:.3f}")
```

Slide 5: Grid Search for Hyperparameter Tuning

Grid Search is a systematic approach to hyperparameter tuning. It exhaustively searches through a predefined set of hyperparameter values to find the best combination. While comprehensive, it can be computationally expensive for large hyperparameter spaces.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define the hyperparameter space
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# Create a base model
rf_model = RandomForestClassifier(random_state=42)

# Perform Grid Search
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)
```

Slide 6: Random Search for Hyperparameter Tuning

Random Search is an alternative to Grid Search that samples random combinations of hyperparameters. It's often more efficient than Grid Search, especially when not all hyperparameters are equally important.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

# Define the hyperparameter distribution
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 11),
    'max_features': uniform(0, 1)  # Fraction of features to consider at each split
}

# Create a base model
rf_model = RandomForestClassifier(random_state=42)

# Perform Random Search
random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, 
                                   n_iter=100, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score_)
```

Slide 7: Bayesian Optimization for Hyperparameter Tuning

Bayesian Optimization is a probabilistic model-based approach to hyperparameter tuning. It uses past evaluation results to form a probabilistic model mapping hyperparameters to a probability of a score on the objective function.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import numpy as np

def objective(params):
    # This would be your actual model evaluation
    return -(params[0]**2 + params[1]**2)  # Negative because we're maximizing

def bayesian_optimization(n_iters, sample_loss, bounds):
    X = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])
    y = np.array([sample_loss(x) for x in X])
    
    for i in range(n_iters):
        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=25)
        gp.fit(X, y)
        
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(10000, 2))
        mean, std = gp.predict(x_tries, return_std=True)
        acquisition_func = mean + norm.ppf(0.99) * std
        x_max = x_tries[acquisition_func.argmax()]
        y_max = sample_loss(x_max)
        
        X = np.vstack((X, x_max))
        y = np.append(y, y_max)
    
    return X[y.argmax()]

bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
result = bayesian_optimization(10, objective, bounds)
print("Best parameters found:", result)
```

Slide 8: Cross-Validation in Hyperparameter Tuning

Cross-validation is a resampling technique used to assess ML model performance and prevent overfitting during hyperparameter tuning. It involves partitioning the data into subsets, training on a subset, and validating on the remaining data.

```python
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def cross_validate(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        scores.append(score)

    return np.mean(scores), np.std(scores)

# Assuming X and y are your feature matrix and target vector
mean_score, std_score = cross_validate(X, y)
print(f"Mean CV Score: {mean_score:.3f} (+/- {std_score:.3f})")
```

Slide 9: Automated Machine Learning (AutoML)

AutoML is an advanced approach to hyperparameter tuning that automates the end-to-end process of applying ML to real-world problems. It includes automated feature engineering, model selection, and hyperparameter tuning.

```python
from autosklearn.classification import AutoSklearnClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Load a sample dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train an AutoML model
automl = AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30)
automl.fit(X_train, y_train)

# Evaluate the model
y_pred = automl.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy on test set: {accuracy:.3f}")

# Print the best model and its hyperparameters
print(automl.sprint_statistics())
print(automl.show_models())
```

Slide 10: Real-Life Example: Image Classification

Image classification is a common application of ML with hyperparameter tuning. In this example, we'll use a Convolutional Neural Network (CNN) for classifying images of fruits.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')  # Assuming 3 classes of fruits
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation and loading
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    'path/to/train/data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

# Train the model
history = model.fit(train_generator, epochs=50, steps_per_epoch=100)

# Note: In practice, you would also have validation data and callbacks for early stopping
```

Slide 11: Real-Life Example: Natural Language Processing

Natural Language Processing (NLP) is another area where hyperparameter tuning is crucial. Here's an example of sentiment analysis using a recurrent neural network (RNN) with hyperparameter tuning.

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Sample data (replace with your own dataset)
texts = ["I love this movie", "This was awful", "Great performance"]
labels = [1, 0, 1]  # 1 for positive, 0 for negative

# Tokenize the text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
max_length = 100
padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2)

# Define the model
model = Sequential([
    Embedding(10000, 16, input_length=max_length),
    LSTM(32),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.3f}')

# Note: In a real scenario, you would use a larger dataset and implement hyperparameter tuning
```

Slide 12: Hyperparameter Tuning Best Practices

When performing hyperparameter tuning, consider these best practices:

1. Start with a broad search space and gradually narrow it down.
2. Use domain knowledge to set reasonable ranges for hyperparameters.
3. Consider the computational cost and use appropriate search strategies.
4. Monitor for overfitting using validation sets or cross-validation.
5. Keep track of all experiments and results for reproducibility.
6. Use visualization tools to understand the impact of different hyperparameters.

Slide 13: Hyperparameter Tuning Best Practices

When performing hyperparameter tuning, consider these best practices:

1. Start with a broad search space and gradually narrow it down.
2. Use domain knowledge to set reasonable ranges for hyperparameters.
3. Consider the computational cost and use appropriate search strategies.
4. Monitor for overfitting using validation sets or cross-validation.
5. Keep track of all experiments and results for reproducibility.
6. Use visualization tools to understand the impact of different hyperparameters.

Slide 14: Hyperparameter Tuning Best Practices

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(10, 200),
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 11)
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, 
                                   n_iter=100, cv=5, random_state=42)
random_search.fit(X, y)

results = random_search.cv_results_

plt.figure(figsize=(10, 6))
plt.scatter(results['param_n_estimators'], results['mean_test_score'])
plt.xlabel('Number of Estimators')
plt.ylabel('Mean Test Score')
plt.title('Impact of n_estimators on Model Performance')
plt.show()
```

Slide 15: Avoiding Overfitting in Hyperparameter Tuning

Overfitting during hyperparameter tuning can lead to poor generalization. To avoid this, use techniques like:

1. K-fold cross-validation
2. Separate validation sets
3. Early stopping
4. Regularization
5. Ensemble methods

Slide 16: Avoiding Overfitting in Hyperparameter Tuning

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a model with specific hyperparameters
rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Train the model on the entire training set
rf.fit(X_train, y_train)

# Evaluate on the test set
test_score = accuracy_score(y_test, rf.predict(X_test))
print(f"Test set score: {test_score:.3f}")
```

Slide 17: Hyperparameter Tuning for Deep Learning

Deep learning models often have many hyperparameters to tune, including:

* Learning rate
* Batch size
* Number of layers and neurons
* Activation functions
* Regularization techniques

Slide 18: Hyperparameter Tuning for Deep Learning

Here's an example using Keras Tuner for hyperparameter optimization:

```python
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    
    return model

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='my_dir',
    project_name='intro_to_kt')

# Assuming X_train, y_train, X_val, y_val are your data
tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps.values}")
```

Slide 19: Additional Resources

For further exploration of Machine Learning and Hyperparameter Tuning, consider these resources:

1. "Hyperparameter Optimization in Machine Learning" by Claesen & De Moor (2015) ArXiv: [https://arxiv.org/abs/1502.02127](https://arxiv.org/abs/1502.02127)
2. "Random Search for Hyper-Parameter Optimization" by Bergstra & Bengio (2012) ArXiv: [https://arxiv.org/abs/1212.5745](https://arxiv.org/abs/1212.5745)
3. "Algorithms for Hyper-Parameter Optimization" by Bergstra et al. (2011) ArXiv: [https://arxiv.org/abs/1206.2944](https://arxiv.org/abs/1206.2944)
4. "Practical Bayesian Optimization of Machine Learning Algorithms" by Snoek et al. (2012) ArXiv: [https://arxiv.org/abs/1206.2944](https://arxiv.org/abs/1206.2944)

These papers provide in-depth discussions on various hyperparameter tuning techniques and their applications in machine learning.

