## Stacking Ensemble Learning with Python
Slide 1: Introduction to Stacking in Ensemble Learning

Stacking, also known as stacked generalization, is an ensemble learning technique that combines multiple models to improve prediction accuracy. It leverages the strengths of various base models by training a meta-model on their outputs. This approach often yields better performance than individual models or simple averaging methods.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
base_models = [
    RandomForestClassifier(n_estimators=10, random_state=42),
    SVC(kernel='rbf', probability=True, random_state=42),
    LogisticRegression(random_state=42)
]

# Train base models
for model in base_models:
    model.fit(X_train, y_train)

print("Base models trained successfully")
```

Slide 2: The Stacking Process

The stacking process involves two main steps: training base models and training a meta-model. First, multiple base models are trained on the original dataset. Then, their predictions are used as features to train a meta-model, which makes the final prediction. This hierarchical approach allows the meta-model to learn how to best combine the base models' predictions.

```python
import numpy as np
from sklearn.model_selection import cross_val_predict

# Generate meta-features
meta_features = np.zeros((X_train.shape[0], len(base_models)))

for i, model in enumerate(base_models):
    meta_features[:, i] = cross_val_predict(model, X_train, y_train, cv=5, method='predict_proba')[:, 1]

# Train meta-model
meta_model = LogisticRegression(random_state=42)
meta_model.fit(meta_features, y_train)

print("Meta-model trained successfully")
```

Slide 3: Implementing Stacking with Scikit-learn

Scikit-learn provides a convenient StackingClassifier class for implementing stacking. This class automatically handles the process of generating meta-features and training the meta-model. It also supports cross-validation for creating out-of-fold predictions during training.

```python
from sklearn.ensemble import StackingClassifier

# Define base models and meta-model
base_models = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('lr', LogisticRegression(random_state=42))
]
meta_model = LogisticRegression(random_state=42)

# Create and train the stacking classifier
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
stacking_clf.fit(X_train, y_train)

print("Stacking classifier trained successfully")
```

Slide 4: Evaluating Stacking Performance

To assess the performance of a stacking model, we compare its accuracy with that of individual base models. This comparison helps us understand the effectiveness of the stacking approach and whether it provides an improvement over single models.

```python
from sklearn.metrics import accuracy_score

# Evaluate base models
for name, model in base_models:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")

# Evaluate stacking model
stacking_pred = stacking_clf.predict(X_test)
stacking_accuracy = accuracy_score(y_test, stacking_pred)
print(f"Stacking Accuracy: {stacking_accuracy:.4f}")
```

Slide 5: Feature Importance in Stacking

Understanding feature importance in a stacking model can provide insights into which base models contribute most to the final prediction. For linear meta-models like logistic regression, we can examine the coefficients to determine the relative importance of each base model.

```python
import matplotlib.pyplot as plt

# Get feature importances (coefficients) from the meta-model
importances = stacking_clf.final_estimator_.coef_[0]
model_names = [name for name, _ in base_models]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(model_names, importances)
plt.title("Base Model Importance in Stacking")
plt.xlabel("Base Models")
plt.ylabel("Importance")
plt.show()
```

Slide 6: Handling Multi-class Classification

Stacking can be extended to multi-class classification problems. In this case, base models need to provide probability estimates for each class, and the meta-model combines these probabilities to make the final prediction.

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

# Load multi-class dataset
iris = load_iris()
X, y = iris.data, iris.target
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models and meta-model for multi-class classification
base_models = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('lr', LogisticRegression(multi_class='ovr', random_state=42))
]
meta_model = LogisticRegression(multi_class='multinomial', random_state=42)

# Create and train the stacking classifier
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
stacking_clf.fit(X_train, y_train)

# Evaluate stacking model
stacking_pred = stacking_clf.predict(X_test)
stacking_accuracy = accuracy_score(y_test, stacking_pred)
print(f"Multi-class Stacking Accuracy: {stacking_accuracy:.4f}")
```

Slide 7: Customizing the Meta-model

The choice of meta-model can significantly impact the performance of a stacking ensemble. While logistic regression is commonly used, other algorithms like gradient boosting or neural networks can potentially capture more complex relationships between base model predictions.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Define different meta-models
meta_models = [
    ('lr', LogisticRegression(random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('nn', MLPClassifier(hidden_layer_sizes=(50, 25), random_state=42))
]

# Train and evaluate stacking with different meta-models
for name, meta_model in meta_models:
    stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
    stacking_clf.fit(X_train, y_train)
    stacking_pred = stacking_clf.predict(X_test)
    accuracy = accuracy_score(y_test, stacking_pred)
    print(f"Stacking with {name} meta-model - Accuracy: {accuracy:.4f}")
```

Slide 8: Handling Imbalanced Datasets

When working with imbalanced datasets, stacking can be combined with resampling techniques to improve performance. This approach involves applying resampling methods to the training data before fitting the base models and meta-model.

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Generate an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models and meta-model
base_models = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('lr', LogisticRegression(random_state=42))
]
meta_model = LogisticRegression(random_state=42)

# Create a pipeline with SMOTE and StackingClassifier
imbalanced_stacking = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('stacking', StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5))
])

# Train and evaluate the imbalanced stacking model
imbalanced_stacking.fit(X_train, y_train)
y_pred = imbalanced_stacking.predict(X_test)
print(classification_report(y_test, y_pred))
```

Slide 9: Stacking with Cross-validation

Cross-validation is crucial in stacking to prevent overfitting and ensure that the meta-model generalizes well. Scikit-learn's StackingClassifier uses cross-validation by default to generate out-of-fold predictions for training the meta-model.

```python
from sklearn.model_selection import cross_val_score

# Define base models and meta-model
base_models = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('lr', LogisticRegression(random_state=42))
]
meta_model = LogisticRegression(random_state=42)

# Create stacking classifier with cross-validation
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# Perform cross-validation on the stacking classifier
cv_scores = cross_val_score(stacking_clf, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")
```

Slide 10: Feature Engineering in Stacking

Feature engineering can be incorporated into the stacking process to create more informative meta-features. This can involve applying transformations to the base model predictions or combining them with original features.

```python
from sklearn.preprocessing import PolynomialFeatures

# Generate base model predictions
base_predictions = np.column_stack([model.predict_proba(X_train)[:, 1] for _, model in base_models])

# Apply polynomial features to base predictions
poly = PolynomialFeatures(degree=2, include_bias=False)
meta_features = poly.fit_transform(base_predictions)

# Combine meta-features with original features
enhanced_features = np.hstack((X_train, meta_features))

# Train meta-model on enhanced features
meta_model = LogisticRegression(random_state=42)
meta_model.fit(enhanced_features, y_train)

# Make predictions on test set
base_predictions_test = np.column_stack([model.predict_proba(X_test)[:, 1] for _, model in base_models])
meta_features_test = poly.transform(base_predictions_test)
enhanced_features_test = np.hstack((X_test, meta_features_test))
y_pred = meta_model.predict(enhanced_features_test)

print(f"Enhanced Stacking Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

Slide 11: Real-life Example: Image Classification

Stacking can be effectively used in image classification tasks. In this example, we'll use a pre-trained convolutional neural network (CNN) as one of the base models, along with traditional machine learning models, to classify images of fruits.

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create image data generator
datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess images (assuming you have a 'fruits' directory with subdirectories for each class)
train_generator = datagen.flow_from_directory('fruits/train', target_size=(224, 224), batch_size=32)
test_generator = datagen.flow_from_directory('fruits/test', target_size=(224, 224), batch_size=32)

# Extract features using VGG16
train_features = base_model.predict(train_generator)
test_features = base_model.predict(test_generator)

# Flatten features
train_features_flat = train_features.reshape(train_features.shape[0], -1)
test_features_flat = test_features.reshape(test_features.shape[0], -1)

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
]

# Create stacking classifier
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5)

# Train stacking classifier
stacking_clf.fit(train_features_flat, train_generator.classes)

# Evaluate stacking classifier
accuracy = stacking_clf.score(test_features_flat, test_generator.classes)
print(f"Stacking Accuracy on Fruit Classification: {accuracy:.4f}")
```

Slide 12: Real-life Example: Text Classification

Stacking can also be applied to text classification tasks. In this example, we'll use a combination of traditional machine learning models and a simple neural network to classify news articles into different categories.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Load text data (assuming you have 'X_text' and 'y' variables with text and labels)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
def create_nn():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(5000,)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

base_models = [
    ('nb', MultinomialNB()),
    ('sgd', SGDClassifier(random_state=42)),
    ('nn', KerasClassifier(build_fn=create_nn, epochs=10, batch_size=32, verbose=0))
]

# Create stacking classifier
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5)

# Train stacking classifier
stacking_clf.fit(X_train, y_train)

# Evaluate stacking classifier
accuracy = stacking_clf.score(X_test, y_test)
print(f"Stacking Accuracy on Text Classification: {accuracy:.4f}")
```

Slide 13: Hyperparameter Tuning for Stacking

Optimizing hyperparameters for both base models and the meta-model can significantly improve stacking performance. Grid search or random search can be used to find the best combination of hyperparameters for the entire stacking ensemble.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define base models with hyperparameter distributions
base_models = [
    ('rf', RandomForestClassifier(), {
        'rf__n_estimators': randint(10, 100),
        'rf__max_depth': randint(2, 10)
    }),
    ('svm', SVC(probability=True), {
        'svm__C': uniform(0.1, 10),
        'svm__kernel': ['rbf', 'poly']
    }),
    ('lr', LogisticRegression(), {
        'lr__C': uniform(0.1, 10)
    })
]

# Create stacking classifier
stacking_clf = StackingClassifier(
    estimators=[('rf', RandomForestClassifier()), ('svm', SVC(probability=True)), ('lr', LogisticRegression())],
    final_estimator=LogisticRegression(),
    cv=5
)

# Define hyperparameter distributions for stacking
param_distributions = {
    'rf__n_estimators': randint(10, 100),
    'rf__max_depth': randint(2, 10),
    'svm__C': uniform(0.1, 10),
    'svm__kernel': ['rbf', 'poly'],
    'lr__C': uniform(0.1, 10),
    'final_estimator__C': uniform(0.1, 10)
}

# Perform random search
random_search = RandomizedSearchCV(stacking_clf, param_distributions, n_iter=100, cv=5, n_jobs=-1)
random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}")
```

Slide 14: Ensemble Diversity in Stacking

Ensuring diversity among base models is crucial for effective stacking. Diverse models capture different aspects of the data, leading to more robust predictions. We can measure diversity using metrics like the correlation between model predictions.

```python
import seaborn as sns

# Train base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('lr', LogisticRegression(random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]

# Get predictions from base models
predictions = []
for name, model in base_models:
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]
    predictions.append(pred)

# Calculate correlation between model predictions
correlation_matrix = np.corrcoef(predictions)

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, xticklabels=[name for name, _ in base_models], yticklabels=[name for name, _ in base_models])
plt.title("Correlation between Base Model Predictions")
plt.show()
```

Slide 15: Advanced Stacking Techniques

Advanced stacking techniques can further improve performance. These include multi-level stacking, where multiple layers of meta-models are used, and blending, which uses a hold-out set for training the meta-model instead of cross-validation.

```python
# Multi-level stacking (pseudocode)
def multi_level_stacking(X_train, y_train, X_test, base_models, meta_models, final_model):
    # Level 1: Train base models and get predictions
    level1_train = get_base_model_predictions(X_train, y_train, base_models)
    level1_test = get_base_model_predictions(X_test, base_models)
    
    # Level 2: Train meta-models on level 1 predictions
    level2_train = get_meta_model_predictions(level1_train, y_train, meta_models)
    level2_test = get_meta_model_predictions(level1_test, meta_models)
    
    # Level 3: Train final model on level 2 predictions
    final_model.fit(level2_train, y_train)
    final_predictions = final_model.predict(level2_test)
    
    return final_predictions

# Blending (pseudocode)
def blending(X_train, y_train, X_val, y_val, X_test, base_models, meta_model):
    # Train base models on training data
    for model in base_models:
        model.fit(X_train, y_train)
    
    # Get base model predictions on validation set
    val_predictions = get_base_model_predictions(X_val, base_models)
    
    # Train meta-model on validation predictions
    meta_model.fit(val_predictions, y_val)
    
    # Get base model predictions on test set
    test_predictions = get_base_model_predictions(X_test, base_models)
    
    # Make final predictions using meta-model
    final_predictions = meta_model.predict(test_predictions)
    
    return final_predictions
```

Slide 16: Additional Resources

For those interested in delving deeper into stacking and ensemble learning, here are some valuable resources:

1. "Stacked Generalization" by David H. Wolpert (1992) - The original paper introducing the concept of stacking. ArXiv: [https://arxiv.org/abs/1503.06451](https://arxiv.org/abs/1503.06451)
2. "Ensemble Methods in Machine Learning" by Thomas G. Dietterich (2000) - A comprehensive overview of ensemble methods, including stacking. ArXiv: [https://arxiv.org/abs/1106.0257](https://arxiv.org/abs/1106.0257)
3. "A Survey of Methods for Managing the Classification and Solution of Data Mining Tasks" by A. Seewald (2002) - Discusses various ensemble methods, including stacking. ArXiv: [https://arxiv.org/abs/1206.5538](https://arxiv.org/abs/1206.5538)

These resources provide in-depth explanations and theoretical foundations for stacking and other ensemble learning techniques.

