## Implementing Machine Learning Algorithms from Scratch in Python

Slide 1: Introduction to Ensemble Learning

Ensemble learning is a powerful machine learning technique that combines multiple models to improve overall performance. By leveraging the strengths of various weak learners, ensemble methods create a strong collective learner. This approach often yields better predictions and more robust models compared to individual algorithms.

```python
# Simple ensemble example
def ensemble_predict(models, input_data):
    predictions = [model.predict(input_data) for model in models]
    return max(set(predictions), key=predictions.count)

# Create mock models and data
class MockModel:
    def predict(self, data):
        return data % 2  # Simple mock prediction

models = [MockModel() for _ in range(5)]
input_data = 7

result = ensemble_predict(models, input_data)
print(f"Ensemble prediction: {result}")
```

Slide 2: Bagging (Bootstrap Aggregating)

Bagging is an ensemble method that trains multiple instances of the same algorithm on different subsets of the training data. These subsets are created through bootstrap sampling, which involves random sampling with replacement. The final prediction is typically made by averaging or voting.

```python
import random

def bootstrap_sample(data, sample_size):
    return [random.choice(data) for _ in range(sample_size)]

def bagging_predict(base_model, data, num_models):
    predictions = []
    for _ in range(num_models):
        sample = bootstrap_sample(data, len(data))
        model = base_model()
        model.fit(sample)
        predictions.append(model.predict(data))
    
    return [max(set(col), key=col.count) for col in zip(*predictions)]

# Example usage
class SimpleClassifier:
    def fit(self, data):
        self.threshold = sum(data) / len(data)
    
    def predict(self, data):
        return [1 if x > self.threshold else 0 for x in data]

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
bagged_predictions = bagging_predict(SimpleClassifier, data, 5)
print(f"Bagged predictions: {bagged_predictions}")
```

Slide 3: Boosting

Boosting is a sequential ensemble method that iteratively trains weak learners, with each subsequent model focusing on the errors of the previous ones. This approach gradually improves the overall model's performance by giving more weight to misclassified instances.

```python
class SimpleBoostingClassifier:
    def __init__(self, num_estimators):
        self.num_estimators = num_estimators
        self.models = []
        self.weights = []

    def fit(self, X, y):
        sample_weights = [1/len(X)] * len(X)
        
        for _ in range(self.num_estimators):
            model = SimpleClassifier()
            model.fit(X, y, sample_weights)
            predictions = model.predict(X)
            
            error = sum((p != y[i]) * sample_weights[i] for i, p in enumerate(predictions))
            alpha = 0.5 * math.log((1 - error) / error)
            
            self.models.append(model)
            self.weights.append(alpha)
            
            # Update sample weights
            for i in range(len(X)):
                sample_weights[i] *= math.exp(-alpha * y[i] * predictions[i])
            
            # Normalize weights
            total = sum(sample_weights)
            sample_weights = [w / total for w in sample_weights]

    def predict(self, X):
        predictions = []
        for x in X:
            weighted_preds = sum(w * m.predict([x])[0] for w, m in zip(self.weights, self.models))
            predictions.append(1 if weighted_preds > 0 else -1)
        return predictions

# Example usage
X = [[1], [2], [3], [4], [5]]
y = [-1, -1, 1, 1, 1]

boosting_model = SimpleBoostingClassifier(num_estimators=3)
boosting_model.fit(X, y)
predictions = boosting_model.predict(X)
print(f"Boosting predictions: {predictions}")
```

Slide 4: Stacking

Stacking is an ensemble method that combines predictions from multiple diverse base models (level-0 models) using a meta-model (level-1 model). This approach allows the meta-model to learn how to best combine the base models' predictions, potentially capturing complex relationships between them.

```python
import random

class BaseModel:
    def __init__(self, name):
        self.name = name
    
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        return [random.choice([0, 1]) for _ in range(len(X))]

class MetaModel:
    def fit(self, X, y):
        self.weights = [random.random() for _ in range(len(X[0]))]
    
    def predict(self, X):
        return [1 if sum(x[i] * self.weights[i] for i in range(len(x))) > 0.5 else 0 for x in X]

def stacking(base_models, meta_model, X, y):
    # Train base models
    for model in base_models:
        model.fit(X, y)
    
    # Generate meta-features
    meta_features = [[model.predict([x])[0] for model in base_models] for x in X]
    
    # Train meta-model
    meta_model.fit(meta_features, y)
    
    return meta_features, meta_model

# Example usage
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [0, 1, 1, 0]

base_models = [BaseModel(f"Model_{i}") for i in range(3)]
meta_model = MetaModel()

meta_features, trained_meta_model = stacking(base_models, meta_model, X, y)
final_predictions = trained_meta_model.predict(meta_features)
print(f"Stacking predictions: {final_predictions}")
```

Slide 5: Real-Life Example: Image Classification

Ensemble methods are widely used in image classification tasks. For instance, in a medical imaging application, multiple models can be combined to improve the accuracy of disease detection.

```python
import random

class ImageClassifier:
    def __init__(self, name):
        self.name = name
    
    def predict(self, image):
        # Simulate prediction (in reality, this would involve complex image processing)
        return random.choice(["benign", "malignant"])

def ensemble_classify(classifiers, image):
    predictions = [clf.predict(image) for clf in classifiers]
    return max(set(predictions), key=predictions.count)

# Simulate image data
image = [[[random.randint(0, 255) for _ in range(3)] for _ in range(64)] for _ in range(64)]

# Create ensemble of classifiers
classifiers = [
    ImageClassifier("CNN"),
    ImageClassifier("ResNet"),
    ImageClassifier("VGG")
]

# Perform ensemble classification
result = ensemble_classify(classifiers, image)
print(f"Ensemble classification result: {result}")
```

Slide 6: Real-Life Example: Weather Forecasting

Ensemble methods are crucial in weather forecasting, where multiple models are combined to improve prediction accuracy.

```python
import random

class WeatherModel:
    def __init__(self, name):
        self.name = name
    
    def predict(self, data):
        # Simulate weather prediction (in reality, this would involve complex atmospheric modeling)
        return random.uniform(0, 35)  # Temperature in Celsius

def ensemble_forecast(models, data):
    predictions = [model.predict(data) for model in models]
    return sum(predictions) / len(predictions)

# Simulate weather data
weather_data = {
    "temperature": 20,
    "humidity": 65,
    "pressure": 1013,
    "wind_speed": 10
}

# Create ensemble of weather models
models = [
    WeatherModel("GFS"),
    WeatherModel("ECMWF"),
    WeatherModel("UKMO")
]

# Perform ensemble forecasting
forecast = ensemble_forecast(models, weather_data)
print(f"Ensemble weather forecast: {forecast:.2f}°C")
```

Slide 7: Implementing Bagging from Scratch

Let's implement a simple bagging classifier from scratch using only built-in Python functions.

```python
import random

class SimpleBaggingClassifier:
    def __init__(self, base_model, n_estimators=10):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models = []

    def bootstrap_sample(self, X, y):
        n_samples = len(X)
        indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
        return [X[i] for i in indices], [y[i] for i in indices]

    def fit(self, X, y):
        self.models = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            model = self.base_model()
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        predictions = [[model.predict([x])[0] for model in self.models] for x in X]
        return [max(set(pred), key=pred.count) for pred in predictions]

# Example usage
class SimpleClassifier:
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        return [self.y[self.X.index(x)] if x in self.X else random.choice(self.y) for x in X]

X = [[1], [2], [3], [4], [5]]
y = [0, 0, 1, 1, 1]

bagging_clf = SimpleBaggingClassifier(SimpleClassifier, n_estimators=5)
bagging_clf.fit(X, y)
predictions = bagging_clf.predict([[2], [4]])
print(f"Bagging predictions: {predictions}")
```

Slide 8: Implementing Boosting from Scratch

Now, let's implement a simple boosting algorithm (similar to AdaBoost) from scratch using only built-in Python functions.

```python
import math

class SimpleBoostingClassifier:
    def __init__(self, base_model, n_estimators=10):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = len(X)
        weights = [1/n_samples] * n_samples

        for _ in range(self.n_estimators):
            model = self.base_model()
            model.fit(X, y, weights)
            predictions = model.predict(X)

            error = sum(w for i, w in enumerate(weights) if predictions[i] != y[i])
            alpha = 0.5 * math.log((1 - error) / max(error, 1e-10))

            self.models.append(model)
            self.alphas.append(alpha)

            # Update weights
            for i in range(n_samples):
                weights[i] *= math.exp(-alpha * y[i] * predictions[i])

            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]

    def predict(self, X):
        return [1 if sum(a * m.predict([x])[0] for a, m in zip(self.alphas, self.models)) > 0 else -1 for x in X]

# Example usage
class SimpleWeakLearner:
    def fit(self, X, y, weights):
        self.X = X
        self.y = y
        self.weights = weights

    def predict(self, X):
        return [self.y[self.X.index(x)] if x in self.X else 1 for x in X]

X = [[1], [2], [3], [4], [5]]
y = [-1, -1, 1, 1, 1]

boosting_clf = SimpleBoostingClassifier(SimpleWeakLearner, n_estimators=5)
boosting_clf.fit(X, y)
predictions = boosting_clf.predict([[2], [4]])
print(f"Boosting predictions: {predictions}")
```

Slide 9: Implementing Stacking from Scratch

Let's implement a simple stacking ensemble from scratch using only built-in Python functions.

```python
import random

class SimpleStackingEnsemble:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        # Train base models
        for model in self.base_models:
            model.fit(X, y)

        # Generate meta-features
        meta_features = self._get_meta_features(X)

        # Train meta-model
        self.meta_model.fit(meta_features, y)

    def _get_meta_features(self, X):
        return [[model.predict([x])[0] for model in self.base_models] for x in X]

    def predict(self, X):
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)

# Example usage
class SimpleClassifier:
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        return [self.y[self.X.index(x)] if x in self.X else random.choice(self.y) for x in X]

class SimpleMetaClassifier:
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        return [self.y[self.X.index(tuple(x))] if tuple(x) in self.X else random.choice(self.y) for x in X]

X = [[1], [2], [3], [4], [5]]
y = [0, 0, 1, 1, 1]

base_models = [SimpleClassifier() for _ in range(3)]
meta_model = SimpleMetaClassifier()

stacking_clf = SimpleStackingEnsemble(base_models, meta_model)
stacking_clf.fit(X, y)
predictions = stacking_clf.predict([[2], [4]])
print(f"Stacking predictions: {predictions}")
```

Slide 10: Ensemble Methods Comparison

Let's compare the performance of different ensemble methods on a simple dataset.

```python
import random

def create_dataset(size):
    X = [[random.uniform(0, 10)] for _ in range(size)]
    y = [1 if x[0] > 5 else 0 for x in X]
    return X, y

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = sum(p == y for p, y in zip(predictions, y_test)) / len(y_test)
    return accuracy

# Create dataset
X_train, y_train = create_dataset(1000)
X_test, y_test = create_dataset(200)

# Train and evaluate models
bagging_clf = SimpleBaggingClassifier(SimpleClassifier, n_estimators=10)
bagging_clf.fit(X_train, y_train)
bagging_accuracy = evaluate_model(bagging_clf, X_test, y_test)

boosting_clf = SimpleBoostingClassifier(SimpleWeakLearner, n_estimators=10)
boosting_clf.fit(X_train, y_train)
boosting_accuracy = evaluate_model(boosting_clf, X_test, y_test)

base_models = [SimpleClassifier() for _ in range(3)]
meta_model = SimpleMetaClassifier()
stacking_clf = SimpleStackingEnsemble(base_models, meta_model)
stacking_clf.fit(X_train, y_train)
stacking_accuracy = evaluate_model(stacking_clf, X_test, y_test)

print(f"Bagging Accuracy: {bagging_accuracy:.4f}")
print(f"Boosting Accuracy: {boosting_accuracy:.4f}")
print(f"Stacking Accuracy: {stacking_accuracy:.4f}")
```

Slide 11: Ensemble Methods in Regression

Ensemble methods are not limited to classification problems; they can also be applied to regression tasks. Let's implement a simple ensemble for regression.

```python
import random

class SimpleRegressionEnsemble:
    def __init__(self, base_models):
        self.base_models = base_models

    def fit(self, X, y):
        for model in self.base_models:
            model.fit(X, y)

    def predict(self, X):
        predictions = [model.predict(X) for model in self.base_models]
        return [sum(pred) / len(pred) for pred in zip(*predictions)]

class SimpleLinearRegression:
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        distances = [abs(x[0] - x_train[0]) for x_train in self.X]
        nearest_index = distances.index(min(distances))
        return self.y[nearest_index]

# Create dataset
X_train = [[i] for i in range(100)]
y_train = [i + random.uniform(-10, 10) for i in range(100)]

# Create and train ensemble
base_models = [SimpleLinearRegression() for _ in range(5)]
ensemble = SimpleRegressionEnsemble(base_models)
ensemble.fit(X_train, y_train)

# Make predictions
X_test = [[25], [50], [75]]
predictions = ensemble.predict(X_test)
print(f"Ensemble Regression Predictions: {predictions}")
```

Slide 12: Handling Imbalanced Datasets with Ensembles

Ensemble methods can be particularly effective in handling imbalanced datasets. Let's implement a simple ensemble approach for this scenario.

```python
import random

class ImbalancedDatasetEnsemble:
    def __init__(self, base_model, n_estimators=5):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        minority_class = min(set(y), key=y.count)
        minority_indices = [i for i, label in enumerate(y) if label == minority_class]
        
        for _ in range(self.n_estimators):
            sampled_indices = random.choices(range(len(X)), k=len(X))
            sampled_indices += minority_indices  # Ensure all minority samples are included
            
            X_sampled = [X[i] for i in sampled_indices]
            y_sampled = [y[i] for i in sampled_indices]
            
            model = self.base_model()
            model.fit(X_sampled, y_sampled)
            self.models.append(model)

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return [max(set(pred), key=pred.count) for pred in zip(*predictions)]

# Create imbalanced dataset
X = [[random.uniform(0, 10)] for _ in range(1000)]
y = [0] * 900 + [1] * 100  # 90% class 0, 10% class 1
random.shuffle(list(zip(X, y)))

# Train ensemble
ensemble = ImbalancedDatasetEnsemble(SimpleClassifier)
ensemble.fit(X, y)

# Make predictions
X_test = [[3], [7]]
predictions = ensemble.predict(X_test)
print(f"Imbalanced Dataset Ensemble Predictions: {predictions}")
```

Slide 13: Ensemble Diversity and Model Selection

The effectiveness of ensemble methods often depends on the diversity of the base models. Let's explore a simple approach to measure and select diverse models for an ensemble.

```python
import random

def model_diversity(model1, model2, X):
    predictions1 = model1.predict(X)
    predictions2 = model2.predict(X)
    return sum(p1 != p2 for p1, p2 in zip(predictions1, predictions2)) / len(X)

def select_diverse_models(models, X, n_select):
    selected = [models[0]]
    for _ in range(n_select - 1):
        max_diversity = 0
        most_diverse_model = None
        for model in models:
            if model not in selected:
                avg_diversity = sum(model_diversity(model, s, X) for s in selected) / len(selected)
                if avg_diversity > max_diversity:
                    max_diversity = avg_diversity
                    most_diverse_model = model
        if most_diverse_model:
            selected.append(most_diverse_model)
    return selected

# Create dataset and models
X = [[random.uniform(0, 10)] for _ in range(100)]
y = [0 if x[0] < 5 else 1 for x in X]
models = [SimpleClassifier() for _ in range(10)]
for model in models:
    model.fit(X, y)

# Select diverse models
diverse_models = select_diverse_models(models, X, n_select=5)

# Create ensemble with diverse models
ensemble = SimpleStackingEnsemble(diverse_models, SimpleMetaClassifier())
ensemble.fit(X, y)

# Make predictions
X_test = [[3], [7]]
predictions = ensemble.predict(X_test)
print(f"Diverse Ensemble Predictions: {predictions}")
```

Slide 14: Ensemble Methods in Time Series Forecasting

Ensemble methods can be applied to time series forecasting to improve prediction accuracy. Let's implement a simple ensemble for time series data.

```python
import random

class SimpleTimeSeriesModel:
    def fit(self, X):
        self.data = X

    def predict(self, steps):
        return [random.choice(self.data) for _ in range(steps)]

class TimeSeriesEnsemble:
    def __init__(self, base_models):
        self.base_models = base_models

    def fit(self, X):
        for model in self.base_models:
            model.fit(X)

    def predict(self, steps):
        predictions = [model.predict(steps) for model in self.base_models]
        return [sum(step_pred) / len(step_pred) for step_pred in zip(*predictions)]

# Generate time series data
time_series = [random.uniform(0, 100) for _ in range(1000)]

# Create and train ensemble
base_models = [SimpleTimeSeriesModel() for _ in range(5)]
ensemble = TimeSeriesEnsemble(base_models)
ensemble.fit(time_series)

# Make predictions
future_steps = 10
predictions = ensemble.predict(future_steps)
print(f"Time Series Ensemble Predictions for next {future_steps} steps:")
print(predictions)
```

Slide 15: Additional Resources

For those interested in delving deeper into ensemble methods, here are some valuable resources:

1.  "Ensemble Methods: Foundations and Algorithms" by Zhi-Hua Zhou (2012) ArXiv: [https://arxiv.org/abs/1404.4088](https://arxiv.org/abs/1404.4088)
2.  "Ensemble Machine Learning: Methods and Applications" edited by Oleg Okun, Giorgio Valentini, and Matteo Re (2011) ArXiv: [https://arxiv.org/abs/1106.0257](https://arxiv.org/abs/1106.0257)
3.  "A Survey of Ensemble Learning Based on Evolutionary Computation" by Xin Yao and Yong Liu (2014) ArXiv: [https://arxiv.org/abs/1412.7880](https://arxiv.org/abs/1412.7880)

These resources provide in-depth coverage of ensemble learning techniques, their theoretical foundations, and practical applications in various domains of machine learning and data science.

