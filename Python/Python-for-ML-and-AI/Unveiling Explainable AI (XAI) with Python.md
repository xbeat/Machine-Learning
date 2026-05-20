## Unveiling Explainable AI (XAI) with Python
Slide 1: Introduction to Explainable AI (XAI)

Explainable AI (XAI) is a set of techniques and methods that allow humans to understand and interpret the decisions made by artificial intelligence systems. As AI becomes more prevalent in our daily lives, the need for transparency and accountability in these systems grows. XAI aims to bridge the gap between complex AI models and human understanding, making AI more trustworthy and accessible.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple example of a decision boundary
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Simple Decision Boundary')
plt.colorbar(label='Class')
plt.show()
```

Slide 2: Importance of XAI

XAI is crucial in various domains where decision-making processes need to be transparent and accountable. It helps in building trust, ensuring fairness, and enabling debugging of AI systems. By making AI models more interpretable, we can identify and mitigate biases, comply with regulations, and improve the overall performance of our models.

```python
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load iris dataset
X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names

# Train a random forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Create a SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)

# Plot SHAP summary
shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar")
```

Slide 3: LIME (Local Interpretable Model-agnostic Explanations)

LIME is a popular XAI technique that explains individual predictions by approximating the model locally with an interpretable model. It works by perturbing the input and observing how the predictions change, then fitting a simple model around the instance to be explained.

```python
from lime import lime_tabular
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load breast cancer dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train a random forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Create a LIME explainer
explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=data.feature_names, class_names=data.target_names, mode='classification')

# Explain a single prediction
exp = explainer.explain_instance(X_test[0], rf_model.predict_proba, num_features=5)
exp.show_in_notebook(show_table=True)
```

Slide 4: SHAP (SHapley Additive exPlanations)

SHAP is another popular XAI method based on game theory. It assigns each feature an importance value for a particular prediction. SHAP values provide a unified measure of feature importance that shows how much each feature contributes, positively or negatively, to the prediction.

```python
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

# Load California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Train a random forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Plot SHAP summary
shap.summary_plot(shap_values, X, feature_names=housing.feature_names)
```

Slide 5: Feature Importance

Feature importance is a simple yet effective way to understand which features contribute most to the predictions of a model. Many machine learning algorithms provide built-in methods to calculate feature importance, which can be used as a starting point for model interpretation.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a random forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get feature importances
importances = rf_model.feature_importances_
feature_names = iris.feature_names

# Sort features by importance
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()
```

Slide 6: Partial Dependence Plots

Partial Dependence Plots (PDPs) show the marginal effect of a feature on the predicted outcome of a machine learning model. They help visualize how changes in a feature affect predictions while keeping other features constant.

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence

# Load California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Train a gradient boosting regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X, y)

# Create partial dependence plot
features = [0, 1, 2]  # Indices of features to plot
fig, ax = plt.subplots(figsize=(12, 4))
plot_partial_dependence(gb_model, X, features, feature_names=housing.feature_names, ax=ax)
plt.tight_layout()
plt.show()
```

Slide 7: Counterfactual Explanations

Counterfactual explanations provide insights into how to change the input to achieve a desired output. They answer questions like "What would need to change for this model to predict a different outcome?" This approach is particularly useful in scenarios where we want to understand how to improve or change a specific prediction.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a random forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Function to find counterfactual
def find_counterfactual(model, instance, desired_class, feature_names):
    counterfactual = instance.()
    for i in range(len(instance)):
        for step in np.linspace(-1, 1, 20):
            counterfactual[i] = instance[i] + step
            if model.predict([counterfactual])[0] == desired_class:
                return counterfactual
    return None

# Example usage
instance = X[0]  # Choose an instance
desired_class = 1  # Desired output class

counterfactual = find_counterfactual(rf_model, instance, desired_class, iris.feature_names)

if counterfactual is not None:
    print("Original instance:", instance)
    print("Counterfactual:", counterfactual)
    print("Changes needed:")
    for i, (orig, count) in enumerate(zip(instance, counterfactual)):
        if orig != count:
            print(f"{iris.feature_names[i]}: {orig:.2f} -> {count:.2f}")
else:
    print("No counterfactual found")
```

Slide 8: Global Surrogate Models

Global surrogate models are interpretable models that approximate the behavior of a complex black-box model. By training a simpler, more interpretable model (like a decision tree) to mimic the predictions of a complex model, we can gain insights into the overall behavior of the black-box model.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a complex model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get predictions from the complex model
y_pred = rf_model.predict(X)

# Train a simple surrogate model (Decision Tree)
surrogate_model = DecisionTreeClassifier(max_depth=3, random_state=42)
surrogate_model.fit(X, y_pred)

# Plot the surrogate decision tree
plt.figure(figsize=(20,10))
plot_tree(surrogate_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Global Surrogate Model (Decision Tree)")
plt.show()

# Evaluate surrogate model fidelity
surrogate_pred = surrogate_model.predict(X)
fidelity = accuracy_score(y_pred, surrogate_pred)
print(f"Surrogate model fidelity: {fidelity:.2f}")
```

Slide 9: Local Surrogate Models

While global surrogate models aim to explain the entire model, local surrogate models focus on explaining individual predictions. These models are trained on a small neighborhood around the instance of interest, providing a more accurate local approximation of the complex model's behavior.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a complex model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Function to create a local surrogate model
def local_surrogate(model, instance, n_samples=1000, radius=0.5):
    # Generate samples around the instance
    local_samples = np.random.normal(instance, radius, (n_samples, len(instance)))
    
    # Get predictions from the complex model
    local_predictions = model.predict(local_samples)
    
    # Train a simple surrogate model on the local samples
    surrogate = DecisionTreeClassifier(max_depth=3)
    surrogate.fit(local_samples, local_predictions)
    
    return surrogate

# Example usage
instance = X[0]  # Choose an instance to explain
local_model = local_surrogate(rf_model, instance)

# Plot the local surrogate decision tree
plt.figure(figsize=(15,10))
plot_tree(local_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Local Surrogate Model (Decision Tree)")
plt.show()
```

Slide 10: Integrated Gradients

Integrated Gradients is a method for attributing a deep network's prediction to its input features. It's especially useful for explaining predictions of deep learning models. The method computes the integral of the gradients along a straight line path from a baseline input to the actual input.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# Integrated Gradients function
def integrated_gradients(model, input_tensor, baseline, steps=50):
    input_tensor.requires_grad = True
    path = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]
    gradients = []
    for p in path:
        output = model(p)
        gradient = torch.autograd.grad(outputs=output, inputs=p)[0]
        gradients.append(gradient)
    avg_gradients = torch.stack(gradients).mean(dim=0)
    integrated_grads = (input_tensor - baseline) * avg_gradients
    return integrated_grads

# Create and use the model
model = SimpleNN()
input_tensor = torch.tensor([[1.0, 2.0]], requires_grad=True)
baseline = torch.zeros_like(input_tensor)

# Compute integrated gradients
ig = integrated_gradients(model, input_tensor, baseline)

# Visualize the results
plt.bar(range(2), ig.detach().numpy()[0])
plt.title("Feature Attributions using Integrated Gradients")
plt.xlabel("Features")
plt.ylabel("Attribution")
plt.show()
```

Slide 11: Real-life Example: Medical Diagnosis

In medical diagnosis, explainable AI can help doctors understand and trust AI-assisted diagnoses. For instance, in skin cancer detection, an XAI system could highlight the specific areas of a skin lesion image that contributed most to the diagnosis, helping doctors verify the AI's decision.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Simulated skin lesion features
features = ['asymmetry', 'border_irregularity', 'color_variegation', 'diameter', 'evolution']
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = (X[:, 0] * 0.3 + X[:, 1] * 0.3 + X[:, 2] * 0.2 + X[:, 3] * 0.1 + X[:, 4] * 0.1 > 0.6).astype(int)

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Function to explain a prediction
def explain_prediction(model, instance, feature_names):
    prediction = model.predict([instance])[0]
    feature_importance = model.feature_importances_
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, instance * feature_importance)
    plt.title(f"Diagnosis: {'Malignant' if prediction == 1 else 'Benign'}")
    plt.xlabel("Features")
    plt.ylabel("Contribution to Prediction")
    plt.show()

# Example usage
sample_instance = X[0]
explain_prediction(model, sample_instance, features)
```

Slide 12: Real-life Example: Environmental Science

In environmental science, XAI can be used to interpret models predicting air quality. This can help policymakers understand which factors contribute most to poor air quality and make informed decisions about environmental regulations.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Simulated air quality data
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'temperature': np.random.normal(25, 5, n_samples),
    'humidity': np.random.normal(60, 10, n_samples),
    'wind_speed': np.random.normal(10, 3, n_samples),
    'vehicle_emissions': np.random.normal(50, 10, n_samples),
    'industrial_emissions': np.random.normal(30, 5, n_samples)
})

# Target variable: Air Quality Index (AQI)
data['AQI'] = (
    0.3 * data['temperature'] +
    0.2 * data['humidity'] -
    0.1 * data['wind_speed'] +
    0.25 * data['vehicle_emissions'] +
    0.15 * data['industrial_emissions'] +
    np.random.normal(0, 5, n_samples)
)

# Split the data
X = data.drop('AQI', axis=1)
y = data['AQI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_
features = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(features, importances)
plt.title('Feature Importances for Air Quality Prediction')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print feature importances
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.4f}")
```

Slide 13: Challenges in XAI

While XAI offers numerous benefits, it also faces several challenges:

1. Trade-off between model performance and interpretability
2. Ensuring the stability and consistency of explanations
3. Handling high-dimensional data and complex model architectures
4. Balancing local and global explanations
5. Addressing the potential for adversarial attacks on explanations

To illustrate the challenge of stability, let's consider a simple example:

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Generate slightly different datasets
np.random.seed(42)
X1 = np.random.rand(100, 2)
X2 = X1 + np.random.normal(0, 0.01, X1.shape)  # Small perturbation
y = (X1[:, 0] + X1[:, 1] > 1).astype(int)

# Train two models on slightly different data
model1 = DecisionTreeClassifier(max_depth=3)
model2 = DecisionTreeClassifier(max_depth=3)
model1.fit(X1, y)
model2.fit(X2, y)

# Compare feature importances
print("Model 1 feature importances:", model1.feature_importances_)
print("Model 2 feature importances:", model2.feature_importances_)
```

This example demonstrates how small changes in the input data can lead to different explanations, highlighting the importance of robust XAI methods.

Slide 14: Future Directions in XAI

The field of Explainable AI is rapidly evolving, with several promising directions for future research and development:

1. Causal explanations: Moving beyond correlations to understand causal relationships in AI decisions.
2. Interactive explanations: Developing tools that allow users to explore and interact with explanations.
3. Multi-modal explanations: Combining different types of explanations (e.g., visual and textual) for more comprehensive understanding.
4. Explanations for deep learning: Improving methods to interpret complex neural network architectures.
5. Standardization and benchmarks: Establishing common metrics and datasets for evaluating XAI methods.

Here's a simple example of how we might approach interactive explanations:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Train a decision tree
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# Function to plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()

# Plot initial decision boundary
plot_decision_boundary(model, X, y)

# In an interactive setting, users could modify the model (e.g., change max_depth)
# and immediately see the impact on the decision boundary and explanations
```

Slide 15: Additional Resources

For those interested in diving deeper into Explainable AI, here are some valuable resources:

1. "Interpretable Machine Learning" by Christoph Molnar ArXiv: [https://arxiv.org/abs/1901.04592](https://arxiv.org/abs/1901.04592)
2. "Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges toward Responsible AI" by Arrieta et al. ArXiv: [https://arxiv.org/abs/1910.10045](https://arxiv.org/abs/1910.10045)
3. "A Survey of Methods for Explaining Black Box Models" by Guidotti et al. ArXiv: [https://arxiv.org/abs/1802.01933](https://arxiv.org/abs/1802.01933)
4. "Explanatory Interactive Machine Learning" by Teso and Kersting ArXiv: [https://arxiv.org/abs/1909.06136](https://arxiv.org/abs/1909.06136)

These resources provide comprehensive overviews of XAI techniques, challenges, and future directions, offering both theoretical foundations and practical insights for implementing explainable AI systems.

