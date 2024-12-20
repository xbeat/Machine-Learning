## Sensitivity Analysis in AI and ML with Python
Slide 1: Introduction to Sensitivity Matrices

Sensitivity matrices play a crucial role in understanding how changes in input parameters affect the output of machine learning models. They provide valuable insights into model behavior and help in optimization and feature selection processes.

```python
import numpy as np
import matplotlib.pyplot as plt

def simple_neural_network(X, W):
    return np.tanh(np.dot(X, W))

X = np.array([[1, 2], [3, 4], [5, 6]])
W = np.array([[0.1], [0.2]])

output = simple_neural_network(X, W)
sensitivity = np.gradient(output, X)

plt.imshow(sensitivity, cmap='viridis')
plt.colorbar()
plt.title('Sensitivity Matrix Visualization')
plt.show()
```

Slide 2: Defining Sensitivity Matrices

A sensitivity matrix represents the rate of change of model outputs with respect to its inputs. It is essentially a Jacobian matrix, where each element (i, j) represents the partial derivative of the i-th output with respect to the j-th input.

```python
def compute_sensitivity_matrix(func, X, epsilon=1e-6):
    n_samples, n_features = X.shape
    n_outputs = func(X).shape[1]
    sensitivity = np.zeros((n_samples, n_outputs, n_features))
    
    for i in range(n_features):
        X_plus = X.()
        X_plus[:, i] += epsilon
        X_minus = X.()
        X_minus[:, i] -= epsilon
        
        sensitivity[:, :, i] = (func(X_plus) - func(X_minus)) / (2 * epsilon)
    
    return sensitivity

# Example usage
def example_function(X):
    return np.column_stack([np.sin(X[:, 0]), np.cos(X[:, 1])])

X = np.random.rand(5, 2)
sensitivity = compute_sensitivity_matrix(example_function, X)
print("Sensitivity Matrix Shape:", sensitivity.shape)
print("Sensitivity Matrix:\n", sensitivity)
```

Slide 3: Importance in Machine Learning

Sensitivity matrices help identify which input features have the most significant impact on the model's predictions. This information is valuable for feature selection, model interpretation, and understanding the model's behavior under different input conditions.

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Compute sensitivity matrix
sensitivity = model.coef_

# Visualize feature importance
plt.bar(range(len(sensitivity)), np.abs(sensitivity))
plt.xlabel('Feature Index')
plt.ylabel('Absolute Sensitivity')
plt.title('Feature Importance based on Sensitivity')
plt.show()
```

Slide 4: Calculating Sensitivity Matrices

To calculate a sensitivity matrix, we compute the partial derivatives of the model's output with respect to each input feature. For complex models, this is often done numerically using finite difference methods.

```python
def numerical_gradient(func, x, epsilon=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.()
        x_plus[i] += epsilon
        x_minus = x.()
        x_minus[i] -= epsilon
        grad[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
    return grad

def model(x):
    return np.sin(x[0]) + np.cos(x[1])

x = np.array([1.0, 2.0])
sensitivity = numerical_gradient(model, x)
print("Sensitivity:", sensitivity)
```

Slide 5: Sensitivity Analysis in Neural Networks

In neural networks, sensitivity analysis helps understand how changes in input features affect the network's output. This is particularly useful for identifying important features and potential vulnerabilities in the model.

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNN()
x = torch.randn(1, 2, requires_grad=True)
output = model(x)
output.backward()

sensitivity = x.grad.numpy()
print("Input sensitivity:", sensitivity)
```

Slide 6: Sensitivity in Image Classification

In image classification tasks, sensitivity matrices can highlight regions of an image that are most influential for the model's decision. This technique is often used in explainable AI to generate saliency maps.

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

# Load and preprocess image
img = Image.open('example_image.jpg')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
input_tensor = transform(img).unsqueeze(0)
input_tensor.requires_grad = True

# Forward pass
output = model(input_tensor)

# Compute gradients
target_class = output.argmax().item()
output[0, target_class].backward()

# Generate saliency map
saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
saliency = saliency.reshape(224, 224).numpy()

# Visualize saliency map
plt.imshow(saliency, cmap='hot')
plt.axis('off')
plt.title('Saliency Map')
plt.show()
```

Slide 7: Sensitivity in Natural Language Processing

In NLP tasks, sensitivity analysis can reveal which words or phrases have the most impact on the model's predictions. This is useful for understanding model behavior and identifying potential biases.

```python
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare input
text = "This movie is great!"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Forward pass
outputs = model(**inputs)
logits = outputs.logits

# Compute gradients
logits[0, 1].backward()  # Assume positive sentiment is class 1

# Get word importance scores
word_importances = inputs['input_ids'][0].grad.abs()

# Print word importances
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
for token, importance in zip(tokens, word_importances):
    print(f"{token}: {importance.item():.4f}")
```

Slide 8: Sensitivity in Reinforcement Learning

In reinforcement learning, sensitivity analysis helps understand how changes in the environment or agent's policy affect the expected rewards. This information can guide policy optimization and environment design.

```python
import numpy as np
import gym

env = gym.make('CartPole-v1')

def simple_policy(observation):
    return 1 if observation[2] + observation[3] > 0 else 0

def evaluate_policy(env, policy, n_episodes=100):
    total_reward = 0
    for _ in range(n_episodes):
        observation = env.reset()
        done = False
        while not done:
            action = policy(observation)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
    return total_reward / n_episodes

def compute_sensitivity(env, policy, param_index, epsilon=0.1):
    def perturbed_policy(observation):
        perturbed_obs = observation.()
        perturbed_obs[param_index] += epsilon
        return policy(perturbed_obs)
    
    base_performance = evaluate_policy(env, policy)
    perturbed_performance = evaluate_policy(env, perturbed_policy)
    
    return (perturbed_performance - base_performance) / epsilon

sensitivities = [compute_sensitivity(env, simple_policy, i) for i in range(4)]
print("Sensitivities:", sensitivities)
```

Slide 9: Sensitivity in Hyperparameter Tuning

Sensitivity analysis can guide hyperparameter tuning by identifying which hyperparameters have the most significant impact on model performance. This helps focus optimization efforts on the most influential parameters.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Define hyperparameter search space
param_dist = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform randomized search
rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=5, random_state=42)
random_search.fit(X, y)

# Analyze hyperparameter importance
importances = random_search.cv_results_['mean_test_score']
param_names = list(param_dist.keys())

plt.bar(param_names, importances)
plt.xlabel('Hyperparameter')
plt.ylabel('Mean Test Score')
plt.title('Hyperparameter Importance')
plt.show()
```

Slide 10: Sensitivity in Time Series Analysis

In time series analysis, sensitivity matrices can reveal which past time steps or features have the most significant impact on future predictions. This information is valuable for feature selection and model interpretation in forecasting tasks.

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Generate synthetic time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
y = np.cumsum(np.random.randn(100)) + 100

# Fit ARIMA model
model = ARIMA(y, order=(1, 1, 1))
results = model.fit()

# Compute sensitivity to past observations
n_steps = 10
sensitivity = np.zeros(n_steps)
for i in range(n_steps):
    y_perturbed = y.()
    y_perturbed[-i-1] += 0.1
    forecast = results.forecast(steps=1, y=y_perturbed)
    sensitivity[i] = (forecast - results.forecast(steps=1))[0] / 0.1

# Plot sensitivity
plt.bar(range(1, n_steps+1), sensitivity[::-1])
plt.xlabel('Lag')
plt.ylabel('Sensitivity')
plt.title('Sensitivity to Past Observations')
plt.show()
```

Slide 11: Sensitivity in Anomaly Detection

Sensitivity analysis in anomaly detection helps identify which features contribute most to the detection of anomalies. This information can be used to refine the anomaly detection algorithm and focus on the most relevant features.

```python
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt

# Generate normal and anomalous data
np.random.seed(42)
X_normal = np.random.randn(1000, 2)
X_anomalies = np.random.uniform(low=-4, high=4, size=(50, 2))
X = np.vstack([X_normal, X_anomalies])

# Train Isolation Forest
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(X)

# Compute feature importance
n_samples = 1000
n_features = X.shape[1]
sensitivities = np.zeros(n_features)

for i in range(n_features):
    X_perturbed = X.()
    X_perturbed[:, i] += 0.1
    scores_original = clf.score_samples(X)
    scores_perturbed = clf.score_samples(X_perturbed)
    sensitivities[i] = np.mean(np.abs(scores_perturbed - scores_original)) / 0.1

# Plot feature importance
plt.bar(range(n_features), sensitivities)
plt.xlabel('Feature')
plt.ylabel('Sensitivity')
plt.title('Feature Importance in Anomaly Detection')
plt.show()
```

Slide 12: Real-Life Example: Predictive Maintenance

In predictive maintenance for industrial equipment, sensitivity analysis helps identify which sensor readings or operational parameters are most indicative of impending failures. This information guides maintenance schedules and sensor placement.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Generate synthetic sensor data
np.random.seed(42)
n_samples = 1000
temperature = np.random.normal(60, 10, n_samples)
vibration = np.random.normal(0.5, 0.2, n_samples)
pressure = np.random.normal(100, 20, n_samples)
failure = (temperature > 75) & (vibration > 0.7) | (pressure > 130)

data = pd.DataFrame({
    'temperature': temperature,
    'vibration': vibration,
    'pressure': pressure,
    'failure': failure
})

# Train a Random Forest classifier
X = data[['temperature', 'vibration', 'pressure']]
y = data['failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Compute feature importance
importances = clf.feature_importances_

# Plot feature importance
plt.bar(X.columns, importances)
plt.xlabel('Sensor')
plt.ylabel('Importance')
plt.title('Sensor Importance in Predictive Maintenance')
plt.show()
```

Slide 13: Real-Life Example: Environmental Impact Assessment

In environmental impact assessments, sensitivity analysis helps identify which factors have the most significant influence on ecological outcomes. This information guides conservation efforts and policy decisions.

```python
import numpy as np
import matplotlib.pyplot as plt

def ecosystem_model(params):
    r, K, a, b = params
    t = np.linspace(0, 100, 1000)
    prey = r * t * (1 - t / K) - a * t
    predator = b * a * t - 0.1 * t
    return np.mean(prey) + np.mean(predator)

baseline_params = [1.5, 1000, 0.01, 0.1]
param_names = ['Growth rate', 'Carrying capacity', 'Predation rate', 'Conversion efficiency']
sensitivities = []

for i in range(len(baseline_params)):
    perturbed_params = baseline_params.()
    perturbed_params[i] *= 1.1
    sensitivity = (ecosystem_model(perturbed_params) - ecosystem_model(baseline_params)) / (0.1 * baseline_params[i])
    sensitivities.append(sensitivity)

plt.bar(param_names, sensitivities)
plt.ylabel('Sensitivity')
plt.title('Parameter Sensitivity in Ecosystem Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

Slide 14: Challenges and Limitations of Sensitivity Analysis

While sensitivity analysis is a powerful tool, it has limitations. It may not capture complex interactions between parameters, and results can be sensitive to the choice of perturbation size. Additionally, for high-dimensional problems, computational cost can be significant.

```python
import numpy as np
import matplotlib.pyplot as plt

def complex_model(x, y):
    return np.sin(x) * np.cos(y) + np.exp(-((x-2)**2 + (y-2)**2))

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = complex_model(X, Y)

plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.contourf(X, Y, Z, levels=20)
plt.colorbar(label='Output')
plt.title('Complex Model Output')
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')

plt.subplot(122)
dx, dy = np.gradient(Z)
sensitivity = np.sqrt(dx**2 + dy**2)
plt.contourf(X, Y, sensitivity, levels=20)
plt.colorbar(label='Sensitivity')
plt.title('Sensitivity Map')
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in delving deeper into sensitivity analysis in AI and ML, here are some valuable resources:

1. "Sensitivity Analysis in Practice: A Guide to Assessing Scientific Models" by Andrea Saltelli et al. (2004)
2. "Global Sensitivity Analysis: The Primer" by Andrea Saltelli et al. (2008)
3. "An Introduction to Sensitivity Analysis" by Bertrand Iooss and Paul Lemaître (ArXiv:1404.2405)
4. "Sensitivity Analysis for Machine Learning" by Wei-Cheng Chang et al. (ArXiv:2003.01747)
5. "Explaining the Predictions of Any Classifier" by Marco Tulio Ribeiro et al. (ArXiv:1602.04938)

These resources provide in-depth coverage of sensitivity analysis techniques, their applications in various domains, and their integration with machine learning models.

