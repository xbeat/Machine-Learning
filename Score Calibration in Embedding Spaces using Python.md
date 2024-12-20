## Score Calibration in Embedding Spaces using Python
Slide 1: Introduction to Score Calibration in Embedding Spaces

Score calibration in embedding spaces is a crucial technique for improving the reliability and interpretability of machine learning models. It involves adjusting the raw scores or probabilities produced by a model to better reflect the true likelihood of predictions. This process is particularly important when working with high-dimensional embedding spaces, where distances and similarities may not directly correspond to meaningful probabilities.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample embeddings
np.random.seed(42)
embeddings = np.random.randn(100, 2)

# Plot embeddings
plt.figure(figsize=(8, 6))
plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6)
plt.title("Sample Embeddings in 2D Space")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
```

Slide 2: Understanding Raw Scores in Embedding Spaces

Raw scores in embedding spaces often represent distances or similarities between points. These scores may not directly translate to probabilities or confidence levels. For example, in a nearest neighbor model, the raw score might be the Euclidean distance between two points in the embedding space.

```python
from sklearn.neighbors import NearestNeighbors

# Create a nearest neighbors model
nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn_model.fit(embeddings)

# Query point
query_point = np.array([[0, 0]])

# Find nearest neighbor and distance
distances, indices = nn_model.kneighbors(query_point)

print(f"Nearest neighbor index: {indices[0][0]}")
print(f"Raw distance score: {distances[0][0]}")
```

Slide 3: The Need for Score Calibration

Raw scores may not accurately represent the true likelihood or confidence of predictions. Calibration helps to address several issues:

1. Different scales across different models or embedding spaces
2. Non-linear relationships between distances and probabilities
3. Overconfidence or underconfidence in model predictions

```python
import scipy.stats as stats

# Generate some example scores
raw_scores = np.random.uniform(0, 10, 1000)

# Plot histogram of raw scores
plt.figure(figsize=(10, 6))
plt.hist(raw_scores, bins=30, edgecolor='black')
plt.title("Distribution of Raw Scores")
plt.xlabel("Raw Score")
plt.ylabel("Frequency")
plt.show()

# Calculate mean and standard deviation
mean = np.mean(raw_scores)
std = np.std(raw_scores)

# Plot normal distribution
x = np.linspace(0, 10, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, stats.norm.pdf(x, mean, std))
plt.title("Ideal Calibrated Score Distribution")
plt.xlabel("Calibrated Score")
plt.ylabel("Probability Density")
plt.show()
```

Slide 4: Isotonic Regression for Score Calibration

Isotonic regression is a non-parametric approach to score calibration. It learns a monotonic function that maps raw scores to calibrated probabilities while preserving the ranking order of the original scores.

```python
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

# Generate synthetic data
np.random.seed(42)
raw_scores = np.random.uniform(0, 1, 1000)
true_labels = (raw_scores + np.random.normal(0, 0.1, 1000) > 0.5).astype(int)

# Fit isotonic regression
iso_reg = IsotonicRegression(out_of_bounds='clip')
calibrated_scores = iso_reg.fit_transform(raw_scores, true_labels)

# Compare Brier scores
print(f"Brier score (raw): {brier_score_loss(true_labels, raw_scores):.4f}")
print(f"Brier score (calibrated): {brier_score_loss(true_labels, calibrated_scores):.4f}")

# Plot calibration curves
plt.figure(figsize=(10, 6))
plt.scatter(raw_scores, true_labels, alpha=0.1, label='Raw scores')
plt.plot(raw_scores, calibrated_scores, color='red', label='Calibrated scores')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
plt.title("Isotonic Regression Calibration")
plt.xlabel("Raw Score")
plt.ylabel("Calibrated Probability")
plt.legend()
plt.show()
```

Slide 5: Platt Scaling for Score Calibration

Platt scaling is a parametric method for score calibration that uses logistic regression to map raw scores to probabilities. It's particularly effective for SVM outputs but can be applied to other models as well.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(42)
raw_scores = np.random.normal(0, 1, 1000)
true_labels = (raw_scores + np.random.normal(0, 0.5, 1000) > 0).astype(int)

# Standardize scores
scaler = StandardScaler()
scaled_scores = scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()

# Fit Platt scaling
platt_model = LogisticRegression(random_state=42)
platt_model.fit(scaled_scores.reshape(-1, 1), true_labels)

# Calibrate scores
calibrated_scores = platt_model.predict_proba(scaled_scores.reshape(-1, 1))[:, 1]

# Plot calibration curves
plt.figure(figsize=(10, 6))
plt.scatter(raw_scores, true_labels, alpha=0.1, label='Raw scores')
plt.scatter(raw_scores, calibrated_scores, alpha=0.1, color='red', label='Calibrated scores')
plt.plot([min(raw_scores), max(raw_scores)], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
plt.title("Platt Scaling Calibration")
plt.xlabel("Raw Score")
plt.ylabel("Calibrated Probability")
plt.legend()
plt.show()
```

Slide 6: Temperature Scaling for Neural Networks

Temperature scaling is a simple yet effective method for calibrating neural network outputs. It involves dividing the logits (pre-softmax outputs) by a learned temperature parameter.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

# Generate synthetic logits and labels
np.random.seed(42)
logits = torch.randn(1000, 10)
labels = torch.randint(0, 10, (1000,))

# Create and train temperature scaling model
temperature_model = TemperatureScaling()
optimizer = optim.LBFGS(temperature_model.parameters(), lr=0.01, max_iter=50)
criterion = nn.CrossEntropyLoss()

def eval():
    optimizer.zero_grad()
    loss = criterion(temperature_model(logits), labels)
    loss.backward()
    return loss

optimizer.step(eval)

print(f"Learned temperature: {temperature_model.temperature.item():.4f}")

# Apply temperature scaling
calibrated_logits = temperature_model(logits)
calibrated_probs = torch.softmax(calibrated_logits, dim=1)

print("Raw probabilities (first 5 samples):")
print(torch.softmax(logits, dim=1)[:5])
print("\nCalibrated probabilities (first 5 samples):")
print(calibrated_probs[:5])
```

Slide 7: Evaluating Calibration: Reliability Diagrams

Reliability diagrams are a visual tool for assessing the calibration of a model. They plot the observed frequency of positive outcomes against the predicted probabilities.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Generate synthetic data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_pred_uncalibrated = np.random.beta(2, 5, 1000)
y_pred_calibrated = np.random.beta(5, 5, 1000)

# Compute calibration curves
fraction_of_positives_uncal, mean_predicted_value_uncal = calibration_curve(y_true, y_pred_uncalibrated, n_bins=10)
fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(y_true, y_pred_calibrated, n_bins=10)

# Plot reliability diagram
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.plot(mean_predicted_value_uncal, fraction_of_positives_uncal, marker='o', label='Uncalibrated')
plt.plot(mean_predicted_value_cal, fraction_of_positives_cal, marker='o', label='Calibrated')
plt.title('Reliability Diagram')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.legend()
plt.show()
```

Slide 8: Evaluating Calibration: Expected Calibration Error (ECE)

Expected Calibration Error (ECE) is a scalar metric that quantifies the difference between predicted probabilities and observed frequencies. A lower ECE indicates better calibration.

```python
def compute_ece(y_true, y_pred, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(y_pred > bin_lower, y_pred <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == (y_pred[in_bin] > 0.5))
            avg_confidence_in_bin = np.mean(y_pred[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    return ece

# Generate synthetic data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_pred_uncalibrated = np.random.beta(2, 5, 1000)
y_pred_calibrated = np.random.beta(5, 5, 1000)

# Compute ECE
ece_uncalibrated = compute_ece(y_true, y_pred_uncalibrated)
ece_calibrated = compute_ece(y_true, y_pred_calibrated)

print(f"ECE (Uncalibrated): {ece_uncalibrated:.4f}")
print(f"ECE (Calibrated): {ece_calibrated:.4f}")
```

Slide 9: Real-life Example: Sentiment Analysis Calibration

In sentiment analysis, raw model outputs may not accurately represent the true probability of a positive or negative sentiment. Calibration can improve the reliability of these predictions.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV

# Sample dataset
texts = [
    "I love this product!", "Terrible experience, never again.",
    "Not bad, but could be better.", "Absolutely amazing service!",
    "Disappointing quality.", "Neutral opinion.", "Highly recommended!"
]
sentiments = [1, 0, 0, 1, 0, 0, 1]  # 1 for positive, 0 for negative/neutral

# Vectorize text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, sentiments)

# Calibrate the classifier
calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
calibrated_clf.fit(X, sentiments)

# Make predictions
new_texts = ["This is great!", "I'm not sure about this."]
new_X = vectorizer.transform(new_texts)

print("Uncalibrated probabilities:")
print(clf.predict_proba(new_X)[:, 1])

print("\nCalibrated probabilities:")
print(calibrated_clf.predict_proba(new_X)[:, 1])
```

Slide 10: Real-life Example: Image Classification Confidence

In image classification tasks, deep learning models often produce overconfident predictions. Calibration can help align the model's confidence with its actual accuracy.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess an example image (replace with your own image path)
image_path = "example_image.jpg"
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0)

# Make a prediction
with torch.no_grad():
    output = model(input_tensor)

# Get the predicted class and confidence
_, predicted_idx = torch.max(output, 1)
confidence = torch.softmax(output, dim=1)[0, predicted_idx].item()

print(f"Predicted class index: {predicted_idx.item()}")
print(f"Uncalibrated confidence: {confidence:.4f}")

# Apply temperature scaling (assuming we've learned a temperature parameter)
temperature = 2.0  # This should be learned on a validation set
calibrated_output = output / temperature
calibrated_confidence = torch.softmax(calibrated_output, dim=1)[0, predicted_idx].item()

print(f"Calibrated confidence: {calibrated_confidence:.4f}")
```

Slide 11: Challenges in Score Calibration

Score calibration in embedding spaces faces several challenges:

1. High-dimensionality: Embedding spaces often have hundreds or thousands of dimensions, making it difficult to visualize and understand the relationship between distances and probabilities.
2. Non-linearity: The mapping between raw scores and probabilities may be highly non-linear, requiring sophisticated calibration techniques.
3. Domain shift: Calibration learned on one dataset may not generalize well to new domains or data distributions.
4. Computational complexity: Some calibration methods can be computationally expensive, especially for large-scale problems.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Generate high-dimensional embeddings
np.random.seed(42)
embeddings = np.random.randn(1000, 100)

# Reduce dimensionality for visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot reduced embeddings
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
plt.title("t-SNE Visualization of High-Dimensional Embeddings")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.colorbar()
plt.show()
```

Slide 12: Ensemble Methods for Robust Calibration

Ensemble methods can improve the robustness and accuracy of score calibration by combining multiple calibration techniques or models.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = np.random.randn(1000, 10), np.random.randint(0, 2, 1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create base classifier
base_clf = RandomForestClassifier(n_estimators=10, random_state=42)

# Create ensemble of calibrated classifiers
calibrated_ensemble = []
for i in range(5):
    clf = CalibratedClassifierCV(base_clf, cv=5, method='sigmoid')
    clf.fit(X_train, y_train)
    calibrated_ensemble.append(clf)

# Make predictions with the ensemble
ensemble_probs = np.mean([clf.predict_proba(X_test)[:, 1] for clf in calibrated_ensemble], axis=0)

print("Ensemble calibrated probabilities (first 5 samples):")
print(ensemble_probs[:5])
```

Slide 13: Calibration in Multi-class Problems

Calibrating scores for multi-class problems requires special consideration, as the probabilities must sum to 1 across all classes.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM classifier
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)

# Calibrate the classifier
calibrated_svm = CalibratedClassifierCV(svm, method='isotonic', cv=5)
calibrated_svm.fit(X_train, y_train)

# Make predictions
svm_probs = svm.decision_function(X_test)
calibrated_probs = calibrated_svm.predict_proba(X_test)

# Compare log loss
svm_loss = log_loss(y_test, svm_probs)
calibrated_loss = log_loss(y_test, calibrated_probs)

print(f"SVM Log Loss: {svm_loss:.4f}")
print(f"Calibrated SVM Log Loss: {calibrated_loss:.4f}")
```

Slide 14: Calibration in Online Learning Scenarios

In online learning scenarios, where data arrives sequentially, calibration methods need to be adapted to handle streaming data and concept drift.

```python
import numpy as np
from scipy.special import expit

class OnlineLogisticCalibration:
    def __init__(self, learning_rate=0.01):
        self.weights = np.array([1.0, 0.0])
        self.learning_rate = learning_rate

    def calibrate(self, score):
        return expit(self.weights[0] * score + self.weights[1])

    def update(self, score, true_label):
        pred = self.calibrate(score)
        error = true_label - pred
        self.weights[0] += self.learning_rate * error * score
        self.weights[1] += self.learning_rate * error

# Simulate online learning
np.random.seed(42)
scores = np.random.randn(1000)
true_labels = (scores + np.random.randn(1000) > 0).astype(int)

calibrator = OnlineLogisticCalibration()

for score, label in zip(scores, true_labels):
    calibrated_score = calibrator.calibrate(score)
    calibrator.update(score, label)

print("Final calibration weights:")
print(f"Slope: {calibrator.weights[0]:.4f}")
print(f"Intercept: {calibrator.weights[1]:.4f}")
```

Slide 15: Additional Resources

For those interested in diving deeper into score calibration in embedding spaces, here are some valuable resources:

1. "On Calibration of Modern Neural Networks" by Guo et al. (2017) ArXiv: [https://arxiv.org/abs/1706.04599](https://arxiv.org/abs/1706.04599)
2. "Calibration of Deep Probabilistic Models with Marginal Posterior Confidence Intervals" by Kuleshov et al. (2018) ArXiv: [https://arxiv.org/abs/1802.03569](https://arxiv.org/abs/1802.03569)
3. "Accurate Uncertainties for Deep Learning Using Calibrated Regression" by Kuleshov et al. (2018) ArXiv: [https://arxiv.org/abs/1807.00263](https://arxiv.org/abs/1807.00263)
4. "Temperature Scaling: A Simple Bayesian Approach for Improving Probability Calibration" by Phan et al. (2020) ArXiv: [https://arxiv.org/abs/2009.12678](https://arxiv.org/abs/2009.12678)

These papers provide in-depth discussions on various calibration techniques, their theoretical foundations, and empirical evaluations in different contexts.

