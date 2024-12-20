## Active Learning in Machine Learning with Python
Slide 1: Introduction to Active Learning in ML

Active Learning is a machine learning paradigm where the algorithm can interactively query a user or other information source to obtain the desired outputs at new data points. This approach is particularly useful when labeled data is scarce or expensive to obtain. By strategically selecting the most informative instances for labeling, active learning aims to achieve high accuracy with a minimal amount of labeled data.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=2, n_classes=2, random_state=42)

# Split the data into labeled and unlabeled sets
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size=0.8, random_state=42)

print(f"Labeled samples: {X_labeled.shape[0]}")
print(f"Unlabeled samples: {X_unlabeled.shape[0]}")
```

Slide 2: Uncertainty Sampling

Uncertainty Sampling is one of the simplest and most commonly used active learning strategies. It selects instances for which the current model is most uncertain about their labels. This approach is based on the idea that the most informative instances are those that the model is least certain about.

```python
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy

def uncertainty_sampling(model, X_unlabeled, n_instances=10):
    # Get probabilities for each class
    probas = model.predict_proba(X_unlabeled)
    
    # Calculate entropy for each instance
    entropies = entropy(probas.T)
    
    # Select instances with highest entropy
    selected_indices = np.argsort(entropies)[-n_instances:]
    
    return selected_indices

# Train initial model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_labeled, y_labeled)

# Select instances using uncertainty sampling
selected_indices = uncertainty_sampling(model, X_unlabeled)

print(f"Selected indices: {selected_indices}")
```

Slide 3: Query by Committee

Query by Committee (QBC) is an active learning approach that maintains a committee of models and selects instances where the committee members disagree the most. This strategy is effective in exploring different regions of the hypothesis space.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def query_by_committee(committee, X_unlabeled, n_instances=10):
    # Get predictions from each committee member
    predictions = np.array([model.predict(X_unlabeled) for model in committee])
    
    # Calculate disagreement (variance) for each instance
    disagreement = np.var(predictions, axis=0)
    
    # Select instances with highest disagreement
    selected_indices = np.argsort(disagreement)[-n_instances:]
    
    return selected_indices

# Create a committee of models
committee = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    SVC(probability=True, random_state=42)
]

# Train each model in the committee
for model in committee:
    model.fit(X_labeled, y_labeled)

# Select instances using QBC
selected_indices = query_by_committee(committee, X_unlabeled)

print(f"Selected indices: {selected_indices}")
```

Slide 4: Expected Model Change

Expected Model Change is an active learning strategy that selects instances that would cause the greatest change in the model if we knew their labels. This approach aims to select instances that have the most impact on the model's parameters.

```python
from sklearn.linear_model import LogisticRegression

def expected_model_change(model, X_unlabeled, n_instances=10):
    probas = model.predict_proba(X_unlabeled)
    
    # Calculate expected gradient length
    expected_change = np.sum(probas * np.abs(1 - probas), axis=1)
    
    # Select instances with highest expected change
    selected_indices = np.argsort(expected_change)[-n_instances:]
    
    return selected_indices

# Train initial model
model = LogisticRegression(random_state=42)
model.fit(X_labeled, y_labeled)

# Select instances using Expected Model Change
selected_indices = expected_model_change(model, X_unlabeled)

print(f"Selected indices: {selected_indices}")
```

Slide 5: Density-Weighted Methods

Density-Weighted Methods combine informativeness and representativeness criteria. They aim to select instances that are not only informative but also representative of the underlying data distribution. This approach helps to avoid selecting outliers or rare instances.

```python
from sklearn.neighbors import NearestNeighbors

def density_weighted_sampling(model, X_unlabeled, n_instances=10, k=5):
    # Calculate uncertainty
    probas = model.predict_proba(X_unlabeled)
    uncertainties = 1 - np.max(probas, axis=1)
    
    # Calculate density
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_unlabeled)
    densities = 1 / nn.kneighbors(X_unlabeled)[0].sum(axis=1)
    
    # Combine uncertainty and density
    scores = uncertainties * densities
    
    # Select instances with highest scores
    selected_indices = np.argsort(scores)[-n_instances:]
    
    return selected_indices

# Train initial model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_labeled, y_labeled)

# Select instances using Density-Weighted Sampling
selected_indices = density_weighted_sampling(model, X_unlabeled)

print(f"Selected indices: {selected_indices}")
```

Slide 6: Active Learning Loop

The Active Learning Loop is the iterative process of selecting instances, querying their labels, and updating the model. This cycle continues until a stopping criterion is met, such as reaching a desired performance level or exhausting the labeling budget.

```python
from sklearn.metrics import accuracy_score

def active_learning_loop(X_labeled, y_labeled, X_unlabeled, y_unlabeled, n_iterations=5, n_instances=10):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    for i in range(n_iterations):
        # Train model
        model.fit(X_labeled, y_labeled)
        
        # Evaluate model
        accuracy = accuracy_score(y_unlabeled, model.predict(X_unlabeled))
        print(f"Iteration {i+1}, Accuracy: {accuracy:.4f}")
        
        # Select instances
        selected_indices = uncertainty_sampling(model, X_unlabeled, n_instances)
        
        # Update datasets
        X_labeled = np.vstack((X_labeled, X_unlabeled[selected_indices]))
        y_labeled = np.concatenate((y_labeled, y_unlabeled[selected_indices]))
        X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
        y_unlabeled = np.delete(y_unlabeled, selected_indices)
    
    return model, X_labeled, y_labeled

# Run active learning loop
final_model, final_X_labeled, final_y_labeled = active_learning_loop(X_labeled, y_labeled, X_unlabeled, y_unlabeled)

print(f"Final labeled set size: {final_X_labeled.shape[0]}")
```

Slide 7: Batch Mode Active Learning

Batch Mode Active Learning selects multiple instances at once for labeling. This approach is particularly useful when it's more efficient to label multiple instances in a batch rather than one at a time. The challenge is to select a diverse and informative batch of instances.

```python
from sklearn.cluster import KMeans

def batch_mode_sampling(model, X_unlabeled, n_instances=10):
    # Get uncertainties
    probas = model.predict_proba(X_unlabeled)
    uncertainties = 1 - np.max(probas, axis=1)
    
    # Select top-k uncertain instances
    top_k = min(n_instances * 10, len(X_unlabeled))
    top_indices = np.argsort(uncertainties)[-top_k:]
    
    # Cluster the top-k instances
    kmeans = KMeans(n_clusters=n_instances, random_state=42)
    kmeans.fit(X_unlabeled[top_indices])
    
    # Select the instance closest to each cluster center
    selected_indices = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(X_unlabeled[top_indices] - center, axis=1)
        selected_indices.append(top_indices[np.argmin(distances)])
    
    return selected_indices

# Train initial model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_labeled, y_labeled)

# Select instances using Batch Mode Sampling
selected_indices = batch_mode_sampling(model, X_unlabeled)

print(f"Selected indices: {selected_indices}")
```

Slide 8: Active Learning for Imbalanced Datasets

When dealing with imbalanced datasets, active learning can be particularly useful. By focusing on the minority class or underrepresented regions of the feature space, we can improve model performance on rare events or underrepresented classes.

```python
from imblearn.over_sampling import SMOTE

def active_learning_imbalanced(X_labeled, y_labeled, X_unlabeled, n_instances=10):
    # Apply SMOTE to balance the labeled set
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_labeled, y_labeled)
    
    # Train model on balanced data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_balanced, y_balanced)
    
    # Select instances using uncertainty sampling
    selected_indices = uncertainty_sampling(model, X_unlabeled, n_instances)
    
    return selected_indices

# Create an imbalanced dataset
X_imb, y_imb = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_imb, y_imb, test_size=0.8, random_state=42)

# Select instances using active learning for imbalanced data
selected_indices = active_learning_imbalanced(X_labeled, y_labeled, X_unlabeled)

print(f"Selected indices: {selected_indices}")
print(f"Class distribution in labeled set: {np.bincount(y_labeled)}")
```

Slide 9: Active Learning for Multi-Label Classification

Active Learning can be extended to multi-label classification problems, where each instance can belong to multiple classes simultaneously. The challenge is to select instances that are informative across multiple labels.

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import hamming_loss

def multi_label_uncertainty_sampling(model, X_unlabeled, n_instances=10):
    # Get probabilities for each label
    probas = model.predict_proba(X_unlabeled)
    
    # Calculate uncertainty as the sum of entropies across all labels
    uncertainties = sum(entropy(p.T) for p in probas)
    
    # Select instances with highest uncertainty
    selected_indices = np.argsort(uncertainties)[-n_instances:]
    
    return selected_indices

# Generate multi-label dataset
X_multi, y_multi = make_classification(n_samples=1000, n_classes=3, n_informative=4, 
                                       n_features=20, n_repeated=2, n_labels=2, 
                                       random_state=42)
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_multi, y_multi, test_size=0.8, random_state=42)

# Train multi-label model
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
model = MultiOutputClassifier(base_model)
model.fit(X_labeled, y_labeled)

# Select instances using multi-label uncertainty sampling
selected_indices = multi_label_uncertainty_sampling(model, X_unlabeled)

print(f"Selected indices: {selected_indices}")
print(f"Hamming loss: {hamming_loss(y_unlabeled, model.predict(X_unlabeled)):.4f}")
```

Slide 10: Active Learning for Regression

Active Learning can also be applied to regression problems. Instead of uncertainty, we can use metrics like expected model change or prediction variance to select informative instances.

```python
from sklearn.ensemble import RandomForestRegressor

def regression_active_learning(model, X_unlabeled, n_instances=10):
    # Get predictions and their variance
    predictions = np.array([tree.predict(X_unlabeled) for tree in model.estimators_])
    variance = np.var(predictions, axis=0)
    
    # Select instances with highest variance
    selected_indices = np.argsort(variance)[-n_instances:]
    
    return selected_indices

# Generate regression dataset
X_reg, y_reg = make_classification(n_samples=1000, n_features=20, n_informative=5, n_targets=1, noise=0.1, random_state=42)
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_reg, y_reg, test_size=0.8, random_state=42)

# Train regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_labeled, y_labeled)

# Select instances using regression active learning
selected_indices = regression_active_learning(model, X_unlabeled)

print(f"Selected indices: {selected_indices}")
print(f"Mean squared error: {np.mean((y_unlabeled - model.predict(X_unlabeled))**2):.4f}")
```

Slide 11: Real-Life Example: Image Classification

Active Learning is particularly useful in image classification tasks where labeling large datasets can be time-consuming and expensive. Here's a simplified example using a small subset of the MNIST dataset.

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load a subset of the MNIST dataset
digits = load_digits(n_class=10, return_X_y=True)
X, y = digits[0][:1000], digits[1][:1000]

# Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size=0.9, random_state=42)

def active_learning_loop_svm(X_labeled, y_labeled, X_unlabeled, y_unlabeled, n_iterations=5, n_instances=10):
    model = SVC(probability=True, random_state=42)
    
    for i in range(n_iterations):
        model.fit(X_labeled, y_labeled)
        accuracy = accuracy_score(y_unlabeled, model.predict(X_unlabeled))
        print(f"Iteration {i+1}, Accuracy: {accuracy:.4f}")
        
        selected_indices = uncertainty_sampling(model, X_unlabeled, n_instances)
        
        X_labeled = np.vstack((X_labeled, X_unlabeled[selected_indices]))
        y_labeled = np.concatenate((y_labeled, y_unlabeled[selected_indices]))
        X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
        y_unlabeled = np.delete(y_unlabeled, selected_indices)
    
    return model, X_labeled, y_labeled

# Run active learning loop
final_model, final_X_labeled, final_y_labeled = active_learning_loop_svm(X_labeled, y_labeled, X_unlabeled, y_unlabeled)

print(f"Final labeled set size: {final_X_labeled.shape[0]}")
print(f"Final accuracy: {accuracy_score(y, final_model.predict(X)):.4f}")
```

Slide 12: Real-Life Example: Text Classification

Active Learning is also valuable in text classification tasks, such as spam detection or sentiment analysis. Here's a simplified example using a small dataset of movie reviews.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups

# Load a subset of the 20 newsgroups dataset
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# Preprocess the data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

# Split the data
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size=0.8, random_state=42)

def active_learning_loop_nb(X_labeled, y_labeled, X_unlabeled, y_unlabeled, n_iterations=5, n_instances=10):
    model = MultinomialNB()
    
    for i in range(n_iterations):
        model.fit(X_labeled, y_labeled)
        accuracy = accuracy_score(y_unlabeled, model.predict(X_unlabeled))
        print(f"Iteration {i+1}, Accuracy: {accuracy:.4f}")
        
        selected_indices = uncertainty_sampling(model, X_unlabeled, n_instances)
        
        X_labeled = vstack((X_labeled, X_unlabeled[selected_indices]))
        y_labeled = np.concatenate((y_labeled, y_unlabeled[selected_indices]))
        X_unlabeled = X_unlabeled[~X_unlabeled.index.isin(selected_indices)]
        y_unlabeled = np.delete(y_unlabeled, selected_indices)
    
    return model, X_labeled, y_labeled

# Run active learning loop
final_model, final_X_labeled, final_y_labeled = active_learning_loop_nb(X_labeled, y_labeled, X_unlabeled, y_unlabeled)

print(f"Final labeled set size: {final_X_labeled.shape[0]}")
print(f"Final accuracy: {accuracy_score(y, final_model.predict(X)):.4f}")
```

Slide 13: Challenges and Considerations in Active Learning

While Active Learning can be highly effective, there are several challenges and considerations to keep in mind:

1. Cold Start Problem: Initial model performance may be poor due to limited labeled data.
2. Sampling Bias: The active learning process may introduce bias in the labeled dataset.
3. Stopping Criteria: Determining when to stop the active learning process can be challenging.
4. Computational Cost: Some query strategies can be computationally expensive.
5. Concept Drift: The underlying data distribution may change over time, affecting model performance.

To address these challenges, researchers have proposed various techniques:

```python
def cold_start_strategy(X_unlabeled, n_instances=10):
    # Use clustering to select diverse initial instances
    kmeans = KMeans(n_clusters=n_instances, random_state=42)
    kmeans.fit(X_unlabeled)
    distances = kmeans.transform(X_unlabeled)
    selected_indices = [np.argmin(distances[:, i]) for i in range(n_instances)]
    return selected_indices

def detect_concept_drift(model, X_new, y_new, threshold=0.1):
    # Monitor model performance on new data
    accuracy_new = accuracy_score(y_new, model.predict(X_new))
    if accuracy_new < model.best_accuracy - threshold:
        print("Concept drift detected. Retraining model...")
        model.fit(X_new, y_new)
        model.best_accuracy = accuracy_new
    return model

# Example usage
initial_indices = cold_start_strategy(X_unlabeled)
X_initial = X_unlabeled[initial_indices]
y_initial = oracle_labeling(X_initial)  # Assume we have an oracle for labeling

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_initial, y_initial)
model.best_accuracy = accuracy_score(y_initial, model.predict(X_initial))

# Simulating new data arrival
X_new, y_new = generate_new_data()  # Assume we have a function to generate new data
model = detect_concept_drift(model, X_new, y_new)
```

Slide 14: Future Directions in Active Learning

Active Learning continues to evolve, with several promising research directions:

1. Deep Active Learning: Combining active learning with deep neural networks for complex tasks.
2. Transfer Active Learning: Leveraging knowledge from related tasks to improve sample selection.
3. Active Learning in Reinforcement Learning: Selecting informative experiences in RL settings.
4. Human-in-the-Loop Active Learning: Incorporating human expertise more effectively in the learning process.
5. Active Learning for Fairness: Ensuring that the active learning process promotes fairness and reduces bias.

Here's a conceptual example of deep active learning:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DeepActivelearner:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
    
    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
    
    def predict_proba(self, X):
        with torch.no_grad():
            return nn.Softmax(dim=1)(self.model(X))
    
    def uncertainty_sampling(self, X_unlabeled, n_instances=10):
        probas = self.predict_proba(X_unlabeled)
        uncertainties = -torch.sum(probas * torch.log(probas), dim=1)
        return torch.argsort(uncertainties, descending=True)[:n_instances]

# Usage would be similar to previous examples, but with PyTorch tensors
```

Slide 15: Additional Resources

For those interested in diving deeper into Active Learning, here are some valuable resources:

1. "Active Learning" by Burr Settles (2012) - A comprehensive survey of active learning methods. ArXiv: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)
2. "Active Learning Literature Survey" by Burr Settles (2009) - An earlier, but still relevant survey. Available at: [http://burrsettles.com/pub/settles.activelearning.pdf](http://burrsettles.com/pub/settles.activelearning.pdf)
3. "Deep Active Learning for Named Entity Recognition" by Shen et al. (2017) - Explores active learning in the context of deep learning for NLP. ArXiv: [https://arxiv.org/abs/1707.05928](https://arxiv.org/abs/1707.05928)
4. "A Survey of Deep Active Learning" by Ren et al. (2020) - A more recent survey focusing on deep active learning. ArXiv: [https://arxiv.org/abs/2009.00236](https://arxiv.org/abs/2009.00236)
5. "Active Learning for Convolutional Neural Networks: A Core-Set Approach" by Sener and Savarese (2018) - Presents a geometric approach to active learning for CNNs. ArXiv: [https://arxiv.org/abs/1708.00489](https://arxiv.org/abs/1708.00489)

These resources provide a mix of foundational knowledge and cutting-edge research in active learning, suitable for both beginners and advanced practitioners.

