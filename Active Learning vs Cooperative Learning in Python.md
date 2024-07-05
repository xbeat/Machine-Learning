## Active Learning vs Cooperative Learning in Python
Slide 1: Introduction to Active Learning

Active learning is a machine learning approach where the algorithm actively selects the most informative data points for labeling. This method aims to reduce the amount of labeled data required for training while maintaining or improving model performance.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from modAL.models import ActiveLearner

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=2, n_classes=2, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the active learner
learner = ActiveLearner(
    estimator=SVC(kernel='rbf', gamma='scale'),
    X_training=X_train[:10],
    y_training=y_train[:10]
)

# Active learning loop
for _ in range(50):
    query_idx, query_instance = learner.query(X_train)
    learner.teach(X_train[query_idx].reshape(1, -1), y_train[query_idx].reshape(1,))
    X_train = np.delete(X_train, query_idx, axis=0)
    y_train = np.delete(y_train, query_idx)

# Evaluate the model
accuracy = learner.score(X_test, y_test)
print(f"Final model accuracy: {accuracy:.2f}")
```

Slide 2: Pool-Based Active Learning

Pool-based active learning involves selecting the most informative instances from a pool of unlabeled data. The algorithm chooses samples that, when labeled, are expected to improve the model's performance the most.

```python
from modAL.uncertainty import uncertainty_sampling

# Initialize pool of unlabeled data
X_pool = X_train[10:]
y_pool = y_train[10:]

# Pool-based active learning loop
n_queries = 50
for idx in range(n_queries):
    query_idx, query_instance = learner.query(X_pool)
    
    # Get the true label for the queried instance
    X, y = X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(1,)
    
    # Teach the model and remove the instance from the pool
    learner.teach(X, y)
    X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx)

    # Evaluate the model
    accuracy = learner.score(X_test, y_test)
    print(f"Query {idx+1} accuracy: {accuracy:.2f}")
```

Slide 3: Uncertainty Sampling in Active Learning

Uncertainty sampling is a common query strategy in active learning. It selects instances for which the model is most uncertain, typically measured by the prediction probability or entropy of the model's output.

```python
from modAL.uncertainty import entropy_sampling

# Custom uncertainty sampling strategy
def custom_uncertainty_sampling(classifier, X, n_instances=1):
    probas = classifier.predict_proba(X)
    uncertainties = entropy_sampling(probas)
    return uncertainties.argsort()[-n_instances:][::-1]

# Initialize active learner with custom uncertainty sampling
learner = ActiveLearner(
    estimator=SVC(kernel='rbf', gamma='scale', probability=True),
    query_strategy=custom_uncertainty_sampling,
    X_training=X_train[:10],
    y_training=y_train[:10]
)

# Active learning loop with custom uncertainty sampling
for _ in range(50):
    query_idx, query_instance = learner.query(X_pool)
    learner.teach(X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(1,))
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)
```

Slide 4: Introduction to Cooperative Learning

Cooperative learning is an educational approach where students work together in small groups to achieve a common goal. In the context of machine learning, it can be seen as a form of ensemble learning where multiple models collaborate to solve a problem.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Create different classifiers
clf1 = RandomForestClassifier(n_estimators=50, random_state=42)
clf2 = DecisionTreeClassifier(random_state=42)
clf3 = GaussianNB()

# Train classifiers on different subsets of data
clf1.fit(X_train[:500], y_train[:500])
clf2.fit(X_train[500:750], y_train[500:750])
clf3.fit(X_train[750:], y_train[750:])

# Make predictions
pred1 = clf1.predict(X_test)
pred2 = clf2.predict(X_test)
pred3 = clf3.predict(X_test)

# Combine predictions (simple majority voting)
final_pred = np.array([1 if (p1 + p2 + p3) > 1 else 0 for p1, p2, p3 in zip(pred1, pred2, pred3)])

# Evaluate the ensemble
accuracy = accuracy_score(y_test, final_pred)
print(f"Ensemble accuracy: {accuracy:.2f}")
```

Slide 5: Cooperative Learning: Weighted Voting

In cooperative learning, we can assign different weights to each model based on their individual performance. This approach allows us to give more importance to more accurate models in the final decision-making process.

```python
from sklearn.metrics import accuracy_score

# Calculate individual model accuracies
acc1 = accuracy_score(y_test, pred1)
acc2 = accuracy_score(y_test, pred2)
acc3 = accuracy_score(y_test, pred3)

# Normalize accuracies to use as weights
total_acc = acc1 + acc2 + acc3
w1, w2, w3 = acc1/total_acc, acc2/total_acc, acc3/total_acc

# Weighted voting
weighted_pred = np.array([1 if (w1*p1 + w2*p2 + w3*p3) > 0.5 else 0 for p1, p2, p3 in zip(pred1, pred2, pred3)])

# Evaluate the weighted ensemble
weighted_accuracy = accuracy_score(y_test, weighted_pred)
print(f"Weighted ensemble accuracy: {weighted_accuracy:.2f}")
```

Slide 6: Active Learning in Cooperative Systems

We can combine active learning with cooperative learning by using an active learning strategy to select instances for labeling, and then training multiple models on different subsets of the labeled data.

```python
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

# Initialize active learners
learner1 = ActiveLearner(estimator=RandomForestClassifier(n_estimators=50, random_state=42), query_strategy=uncertainty_sampling)
learner2 = ActiveLearner(estimator=DecisionTreeClassifier(random_state=42), query_strategy=uncertainty_sampling)
learner3 = ActiveLearner(estimator=GaussianNB(), query_strategy=uncertainty_sampling)

# Active learning loop
for _ in range(50):
    # Query instance using the first learner
    query_idx, query_instance = learner1.query(X_pool)
    
    # Teach all learners with the same instance
    X, y = X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(1,)
    learner1.teach(X, y)
    learner2.teach(X, y)
    learner3.teach(X, y)
    
    # Remove labeled instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)

# Make predictions
pred1 = learner1.predict(X_test)
pred2 = learner2.predict(X_test)
pred3 = learner3.predict(X_test)

# Combine predictions
final_pred = np.array([1 if (p1 + p2 + p3) > 1 else 0 for p1, p2, p3 in zip(pred1, pred2, pred3)])

# Evaluate the ensemble
accuracy = accuracy_score(y_test, final_pred)
print(f"Active cooperative ensemble accuracy: {accuracy:.2f}")
```

Slide 7: Query by Committee

Query by Committee (QBC) is an active learning approach that uses a committee of models to decide which instances to query. It's a form of cooperative active learning where models collaborate to improve the overall performance.

```python
from modAL.models import Committee
from modAL.disagreement import vote_entropy_sampling

# Initialize committee members
learner1 = ActiveLearner(estimator=RandomForestClassifier(n_estimators=50, random_state=42))
learner2 = ActiveLearner(estimator=DecisionTreeClassifier(random_state=42))
learner3 = ActiveLearner(estimator=GaussianNB())

# Create committee
committee = Committee(
    learner_list=[learner1, learner2, learner3],
    query_strategy=vote_entropy_sampling
)

# Initial training
initial_idx = np.random.choice(range(len(X_pool)), size=10, replace=False)
committee.teach(X_pool[initial_idx], y_pool[initial_idx])

# Remove initial instances from pool
X_pool = np.delete(X_pool, initial_idx, axis=0)
y_pool = np.delete(y_pool, initial_idx)

# Active learning loop
for _ in range(40):
    query_idx, query_instance = committee.query(X_pool)
    committee.teach(X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(1,))
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)

# Evaluate the committee
accuracy = committee.score(X_test, y_test)
print(f"Committee accuracy: {accuracy:.2f}")
```

Slide 8: Diversity in Cooperative Learning

Diversity among models is crucial in cooperative learning. By using different types of models or training them on different subsets of data, we can create a more robust ensemble that captures various aspects of the problem.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Create diverse classifiers
clf1 = RandomForestClassifier(n_estimators=50, random_state=42)
clf2 = LogisticRegression(random_state=42)
clf3 = SVC(kernel='rbf', probability=True, random_state=42)
clf4 = GaussianNB()

# Train classifiers on different data subsets
clf1.fit(X_train[:250], y_train[:250])
clf2.fit(X_train[250:500], y_train[250:500])
clf3.fit(X_train[500:750], y_train[500:750])
clf4.fit(X_train[750:], y_train[750:])

# Make predictions
pred1 = clf1.predict(X_test)
pred2 = clf2.predict(X_test)
pred3 = clf3.predict(X_test)
pred4 = clf4.predict(X_test)

# Combine predictions
final_pred = np.array([1 if (p1 + p2 + p3 + p4) > 2 else 0 for p1, p2, p3, p4 in zip(pred1, pred2, pred3, pred4)])

# Evaluate the diverse ensemble
accuracy = accuracy_score(y_test, final_pred)
print(f"Diverse ensemble accuracy: {accuracy:.2f}")
```

Slide 9: Cooperative Learning with Boosting

Boosting is a cooperative learning technique where models are trained sequentially, with each model focusing on the errors of the previous ones. This approach can be very effective in creating strong ensembles.

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Create base estimator
base_estimator = DecisionTreeClassifier(max_depth=1)

# Create AdaBoost classifier
adaboost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)

# Train AdaBoost
adaboost.fit(X_train, y_train)

# Make predictions
y_pred = adaboost.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"AdaBoost accuracy: {accuracy:.2f}")

# Visualize the importance of each weak learner
importances = adaboost.estimator_weights_
plt.bar(range(len(importances)), importances)
plt.title("Importance of Weak Learners in AdaBoost")
plt.xlabel("Weak Learner Index")
plt.ylabel("Weight")
plt.show()
```

Slide 10: Active Learning with Uncertainty Disagreement

In this approach, we combine active learning with cooperative learning by using the disagreement between models as a measure of uncertainty. This can help in selecting the most informative instances for labeling.

```python
from modAL.disagreement import ConsensusEntropy

# Initialize classifiers
clf1 = RandomForestClassifier(n_estimators=50, random_state=42)
clf2 = SVC(probability=True, random_state=42)
clf3 = GaussianNB()

# Create committee
committee = Committee([
    ActiveLearner(estimator=clf1, X_training=X_train[:10], y_training=y_train[:10]),
    ActiveLearner(estimator=clf2, X_training=X_train[:10], y_training=y_train[:10]),
    ActiveLearner(estimator=clf3, X_training=X_train[:10], y_training=y_train[:10])
])

# Define query strategy
query_strategy = ConsensusEntropy()

# Active learning loop
for _ in range(50):
    query_idx, query_instance = query_strategy(committee, X_pool)
    committee.teach(X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(1,))
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)

# Evaluate the committee
accuracy = committee.score(X_test, y_test)
print(f"Uncertainty disagreement committee accuracy: {accuracy:.2f}")
```

Slide 11: Transfer Learning in Cooperative Systems

Transfer learning allows models to share knowledge across different but related tasks. In a cooperative learning setting, we can use transfer learning to improve the performance of individual models in the ensemble.

```python
from sklearn.neural_network import MLPClassifier
import numpy as np

# Create a base model
base_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

# Train on a related task (assuming we have X_related, y_related)
X_related = np.random.rand(1000, 20)
y_related = np.random.randint(0, 2, 1000)
base_model.fit(X_related, y_related)

# Create transfer models
transfer_model1 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
transfer_model2 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=43)

# Transfer learned weights from base model to transfer models
transfer_model1.coefs_ = base_model.coefs_
transfer_model1.intercepts_ = base_model.intercepts_
transfer_model2.coefs_ = base_model.coefs_
transfer_model2.intercepts_ = base_model.intercepts_

# Fine-tune transfer models on target task
transfer_model1.partial_fit(X_train, y_train, classes=np.unique(y_train))
transfer_model2.partial_fit(X_train, y_train, classes=np.unique(y_train))

# Make predictions
pred1 = transfer_model1.predict(X_test)
pred2 = transfer_model2.predict(X_test)

# Combine predictions
final_pred = np.array([1 if (p1 + p2) > 0 else 0 for p1, p2 in zip(pred1, pred2)])

# Evaluate the ensemble
accuracy = accuracy_score(y_test, final_pred)
print(f"Transfer learning ensemble accuracy: {accuracy:.2f}")
```

Slide 12: Multi-Task Learning in Cooperative Systems

Multi-task learning is an approach where a model learns to perform multiple related tasks simultaneously. In a cooperative learning context, we can use multi-task learning to create more versatile models that can contribute to multiple aspects of the problem.

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Create synthetic multi-task data
X_multi = np.random.rand(1000, 20)
y_multi = np.random.randint(0, 2, (1000, 3))  # 3 related tasks

# Create and train multi-task model
multi_task_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
multi_task_model.fit(X_multi, y_multi)

# Use the multi-task model in a cooperative ensemble
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
base_model.fit(X_train, y_train)

# Make predictions
pred_base = base_model.predict(X_test)
pred_multi = multi_task_model.predict(X_test)[:, 0]  # Use predictions from the first task

# Combine predictions
final_pred = np.array([1 if (p1 + p2) > 0 else 0 for p1, p2 in zip(pred_base, pred_multi)])

# Evaluate the ensemble
accuracy = accuracy_score(y_test, final_pred)
print(f"Multi-task cooperative ensemble accuracy: {accuracy:.2f}")
```

Slide 13: Active Learning with Model Uncertainty

This approach combines active learning with model uncertainty estimation. We use dropout in neural networks as a way to estimate model uncertainty, which guides the active learning process in a cooperative setting.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Create a model with dropout for uncertainty estimation
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(20,)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to get model uncertainty
def get_uncertainty(model, X, num_samples=10):
    predictions = np.array([model(X, training=True) for _ in range(num_samples)])
    return np.std(predictions, axis=0)

# Active learning loop
model = create_model()
X_labeled, y_labeled = X_train[:10], y_train[:10]
X_pool, y_pool = X_train[10:], y_train[10:]

for _ in range(50):
    model.fit(X_labeled, y_labeled, epochs=10, verbose=0)
    uncertainties = get_uncertainty(model, X_pool)
    query_idx = np.argmax(uncertainties)
    
    X_labeled = np.vstack([X_labeled, X_pool[query_idx]])
    y_labeled = np.append(y_labeled, y_pool[query_idx])
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print(f"Active learning with model uncertainty accuracy: {accuracy:.2f}")
```

Slide 14: Cooperative Learning with Federated Learning

Federated learning is a technique where multiple parties collaboratively train a model without sharing their raw data. This approach can be seen as a form of cooperative learning where privacy is a key concern.

```python
import numpy as np
from sklearn.linear_model import SGDClassifier

def federated_average(models):
    avg_coef = np.mean([model.coef_ for model in models], axis=0)
    avg_intercept = np.mean([model.intercept_ for model in models], axis=0)
    return avg_coef, avg_intercept

# Simulate multiple parties
num_parties = 3
party_data = np.array_split(X_train, num_parties)
party_labels = np.array_split(y_train, num_parties)

# Initialize local models
local_models = [SGDClassifier(loss='log', random_state=42) for _ in range(num_parties)]

# Federated learning loop
for _ range(10):  # 10 rounds of federated learning
    # Local training
    for i, model in enumerate(local_models):
        model.partial_fit(party_data[i], party_labels[i], classes=np.unique(y_train))
    
    # Federated averaging
    avg_coef, avg_intercept = federated_average(local_models)
    
    # Update local models
    for model in local_models:
        model.coef_ = avg_coef
        model.intercept_ = avg_intercept

# Evaluate the federated model
federated_model = SGDClassifier(loss='log', random_state=42)
federated_model.coef_ = avg_coef
federated_model.intercept_ = avg_intercept

accuracy = federated_model.score(X_test, y_test)
print(f"Federated learning accuracy: {accuracy:.2f}")
```

Slide 15: Conclusion and Future Directions

Active learning and cooperative learning are powerful techniques that can significantly improve model performance and data efficiency. Future research directions include combining these approaches with advanced deep learning models, exploring new query strategies for active learning, and developing more efficient federated learning algorithms for large-scale cooperative systems.

```python
# Pseudocode for a future advanced active cooperative learning system

class AdvancedActiveCooperativeLearner:
    def __init__(self, models, query_strategy, aggregation_method):
        self.models = models
        self.query_strategy = query_strategy
        self.aggregation_method = aggregation_method

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        while not stopping_criterion:
            # Train individual models
            for model in self.models:
                model.fit(X_labeled, y_labeled)

            # Select most informative instance
            query_idx = self.query_strategy(self.models, X_unlabeled)

            # Get true label (simulated here)
            X_new, y_new = get_label(X_unlabeled[query_idx])

            # Update labeled and unlabeled datasets
            X_labeled = np.vstack([X_labeled, X_new])
            y_labeled = np.append(y_labeled, y_new)
            X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)

        # Final aggregation of models
        return self.aggregation_method(self.models)

# Usage
learner = AdvancedActiveCooperativeLearner(
    models=[DeepNeuralNetwork(), RandomForest(), SVM()],
    query_strategy=entropy_disagreement_sampling,
    aggregation_method=weighted_average
)

final_model = learner.fit(X_train, y_train, X_unlabeled)
```

Slide 16: Additional Resources

For those interested in diving deeper into active learning and cooperative learning, here are some valuable resources:

1. "Active Learning" by Burr Settles (2012) - Comprehensive overview of active learning techniques. ArXiv: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)
2. "A Survey of Deep Active Learning" by Ren et al. (2020) - Recent advancements in active learning with deep neural networks. ArXiv: [https://arxiv.org/abs/2009.00236](https://arxiv.org/abs/2009.00236)
3. "Cooperative Machine Learning" by Pang et al. (2018) - Overview of cooperative learning approaches in machine learning. ArXiv: [https://arxiv.org/abs/1808.04736](https://arxiv.org/abs/1808.04736)
4. "Federated Learning: Challenges, Methods, and Future Directions" by Yang et al. (2019) - Comprehensive survey on federated learning. ArXiv: [https://arxiv.org/abs/1908.07873](https://arxiv.org/abs/1908.07873)

These resources provide in-depth discussions and the latest research in active and cooperative learning, offering valuable insights for both beginners and advanced practitioners.

