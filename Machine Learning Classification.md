## Machine Learning Classification
Slide 1: Introduction to Classification

Classification is a fundamental task in machine learning where we predict discrete class labels for input data. It's widely used in various applications, from spam detection to medical diagnosis.

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

print(f"Accuracy: {clf.score(X_test, y_test):.2f}")
```

Slide 2: Binary Classification

Binary classification involves categorizing instances into one of two classes. It's commonly used in scenarios like spam detection or disease diagnosis.

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict for a new point
new_point = np.array([[1.5, 0.5]])
prediction = model.predict(new_point)
probability = model.predict_proba(new_point)

print(f"Predicted class: {prediction[0]}")
print(f"Probability of class 1: {probability[0][1]:.2f}")
```

Slide 3: Multiclass Classification

Multiclass classification extends binary classification to problems with more than two classes. It's used in scenarios like digit recognition or species identification.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load iris dataset (3 classes)
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM classifier
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_scaled, y)

# Predict for a new sample
new_sample = scaler.transform([[5.1, 3.5, 1.4, 0.2]])
prediction = svm.predict(new_sample)

print(f"Predicted class: {iris.target_names[prediction[0]]}")
```

Slide 4: Multilabel Classification

Multilabel classification allows each instance to belong to multiple classes simultaneously. It's useful in scenarios like image tagging or document categorization.

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Generate synthetic multilabel data
np.random.seed(42)
X = np.random.randn(100, 5)
y = np.random.randint(2, size=(100, 3))

# Train multilabel classifier
forest = RandomForestClassifier(n_estimators=100, random_state=42)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest.fit(X, y)

# Predict for a new sample
new_sample = np.array([[0.5, 1.2, -0.3, 0.8, -1.5]])
prediction = multi_target_forest.predict(new_sample)

print(f"Predicted labels: {prediction[0]}")
```

Slide 5: Confusion Matrix

A confusion matrix is a table that visualizes the performance of a classification model, showing the counts of true positives, true negatives, false positives, and false negatives.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate synthetic predictions
y_true = np.array([0, 1, 2, 2, 1, 0, 1, 0, 2, 1])
y_pred = np.array([0, 2, 1, 2, 1, 0, 1, 0, 2, 1])

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

Slide 6: Precision

Precision measures the accuracy of positive predictions. It's the ratio of true positives to the total number of positive predictions.

```python
from sklearn.metrics import precision_score

# Binary classification example
y_true = [0, 1, 1, 0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1]

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.2f}")

# Calculate precision manually
true_positives = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_true, y_pred))
predicted_positives = sum(yp == 1 for yp in y_pred)
manual_precision = true_positives / predicted_positives
print(f"Manual Precision: {manual_precision:.2f}")
```

Slide 7: Recall

Recall measures the ability to find all positive instances. It's the ratio of true positives to the total number of actual positive instances.

```python
from sklearn.metrics import recall_score

# Binary classification example
y_true = [0, 1, 1, 0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1]

recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.2f}")

# Calculate recall manually
true_positives = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_true, y_pred))
actual_positives = sum(yt == 1 for yt in y_true)
manual_recall = true_positives / actual_positives
print(f"Manual Recall: {manual_recall:.2f}")
```

Slide 8: F1 Score

The F1 score is the harmonic mean of precision and recall, providing a single score that balances both metrics. It's particularly useful when you have an uneven class distribution.

```python
from sklearn.metrics import f1_score

# Binary classification example
y_true = [0, 1, 1, 0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1]

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.2f}")

# Calculate F1 score manually
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
manual_f1 = 2 * (precision * recall) / (precision + recall)
print(f"Manual F1 Score: {manual_f1:.2f}")
```

Slide 9: ROC Curve and AUC

The Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) are used to evaluate the performance of binary classifiers across various threshold settings.

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
y_true = np.random.randint(2, size=100)
y_scores = np.random.rand(100)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

Slide 10: Cross-Validation

Cross-validation is a technique used to assess model performance and prevent overfitting by splitting the data into multiple training and validation sets.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Create a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5)

print("Cross-validation scores:", cv_scores)
print(f"Mean CV score: {cv_scores.mean():.2f}")
print(f"Standard deviation of CV scores: {cv_scores.std():.2f}")
```

Slide 11: Real-Life Example: Sentiment Analysis

Sentiment analysis is a common application of text classification, used to determine the emotional tone behind words.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample tweets
tweets = [
    "I love this product! It's amazing!",
    "This is the worst experience ever.",
    "Neutral opinion about this service.",
    "Absolutely fantastic customer support!",
    "Disappointed with the quality."
]
sentiments = [1, 0, 2, 1, 0]  # 1: positive, 0: negative, 2: neutral

# Create a pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(tweets, sentiments)

# Predict sentiment for a new tweet
new_tweet = ["The product exceeded my expectations!"]
prediction = pipeline.predict(new_tweet)

sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
print(f"Predicted sentiment: {sentiment_map[prediction[0]]}")
```

Slide 12: Real-Life Example: Image Classification

Image classification is widely used in computer vision applications, from facial recognition to medical imaging diagnostics.

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load and preprocess an image
img_path = 'path_to_your_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make prediction
preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]

# Print top 3 predictions
for i, (imagenet_id, label, score) in enumerate(decoded_preds):
    print(f"{i + 1}: {label} ({score:.2f})")
```

Slide 13: Error Analysis

Error analysis involves examining misclassified instances to understand model weaknesses and guide improvements.

```python
from sklearn.metrics import classification_report
import pandas as pd

# Assuming we have true labels and predictions
y_true = [0, 1, 2, 2, 1, 0, 1, 0, 2, 1]
y_pred = [0, 2, 1, 2, 1, 0, 1, 0, 2, 1]

# Generate classification report
report = classification_report(y_true, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

print(df_report)

# Identify misclassified instances
misclassified = [(true, pred) for true, pred in zip(y_true, y_pred) if true != pred]
print("\nMisclassified instances (true label, predicted label):")
for true, pred in misclassified:
    print(f"True: {true}, Predicted: {pred}")
```

Slide 14: Model Interpretation: Feature Importance

Understanding feature importance helps interpret model decisions and can guide feature engineering efforts.

```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_
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

Slide 15: Additional Resources

For further exploration of classification techniques and performance metrics, consider these peer-reviewed articles:

1. "A Survey of Deep Learning Techniques for Image Classification" - arXiv:2009.09809
2. "Understanding Confusion Matrices" - arXiv:2008.05786
3. "An Introduction to ROC Analysis" - arXiv:2008.04635

These resources provide in-depth discussions of advanced topics in machine learning classification.

