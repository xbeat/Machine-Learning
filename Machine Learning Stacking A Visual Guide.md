## Machine Learning Stacking A Visual Guide
Slide 1: Introduction to Stacking

Stacking is an advanced ensemble learning technique that combines predictions from multiple models to create a more powerful and accurate predictor. This method leverages the strengths of different algorithms while mitigating their individual weaknesses. Stacking typically involves two layers of models: base models and a meta-model.

Slide 2: Source Code for Introduction to Stacking

```python
# Simplified illustration of stacking concept
class StackingEnsemble:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        # Train base models
        for model in self.base_models:
            model.fit(X, y)
        
        # Generate meta-features
        meta_features = self.generate_meta_features(X)
        
        # Train meta-model
        self.meta_model.fit(meta_features, y)

    def generate_meta_features(self, X):
        return [model.predict(X) for model in self.base_models]

    def predict(self, X):
        meta_features = self.generate_meta_features(X)
        return self.meta_model.predict(meta_features)
```

Slide 3: Base Models in Stacking

Base models form the foundation of a stacking ensemble. These are diverse algorithms trained on the same dataset to make independent predictions. Common choices include decision trees, neural networks, support vector machines, and k-nearest neighbors. The key is to select models with different strengths and weaknesses to capture various aspects of the data.

Slide 4: Source Code for Base Models in Stacking

```python
# Implementing simple base models
class DecisionStump:
    def fit(self, X, y):
        self.feature = max(range(len(X[0])), key=lambda i: abs(sum(X[j][i]*y[j] for j in range(len(X)))))
        self.threshold = sum(X[i][self.feature] for i in range(len(X))) / len(X)

    def predict(self, X):
        return [1 if x[self.feature] > self.threshold else -1 for x in X]

class SimplePerceptron:
    def fit(self, X, y, epochs=100, lr=0.01):
        self.weights = [0] * len(X[0])
        for _ in range(epochs):
            for x, target in zip(X, y):
                prediction = sum(w*xi for w, xi in zip(self.weights, x))
                error = target - (1 if prediction > 0 else -1)
                self.weights = [w + lr * error * xi for w, xi in zip(self.weights, x)]

    def predict(self, X):
        return [1 if sum(w*xi for w, xi in zip(self.weights, x)) > 0 else -1 for x in X]

# Usage
base_models = [DecisionStump(), SimplePerceptron()]
```

Slide 5: Meta-Model in Stacking

The meta-model, also known as the blender or second-level learner, combines the predictions of the base models. It learns how to best weigh and combine these predictions to make a final, more accurate prediction. The meta-model can be any machine learning algorithm, but it's often chosen to be flexible enough to capture complex relationships between base model predictions.

Slide 6: Source Code for Meta-Model in Stacking

```python
class LogisticRegression:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + sum([-x for x in z]))

    def fit(self, X, y):
        self.weights = [0] * len(X[0])
        for _ in range(self.epochs):
            for x, target in zip(X, y):
                z = sum(w*xi for w, xi in zip(self.weights, x))
                prediction = self.sigmoid(z)
                error = target - prediction
                self.weights = [w + self.lr * error * xi for w, xi in zip(self.weights, x)]

    def predict(self, X):
        return [1 if self.sigmoid(sum(w*xi for w, xi in zip(self.weights, x))) > 0.5 else 0 for x in X]

# Usage as meta-model
meta_model = LogisticRegression()
```

Slide 7: Training Process in Stacking

The training process in stacking involves two main steps. First, the base models are trained on the original dataset. Then, these trained base models generate predictions on a validation set or through cross-validation. These predictions become the features for training the meta-model. This two-step process allows the meta-model to learn how to best combine the base models' predictions.

Slide 8: Source Code for Training Process in Stacking

```python
def train_stacking_ensemble(X_train, y_train, X_val, y_val, base_models, meta_model):
    # Train base models
    for model in base_models:
        model.fit(X_train, y_train)
    
    # Generate meta-features
    meta_features_train = [[model.predict([x])[0] for model in base_models] for x in X_train]
    meta_features_val = [[model.predict([x])[0] for model in base_models] for x in X_val]
    
    # Train meta-model
    meta_model.fit(meta_features_train, y_train)
    
    # Evaluate
    predictions = meta_model.predict(meta_features_val)
    accuracy = sum(1 for p, y in zip(predictions, y_val) if p == y) / len(y_val)
    
    return accuracy

# Usage
base_models = [DecisionStump(), SimplePerceptron()]
meta_model = LogisticRegression()
accuracy = train_stacking_ensemble(X_train, y_train, X_val, y_val, base_models, meta_model)
print(f"Stacking Ensemble Accuracy: {accuracy:.2f}")
```

Slide 9: Prediction Process in Stacking

During prediction, each base model generates a prediction for the new instance. These predictions are then fed into the meta-model, which makes the final prediction. This process allows the ensemble to leverage the strengths of each base model while mitigating their individual weaknesses, often resulting in improved overall performance.

Slide 10: Source Code for Prediction Process in Stacking

```python
def predict_stacking_ensemble(X_new, base_models, meta_model):
    # Generate meta-features for new data
    meta_features_new = [[model.predict([x])[0] for model in base_models] for x in X_new]
    
    # Make final predictions using meta-model
    predictions = meta_model.predict(meta_features_new)
    
    return predictions

# Usage
X_new = [[1.2, 2.3, 3.4], [4.5, 5.6, 6.7]]  # New data points
predictions = predict_stacking_ensemble(X_new, base_models, meta_model)
print("Predictions for new data:", predictions)
```

Slide 11: Real-Life Example: Image Classification

In image classification tasks, stacking can combine models specialized in different aspects. For instance, a convolutional neural network might excel at capturing spatial features, while a gradient boosting machine might better handle color histograms. By stacking these models, we can create a more robust classifier that leverages the strengths of both approaches.

Slide 12: Source Code for Image Classification Example

```python
# Simplified image classification example
class SimpleConvNet:
    def __init__(self):
        self.filters = [[random.random() for _ in range(9)] for _ in range(10)]
    
    def convolve(self, image):
        return sum(f*p for f, p in zip(self.filters, image[:90]))  # Simplified convolution
    
    def fit(self, images, labels):
        for _ in range(100):  # Simple training loop
            for img, label in zip(images, labels):
                pred = self.convolve(img)
                error = label - (1 if pred > 0 else 0)
                self.filters = [[f + 0.01 * error * p for f, p in zip(filter, img[:90])] for filter in self.filters]
    
    def predict(self, images):
        return [1 if self.convolve(img) > 0 else 0 for img in images]

class ColorHistogram:
    def fit(self, images, labels):
        self.avg_pos = [sum(img[i] for img, label in zip(images, labels) if label == 1) / sum(labels) for i in range(len(images[0]))]
        self.avg_neg = [sum(img[i] for img, label in zip(images, labels) if label == 0) / (len(labels) - sum(labels)) for i in range(len(images[0]))]
    
    def predict(self, images):
        return [1 if sum((img[i] - self.avg_neg[i])**2 for i in range(len(img))) < 
                   sum((img[i] - self.avg_pos[i])**2 for i in range(len(img))) else 0 for img in images]

# Stacking for image classification
base_models = [SimpleConvNet(), ColorHistogram()]
meta_model = LogisticRegression()

# Assuming we have image data: X_train, y_train, X_val, y_val
accuracy = train_stacking_ensemble(X_train, y_train, X_val, y_val, base_models, meta_model)
print(f"Image Classification Stacking Ensemble Accuracy: {accuracy:.2f}")
```

Slide 13: Real-Life Example: Text Sentiment Analysis

In sentiment analysis, stacking can combine different text processing techniques. A model based on TF-IDF vectors might capture important keywords, while a recurrent neural network could better understand context and word order. Stacking these models allows for a more comprehensive analysis of sentiment in text data.

Slide 14: Source Code for Text Sentiment Analysis Example

```python
class SimpleTFIDF:
    def fit(self, texts, labels):
        self.vocab = list(set(word for text in texts for word in text.split()))
        self.idf = {w: sum(1 for t in texts if w in t.split()) for w in self.vocab}
        self.weights = [0] * len(self.vocab)
        for _ in range(100):  # Simple training loop
            for text, label in zip(texts, labels):
                pred = sum(self.weights[self.vocab.index(w)] * text.count(w) / self.idf[w] for w in set(text.split()) if w in self.vocab)
                error = label - (1 if pred > 0 else 0)
                for w in set(text.split()):
                    if w in self.vocab:
                        self.weights[self.vocab.index(w)] += 0.01 * error * text.count(w) / self.idf[w]
    
    def predict(self, texts):
        return [1 if sum(self.weights[self.vocab.index(w)] * text.count(w) / self.idf[w] for w in set(text.split()) if w in self.vocab) > 0 else 0 for text in texts]

class SimpleRNN:
    def __init__(self, hidden_size=10):
        self.hidden_size = hidden_size
        self.Wxh = [[random.random() for _ in range(hidden_size)] for _ in range(26)]  # Assume 26 characters
        self.Whh = [[random.random() for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.Why = [random.random() for _ in range(hidden_size)]
    
    def forward(self, text):
        h = [0] * self.hidden_size
        for char in text.lower():
            x = [0] * 26
            x[ord(char) - ord('a')] = 1 if ord('a') <= ord(char) <= ord('z') else 0
            h = [sum(self.Wxh[i][j] * x[i] + self.Whh[i][j] * h[i] for i in range(len(x))) for j in range(self.hidden_size)]
        return sum(self.Why[i] * h[i] for i in range(self.hidden_size))
    
    def fit(self, texts, labels):
        for _ in range(100):  # Simple training loop
            for text, label in zip(texts, labels):
                pred = self.forward(text)
                error = label - (1 if pred > 0 else 0)
                # Simplified backpropagation (not accurate, just for illustration)
                self.Why = [w + 0.01 * error for w in self.Why]
    
    def predict(self, texts):
        return [1 if self.forward(text) > 0 else 0 for text in texts]

# Stacking for sentiment analysis
base_models = [SimpleTFIDF(), SimpleRNN()]
meta_model = LogisticRegression()

# Assuming we have text data: X_train, y_train, X_val, y_val
accuracy = train_stacking_ensemble(X_train, y_train, X_val, y_val, base_models, meta_model)
print(f"Sentiment Analysis Stacking Ensemble Accuracy: {accuracy:.2f}")
```

Slide 15: Additional Resources

For a deeper understanding of stacking and ensemble methods, consider exploring these research papers:

1.  "Stacked Generalization" by David H. Wolpert (1992) ArXiv: [https://arxiv.org/abs/neural/9205001](https://arxiv.org/abs/neural/9205001)
2.  "Ensemble Methods in Machine Learning" by Thomas G. Dietterich (2000) Available at: [https://arxiv.org/abs/cs/0004003](https://arxiv.org/abs/cs/0004003)

These papers provide foundational insights into stacking and its applications in various domains of machine learning.

