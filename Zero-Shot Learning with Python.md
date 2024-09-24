## Zero-Shot Learning with Python
Slide 1: Introduction to Zero-Shot Learning

Zero-Shot Learning (ZSL) is a machine learning paradigm where a model can classify or make predictions for classes it has never seen during training. This approach leverages the semantic relationships between known and unknown classes to generalize knowledge.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Simulating semantic embeddings for known and unknown classes
known_classes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
unknown_classes = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

# Training a simple ZSL model
model = LogisticRegression()
model.fit(known_classes, [0, 1, 2])

# Predicting unknown classes
predictions = model.predict(unknown_classes)
print("Predictions for unknown classes:", predictions)
```

Slide 2: Key Concepts in Zero-Shot Learning

Zero-Shot Learning relies on three main components: a feature extractor, a semantic embedding space, and a compatibility function. The feature extractor transforms input data into a meaningful representation, while the semantic embedding space captures relationships between classes. The compatibility function measures how well an input matches a class description.

```python
import torch
import torch.nn as nn

class ZeroShotModel(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.feature_extractor = nn.Linear(input_dim, 128)
        self.compatibility_function = nn.Linear(128, embedding_dim)
    
    def forward(self, x, class_embeddings):
        features = self.feature_extractor(x)
        compatibility = self.compatibility_function(features)
        scores = torch.matmul(compatibility, class_embeddings.T)
        return scores

# Example usage
model = ZeroShotModel(input_dim=300, embedding_dim=100)
input_data = torch.randn(1, 300)
class_embeddings = torch.randn(5, 100)  # 5 classes, 100-dim embeddings
scores = model(input_data, class_embeddings)
print("Class scores:", scores)
```

Slide 3: Semantic Embeddings

Semantic embeddings are crucial in Zero-Shot Learning as they bridge the gap between known and unknown classes. These embeddings can be derived from various sources such as word vectors, attribute descriptions, or knowledge graphs.

```python
from gensim.models import KeyedVectors

# Load pre-trained word vectors
word_vectors = KeyedVectors.load_word2vec_format('path_to_word_vectors.bin', binary=True)

# Get embeddings for class names
class_names = ['dog', 'cat', 'bird', 'fish']
class_embeddings = [word_vectors[name] for name in class_names]

# Function to find the most similar class
def find_similar_class(query, class_embeddings, class_names):
    query_embedding = word_vectors[query]
    similarities = [np.dot(query_embedding, emb) for emb in class_embeddings]
    most_similar_idx = np.argmax(similarities)
    return class_names[most_similar_idx]

# Example: Finding the most similar class for 'whale'
print(find_similar_class('whale', class_embeddings, class_names))
```

Slide 4: Feature Extraction in Zero-Shot Learning

Feature extraction is a critical step in Zero-Shot Learning, transforming raw input data into a meaningful representation that can be used for classification. In many cases, pre-trained models are used as feature extractors to leverage their learned representations.

```python
import torch
from torchvision import models, transforms

# Load a pre-trained ResNet model
resnet = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

# Prepare image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features from an image
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(image)
    return features.squeeze()

# Example usage
image_features = extract_features('path_to_image.jpg')
print("Feature shape:", image_features.shape)
```

Slide 5: Compatibility Functions

Compatibility functions in Zero-Shot Learning measure how well an input matches a class description. These functions can range from simple dot products to more complex neural networks.

```python
import torch
import torch.nn as nn

class BilinearCompatibility(nn.Module):
    def __init__(self, feature_dim, embedding_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(feature_dim, embedding_dim))
    
    def forward(self, features, class_embeddings):
        return torch.matmul(features, torch.matmul(self.W, class_embeddings.T))

# Example usage
feature_dim, embedding_dim = 2048, 300
compatibility_func = BilinearCompatibility(feature_dim, embedding_dim)

features = torch.randn(1, feature_dim)
class_embeddings = torch.randn(5, embedding_dim)  # 5 classes

scores = compatibility_func(features, class_embeddings)
print("Compatibility scores:", scores)
```

Slide 6: Training a Zero-Shot Learning Model

Training a Zero-Shot Learning model involves optimizing the feature extractor and compatibility function to align visual features with semantic embeddings. This is often done using a ranking loss or cross-entropy loss on seen classes.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ZSLModel(nn.Module):
    def __init__(self, feature_dim, embedding_dim, num_classes):
        super().__init__()
        self.feature_extractor = nn.Linear(feature_dim, 512)
        self.compatibility = nn.Linear(512, embedding_dim)
    
    def forward(self, x, class_embeddings):
        features = self.feature_extractor(x)
        compatibility = self.compatibility(features)
        scores = torch.matmul(compatibility, class_embeddings.T)
        return scores

# Simulated data
feature_dim, embedding_dim, num_classes = 1000, 300, 50
model = ZSLModel(feature_dim, embedding_dim, num_classes)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    inputs = torch.randn(32, feature_dim)
    labels = torch.randint(0, num_classes, (32,))
    class_embeddings = torch.randn(num_classes, embedding_dim)
    
    optimizer.zero_grad()
    outputs = model(inputs, class_embeddings)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

Slide 7: Generalized Zero-Shot Learning

Generalized Zero-Shot Learning (GZSL) extends the ZSL paradigm by considering both seen and unseen classes during testing. This approach is more realistic but also more challenging due to the bias towards seen classes.

```python
import numpy as np
from sklearn.metrics import accuracy_score

def gzsl_evaluation(model, seen_data, unseen_data, seen_labels, unseen_labels):
    all_data = np.vstack((seen_data, unseen_data))
    all_labels = np.concatenate((seen_labels, unseen_labels))
    
    predictions = model.predict(all_data)
    
    seen_acc = accuracy_score(seen_labels, predictions[:len(seen_labels)])
    unseen_acc = accuracy_score(unseen_labels, predictions[len(seen_labels):])
    
    harmonic_mean = 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc)
    
    return {
        'Seen Accuracy': seen_acc,
        'Unseen Accuracy': unseen_acc,
        'Harmonic Mean': harmonic_mean
    }

# Simulated GZSL evaluation
np.random.seed(42)
seen_data = np.random.randn(100, 10)
unseen_data = np.random.randn(50, 10)
seen_labels = np.random.randint(0, 5, 100)
unseen_labels = np.random.randint(5, 10, 50)

class DummyModel:
    def predict(self, X):
        return np.random.randint(0, 10, len(X))

model = DummyModel()
results = gzsl_evaluation(model, seen_data, unseen_data, seen_labels, unseen_labels)
print("GZSL Evaluation Results:", results)
```

Slide 8: Attribute-Based Zero-Shot Learning

Attribute-based ZSL uses human-interpretable attributes to describe classes. This approach allows for more intuitive class descriptions and can improve model interpretability.

```python
import numpy as np
from scipy.spatial.distance import cosine

# Simulated attribute descriptions for animals
attributes = {
    'dog': [1, 1, 1, 0],  # [furry, four_legs, barks, flies]
    'cat': [1, 1, 0, 0],
    'bird': [0, 0, 0, 1],
    'fish': [0, 0, 0, 0]
}

# Function to predict class based on attributes
def predict_class(input_attributes, attribute_dict):
    similarities = {}
    for class_name, class_attributes in attribute_dict.items():
        similarity = 1 - cosine(input_attributes, class_attributes)
        similarities[class_name] = similarity
    return max(similarities, key=similarities.get)

# Example: Predicting class for a new animal
new_animal_attributes = [1, 1, 0, 0]  # furry, four legs, doesn't bark, doesn't fly
predicted_class = predict_class(new_animal_attributes, attributes)
print(f"Predicted class for new animal: {predicted_class}")
```

Slide 9: Embedding-Based Zero-Shot Learning

Embedding-based ZSL leverages pre-trained word embeddings or sentence encoders to create semantic representations of class names or descriptions. This approach can capture rich semantic relationships between classes.

```python
from sentence_transformers import SentenceTransformer
import torch

# Load a pre-trained sentence encoder
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Class descriptions
class_descriptions = [
    "A four-legged mammal that barks and is often kept as a pet.",
    "A small, furry mammal that meows and is known for its independent nature.",
    "A feathered animal that can fly and often sings.",
    "An aquatic animal with scales and fins that lives in water."
]

# Encode class descriptions
class_embeddings = model.encode(class_descriptions)

# Function to find the most similar class
def find_similar_class(query, class_embeddings, class_names):
    query_embedding = model.encode([query])
    similarities = torch.nn.functional.cosine_similarity(
        torch.tensor(query_embedding), 
        torch.tensor(class_embeddings)
    )
    most_similar_idx = similarities.argmax().item()
    return class_names[most_similar_idx]

# Example: Finding the most similar class for a new description
new_description = "A large marine mammal that lives in the ocean and breathes air."
class_names = ["dog", "cat", "bird", "fish"]
predicted_class = find_similar_class(new_description, class_embeddings, class_names)
print(f"Predicted class for new description: {predicted_class}")
```

Slide 10: Transductive Zero-Shot Learning

Transductive Zero-Shot Learning assumes access to unlabeled data from unseen classes during training. This approach can leverage the structure of unseen class data to improve performance.

```python
import numpy as np
from sklearn.cluster import KMeans

def transductive_zsl(seen_data, unseen_data, seen_labels, num_unseen_classes):
    # Combine seen and unseen data
    all_data = np.vstack((seen_data, unseen_data))
    
    # Perform clustering on all data
    kmeans = KMeans(n_clusters=len(np.unique(seen_labels)) + num_unseen_classes)
    cluster_labels = kmeans.fit_predict(all_data)
    
    # Assign seen classes to clusters
    cluster_to_class = {}
    for i, label in enumerate(seen_labels):
        cluster = cluster_labels[i]
        if cluster not in cluster_to_class:
            cluster_to_class[cluster] = label
    
    # Predict labels for unseen data
    unseen_predictions = []
    for i in range(len(seen_data), len(all_data)):
        cluster = cluster_labels[i]
        if cluster in cluster_to_class:
            unseen_predictions.append(cluster_to_class[cluster])
        else:
            unseen_predictions.append(-1)  # Unknown class
    
    return unseen_predictions

# Simulated data
np.random.seed(42)
seen_data = np.random.randn(100, 10)
unseen_data = np.random.randn(50, 10)
seen_labels = np.random.randint(0, 5, 100)
num_unseen_classes = 3

predictions = transductive_zsl(seen_data, unseen_data, seen_labels, num_unseen_classes)
print("Predictions for unseen data:", predictions[:10])
```

Slide 11: Few-Shot Learning vs. Zero-Shot Learning

While Zero-Shot Learning deals with classes unseen during training, Few-Shot Learning allows for a small number of examples per new class. Let's compare these approaches using a simple implementation.

```python
import numpy as np
from sklearn.metrics import accuracy_score

class SimpleLearner:
    def __init__(self):
        self.prototypes = {}
    
    def fit(self, X, y):
        for label in np.unique(y):
            self.prototypes[label] = X[y == label].mean(axis=0)
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = [np.linalg.norm(x - proto) for proto in self.prototypes.values()]
            pred_label = list(self.prototypes.keys())[np.argmin(distances)]
            predictions.append(pred_label)
        return np.array(predictions)

# Simulated data
np.random.seed(42)
train_data = np.random.randn(100, 10)
train_labels = np.random.randint(0, 5, 100)
test_data = np.random.randn(20, 10)
test_labels = np.random.randint(0, 7, 20)  # Including unseen classes

# Zero-Shot scenario
zsl_model = SimpleLearner()
zsl_model.fit(train_data, train_labels)
zsl_predictions = zsl_model.predict(test_data)
zsl_accuracy = accuracy_score(test_labels, zsl_predictions)

# Few-Shot scenario (3 shots per new class)
fsl_model = SimpleLearner()
fsl_model.fit(train_data, train_labels)
for new_class in range(5, 7):
    few_shot_data = np.random.randn(3, 10)
    few_shot_labels = np.full(3, new_class)
    fsl_model.fit(few_shot_data, few_shot_labels)

fsl_predictions = fsl_model.predict(test_data)
fsl_accuracy = accuracy_score(test_labels, fsl_predictions)

print(f"Zero-Shot Learning Accuracy: {zsl_accuracy:.4f}")
print(f"Few-Shot Learning Accuracy: {fsl_accuracy:.4f}")
```

Slide 12: Real-Life Example: Image Classification

Zero-Shot Learning can be applied to image classification tasks where new classes emerge frequently. Here's a simplified example using a pre-trained image encoder and word embeddings.

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load pre-trained ResNet model
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Simulated word embeddings
class_embeddings = {
    'dog': torch.randn(1000),
    'cat': torch.randn(1000),
    'bird': torch.randn(1000),
    'fish': torch.randn(1000)
}

def classify_image(image_path, class_embeddings):
    # Load and preprocess image
    image = Image.open(image_path)
    input_tensor = preprocess(image).unsqueeze(0)
    
    # Extract features
    with torch.no_grad():
        features = resnet(input_tensor)
    
    # Compare with class embeddings
    similarities = {cls: torch.cosine_similarity(features, emb.unsqueeze(0)) 
                    for cls, emb in class_embeddings.items()}
    
    return max(similarities, key=similarities.get)

# Example usage
image_path = 'path_to_image.jpg'
predicted_class = classify_image(image_path, class_embeddings)
print(f"Predicted class: {predicted_class}")
```

Slide 13: Real-Life Example: Text Classification

Zero-Shot Learning can be applied to text classification tasks, such as sentiment analysis or topic classification, where new categories may emerge over time.

```python
from transformers import pipeline

# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification")

# Example texts
texts = [
    "The new smartphone has an amazing camera and long battery life.",
    "The restaurant's service was slow, and the food was cold.",
    "Scientists have discovered a new species of deep-sea fish."
]

# Candidate labels
labels = ["technology review", "restaurant review", "scientific discovery"]

# Perform zero-shot classification
for text in texts:
    result = classifier(text, candidate_labels=labels)
    print(f"Text: {text}")
    print(f"Predicted label: {result['labels'][0]}")
    print(f"Confidence: {result['scores'][0]:.4f}")
    print()
```

Slide 14: Challenges and Future Directions in Zero-Shot Learning

Zero-Shot Learning faces several challenges, including the semantic gap between visual and semantic spaces, bias towards seen classes, and the need for high-quality semantic embeddings. Future research directions include:

1. Improving the alignment between visual and semantic spaces
2. Developing more robust compatibility functions
3. Incorporating external knowledge sources
4. Addressing the domain shift problem in real-world applications

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulating the semantic gap problem
np.random.seed(42)
visual_features = np.random.randn(100, 2)
semantic_embeddings = np.random.randn(100, 2)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(visual_features[:, 0], visual_features[:, 1], c='blue', alpha=0.5)
plt.title("Visual Space")
plt.subplot(122)
plt.scatter(semantic_embeddings[:, 0], semantic_embeddings[:, 1], c='red', alpha=0.5)
plt.title("Semantic Space")
plt.tight_layout()
plt.show()

# Function to measure alignment
def measure_alignment(visual, semantic):
    return np.mean(np.abs(visual - semantic))

alignment_score = measure_alignment(visual_features, semantic_embeddings)
print(f"Alignment score: {alignment_score:.4f}")
```

Slide 15: Additional Resources

For those interested in diving deeper into Zero-Shot Learning, here are some valuable resources:

1. "Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly" by Y. Xian et al. (2018) ArXiv: [https://arxiv.org/abs/1707.00600](https://arxiv.org/abs/1707.00600)
2. "A Survey of Zero-Shot Learning: Settings, Methods, and Applications" by W. Wang et al. (2019) ArXiv: [https://arxiv.org/abs/1907.13258](https://arxiv.org/abs/1907.13258)
3. "Generalized Zero-Shot Learning via Synthesized Examples" by V. K. Verma et al. (2018) ArXiv: [https://arxiv.org/abs/1712.03878](https://arxiv.org/abs/1712.03878)
4. "Transductive Zero-Shot Learning with Visual Structure Constraint" by Y. Li et al. (2019) ArXiv: [https://arxiv.org/abs/1901.01570](https://arxiv.org/abs/1901.01570)

These papers provide a comprehensive overview of Zero-Shot Learning techniques, challenges, and recent advancements in the field.

