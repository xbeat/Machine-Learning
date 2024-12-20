## Boosting Model Accuracy with Self-Training
Slide 1: Introduction to Semi-Supervised Learning

Semi-supervised learning (SSL) is a machine learning paradigm that leverages both labeled and unlabeled data. This approach is particularly useful when labeled data is scarce or expensive to obtain. SSL aims to improve model performance by utilizing the abundant unlabeled data alongside a limited amount of labeled data.

```python
import numpy as np
from sklearn.semi_supervised import LabelSpreading

# Generate sample data
rng = np.random.RandomState(0)
X = rng.randn(200, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# Labeled and unlabeled split
labeled_indices = [0, 1, 2, 3]
y_train = np.(y)
y_train[~np.in1d(np.arange(len(y)), labeled_indices)] = -1

# Train Label Spreading model
model = LabelSpreading(kernel='rbf', alpha=0.5)
model.fit(X, y_train)

# Predict and print accuracy
print(f"Accuracy: {model.score(X, y):.2f}")
```

Slide 2: Pre-Training in Semi-Supervised Learning

Pre-training is a common SSL technique where a model is initially trained on a large amount of unlabeled data in a self-supervised manner. This pre-trained model is then fine-tuned on a smaller labeled dataset for specific tasks. The pre-training phase helps the model learn general features from the data, which can be beneficial for downstream tasks.

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Load pre-trained BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example text for pre-training
text = "The quick brown fox jumps over the lazy dog."

# Tokenize and encode the text
inputs = tokenizer(text, return_tensors="pt")

# Get the output from the pre-trained model
outputs = model(**inputs)

# Access the last hidden states
last_hidden_states = outputs.last_hidden_state
print(f"Shape of last hidden states: {last_hidden_states.shape}")
```

Slide 3: Fine-Tuning in Semi-Supervised Learning

Fine-tuning is the supervised component of semi-supervised pre-training. After pre-training, the model is further trained on a smaller labeled dataset to adapt it to a specific task. This process allows the model to leverage the general knowledge gained during pre-training and specialize it for the target task.

```python
import torch.optim as optim

# Assume we have a pre-trained model 'model' and labeled data 'X_labeled', 'y_labeled'

# Define a simple classifier on top of the pre-trained model
class Classifier(nn.Module):
    def __init__(self, pre_trained_model, num_classes):
        super().__init__()
        self.pre_trained = pre_trained_model
        self.classifier = nn.Linear(768, num_classes)  # 768 is BERT's hidden size

    def forward(self, x):
        features = self.pre_trained(**x).last_hidden_state[:, 0, :]
        return self.classifier(features)

# Create classifier and optimizer
classifier = Classifier(model, num_classes=2)
optimizer = optim.Adam(classifier.parameters())

# Fine-tuning loop (simplified)
for epoch in range(5):
    for batch in get_batches(X_labeled, y_labeled):
        optimizer.zero_grad()
        outputs = classifier(batch['input_ids'])
        loss = nn.functional.cross_entropy(outputs, batch['labels'])
        loss.backward()
        optimizer.step()

print("Fine-tuning completed")
```

Slide 4: Self-Training in Semi-Supervised Learning

Self-training is another SSL technique where a model is initially trained on a small set of labeled data. This trained model, called the teacher model, then makes predictions on unlabeled data. The high-confidence predictions are treated as pseudo-labels and added to the labeled dataset to train a student model.

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def self_training(X_labeled, y_labeled, X_unlabeled, confidence_threshold=0.8):
    # Train initial model on labeled data
    model = SVC(probability=True)
    model.fit(X_labeled, y_labeled)

    while len(X_unlabeled) > 0:
        # Predict on unlabeled data
        predictions = model.predict_proba(X_unlabeled)
        
        # Find high-confidence predictions
        max_probs = predictions.max(axis=1)
        confident_idx = max_probs >= confidence_threshold
        
        if not any(confident_idx):
            break
        
        # Add high-confidence predictions to labeled data
        new_X = X_unlabeled[confident_idx]
        new_y = model.predict(new_X)
        X_labeled = np.vstack((X_labeled, new_X))
        y_labeled = np.concatenate((y_labeled, new_y))
        
        # Remove pseudo-labeled data from unlabeled set
        X_unlabeled = X_unlabeled[~confident_idx]
        
        # Retrain model
        model.fit(X_labeled, y_labeled)
    
    return model

# Usage
model = self_training(X_labeled, y_labeled, X_unlabeled)
print(f"Final model accuracy: {accuracy_score(y_true, model.predict(X_test)):.2f}")
```

Slide 5: Comparing Self-Training and Pre-Training

Recent research suggests that self-training can outperform pre-training in certain scenarios, especially when combined with data augmentation and increased availability of labeled data. However, it's important to note that the effectiveness of each method can vary depending on the specific task and dataset.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curves(estimator1, estimator2, X, y, title):
    train_sizes, train_scores1, test_scores1 = learning_curve(estimator1, X, y)
    train_sizes, train_scores2, test_scores2 = learning_curve(estimator2, X, y)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(test_scores1, axis=1), label='Self-Training')
    plt.plot(train_sizes, np.mean(test_scores2, axis=1), label='Pre-Training')
    plt.title(title)
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.show()

# Assume we have self_trained_model and pre_trained_model
plot_learning_curves(self_trained_model, pre_trained_model, X, y, 
                     'Learning Curves: Self-Training vs Pre-Training')
```

Slide 6: Data Augmentation in SSL

Data augmentation is a technique used to increase the diversity of the training data by applying various transformations. When combined with self-training or pre-training, it can significantly improve model performance, especially in image recognition tasks.

```python
from torchvision import transforms

# Define data augmentation transforms
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Apply augmentation to an image
augmented_image = data_transforms(original_image)

# Visualize original and augmented images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(original_image)
ax1.set_title('Original Image')
ax2.imshow(augmented_image.permute(1, 2, 0))
ax2.set_title('Augmented Image')
plt.show()
```

Slide 7: Impact of Pre-Training Knowledge

The strength of pre-training knowledge can significantly affect model performance. For instance, models pre-trained on larger datasets or with more complex architectures (e.g., BERT-base vs BERT-medium) tend to perform better, especially under conditions of no or moderate data augmentation.

```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

def compare_models(text, model_names):
    results = {}
    for name in model_names:
        model = BertForSequenceClassification.from_pretrained(name)
        tokenizer = BertTokenizer.from_pretrained(name)
        
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        results[name] = outputs.logits.softmax(dim=1)
    
    return results

text = "This movie is great!"
model_names = ['bert-base-uncased', 'bert-medium-uncased']
results = compare_models(text, model_names)

for name, probs in results.items():
    print(f"{name} prediction: {probs}")
```

Slide 8: Challenges in SSL: Accuracy Decline

While SSL techniques like self-training can initially boost model accuracy, there's often a point where accuracy starts to decline. This phenomenon is known as "pseudo-label noise" and occurs when the model begins to reinforce its own mistakes through the pseudo-labeling process.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_ssl_accuracy(iterations, initial_accuracy, peak_iteration, decline_rate):
    accuracies = []
    for i in range(iterations):
        if i <= peak_iteration:
            accuracy = initial_accuracy + (1 - initial_accuracy) * (i / peak_iteration)
        else:
            accuracy = 1 - decline_rate * (i - peak_iteration)
        accuracies.append(max(min(accuracy, 1), initial_accuracy))
    return accuracies

iterations = 100
initial_accuracy = 0.7
peak_iteration = 40
decline_rate = 0.005

accuracies = simulate_ssl_accuracy(iterations, initial_accuracy, peak_iteration, decline_rate)

plt.figure(figsize=(10, 6))
plt.plot(range(iterations), accuracies)
plt.title('Simulated SSL Accuracy Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.show()
```

Slide 9: Real-Life Example: Image Classification

Consider a wildlife conservation project that aims to identify different species of birds from camera trap images. With limited labeled data available, SSL techniques can be employed to improve the classification accuracy.

```python
import torchvision.models as models
import torch.nn as nn

def create_ssl_model(num_classes):
    # Load a pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    
    # Freeze the pre-trained layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# Create model for bird species classification
num_bird_species = 200
bird_classifier = create_ssl_model(num_bird_species)

print(f"Model created with {num_bird_species} output classes")
```

Slide 10: Real-Life Example: Text Classification

In a customer service scenario, a company wants to automatically categorize incoming support tickets. With a large volume of unlabeled historical tickets and a small set of manually labeled ones, SSL can be used to improve the classification accuracy.

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load pre-trained DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Example function to classify a support ticket
def classify_ticket(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    categories = ['Billing', 'Technical', 'Account', 'Product', 'General']
    return categories[predicted_class]

# Example usage
ticket_text = "I can't log into my account. It says my password is incorrect."
category = classify_ticket(ticket_text)
print(f"Predicted category: {category}")
```

Slide 11: Implementing SSL with PyTorch

Here's a basic implementation of a semi-supervised learning pipeline using PyTorch. This example demonstrates how to combine labeled and unlabeled data in the training process.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SSLModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

def ssl_train_step(model, labeled_data, unlabeled_data, optimizer, criterion, lambda_u):
    model.train()
    optimizer.zero_grad()
    
    # Supervised loss
    outputs = model(labeled_data['x'])
    loss_s = criterion(outputs, labeled_data['y'])
    
    # Unsupervised loss (pseudo-labeling)
    with torch.no_grad():
        pseudo_labels = model(unlabeled_data).argmax(dim=1)
    outputs_u = model(unlabeled_data)
    loss_u = criterion(outputs_u, pseudo_labels)
    
    # Combined loss
    loss = loss_s + lambda_u * loss_u
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Usage
model = SSLModel(input_dim=10, hidden_dim=50, output_dim=2)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    loss = ssl_train_step(model, labeled_batch, unlabeled_batch, optimizer, criterion, lambda_u=0.5)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
```

Slide 12: Evaluating SSL Models

Evaluating SSL models requires careful consideration of both labeled and unlabeled data. Here's an example of how to evaluate a semi-supervised model using a holdout test set and various metrics.

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_ssl_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return accuracy, precision, recall, f1

# Usage
model = SSLModel(input_dim=10, hidden_dim=50, output_dim=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
accuracy, precision, recall, f1 = evaluate_ssl_model(model, test_loader)
```

Slide 13: Challenges and Future Directions in SSL

While SSL has shown promising results, there are still challenges to overcome and areas for future research. These include:

1. Handling domain shift between labeled and unlabeled data
2. Improving the reliability of pseudo-labels in self-training
3. Developing more effective ways to leverage unlabeled data
4. Addressing the problem of confirmation bias in iterative SSL methods

Researchers are exploring various approaches to tackle these challenges, such as improved consistency regularization techniques, adaptive pseudo-labeling strategies, and novel architectures designed specifically for SSL tasks.

Slide 14: Challenges and Future Directions in SSL

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_ssl_challenges():
    challenges = ['Domain Shift', 'Pseudo-Label Reliability', 'Unlabeled Data Usage', 'Confirmation Bias']
    difficulty = [0.8, 0.7, 0.6, 0.9]
    
    plt.figure(figsize=(10, 6))
    plt.bar(challenges, difficulty)
    plt.title('Challenges in Semi-Supervised Learning')
    plt.xlabel('Challenges')
    plt.ylabel('Relative Difficulty')
    plt.ylim(0, 1)
    plt.show()

plot_ssl_challenges()
```

Slide 15: Additional Resources

For those interested in delving deeper into semi-supervised learning, here are some valuable resources:

1. "Semi-Supervised Learning" by Olivier Chapelle, Bernhard Schölkopf, and Alexander Zien (MIT Press)
2. "A Survey on Semi-Supervised Learning" by Jesper E. van Engelen and Holger H. Hoos (arXiv:1905.00303)
3. "Realistic Evaluation of Deep Semi-Supervised Learning Algorithms" by Avital Oliver et al. (arXiv:1804.09170)
4. "MixMatch: A Holistic Approach to Semi-Supervised Learning" by David Berthelot et al. (arXiv:1905.02249)

These resources provide comprehensive overviews of SSL techniques, algorithms, and recent advancements in the field. They offer valuable insights for both beginners and experienced practitioners looking to expand their knowledge of semi-supervised learning.

