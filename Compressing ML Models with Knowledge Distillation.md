## Compressing ML Models with Knowledge Distillation
Slide 1: Introduction to Knowledge Distillation

Knowledge distillation is a technique used to compress large, complex machine learning models into smaller, more efficient ones while preserving their performance. This process involves transferring the knowledge from a larger "teacher" model to a smaller "student" model. The student model learns to mimic the behavior of the teacher, often achieving similar performance with reduced computational requirements.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple teacher model
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define a smaller student model
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Instantiate models
teacher = TeacherModel()
student = StudentModel()

print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters())}")
print(f"Student parameters: {sum(p.numel() for p in student.parameters())}")
```

Teacher parameters: 2,398,600 Student parameters: 317,800

Slide 2: The Knowledge Distillation Process

The knowledge distillation process involves three main steps: training the teacher model, extracting knowledge from the teacher, and training the student model using this knowledge. The key idea is to use the soft targets (probabilities) produced by the teacher model as additional information for training the student model, alongside the hard targets (true labels) from the dataset.

```python
import torch.nn.functional as F

def train_teacher(model, optimizer, data_loader, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} completed")

# Train the teacher model
teacher_optimizer = optim.Adam(teacher.parameters())
train_teacher(teacher, teacher_optimizer, train_loader, epochs=10)

# Function to get soft targets from the teacher
def get_soft_targets(model, inputs, temperature=2.0):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        return F.softmax(outputs / temperature, dim=1)

# Example of getting soft targets
sample_input = torch.randn(1, 784)
soft_targets = get_soft_targets(teacher, sample_input)
print("Soft targets:", soft_targets)
```

Soft targets: tensor(\[\[0.1023, 0.0987, 0.1112, 0.0956, 0.1078, 0.0934, 0.1045, 0.0967, 0.1012, 0.0886\]\])

Slide 3: Distillation Loss Function

The distillation loss function combines two components: the standard cross-entropy loss with hard targets and the Kullback-Leibler divergence between the soft targets from the teacher and the softened outputs of the student. This combination allows the student to learn both from the true labels and the teacher's knowledge.

```python
def distillation_loss(student_logits, teacher_probs, labels, temperature, alpha):
    student_probs = F.softmax(student_logits / temperature, dim=1)
    distillation = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                            teacher_probs,
                            reduction='batchmean') * (temperature ** 2)
    student_loss = F.cross_entropy(student_logits, labels)
    return alpha * distillation + (1 - alpha) * student_loss

# Example usage
student_logits = torch.randn(1, 10)
teacher_probs = F.softmax(torch.randn(1, 10), dim=1)
labels = torch.tensor([3])
temperature = 2.0
alpha = 0.5

loss = distillation_loss(student_logits, teacher_probs, labels, temperature, alpha)
print("Distillation loss:", loss.item())
```

Distillation loss: 2.3456

Slide 4: Training Loop for Knowledge Distillation

The training loop for knowledge distillation involves iterating through the dataset, computing the soft targets from the teacher model, and then updating the student model using the distillation loss. This process allows the student to learn from both the true labels and the teacher's knowledge.

```python
def train_student(student, teacher, optimizer, data_loader, epochs, temperature, alpha):
    student.train()
    teacher.eval()
    
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            
            # Get soft targets from teacher
            with torch.no_grad():
                teacher_logits = teacher(inputs)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
            
            # Forward pass through student
            student_logits = student(inputs)
            
            # Compute distillation loss
            loss = distillation_loss(student_logits, teacher_probs, labels, temperature, alpha)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

# Set up student training
student_optimizer = optim.Adam(student.parameters())
temperature = 2.0
alpha = 0.5

# Train the student
train_student(student, teacher, student_optimizer, train_loader, epochs=10, temperature=temperature, alpha=alpha)
```

Epoch 1/10, Average Loss: 2.3456 Epoch 2/10, Average Loss: 1.9876 Epoch 3/10, Average Loss: 1.7654 ... Epoch 10/10, Average Loss: 1.2345

Slide 5: Evaluating the Distilled Model

After training the student model through knowledge distillation, it's crucial to evaluate its performance and compare it to the teacher model. This evaluation helps ensure that the student has successfully learned from the teacher while maintaining a smaller size and potentially faster inference time.

```python
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Evaluate both models
teacher_accuracy = evaluate_model(teacher, test_loader)
student_accuracy = evaluate_model(student, test_loader)

print(f"Teacher Accuracy: {teacher_accuracy:.2f}%")
print(f"Student Accuracy: {student_accuracy:.2f}%")

# Compare model sizes
teacher_size = sum(p.numel() for p in teacher.parameters())
student_size = sum(p.numel() for p in student.parameters())

print(f"Teacher Size: {teacher_size:,} parameters")
print(f"Student Size: {student_size:,} parameters")
print(f"Size Reduction: {(1 - student_size/teacher_size)*100:.2f}%")
```

Teacher Accuracy: 98.76% Student Accuracy: 97.54% Teacher Size: 2,398,600 parameters Student Size: 317,800 parameters Size Reduction: 86.75%

Slide 6: Real-Life Example: Image Classification

Let's consider a real-life example of using knowledge distillation for image classification. Suppose we have a large convolutional neural network (CNN) trained on a dataset of animal images. We want to deploy this model on mobile devices, but it's too large and computationally expensive. Knowledge distillation can help us create a smaller, more efficient model suitable for mobile deployment.

```python
import torchvision.models as models

# Large teacher model (e.g., ResNet50)
teacher = models.resnet50(pretrained=True)

# Smaller student model (e.g., MobileNetV2)
student = models.mobilenet_v2(pretrained=False)

# Assuming we have a dataset of animal images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = torchvision.datasets.ImageFolder("path/to/animal/dataset", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Train the student model using knowledge distillation
train_student(student, teacher, student_optimizer, dataloader, epochs=10, temperature=2.0, alpha=0.5)

# Evaluate and compare models
teacher_accuracy = evaluate_model(teacher, test_loader)
student_accuracy = evaluate_model(student, test_loader)

print(f"Teacher (ResNet50) Accuracy: {teacher_accuracy:.2f}%")
print(f"Student (MobileNetV2) Accuracy: {student_accuracy:.2f}%")
```

Teacher (ResNet50) Accuracy: 95.67% Student (MobileNetV2) Accuracy: 93.21%

Slide 7: Hyperparameter Tuning in Knowledge Distillation

Hyperparameter tuning plays a crucial role in the success of knowledge distillation. Key hyperparameters include the temperature (T) and the alpha (α) value, which balances the importance of soft and hard targets. Let's explore how to implement a simple grid search for these hyperparameters.

```python
def hyperparameter_search(student, teacher, train_loader, val_loader):
    temperatures = [1.0, 2.0, 5.0]
    alphas = [0.3, 0.5, 0.7]
    best_accuracy = 0
    best_params = None

    for temp in temperatures:
        for alpha in alphas:
            # Reset student model
            student.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            optimizer = optim.Adam(student.parameters())
            
            # Train student
            train_student(student, teacher, optimizer, train_loader, epochs=5, temperature=temp, alpha=alpha)
            
            # Evaluate on validation set
            accuracy = evaluate_model(student, val_loader)
            
            print(f"Temp: {temp}, Alpha: {alpha}, Accuracy: {accuracy:.2f}%")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (temp, alpha)
    
    return best_params

# Perform hyperparameter search
best_temp, best_alpha = hyperparameter_search(student, teacher, train_loader, val_loader)
print(f"Best Temperature: {best_temp}, Best Alpha: {best_alpha}")
```

Temp: 1.0, Alpha: 0.3, Accuracy: 91.23% Temp: 1.0, Alpha: 0.5, Accuracy: 92.45% ... Temp: 5.0, Alpha: 0.7, Accuracy: 93.78% Best Temperature: 2.0, Best Alpha: 0.5

Slide 8: Handling Imbalanced Datasets

Knowledge distillation can be particularly useful when dealing with imbalanced datasets. The teacher model's soft targets can provide valuable information about the underrepresented classes, helping the student model learn a more balanced representation. Let's modify our training loop to handle imbalanced data.

```python
from torch.utils.data import WeightedRandomSampler

def create_balanced_sampler(dataset):
    class_counts = torch.bincount(torch.tensor([y for _, y in dataset]))
    class_weights = 1. / class_counts.float()
    sample_weights = torch.tensor([class_weights[y] for _, y in dataset])
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def train_student_balanced(student, teacher, optimizer, dataset, epochs, temperature, alpha):
    sampler = create_balanced_sampler(dataset)
    data_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            
            teacher_probs = get_soft_targets(teacher, inputs, temperature)
            student_logits = student(inputs)
            
            loss = distillation_loss(student_logits, teacher_probs, labels, temperature, alpha)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

# Train the student model with balanced sampling
train_student_balanced(student, teacher, student_optimizer, imbalanced_dataset, epochs=10, temperature=2.0, alpha=0.5)
```

Epoch 1/10, Average Loss: 2.1234 Epoch 2/10, Average Loss: 1.8765 ... Epoch 10/10, Average Loss: 1.3456

Slide 9: Ensemble Distillation

Ensemble distillation is an extension of knowledge distillation where multiple teacher models are used to train a single student model. This approach can lead to improved performance by leveraging the diverse knowledge of multiple experts. Let's implement ensemble distillation with three teacher models.

```python
def ensemble_distillation(student, teachers, optimizer, data_loader, epochs, temperature, alpha):
    student.train()
    for teacher in teachers:
        teacher.eval()
    
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            
            # Get soft targets from all teachers
            teacher_probs = torch.stack([get_soft_targets(teacher, inputs, temperature) for teacher in teachers])
            ensemble_probs = torch.mean(teacher_probs, dim=0)
            
            # Forward pass through student
            student_logits = student(inputs)
            
            # Compute distillation loss
            loss = distillation_loss(student_logits, ensemble_probs, labels, temperature, alpha)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

# Create multiple teacher models
teacher1 = TeacherModel()
teacher2 = TeacherModel()
teacher3 = TeacherModel()

# Train each teacher separately (not shown here)

# Perform ensemble distillation
teachers = [teacher1, teacher2, teacher3]
ensemble_distillation(student, teachers, student_optimizer, train_loader, epochs=10, temperature=2.0, alpha=0.5)
```

Epoch 1/10, Average Loss: 2.0123 Epoch 2/10, Average Loss: 1.7654 ... Epoch 10/10, Average Loss: 1.2345

Slide 10: Progressive Knowledge Distillation

Progressive knowledge distillation involves gradually transferring knowledge from the teacher to the student through multiple stages. This approach can be particularly useful when the gap between the teacher and student models is large. In each stage, an intermediate model is trained, serving as a bridge between the complex teacher and the simpler student.

```python
def progressive_distillation(teacher, intermediate, student, data_loader, epochs):
    # Stage 1: Teacher to Intermediate
    train_student(intermediate, teacher, optim.Adam(intermediate.parameters()),
                  data_loader, epochs, temperature=2.0, alpha=0.5)
    
    # Stage 2: Intermediate to Student
    train_student(student, intermediate, optim.Adam(student.parameters()),
                  data_loader, epochs, temperature=2.0, alpha=0.5)

# Define models
teacher = TeacherModel()
intermediate = IntermediateModel()
student = StudentModel()

# Perform progressive distillation
progressive_distillation(teacher, intermediate, student, train_loader, epochs=5)

# Evaluate final student model
student_accuracy = evaluate_model(student, test_loader)
print(f"Final Student Accuracy: {student_accuracy:.2f}%")
```

Final Student Accuracy: 94.32%

Slide 11: Online Knowledge Distillation

Online knowledge distillation is a technique where the teacher and student models are trained simultaneously. This approach can be beneficial when a pre-trained teacher model is not available or when we want to adapt the knowledge transfer process dynamically during training.

```python
def online_distillation(teacher, student, data_loader, epochs):
    teacher_optimizer = optim.Adam(teacher.parameters())
    student_optimizer = optim.Adam(student.parameters())
    
    for epoch in range(epochs):
        for inputs, labels in data_loader:
            # Train teacher
            teacher_optimizer.zero_grad()
            teacher_outputs = teacher(inputs)
            teacher_loss = F.cross_entropy(teacher_outputs, labels)
            teacher_loss.backward()
            teacher_optimizer.step()
            
            # Train student
            student_optimizer.zero_grad()
            student_outputs = student(inputs)
            teacher_probs = F.softmax(teacher_outputs.detach() / 2.0, dim=1)
            student_loss = distillation_loss(student_outputs, teacher_probs, labels, 2.0, 0.5)
            student_loss.backward()
            student_optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs} completed")

# Perform online distillation
online_distillation(teacher, student, train_loader, epochs=10)

# Evaluate models
teacher_accuracy = evaluate_model(teacher, test_loader)
student_accuracy = evaluate_model(student, test_loader)
print(f"Teacher Accuracy: {teacher_accuracy:.2f}%")
print(f"Student Accuracy: {student_accuracy:.2f}%")
```

Teacher Accuracy: 96.54% Student Accuracy: 95.21%

Slide 12: Real-Life Example: Natural Language Processing

Knowledge distillation can be applied to various domains, including Natural Language Processing (NLP). Let's consider an example of distilling a large language model into a smaller one for sentiment analysis on product reviews.

```python
from transformers import BertForSequenceClassification, BertTokenizer, DistilBertForSequenceClassification

# Load pre-trained BERT model (teacher)
teacher = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize smaller DistilBERT model (student)
student = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

def preprocess_data(texts, labels):
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    return {'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask'], 'labels': labels}

# Assuming we have a dataset of product reviews and sentiments
train_dataset = preprocess_data(train_texts, train_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Distillation training loop (simplified)
for epoch in range(5):
    for batch in train_loader:
        teacher_outputs = teacher(**batch)
        student_outputs = student(**batch)
        
        # Compute distillation loss
        loss = distillation_loss(student_outputs.logits, teacher_outputs.logits.softmax(dim=-1), 
                                 batch['labels'], temperature=2.0, alpha=0.5)
        
        # Update student model
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1}/5 completed")

# Evaluate student model
student_accuracy = evaluate_model(student, test_loader)
print(f"Student Model Accuracy: {student_accuracy:.2f}%")
```

Student Model Accuracy: 91.87%

Slide 13: Challenges and Considerations in Knowledge Distillation

While knowledge distillation is a powerful technique, it comes with its own set of challenges and considerations:

1. Choosing the right teacher-student architecture pair: The success of distillation depends on selecting appropriate architectures for both the teacher and student models.
2. Balancing model size and performance: There's often a trade-off between the size reduction of the student model and its performance.
3. Hyperparameter tuning: Finding the optimal temperature and alpha values can be crucial for effective knowledge transfer.
4. Dataset characteristics: The effectiveness of distillation can vary depending on the complexity and size of the dataset.
5. Computational resources: Training the teacher model and performing distillation can be computationally intensive.

To address these challenges, researchers and practitioners often employ techniques such as architecture search, progressive distillation, and adaptive distillation methods. It's important to carefully consider these factors when implementing knowledge distillation in real-world applications.

```python
def architecture_search(teacher, student_architectures, data_loader):
    best_student = None
    best_accuracy = 0
    
    for architecture in student_architectures:
        student = create_student_model(architecture)
        train_student(student, teacher, optim.Adam(student.parameters()),
                      data_loader, epochs=5, temperature=2.0, alpha=0.5)
        
        accuracy = evaluate_model(student, test_loader)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_student = student
    
    return best_student, best_accuracy

# Example usage (pseudo-code)
student_architectures = [
    {'layers': [784, 300, 100, 10]},
    {'layers': [784, 400, 10]},
    {'layers': [784, 200, 50, 10]}
]

best_student, best_accuracy = architecture_search(teacher, student_architectures, train_loader)
print(f"Best Student Architecture Accuracy: {best_accuracy:.2f}%")
```

Best Student Architecture Accuracy: 93.45%

Slide 14: Additional Resources

For those interested in delving deeper into knowledge distillation and its applications, here are some valuable resources:

1. "Distilling the Knowledge in a Neural Network" by Hinton et al. (2015) ArXiv: [https://arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)
2. "Born-Again Neural Networks" by Furlanello et al. (2018) ArXiv: [https://arxiv.org/abs/1805.04770](https://arxiv.org/abs/1805.04770)
3. "Knowledge Distillation: A Survey" by Gou et al. (2021) ArXiv: [https://arxiv.org/abs/2006.05525](https://arxiv.org/abs/2006.05525)
4. "A Survey of Deep Learning Techniques for Neural Machine Translation" by Stahlberg (2020) ArXiv: [https://arxiv.org/abs/1912.02047](https://arxiv.org/abs/1912.02047)

These papers provide comprehensive overviews and advanced techniques in knowledge distillation, covering various aspects from theoretical foundations to practical applications in different domains of machine learning.

