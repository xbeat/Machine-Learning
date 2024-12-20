## Implementing CNN Image Classification with PyTorch
Slide 1: Introduction to CNNs and PyTorch

Convolutional Neural Networks (CNNs) are a powerful class of deep learning models particularly effective for image classification tasks. PyTorch, a popular deep learning framework, provides an intuitive way to implement CNNs. This slideshow will guide you through the process of creating a CNN for image classification using PyTorch and Python.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

Slide 2: Preparing the Dataset

Before building our CNN, we need to prepare our dataset. We'll use the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes. PyTorch provides convenient methods to load and preprocess this dataset.

```python
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
```

Slide 3: Defining the CNN Architecture

Now, let's define our CNN architecture. We'll create a simple CNN with two convolutional layers followed by three fully connected layers.

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().to(device)
```

Slide 4: Loss Function and Optimizer

To train our CNN, we need to define a loss function and an optimizer. We'll use Cross-Entropy Loss and Stochastic Gradient Descent (SGD) optimizer.

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

Slide 5: Training the CNN

Now, let's train our CNN. We'll iterate over our dataset multiple times (epochs), and in each epoch, we'll perform forward and backward passes to update our model's parameters.

```python
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
```

Slide 6: Evaluating the Model

After training, we need to evaluate our model's performance on the test set to see how well it generalizes to unseen data.

```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
```

Slide 7: Class-wise Accuracy

Let's analyze the model's performance for each class to identify any biases or weaknesses.

```python
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print(f'Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]}%')
```

Slide 8: Visualizing Convolutional Filters

Understanding what our CNN has learned can be challenging. One way to gain insight is by visualizing the filters in the convolutional layers.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_filters(model, layer_num, single_channel=True, collated=False):
    filters = model.conv1.weight.data.cpu().numpy()
    if single_channel:
        if collated:
            filters = filters.reshape(filters.shape[0]*filters.shape[1], filters.shape[2], filters.shape[3])
        else:
            filters = filters[:,0,:,:]
    n_filters = filters.shape[0]
    ix = 1
    for i in range(n_filters):
        f = filters[i]
        ax = plt.subplot(n_filters//8 + 1, 8, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f, cmap='gray')
        ix += 1
    plt.show()

plot_filters(net, 0)
```

Slide 9: Feature Maps Visualization

Another way to understand our CNN is by visualizing the feature maps, which show how the input is transformed as it passes through the network.

```python
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

activation = {}
net.conv1.register_forward_hook(get_activation('conv1'))
net.conv2.register_forward_hook(get_activation('conv2'))

dataiter = iter(testloader)
images, labels = next(dataiter)

output = net(images.to(device))

plt.imshow(images[0].permute(1, 2, 0))
plt.show()

plt.imshow(activation['conv1'][0, 0].cpu(), cmap='viridis')
plt.show()

plt.imshow(activation['conv2'][0, 0].cpu(), cmap='viridis')
plt.show()
```

Slide 10: Transfer Learning

Transfer learning allows us to leverage pre-trained models on large datasets to improve performance on smaller, similar datasets. Let's use a pre-trained ResNet model for our CIFAR-10 classification task.

```python
import torchvision.models as models

# Load pre-trained ResNet
resnet = models.resnet18(pretrained=True)

# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace the last fully connected layer
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)

# Move model to device
resnet = resnet.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
```

Slide 11: Data Augmentation

Data augmentation is a technique to increase the diversity of your training set by applying random transformations. This can help improve model generalization and reduce overfitting.

```python
# Define augmented transformations
augmented_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset with augmented transformations
augmented_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=augmented_transform)
augmented_trainloader = torch.utils.data.DataLoader(augmented_trainset, batch_size=4,
                                                    shuffle=True, num_workers=2)

# Visualize augmented images
dataiter = iter(augmented_trainloader)
images, labels = next(dataiter)

plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i].permute(1, 2, 0))
    plt.title(classes[labels[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

Slide 12: Real-life Example: Pet Breed Classification

One practical application of CNNs is pet breed classification. This can be used in animal shelters to automatically identify dog or cat breeds from photos, helping in the adoption process.

```python
# Assume we have a pre-trained model for pet breed classification
class PetBreedClassifier(nn.Module):
    def __init__(self, num_breeds):
        super(PetBreedClassifier, self).__init__()
        self.features = models.resnet50(pretrained=True)
        num_ftrs = self.features.fc.in_features
        self.features.fc = nn.Linear(num_ftrs, num_breeds)

    def forward(self, x):
        return self.features(x)

# Load the model
num_breeds = 120  # Example: 120 dog breeds
model = PetBreedClassifier(num_breeds).to(device)
model.load_state_dict(torch.load('pet_breed_classifier.pth'))
model.eval()

# Function to predict breed
def predict_breed(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()

# Example usage
image_path = 'golden_retriever.jpg'
breed_index = predict_breed(image_path, model, device)
print(f"Predicted breed: {breeds[breed_index]}")
```

Slide 13: Real-life Example: Plant Disease Detection

Another practical application of CNNs is in agriculture for plant disease detection. This can help farmers quickly identify and address crop diseases, potentially saving entire harvests.

```python
# Assume we have a pre-trained model for plant disease detection
class PlantDiseaseDetector(nn.Module):
    def __init__(self, num_diseases):
        super(PlantDiseaseDetector, self).__init__()
        self.features = models.densenet121(pretrained=True)
        num_ftrs = self.features.classifier.in_features
        self.features.classifier = nn.Linear(num_ftrs, num_diseases)

    def forward(self, x):
        return self.features(x)

# Load the model
num_diseases = 38  # Example: 38 different plant diseases
model = PlantDiseaseDetector(num_diseases).to(device)
model.load_state_dict(torch.load('plant_disease_detector.pth'))
model.eval()

# Function to detect disease
def detect_disease(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()

# Example usage
image_path = 'tomato_leaf.jpg'
disease_index = detect_disease(image_path, model, device)
print(f"Detected disease: {diseases[disease_index]}")
```

Slide 14: Additional Resources

For further exploration of CNNs and PyTorch, consider these resources:

1. "ImageNet Classification with Deep Convolutional Neural Networks" by Krizhevsky et al. (2012): A seminal paper on CNNs for image classification. (arXiv:1207.0580)
2. "Very Deep Convolutional Networks for Large-Scale Image Recognition" by Simonyan and Zisserman (2014): Introduces the VGG network architecture. (arXiv:1409.1556)
3. "Deep Residual Learning for Image Recognition" by He et al. (2015): Presents the ResNet architecture, which allows training of very deep networks. (arXiv:1512.03385)
4. PyTorch documentation ([https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)): Comprehensive guide to PyTorch's features and APIs.
5. "Visualizing and Understanding Convolutional Networks" by Zeiler and Fergus (2013): Provides techniques for visualizing CNN features. (arXiv:1311.2901)

