## Convolutional Neural Network Fundamentals
Slide 1: CNN Basic Architecture Implementation

A convolutional neural network implementation focusing on the fundamental building blocks using NumPy. This base architecture demonstrates the core concepts of convolution operations, activation functions, and forward propagation through multiple layers.

```python
import numpy as np

class CNN:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        # Initialize kernels with random weights
        self.conv1_kernel = np.random.randn(3, 3, input_shape[2], 16) * 0.1
        self.conv2_kernel = np.random.randn(3, 3, 16, 32) * 0.1
        
    def convolution2d(self, input_data, kernel, stride=1, padding=0):
        h_in, w_in, c_in = input_data.shape
        k_h, k_w, _, c_out = kernel.shape
        
        # Calculate output dimensions
        h_out = (h_in + 2*padding - k_h)//stride + 1
        w_out = (w_in + 2*padding - k_w)//stride + 1
        
        output = np.zeros((h_out, w_out, c_out))
        padded_data = np.pad(input_data, ((padding,padding), 
                                        (padding,padding), (0,0)))
        
        for i in range(h_out):
            for j in range(w_out):
                for k in range(c_out):
                    output[i,j,k] = np.sum(
                        padded_data[i*stride:i*stride+k_h, 
                                  j*stride:j*stride+k_w, :] * kernel[:,:,:,k]
                    )
        return output
```

Slide 2: Activation Functions and Pooling

Essential components of CNNs include activation functions for introducing non-linearity and pooling operations for reducing spatial dimensions. This implementation shows ReLU activation and max pooling operations.

```python
class CNNComponents:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def max_pooling(input_data, pool_size=2, stride=2):
        h_in, w_in, c = input_data.shape
        h_out = (h_in - pool_size)//stride + 1
        w_out = (w_in - pool_size)//stride + 1
        
        output = np.zeros((h_out, w_out, c))
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                h_end = h_start + pool_size
                w_start = j * stride
                w_end = w_start + pool_size
                
                output[i,j,:] = np.max(
                    input_data[h_start:h_end, w_start:w_end, :],
                    axis=(0,1)
                )
        return output

# Example usage
input_data = np.random.randn(28, 28, 1)
cnn_comp = CNNComponents()
activated = cnn_comp.relu(input_data)
pooled = cnn_comp.max_pooling(activated)
print(f"Input shape: {input_data.shape}")
print(f"Pooled shape: {pooled.shape}")
```

Slide 3: Forward Propagation Implementation

Forward propagation in CNNs involves sequential application of convolution, activation, and pooling operations. This implementation demonstrates the complete forward pass through multiple layers of the network.

```python
class CNNForward(CNN):
    def __init__(self, input_shape):
        super().__init__(input_shape)
        
    def forward(self, x):
        # First convolution layer
        conv1 = self.convolution2d(x, self.conv1_kernel, 
                                 stride=1, padding=1)
        relu1 = CNNComponents.relu(conv1)
        pool1 = CNNComponents.max_pooling(relu1)
        
        # Second convolution layer
        conv2 = self.convolution2d(pool1, self.conv2_kernel, 
                                 stride=1, padding=1)
        relu2 = CNNComponents.relu(conv2)
        pool2 = CNNComponents.max_pooling(relu2)
        
        # Store intermediate outputs for backpropagation
        self.cache = {
            'conv1': conv1, 'relu1': relu1, 'pool1': pool1,
            'conv2': conv2, 'relu2': relu2, 'pool2': pool2
        }
        
        return pool2

# Example usage
cnn = CNNForward((28, 28, 1))
input_image = np.random.randn(28, 28, 1)
output = cnn.forward(input_image)
print(f"Output shape: {output.shape}")
```

Slide 4: Loss Function and Gradient Computation

The implementation of loss computation and gradient calculation is crucial for training CNNs. This code demonstrates the categorical cross-entropy loss and its gradient computation for backpropagation.

```python
def categorical_crossentropy(predictions, targets):
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    N = predictions.shape[0]
    ce_loss = -np.sum(targets * np.log(predictions)) / N
    ce_gradient = predictions - targets
    return ce_loss, ce_gradient

class LossComputation:
    def compute_gradients(self, output, target):
        """
        Compute gradients for backpropagation
        """
        # Softmax activation for output layer
        exp_scores = np.exp(output)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Compute loss and gradient
        N = output.shape[0]
        loss, gradient = categorical_crossentropy(probs, target)
        
        return loss, gradient / N

# Example usage
output = np.random.randn(10, 10)  # 10 samples, 10 classes
target = np.eye(10)  # One-hot encoded targets
loss_computer = LossComputation()
loss, gradient = loss_computer.compute_gradients(output, target)
print(f"Loss: {loss:.4f}")
```

Slide 5: Backpropagation Through Convolution Layers

A detailed implementation of backpropagation through convolutional layers, showing how gradients flow backward through the network to update weights. This process is essential for training CNNs effectively.

```python
class CNNBackprop(CNNForward):
    def backward(self, gradient):
        # Gradient of pool2
        dpool2 = self.pool_backward(gradient, self.cache['relu2'],
                                  pool_size=2)
        # Gradient of relu2
        drelu2 = self.relu_backward(dpool2, self.cache['conv2'])
        # Gradient of conv2
        dconv2, self.dconv2_kernel = self.conv_backward(
            drelu2, self.cache['pool1'], self.conv2_kernel
        )
        
        # Gradient of pool1
        dpool1 = self.pool_backward(dconv2, self.cache['relu1'],
                                  pool_size=2)
        # Gradient of relu1
        drelu1 = self.relu_backward(dpool1, self.cache['conv1'])
        # Gradient of conv1
        dconv1, self.dconv1_kernel = self.conv_backward(
            drelu1, self.input_data, self.conv1_kernel
        )
        
        return dconv1
    
    def conv_backward(self, dout, cache, kernel):
        x_pad = np.pad(cache, ((1,1), (1,1), (0,0)))
        dx = np.zeros_like(cache)
        dw = np.zeros_like(kernel)
        
        for i in range(dout.shape[0]):
            for j in range(dout.shape[1]):
                dx[i:i+3, j:j+3] += np.sum(
                    kernel * dout[i,j], axis=-1
                )
                dw += x_pad[i:i+3, j:j+3].reshape(
                    3,3,-1,1) * dout[i,j]
        
        return dx, dw
```

Slide 6: Training Loop Implementation

The complete training loop implementation incorporates batch processing, optimizer updates, and learning rate scheduling. This code demonstrates how to train a CNN model with mini-batch gradient descent and momentum optimization.

```python
class CNNTrainer:
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        self.model = model
        self.lr = learning_rate
        self.momentum = momentum
        self.v_conv1 = np.zeros_like(model.conv1_kernel)
        self.v_conv2 = np.zeros_like(model.conv2_kernel)
    
    def train_step(self, X_batch, y_batch):
        batch_size = X_batch.shape[0]
        loss = 0
        
        # Forward pass
        output = self.model.forward(X_batch)
        
        # Compute loss and gradients
        loss_computer = LossComputation()
        batch_loss, gradient = loss_computer.compute_gradients(
            output, y_batch
        )
        
        # Backward pass
        dx = self.model.backward(gradient)
        
        # Update weights with momentum
        self.v_conv1 = (self.momentum * self.v_conv1 - 
                       self.lr * self.model.dconv1_kernel)
        self.v_conv2 = (self.momentum * self.v_conv2 - 
                       self.lr * self.model.dconv2_kernel)
        
        self.model.conv1_kernel += self.v_conv1
        self.model.conv2_kernel += self.v_conv2
        
        return batch_loss

# Training loop example
trainer = CNNTrainer(model=CNNBackprop((28, 28, 1)))
for epoch in range(10):
    epoch_loss = trainer.train_step(
        X_batch=np.random.randn(32, 28, 28, 1),
        y_batch=np.eye(32)
    )
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
```

Slide 7: Data Processing and Augmentation

Implementation of data preprocessing and augmentation techniques crucial for improving CNN performance. This code shows image normalization, random rotations, flips, and other transformations.

```python
import cv2
from scipy.ndimage import rotate

class DataAugmentation:
    def __init__(self, rotation_range=20, flip_prob=0.5):
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
    
    def normalize_image(self, image):
        """Normalize image to range [0,1]"""
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    
    def random_rotation(self, image):
        """Apply random rotation"""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        return rotate(image, angle, reshape=False)
    
    def random_flip(self, image):
        """Apply random horizontal flip"""
        if np.random.random() < self.flip_prob:
            return np.fliplr(image)
        return image
    
    def augment(self, image):
        """Apply all augmentations"""
        image = self.normalize_image(image)
        image = self.random_rotation(image)
        image = self.random_flip(image)
        return image

# Example usage
augmenter = DataAugmentation()
sample_image = np.random.randn(28, 28, 1)
augmented = augmenter.augment(sample_image)
print(f"Original shape: {sample_image.shape}")
print(f"Augmented shape: {augmented.shape}")
```

Slide 8: CNN Image Classification Example

A complete example of using CNN for image classification, including data preparation, model training, and evaluation. This implementation demonstrates the practical application of CNNs for real-world image recognition tasks.

```python
class ImageClassifier(CNNBackprop):
    def __init__(self, input_shape, num_classes):
        super().__init__(input_shape)
        self.num_classes = num_classes
        self.fc = np.random.randn(
            32 * (input_shape[0]//4) * (input_shape[1]//4),
            num_classes
        ) * 0.1
    
    def classify(self, image):
        # Forward pass through conv layers
        features = self.forward(image)
        
        # Flatten and pass through FC layer
        flattened = features.reshape(features.shape[0], -1)
        scores = np.dot(flattened, self.fc)
        
        # Softmax activation
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probs

# Example usage with MNIST-like data
classifier = ImageClassifier((28, 28, 1), num_classes=10)
test_image = np.random.randn(1, 28, 28, 1)
predictions = classifier.classify(test_image)
predicted_class = np.argmax(predictions)
print(f"Predicted class: {predicted_class}")
print(f"Class probabilities:\n{predictions[0]}")
```

Slide 9: Model Evaluation Metrics

Implementation of comprehensive evaluation metrics for CNN models, including accuracy, precision, recall, and F1-score calculations. This code provides essential tools for assessing model performance.

```python
class ModelEvaluator:
    @staticmethod
    def compute_metrics(y_true, y_pred):
        """
        Compute classification metrics
        """
        # Convert predictions to class labels
        pred_classes = np.argmax(y_pred, axis=1)
        true_classes = np.argmax(y_true, axis=1)
        
        # Accuracy
        accuracy = np.mean(pred_classes == true_classes)
        
        # Per-class metrics
        metrics = {}
        for class_idx in range(y_true.shape[1]):
            # True positives, false positives, false negatives
            tp = np.sum((pred_classes == class_idx) & 
                       (true_classes == class_idx))
            fp = np.sum((pred_classes == class_idx) & 
                       (true_classes != class_idx))
            fn = np.sum((pred_classes != class_idx) & 
                       (true_classes == class_idx))
            
            # Precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall / 
                 (precision + recall)) if (precision + recall) > 0 else 0
            
            metrics[f"class_{class_idx}"] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        metrics["accuracy"] = accuracy
        return metrics

# Example usage
evaluator = ModelEvaluator()
y_true = np.eye(10)[np.random.randint(0, 10, 100)]
y_pred = np.random.random((100, 10))
metrics = evaluator.compute_metrics(y_true, y_pred)
print("Model Performance Metrics:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

Slide 10: Real-world Application: Face Detection CNN

Implementation of a face detection system using CNN architecture. This practical example demonstrates handling real image data, preprocessing, and detection pipeline implementation for face recognition tasks.

```python
import numpy as np

class FaceDetectionCNN:
    def __init__(self, input_size=(64, 64)):
        self.input_size = input_size
        self.detection_threshold = 0.8
        
        # Initialize specialized kernels for face features
        self.edge_kernel = np.random.randn(3, 3, 1, 16) * 0.1
        self.feature_kernel = np.random.randn(3, 3, 16, 32) * 0.1
        self.face_kernel = np.random.randn(3, 3, 32, 64) * 0.1
    
    def preprocess_image(self, image):
        # Convert to grayscale if colored
        if len(image.shape) == 3:
            image = np.mean(image, axis=2, keepdims=True)
        
        # Resize to input size
        image = self._resize_image(image, self.input_size)
        
        # Normalize
        image = (image - np.mean(image)) / np.std(image)
        return image
    
    def sliding_window_detect(self, image, window_size=(64, 64), stride=32):
        detections = []
        h, w = image.shape[:2]
        
        for y in range(0, h - window_size[0], stride):
            for x in range(0, w - window_size[1], stride):
                window = image[y:y+window_size[0], x:x+window_size[1]]
                if window.shape[:2] == window_size:
                    score = self._evaluate_window(window)
                    if score > self.detection_threshold:
                        detections.append((x, y, score))
        
        return self._non_max_suppression(detections)
    
    def _evaluate_window(self, window):
        # Forward pass through specialized layers
        x = self.convolution2d(window, self.edge_kernel)
        x = self.relu(x)
        x = self.max_pooling(x)
        
        x = self.convolution2d(x, self.feature_kernel)
        x = self.relu(x)
        x = self.max_pooling(x)
        
        x = self.convolution2d(x, self.face_kernel)
        x = self.relu(x)
        
        # Final confidence score
        return np.mean(x)

# Example usage
detector = FaceDetectionCNN()
sample_image = np.random.randn(256, 256)
processed = detector.preprocess_image(sample_image)
detections = detector.sliding_window_detect(processed)
print(f"Found {len(detections)} potential faces")
```

Slide 11: Transfer Learning Implementation

A comprehensive implementation of transfer learning capabilities for CNNs, allowing the reuse of pre-trained weights and fine-tuning for specific tasks. This approach significantly reduces training time and improves performance on limited datasets.

```python
class TransferLearningCNN:
    def __init__(self, base_model, num_new_classes, frozen_layers=None):
        self.base_model = base_model
        self.num_new_classes = num_new_classes
        self.frozen_layers = frozen_layers or []
        
        # Initialize new classification head
        self.new_head = self._create_new_head()
        
    def _create_new_head(self):
        """Create new classification layers"""
        return {
            'fc1': np.random.randn(512, 256) * 0.1,
            'fc2': np.random.randn(256, self.num_new_classes) * 0.1,
            'bn1': {'mean': np.zeros(256), 'var': np.ones(256)},
            'bn2': {'mean': np.zeros(self.num_new_classes), 
                   'var': np.ones(self.num_new_classes)}
        }
    
    def freeze_layers(self):
        """Freeze specified layers during training"""
        for layer_name in self.frozen_layers:
            if hasattr(self.base_model, layer_name):
                layer = getattr(self.base_model, layer_name)
                layer.trainable = False
    
    def unfreeze_layers(self):
        """Unfreeze all layers for fine-tuning"""
        for layer_name in self.frozen_layers:
            if hasattr(self.base_model, layer_name):
                layer = getattr(self.base_model, layer_name)
                layer.trainable = True
    
    def forward(self, x):
        # Get features from base model
        features = self.base_model.forward(x)
        
        # Pass through new classification head
        x = self.dense_forward(features, self.new_head['fc1'])
        x = self.batch_norm(x, self.new_head['bn1'])
        x = self.relu(x)
        
        x = self.dense_forward(x, self.new_head['fc2'])
        x = self.batch_norm(x, self.new_head['bn2'])
        
        return self.softmax(x)

# Example usage
base_cnn = CNNBackprop((224, 224, 3))
transfer_model = TransferLearningCNN(
    base_model=base_cnn,
    num_new_classes=5,
    frozen_layers=['conv1', 'conv2']
)

# Test forward pass
test_input = np.random.randn(1, 224, 224, 3)
predictions = transfer_model.forward(test_input)
print(f"Output shape: {predictions.shape}")
```

Slide 12: Attention Mechanism in CNNs

Implementation of attention mechanisms in CNNs to focus on relevant features in the input. This advanced technique improves model performance by learning to weight important spatial locations differently.

```python
class AttentionCNN:
    def __init__(self, input_shape, num_heads=8):
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.attention_dim = 64
        
        # Initialize attention parameters
        self.query_conv = np.random.randn(1, 1, input_shape[-1], 
                                        self.attention_dim) * 0.1
        self.key_conv = np.random.randn(1, 1, input_shape[-1], 
                                      self.attention_dim) * 0.1
        self.value_conv = np.random.randn(1, 1, input_shape[-1], 
                                        self.attention_dim) * 0.1
    
    def attention_forward(self, x):
        batch_size, h, w, c = x.shape
        
        # Generate Q, K, V
        queries = self.convolution2d(x, self.query_conv)
        keys = self.convolution2d(x, self.key_conv)
        values = self.convolution2d(x, self.value_conv)
        
        # Reshape for multi-head attention
        queries = self._reshape_multihead(queries)
        keys = self._reshape_multihead(keys)
        values = self._reshape_multihead(values)
        
        # Compute attention scores
        scores = np.matmul(queries, keys.transpose(0, 1, 3, 2))
        scores = scores / np.sqrt(self.attention_dim // self.num_heads)
        attention_weights = self.softmax(scores)
        
        # Apply attention
        out = np.matmul(attention_weights, values)
        out = self._reshape_output(out, h, w)
        
        return out, attention_weights
    
    def _reshape_multihead(self, x):
        batch_size, h, w, c = x.shape
        x = x.reshape(batch_size, h*w, self.num_heads, -1)
        return x.transpose(0, 2, 1, 3)
    
    def _reshape_output(self, x, h, w):
        batch_size = x.shape[0]
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, h, w, -1)

# Example usage
attention_cnn = AttentionCNN((28, 28, 64))
feature_map = np.random.randn(1, 28, 28, 64)
output, attention_weights = attention_cnn.attention_forward(feature_map)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
```

Slide 13: Visualization and Interpretability

Implementation of visualization techniques for understanding CNN decisions, including activation maps, gradient-based saliency, and class activation mapping (CAM) to provide insights into model behavior.

```python
class CNNVisualizer:
    def __init__(self, model):
        self.model = model
        
    def compute_activation_maps(self, input_image):
        """Generate activation maps for each conv layer"""
        activations = {}
        x = input_image
        
        # Forward pass storing activations
        for layer_name, layer in self.model.layers.items():
            if 'conv' in layer_name:
                x = self.model.forward_layer(x, layer)
                activations[layer_name] = np.mean(x, axis=-1)
        
        return activations
    
    def compute_gradcam(self, input_image, target_class):
        """Compute Grad-CAM visualization"""
        # Forward pass
        conv_outputs = {}
        x = input_image
        
        def save_conv_output(layer_output, layer_name):
            conv_outputs[layer_name] = layer_output
        
        # Get final conv layer activations and gradients
        final_conv_output = self.model.forward_with_activation_hook(
            x, save_conv_output)
        
        # Calculate gradients
        grads = self.model.backward_to_conv(target_class)
        
        # Global average pooling of gradients
        weights = np.mean(grads, axis=(0, 1))
        
        # Compute weighted combination of forward activation maps
        cam = np.zeros(conv_outputs['final_conv'].shape[:-1])
        for i, w in enumerate(weights):
            cam += w * conv_outputs['final_conv'][..., i]
        
        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        
        return cam
    
    def visualize_filters(self, layer_name):
        """Visualize convolutional filters"""
        layer = self.model.layers[layer_name]
        filters = layer.weights
        
        # Normalize filters for visualization
        normalized_filters = []
        for i in range(filters.shape[-1]):
            filt = filters[..., i]
            filt = (filt - np.min(filt)) / (np.max(filt) - np.min(filt))
            normalized_filters.append(filt)
            
        return np.array(normalized_filters)

# Example usage
model = CNN((224, 224, 3))
visualizer = CNNVisualizer(model)

# Generate visualizations
sample_image = np.random.randn(224, 224, 3)
activation_maps = visualizer.compute_activation_maps(sample_image)
gradcam = visualizer.compute_gradcam(sample_image, target_class=0)
filters = visualizer.visualize_filters('conv1')

print("Activation maps shapes:")
for layer, act_map in activation_maps.items():
    print(f"{layer}: {act_map.shape}")
print(f"Grad-CAM shape: {gradcam.shape}")
print(f"Filter visualizations shape: {filters.shape}")
```

Slide 14: Advanced Loss Functions

Implementation of specialized loss functions for CNN training, including focal loss for handling class imbalance and contrastive loss for similarity learning tasks.

```python
class AdvancedLossFunctions:
    def focal_loss(self, y_pred, y_true, gamma=2.0, alpha=0.25):
        """
        Focal Loss implementation for handling class imbalance
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * np.log(y_pred)
        
        # Calculate focal term
        focal_term = np.power(1 - y_pred, gamma)
        
        # Calculate focal loss
        focal_loss = alpha * focal_term * cross_entropy
        
        return np.mean(focal_loss)
    
    def contrastive_loss(self, embeddings1, embeddings2, labels, margin=1.0):
        """
        Contrastive Loss for similarity learning
        """
        # Calculate euclidean distance
        distances = np.sqrt(np.sum(
            np.square(embeddings1 - embeddings2), axis=1))
        
        # Calculate loss for similar and dissimilar pairs
        similar_loss = labels * np.square(distances)
        dissimilar_loss = (1 - labels) * np.square(
            np.maximum(0, margin - distances))
        
        # Combine losses
        loss = np.mean(similar_loss + dissimilar_loss)
        
        return loss, distances
    
    def center_loss(self, features, labels, centers, alpha=0.5):
        """
        Center Loss for deep feature learning
        """
        num_classes = centers.shape[0]
        batch_size = features.shape[0]
        
        # Calculate distances to centers
        distances = np.zeros((batch_size, num_classes))
        for i in range(num_classes):
            distances[:, i] = np.sum(
                np.square(features - centers[i]), axis=1)
        
        # Calculate loss
        mask = np.zeros_like(distances)
        mask[np.arange(batch_size), labels] = 1
        loss = np.sum(distances * mask) / batch_size
        
        # Update centers
        for i in range(num_classes):
            class_features = features[labels == i]
            if len(class_features) > 0:
                centers[i] = (1 - alpha) * centers[i] + \
                           alpha * np.mean(class_features, axis=0)
        
        return loss, centers

# Example usage
loss_functions = AdvancedLossFunctions()

# Test focal loss
predictions = np.random.random((100, 10))
targets = np.eye(10)[np.random.randint(0, 10, 100)]
focal_loss = loss_functions.focal_loss(predictions, targets)

# Test contrastive loss
emb1 = np.random.randn(32, 128)
emb2 = np.random.randn(32, 128)
pair_labels = np.random.randint(0, 2, 32)
cont_loss, distances = loss_functions.contrastive_loss(
    emb1, emb2, pair_labels)

print(f"Focal Loss: {focal_loss:.4f}")
print(f"Contrastive Loss: {cont_loss:.4f}")
```

Slide 15: Additional Resources

*   "Deep Residual Learning for Image Recognition" [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
*   "Squeeze-and-Excitation Networks" [https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507)
*   "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
*   "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)
*   "Focal Loss for Dense Object Detection" [https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)

