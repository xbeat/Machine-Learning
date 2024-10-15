## Visualizing CNN Decision-Making with Grad-CAM
Slide 1: Introduction to Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) is a powerful technique for visualizing and understanding the decision-making process of convolutional neural networks (CNNs). It helps identify which regions of an input image are most important for the model's predictions.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def grad_cam(model, img_array, layer_name, class_index):
    # Create a model that maps the input image to the activations
    # of the last convolutional layer and the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    
    # Rest of the Grad-CAM implementation will follow in subsequent slides
```

Slide 2: Preparing the Input

Before applying Grad-CAM, we need to prepare our input image and model. This involves loading and preprocessing the image, as well as ensuring our model is ready for inference.

```python
# Load and preprocess the image
img_path = 'path/to/your/image.jpg'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

# Load a pre-trained model (e.g., ResNet50)
model = tf.keras.applications.ResNet50(weights='imagenet')

# Choose the last convolutional layer
layer_name = 'conv5_block3_out'
```

Slide 3: Computing Gradients

The core of Grad-CAM involves computing the gradients of the output with respect to the feature maps of a specific convolutional layer. This helps us understand which features are most important for the prediction.

```python
def compute_gradients(grad_model, img_array, class_index):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    return conv_outputs, grads

# Assume we're interested in the top predicted class
class_index = tf.argmax(model.predict(img_array)[0])
conv_outputs, grads = compute_gradients(grad_model, img_array, class_index)
```

Slide 4: Calculating Class Activation Map

Once we have the gradients, we can calculate the class activation map. This involves taking the global average pool of the gradients and using it to weight the feature maps.

```python
def calculate_cam(conv_outputs, grads):
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    cam = tf.nn.relu(cam)  # ReLU to only show positive influences
    return cam

cam = calculate_cam(conv_outputs[0], grads[0])
```

Slide 5: Visualizing the Heatmap

To make the class activation map interpretable, we need to resize it to match the input image dimensions and overlay it on the original image.

```python
def create_heatmap(cam, img):
    cam = cv2.resize(cam.numpy(), (img.shape[1], img.shape[0]))
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed = heatmap * 0.4 + img
    return superimposed / 255.0

heatmap = create_heatmap(cam, img)
plt.imshow(heatmap)
plt.axis('off')
plt.show()
```

Slide 6: Interpreting Grad-CAM Results

Grad-CAM produces a heatmap that highlights the regions of the input image that most strongly influence the model's prediction for a specific class. Red areas indicate high importance, while blue areas are less important.

```python
def interpret_prediction(model, img_array, class_index):
    predictions = model.predict(img_array)
    predicted_class = tf.keras.applications.resnet50.decode_predictions(predictions, top=1)[0][0]
    class_name = predicted_class[1]
    confidence = predicted_class[2]
    
    print(f"Predicted class: {class_name}")
    print(f"Confidence: {confidence:.2f}")

interpret_prediction(model, img_array, class_index)
```

Slide 7: Real-Life Example: Object Detection

Let's apply Grad-CAM to a real-life scenario of object detection in an image of a city street.

```python
img_path = 'path/to/street_image.jpg'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

class_index = tf.argmax(model.predict(img_array)[0])
conv_outputs, grads = compute_gradients(grad_model, img_array, class_index)
cam = calculate_cam(conv_outputs[0], grads[0])
heatmap = create_heatmap(cam, img)

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(heatmap)
plt.title('Grad-CAM Heatmap')
plt.axis('off')

plt.show()

interpret_prediction(model, img_array, class_index)
```

Slide 8: Understanding Model Focus

By examining the Grad-CAM heatmap, we can see which parts of the image the model is focusing on to make its prediction. This can help us understand if the model is using relevant features or if it's being influenced by irrelevant background elements.

```python
def analyze_focus(heatmap, threshold=0.5):
    high_focus = np.mean(heatmap > threshold)
    print(f"Percentage of image with high focus: {high_focus:.2%}")
    
    if high_focus > 0.7:
        print("The model is focusing on a large portion of the image.")
    elif high_focus > 0.3:
        print("The model is focusing on specific regions of the image.")
    else:
        print("The model is highly focused on small, specific areas.")

analyze_focus(heatmap)
```

Slide 9: Comparing Multiple Classes

Grad-CAM can be used to compare how the model focuses on different classes within the same image. This is particularly useful for understanding multi-class classification problems.

```python
def compare_classes(model, img_array, class_indices):
    fig, axes = plt.subplots(1, len(class_indices), figsize=(15, 5))
    
    for i, class_index in enumerate(class_indices):
        conv_outputs, grads = compute_gradients(grad_model, img_array, class_index)
        cam = calculate_cam(conv_outputs[0], grads[0])
        heatmap = create_heatmap(cam, img)
        
        axes[i].imshow(heatmap)
        axes[i].set_title(f"Class {class_index}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

top_3_classes = tf.argsort(model.predict(img_array)[0])[-3:]
compare_classes(model, img_array, top_3_classes)
```

Slide 10: Grad-CAM for Model Debugging

Grad-CAM can be a powerful tool for debugging and improving neural networks. By visualizing what the model is focusing on, we can identify potential biases or mistakes in the model's decision-making process.

```python
def debug_model(model, img_array, expected_class):
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions[0])
    
    if predicted_class != expected_class:
        print("Model prediction doesn't match expected class.")
        print("Analyzing model focus...")
        
        conv_outputs, grads = compute_gradients(grad_model, img_array, predicted_class)
        cam = calculate_cam(conv_outputs[0], grads[0])
        heatmap = create_heatmap(cam, img)
        
        plt.imshow(heatmap)
        plt.title(f"Focus for predicted class {predicted_class}")
        plt.axis('off')
        plt.show()
        
        print("Check if the model is focusing on relevant features.")
    else:
        print("Model prediction matches expected class.")

debug_model(model, img_array, expected_class=242)  # 242 is the class index for 'bull mastiff' in ImageNet
```

Slide 11: Grad-CAM for Different Network Architectures

Grad-CAM can be applied to various CNN architectures. Here's an example of how to use it with a different model, such as VGG16.

```python
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

vgg_model = VGG16(weights='imagenet')
vgg_layer_name = 'block5_conv3'

vgg_grad_model = tf.keras.models.Model(
    [vgg_model.inputs], [vgg_model.get_layer(vgg_layer_name).output, vgg_model.output]
)

img_array = preprocess_input(img_array)
class_index = tf.argmax(vgg_model.predict(img_array)[0])

conv_outputs, grads = compute_gradients(vgg_grad_model, img_array, class_index)
cam = calculate_cam(conv_outputs[0], grads[0])
heatmap = create_heatmap(cam, img)

plt.imshow(heatmap)
plt.title("Grad-CAM on VGG16")
plt.axis('off')
plt.show()

print(decode_predictions(vgg_model.predict(img_array), top=1)[0])
```

Slide 12: Real-Life Example: Medical Imaging

Let's apply Grad-CAM to a medical imaging scenario, such as identifying pneumonia in chest X-rays. This example demonstrates how Grad-CAM can be used to enhance interpretability in critical applications like healthcare.

```python
# Assume we have a pre-trained model for pneumonia detection
pneumonia_model = tf.keras.models.load_model('path/to/pneumonia_model.h5')

# Load and preprocess a chest X-ray image
xray_path = 'path/to/chest_xray.jpg'
xray = tf.keras.preprocessing.image.load_img(xray_path, target_size=(224, 224))
xray_array = tf.keras.preprocessing.image.img_to_array(xray)
xray_array = np.expand_dims(xray_array, axis=0)

# Apply Grad-CAM
pneumonia_grad_model = tf.keras.models.Model(
    [pneumonia_model.inputs], 
    [pneumonia_model.get_layer('conv5_block3_out').output, pneumonia_model.output]
)

class_index = 1  # Assume 1 represents pneumonia
conv_outputs, grads = compute_gradients(pneumonia_grad_model, xray_array, class_index)
cam = calculate_cam(conv_outputs[0], grads[0])
heatmap = create_heatmap(cam, xray)

plt.imshow(heatmap, cmap='gray')
plt.title("Pneumonia Detection Heatmap")
plt.axis('off')
plt.show()

prediction = pneumonia_model.predict(xray_array)[0][0]
print(f"Probability of pneumonia: {prediction:.2%}")
```

Slide 13: Limitations and Considerations

While Grad-CAM is a powerful tool, it's important to be aware of its limitations:

1. It only works for CNNs and may not be suitable for other architectures.
2. The resolution of the heatmap is limited by the size of the feature maps in the chosen layer.
3. It may not capture fine-grained details or complex relationships between features.
4. The choice of the convolutional layer can significantly affect the results.

```python
def grad_cam_resolution_demo(model, img_array, layer_names):
    fig, axes = plt.subplots(1, len(layer_names), figsize=(15, 5))
    
    for i, layer_name in enumerate(layer_names):
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )
        
        conv_outputs, grads = compute_gradients(grad_model, img_array, class_index)
        cam = calculate_cam(conv_outputs[0], grads[0])
        heatmap = create_heatmap(cam, img)
        
        axes[i].imshow(heatmap)
        axes[i].set_title(f"Layer: {layer_name}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

layer_names = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
grad_cam_resolution_demo(model, img_array, layer_names)
```

Slide 14: Additional Resources

For those interested in diving deeper into Grad-CAM and related techniques, here are some valuable resources:

1. Original Grad-CAM paper: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" by Selvaraju et al. (2017) ArXiv link: [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)
2. "Axiomatic Attribution for Deep Networks" by Sundararajan et al. (2017) ArXiv link: [https://arxiv.org/abs/1703.01365](https://arxiv.org/abs/1703.01365)
3. "Sanity Checks for Saliency Maps" by Adebayo et al. (2018) ArXiv link: [https://arxiv.org/abs/1810.03292](https://arxiv.org/abs/1810.03292)
4. "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks" by Chattopadhyay et al. (2018) ArXiv link: [https://arxiv.org/abs/1710.11063](https://arxiv.org/abs/1710.11063)

These papers provide in-depth discussions on the theoretical foundations, improvements, and evaluations of Grad-CAM and related visualization techniques for deep learning models.
