## Advanced ECG Analysis Using Transfer Learning and CNNs
Slide 1: Project Overview: ECG Analysis with Deep Learning

This project focuses on classifying ECG images using advanced deep learning techniques. We'll explore data preparation, model development, and the application of transfer learning to improve classification accuracy for various heart conditions.

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and prepare the data
train_generator = train_datagen.flow_from_directory(
    'path/to/train_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'path/to/test_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

Slide 2: Data Exploration and Visualization

Understanding the dataset is crucial. We'll visualize samples from each ECG category to gain insights into the data distribution and characteristics.

```python
import matplotlib.pyplot as plt
import numpy as np

# Function to plot sample images
def plot_samples(generator, n=4):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    
    for i in range(n):
        images, labels = next(generator)
        ax = axes[i]
        ax.imshow(images[0])
        ax.set_title(f"Class: {np.argmax(labels[0])}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Plot sample images
plot_samples(train_generator)
```

Slide 3: Baseline Model: Custom CNN

We'll start with a custom Convolutional Neural Network (CNN) as our baseline model to establish initial performance metrics.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the baseline CNN model
def create_baseline_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax')  # 4 classes
    ])
    return model

baseline_model = create_baseline_model()
baseline_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the baseline model
history = baseline_model.fit(train_generator, epochs=10, validation_data=test_generator)
```

Slide 4: Baseline Model Performance

Let's evaluate the performance of our baseline CNN model and visualize the training progress.

```python
# Evaluate the baseline model
test_loss, test_acc = baseline_model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.2f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 5: Transfer Learning with VGG16

To improve our model's performance, we'll leverage transfer learning using the pre-trained VGG16 model.

```python
# Load VGG16 model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Create new model on top
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=20, validation_data=test_generator)
```

Slide 6: Transfer Learning Model Performance

We'll evaluate the performance of our transfer learning model and compare it with the baseline model.

```python
# Evaluate the transfer learning model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.2f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Transfer Learning Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Transfer Learning Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 7: Fine-tuning the Transfer Learning Model

To further improve performance, we'll fine-tune the last few layers of the VGG16 model.

```python
# Unfreeze the last 4 layers of the base model
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tune the model
history_fine = model.fit(train_generator,
                         epochs=10,
                         validation_data=test_generator)
```

Slide 8: Fine-tuned Model Performance

Let's evaluate the performance of our fine-tuned transfer learning model.

```python
# Evaluate the fine-tuned model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy after fine-tuning: {test_acc:.2f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_fine.history['accuracy'], label='Training Accuracy')
plt.plot(history_fine.history['val_accuracy'], label='Validation Accuracy')
plt.title('Fine-tuned Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_fine.history['loss'], label='Training Loss')
plt.plot(history_fine.history['val_loss'], label='Validation Loss')
plt.title('Fine-tuned Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 9: Model Predictions and Visualization

We'll use our trained model to make predictions on test data and visualize the results.

```python
import numpy as np

# Get a batch of test images
test_images, test_labels = next(test_generator)

# Make predictions
predictions = model.predict(test_images)

# Function to plot images with predictions
def plot_predictions(images, true_labels, predictions, n=4):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i])
        true_class = np.argmax(true_labels[i])
        pred_class = np.argmax(predictions[i])
        ax.set_title(f"True: {true_class}, Pred: {pred_class}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Plot predictions
plot_predictions(test_images, test_labels, predictions)
```

Slide 10: Model Interpretability with Grad-CAM

To understand what features our model focuses on, we'll use Gradient-weighted Class Activation Mapping (Grad-CAM).

```python
from tensorflow.keras.models import Model

# Create a Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Generate and plot Grad-CAM for a sample image
img = test_images[0]
heatmap = make_gradcam_heatmap(img[np.newaxis, ...], model, 'block5_conv3')

plt.matshow(heatmap)
plt.title("Grad-CAM Heatmap")
plt.show()
```

Slide 11: Real-life Example: Automated ECG Screening

In a hospital setting, our model can be used to quickly screen ECGs for potential abnormalities, allowing medical professionals to prioritize cases that require immediate attention.

```python
def ecg_screening(ecg_image_path, model):
    # Load and preprocess the ECG image
    img = tf.keras.preprocessing.image.load_img(ecg_image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    confidence = prediction[0][class_index]

    classes = ['Normal', 'Abnormal Beat', 'Myocardial Infarction', 'Other']
    result = f"ECG Classification: {classes[class_index]}"
    result += f"\nConfidence: {confidence:.2f}"

    return result

# Example usage
ecg_path = 'path/to/patient_ecg.jpg'
screening_result = ecg_screening(ecg_path, model)
print(screening_result)
```

Slide 12: Real-life Example: ECG Monitoring System

Our model can be integrated into a continuous ECG monitoring system for patients in intensive care units, alerting medical staff to potential cardiac events in real-time.

```python
import time

def continuous_ecg_monitoring(model, interval=60):
    while True:
        # Simulate getting a new ECG reading every 'interval' seconds
        ecg_data = simulate_ecg_reading()  # This function would capture real ECG data
        
        # Preprocess the ECG data
        processed_ecg = preprocess_ecg(ecg_data)
        
        # Make prediction
        prediction = model.predict(processed_ecg)
        class_index = np.argmax(prediction[0])
        confidence = prediction[0][class_index]
        
        classes = ['Normal', 'Abnormal Beat', 'Myocardial Infarction', 'Other']
        
        if class_index != 0:  # If not normal
            alert_medical_staff(classes[class_index], confidence)
        
        time.sleep(interval)

def simulate_ecg_reading():
    # This function would be replaced with actual ECG data acquisition
    return np.random.rand(224, 224, 3)

def preprocess_ecg(ecg_data):
    # Preprocess the ECG data for the model
    return np.expand_dims(ecg_data, axis=0) / 255.0

def alert_medical_staff(condition, confidence):
    print(f"ALERT: Possible {condition} detected. Confidence: {confidence:.2f}")
    # In a real system, this would send an alert to the medical staff

# Example usage
continuous_ecg_monitoring(model, interval=10)  # Check every 10 seconds for demonstration
```

Slide 13: Future Improvements and Considerations

While our model shows promising results, there's always room for improvement. Consider the following steps for future enhancements:

1. Collect more diverse ECG data to improve model generalization.
2. Experiment with other pre-trained models like ResNet or EfficientNet.
3. Implement explainable AI techniques for better model interpretability.
4. Conduct clinical trials to validate the model's performance in real-world scenarios.

```python
# Example of using a different pre-trained model (ResNet50)
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=20, validation_data=test_generator)
```

Slide 14: Additional Resources

For further exploration of ECG analysis using deep learning, consider the following resources:

1. "Automatic Detection of Electrocardiogram ST Segment: Application in Ischemic Disease Diagnosis" (ArXiv:1809.03452)
2. "ECG Arrhythmia Classification Using a 2-D Convolutional Neural Network" (ArXiv:1804.06812)
3. "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network" (Nature Medicine, 2019)

These papers provide valuable insights into advanced techniques and methodologies in ECG analysis using machine learning.

