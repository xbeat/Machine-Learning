## Computational Intelligence for Early Infertility Detection
Slide 1: Dataset Preprocessing for Fetal Ultrasound

The initial phase involves loading and preprocessing ultrasound images using OpenCV and NumPy. The process includes resizing images to a standard dimension, normalizing pixel values, and implementing basic data augmentation techniques for enhanced model training.

```python
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_images(image_paths, target_size=(224, 224)):
    processed_images = []
    for path in image_paths:
        # Load and preprocess image
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        
        processed_images.append(img)
    
    return np.array(processed_images)

# Example usage
X = preprocess_images(image_paths)
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
```

Slide 2: Data Augmentation Pipeline

Data augmentation is crucial for improving model generalization. This implementation creates a robust augmentation pipeline using geometric transformations and intensity adjustments to expand the dataset artificially while maintaining clinical relevance.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_augmentation_pipeline():
    augmentor = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        zoom_range=0.2
    )
    return augmentor

# Create and apply augmentation
augmentor = create_augmentation_pipeline()
train_generator = augmentor.flow(
    X_train, y_train,
    batch_size=32,
    shuffle=True
)
```

Slide 3: Simple CNN Architecture

The baseline CNN model establishes fundamental performance metrics. This architecture incorporates convolutional layers with batch normalization and dropout for regularization, proving surprisingly effective for ultrasound image classification.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout

def create_simple_cnn(input_shape=(224, 224, 3), num_classes=4):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model
```

Slide 4: Inception Module Implementation

This implementation creates a custom Inception module that processes input features at multiple scales simultaneously. The module concatenates different convolutional paths to capture diverse feature representations.

```python
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model

def inception_module(x, filters):
    # 1x1 convolution path
    path1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)
    
    # 1x1 -> 3x3 convolution path
    path2 = Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
    path2 = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(path2)
    
    # 1x1 -> 5x5 convolution path
    path3 = Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    path3 = Conv2D(filters[4], (5, 5), padding='same', activation='relu')(path3)
    
    # 3x3 max pooling -> 1x1 convolution path
    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = Conv2D(filters[5], (1, 1), padding='same', activation='relu')(path4)
    
    return Concatenate(axis=-1)([path1, path2, path3, path4])
```

Slide 5: Xception Building Blocks

Implementation of the core Xception architecture components featuring depthwise separable convolutions. This approach significantly reduces computational complexity while maintaining model expressiveness for ultrasound image analysis.

```python
from tensorflow.keras.layers import SeparableConv2D

def xception_block(inputs, filters, strides=1):
    residual = Conv2D(filters, (1, 1), strides=strides)(inputs)
    
    x = SeparableConv2D(filters, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SeparableConv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=strides, padding='same')(x)
    
    return Add()([x, residual])

def create_xception_model(input_shape=(224, 224, 3), num_classes=4):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), strides=(2, 2))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = xception_block(x, 64)
    x = xception_block(x, 128, strides=2)
    x = xception_block(x, 256, strides=2)
    
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)
```

Slide 6: Attention Mechanism Implementation

A custom attention mechanism that enables the model to focus on relevant features within ultrasound images. The mechanism computes attention weights and applies them to feature maps for enhanced feature extraction.

```python
def attention_block(inputs, ratio=8):
    channels = inputs.shape[-1]
    
    # Global average pooling
    x = GlobalAveragePooling2D()(inputs)
    
    # Squeeze and excitation
    x = Dense(channels // ratio, activation='relu')(x)
    x = Dense(channels, activation='sigmoid')(x)
    
    # Reshape for multiplication
    x = Reshape((1, 1, channels))(x)
    
    # Apply attention weights
    return Multiply()([inputs, x])

def create_attention_layer(inputs, filters):
    x = Conv2D(filters, (3, 3), padding='same')(inputs)
    x = attention_block(x)
    x = BatchNormalization()(x)
    return Activation('relu')(x)
```

Slide 7: Model Training Configuration

Advanced training setup incorporating learning rate scheduling, early stopping, and model checkpointing. This configuration ensures optimal convergence and prevents overfitting during the training process.

```python
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def configure_training(model_name='ultrasound_model'):
    callbacks = [
        ModelCheckpoint(
            f'best_{model_name}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    return callbacks

# Training configuration
batch_size = 32
epochs = 100
callbacks = configure_training()

history = model.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks
)
```

Slide 8: Performance Metrics Implementation

Comprehensive evaluation metrics implementation including precision, recall, F1-score, and confusion matrix visualization for detailed model performance analysis.

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, class_names):
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    report = classification_report(y_test_classes, y_pred_classes, 
                                 target_names=class_names, output_dict=True)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return report
```

Slide 9: Results Analysis Framework

Implementation of a comprehensive results analysis system that tracks and visualizes training metrics, learning curves, and model performance across different architectures for comparative analysis.

```python
import pandas as pd
import matplotlib.pyplot as plt

def analyze_training_history(history_dict, model_name):
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['accuracy'], label='Training')
    plt.plot(history_dict['val_accuracy'], label='Validation')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['loss'], label='Training')
    plt.plot(history_dict['val_loss'], label='Validation')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Return metrics summary
    metrics_summary = {
        'final_train_acc': history_dict['accuracy'][-1],
        'final_val_acc': history_dict['val_accuracy'][-1],
        'final_train_loss': history_dict['loss'][-1],
        'final_val_loss': history_dict['val_loss'][-1]
    }
    return pd.DataFrame([metrics_summary])
```

Slide 10: Model Comparison Framework

A systematic approach to compare different model architectures, tracking performance metrics and computational efficiency. This framework enables objective evaluation of each model's strengths and weaknesses.

```python
class ModelComparator:
    def __init__(self):
        self.results = {}
        self.training_times = {}
        
    def evaluate_model(self, model, model_name, X_train, y_train, X_val, y_val):
        # Time training
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=configure_training(model_name),
            verbose=0
        )
        training_time = time.time() - start_time
        
        # Store results
        self.results[model_name] = {
            'history': history.history,
            'final_val_acc': history.history['val_accuracy'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }
        self.training_times[model_name] = training_time
        
    def generate_comparison_report(self):
        comparison_df = pd.DataFrame({
            model_name: {
                'Validation Accuracy': results['final_val_acc'],
                'Training Time (s)': self.training_times[model_name]
            }
            for model_name, results in self.results.items()
        }).T
        
        return comparison_df.round(4)
```

Slide 11: Real-world Implementation - Clinical Validation

Implementation of a clinical validation pipeline that processes real ultrasound images through the trained models while maintaining medical data handling standards and preprocessing consistency.

```python
class ClinicalValidator:
    def __init__(self, model, preprocessing_pipeline):
        self.model = model
        self.preprocessing_pipeline = preprocessing_pipeline
        self.confidence_threshold = 0.85
        
    def process_clinical_image(self, image_path):
        # Load and preprocess clinical image
        img = cv2.imread(image_path)
        processed_img = self.preprocessing_pipeline(img)
        
        # Generate prediction
        prediction = self.model.predict(np.expand_dims(processed_img, axis=0))[0]
        confidence = np.max(prediction)
        predicted_class = np.argmax(prediction)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'requires_review': confidence < self.confidence_threshold
        }
        
    def validate_batch(self, image_paths, ground_truth):
        results = []
        for img_path, true_label in zip(image_paths, ground_truth):
            result = self.process_clinical_image(img_path)
            result['true_label'] = true_label
            results.append(result)
            
        return pd.DataFrame(results)
```

Slide 12: Performance Optimization and Model Deployment

Implementation of model optimization techniques including quantization and pruning, alongside deployment utilities for clinical integration. This ensures efficient model execution in resource-constrained environments.

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile

class ModelOptimizer:
    def __init__(self, model):
        self.model = model
        self.optimized_model = None
        
    def quantize_model(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save quantized model
        with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as f:
            f.write(tflite_model)
            return f.name
            
    def prune_model(self, target_sparsity=0.5):
        pruning_params = {
            'pruning_schedule': tf.keras.optimizers.schedules.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=1000
            )
        }
        
        self.optimized_model = tf.keras.models.clone_model(
            self.model,
            clone_function=lambda layer: tf.keras.layers.prune.prune_low_magnitude(
                layer, **pruning_params
            ) if isinstance(layer, tf.keras.layers.Conv2D) else layer
        )
        
        return self.optimized_model
```

Slide 13: Advanced Loss Functions and Metrics

Implementation of specialized loss functions and metrics tailored for medical image classification, incorporating class imbalance handling and confidence penalties.

```python
import tensorflow as tf
from tensorflow.keras import backend as K

class MedicalImageMetrics:
    @staticmethod
    def focal_loss(gamma=2., alpha=.25):
        def focal_loss_fixed(y_true, y_pred):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            
            loss = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - \
                   K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
            return loss
        return focal_loss_fixed
    
    @staticmethod
    def weighted_categorical_crossentropy(class_weights):
        def loss(y_true, y_pred):
            y_true = K.cast(y_true, K.dtype(y_pred))
            return K.mean(
                K.sum(y_true * class_weights * K.log(K.clip(y_pred, K.epsilon(), 1.0)), axis=-1))
        return loss
        
    @staticmethod
    def specificity(y_true, y_pred):
        true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())
```

Slide 14: Additional Resources

ArXiv papers for further reading and implementation details:

*   [https://arxiv.org/abs/2103.12465](https://arxiv.org/abs/2103.12465) - Automated Fetal Head Detection and Measurement in Ultrasound Images through Deep Learning
*   [https://arxiv.org/abs/1905.13105](https://arxiv.org/abs/1905.13105) - Deep Learning Approaches for Medical Image Analysis in Obstetrics and Gynecology
*   [https://arxiv.org/abs/2007.09346](https://arxiv.org/abs/2007.09346) - A Survey on Deep Learning for Medical Image Analysis in Reproductive Healthcare
*   [https://arxiv.org/abs/2109.04368](https://arxiv.org/abs/2109.04368) - Attention Mechanisms in Medical Image Analysis: A Survey
*   [https://arxiv.org/abs/2008.06167](https://arxiv.org/abs/2008.06167) - Deep Learning in Medical Image Analysis: Challenges and Future Directions

