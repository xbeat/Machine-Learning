## Keras Model Train in Python, Predict in C++
Slide 1: Introduction to Keras

Keras is a high-level neural network library that runs on top of TensorFlow. It provides a user-friendly interface for building and training deep learning models. Keras simplifies the process of creating complex neural networks by offering pre-built layers and models.

```python
import tensorflow as tf
from tensorflow import keras

# Create a simple sequential model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
```

Slide 2: Building a Keras Model

Keras offers various ways to build models, including the Sequential API and the Functional API. The Sequential API is straightforward and suitable for linear stack of layers, while the Functional API allows for more complex model architectures.

```python
# Sequential API
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Functional API
inputs = keras.Input(shape=(10,))
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(32, activation='relu')(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

Slide 3: Preparing Data for Training

Before training a Keras model, it's essential to prepare and preprocess the data. This typically involves splitting the data into training and validation sets, normalizing the features, and converting labels to the appropriate format.

```python
import numpy as np

# Generate sample data
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, (1000, 1))

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("Training data shape:", X_train_scaled.shape)
print("Validation data shape:", X_val_scaled.shape)
```

Slide 4: Training a Keras Model

Training a Keras model involves calling the `fit` method, specifying the training data, number of epochs, and batch size. During training, you can monitor various metrics and use callbacks for early stopping or learning rate scheduling.

```python
# Define the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
)

# Plot training history
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
```

Slide 5: Evaluating and Making Predictions

After training, it's crucial to evaluate the model's performance on unseen data and use it to make predictions. Keras provides methods for both evaluation and prediction.

```python
# Evaluate the model on validation data
loss, accuracy = model.evaluate(X_val_scaled, y_val)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Make predictions on new data
new_data = np.random.rand(5, 10)
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
print("Predictions:")
print(predictions)
```

Slide 6: Saving and Loading Keras Models

Keras allows you to save trained models for later use or deployment. You can save the entire model or just the weights, and load them when needed.

```python
# Save the entire model
model.save('my_model.h5')

# Save only the weights
model.save_weights('my_model_weights.h5')

# Load the entire model
loaded_model = keras.models.load_model('my_model.h5')

# Load weights into a new model
new_model = keras.Sequential([...])  # Define the model architecture
new_model.load_weights('my_model_weights.h5')

# Make predictions with the loaded model
predictions = loaded_model.predict(X_val_scaled)
print("Predictions from loaded model:")
print(predictions[:5])
```

Slide 7: Converting Keras Model to TensorFlow Lite

To use a Keras model in C++, you can convert it to TensorFlow Lite format. This process involves using the TFLiteConverter to create a smaller, optimized version of your model.

```python
# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Optionally, you can quantize the model for further optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()

with open('quantized_model.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

print("TensorFlow Lite model saved successfully.")
```

Slide 8: Setting Up C++ Environment for TensorFlow Lite

To use the TensorFlow Lite model in C++, you need to set up the TensorFlow Lite C++ library. This typically involves downloading the TensorFlow Lite C++ library and including it in your C++ project.

```cpp
// Example C++ code structure for TensorFlow Lite
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

int main() {
    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile("model.tflite");

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    // Allocate tensor buffers
    interpreter->AllocateTensors();

    // Perform inference
    // ...

    return 0;
}
```

Slide 9: Loading and Preparing Input Data in C++

Before making predictions with the TensorFlow Lite model in C++, you need to load and prepare the input data. This involves reading the input data and setting it in the model's input tensor.

```cpp
#include <vector>
#include <iostream>

// ... (previous includes)

int main() {
    // ... (previous code for loading model and building interpreter)

    // Get input and output tensors
    int input = interpreter->inputs()[0];
    int output = interpreter->outputs()[0];

    // Prepare input data (example with random data)
    std::vector<float> input_data(10);
    for (int i = 0; i < 10; i++) {
        input_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    //  input data to model's input tensor
    float* input_tensor = interpreter->typed_input_tensor<float>(0);
    std::(input_data.begin(), input_data.end(), input_tensor);

    std::cout << "Input data prepared for inference." << std::endl;

    // ... (continue with inference)
    return 0;
}
```

Slide 10: Performing Inference in C++

After preparing the input data, you can perform inference using the TensorFlow Lite model in C++. This involves invoking the interpreter and accessing the output tensor.

```cpp
// ... (previous includes and code)

int main() {
    // ... (previous code for loading model, building interpreter, and preparing input)

    // Perform inference
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        return 1;
    }

    // Get the output tensor
    float* output_tensor = interpreter->typed_output_tensor<float>(0);

    // Print the results
    std::cout << "Inference results:" << std::endl;
    for (int i = 0; i < interpreter->outputs().size(); i++) {
        std::cout << output_tensor[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

Slide 11: Post-processing and Interpreting Results

After performing inference, you may need to post-process the results to interpret them in a meaningful way. This could involve applying thresholds, converting probabilities to class labels, or any other application-specific processing.

```cpp
// ... (previous includes and code)

int main() {
    // ... (previous code for inference)

    // Post-process results (example: applying threshold for binary classification)
    const float threshold = 0.5;
    std::cout << "Interpreted results:" << std::endl;
    for (int i = 0; i < interpreter->outputs().size(); i++) {
        float probability = output_tensor[i];
        int predicted_class = (probability > threshold) ? 1 : 0;
        std::cout << "Output " << i << ": Probability = " << probability
                  << ", Predicted Class = " << predicted_class << std::endl;
    }

    return 0;
}
```

Slide 12: Real-life Example: Image Classification

Let's consider an image classification task using a pre-trained MobileNetV2 model. We'll use Keras to load the model, make predictions on an image, and then show how to use this model in C++.

```python
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load and preprocess an image
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make predictions
preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]

print("Predictions:")
for _, label, score in decoded_preds:
    print(f"{label}: {score:.2f}")

# Convert model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('mobilenet_v2.tflite', 'wb') as f:
    f.write(tflite_model)

print("TensorFlow Lite model saved successfully.")
```

Slide 13: Real-life Example: Image Classification in C++

Now, let's use the converted MobileNetV2 model in C++ to perform image classification. This example demonstrates how to load an image, preprocess it, and run inference using the TensorFlow Lite model.

```cpp
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

int main() {
    // Load TFLite model
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile("mobilenet_v2.tflite");

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    interpreter->AllocateTensors();

    // Load and preprocess image
    cv::Mat img = cv::imread("elephant.jpg");
    cv::resize(img, img, cv::Size(224, 224));
    cv::Mat float_img;
    img.convertTo(float_img, CV_32F, 1.0 / 255);

    // Set input tensor
    float* input = interpreter->typed_input_tensor<float>(0);
    for (int i = 0; i < 224 * 224 * 3; i++) {
        input[i] = float_img.at<float>(i);
    }

    // Run inference
    interpreter->Invoke();

    // Get output tensor
    float* output = interpreter->typed_output_tensor<float>(0);

    // Find top prediction
    int max_index = 0;
    float max_score = output[0];
    for (int i = 1; i < 1000; i++) {
        if (output[i] > max_score) {
            max_score = output[i];
            max_index = i;
        }
    }

    std::cout << "Predicted class index: " << max_index << std::endl;
    std::cout << "Confidence score: " << max_score << std::endl;

    return 0;
}
```

Slide 14: Additional Resources

For further learning and exploration of Keras, TensorFlow, and TensorFlow Lite, consider the following resources:

1. TensorFlow Official Documentation: [https://www.tensorflow.org/guide](https://www.tensorflow.org/guide)
2. Keras Official Documentation: [https://keras.io/](https://keras.io/)
3. TensorFlow Lite Guide: [https://www.tensorflow.org/lite/guide](https://www.tensorflow.org/lite/guide)
4. "Efficient Deep Learning: A Survey on Making Deep Learning Models Smaller, Faster, and Better" by Gaurav Menghani (arXiv:2106.08962): [https://arxiv.org/abs/2106.08962](https://arxiv.org/abs/2106.08962)
5. "A Survey of Deep Learning Techniques for Mobile Robot Applications" by Tai et al. (arXiv:2009.08819): [https://arxiv.org/abs/2009.08819](https://arxiv.org/abs/2009.08819)

These resources provide in-depth information on model development, optimization, and deployment across various platforms.

