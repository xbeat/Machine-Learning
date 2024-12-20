## Machine Learning Lifecycle in Python
Slide 1: Data Collection in Machine Learning

The foundation of any machine learning project begins with gathering relevant data. High-quality data collection is crucial for model performance. Data can be structured (tables, spreadsheets) or unstructured (images, text, audio). The collection process should focus on obtaining diverse, representative samples that capture the full spectrum of scenarios the model will encounter.

```python
# Example of collecting text data from files
def collect_text_data(directory_path):
    text_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            with open(os.path.join(directory_path, filename)) as f:
                text_data.append(f.read())
    return text_data
```

Slide 2: Data Exploration and Analysis

Before processing the data, we must understand its characteristics, distribution, and potential issues. This involves examining data types, identifying missing values, and detecting outliers. Statistical analysis helps reveal patterns and relationships within the data.

```python
def explore_data(data):
    summary = {
        'total_samples': len(data),
        'missing_values': sum(1 for x in data if x is None),
        'unique_values': len(set(data)),
        'data_range': (min(data), max(data)) if data else None
    }
    return summary
```

Slide 3: Data Cleaning and Preprocessing

Raw data often contains noise, missing values, and inconsistencies. Cleaning involves removing duplicates, handling missing values, and standardizing formats. This step ensures the data is suitable for model training.

```python
def clean_data(data):
    # Remove None values
    cleaned = [x for x in data if x is not None]
    # Remove duplicates while preserving order
    seen = set()
    cleaned = [x for x in cleaned if not (x in seen or seen.add(x))]
    return cleaned
```

Slide 4: Data Labeling Fundamentals

Data labeling assigns meaningful tags or categories to raw data. This process transforms unlabeled data into training examples that machine learning models can learn from. The quality of labels directly impacts model performance.

```python
def label_data(raw_data, labeling_function):
    labeled_data = []
    for item in raw_data:
        label = labeling_function(item)
        labeled_data.append((item, label))
    return labeled_data
```

Slide 5: Data Splitting

Before training, data must be divided into training, validation, and test sets. This separation helps evaluate model performance and prevent overfitting.

```python
def split_data(data, train_ratio=0.7, val_ratio=0.15):
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train = data[:train_size]
    val = data[train_size:train_size + val_size]
    test = data[train_size + val_size:]
    
    return train, val, test
```

Slide 6: Feature Engineering

Feature engineering transforms raw data into meaningful representations that machine learning models can understand better. This process creates new features or modifies existing ones to improve model performance.

```python
def engineer_features(data):
    features = {}
    for text in data:
        # Calculate text length
        features['length'] = len(text)
        # Count words
        features['word_count'] = len(text.split())
        # Calculate average word length
        words = text.split()
        features['avg_word_length'] = sum(len(word) for word in words) / len(words)
    return features
```

Slide 7: Model Training Process

Model training is an iterative process where the algorithm learns patterns from the training data. The model adjusts its parameters to minimize prediction errors.

```python
def train_model(X_train, y_train, learning_rate=0.01, epochs=100):
    weights = [0] * len(X_train[0])
    bias = 0
    
    for _ in range(epochs):
        for x, y in zip(X_train, y_train):
            prediction = sum(w * xi for w, xi in zip(weights, x)) + bias
            error = prediction - y
            
            # Update weights and bias
            weights = [w - learning_rate * error * xi 
                      for w, xi in zip(weights, x)]
            bias -= learning_rate * error
            
    return weights, bias
```

Slide 8: Model Evaluation

Evaluation metrics help assess model performance and identify areas for improvement. Different metrics are suitable for different types of problems.

```python
def evaluate_model(y_true, y_pred):
    # Calculate accuracy
    accuracy = sum(1 for yt, yp in zip(y_true, y_pred) 
                  if yt == yp) / len(y_true)
    
    # Calculate mean squared error
    mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)
    
    return {'accuracy': accuracy, 'mse': mse}
```

Slide 9: Model Deployment

Deployment involves integrating the trained model into a production environment where it can make predictions on new data.

```python
class ModelDeployment:
    def __init__(self, model_weights, bias):
        self.weights = model_weights
        self.bias = bias
    
    def predict(self, input_data):
        prediction = sum(w * x for w, x in zip(self.weights, input_data))
        prediction += self.bias
        return prediction
```

Slide 10: Real-Life Example - Image Classification

An example of classifying images of handwritten digits demonstrates the complete machine learning lifecycle.

```python
def process_image(image_path):
    # Convert image to grayscale values (0-255)
    pixels = []
    with open(image_path, 'rb') as img:
        # Simplified image processing
        raw_pixels = img.read()
        pixels = [ord(byte) for byte in raw_pixels]
    return normalize_pixels(pixels)
```

Slide 11: Real-Life Example - Weather Prediction

A weather prediction system using historical temperature and humidity data illustrates practical machine learning application.

```python
def prepare_weather_data(temperatures, humidity):
    features = []
    for t, h in zip(temperatures, humidity):
        feature_vector = [
            t,  # temperature
            h,  # humidity
            t * h,  # interaction term
            t ** 2,  # quadratic temperature term
            math.sin(2 * math.pi * t / 24)  # time of day cycle
        ]
        features.append(feature_vector)
    return features
```

Slide 12: Continuous Model Monitoring

Monitoring deployed models ensures they maintain performance over time and detect when retraining is needed.

```python
def monitor_model_performance(predictions, actual_values, threshold=0.8):
    performance_metrics = {
        'timestamp': time.time(),
        'accuracy': calculate_accuracy(predictions, actual_values),
        'data_drift': detect_drift(predictions, actual_values)
    }
    
    if performance_metrics['accuracy'] < threshold:
        trigger_retraining_alert()
    
    return performance_metrics
```

Slide 13: Additional Resources

Recent research papers from ArXiv.org for further reading:

*   "A Survey of Deep Learning Lifecycle: From a Software Engineering Perspective" (arXiv:2205.03656)
*   "MLOps: A Survey on Challenges and Opportunities" (arXiv:2110.14711)
*   "Efficient Machine Learning Lifecycle Management" (arXiv:2010.00029)

These papers provide comprehensive insights into modern machine learning lifecycle management and best practices.

