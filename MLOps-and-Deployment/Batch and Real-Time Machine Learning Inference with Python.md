## Batch and Real-Time Machine Learning Inference with Python
Slide 1: Introduction to Batch and Real-Time Inference

Batch and real-time inference are two fundamental approaches to deploying machine learning models. Batch inference processes large volumes of data at scheduled intervals, while real-time inference handles individual requests as they arrive. This slideshow will explore both methods using Python, providing practical examples and code snippets.

```python
import time

def batch_inference(data):
    start_time = time.time()
    results = [process_item(item) for item in data]
    end_time = time.time()
    print(f"Batch inference time: {end_time - start_time:.2f} seconds")
    return results

def real_time_inference(item):
    start_time = time.time()
    result = process_item(item)
    end_time = time.time()
    print(f"Real-time inference time: {end_time - start_time:.2f} seconds")
    return result

def process_item(item):
    # Simulating processing time
    time.sleep(0.1)
    return item * 2

# Example usage
batch_data = list(range(100))
batch_results = batch_inference(batch_data)
real_time_result = real_time_inference(42)
```

Slide 2: Batch Inference Basics

Batch inference involves processing multiple data points simultaneously. It's ideal for scenarios where immediate results aren't required and efficiency in processing large datasets is crucial. Let's implement a simple batch inference system for image classification.

```python
import numpy as np
from PIL import Image

def load_image(path):
    return np.array(Image.open(path))

def preprocess_batch(images):
    return np.array([img / 255.0 for img in images])

def batch_classify(model, image_paths):
    images = [load_image(path) for path in image_paths]
    preprocessed = preprocess_batch(images)
    return model.predict(preprocessed)

# Assuming we have a pre-trained model
model = load_pretrained_model()
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = batch_classify(model, image_paths)
print(f"Batch classification results: {results}")
```

Slide 3: Advantages of Batch Inference

Batch inference offers several benefits, including improved resource utilization and reduced overhead. It's particularly useful for tasks like overnight data processing or periodic report generation. Let's explore a scenario where batch inference shines: analyzing customer reviews.

```python
import pandas as pd
from textblob import TextBlob

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

def batch_sentiment_analysis(reviews_df):
    reviews_df['sentiment'] = reviews_df['review'].apply(analyze_sentiment)
    return reviews_df

# Load customer reviews
reviews = pd.read_csv('customer_reviews.csv')

# Perform batch sentiment analysis
analyzed_reviews = batch_sentiment_analysis(reviews)

# Aggregate results
sentiment_summary = analyzed_reviews.groupby('product_id')['sentiment'].mean()
print(sentiment_summary)
```

Slide 4: Implementing Batch Inference with Multiprocessing

To further optimize batch inference, we can leverage multiprocessing to parallelize the workload. This is especially useful for CPU-bound tasks. Let's modify our sentiment analysis example to use multiprocessing.

```python
import multiprocessing as mp

def process_chunk(chunk):
    chunk['sentiment'] = chunk['review'].apply(analyze_sentiment)
    return chunk

def parallel_batch_sentiment_analysis(reviews_df, num_processes=4):
    pool = mp.Pool(processes=num_processes)
    chunks = np.array_split(reviews_df, num_processes)
    results = pool.map(process_chunk, chunks)
    return pd.concat(results)

# Load customer reviews
reviews = pd.read_csv('customer_reviews.csv')

# Perform parallel batch sentiment analysis
analyzed_reviews = parallel_batch_sentiment_analysis(reviews)

# Aggregate results
sentiment_summary = analyzed_reviews.groupby('product_id')['sentiment'].mean()
print(sentiment_summary)
```

Slide 5: Real-Time Inference Introduction

Real-time inference involves processing individual data points as they arrive, providing immediate results. It's crucial for applications requiring instant feedback, such as recommendation systems or fraud detection. Let's implement a simple real-time inference system for text classification.

```python
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load pre-trained model and vectorizer
model = joblib.load('text_classifier.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

def real_time_text_classify(text):
    # Preprocess and vectorize the input text
    vectorized_text = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0].max()
    
    return prediction, probability

# Example usage
input_text = "This product is amazing! I love it."
category, confidence = real_time_text_classify(input_text)
print(f"Category: {category}, Confidence: {confidence:.2f}")
```

Slide 6: Real-Time Inference with API

In practice, real-time inference often involves deploying models as APIs. Let's create a simple Flask API for our text classification model.

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.json
    text = data['text']
    category, confidence = real_time_text_classify(text)
    return jsonify({
        'category': category,
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True)

# To test the API:
# import requests
# response = requests.post('http://localhost:5000/classify', 
#                          json={'text': 'This product is amazing! I love it.'})
# print(response.json())
```

Slide 7: Scaling Real-Time Inference

As demand increases, scaling real-time inference becomes crucial. Let's explore using Redis as a caching layer to improve performance and reduce load on our classification service.

```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached_real_time_text_classify(text):
    # Check if result is in cache
    cached_result = redis_client.get(text)
    if cached_result:
        return json.loads(cached_result)
    
    # If not in cache, perform classification
    category, confidence = real_time_text_classify(text)
    result = {'category': category, 'confidence': float(confidence)}
    
    # Store in cache for future use
    redis_client.setex(text, 3600, json.dumps(result))  # Cache for 1 hour
    
    return result

# Modified Flask route
@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.json
    text = data['text']
    result = cached_real_time_text_classify(text)
    return jsonify(result)
```

Slide 8: Batch vs Real-Time: When to Use Each

The choice between batch and real-time inference depends on the specific use case. Let's create a decision helper function to illustrate this concept.

```python
def inference_type_recommendation(latency_requirement_ms, data_volume, update_frequency):
    if latency_requirement_ms <= 100:
        return "Real-time inference"
    elif data_volume > 1000000 and update_frequency == "daily":
        return "Batch inference"
    elif 100 < latency_requirement_ms <= 1000 and data_volume <= 1000000:
        return "Consider real-time or mini-batch inference"
    else:
        return "Requires further analysis"

# Example usage
use_cases = [
    {"name": "Fraud Detection", "latency": 50, "volume": 10000, "frequency": "continuous"},
    {"name": "Daily Sales Report", "latency": 3600000, "volume": 5000000, "frequency": "daily"},
    {"name": "Product Recommendation", "latency": 500, "volume": 100000, "frequency": "hourly"}
]

for case in use_cases:
    recommendation = inference_type_recommendation(case["latency"], case["volume"], case["frequency"])
    print(f"{case['name']}: {recommendation}")
```

Slide 9: Mini-Batch Inference

Mini-batch inference is a hybrid approach that combines aspects of both batch and real-time inference. It processes small groups of data points, balancing between latency and throughput. Let's implement a mini-batch inference system for image classification.

```python
import numpy as np
from collections import deque

class MiniBatchClassifier:
    def __init__(self, model, batch_size=32, max_wait_time=0.5):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.queue = deque()
        self.last_process_time = time.time()

    def classify(self, image):
        self.queue.append(image)
        current_time = time.time()
        
        if len(self.queue) >= self.batch_size or (current_time - self.last_process_time) > self.max_wait_time:
            batch = [self.queue.popleft() for _ in range(min(self.batch_size, len(self.queue)))]
            results = self.process_batch(batch)
            self.last_process_time = current_time
            return results[0]  # Return result for the current image
        
        return None  # No result yet, still batching

    def process_batch(self, batch):
        preprocessed = np.array([img / 255.0 for img in batch])
        return self.model.predict(preprocessed)

# Usage
model = load_pretrained_model()
mini_batch_classifier = MiniBatchClassifier(model)

# Simulating incoming images
for i in range(100):
    image = load_image(f"image_{i}.jpg")
    result = mini_batch_classifier.classify(image)
    if result is not None:
        print(f"Classification result for image {i}: {result}")
```

Slide 10: Real-Life Example: Content Moderation System

Let's design a content moderation system that uses both batch and real-time inference. This system will process user-generated content, flagging inappropriate text and images.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ContentModerationSystem:
    def __init__(self, text_model, image_model):
        self.text_model = text_model
        self.image_model = image_model
        self.text_queue = asyncio.Queue()
        self.image_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def moderate_text(self, text):
        return await self.text_queue.put((text, self.text_model.predict([text])[0]))

    async def moderate_image(self, image):
        return await self.image_queue.put((image, self.image_model.predict([image])[0]))

    async def batch_process_text(self):
        while True:
            batch = []
            while len(batch) < 32:
                if self.text_queue.empty():
                    if batch:
                        break
                    await asyncio.sleep(0.1)
                    continue
                batch.append(await self.text_queue.get())
            
            texts, _ = zip(*batch)
            results = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.text_model.predict, texts
            )
            for (text, _), result in zip(batch, results):
                print(f"Text moderation result: {text[:20]}... -> {result}")

    async def batch_process_images(self):
        while True:
            batch = []
            while len(batch) < 16:
                if self.image_queue.empty():
                    if batch:
                        break
                    await asyncio.sleep(0.1)
                    continue
                batch.append(await self.image_queue.get())
            
            images, _ = zip(*batch)
            results = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.image_model.predict, images
            )
            for (image, _), result in zip(batch, results):
                print(f"Image moderation result: {image[:20]}... -> {result}")

    async def run(self):
        await asyncio.gather(
            self.batch_process_text(),
            self.batch_process_images()
        )

# Usage
text_model = load_text_moderation_model()
image_model = load_image_moderation_model()
moderation_system = ContentModerationSystem(text_model, image_model)

async def simulate_content():
    for i in range(100):
        if i % 2 == 0:
            await moderation_system.moderate_text(f"User comment {i}")
        else:
            await moderation_system.moderate_image(f"image_data_{i}")
        await asyncio.sleep(0.1)

asyncio.run(asyncio.gather(moderation_system.run(), simulate_content()))
```

Slide 11: Real-Life Example: IoT Sensor Data Processing

In this example, we'll create a system that processes data from IoT sensors, using both batch and real-time inference for different aspects of the analysis.

```python
import numpy as np
from scipy import stats

class IoTSensorProcessor:
    def __init__(self, anomaly_model):
        self.anomaly_model = anomaly_model
        self.data_buffer = []
        self.batch_size = 1000

    def process_reading(self, sensor_id, timestamp, value):
        # Real-time anomaly detection
        is_anomaly = self.real_time_anomaly_detection(value)
        
        # Add to batch processing buffer
        self.data_buffer.append((sensor_id, timestamp, value))
        
        # Perform batch processing if buffer is full
        if len(self.data_buffer) >= self.batch_size:
            self.batch_process()
        
        return is_anomaly

    def real_time_anomaly_detection(self, value):
        # Using pre-trained model for quick anomaly detection
        return self.anomaly_model.predict([value])[0]

    def batch_process(self):
        sensor_ids, timestamps, values = zip(*self.data_buffer)
        
        # Perform more complex analysis on the batch data
        mean_value = np.mean(values)
        median_value = np.median(values)
        mode_value = stats.mode(values)[0]
        
        # Identify trends
        trend = np.polyfit(range(len(values)), values, 1)[0]
        
        print(f"Batch processing results:")
        print(f"Mean: {mean_value:.2f}, Median: {median_value:.2f}, Mode: {mode_value:.2f}")
        print(f"Trend: {'Increasing' if trend > 0 else 'Decreasing'}")
        
        # Clear the buffer
        self.data_buffer.clear()

# Usage
anomaly_model = load_pretrained_anomaly_model()
iot_processor = IoTSensorProcessor(anomaly_model)

# Simulate incoming sensor data
for i in range(1500):
    sensor_id = f"sensor_{i % 10}"
    timestamp = time.time()
    value = np.random.normal(100, 10)  # Simulated sensor reading
    
    is_anomaly = iot_processor.process_reading(sensor_id, timestamp, value)
    
    if is_anomaly:
        print(f"Anomaly detected for {sensor_id} at {timestamp}: {value}")

# Ensure final batch is processed
if iot_processor.data_buffer:
    iot_processor.batch_process()
```

Slide 12: Optimizing Inference Performance

Optimizing inference performance is crucial for both batch and real-time scenarios. Let's explore some techniques to improve inference speed and efficiency using PyTorch as an example.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Assume we have a pre-trained model
model = load_pretrained_model()

# 1. Use GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 2. Enable inference mode
model.eval()

# 3. Use torch.no_grad() to disable gradient computation
def optimized_inference(model, input_data):
    with torch.no_grad():
        return model(input_data)

# 4. Batch processing
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 5. Mixed precision inference
if torch.cuda.is_available():
    model = model.half()  # Convert model to half precision

# 6. Model quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)

# Example usage
for batch in dataloader:
    batch = batch.to(device)
    if torch.cuda.is_available():
        batch = batch.half()  # Convert input to half precision
    outputs = optimized_inference(model, batch)
    # Process outputs...

# Benchmark
import time

def benchmark_inference(model, input_data, num_runs=100):
    start_time = time.time()
    for _ in range(num_runs):
        _ = optimized_inference(model, input_data)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    print(f"Average inference time: {avg_time:.4f} seconds")

# Run benchmark
sample_input = torch.randn(1, 3, 224, 224).to(device)
if torch.cuda.is_available():
    sample_input = sample_input.half()
benchmark_inference(model, sample_input)
```

Slide 13: Handling Drift in Batch and Real-Time Inference

Data drift and concept drift can significantly impact model performance over time. Let's implement a simple drift detection system for both batch and real-time scenarios.

```python
import numpy as np
from scipy import stats

class DriftDetector:
    def __init__(self, reference_data, threshold=0.05):
        self.reference_data = reference_data
        self.threshold = threshold

    def detect_drift(self, new_data):
        # Perform Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(self.reference_data, new_data)
        return p_value < self.threshold

class BatchInferenceWithDriftDetection:
    def __init__(self, model, drift_detector):
        self.model = model
        self.drift_detector = drift_detector

    def process_batch(self, batch_data):
        # Check for drift
        if self.drift_detector.detect_drift(batch_data):
            print("Drift detected in batch data!")
            # Trigger model retraining or adaptation
            self.adapt_model(batch_data)
        
        # Perform inference
        return self.model.predict(batch_data)

    def adapt_model(self, new_data):
        # Implement model adaptation strategy
        pass

class RealTimeInferenceWithDriftDetection:
    def __init__(self, model, drift_detector, window_size=1000):
        self.model = model
        self.drift_detector = drift_detector
        self.window_size = window_size
        self.recent_data = []

    def process_sample(self, sample):
        # Add sample to recent data
        self.recent_data.append(sample)
        if len(self.recent_data) > self.window_size:
            self.recent_data.pop(0)

        # Check for drift periodically
        if len(self.recent_data) == self.window_size:
            if self.drift_detector.detect_drift(self.recent_data):
                print("Drift detected in real-time data!")
                # Trigger model retraining or adaptation
                self.adapt_model(self.recent_data)

        # Perform inference
        return self.model.predict([sample])[0]

    def adapt_model(self, new_data):
        # Implement model adaptation strategy
        pass

# Usage example
reference_data = np.random.normal(0, 1, 10000)
drift_detector = DriftDetector(reference_data)

model = load_pretrained_model()
batch_inference = BatchInferenceWithDriftDetection(model, drift_detector)
real_time_inference = RealTimeInferenceWithDriftDetection(model, drift_detector)

# Simulate batch inference
batch_data = np.random.normal(0, 1, 1000)
batch_results = batch_inference.process_batch(batch_data)

# Simulate real-time inference
for _ in range(2000):
    sample = np.random.normal(0, 1)
    result = real_time_inference.process_sample(sample)
```

Slide 14: Monitoring and Logging in Inference Systems

Effective monitoring and logging are essential for maintaining the health and performance of inference systems. Let's implement a simple monitoring and logging system for both batch and real-time inference.

```python
import logging
import time
from prometheus_client import start_http_server, Summary, Counter, Gauge

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up Prometheus metrics
BATCH_INFERENCE_TIME = Summary('batch_inference_seconds', 'Time spent processing a batch')
REAL_TIME_INFERENCE_TIME = Summary('real_time_inference_seconds', 'Time spent on a single real-time inference')
INFERENCE_REQUESTS = Counter('inference_requests_total', 'Total number of inference requests')
MODEL_VERSION = Gauge('model_version', 'Current model version in use')

class MonitoredInferenceSystem:
    def __init__(self, model):
        self.model = model
        self.model_version = 1
        MODEL_VERSION.set(self.model_version)

    @BATCH_INFERENCE_TIME.time()
    def batch_inference(self, batch_data):
        INFERENCE_REQUESTS.inc(len(batch_data))
        start_time = time.time()
        results = self.model.predict(batch_data)
        end_time = time.time()
        
        logging.info(f"Processed batch of {len(batch_data)} items in {end_time - start_time:.2f} seconds")
        return results

    @REAL_TIME_INFERENCE_TIME.time()
    def real_time_inference(self, sample):
        INFERENCE_REQUESTS.inc()
        start_time = time.time()
        result = self.model.predict([sample])[0]
        end_time = time.time()
        
        logging.info(f"Processed real-time inference in {end_time - start_time:.4f} seconds")
        return result

    def update_model(self, new_model):
        self.model = new_model
        self.model_version += 1
        MODEL_VERSION.set(self.model_version)
        logging.info(f"Model updated to version {self.model_version}")

# Start Prometheus HTTP server
start_http_server(8000)

# Usage
model = load_pretrained_model()
inference_system = MonitoredInferenceSystem(model)

# Simulate batch inference
batch_data = generate_batch_data(1000)
batch_results = inference_system.batch_inference(batch_data)

# Simulate real-time inference
for _ in range(100):
    sample = generate_sample()
    result = inference_system.real_time_inference(sample)

# Simulate model update
new_model = train_new_model()
inference_system.update_model(new_model)

# Note: In a real-world scenario, you would typically run this system
# continuously and have a separate process for collecting and analyzing
# the metrics exposed by Prometheus.
```

Slide 15: Additional Resources

For further exploration of batch and real-time inference techniques, consider the following resources:

1. "Serving Machine Learning Models: A Guide to Architecture, Stream Processing Engines, and Frameworks" (arXiv:1712.06139) [https://arxiv.org/abs/1712.06139](https://arxiv.org/abs/1712.06139)
2. "MLOps: Continuous delivery and automation pipelines in machine learning" (arXiv:2006.01113) [https://arxiv.org/abs/2006.01113](https://arxiv.org/abs/2006.01113)
3. "Efficient Processing of Deep Neural Networks: A Tutorial and Survey" (arXiv:1703.09039) [https://arxiv.org/abs/1703.09039](https://arxiv.org/abs/1703.09039)

These papers provide in-depth discussions on model serving architectures, MLOps practices, and optimization techniques for neural network inference.

