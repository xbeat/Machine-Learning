## Integrating ML Models in FastAPI with Python
Slide 1: Introduction to ML Model Integration in FastAPI

FastAPI is a modern, fast web framework for building APIs with Python. Integrating machine learning models into FastAPI allows for easy deployment and scalability of ML-powered applications. This slideshow will guide you through the process of integrating an ML model into a FastAPI application.

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load pre-trained model
model = joblib.load('model.joblib')

class InputData(BaseModel):
    feature1: float
    feature2: float

@app.post("/predict")
async def predict(data: InputData):
    features = [[data.feature1, data.feature2]]
    prediction = model.predict(features)
    return {"prediction": prediction[0]}
```

Slide 2: Setting Up the FastAPI Application

To begin, we need to set up a basic FastAPI application. This involves importing the necessary modules, creating a FastAPI instance, and defining a simple endpoint.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the ML Model API"}

# Run the app with: uvicorn main:app --reload
```

Slide 3: Loading the ML Model

Before we can make predictions, we need to load our pre-trained ML model. We'll use joblib to load the model from a file.

```python
import joblib

# Load the pre-trained model
model = joblib.load('path/to/your/model.joblib')

# Example of how the model might have been saved
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# joblib.dump(model, 'path/to/your/model.joblib')
```

Slide 4: Creating Input Data Models

To ensure proper data validation, we'll use Pydantic to create input data models. This helps in automatically validating incoming requests.

```python
from pydantic import BaseModel

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

# Example usage:
# input_data = InputData(feature1=1.0, feature2=2.0, feature3=3.0)
```

Slide 5: Implementing the Prediction Endpoint

Now, let's create an endpoint that accepts input data and returns predictions using our ML model.

```python
@app.post("/predict")
async def predict(data: InputData):
    features = [[data.feature1, data.feature2, data.feature3]]
    prediction = model.predict(features)
    return {"prediction": prediction[0]}

# Example request:
# curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}'
```

Slide 6: Handling Multiple Input Samples

Let's extend our API to handle multiple input samples in a single request.

```python
from typing import List

class InputDataBatch(BaseModel):
    samples: List[InputData]

@app.post("/predict_batch")
async def predict_batch(data: InputDataBatch):
    features = [[sample.feature1, sample.feature2, sample.feature3] for sample in data.samples]
    predictions = model.predict(features)
    return {"predictions": predictions.tolist()}

# Example request:
# curl -X POST "http://localhost:8000/predict_batch" -H "Content-Type: application/json" -d '{"samples": [{"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}, {"feature1": 4.0, "feature2": 5.0, "feature3": 6.0}]}'
```

Slide 7: Adding Error Handling

To make our API more robust, let's add error handling to manage potential issues during prediction.

```python
from fastapi import HTTPException

@app.post("/predict_with_error_handling")
async def predict_with_error_handling(data: InputData):
    try:
        features = [[data.feature1, data.feature2, data.feature3]]
        prediction = model.predict(features)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Example of how this might be triggered:
# curl -X POST "http://localhost:8000/predict_with_error_handling" -H "Content-Type: application/json" -d '{"feature1": "invalid", "feature2": 2.0, "feature3": 3.0}'
```

Slide 8: Model Versioning

Implementing model versioning allows for easier management of different model iterations.

```python
models = {
    "v1": joblib.load('path/to/model_v1.joblib'),
    "v2": joblib.load('path/to/model_v2.joblib')
}

@app.post("/predict/{version}")
async def predict_versioned(version: str, data: InputData):
    if version not in models:
        raise HTTPException(status_code=404, detail="Model version not found")
    
    model = models[version]
    features = [[data.feature1, data.feature2, data.feature3]]
    prediction = model.predict(features)
    return {"prediction": prediction[0], "model_version": version}

# Example usage:
# curl -X POST "http://localhost:8000/predict/v2" -H "Content-Type: application/json" -d '{"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}'
```

Slide 9: Asynchronous Prediction

For long-running predictions, we can implement asynchronous processing using background tasks.

```python
from fastapi import BackgroundTasks

def process_prediction(data: InputData):
    features = [[data.feature1, data.feature2, data.feature3]]
    prediction = model.predict(features)
    # Store or send the prediction result
    print(f"Prediction result: {prediction[0]}")

@app.post("/predict_async")
async def predict_async(data: InputData, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_prediction, data)
    return {"message": "Prediction task added to queue"}

# Example usage:
# curl -X POST "http://localhost:8000/predict_async" -H "Content-Type: application/json" -d '{"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}'
```

Slide 10: Model Retraining Endpoint

Let's create an endpoint to trigger model retraining, which can be useful for updating the model with new data.

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

@app.post("/retrain")
async def retrain_model(X: List[List[float]], y: List[int]):
    global model
    X_array = np.array(X)
    y_array = np.array(y)
    
    model = RandomForestClassifier()
    model.fit(X_array, y_array)
    
    joblib.dump(model, 'path/to/updated_model.joblib')
    return {"message": "Model retrained and saved successfully"}

# Example usage:
# curl -X POST "http://localhost:8000/retrain" -H "Content-Type: application/json" -d '{"X": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "y": [0, 1]}'
```

Slide 11: Real-life Example: Sentiment Analysis

Let's implement a sentiment analysis endpoint using a pre-trained NLTK model.

```python
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

class TextInput(BaseModel):
    text: str

@app.post("/analyze_sentiment")
async def analyze_sentiment(data: TextInput):
    sentiment_scores = sia.polarity_scores(data.text)
    sentiment = "positive" if sentiment_scores['compound'] > 0 else "negative" if sentiment_scores['compound'] < 0 else "neutral"
    return {"sentiment": sentiment, "scores": sentiment_scores}

# Example usage:
# curl -X POST "http://localhost:8000/analyze_sentiment" -H "Content-Type: application/json" -d '{"text": "I love using FastAPI for ML model deployment!"}'
# Output: {"sentiment": "positive", "scores": {"neg": 0.0, "neu": 0.508, "pos": 0.492, "compound": 0.6369}}
```

Slide 12: Real-life Example: Image Classification

Let's create an endpoint for image classification using a pre-trained ResNet model.

```python
from fastapi import File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/classify_image")
async def classify_image(file: UploadFile = File(...)):
    image = Image.open(file.file)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)
    
    _, predicted_idx = torch.max(output, 1)
    predicted_label = ResNet50_Weights.DEFAULT.meta["categories"][predicted_idx.item()]
    
    return {"predicted_class": predicted_label}

# Example usage:
# curl -X POST "http://localhost:8000/classify_image" -H "Content-Type: multipart/form-data" -F "file=@path/to/your/image.jpg"
```

Slide 13: Monitoring and Logging

Implementing monitoring and logging is crucial for maintaining and debugging your ML-powered API.

```python
import logging
from fastapi import Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@app.post("/predict_with_logging")
async def predict_with_logging(data: InputData):
    logger.info(f"Received input: {data}")
    features = [[data.feature1, data.feature2, data.feature3]]
    prediction = model.predict(features)
    logger.info(f"Prediction result: {prediction[0]}")
    return {"prediction": prediction[0]}

# Example usage:
# curl -X POST "http://localhost:8000/predict_with_logging" -H "Content-Type: application/json" -d '{"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}'
# Check your console or log file for the logged information
```

Slide 14: Additional Resources

1. FastAPI Documentation: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
2. Scikit-learn Documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
3. "Deploying Machine Learning Models with FastAPI" by Adrian Tam: [https://arxiv.org/abs/2110.06380](https://arxiv.org/abs/2110.06380)
4. "A Survey on Machine Learning Model Serving Frameworks" by Xu et al.: [https://arxiv.org/abs/2111.07221](https://arxiv.org/abs/2111.07221)

