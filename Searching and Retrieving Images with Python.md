## Searching and Retrieving Images with Python
Slide 1: Image Search and Retrieval in Python

Python offers powerful tools for searching and retrieving images from datasets. This presentation will guide you through the process, from setting up your environment to implementing advanced search techniques.

```python
import os
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Example: Load an image and convert it to a numpy array
image_path = "path/to/your/image.jpg"
img = Image.open(image_path)
img_array = np.array(img)

print(f"Image shape: {img_array.shape}")
print(f"Image data type: {img_array.dtype}")
```

Slide 2: Setting Up Your Environment

Before diving into image search, ensure you have the necessary libraries installed. We'll be using Python Imaging Library (PIL), NumPy, and scikit-learn for our examples.

```python
# Install required libraries
!pip install Pillow numpy scikit-learn

# Verify installations
import PIL
import numpy
import sklearn

print(f"PIL version: {PIL.__version__}")
print(f"NumPy version: {numpy.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
```

Slide 3: Loading and Preprocessing Images

To search through an image dataset, we first need to load and preprocess the images. This involves reading image files, resizing them to a consistent dimension, and converting them to a format suitable for comparison.

```python
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return img_array.flatten()  # Flatten the 3D array to 1D

# Example usage
image_path = "path/to/your/image.jpg"
processed_image = load_and_preprocess_image(image_path)
print(f"Processed image shape: {processed_image.shape}")
```

Slide 4: Building an Image Dataset

To search through multiple images, we need to create a dataset. We'll walk through a directory, process each image, and store the results in a list along with their file paths.

```python
def build_image_dataset(directory):
    dataset = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                processed_image = load_and_preprocess_image(image_path)
                dataset.append((image_path, processed_image))
    return dataset

# Example usage
dataset_directory = "path/to/your/image/directory"
image_dataset = build_image_dataset(dataset_directory)
print(f"Dataset size: {len(image_dataset)} images")
```

Slide 5: Implementing a Basic Search Function

Now that we have our dataset, let's implement a basic search function using cosine similarity. This function will compare a query image against all images in the dataset and return the most similar ones.

```python
def search_similar_images(query_image, dataset, top_n=5):
    query_vector = load_and_preprocess_image(query_image)
    similarities = []
    
    for path, image_vector in dataset:
        similarity = cosine_similarity([query_vector], [image_vector])[0][0]
        similarities.append((path, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Example usage
query_image_path = "path/to/query/image.jpg"
results = search_similar_images(query_image_path, image_dataset)

for path, similarity in results:
    print(f"Image: {path}, Similarity: {similarity:.4f}")
```

Slide 6: Enhancing Search with Feature Extraction

To improve our search results, we can use more advanced feature extraction techniques. Let's use a pre-trained convolutional neural network (CNN) to extract meaningful features from our images.

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

model = VGG16(weights='imagenet', include_top=False)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

# Update our dataset with CNN features
cnn_dataset = [(path, extract_features(path)) for path, _ in image_dataset]
```

Slide 7: Implementing Advanced Search

With our CNN-based feature extraction, we can now implement a more advanced search function that leverages these high-level features for better results.

```python
def advanced_search(query_image, dataset, top_n=5):
    query_features = extract_features(query_image)
    similarities = []
    
    for path, features in dataset:
        similarity = cosine_similarity([query_features], [features])[0][0]
        similarities.append((path, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Example usage
query_image_path = "path/to/query/image.jpg"
results = advanced_search(query_image_path, cnn_dataset)

for path, similarity in results:
    print(f"Image: {path}, Similarity: {similarity:.4f}")
```

Slide 8: Real-Life Example: Content-Based Image Retrieval System

Let's create a content-based image retrieval system for a digital art gallery. This system will help users find artworks similar to a given piece.

```python
class ArtGallerySearch:
    def __init__(self, gallery_directory):
        self.dataset = self.build_gallery_dataset(gallery_directory)
    
    def build_gallery_dataset(self, directory):
        return [(path, extract_features(path)) for path in glob.glob(f"{directory}/*.jpg")]
    
    def find_similar_artworks(self, query_artwork, top_n=5):
        return advanced_search(query_artwork, self.dataset, top_n)

# Usage
gallery = ArtGallerySearch("path/to/art/gallery")
similar_artworks = gallery.find_similar_artworks("path/to/query/artwork.jpg")

for artwork, similarity in similar_artworks:
    print(f"Similar artwork: {artwork}, Similarity: {similarity:.4f}")
```

Slide 9: Optimizing Search Performance

As your image dataset grows, search performance may become a concern. Let's explore some optimization techniques to speed up our search process.

```python
import faiss

def build_faiss_index(dataset):
    features = np.array([features for _, features in dataset])
    dimension = features.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(features.astype('float32'))
    return index

def faiss_search(query_features, index, dataset, top_n=5):
    D, I = index.search(query_features.astype('float32').reshape(1, -1), top_n)
    return [(dataset[i][0], 1 / (1 + d)) for d, i in zip(D[0], I[0])]

# Build FAISS index
faiss_index = build_faiss_index(cnn_dataset)

# Perform fast search
query_image_path = "path/to/query/image.jpg"
query_features = extract_features(query_image_path)
results = faiss_search(query_features, faiss_index, cnn_dataset)

for path, similarity in results:
    print(f"Image: {path}, Similarity: {similarity:.4f}")
```

Slide 10: Handling Large-Scale Image Datasets

When dealing with millions of images, we need to consider distributed processing and storage solutions. Let's explore using Apache Spark for distributed image processing.

```python
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName("LargeScaleImageSearch").getOrCreate()

def process_image_spark(row):
    path = row['path']
    features = extract_features(path)
    return (path, Vectors.dense(features))

# Create a Spark DataFrame of image paths
image_df = spark.createDataFrame([(path,) for path, _ in image_dataset], ['path'])

# Process images in parallel
processed_df = image_df.rdd.map(process_image_spark).toDF(['path', 'features'])

# Perform similarity search using Spark
def spark_similarity_search(query_features, processed_df, top_n=5):
    broadcast_query = spark.sparkContext.broadcast(query_features)
    
    def compute_similarity(row):
        path = row['path']
        features = row['features']
        similarity = cosine_similarity([broadcast_query.value], [features.toArray()])[0][0]
        return (path, similarity)
    
    return processed_df.rdd.map(compute_similarity).top(top_n, key=lambda x: x[1])

# Example usage
query_image_path = "path/to/query/image.jpg"
query_features = extract_features(query_image_path)
results = spark_similarity_search(query_features, processed_df)

for path, similarity in results:
    print(f"Image: {path}, Similarity: {similarity:.4f}")
```

Slide 11: Real-Life Example: Satellite Image Analysis

Let's create a system to analyze satellite images and find areas with similar geographical features. This could be useful for urban planning or environmental monitoring.

```python
import rasterio
from rasterio.plot import show

class SatelliteImageAnalysis:
    def __init__(self, image_directory):
        self.dataset = self.build_satellite_dataset(image_directory)
    
    def build_satellite_dataset(self, directory):
        dataset = []
        for image_path in glob.glob(f"{directory}/*.tif"):
            with rasterio.open(image_path) as src:
                image_array = src.read()
                features = self.extract_satellite_features(image_array)
                dataset.append((image_path, features))
        return dataset
    
    def extract_satellite_features(self, image_array):
        # Simplified feature extraction (you might want to use more advanced techniques)
        return np.mean(image_array, axis=(1, 2))
    
    def find_similar_areas(self, query_image_path, top_n=5):
        with rasterio.open(query_image_path) as src:
            query_array = src.read()
            query_features = self.extract_satellite_features(query_array)
        
        similarities = []
        for path, features in self.dataset:
            similarity = cosine_similarity([query_features], [features])[0][0]
            similarities.append((path, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

# Usage
satellite_analysis = SatelliteImageAnalysis("path/to/satellite/images")
similar_areas = satellite_analysis.find_similar_areas("path/to/query/satellite_image.tif")

for area, similarity in similar_areas:
    print(f"Similar area: {area}, Similarity: {similarity:.4f}")
    with rasterio.open(area) as src:
        show(src)
```

Slide 12: Incorporating User Feedback

To improve search results over time, we can incorporate user feedback. Let's implement a simple relevance feedback mechanism.

```python
class ImageSearchWithFeedback:
    def __init__(self, dataset):
        self.dataset = dataset
        self.user_preferences = {}
    
    def search(self, query_image, top_n=5):
        query_features = extract_features(query_image)
        similarities = []
        
        for path, features in self.dataset:
            similarity = cosine_similarity([query_features], [features])[0][0]
            user_preference = self.user_preferences.get(path, 1.0)
            adjusted_similarity = similarity * user_preference
            similarities.append((path, adjusted_similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    
    def update_preferences(self, image_path, relevance):
        # relevance: 1 for relevant, -1 for irrelevant
        current_preference = self.user_preferences.get(image_path, 1.0)
        self.user_preferences[image_path] = max(0.1, min(2.0, current_preference + 0.1 * relevance))

# Usage
search_engine = ImageSearchWithFeedback(cnn_dataset)
results = search_engine.search("path/to/query/image.jpg")

# User provides feedback
search_engine.update_preferences(results[0][0], 1)  # Mark first result as relevant
search_engine.update_preferences(results[1][0], -1)  # Mark second result as irrelevant

# Perform search again with updated preferences
updated_results = search_engine.search("path/to/query/image.jpg")
```

Slide 13: Conclusion and Future Directions

We've explored various techniques for image search and retrieval using Python, from basic similarity measures to advanced CNN-based feature extraction and distributed processing. As the field evolves, consider exploring these future directions:

1. Multimodal search: Combining image and text data for more contextual searches.
2. Self-supervised learning: Training models on unlabeled image data for better feature extraction.
3. Federated learning: Collaboratively training models across decentralized devices while preserving privacy.
4. Quantum computing: Leveraging quantum algorithms for faster similarity computations on large-scale datasets.

Slide 14: Additional Resources

For further exploration of image search and retrieval techniques, consider these peer-reviewed articles:

1. "A Survey of Content-Based Image Retrieval with High-Level Semantics" by Liu et al. (2007) ArXiv: [https://arxiv.org/abs/0707.1217](https://arxiv.org/abs/0707.1217)
2. "Deep Learning for Content-Based Image Retrieval: A Comprehensive Study" by Wan et al. (2014) ArXiv: [https://arxiv.org/abs/1406.4774](https://arxiv.org/abs/1406.4774)
3. "Visual Search at Pinterest" by Jing et al. (2015) ArXiv: [https://arxiv.org/abs/1505.07647](https://arxiv.org/abs/1505.07647)

These resources provide in-depth discussions on advanced topics and real-world applications of image search and retrieval systems.

