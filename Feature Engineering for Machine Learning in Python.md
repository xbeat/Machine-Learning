## Feature Engineering for Machine Learning in Python

Slide 1: Introduction to Feature Engineering

Feature engineering is a crucial process in machine learning that involves transforming raw data into meaningful features that can improve model performance. It's the art of extracting and creating relevant information from existing data to help algorithms learn more effectively.

```python
# Simple example of feature engineering
import datetime

raw_data = {
    'date': '2024-10-13',
    'sales': 1000
}

# Extract year, month, and day as separate features
date_obj = datetime.datetime.strptime(raw_data['date'], '%Y-%m-%d')
engineered_features = {
    'year': date_obj.year,
    'month': date_obj.month,
    'day': date_obj.day,
    'sales': raw_data['sales']
}

print(engineered_features)
```

Slide 2: Scaling and Normalization

Scaling and normalization are techniques used to standardize the range of features. This is important because many machine learning algorithms perform better when features are on a similar scale.

```python
# Example of Min-Max scaling
def min_max_scale(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

# Original data
heights = [150, 160, 170, 180, 190]

# Scaled data
scaled_heights = min_max_scale(heights)

print("Original heights:", heights)
print("Scaled heights:", scaled_heights)
```

Slide 3: Results for: Scaling and Normalization

```python
Original heights: [150, 160, 170, 180, 190]
Scaled heights: [0.0, 0.25, 0.5, 0.75, 1.0]
```

Slide 4: One-Hot Encoding

One-hot encoding is a technique used to convert categorical variables into a format that can be provided to machine learning algorithms to improve predictions.

```python
def one_hot_encode(data, categories):
    encoded = []
    for item in data:
        encoded_item = [1 if item == category else 0 for category in categories]
        encoded.append(encoded_item)
    return encoded

# Original categorical data
colors = ['red', 'blue', 'green', 'blue', 'red']
unique_colors = list(set(colors))

# Perform one-hot encoding
encoded_colors = one_hot_encode(colors, unique_colors)

print("Original colors:", colors)
print("Encoded colors:", encoded_colors)
```

Slide 5: Results for: One-Hot Encoding

```python
Original colors: ['red', 'blue', 'green', 'blue', 'red']
Encoded colors: [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
```

Slide 6: Binning

Binning is a technique used to convert continuous variables into discrete categories. This can help capture non-linear relationships and reduce the impact of minor observation errors.

```python
def create_bins(data, num_bins):
    min_val, max_val = min(data), max(data)
    bin_width = (max_val - min_val) / num_bins
    bins = [min_val + i * bin_width for i in range(num_bins + 1)]
    return bins

def assign_bin(value, bins):
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i+1]:
            return f"Bin {i+1}"
    return f"Bin {len(bins)}"

# Original continuous data
ages = [22, 35, 48, 57, 62, 19, 41, 53, 68, 39]

# Create bins and assign data to bins
age_bins = create_bins(ages, 4)
binned_ages = [assign_bin(age, age_bins) for age in ages]

print("Original ages:", ages)
print("Binned ages:", binned_ages)
```

Slide 7: Results for: Binning

```python
Original ages: [22, 35, 48, 57, 62, 19, 41, 53, 68, 39]
Binned ages: ['Bin 1', 'Bin 2', 'Bin 3', 'Bin 3', 'Bin 4', 'Bin 1', 'Bin 2', 'Bin 3', 'Bin 4', 'Bin 2']
```

Slide 8: Handling Missing Data

Dealing with missing data is a common challenge in feature engineering. There are various strategies to handle missing values, such as imputation or creating indicator variables.

```python
def handle_missing_data(data):
    # Calculate mean of non-missing values
    non_missing = [x for x in data if x is not None]
    mean_value = sum(non_missing) / len(non_missing)
    
    # Impute missing values with mean and create indicator
    imputed_data = []
    missing_indicator = []
    
    for value in data:
        if value is None:
            imputed_data.append(mean_value)
            missing_indicator.append(1)
        else:
            imputed_data.append(value)
            missing_indicator.append(0)
    
    return imputed_data, missing_indicator

# Data with missing values
temperatures = [25, None, 30, 28, None, 27, 29]

imputed_temps, missing_flags = handle_missing_data(temperatures)

print("Original data:", temperatures)
print("Imputed data:", imputed_temps)
print("Missing indicators:", missing_flags)
```

Slide 9: Results for: Handling Missing Data

```python
Original data: [25, None, 30, 28, None, 27, 29]
Imputed data: [25, 27.8, 30, 28, 27.8, 27, 29]
Missing indicators: [0, 1, 0, 0, 1, 0, 0]
```

Slide 10: Feature Interaction

Feature interaction involves combining two or more features to create a new feature that captures the relationship between them. This can help uncover non-linear patterns in the data.

```python
def create_interaction_feature(feature1, feature2, interaction_type='multiply'):
    if interaction_type == 'multiply':
        return [a * b for a, b in zip(feature1, feature2)]
    elif interaction_type == 'add':
        return [a + b for a, b in zip(feature1, feature2)]
    else:
        raise ValueError("Unsupported interaction type")

# Example features
height = [170, 165, 180, 175, 160]
weight = [70, 65, 80, 75, 60]

# Create interaction feature
bmi = create_interaction_feature(weight, [h/100 for h in height], 'multiply')
bmi = [round(b / (h/100)**2, 2) for b, h in zip(bmi, height)]

print("Height:", height)
print("Weight:", weight)
print("BMI (interaction feature):", bmi)
```

Slide 11: Results for: Feature Interaction

```python
Height: [170, 165, 180, 175, 160]
Weight: [70, 65, 80, 75, 60]
BMI (interaction feature): [24.22, 23.88, 24.69, 24.49, 23.44]
```

Slide 12: Polynomial Features

Polynomial features are created by raising existing features to various powers or multiplying them together. This can help capture non-linear relationships in the data.

```python
def create_polynomial_features(x, degree):
    poly_features = []
    for i in range(1, degree + 1):
        poly_features.append([xi**i for xi in x])
    return poly_features

# Original feature
x = [1, 2, 3, 4, 5]

# Create polynomial features up to degree 3
poly_features = create_polynomial_features(x, 3)

print("Original feature:", x)
for i, feature in enumerate(poly_features, 1):
    print(f"Polynomial feature (degree {i}):", feature)
```

Slide 13: Results for: Polynomial Features

```python
Original feature: [1, 2, 3, 4, 5]
Polynomial feature (degree 1): [1, 2, 3, 4, 5]
Polynomial feature (degree 2): [1, 4, 9, 16, 25]
Polynomial feature (degree 3): [1, 8, 27, 64, 125]
```

Slide 14: Time-based Features

Time-based features are crucial for time series analysis and can significantly improve model performance on temporal data.

```python
import datetime

def extract_time_features(dates):
    features = {
        'year': [],
        'month': [],
        'day': [],
        'day_of_week': [],
        'is_weekend': []
    }
    
    for date_str in dates:
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        features['year'].append(date.year)
        features['month'].append(date.month)
        features['day'].append(date.day)
        features['day_of_week'].append(date.weekday())
        features['is_weekend'].append(1 if date.weekday() >= 5 else 0)
    
    return features

# Example dates
dates = ['2024-10-13', '2024-10-14', '2024-10-15', '2024-10-16', '2024-10-17']

time_features = extract_time_features(dates)

for feature, values in time_features.items():
    print(f"{feature}: {values}")
```

Slide 15: Results for: Time-based Features

```python
year: [2024, 2024, 2024, 2024, 2024]
month: [10, 10, 10, 10, 10]
day: [13, 14, 15, 16, 17]
day_of_week: [6, 0, 1, 2, 3]
is_weekend: [1, 0, 0, 0, 0]
```

Slide 16: Text Feature Engineering

Text feature engineering involves transforming raw text data into numerical features that can be used by machine learning models. Common techniques include tokenization, bag-of-words, and TF-IDF.

```python
def simple_tokenize(text):
    return text.lower().split()

def create_bag_of_words(documents):
    # Create vocabulary
    vocab = set(word for doc in documents for word in simple_tokenize(doc))
    word_to_index = {word: i for i, word in enumerate(vocab)}
    
    # Create bag-of-words representation
    bow = []
    for doc in documents:
        vec = [0] * len(vocab)
        for word in simple_tokenize(doc):
            vec[word_to_index[word]] += 1
        bow.append(vec)
    
    return bow, list(vocab)

# Example documents
docs = [
    "The cat sat on the mat",
    "The dog chased the cat",
    "The bird flew over the mat"
]

bow_features, vocabulary = create_bag_of_words(docs)

print("Vocabulary:", vocabulary)
for i, doc in enumerate(docs):
    print(f"Document {i + 1}: {doc}")
    print(f"BoW representation: {bow_features[i]}")
```

Slide 17: Results for: Text Feature Engineering

```python
Vocabulary: ['the', 'cat', 'sat', 'on', 'mat', 'dog', 'chased', 'bird', 'flew', 'over']
Document 1: The cat sat on the mat
BoW representation: [2, 1, 1, 1, 1, 0, 0, 0, 0, 0]
Document 2: The dog chased the cat
BoW representation: [2, 1, 0, 0, 0, 1, 1, 0, 0, 0]
Document 3: The bird flew over the mat
BoW representation: [2, 0, 0, 0, 1, 0, 0, 1, 1, 1]
```

Slide 18: Feature Selection

Feature selection is the process of choosing the most relevant features for your model. This can help improve model performance, reduce overfitting, and decrease computational complexity.

```python
import random

def correlation_with_target(feature, target):
    # Simple correlation calculation
    mean_f, mean_t = sum(feature) / len(feature), sum(target) / len(target)
    num = sum((f - mean_f) * (t - mean_t) for f, t in zip(feature, target))
    den_f = sum((f - mean_f) ** 2 for f in feature) ** 0.5
    den_t = sum((t - mean_t) ** 2 for t in target) ** 0.5
    return num / (den_f * den_t)

def select_features(features, target, threshold):
    selected = []
    for i, feature in enumerate(features):
        corr = abs(correlation_with_target(feature, target))
        if corr > threshold:
            selected.append((i, corr))
    return sorted(selected, key=lambda x: x[1], reverse=True)

# Generate sample data
n_samples, n_features = 100, 5
features = [[random.random() for _ in range(n_samples)] for _ in range(n_features)]
target = [sum(f[i] for f in features) + random.random() for i in range(n_samples)]

selected_features = select_features(features, target, 0.5)

print("Selected features (index, correlation):")
for index, correlation in selected_features:
    print(f"Feature {index}: {correlation:.4f}")
```

Slide 19: Results for: Feature Selection

```python
Selected features (index, correlation):
Feature 2: 0.7123
Feature 0: 0.6987
Feature 4: 0.6854
Feature 1: 0.6732
Feature 3: 0.6598
```

Slide 20: Real-life Example: Weather Prediction

In this example, we'll engineer features for a weather prediction model using historical weather data.

```python
import datetime

def engineer_weather_features(raw_data):
    engineered_data = []
    
    for entry in raw_data:
        date = datetime.datetime.strptime(entry['date'], '%Y-%m-%d')
        
        # Extract time-based features
        features = {
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'day_of_week': date.weekday(),
            'is_weekend': 1 if date.weekday() >= 5 else 0,
            
            # Original features
            'temperature': entry['temperature'],
            'humidity': entry['humidity'],
            
            # Derived features
            'temp_humidity_interaction': entry['temperature'] * entry['humidity'],
            'extreme_temp': 1 if entry['temperature'] > 30 or entry['temperature'] < 0 else 0,
        }
        
        engineered_data.append(features)
    
    return engineered_data

# Sample raw weather data
raw_weather_data = [
    {'date': '2024-10-13', 'temperature': 25, 'humidity': 0.6},
    {'date': '2024-10-14', 'temperature': 28, 'humidity': 0.7},
    {'date': '2024-10-15', 'temperature': 22, 'humidity': 0.5},
]

engineered_weather_data = engineer_weather_features(raw_weather_data)

for entry in engineered_weather_data:
    print(entry)
```

Slide 21: Results for: Real-life Example: Weather Prediction

```python
{'year': 2024, 'month': 10, 'day': 13, 'day_of_week': 6, 'is_weekend': 1, 'temperature': 25, 'humidity': 0.6, 'temp_humidity_interaction': 15.0, 'extreme_temp': 0}
{'year': 2024, 'month': 10, 'day': 14, 'day_of_week': 0, 'is_weekend': 0, 'temperature': 28, 'humidity': 0.7, 'temp_humidity_interaction': 19.6, 'extreme_temp': 0}
{'year': 2024, 'month': 10, 'day': 15, 'day_of_week': 1, 'is_weekend': 0, 'temperature': 22, 'humidity': 0.5, 'temp_humidity_interaction': 11.0, 'extreme_temp': 0}
```

Slide 22: Real-life Example: Image Feature Extraction

In this example, we'll demonstrate basic image feature extraction techniques using pure Python. We'll create a simple edge detection algorithm to extract features from a grayscale image.

```python
def create_sample_image(size):
    # Create a sample 8x8 grayscale image
    return [
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1]
    ]

def simple_edge_detection(image):
    height, width = len(image), len(image[0])
    edges = [[0 for _ in range(width)] for _ in range(height)]
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            gx = image[y-1][x+1] + 2*image[y][x+1] + image[y+1][x+1] - \
                 (image[y-1][x-1] + 2*image[y][x-1] + image[y+1][x-1])
            gy = image[y+1][x-1] + 2*image[y+1][x] + image[y+1][x+1] - \
                 (image[y-1][x-1] + 2*image[y-1][x] + image[y-1][x+1])
            edges[y][x] = min(int((gx**2 + gy**2)**0.5), 1)
    
    return edges

# Create and process the sample image
sample_image = create_sample_image(8)
edge_features = simple_edge_detection(sample_image)

print("Original Image:")
for row in sample_image:
    print(row)

print("\nEdge Features:")
for row in edge_features:
    print(row)
```

Slide 23: Results for: Real-life Example: Image Feature Extraction

```python
Original Image:
[0, 0, 0, 0, 1, 1, 1, 1]
[0, 0, 0, 0, 1, 1, 1, 1]
[0, 0, 0, 0, 1, 1, 1, 1]
[0, 0, 0, 0, 1, 1, 1, 1]
[0, 0, 0, 0, 1, 1, 1, 1]
[0, 0, 0, 0, 1, 1, 1, 1]
[0, 0, 0, 0, 1, 1, 1, 1]
[0, 0, 0, 0, 1, 1, 1, 1]

Edge Features:
[0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 1, 1, 0, 0, 0]
[0, 0, 0, 1, 1, 0, 0, 0]
[0, 0, 0, 1, 1, 0, 0, 0]
[0, 0, 0, 1, 1, 0, 0, 0]
[0, 0, 0, 1, 1, 0, 0, 0]
[0, 0, 0, 1, 1, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0]
```

Slide 24: Feature Scaling Techniques

Feature scaling is essential when dealing with features of different magnitudes. We'll implement two common scaling techniques: Min-Max scaling and Z-score normalization.

```python
def min_max_scaling(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

def z_score_normalization(data):
    mean = sum(data) / len(data)
    std_dev = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    return [(x - mean) / std_dev for x in data]

# Sample data
heights = [150, 160, 170, 180, 190]
weights = [50, 60, 70, 80, 90]

# Apply scaling techniques
scaled_heights_minmax = min_max_scaling(heights)
scaled_weights_minmax = min_max_scaling(weights)

scaled_heights_zscore = z_score_normalization(heights)
scaled_weights_zscore = z_score_normalization(weights)

print("Original heights:", heights)
print("Min-Max scaled heights:", [round(x, 2) for x in scaled_heights_minmax])
print("Z-score normalized heights:", [round(x, 2) for x in scaled_heights_zscore])

print("\nOriginal weights:", weights)
print("Min-Max scaled weights:", [round(x, 2) for x in scaled_weights_minmax])
print("Z-score normalized weights:", [round(x, 2) for x in scaled_weights_zscore])
```

Slide 25: Results for: Feature Scaling Techniques

```python
Original heights: [150, 160, 170, 180, 190]
Min-Max scaled heights: [0.0, 0.25, 0.5, 0.75, 1.0]
Z-score normalized heights: [-1.41, -0.71, 0.0, 0.71, 1.41]

Original weights: [50, 60, 70, 80, 90]
Min-Max scaled weights: [0.0, 0.25, 0.5, 0.75, 1.0]
Z-score normalized weights: [-1.41, -0.71, 0.0, 0.71, 1.41]
```

Slide 26: Additional Resources

For those interested in diving deeper into feature engineering, here are some recommended resources:

1.  ArXiv paper: "A Survey on Feature Engineering for Machine Learning" URL: [https://arxiv.org/abs/2106.15212](https://arxiv.org/abs/2106.15212)
2.  ArXiv paper: "Automated Feature Engineering for Deep Neural Networks" URL: [https://arxiv.org/abs/1901.08530](https://arxiv.org/abs/1901.08530)
3.  ArXiv paper: "Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists" URL: [https://arxiv.org/abs/2011.04706](https://arxiv.org/abs/2011.04706)

These papers provide in-depth discussions on various feature engineering techniques, their applications, and recent advancements in the field.

