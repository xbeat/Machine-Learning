## Mastering Analytics Frameworks with Python
Slide 1: Introduction to Analytics Frameworks

Analytics frameworks provide a structured approach to data analysis, enabling data scientists to extract meaningful insights from complex datasets. These frameworks encompass various methodologies, tools, and best practices that guide the entire analytics process, from data collection to interpretation and decision-making.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and preview a dataset
data = pd.read_csv('example_dataset.csv')
print(data.head())

# Basic statistical summary
print(data.describe())

# Visualize data distribution
plt.figure(figsize=(10, 6))
data.hist()
plt.tight_layout()
plt.show()
```

Slide 2: The Data Analytics Lifecycle

The data analytics lifecycle is a fundamental framework that outlines the stages of an analytics project. It typically includes phases such as problem definition, data collection, data preparation, exploratory data analysis, modeling, and deployment.

```python
# Simplified representation of the data analytics lifecycle
lifecycle_stages = ['Problem Definition', 'Data Collection', 'Data Preparation',
                    'Exploratory Analysis', 'Modeling', 'Deployment']

# Visualize the lifecycle
plt.figure(figsize=(12, 6))
plt.plot(lifecycle_stages, 'bo-')
plt.title('Data Analytics Lifecycle')
plt.ylabel('Progress')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Slide 3: Exploratory Data Analysis (EDA)

EDA is a crucial step in understanding the underlying patterns, relationships, and anomalies in your data. It involves using various statistical and visualization techniques to gain insights before formal modeling.

```python
import seaborn as sns

# Load a sample dataset
tips = sns.load_dataset('tips')

# Pairplot for multivariate analysis
sns.pairplot(tips, hue='time')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(tips.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

Slide 4: Feature Engineering

Feature engineering is the process of creating new features or transforming existing ones to improve model performance. It requires domain knowledge and creativity to extract meaningful information from raw data.

```python
from sklearn.preprocessing import PolynomialFeatures

# Create sample data
X = np.array([[1, 2], [3, 4], [5, 6]])

# Generate polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

print("Original features:")
print(X)
print("\nPolynomial features:")
print(X_poly)
```

Slide 5: Dimensionality Reduction

Dimensionality reduction techniques help in managing high-dimensional datasets by projecting data onto a lower-dimensional space while preserving important information. Principal Component Analysis (PCA) is a commonly used method.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X = iris.data

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize results
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
plt.title('PCA of Iris Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
```

Slide 6: Supervised Learning: Classification

Classification is a supervised learning task where the goal is to predict categorical labels. Decision trees are interpretable models often used for classification tasks.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

# Train a decision tree
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Visualize the tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=iris.feature_names, 
          class_names=iris.target_names, filled=True)
plt.show()
```

Slide 7: Supervised Learning: Regression

Regression is used to predict continuous numerical values. Linear regression is a simple yet powerful technique for understanding relationships between variables.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = 2 * X + 1 + np.random.randn(5, 1)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean squared error: {mse:.2f}")
print(f"R-squared score: {r2:.2f}")

# Plot results
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.legend()
plt.title('Linear Regression Example')
plt.show()
```

Slide 8: Unsupervised Learning: Clustering

Clustering is an unsupervised learning technique used to group similar data points. K-means is a popular algorithm for partitioning data into distinct clusters.

```python
from sklearn.cluster import KMeans

# Generate sample data
X = np.random.randn(300, 2)
X[:100, :] += 3

# Perform K-means clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='r')
plt.title('K-means Clustering')
plt.show()
```

Slide 9: Time Series Analysis

Time series analysis involves studying data points collected over time to identify trends, seasonality, and other patterns. It's crucial for forecasting and understanding temporal dependencies.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate sample time series data
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
ts = pd.Series(np.random.randn(len(dates)).cumsum() + 20, index=dates)

# Add seasonality
ts += np.sin(np.arange(len(ts)) * 2 * np.pi / 365) * 10

# Perform seasonal decomposition
result = seasonal_decompose(ts, model='additive', period=365)

# Plot components
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
result.observed.plot(ax=ax1)
ax1.set_title('Observed')
result.trend.plot(ax=ax2)
ax2.set_title('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()
```

Slide 10: Natural Language Processing (NLP)

NLP is a branch of AI that focuses on the interaction between computers and human language. It involves tasks such as text classification, sentiment analysis, and named entity recognition.

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.
"""

# Tokenize and remove stopwords
tokens = word_tokenize(text.lower())
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

# Count word frequencies
word_freq = Counter(filtered_tokens)

# Plot top 10 most common words
plt.figure(figsize=(12, 6))
word_freq.most_common(10)[::-1]
plt.barh(*zip(*word_freq.most_common(10)))
plt.title('Top 10 Most Common Words')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.show()
```

Slide 11: Deep Learning with Neural Networks

Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model complex patterns in data. It has revolutionized fields such as computer vision and natural language processing.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a simple neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model architecture
model.summary()

# Visualize the model
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
img = plt.imread('model.png')
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
```

Slide 12: Model Evaluation and Validation

Proper evaluation and validation of machine learning models are crucial to ensure their generalization and reliability. Techniques like cross-validation and performance metrics help assess model quality.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load a dataset
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

# Create a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform 5-fold cross-validation
scores = cross_val_score(rf, X, y, cv=5)

print("Cross-validation scores:", scores)
print("Mean accuracy: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std() * 2))

# Plot cross-validation results
plt.figure(figsize=(8, 6))
plt.boxplot(scores)
plt.title('5-Fold Cross-Validation Results')
plt.ylabel('Accuracy')
plt.show()
```

Slide 13: Real-Life Example: Weather Prediction

Weather prediction is a common application of data science and machine learning. It involves analyzing historical weather data to forecast future conditions.

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Generate sample weather data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
temperature = pd.Series(np.random.randn(len(dates)).cumsum() + 20, index=dates)
temperature += 10 * np.sin(np.arange(len(temperature)) * 2 * np.pi / 365)  # Add seasonality

# Fit ARIMA model
model = ARIMA(temperature, order=(1, 1, 1))
results = model.fit()

# Forecast next 30 days
forecast = results.forecast(steps=30)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(temperature.index, temperature, label='Historical')
plt.plot(forecast.index, forecast, color='red', label='Forecast')
plt.title('Temperature Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
```

Slide 14: Real-Life Example: Image Classification

Image classification is widely used in various applications, from facial recognition to medical diagnosis. Here's a simple example using a pre-trained model.

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load and preprocess an image
img_path = 'example_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make predictions
preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]

# Display results
plt.imshow(img)
plt.axis('off')
plt.title('Predictions:')
for i, (imagenet_id, label, score) in enumerate(decoded_preds):
    plt.text(10, 10 + i * 20, f"{label}: {score:.2f}", fontsize=14, color='white', 
             bbox=dict(facecolor='black', alpha=0.8))
plt.show()
```

Slide 15: Additional Resources

For further exploration of analytics frameworks and data science techniques, consider the following resources:

1. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (Available on ArXiv: [https://arxiv.org/abs/1601.06615](https://arxiv.org/abs/1601.06615))
2. "A Survey of Deep Learning Techniques for Neural Machine Translation" by Shuohang Wang and Jing Jiang (ArXiv: [https://arxiv.org/abs/1804.09139](https://arxiv.org/abs/1804.09139))
3. "Machine Learning: A Probabilistic Perspective" by Kevin P. Murphy
4. Python Data Science Handbook by Jake VanderPlas (Available online: [https://jakevdp.github.io/PythonDataScienceHandbook/](https://jakevdp.github.io/PythonDataScienceHandbook/))
5. Scikit-learn documentation ([https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html))

