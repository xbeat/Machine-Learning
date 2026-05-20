## Fractal Patterns in Neural Network Hyperparameter Grids
Slide 1: Fractal Patterns in Neural Network Hyperparameter Grids

Neural networks are complex systems with numerous hyperparameters. When performing a grid search over these parameters, unexpected patterns can emerge. This presentation explores the fascinating world of fractal patterns found in hyperparameter landscapes and how to visualize them using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_fractal_grid(size):
    grid = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            c = complex(i / size - 1.5, j / size - 1)
            z = 0
            for _ in range(100):
                if abs(z) > 2:
                    break
                z = z**2 + c
            grid[i, j] = abs(z)
    return grid

fractal = generate_fractal_grid(1000)
plt.imshow(fractal, cmap='hot', extent=[-2, 1, -1.5, 1.5])
plt.title('Mandelbrot Set: A Classic Fractal')
plt.colorbar(label='Absolute value of z')
plt.show()
```

Slide 2: Understanding Hyperparameter Landscapes

Hyperparameter landscapes represent the performance of a neural network across different hyperparameter configurations. These landscapes can be surprisingly complex, often exhibiting fractal-like properties. Let's create a simple function to simulate a hyperparameter landscape.

```python
import numpy as np
import matplotlib.pyplot as plt

def hyperparameter_landscape(x, y):
    return np.sin(10 * x) * np.cos(10 * y) * np.exp(-x**2 - y**2)

x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = hyperparameter_landscape(X, Y)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='Performance')
plt.title('Simulated Hyperparameter Landscape')
plt.xlabel('Hyperparameter 1')
plt.ylabel('Hyperparameter 2')
plt.show()
```

Slide 3: Grid Search Implementation

Grid search is a common method for hyperparameter tuning. Let's implement a basic grid search function and visualize its results.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

def grid_search(param_grid, performance_func):
    results = []
    for params in ParameterGrid(param_grid):
        score = performance_func(**params)
        results.append((params, score))
    return results

def performance_func(learning_rate, hidden_units):
    return np.sin(10 * learning_rate) * np.cos(hidden_units) * np.exp(-learning_rate**2 - (hidden_units/100)**2)

param_grid = {
    'learning_rate': np.linspace(0.001, 0.1, 50),
    'hidden_units': np.linspace(10, 100, 50)
}

results = grid_search(param_grid, performance_func)

X = [r[0]['learning_rate'] for r in results]
Y = [r[0]['hidden_units'] for r in results]
Z = [r[1] for r in results]

plt.figure(figsize=(10, 8))
plt.scatter(X, Y, c=Z, cmap='viridis')
plt.colorbar(label='Performance')
plt.title('Grid Search Results')
plt.xlabel('Learning Rate')
plt.ylabel('Hidden Units')
plt.show()
```

Slide 4: Identifying Fractal Patterns

Fractal patterns in hyperparameter landscapes often manifest as self-similar structures across different scales. Let's create a function to zoom into a specific region of our hyperparameter space and observe potential fractal-like behavior.

```python
import numpy as np
import matplotlib.pyplot as plt

def zoom_hyperparameter_landscape(x_range, y_range, resolution=1000):
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = hyperparameter_landscape(X, Y)
    return X, Y, Z

def plot_zoomed_landscape(x_range, y_range, title):
    X, Y, Z = zoom_hyperparameter_landscape(x_range, y_range)
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Performance')
    plt.title(title)
    plt.xlabel('Hyperparameter 1')
    plt.ylabel('Hyperparameter 2')
    plt.show()

# Plot full landscape
plot_zoomed_landscape((-2, 2), (-2, 2), 'Full Hyperparameter Landscape')

# Plot zoomed-in region
plot_zoomed_landscape((-0.5, 0.5), (-0.5, 0.5), 'Zoomed Hyperparameter Landscape')
```

Slide 5: Fractal Dimension Calculation

To quantify the fractal-like nature of our hyperparameter landscape, we can calculate its fractal dimension using the box-counting method. This technique involves counting the number of boxes needed to cover the fractal at different scales.

```python
import numpy as np
import matplotlib.pyplot as plt

def box_count(Z, threshold):
    return np.sum(Z > threshold)

def fractal_dimension(Z, thresholds):
    box_counts = [box_count(Z, t) for t in thresholds]
    coeffs = np.polyfit(np.log(1/thresholds), np.log(box_counts), 1)
    return coeffs[0]

X, Y, Z = zoom_hyperparameter_landscape((-2, 2), (-2, 2))
thresholds = np.logspace(-3, 0, 20)
fd = fractal_dimension(Z, thresholds)

plt.figure(figsize=(10, 8))
plt.loglog(1/thresholds, [box_count(Z, t) for t in thresholds], 'bo-')
plt.title(f'Box Counting: Fractal Dimension ≈ {fd:.2f}')
plt.xlabel('1 / Threshold')
plt.ylabel('Box Count')
plt.show()
```

Slide 6: Real-Life Example: Image Classification

Let's explore how fractal patterns might emerge in a real-world scenario of tuning a convolutional neural network for image classification. We'll use a simplified model and dataset for illustration.

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

def create_model(conv_filters, dense_units):
    model = Sequential([
        Conv2D(conv_filters, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

def evaluate_model(conv_filters, dense_units):
    model = create_model(conv_filters, dense_units)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=1, validation_split=0.2, verbose=0)
    return history.history['val_accuracy'][-1]

# Perform grid search
conv_filters = np.linspace(8, 64, 20, dtype=int)
dense_units = np.linspace(32, 256, 20, dtype=int)
results = np.array([[evaluate_model(cf, du) for du in dense_units] for cf in conv_filters])

plt.figure(figsize=(10, 8))
plt.imshow(results, cmap='viridis', aspect='auto', extent=[32, 256, 8, 64])
plt.colorbar(label='Validation Accuracy')
plt.title('CNN Hyperparameter Landscape')
plt.xlabel('Dense Units')
plt.ylabel('Conv Filters')
plt.show()
```

Slide 7: Analyzing the CNN Hyperparameter Landscape

The previous slide's grid search results reveal interesting patterns in the hyperparameter space of our CNN model. Let's analyze these patterns and their implications for model tuning.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Assuming 'results' is available from the previous slide
smoothed_results = gaussian_filter(results, sigma=1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(results, cmap='viridis', aspect='auto', extent=[32, 256, 8, 64])
plt.colorbar(label='Validation Accuracy')
plt.title('Original Landscape')
plt.xlabel('Dense Units')
plt.ylabel('Conv Filters')

plt.subplot(1, 2, 2)
plt.imshow(smoothed_results, cmap='viridis', aspect='auto', extent=[32, 256, 8, 64])
plt.colorbar(label='Smoothed Validation Accuracy')
plt.title('Smoothed Landscape')
plt.xlabel('Dense Units')
plt.ylabel('Conv Filters')

plt.tight_layout()
plt.show()

# Calculate gradient magnitude
gy, gx = np.gradient(smoothed_results)
gradient_mag = np.sqrt(gx**2 + gy**2)

plt.figure(figsize=(10, 8))
plt.imshow(gradient_mag, cmap='hot', aspect='auto', extent=[32, 256, 8, 64])
plt.colorbar(label='Gradient Magnitude')
plt.title('Gradient Magnitude of Smoothed Landscape')
plt.xlabel('Dense Units')
plt.ylabel('Conv Filters')
plt.show()
```

Slide 8: Fractal Analysis of CNN Hyperparameter Landscape

To investigate the potential fractal nature of our CNN hyperparameter landscape, we'll apply fractal analysis techniques to the gradient magnitude of the smoothed results.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def box_count(image, box_size):
    return np.sum(image.reshape(-1, box_size, image.shape[1]//box_size, box_size).any(axis=(1,3)))

def fractal_dimension(image, box_sizes):
    counts = [box_count(image, size) for size in box_sizes]
    coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)
    return -coeffs[0]

# Assuming 'gradient_mag' is available from the previous slide
threshold = np.mean(gradient_mag)
binary_image = gradient_mag > threshold

box_sizes = np.arange(2, 20, 2)
fd = fractal_dimension(binary_image, box_sizes)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(binary_image, cmap='binary', aspect='auto', extent=[32, 256, 8, 64])
plt.title('Thresholded Gradient Magnitude')
plt.xlabel('Dense Units')
plt.ylabel('Conv Filters')

plt.subplot(1, 2, 2)
counts = [box_count(binary_image, size) for size in box_sizes]
slope, intercept, r_value, p_value, std_err = linregress(np.log(box_sizes), np.log(counts))
plt.loglog(box_sizes, counts, 'bo-')
plt.loglog(box_sizes, np.exp(intercept + slope * np.log(box_sizes)), 'r--')
plt.title(f'Box Counting: D ≈ {-slope:.2f}')
plt.xlabel('Box Size')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

print(f"Estimated fractal dimension: {fd:.2f}")
```

Slide 9: Interpreting Fractal Patterns in Hyperparameter Space

The fractal analysis of our CNN hyperparameter landscape reveals interesting insights about the model's behavior. Let's interpret these findings and discuss their implications for hyperparameter tuning strategies.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Assuming 'results' is available from previous slides
accuracy_range = np.max(results) - np.min(results)
normalized_results = (results - np.min(results)) / accuracy_range

def local_complexity(image, window_size):
    pad_width = window_size // 2
    padded = np.pad(image, pad_width, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, (window_size, window_size))
    return np.std(windows, axis=(2, 3))

complexity = local_complexity(normalized_results, 5)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(normalized_results, cmap='viridis', aspect='auto', extent=[32, 256, 8, 64])
plt.colorbar(label='Normalized Accuracy')
plt.title('Normalized CNN Performance')
plt.xlabel('Dense Units')
plt.ylabel('Conv Filters')

plt.subplot(1, 2, 2)
plt.imshow(complexity, cmap='plasma', aspect='auto', extent=[32, 256, 8, 64])
plt.colorbar(label='Local Complexity')
plt.title('Local Complexity of Performance Landscape')
plt.xlabel('Dense Units')
plt.ylabel('Conv Filters')

plt.tight_layout()
plt.show()

correlation, _ = pearsonr(normalized_results.flatten(), complexity.flatten())
print(f"Correlation between performance and local complexity: {correlation:.2f}")
```

Slide 10: Optimizing Hyperparameters with Fractal Insights

Understanding the fractal nature of hyperparameter landscapes can inform more efficient optimization strategies. Let's implement a simple adaptive grid search that leverages this knowledge.

```python
import numpy as np
import matplotlib.pyplot as plt

def adaptive_grid_search(param_ranges, performance_func, iterations=5, points_per_dim=10):
    results = []
    
    for i in range(iterations):
        grid_points = [np.linspace(r[0], r[1], points_per_dim) for r in param_ranges]
        X, Y = np.meshgrid(*grid_points)
        Z = np.array([performance_func(x, y) for x, y in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
        
        results.append((X, Y, Z))
        best_idx = np.unravel_index(np.argmax(Z), Z.shape)
        
        param_ranges = [
            (max(r[0], grid_points[j][max(0, best_idx[j]-1)]),
             min(r[1], grid_points[j][min(points_per_dim-1, best_idx[j]+1)]))
            for j, r in enumerate(param_ranges)
        ]
    
    return results

def plot_adaptive_search(results):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, (X, Y, Z) in enumerate(results[:6]):
        ax = axes[i//3, i%3]
        c = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
        ax.set_title(f'Iteration {i+1}')
        fig.colorbar(c, ax=ax)
    plt.tight_layout()
    plt.show()

def performance_func(x, y):
    return np.sin(10*x) * np.cos(10*y) * np.exp(-x**2 - y**2)

initial_ranges = [(-2, 2), (-2, 2)]
search_results = adaptive_grid_search(initial_ranges, performance_func)
plot_adaptive_search(search_results)
```

Slide 11: Real-Life Example: Natural Language Processing

Let's explore how fractal patterns might emerge in a natural language processing task, such as text classification using a simple neural network.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load and preprocess data
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(newsgroups.data).toarray()
y = newsgroups.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_model(hidden_units, learning_rate):
    model = Sequential([
        Dense(hidden_units, activation='relu', input_shape=(5000,)),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(hidden_units, learning_rate):
    model = create_model(hidden_units, learning_rate)
    history = model.fit(X_train, y_train, epochs=5, validation_split=0.2, verbose=0)
    return history.history['val_accuracy'][-1]

hidden_units = np.linspace(32, 256, 20, dtype=int)
learning_rates = np.logspace(-4, -1, 20)
results = np.array([[evaluate_model(hu, lr) for lr in learning_rates] for hu in hidden_units])

plt.figure(figsize=(10, 8))
plt.imshow(results, cmap='viridis', aspect='auto', extent=[-4, -1, 32, 256])
plt.colorbar(label='Validation Accuracy')
plt.title('NLP Model Hyperparameter Landscape')
plt.xlabel('Log Learning Rate')
plt.ylabel('Hidden Units')
plt.show()
```

Slide 12: Analyzing NLP Hyperparameter Landscape

The previous slide's grid search results reveal interesting patterns in the hyperparameter space of our NLP model. Let's analyze these patterns and their implications for model tuning.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Assuming 'results' is available from the previous slide
smoothed_results = gaussian_filter(results, sigma=1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(results, cmap='viridis', aspect='auto', extent=[-4, -1, 32, 256])
plt.colorbar(label='Validation Accuracy')
plt.title('Original Landscape')
plt.xlabel('Log Learning Rate')
plt.ylabel('Hidden Units')

plt.subplot(1, 2, 2)
plt.imshow(smoothed_results, cmap='viridis', aspect='auto', extent=[-4, -1, 32, 256])
plt.colorbar(label='Smoothed Validation Accuracy')
plt.title('Smoothed Landscape')
plt.xlabel('Log Learning Rate')
plt.ylabel('Hidden Units')

plt.tight_layout()
plt.show()

# Calculate gradient magnitude
gy, gx = np.gradient(smoothed_results)
gradient_mag = np.sqrt(gx**2 + gy**2)

plt.figure(figsize=(10, 8))
plt.imshow(gradient_mag, cmap='hot', aspect='auto', extent=[-4, -1, 32, 256])
plt.colorbar(label='Gradient Magnitude')
plt.title('Gradient Magnitude of Smoothed Landscape')
plt.xlabel('Log Learning Rate')
plt.ylabel('Hidden Units')
plt.show()
```

Slide 13: Fractal Analysis of NLP Hyperparameter Landscape

To investigate the potential fractal nature of our NLP hyperparameter landscape, we'll apply fractal analysis techniques to the gradient magnitude of the smoothed results.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def box_count(image, box_size):
    return np.sum(image.reshape(-1, box_size, image.shape[1]//box_size, box_size).any(axis=(1,3)))

def fractal_dimension(image, box_sizes):
    counts = [box_count(image, size) for size in box_sizes]
    coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)
    return -coeffs[0]

# Assuming 'gradient_mag' is available from the previous slide
threshold = np.mean(gradient_mag)
binary_image = gradient_mag > threshold

box_sizes = np.arange(2, 20, 2)
fd = fractal_dimension(binary_image, box_sizes)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(binary_image, cmap='binary', aspect='auto', extent=[-4, -1, 32, 256])
plt.title('Thresholded Gradient Magnitude')
plt.xlabel('Log Learning Rate')
plt.ylabel('Hidden Units')

plt.subplot(1, 2, 2)
counts = [box_count(binary_image, size) for size in box_sizes]
slope, intercept, r_value, p_value, std_err = linregress(np.log(box_sizes), np.log(counts))
plt.loglog(box_sizes, counts, 'bo-')
plt.loglog(box_sizes, np.exp(intercept + slope * np.log(box_sizes)), 'r--')
plt.title(f'Box Counting: D ≈ {-slope:.2f}')
plt.xlabel('Box Size')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

print(f"Estimated fractal dimension: {fd:.2f}")
```

Slide 14: Implications and Future Directions

The fractal analysis of neural network hyperparameter landscapes reveals complex patterns that can inform optimization strategies. Future research directions include:

1. Developing adaptive optimization algorithms that leverage fractal properties
2. Investigating the relationship between model architecture and fractal dimension of hyperparameter landscapes
3. Exploring the impact of dataset characteristics on the fractal nature of hyperparameter spaces
4. Applying fractal analysis to other machine learning models and tasks

These insights can lead to more efficient hyperparameter tuning methods and a deeper understanding of neural network behavior.

Slide 15: Additional Resources

For those interested in further exploring the intersection of fractals and machine learning, consider the following resources:

1. "Fractal Geometry of Nature" by Benoit Mandelbrot (1982) - A seminal work on fractals and their prevalence in natural systems.
2. "Scaling Laws in Deep Learning" by J. Kaplan et al. (2020) - ArXiv: [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361) This paper discusses how neural network performance scales with model size, dataset size, and compute budget.
3. "Neural Tangent Kernel: Convergence and Generalization in Neural Networks" by A. Jacot et al. (2018) - ArXiv: [https://arxiv.org/abs/1806.07572](https://arxiv.org/abs/1806.07572) This work provides theoretical insights into the behavior of neural networks in the limit of infinite width.
4. "Visualizing the Loss Landscape of Neural Nets" by H. Li et al. (2018) - ArXiv: [https://arxiv.org/abs/1712.09913](https://arxiv.org/abs/1712.09913) This paper introduces techniques for visualizing the loss landscapes of neural networks, which can exhibit fractal-like properties.

These resources offer a mix of foundational knowledge and cutting-edge research in the field of neural network optimization and analysis.

