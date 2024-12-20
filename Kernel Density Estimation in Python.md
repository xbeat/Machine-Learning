## Claude-Kernel Density Estimation in Python
Slide 1: Introduction to Kernel Density Estimation

Kernel Density Estimation (KDE) is a non-parametric method for estimating the probability density function of a random variable based on a finite data sample. It's a powerful technique used in data analysis and visualization to smooth out the data and reveal underlying patterns.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate sample data
data = np.random.normal(0, 1, 1000)

# Compute KDE
kde = gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 100)
kde_values = kde(x_range)

# Plot the results
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue')
plt.plot(x_range, kde_values, 'r-', linewidth=2)
plt.title('Kernel Density Estimation')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
```

Slide 2: The Kernel Function

The kernel function is a key component of KDE. It's a non-negative function that integrates to one and is symmetric. Common kernel functions include Gaussian, Epanechnikov, and Triangular kernels. The Gaussian kernel is most widely used due to its smooth properties.

```python
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

def epanechnikov_kernel(x):
    return 0.75 * (1 - x**2) * (np.abs(x) <= 1)

def triangular_kernel(x):
    return (1 - np.abs(x)) * (np.abs(x) <= 1)

x = np.linspace(-3, 3, 100)

plt.figure(figsize=(10, 6))
plt.plot(x, gaussian_kernel(x), label='Gaussian')
plt.plot(x, epanechnikov_kernel(x), label='Epanechnikov')
plt.plot(x, triangular_kernel(x), label='Triangular')
plt.legend()
plt.title('Common Kernel Functions')
plt.xlabel('x')
plt.ylabel('K(x)')
plt.show()
```

Slide 3: Bandwidth Selection

Bandwidth is a crucial parameter in KDE that controls the smoothness of the resulting density estimate. A small bandwidth leads to a spiky estimate, while a large bandwidth results in an oversmoothed estimate. Various methods exist for optimal bandwidth selection, including cross-validation and rule-of-thumb estimators.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate sample data
np.random.seed(42)
data = np.concatenate([np.random.normal(-2, 1, 200), np.random.normal(2, 1, 200)])

# Compute KDE with different bandwidths
x_range = np.linspace(data.min(), data.max(), 100)
bandwidths = [0.1, 0.5, 2.0]

plt.figure(figsize=(12, 6))
for bw in bandwidths:
    kde = gaussian_kde(data, bw_method=bw)
    plt.plot(x_range, kde(x_range), label=f'Bandwidth = {bw}')

plt.hist(data, bins=30, density=True, alpha=0.3, color='gray')
plt.legend()
plt.title('Effect of Bandwidth on KDE')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
```

Slide 4: Multivariate KDE

KDE can be extended to multiple dimensions, allowing for the estimation of joint probability density functions. This is particularly useful for visualizing the relationship between two or more variables.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate 2D sample data
np.random.seed(42)
x = np.random.normal(0, 1, 1000)
y = x * 0.5 + np.random.normal(0, 0.5, 1000)

# Compute 2D KDE
xy = np.vstack([x, y])
kde = gaussian_kde(xy)

# Create a grid of points
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])

# Evaluate KDE on the grid
z = kde(positions).reshape(xx.shape)

# Plot the results
plt.figure(figsize=(10, 8))
plt.imshow(np.rot90(z), extent=[xmin, xmax, ymin, ymax], cmap='viridis')
plt.scatter(x, y, alpha=0.1, color='white')
plt.colorbar(label='Density')
plt.title('2D Kernel Density Estimation')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

Slide 5: Adaptive KDE

Adaptive KDE adjusts the bandwidth locally based on the density of data points. This allows for better estimation in regions with varying data density, providing a more accurate representation of the underlying distribution.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def adaptive_kde(data, x_range, pilot_bandwidth=0.5, alpha=0.5):
    pilot_kde = gaussian_kde(data, bw_method=pilot_bandwidth)
    pilot_density = pilot_kde(data)
    adaptive_bw = (pilot_density / np.geometric_mean(pilot_density)) ** (-alpha)
    
    kde_values = np.zeros_like(x_range)
    for i, x in enumerate(x_range):
        local_kde = gaussian_kde(data, bw_method=lambda _: adaptive_bw)
        kde_values[i] = local_kde(x)
    
    return kde_values

# Generate sample data
np.random.seed(42)
data = np.concatenate([np.random.normal(-2, 0.5, 200), np.random.normal(2, 1, 200)])

# Compute adaptive KDE
x_range = np.linspace(data.min(), data.max(), 100)
adaptive_kde_values = adaptive_kde(data, x_range)

# Plot the results
plt.figure(figsize=(12, 6))
plt.hist(data, bins=30, density=True, alpha=0.3, color='gray')
plt.plot(x_range, adaptive_kde_values, 'r-', linewidth=2, label='Adaptive KDE')
plt.plot(x_range, gaussian_kde(data)(x_range), 'b--', linewidth=2, label='Standard KDE')
plt.legend()
plt.title('Adaptive vs Standard KDE')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
```

Slide 6: KDE for Outlier Detection

KDE can be used for outlier detection by identifying data points with low density estimates. This method is particularly useful when the underlying distribution is unknown or non-normal.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate sample data with outliers
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
outliers = np.random.uniform(-5, 5, 20)
data = np.concatenate([normal_data, outliers])

# Compute KDE
kde = gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 100)
kde_values = kde(data)

# Define outlier threshold (e.g., bottom 1% of density values)
threshold = np.percentile(kde_values, 1)

# Identify outliers
outlier_mask = kde_values < threshold

# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(data[~outlier_mask], kde_values[~outlier_mask], alpha=0.5, label='Normal points')
plt.scatter(data[outlier_mask], kde_values[outlier_mask], color='red', label='Outliers')
plt.plot(x_range, kde(x_range), 'k-', linewidth=2, label='KDE')
plt.axhline(threshold, color='r', linestyle='--', label='Outlier threshold')
plt.legend()
plt.title('Outlier Detection using KDE')
plt.xlabel('Value')
plt.ylabel('Density')
plt.yscale('log')
plt.show()
```

Slide 7: KDE for Classification

KDE can be used for classification tasks by estimating the probability density function for each class separately. The class with the highest density at a given point is then assigned as the predicted class.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate sample data for two classes
np.random.seed(42)
class1 = np.random.multivariate_normal([-1, -1], [[1, 0.5], [0.5, 1]], 200)
class2 = np.random.multivariate_normal([1, 1], [[1, -0.5], [-0.5, 1]], 200)

# Compute KDE for each class
kde1 = gaussian_kde(class1.T)
kde2 = gaussian_kde(class2.T)

# Create a grid of points
x, y = np.mgrid[-4:4:100j, -4:4:100j]
positions = np.vstack([x.ravel(), y.ravel()])

# Evaluate KDE for both classes
z1 = kde1(positions).reshape(x.shape)
z2 = kde2(positions).reshape(x.shape)

# Classify based on higher density
classification = np.where(z1 > z2, 1, 2)

# Plot the results
plt.figure(figsize=(10, 8))
plt.contourf(x, y, classification, levels=[0.5, 1.5, 2.5], colors=['skyblue', 'salmon'], alpha=0.5)
plt.scatter(class1[:, 0], class1[:, 1], c='blue', label='Class 1', alpha=0.5)
plt.scatter(class2[:, 0], class2[:, 1], c='red', label='Class 2', alpha=0.5)
plt.legend()
plt.title('KDE-based Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

Slide 8: KDE for Data Smoothing

KDE can be used to smooth noisy data, revealing underlying trends and patterns. This is particularly useful in time series analysis and signal processing.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate noisy time series data
np.random.seed(42)
t = np.linspace(0, 10, 1000)
signal = np.sin(t) + 0.5 * np.sin(5 * t)
noise = np.random.normal(0, 0.2, 1000)
noisy_signal = signal + noise

# Apply KDE for smoothing
kde = gaussian_kde(noisy_signal, bw_method=0.1)
smoothed_signal = kde(noisy_signal)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(t, noisy_signal, 'b.', alpha=0.3, label='Noisy data')
plt.plot(t, smoothed_signal, 'r-', linewidth=2, label='KDE smoothed')
plt.plot(t, signal, 'g--', linewidth=2, label='True signal')
plt.legend()
plt.title('KDE for Data Smoothing')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
```

Slide 9: KDE vs. Histograms

KDE offers several advantages over traditional histograms, including smoothness and independence from bin size and origin. This comparison demonstrates how KDE can provide a more accurate representation of the underlying distribution.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate sample data
np.random.seed(42)
data = np.concatenate([np.random.normal(-2, 1, 300), np.random.normal(2, 1, 700)])

# Compute KDE
kde = gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 100)
kde_values = kde(x_range)

# Plot histograms with different bin sizes and KDE
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
bin_sizes = [10, 20, 50, 100]

for ax, bins in zip(axs.ravel(), bin_sizes):
    ax.hist(data, bins=bins, density=True, alpha=0.7, label=f'Histogram ({bins} bins)')
    ax.plot(x_range, kde_values, 'r-', linewidth=2, label='KDE')
    ax.set_title(f'Histogram ({bins} bins) vs KDE')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()

plt.tight_layout()
plt.show()
```

Slide 10: KDE for Density Ratio Estimation

KDE can be used to estimate the ratio of two probability densities, which is useful in various machine learning tasks such as covariate shift adaptation and anomaly detection.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate sample data from two distributions
np.random.seed(42)
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(1, 1.5, 1000)

# Compute KDE for both distributions
kde1 = gaussian_kde(data1)
kde2 = gaussian_kde(data2)

# Compute density ratio
x_range = np.linspace(-5, 7, 100)
density_ratio = kde1(x_range) / kde2(x_range)

# Plot the results
plt.figure(figsize=(12, 6))
plt.hist(data1, bins=30, density=True, alpha=0.5, label='Distribution 1')
plt.hist(data2, bins=30, density=True, alpha=0.5, label='Distribution 2')
plt.plot(x_range, kde1(x_range), 'b-', linewidth=2, label='KDE 1')
plt.plot(x_range, kde2(x_range), 'r-', linewidth=2, label='KDE 2')
plt.plot(x_range, density_ratio, 'g-', linewidth=2, label='Density Ratio')
plt.legend()
plt.title('Density Ratio Estimation using KDE')
plt.xlabel('Value')
plt.ylabel('Density / Ratio')
plt.show()
```

Slide 11: KDE for Image Processing

KDE can be applied to image processing tasks, such as edge detection and image segmentation. In this example, we'll use KDE to enhance edges in a grayscale image.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from skimage import data, filters

# Load a sample image
image = data.camera()

# Compute gradients
gx, gy = np.gradient(image)
gradient_magnitude = np.sqrt(gx**2 + gy**2)

# Apply KDE to gradient magnitude
kde = gaussian_kde(gradient_magnitude.ravel())
enhanced_edges = kde(gradient_magnitude.ravel()).reshape(image.shape)

# Plot the results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(gradient_magnitude, cmap='gray')
axs[1].set_title('Gradient Magnitude')
axs[1].axis('off')

axs[2].imshow(enhanced_edges, cmap='gray')
axs[2].set_title('KDE Enhanced Edges')
axs[2].axis('off')

plt.tight_layout()
plt.show()
```

Slide 12: KDE for Spatial Analysis

KDE is useful in spatial analysis for creating heat maps and identifying areas of high density. This example demonstrates how to use KDE to visualize the density of points in a 2D space.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate sample spatial data
np.random.seed(42)
x = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(3, 1, 500)])
y = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(3, 1, 500)])

# Perform KDE
xy = np.vstack([x, y])
kde = gaussian_kde(xy)

# Create a grid and compute the density on it
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
xi, yi = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

# Plot the results
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(x, y, alpha=0.5, s=5)
ax.contourf(xi, yi, zi, cmap='viridis', alpha=0.5)
ax.set_title('Spatial Density Estimation using KDE')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
plt.colorbar(ax.contourf(xi, yi, zi, cmap='viridis', alpha=0.5), label='Density')
plt.show()
```

Slide 13: KDE for Probability Density Function Estimation

KDE is particularly useful for estimating probability density functions (PDFs) of unknown distributions. This example compares a KDE estimate with the true PDF of a mixture of normal distributions.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

# Generate sample data from a mixture of two normal distributions
np.random.seed(42)
data = np.concatenate([np.random.normal(-2, 1, 1000), np.random.normal(2, 1, 1000)])

# Define the true PDF
def true_pdf(x):
    return 0.5 * (norm.pdf(x, -2, 1) + norm.pdf(x, 2, 1))

# Compute KDE
kde = gaussian_kde(data)
x_range = np.linspace(-6, 6, 200)

# Plot the results
plt.figure(figsize=(12, 6))
plt.hist(data, bins=50, density=True, alpha=0.5, label='Sample data')
plt.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE estimate')
plt.plot(x_range, true_pdf(x_range), 'g--', linewidth=2, label='True PDF')
plt.legend()
plt.title('KDE for PDF Estimation')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
```

Slide 14: Real-life Example: Air Quality Analysis

KDE can be used to analyze air quality data, helping identify patterns and trends in pollutant concentrations. This example demonstrates how to use KDE to visualize the distribution of PM2.5 concentrations over time.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate sample air quality data (PM2.5 concentrations)
np.random.seed(42)
time = np.linspace(0, 24, 1000)  # 24-hour period
base_concentration = 15 + 5 * np.sin(2 * np.pi * time / 24)  # Daily cycle
noise = np.random.normal(0, 2, 1000)
pm25_concentration = base_concentration + noise

# Compute KDE
kde = gaussian_kde(pm25_concentration)
x_range = np.linspace(pm25_concentration.min(), pm25_concentration.max(), 100)

# Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.scatter(time, pm25_concentration, alpha=0.5, s=5)
ax1.set_title('PM2.5 Concentration over 24 hours')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('PM2.5 Concentration (μg/m³)')

ax2.hist(pm25_concentration, bins=30, density=True, alpha=0.5)
ax2.plot(x_range, kde(x_range), 'r-', linewidth=2)
ax2.set_title('Distribution of PM2.5 Concentrations')
ax2.set_xlabel('PM2.5 Concentration (μg/m³)')
ax2.set_ylabel('Density')

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into Kernel Density Estimation, here are some valuable resources:

1. ArXiv paper: "A Review of Kernel Density Estimation with Applications to Econometrics" by Jianqing Fan and Qiwei Yao ([https://arxiv.org/abs/1212.2812](https://arxiv.org/abs/1212.2812))
2. ArXiv paper: "Kernel Density Estimation via Diffusion" by Z. I. Botev, J. F. Grotowski, and D. P. Kroese ([https://arxiv.org/abs/1011.2602](https://arxiv.org/abs/1011.2602))

These papers provide comprehensive overviews of KDE techniques, applications, and recent developments in the field.

