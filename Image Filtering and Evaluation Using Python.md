## Image Filtering and Evaluation Using Python
Slide 1: Introduction to Image Filtering and MSE

Image filtering is a fundamental technique in image processing used to enhance or modify digital images. Mean Squared Error (MSE) is a common metric for evaluating the quality of filtered images. This presentation will explore various image filtering techniques and how to use MSE for quality assessment using Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters

# Load an example image
image = io.imread('example_image.jpg')

# Display the original image
plt.imshow(image)
plt.title('Original Image')
plt.show()
```

Slide 2: Gaussian Blur Filter

Gaussian blur is a widely used image smoothing filter. It reduces noise and detail by convolving the image with a Gaussian function. This filter is particularly effective for reducing high-frequency noise while preserving edges.

```python
# Apply Gaussian blur
sigma = 2.0  # Standard deviation of the Gaussian distribution
blurred = filters.gaussian(image, sigma=sigma, multichannel=True)

# Display the blurred image
plt.imshow(blurred)
plt.title(f'Gaussian Blur (sigma={sigma})')
plt.show()
```

Slide 3: Median Filter

The median filter is excellent for removing salt-and-pepper noise while preserving edges. It replaces each pixel with the median value of its neighboring pixels within a specified window size.

```python
from scipy import ndimage

# Apply median filter
window_size = 3
median_filtered = ndimage.median_filter(image, size=window_size)

# Display the median filtered image
plt.imshow(median_filtered)
plt.title(f'Median Filter (window size={window_size})')
plt.show()
```

Slide 4: Sobel Edge Detection

Sobel edge detection is used to emphasize edges in an image. It calculates the gradient of image intensity at each pixel, highlighting areas of rapid intensity change.

```python
# Apply Sobel edge detection
edges = filters.sobel(image)

# Display the edge-detected image
plt.imshow(edges, cmap='gray')
plt.title('Sobel Edge Detection')
plt.show()
```

Slide 5: Mean Squared Error (MSE)

MSE is a metric used to quantify the difference between an original image and its processed version. It calculates the average squared difference between corresponding pixel values.

```python
def calculate_mse(original, processed):
    return np.mean((original - processed) ** 2)

# Calculate MSE for Gaussian blur
mse_gaussian = calculate_mse(image, blurred)
print(f"MSE for Gaussian blur: {mse_gaussian:.2f}")

# Calculate MSE for median filter
mse_median = calculate_mse(image, median_filtered)
print(f"MSE for median filter: {mse_median:.2f}")
```

Slide 6: Comparing Filters Using MSE

We can use MSE to compare the effectiveness of different filters. Lower MSE values indicate that the filtered image is more similar to the original.

```python
# Apply different levels of Gaussian blur
sigmas = [1.0, 2.0, 3.0, 4.0]
mse_values = []

for sigma in sigmas:
    blurred = filters.gaussian(image, sigma=sigma, multichannel=True)
    mse = calculate_mse(image, blurred)
    mse_values.append(mse)

# Plot MSE vs. sigma
plt.plot(sigmas, mse_values, marker='o')
plt.xlabel('Sigma')
plt.ylabel('MSE')
plt.title('MSE vs. Gaussian Blur Sigma')
plt.show()
```

Slide 7: Real-life Example: Noise Reduction in Medical Imaging

Medical images often contain noise that can interfere with diagnosis. Applying filters and evaluating their performance using MSE can help improve image quality for more accurate medical assessments.

```python
# Simulate a noisy medical image
medical_image = io.imread('brain_scan.jpg', as_gray=True)
noisy_image = medical_image + np.random.normal(0, 0.1, medical_image.shape)

# Apply Gaussian filter
filtered_image = filters.gaussian(noisy_image, sigma=1.5)

# Calculate MSE
mse = calculate_mse(medical_image, filtered_image)

# Display results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(medical_image, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(noisy_image, cmap='gray')
axs[1].set_title('Noisy Image')
axs[2].imshow(filtered_image, cmap='gray')
axs[2].set_title(f'Filtered Image (MSE: {mse:.4f})')
plt.show()
```

Slide 8: Real-life Example: Enhancing Satellite Imagery

Satellite images often require enhancement to improve visibility and extract useful information. Filters can be applied to reduce atmospheric haze or enhance specific features, with MSE used to evaluate the effectiveness of different techniques.

```python
# Load a satellite image
satellite_image = io.imread('satellite_image.jpg')

# Apply contrast enhancement
enhanced_image = filters.unsharp_mask(satellite_image, radius=5, amount=2)

# Calculate MSE
mse = calculate_mse(satellite_image, enhanced_image)

# Display results
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(satellite_image)
axs[0].set_title('Original Satellite Image')
axs[1].imshow(enhanced_image)
axs[1].set_title(f'Enhanced Image (MSE: {mse:.4f})')
plt.show()
```

Slide 9: Implementing Custom Filters

Python allows for easy implementation of custom filters. Here's an example of a simple sharpening filter and its evaluation using MSE.

```python
def sharpen_filter(image):
    blurred = filters.gaussian(image, sigma=1, multichannel=True)
    return image + (image - blurred)

# Apply custom sharpening filter
sharpened = sharpen_filter(image)

# Calculate MSE
mse = calculate_mse(image, sharpened)

# Display results
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[1].imshow(sharpened)
axs[1].set_title(f'Sharpened Image (MSE: {mse:.4f})')
plt.show()
```

Slide 10: MSE Limitations and Alternatives

While MSE is widely used, it has limitations. It doesn't always correlate well with human perception of image quality. Alternative metrics like Structural Similarity Index (SSIM) can provide more perceptually relevant evaluations.

```python
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(original, processed):
    return ssim(original, processed, multichannel=True)

# Calculate SSIM for Gaussian blur
ssim_value = calculate_ssim(image, blurred)
print(f"SSIM for Gaussian blur: {ssim_value:.4f}")

# Compare MSE and SSIM
mse = calculate_mse(image, blurred)
print(f"MSE for Gaussian blur: {mse:.4f}")
```

Slide 11: Batch Processing and MSE Evaluation

In real-world scenarios, we often need to process and evaluate multiple images. Here's an example of batch processing with MSE evaluation.

```python
import os

def process_image_batch(folder_path, filter_func):
    mse_values = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = io.imread(image_path)
            processed = filter_func(image)
            mse = calculate_mse(image, processed)
            mse_values.append(mse)
    return mse_values

# Example usage
folder_path = 'image_folder'
mse_values = process_image_batch(folder_path, lambda img: filters.gaussian(img, sigma=2))

# Plot MSE distribution
plt.hist(mse_values, bins=20)
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.title('MSE Distribution for Batch Processing')
plt.show()
```

Slide 12: Optimizing Filter Parameters

We can use MSE to optimize filter parameters. This example demonstrates finding the optimal sigma value for Gaussian blur.

```python
def optimize_gaussian_sigma(image, sigma_range):
    mse_values = []
    for sigma in sigma_range:
        blurred = filters.gaussian(image, sigma=sigma, multichannel=True)
        mse = calculate_mse(image, blurred)
        mse_values.append(mse)
    optimal_sigma = sigma_range[np.argmin(mse_values)]
    return optimal_sigma, mse_values

sigma_range = np.linspace(0.1, 5, 50)
optimal_sigma, mse_values = optimize_gaussian_sigma(image, sigma_range)

plt.plot(sigma_range, mse_values)
plt.xlabel('Sigma')
plt.ylabel('MSE')
plt.title(f'Optimal Sigma: {optimal_sigma:.2f}')
plt.show()
```

Slide 13: Conclusion

Image filtering and evaluation using MSE are essential techniques in image processing. They allow us to enhance images, reduce noise, and quantitatively assess the quality of processed images. While MSE has limitations, it remains a valuable tool when used in conjunction with other metrics and visual inspection.

```python
# Final demonstration: Apply optimal Gaussian blur
optimal_blurred = filters.gaussian(image, sigma=optimal_sigma, multichannel=True)
final_mse = calculate_mse(image, optimal_blurred)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[1].imshow(optimal_blurred)
axs[1].set_title(f'Optimal Gaussian Blur (MSE: {final_mse:.4f})')
plt.show()
```

Slide 14: Additional Resources

For further exploration of image filtering and evaluation techniques, consider the following resources:

1. "Image Quality Assessment: From Error Visibility to Structural Similarity" by Z. Wang et al. (2004) ArXiv: [https://arxiv.org/abs/cs/0308034](https://arxiv.org/abs/cs/0308034)
2. "A Survey of Recent Advances in CNN-based Single Image Super-Resolution" by W. Yang et al. (2019) ArXiv: [https://arxiv.org/abs/1902.07392](https://arxiv.org/abs/1902.07392)
3. "Deep Learning for Image Super-Resolution: A Survey" by Z. Wang et al. (2020) ArXiv: [https://arxiv.org/abs/1902.06068](https://arxiv.org/abs/1902.06068)

These papers provide in-depth discussions on image quality assessment, super-resolution techniques, and deep learning approaches to image processing, which can complement the concepts covered in this presentation.

