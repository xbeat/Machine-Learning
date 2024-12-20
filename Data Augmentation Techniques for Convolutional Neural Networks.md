## Data Augmentation Techniques for Convolutional Neural Networks
Slide 1: Data Augmentation in CNNs

Data augmentation is a powerful technique used to increase the diversity of training data for convolutional neural networks (CNNs). It involves creating new training samples by applying various transformations to existing images. This process helps improve model generalization and reduces overfitting, especially when working with limited datasets.

```python
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

# Load MNIST dataset
(X_train, _), (_, _) = mnist.load_data()

# Select a sample image
sample_image = X_train[0]

# Create an ImageDataGenerator instance
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Reshape the image to (1, 28, 28, 1)
sample_image = sample_image.reshape((1, 28, 28, 1))

# Generate augmented images
augmented_images = [sample_image]
for _ in range(5):
    augmented_images.append(datagen.flow(sample_image, batch_size=1)[0])

# Plot original and augmented images
plt.figure(figsize=(10, 2))
for i, img in enumerate(augmented_images):
    plt.subplot(1, 6, i+1)
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.title('Original')
    else:
        plt.title(f'Augmented {i}')
plt.tight_layout()
plt.show()
```

Slide 2: Image Rotation

Rotation is a common augmentation technique that involves rotating the image by a random angle within a specified range. This helps the model become invariant to the orientation of objects in the image.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

# Load a sample image
image = cv2.imread('sample_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate rotated images
angles = [0, 45, 90, 135, 180]
rotated_images = [rotate_image(image, angle) for angle in angles]

# Display the results
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, (img, angle) in enumerate(zip(rotated_images, angles)):
    axes[i].imshow(img)
    axes[i].set_title(f'Rotation: {angle}°')
    axes[i].axis('off')
plt.tight_layout()
plt.show()
```

Slide 3: Horizontal and Vertical Flipping

Flipping is another effective augmentation technique that creates mirror images of the original data. This is particularly useful for objects that can appear in different orientations.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def flip_image(image, flip_code):
    return cv2.flip(image, flip_code)

# Load a sample image
image = cv2.imread('sample_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate flipped images
flipped_horizontal = flip_image(image, 1)
flipped_vertical = flip_image(image, 0)
flipped_both = flip_image(image, -1)

# Display the results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(image)
axes[0, 0].set_title('Original')
axes[0, 1].imshow(flipped_horizontal)
axes[0, 1].set_title('Horizontal Flip')
axes[1, 0].imshow(flipped_vertical)
axes[1, 0].set_title('Vertical Flip')
axes[1, 1].imshow(flipped_both)
axes[1, 1].set_title('Both Flips')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
```

Slide 4: Random Cropping

Random cropping involves selecting a random portion of the image and using it as a new training sample. This technique helps the model focus on different parts of the image and become more robust to partial occlusions.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]
    return crop

# Load a sample image
image = cv2.imread('sample_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate random crops
crops = [random_crop(image, 200, 200) for _ in range(4)]

# Display the results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, crop in enumerate(crops):
    row = i // 2
    col = i % 2
    axes[row, col].imshow(crop)
    axes[row, col].set_title(f'Random Crop {i+1}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

Slide 5: Color Jittering

Color jittering involves randomly altering the brightness, contrast, saturation, and hue of an image. This technique helps the model become more robust to variations in lighting conditions and color balance.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_jitter(image, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Jitter brightness
    brightness_factor = 1.0 + np.random.uniform(-brightness, brightness)
    image[:,:,2] = np.clip(image[:,:,2] * brightness_factor, 0, 255)

    # Jitter contrast
    contrast_factor = 1.0 + np.random.uniform(-contrast, contrast)
    image[:,:,2] = np.clip(((image[:,:,2] - 128) * contrast_factor) + 128, 0, 255)

    # Jitter saturation
    saturation_factor = 1.0 + np.random.uniform(-saturation, saturation)
    image[:,:,1] = np.clip(image[:,:,1] * saturation_factor, 0, 255)

    # Jitter hue
    hue_factor = np.random.uniform(-hue, hue)
    image[:,:,0] = (image[:,:,0] + hue_factor * 180) % 180

    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image.astype(np.uint8)

# Load a sample image
image = cv2.imread('sample_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate color jittered images
jittered_images = [color_jitter(image) for _ in range(4)]

# Display the results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, img in enumerate(jittered_images):
    row = i // 2
    col = i % 2
    axes[row, col].imshow(img)
    axes[row, col].set_title(f'Color Jittered {i+1}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

Slide 6: Gaussian Noise Addition

Adding Gaussian noise to images can help improve the model's robustness to noise in real-world scenarios. This technique simulates imperfections in image capture or transmission.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

# Load a sample image
image = cv2.imread('sample_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate noisy images with different standard deviations
std_devs = [10, 25, 50]
noisy_images = [add_gaussian_noise(image, std=std) for std in std_devs]

# Display the results
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(image)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

for i, (img, std) in enumerate(zip(noisy_images, std_devs)):
    row = (i + 1) // 2
    col = (i + 1) % 2
    axes[row, col].imshow(img)
    axes[row, col].set_title(f'Noise std: {std}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

Slide 7: Elastic Deformation

Elastic deformation is an advanced augmentation technique that applies non-linear transformations to images. This is particularly useful for handwritten digit recognition tasks, as it simulates natural variations in handwriting.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter

def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)

# Load a sample image (assuming it's a grayscale image)
image = cv2.imread('sample_digit.png', 0)

# Generate elastically deformed images
alphas = [10, 30, 50]
sigma = 5
deformed_images = [elastic_transform(image, alpha, sigma) for alpha in alphas]

# Display the results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

for i, (img, alpha) in enumerate(zip(deformed_images, alphas)):
    row = (i + 1) // 2
    col = (i + 1) % 2
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(f'Alpha: {alpha}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

Slide 8: Mixup

Mixup is a data augmentation technique that creates new training samples by linearly interpolating between pairs of images and their labels. This helps the model learn smoother decision boundaries and improve generalization.

```python
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10

def mixup(x1, x2, y1, y2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y

# Load CIFAR-10 dataset
(x_train, y_train), (_, _) = cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0

# Select two random images
idx1, idx2 = np.random.randint(0, len(x_train), 2)
img1, img2 = x_train[idx1], x_train[idx2]
label1, label2 = y_train[idx1], y_train[idx2]

# Apply mixup
mixed_img, mixed_label = mixup(img1, img2, label1, label2)

# Display the results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img1)
axes[0].set_title(f'Image 1 (Label: {label1[0]})')
axes[0].axis('off')

axes[1].imshow(img2)
axes[1].set_title(f'Image 2 (Label: {label2[0]})')
axes[1].axis('off')

axes[2].imshow(mixed_img)
axes[2].set_title(f'Mixed Image (Label: {mixed_label[0]:.2f}, {1-mixed_label[0]:.2f})')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

Slide 9: Random Erasing

Random Erasing is an augmentation technique that randomly selects rectangular regions in an image and replaces them with random noise or a constant value. This helps the model become more robust to occlusions and missing parts in images.

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def random_erasing(image, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=1/0.3):
    if np.random.rand() > p:
        return image
    
    h, w, c = image.shape
    s = np.random.uniform(sl, sh) * h * w
    r = np.random.uniform(r1, r2)
    
    new_h = int(np.sqrt(s / r))
    new_w = int(np.sqrt(s * r))
    
    left = np.random.randint(0, w - new_w)
    top = np.random.randint(0, h - new_h)
    
    erased_area = image[top:top+new_h, left:left+new_w, :]
    erased_area[:] = np.random.randint(0, 256, size=erased_area.shape)
    
    return image

# Load a sample image
image = np.array(Image.open('sample_image.jpg'))

# Apply random erasing multiple times
erased_images = [random_erasing(image.()) for _ in range(4)]

# Display the results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(image)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

for i, img in enumerate(erased_images):
    row = i // 3
    col = i % 3 + 1 if i < 3 else i % 3
    axes[row, col].imshow(img)
    axes[row, col].set_title(f'Erased {i+1}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

Slide 10: Cutout

Cutout is a simple yet effective data augmentation technique that involves randomly masking out square regions of input images. This encourages the model to focus on the entire object in the image, rather than relying on specific features.

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def cutout(image, n_holes=1, length=50):
    h, w = image.shape[:2]
    mask = np.ones((h, w), np.float32)

    for _ in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0

    masked_image = image.()
    masked_image[:,:,0] = image[:,:,0] * mask
    masked_image[:,:,1] = image[:,:,1] * mask
    masked_image[:,:,2] = image[:,:,2] * mask

    return masked_image

# Load a sample image
image = np.array(Image.open('sample_image.jpg'))

# Apply cutout with different parameters
cutout_images = [
    cutout(image.(), n_holes=1, length=50),
    cutout(image.(), n_holes=2, length=40),
    cutout(image.(), n_holes=3, length=30)
]

# Display the results
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(image)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

for i, img in enumerate(cutout_images):
    row = (i + 1) // 2
    col = (i + 1) % 2
    axes[row, col].imshow(img)
    axes[row, col].set_title(f'Cutout {i+1}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

Slide 11: CutMix

CutMix is an advanced data augmentation technique that combines aspects of both Mixup and Cutout. It involves cutting and pasting patches from one training image onto another, adjusting the labels proportionally to the area of the patch.

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def cutmix(image1, image2, alpha=1.0):
    h, w = image1.shape[:2]
    
    # Generate random bounding box
    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Create mixed image
    mixed_image = image1.()
    mixed_image[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    
    return mixed_image, lam

# Load two sample images
image1 = np.array(Image.open('sample_image1.jpg'))
image2 = np.array(Image.open('sample_image2.jpg'))

# Apply CutMix
mixed_image, lam = cutmix(image1, image2)

# Display the results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image1)
axes[0].set_title('Image 1')
axes[0].axis('off')

axes[1].imshow(image2)
axes[1].set_title('Image 2')
axes[1].axis('off')

axes[2].imshow(mixed_image)
axes[2].set_title(f'CutMix (λ = {lam:.2f})')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

Slide 12: Real-life Example: Facial Expression Recognition

Data augmentation is crucial in facial expression recognition tasks to improve model performance and generalization. Here's an example of how various augmentation techniques can be applied to a facial expression dataset.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (48, 48))
    return image

# Load a sample facial expression image
image = load_and_preprocess_image('sample_face.jpg')

# Create an ImageDataGenerator instance
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generate augmented images
augmented_images = [image]
for _ in range(5):
    augmented_images.append(datagen.random_transform(image))

# Display the results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, img in enumerate(augmented_images):
    row = i // 3
    col = i % 3
    axes[row, col].imshow(img)
    axes[row, col].set_title('Original' if i == 0 else f'Augmented {i}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

Slide 13: Real-life Example: Plant Disease Detection

Data augmentation plays a crucial role in improving plant disease detection models, especially when dealing with limited datasets. Here's an example of how various augmentation techniques can be applied to plant leaf images for disease classification.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    return image

# Load a sample plant leaf image
image = load_and_preprocess_image('sample_leaf.jpg')

# Create an ImageDataGenerator instance
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Generate augmented images
augmented_images = [image]
for _ in range(5):
    augmented_images.append(datagen.random_transform(image))

# Display the results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, img in enumerate(augmented_images):
    row = i // 3
    col = i % 3
    axes[row, col].imshow(img)
    axes[row, col].set_title('Original' if i == 0 else f'Augmented {i}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

For more in-depth information on data augmentation techniques and their applications in CNNs, consider exploring the following research papers:

1. "A survey on Image Data Augmentation for Deep Learning" by Connor Shorten and Taghi M. Khoshgoftaar (2019) ArXiv: [https://arxiv.org/abs/1912.11899](https://arxiv.org/abs/1912.11899)
2. "AutoAugment: Learning Augmentation Strategies from Data" by Ekin D. Cubuk et al. (2018) ArXiv: [https://arxiv.org/abs/1805.09501](https://arxiv.org/abs/1805.09501)
3. "RandAugment: Practical automated data augmentation with a reduced search space" by Ekin D. Cubuk et al. (2019) ArXiv: [https://arxiv.org/abs/1909.13719](https://arxiv.org/abs/1909.13719)
4. "Improved Regularization of Convolutional Neural Networks with Cutout" by Terrance DeVries and Graham W. Taylor (2017) ArXiv: [https://arxiv.org/abs/1708.04552](https://arxiv.org/abs/1708.04552)
5. "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features" by Sangdoo Yun et al. (2019) ArXiv: [https://arxiv.org/abs/1905.04899](https://arxiv.org/abs/1905.04899)

These resources provide valuable insights into the latest advancements in data augmentation techniques for CNNs and their impact on model performance.

