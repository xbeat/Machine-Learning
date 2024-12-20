## Enhancing Low-Res Images with Python
Slide 1: Introduction to Image Super-Resolution

Image super-resolution is the process of enhancing the resolution and quality of low-resolution images. This technique has numerous applications in various fields, including medical imaging, satellite imagery, and digital photography. In this presentation, we'll explore how to use Python to implement image super-resolution techniques.

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load a low-resolution image
low_res_img = Image.open('low_res_image.jpg')

# Display the low-resolution image
plt.imshow(low_res_img)
plt.title('Low-Resolution Image')
plt.show()
```

Slide 2: Understanding Image Resolution

Image resolution refers to the level of detail in an image. It's typically measured in pixels per inch (PPI) or dots per inch (DPI). Higher resolution images contain more pixels and thus more detail. When we upscale a low-resolution image, we aim to intelligently fill in the missing information.

```python
# Get the dimensions of the low-resolution image
width, height = low_res_img.size
print(f"Low-res image dimensions: {width}x{height}")

# Create a higher resolution version (2x upscaling)
high_res_img = low_res_img.resize((width*2, height*2), Image.BICUBIC)
print(f"High-res image dimensions: {high_res_img.size[0]}x{high_res_img.size[1]}")

# Display both images side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(low_res_img)
ax1.set_title('Low-Resolution')
ax2.imshow(high_res_img)
ax2.set_title('High-Resolution (2x upscale)')
plt.show()
```

Slide 3: Bicubic Interpolation

Bicubic interpolation is a common method for image upscaling. It considers the 16 nearest pixels (a 4x4 grid) to estimate the value of a new pixel. This method produces smoother images compared to simpler techniques like nearest-neighbor or bilinear interpolation.

```python
from scipy.ndimage import zoom

# Convert image to numpy array
low_res_array = np.array(low_res_img)

# Apply bicubic interpolation
high_res_array = zoom(low_res_array, (2, 2, 1), order=3)

# Convert back to image
high_res_img = Image.fromarray(high_res_array.astype('uint8'))

# Display the result
plt.imshow(high_res_img)
plt.title('Bicubic Interpolation (2x upscale)')
plt.show()
```

Slide 4: Super-Resolution Convolutional Neural Network (SRCNN)

SRCNN is a deep learning approach to super-resolution. It uses convolutional neural networks to learn the mapping between low and high-resolution images. The network consists of three layers: patch extraction, non-linear mapping, and reconstruction.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

# Define the SRCNN model
model = Sequential([
    Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(None, None, 1)),
    Conv2D(32, (1, 1), activation='relu', padding='same'),
    Conv2D(1, (5, 5), padding='same')
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Display model summary
model.summary()
```

Slide 5: Preparing Data for SRCNN

To train an SRCNN model, we need pairs of low and high-resolution images. We'll create a dataset by downscaling high-resolution images and using them as input, with the original images as targets.

```python
import os
from sklearn.model_selection import train_test_split

def create_dataset(hr_folder, scale_factor=2):
    X, y = [], []
    for img_name in os.listdir(hr_folder):
        img = Image.open(os.path.join(hr_folder, img_name)).convert('L')
        hr_img = np.array(img)
        lr_img = np.array(img.resize((img.width // scale_factor, img.height // scale_factor), Image.BICUBIC))
        lr_img = np.array(Image.fromarray(lr_img).resize((img.width, img.height), Image.BICUBIC))
        X.append(lr_img)
        y.append(hr_img)
    return np.array(X), np.array(y)

X, y = create_dataset('high_res_images')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
```

Slide 6: Training the SRCNN Model

Now that we have prepared our dataset, we can train the SRCNN model. We'll use the Adam optimizer and mean squared error as our loss function. The model will learn to map low-resolution inputs to high-resolution outputs.

```python
# Normalize the data
X_train, y_train = X_train.astype('float32') / 255.0, y_train.astype('float32') / 255.0
X_test, y_test = X_test.astype('float32') / 255.0, y_test.astype('float32') / 255.0

# Reshape for single channel (grayscale) images
X_train = X_train.reshape(X_train.shape + (1,))
y_train = y_train.reshape(y_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))
y_test = y_test.reshape(y_test.shape + (1,))

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.1)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
```

Slide 7: Evaluating the SRCNN Model

After training, we need to evaluate our model's performance. We'll use the test set to generate super-resolution images and compare them with the original high-resolution images.

```python
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = np.mean((y_test - predictions) ** 2)
    psnr_value = psnr(y_test, predictions, data_range=1)
    return mse, psnr_value

mse, psnr_value = evaluate_model(model, X_test, y_test)
print(f"Mean Squared Error: {mse}")
print(f"Peak Signal-to-Noise Ratio: {psnr_value} dB")

# Visualize a sample result
sample_idx = np.random.randint(0, len(X_test))
input_img = X_test[sample_idx].squeeze()
true_img = y_test[sample_idx].squeeze()
pred_img = model.predict(X_test[sample_idx:sample_idx+1]).squeeze()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(input_img, cmap='gray')
ax1.set_title('Low Resolution')
ax2.imshow(pred_img, cmap='gray')
ax2.set_title('SRCNN Output')
ax3.imshow(true_img, cmap='gray')
ax3.set_title('Original High Resolution')
plt.show()
```

Slide 8: Real-Life Example: Enhancing Satellite Imagery

Satellite imagery often suffers from low resolution due to the limitations of space-based sensors. Super-resolution techniques can be applied to enhance these images, providing more detailed views of Earth's surface for applications in urban planning, agriculture, and environmental monitoring.

```python
import rasterio
from rasterio.enums import Resampling

# Load a low-resolution satellite image
with rasterio.open('low_res_satellite.tif') as src:
    low_res_data = src.read(1)  # Read the first band
    profile = src.profile

# Upscale the image using our trained SRCNN model
upscaled_data = model.predict(low_res_data.reshape(1, *low_res_data.shape, 1)).squeeze()

# Update the profile for the new resolution
profile.update(width=upscaled_data.shape[1], height=upscaled_data.shape[0])

# Save the upscaled image
with rasterio.open('high_res_satellite.tif', 'w', **profile) as dst:
    dst.write(upscaled_data, 1)

# Visualize the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(low_res_data, cmap='gray')
ax1.set_title('Original Low-Resolution Satellite Image')
ax2.imshow(upscaled_data, cmap='gray')
ax2.set_title('Super-Resolution Satellite Image')
plt.show()
```

Slide 9: Real-Life Example: Enhancing Medical Imaging

In medical imaging, high-resolution images are crucial for accurate diagnosis. Super-resolution techniques can be applied to enhance low-resolution medical images, potentially improving diagnostic accuracy without the need for new, expensive imaging equipment.

```python
import SimpleITK as sitk

# Load a low-resolution medical image (e.g., MRI or CT scan)
low_res_image = sitk.ReadImage('low_res_mri.nii')
low_res_array = sitk.GetArrayFromImage(low_res_image)

# Apply super-resolution to each slice
high_res_slices = []
for slice in low_res_array:
    high_res_slice = model.predict(slice.reshape(1, *slice.shape, 1)).squeeze()
    high_res_slices.append(high_res_slice)

high_res_array = np.stack(high_res_slices)

# Create a new SimpleITK image with the enhanced resolution
high_res_image = sitk.GetImageFromArray(high_res_array)
high_res_image.SetSpacing([s/2 for s in low_res_image.GetSpacing()])  # Adjust spacing for 2x upscaling

# Save the high-resolution image
sitk.WriteImage(high_res_image, 'high_res_mri.nii')

# Visualize a sample slice
sample_slice = np.random.randint(0, high_res_array.shape[0])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(low_res_array[sample_slice], cmap='gray')
ax1.set_title('Original Low-Resolution MRI Slice')
ax2.imshow(high_res_array[sample_slice], cmap='gray')
ax2.set_title('Super-Resolution MRI Slice')
plt.show()
```

Slide 10: Generative Adversarial Networks (GANs) for Super-Resolution

GANs have shown remarkable results in image super-resolution tasks. They consist of two neural networks: a generator that creates high-resolution images, and a discriminator that tries to distinguish between real and generated high-resolution images.

```python
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_generator():
    input_shape = (None, None, 1)
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=9, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(32, kernel_size=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Conv2D(1, kernel_size=5, padding='same', activation='tanh')(x)
    return Model(inputs, outputs)

def build_discriminator():
    input_shape = (None, None, 1)
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

# Build and compile the GAN model
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
discriminator.trainable = False
gan_input = Input(shape=(None, None, 1))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

print(generator.summary())
print(discriminator.summary())
```

Slide 11: Training the GAN for Super-Resolution

Training a GAN involves alternating between training the discriminator and the generator. The generator aims to produce realistic high-resolution images, while the discriminator tries to distinguish between real and generated images.

```python
def train_gan(epochs, batch_size=32):
    for epoch in range(epochs):
        # Select a random batch of low-resolution images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        low_res_imgs = X_train[idx]
        
        # Generate high-resolution images
        generated_imgs = generator.predict(low_res_imgs)
        
        # Get a random batch of real high-resolution images
        real_imgs = y_train[idx]
        
        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_imgs, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the generator
        g_loss = gan.train_on_batch(low_res_imgs, np.ones((batch_size, 1)))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Train the GAN
train_gan(epochs=1000)

# Generate and display a super-resolution image
sample_idx = np.random.randint(0, len(X_test))
low_res_sample = X_test[sample_idx:sample_idx+1]
generated_img = generator.predict(low_res_sample).squeeze()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(low_res_sample.squeeze(), cmap='gray')
ax1.set_title('Low Resolution')
ax2.imshow(generated_img, cmap='gray')
ax2.set_title('GAN Generated')
ax3.imshow(y_test[sample_idx].squeeze(), cmap='gray')
ax3.set_title('Original High Resolution')
plt.show()
```

Slide 12: Perceptual Loss for Enhanced Super-Resolution

Perceptual loss uses features extracted from pre-trained deep learning models to compare the generated and target images. This approach often results in visually more pleasing super-resolution results compared to pixel-wise loss functions.

```python
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

def build_vgg():
    vgg = VGG19(weights='imagenet', include_top=False, input_shape=(None, None, 3))
    return Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output)

vgg = build_vgg()
vgg.trainable = False

def perceptual_loss(y_true, y_pred):
    return K.mean(K.square(vgg(y_true) - vgg(y_pred)))

# Modify the generator to output 3-channel images
def build_generator_rgb():
    input_shape = (None, None, 3)
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=9, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(32, kernel_size=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)
    return Model(inputs, outputs)

generator_rgb = build_generator_rgb()
gan_rgb = Model(gan_input, discriminator(generator_rgb(gan_input)))
gan_rgb.compile(loss=[perceptual_loss, 'binary_crossentropy'], 
                loss_weights=[1, 1e-3], 
                optimizer=Adam(0.0002, 0.5))

# Training loop would be similar to the previous GAN training,
# but using the perceptual loss for the generator
```

Slide 13: Evaluation Metrics for Super-Resolution

To assess the quality of super-resolution results, we use various metrics such as Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and perceptual metrics like LPIPS (Learned Perceptual Image Patch Similarity).

```python
from skimage.metrics import structural_similarity as ssim
import lpips

def evaluate_sr(y_true, y_pred):
    psnr = tf.image.psnr(y_true, y_pred, max_val=1.0)
    ssim_value = ssim(y_true.squeeze(), y_pred.squeeze(), data_range=1, multichannel=True)
    
    # LPIPS (you need to install the lpips package)
    loss_fn = lpips.LPIPS(net='alex')
    lpips_value = loss_fn(y_true, y_pred)
    
    return psnr.numpy(), ssim_value, lpips_value.item()

# Evaluate on a test sample
sample_idx = np.random.randint(0, len(X_test))
low_res = X_test[sample_idx:sample_idx+1]
high_res_true = y_test[sample_idx:sample_idx+1]
high_res_pred = generator.predict(low_res)

psnr, ssim_value, lpips_value = evaluate_sr(high_res_true, high_res_pred)
print(f"PSNR: {psnr:.2f}")
print(f"SSIM: {ssim_value:.4f}")
print(f"LPIPS: {lpips_value:.4f}")
```

Slide 14: Challenges and Future Directions

While significant progress has been made in image super-resolution, challenges remain. These include handling diverse image types, real-time processing for video, and achieving consistent quality across different upscaling factors. Future research directions include:

1. Incorporating attention mechanisms for better feature extraction.
2. Exploring unsupervised and self-supervised learning approaches.
3. Developing adaptive models that can handle multiple upscaling factors.
4. Improving efficiency for deployment on edge devices.

```python
# Pseudocode for an attention-based super-resolution model
class AttentionSR(nn.Module):
    def __init__(self):
        super(AttentionSR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.attention = SelfAttentionBlock(64)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        return self.conv2(x)

class SelfAttentionBlock(nn.Module):
    # Implementation of self-attention mechanism
    ...

# Future work: Implement and train this attention-based model
```

Slide 15: Additional Resources

For those interested in diving deeper into image super-resolution techniques, here are some valuable resources:

1. "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" by Ledig et al. (2017) ArXiv: [https://arxiv.org/abs/1609.04802](https://arxiv.org/abs/1609.04802)
2. "Enhanced Deep Residual Networks for Single Image Super-Resolution" by Lim et al. (2017) ArXiv: [https://arxiv.org/abs/1707.02921](https://arxiv.org/abs/1707.02921)
3. "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data" by Wang et al. (2021) ArXiv: [https://arxiv.org/abs/2107.10833](https://arxiv.org/abs/2107.10833)

These papers provide in-depth explanations of advanced super-resolution techniques and serve as excellent starting points for further exploration in this field.

