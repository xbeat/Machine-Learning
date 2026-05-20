## Building a Generative Neural Network from Scratch in Python
Slide 1: Introduction to Generative Neural Networks

Generative Neural Networks (GNNs) are a class of artificial intelligence models capable of creating new data that resembles the training data. They learn the underlying patterns and distributions of the input data, allowing them to generate novel, realistic samples. In this slideshow, we'll explore how to build a simple GNN from scratch using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize the concept of data generation
np.random.seed(42)
real_data = np.random.normal(0, 1, 1000)
generated_data = np.random.normal(0.5, 1.2, 1000)

plt.hist(real_data, alpha=0.5, label='Real Data')
plt.hist(generated_data, alpha=0.5, label='Generated Data')
plt.legend()
plt.title('Real vs Generated Data Distribution')
plt.show()
```

Slide 2: Basic Architecture of a GNN

A simple GNN consists of a generator and a discriminator. The generator creates fake data, while the discriminator tries to distinguish between real and fake data. These two components are trained adversarially, improving each other's performance over time.

```python
import tensorflow as tf

def build_generator(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_dim=input_dim, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='tanh')
    ])
    return model

def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, input_dim=input_dim, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Example usage
generator = build_generator(100, 784)  # For generating 28x28 images
discriminator = build_discriminator(784)
```

Slide 3: Generating Random Noise

The generator takes random noise as input and transforms it into fake data. We'll create a function to generate this noise, which will serve as the starting point for our generated samples.

```python
def generate_noise(batch_size, noise_dim):
    return np.random.normal(0, 1, size=(batch_size, noise_dim))

# Example usage
batch_size = 64
noise_dim = 100
noise = generate_noise(batch_size, noise_dim)

print("Shape of noise:", noise.shape)
print("Sample noise vector:", noise[0][:10])  # First 10 elements of the first noise vector
```

Slide 4: Loss Functions for GNN

The generator and discriminator have different loss functions. The generator aims to minimize the difference between real and generated data, while the discriminator tries to maximize its ability to distinguish between them.

```python
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Example usage
fake_output = tf.random.uniform((64, 1))
gen_loss = generator_loss(fake_output)
print("Generator loss:", gen_loss.numpy())

real_output = tf.random.uniform((64, 1))
disc_loss = discriminator_loss(real_output, fake_output)
print("Discriminator loss:", disc_loss.numpy())
```

Slide 5: Training Loop

The training process involves alternating between training the discriminator and the generator. We'll create a basic training loop that demonstrates this process.

```python
@tf.function
def train_step(real_images):
    noise = generate_noise(batch_size, noise_dim)
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# Example usage (assuming we have a dataset)
for epoch in range(num_epochs):
    for batch in dataset:
        gen_loss, disc_loss = train_step(batch)
    print(f"Epoch {epoch + 1}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")
```

Slide 6: Generating Samples

After training, we can use the generator to create new samples. Let's create a function to generate and visualize these samples.

```python
def generate_and_plot_images(generator, noise_dim, num_examples=16):
    noise = generate_noise(num_examples, noise_dim)
    generated_images = generator(noise, training=False)
    
    fig = plt.figure(figsize=(4, 4))
    for i in range(num_examples):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_images[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
generate_and_plot_images(generator, noise_dim)
```

Slide 7: Real-life Example: Generating Handwritten Digits

Let's apply our GNN to generate handwritten digits similar to those in the MNIST dataset. We'll use a simple version of our GNN to create new digit images.

```python
from tensorflow.keras.datasets import mnist

# Load and preprocess MNIST data
(train_images, _), (_, _) = mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]

# Build and train the GNN (assuming previous code is available)
generator = build_generator(100, 784)
discriminator = build_discriminator(784)

# Training loop (simplified for brevity)
for epoch in range(50):
    for batch in train_images:
        noise = generate_noise(batch_size, 100)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)
            real_output = discriminator(batch, training=True)
            fake_output = discriminator(generated_images, training=True)
            
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)
        
        # Apply gradients (omitted for brevity)

# Generate and display results
generate_and_plot_images(generator, 100)
```

Slide 8: Conditional Generative Neural Networks

Conditional GNNs allow us to generate samples with specific attributes. We'll modify our previous GNN to generate digits of a particular class.

```python
def build_conditional_generator(noise_dim, num_classes):
    noise_input = tf.keras.Input(shape=(noise_dim,))
    label_input = tf.keras.Input(shape=(1,))
    
    label_embedding = tf.keras.layers.Embedding(num_classes, 50)(label_input)
    label_embedding = tf.keras.layers.Flatten()(label_embedding)
    
    combined_input = tf.keras.layers.Concatenate()([noise_input, label_embedding])
    
    x = tf.keras.layers.Dense(256, activation='relu')(combined_input)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    output = tf.keras.layers.Dense(784, activation='tanh')(x)
    
    return tf.keras.Model([noise_input, label_input], output)

# Example usage
conditional_generator = build_conditional_generator(100, 10)  # 10 classes for digits 0-9

# Generate images of a specific digit
noise = generate_noise(16, 100)
labels = np.full((16, 1), 5)  # Generate 16 images of digit '5'
generated_images = conditional_generator.predict([noise, labels])

# Visualize the generated images
plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
```

Slide 9: Evaluating GNN Performance

Evaluating the quality of generated samples is crucial. We'll implement a simple evaluation metric called Frechet Inception Distance (FID) to measure the similarity between real and generated images.

```python
from scipy.linalg import sqrtm

def calculate_fid(real_images, generated_images):
    # Assume we have a pre-trained model for feature extraction
    feature_extractor = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    
    def preprocess_images(images):
        images = tf.image.resize(images, (299, 299))
        images = tf.keras.applications.inception_v3.preprocess_input(images)
        return images
    
    real_features = feature_extractor.predict(preprocess_images(real_images))
    generated_features = feature_extractor.predict(preprocess_images(generated_images))
    
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Example usage
real_samples = # ... load real samples
generated_samples = generator.predict(generate_noise(1000, noise_dim))
fid_score = calculate_fid(real_samples, generated_samples)
print(f"FID Score: {fid_score}")
```

Slide 10: Real-life Example: Generating Synthetic Images of Clothing

Let's apply our GNN to generate synthetic images of clothing items, similar to those in the Fashion MNIST dataset. This example demonstrates how GNNs can be used in the fashion industry for design inspiration or augmenting product catalogs.

```python
from tensorflow.keras.datasets import fashion_mnist

# Load and preprocess Fashion MNIST data
(train_images, _), (_, _) = fashion_mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]

# Build and train the GNN (assuming previous code is available)
generator = build_generator(100, 784)
discriminator = build_discriminator(784)

# Training loop (simplified for brevity)
for epoch in range(50):
    for batch in train_images:
        noise = generate_noise(batch_size, 100)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)
            real_output = discriminator(batch, training=True)
            fake_output = discriminator(generated_images, training=True)
            
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)
        
        # Apply gradients (omitted for brevity)

# Generate and display results
generate_and_plot_images(generator, 100)
```

Slide 11: Advanced Techniques: Wasserstein GAN

Wasserstein GAN (WGAN) improves upon traditional GANs by using the Wasserstein distance as its loss function. This approach often leads to more stable training and better quality results. Here's a simplified implementation of WGAN:

```python
import tensorflow as tf

class WGAN(tf.keras.Model):
    def __init__(self, generator, critic):
        super(WGAN, self).__init__()
        self.generator = generator
        self.critic = critic

    def compile(self, g_optimizer, c_optimizer, g_loss_fn, c_loss_fn):
        super(WGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer
        self.g_loss_fn = g_loss_fn
        self.c_loss_fn = c_loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.generator.input_shape[1]])

        # Train the critic
        for _ in range(5):
            with tf.GradientTape() as tape:
                fake_images = self.generator(noise, training=True)
                fake_logits = self.critic(fake_images, training=True)
                real_logits = self.critic(real_images, training=True)
                c_loss = self.c_loss_fn(real_logits, fake_logits)

            c_gradients = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_gradients, self.critic.trainable_variables))

        # Train the generator
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            fake_logits = self.critic(fake_images, training=True)
            g_loss = self.g_loss_fn(fake_logits)

        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return {"c_loss": c_loss, "g_loss": g_loss}

# Usage example
generator = build_generator(100, 784)
critic = build_critic(784)
wgan = WGAN(generator, critic)
wgan.compile(
    g_optimizer=tf.keras.optimizers.RMSprop(0.00005),
    c_optimizer=tf.keras.optimizers.RMSprop(0.00005),
    g_loss_fn=lambda fake_logits: -tf.reduce_mean(fake_logits),
    c_loss_fn=lambda real_logits, fake_logits: tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
)
```

Slide 12: Handling Mode Collapse

Mode collapse is a common issue in GAN training where the generator produces limited varieties of samples. Let's implement a simple technique to detect and mitigate mode collapse:

```python
def detect_mode_collapse(generated_samples, n_bins=10):
    # Assuming generated_samples are images flattened to 1D
    hist, _ = np.histogram(generated_samples.flatten(), bins=n_bins)
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log(hist + 1e-8))
    max_entropy = np.log(n_bins)
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

def generate_diverse_samples(generator, noise_dim, batch_size, diversity_factor=0.1):
    base_noise = tf.random.normal([batch_size, noise_dim])
    diverse_noise = base_noise + tf.random.normal([batch_size, noise_dim]) * diversity_factor
    return generator(diverse_noise, training=False)

# Usage in training loop
for epoch in range(num_epochs):
    for batch in dataset:
        # ... (normal training step)
        
        # Check for mode collapse
        generated_samples = generator(generate_noise(1000, noise_dim))
        entropy = detect_mode_collapse(generated_samples.numpy())
        
        if entropy < 0.6:  # Arbitrary threshold
            print(f"Potential mode collapse detected. Entropy: {entropy:.2f}")
            diverse_samples = generate_diverse_samples(generator, noise_dim, 1000)
            # Use diverse_samples for additional training or adjustment
```

Slide 13: Progressive Growing of GANs

Progressive growing is a technique to improve the quality and stability of GAN training, especially for high-resolution images. Here's a simplified implementation:

```python
def build_progressive_generator(final_dim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4*4*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.Reshape((4, 4, 256)))
    
    resolutions = [8, 16, 32, 64, final_dim]
    for res in resolutions:
        model.add(tf.keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model

def train_progressive_gan(generator, discriminator, dataset, num_epochs, steps_per_epoch):
    current_resolution = 4
    for epoch in range(num_epochs):
        if epoch % (num_epochs // 5) == 0 and current_resolution < 128:
            current_resolution *= 2
            print(f"Increasing resolution to {current_resolution}x{current_resolution}")
        
        for step in range(steps_per_epoch):
            real_images = next(iter(dataset))
            real_images = tf.image.resize(real_images, (current_resolution, current_resolution))
            
            # ... (perform training step with resized images)

# Usage
generator = build_progressive_generator(128)
discriminator = build_progressive_discriminator(128)
train_progressive_gan(generator, discriminator, dataset, num_epochs=100, steps_per_epoch=1000)
```

Slide 14: Additional Resources

For those interested in diving deeper into Generative Neural Networks, here are some valuable resources:

1. "Generative Deep Learning" by David Foster - A comprehensive book on various GAN architectures and applications.
2. "Improved Training of Wasserstein GANs" by Gulrajani et al. (2017) - ArXiv:1704.00028 [https://arxiv.org/abs/1704.00028](https://arxiv.org/abs/1704.00028)
3. "Progressive Growing of GANs for Improved Quality, Stability, and Variation" by Karras et al. (2017) - ArXiv:1710.10196 [https://arxiv.org/abs/1710.10196](https://arxiv.org/abs/1710.10196)
4. TensorFlow GAN (TFGAN) - An open-source library for working with GANs in TensorFlow.
5. "A Survey on GANs for Anomaly Detection" by Di Mattia et al. (2019) - ArXiv:1906.11632 [https://arxiv.org/abs/1906.11632](https://arxiv.org/abs/1906.11632)

These resources provide in-depth explanations, mathematical foundations, and advanced techniques for working with Generative Neural Networks.

