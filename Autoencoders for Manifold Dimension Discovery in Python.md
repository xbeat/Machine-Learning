## Autoencoders for Manifold Dimension Discovery in Python
Slide 1: Introduction to Autoencoders for Manifold Dimension Discovery

Autoencoders are neural networks designed to learn efficient data representations in an unsupervised manner. They can be used to discover the intrinsic dimensionality of data, often referred to as the manifold dimension. This process involves compressing the input data into a lower-dimensional representation and then reconstructing it, allowing us to understand the underlying structure of complex datasets.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Simple autoencoder architecture
def build_autoencoder(input_dim, encoding_dim):
    # Encoder
    inputs = tf.keras.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(inputs)
    
    # Decoder
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
    
    # Autoencoder model
    autoencoder = tf.keras.Model(inputs, decoded)
    
    return autoencoder

# Example usage
input_dim = 784  # e.g., for MNIST dataset (28x28 images)
encoding_dim = 32  # Reduced dimension

model = build_autoencoder(input_dim, encoding_dim)
model.compile(optimizer='adam', loss='binary_crossentropy')
```

Slide 2: The Concept of Manifold Learning

Manifold learning is based on the idea that high-dimensional data often lies on or near a lower-dimensional manifold. Autoencoders can help us discover this manifold by learning a compact representation of the data. This process involves finding the most important features that capture the essence of the data while discarding less relevant information.

```python
import numpy as np
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

# Generate Swiss Roll dataset
n_samples = 1000
noise = 0.1
X, _ = make_swiss_roll(n_samples, noise=noise)

# Visualize the original 3D dataset
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 2], cmap=plt.cm.viridis)
ax.set_title("Swiss Roll Dataset (3D)")
plt.show()

# The goal is to "unroll" this 3D structure into a 2D representation
```

Slide 3: Architecture of Autoencoders for Manifold Discovery

Autoencoders for manifold discovery typically consist of an encoder that compresses the input data into a lower-dimensional representation, and a decoder that attempts to reconstruct the original input from this compressed representation. The architecture is designed to capture the most important features of the data in its compressed form.

```python
import tensorflow as tf

class ManifoldAutoencoder(tf.keras.Model):
    def __init__(self, input_dim, encoding_dim):
        super(ManifoldAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(encoding_dim, activation='relu')
        ])
        
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(encoding_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='sigmoid')
        ])
        
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# Example usage
model = ManifoldAutoencoder(input_dim=784, encoding_dim=2)
model.compile(optimizer='adam', loss='mse')
```

Slide 4: Training the Autoencoder

Training an autoencoder involves minimizing the reconstruction error between the input and the output. This process forces the network to learn an efficient encoding of the data that preserves its most important features. The choice of loss function and optimization algorithm can significantly impact the quality of the learned manifold.

```python
import numpy as np
from sklearn.datasets import fetch_openml

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype('float32') / 255.0

# Split the data
X_train, X_test = X[:60000], X[60000:]

# Train the model
history = model.fit(
    X_train, X_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(X_test, X_test)
)

# Plot training history
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
```

Slide 5: Visualizing the Learned Manifold

After training, we can visualize the learned manifold by encoding the data into the lower-dimensional space and plotting it. This visualization can reveal clusters, patterns, and the overall structure of the data in a more interpretable form.

```python
# Encode the test data
encoded_imgs = model.encoder(X_test).numpy()

# Plot the 2D manifold
plt.figure(figsize=(10, 8))
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=mnist.target[60000:].astype(int), cmap='viridis')
plt.colorbar()
plt.title('MNIST digits projected onto 2D manifold')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

# This plot shows how different digits are distributed in the learned 2D space
```

Slide 6: Determining the Optimal Manifold Dimension

Choosing the right dimension for the manifold is crucial. Too low, and important information may be lost; too high, and noise might be retained. We can use techniques like the elbow method or more advanced approaches like the intrinsic dimension estimation to find the optimal encoding dimension.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def compute_reconstruction_error(model, data, encoding_dims):
    errors = []
    for dim in encoding_dims:
        model.encoding_dim = dim
        model.build((None, data.shape[1]))
        model.compile(optimizer='adam', loss='mse')
        model.fit(data, data, epochs=10, batch_size=256, verbose=0)
        reconstructed = model.predict(data)
        mse = np.mean(np.square(data - reconstructed))
        errors.append(mse)
    return errors

# Example usage
encoding_dims = [1, 2, 4, 8, 16, 32, 64]
reconstruction_errors = compute_reconstruction_error(model, X_train, encoding_dims)

plt.figure(figsize=(10, 6))
plt.plot(encoding_dims, reconstruction_errors, 'b-o')
plt.title('Reconstruction Error vs. Encoding Dimension')
plt.xlabel('Encoding Dimension')
plt.ylabel('Reconstruction Error')
plt.xscale('log2')
plt.show()

# The "elbow" in this plot suggests the optimal encoding dimension
```

Slide 7: Real-Life Example: Image Compression

Autoencoders can be used for image compression by learning a compact representation of images. This technique is particularly useful for reducing storage requirements and transmission bandwidth while preserving important visual information.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

# Load sample image
china = load_sample_image("china.jpg")
china = china / 255.0  # Normalize pixel values

# Reshape the image
h, w, c = china.shape
china_flat = china.reshape((h * w, c))

# Train autoencoder
autoencoder = build_autoencoder(input_dim=c, encoding_dim=2)
autoencoder.fit(china_flat, china_flat, epochs=50, batch_size=256, shuffle=True, verbose=0)

# Compress and reconstruct
compressed = autoencoder.encoder(china_flat).numpy()
reconstructed = autoencoder.decoder(compressed).numpy()

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(china)
ax1.set_title("Original Image")
ax2.imshow(reconstructed.reshape(h, w, c))
ax2.set_title("Reconstructed Image")
plt.show()

print(f"Compression ratio: {c / 2:.2f}")
```

Slide 8: Real-Life Example: Anomaly Detection

Autoencoders can be employed for anomaly detection by learning the normal patterns in data. When presented with anomalous data, the reconstruction error will be higher, allowing us to identify outliers or unusual events.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate normal and anomalous data
n_samples = 1000
n_outliers = 50
X, _ = make_blobs(n_samples=n_samples, n_features=2, centers=1, cluster_std=0.5, random_state=42)
X_outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))
X_combined = np.vstack([X, X_outliers])

# Train autoencoder
autoencoder = build_autoencoder(input_dim=2, encoding_dim=1)
autoencoder.fit(X, X, epochs=50, batch_size=32, shuffle=True, verbose=0)

# Compute reconstruction error
reconstructed = autoencoder.predict(X_combined)
mse = np.mean(np.square(X_combined - reconstructed), axis=1)

# Visualize results
plt.figure(figsize=(10, 8))
plt.scatter(X_combined[:, 0], X_combined[:, 1], c=mse, cmap='viridis')
plt.colorbar(label='Reconstruction Error')
plt.title('Anomaly Detection using Autoencoder')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Higher reconstruction errors (brighter colors) indicate potential anomalies
```

Slide 9: Variational Autoencoders (VAEs)

Variational Autoencoders are a probabilistic twist on traditional autoencoders. They learn a probability distribution of the latent space, allowing for generation of new samples and providing a more robust manifold representation.

```python
import tensorflow as tf

class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(784,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(latent_dim * 2)
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(784, activation='sigmoid')
        ])
        
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):
        return self.decoder(z)
    
    def call(self, inputs):
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

# Example usage
vae = VAE(latent_dim=2)
```

Slide 10: Training and Using VAEs

Training a VAE involves optimizing both the reconstruction loss and the KL divergence between the learned latent distribution and a prior (usually a standard normal distribution). This process results in a smooth, continuous latent space that can be sampled to generate new data points.

```python
@tf.function
def compute_loss(model, x):
    x_reconstructed, mean, logvar = model(x)
    reconstruction_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(x, x_reconstructed)
    )
    kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
    return reconstruction_loss + kl_loss

optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
for epoch in range(50):
    for batch in X_train:
        loss = train_step(vae, batch)
    print(f'Epoch {epoch + 1}, Loss: {loss.numpy():.4f}')

# Generate new samples
random_vector = tf.random.normal(shape=[16, latent_dim])
generated_images = vae.decode(random_vector)

# Visualize generated images
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated_images[i].numpy().reshape(28, 28), cmap='gray')
    ax.axis('off')
plt.show()
```

Slide 11: Manifold Interpolation

One powerful application of autoencoders in manifold learning is the ability to perform smooth interpolations in the latent space. This allows us to generate new, meaningful data points by moving along the learned manifold.

```python
def interpolate_points(p1, p2, n_steps=10):
    ratios = np.linspace(0, 1, num=n_steps)
    vectors = []
    for ratio in ratios:
        v = p1 * (1.0 - ratio) + p2 * ratio
        vectors.append(v)
    return np.array(vectors)

# Select two random points in the latent space
z1 = tf.random.normal((1, latent_dim))
z2 = tf.random.normal((1, latent_dim))

# Generate interpolations
interpolations = interpolate_points(z1, z2)
generated = vae.decode(interpolations)

# Visualize interpolations
fig, axes = plt.subplots(1, n_steps, figsize=(20, 2))
for i, ax in enumerate(axes):
    ax.imshow(generated[i].numpy().reshape(28, 28), cmap='gray')
    ax.axis('off')
plt.show()

# This demonstrates smooth transitions between different digit styles
```

Slide 12: Challenges and Limitations

While autoencoders are powerful tools for manifold discovery, they face several challenges. Determining the optimal latent dimension can be difficult and often requires trial and error. The learned manifold may not always correspond to human-interpretable features. Autoencoders can struggle with very high-dimensional or sparse data. The quality of the learned manifold heavily depends on the choice of architecture and hyperparameters.

```python
# Visualizing the impact of different latent dimensions
latent_dims = [1, 2, 5, 10]
fig, axes = plt.subplots(1, len(latent_dims), figsize=(20, 5))

for i, dim in enumerate(latent_dims):
    vae = VAE(latent_dim=dim)
    vae.fit(X_train, X_train, epochs=10, batch_size=128, verbose=0)
    
    # Generate samples
    samples = vae.decode(tf.random.normal((16, dim)))
    
    # Plot samples
    for j in range(16):
        ax = plt.subplot(4, 4, j+1)
        ax.imshow(samples[j].numpy().reshape(28, 28), cmap='gray')
        ax.axis('off')
    
    axes[i].set_title(f'Latent Dim: {dim}')

plt.tight_layout()
plt.show()
```

Slide 13: Advanced Techniques: Disentangled Representations

Disentangled representation learning aims to find latent representations where individual dimensions correspond to meaningful and independent factors of variation in the data. This can lead to more interpretable and controllable generative models.

```python
class BetaVAE(tf.keras.Model):
    def __init__(self, latent_dim, beta=4.0):
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(784,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(latent_dim * 2)
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(784, activation='sigmoid')
        ])
    
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def decode(self, z):
        return self.decoder(z)
    
    def call(self, inputs):
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

# Training and visualization code would follow here
```

Slide 14: Future Directions and Research

Research in autoencoder-based manifold discovery continues to evolve. Current areas of interest include:

1. Improving disentanglement in learned representations
2. Developing more robust methods for determining optimal latent dimensionality
3. Exploring the use of autoencoders in transfer learning and domain adaptation
4. Investigating the theoretical connections between autoencoders and other dimensionality reduction techniques
5. Applying autoencoders to increasingly complex and high-dimensional datasets

```python
# Pseudocode for a hypothetical future research direction
def adaptive_latent_dimension_autoencoder(data, max_dim):
    model = initialize_autoencoder(max_dim)
    for epoch in range(num_epochs):
        train_model(model, data)
        current_dim = evaluate_effective_dimension(model)
        if current_dim < max_dim:
            model = expand_latent_space(model, current_dim + 1)
    return model

# This represents a conceptual approach to dynamically adjusting 
# the latent space dimension during training
```

Slide 15: Additional Resources

For those interested in diving deeper into autoencoders and manifold learning, here are some valuable resources:

1. "Auto-Encoding Variational Bayes" by Kingma and Welling (2013) ArXiv: [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
2. "Representation Learning: A Review and New Perspectives" by Bengio et al. (2013) ArXiv: [https://arxiv.org/abs/1206.5538](https://arxiv.org/abs/1206.5538)
3. "Understanding disentangling in β-VAE" by Burgess et al. (2018) ArXiv: [https://arxiv.org/abs/1804.03599](https://arxiv.org/abs/1804.03599)
4. "Challenges in Disentangled Representation Learning" by Locatello et al. (2019) ArXiv: [https://arxiv.org/abs/1811.12359](https://arxiv.org/abs/1811.12359)

These papers provide in-depth discussions on the theory and applications of autoencoders in manifold learning and representation discovery.

