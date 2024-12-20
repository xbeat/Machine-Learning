## Advanced Generative Adversarial Networks with Python
Slide 1: Introduction to Advanced Generative Adversarial Networks (GANs)

Generative Adversarial Networks (GANs) are a class of deep learning models consisting of two neural networks: a generator and a discriminator. These networks are trained simultaneously in a competitive process, where the generator creates synthetic data and the discriminator attempts to distinguish between real and generated data. This slide introduces the concept of advanced GANs and their applications in various fields.

```python
import tensorflow as tf

# Basic GAN structure
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # Define generator layers

    def call(self, inputs):
        # Generate synthetic data
        pass

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Define discriminator layers

    def call(self, inputs):
        # Classify real/fake data
        pass

# Create instances
generator = Generator()
discriminator = Discriminator()
```

Slide 2: Deep Convolutional GAN (DCGAN)

DCGANs incorporate convolutional layers in both the generator and discriminator, making them particularly effective for image generation tasks. This architecture leverages the power of convolutional neural networks to capture spatial hierarchies in data, resulting in higher quality generated images.

```python
import tensorflow as tf

def make_generator_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        
        tf.keras.layers.Reshape((7, 7, 256)),
        tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        
        tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        
        tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

generator = make_generator_model()
```

Slide 3: Conditional GAN (cGAN)

Conditional GANs extend the basic GAN framework by incorporating additional information to guide the generation process. This allows for more controlled and targeted output. The generator and discriminator receive both a noise vector and a condition vector as input, enabling the model to generate samples conditioned on specific attributes.

```python
import tensorflow as tf

class ConditionalGAN(tf.keras.Model):
    def __init__(self):
        super(ConditionalGAN, self).__init__()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        noise_input = tf.keras.Input(shape=(100,))
        condition_input = tf.keras.Input(shape=(10,))
        x = tf.keras.layers.Concatenate()([noise_input, condition_input])
        # Add generator layers
        return tf.keras.Model([noise_input, condition_input], x)

    def build_discriminator(self):
        img_input = tf.keras.Input(shape=(28, 28, 1))
        condition_input = tf.keras.Input(shape=(10,))
        x = tf.keras.layers.Concatenate(axis=1)([img_input, condition_input])
        # Add discriminator layers
        return tf.keras.Model([img_input, condition_input], x)

cgan = ConditionalGAN()
```

Slide 4: Wasserstein GAN (WGAN)

Wasserstein GANs address some of the training instability issues present in traditional GANs by using the Wasserstein distance as a loss function. This approach provides a more stable gradient for training, leading to improved convergence and reduced mode collapse. WGANs replace the discriminator with a critic that estimates the Wasserstein distance between real and generated distributions.

```python
import tensorflow as tf

class WGAN(tf.keras.Model):
    def __init__(self):
        super(WGAN, self).__init__()
        self.generator = self.build_generator()
        self.critic = self.build_critic()

    def build_generator(self):
        # Similar to traditional GAN generator
        pass

    def build_critic(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)  # No activation for Wasserstein distance
        ])
        return model

    def critic_loss(self, real_output, fake_output):
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

wgan = WGAN()
```

Slide 5: Progressive Growing of GANs (PGGAN)

Progressive Growing of GANs is a technique that gradually increases the resolution of generated images during training. This approach starts with low-resolution images and progressively adds layers to both the generator and discriminator, allowing the model to learn coarse-to-fine details. PGGANs have shown remarkable success in generating high-resolution, photorealistic images.

```python
import tensorflow as tf

class PGGAN(tf.keras.Model):
    def __init__(self):
        super(PGGAN, self).__init__()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        model = tf.keras.Sequential()
        # Initial block for 4x4 resolution
        model.add(tf.keras.layers.Dense(4 * 4 * 512, input_dim=512))
        model.add(tf.keras.layers.Reshape((4, 4, 512)))
        
        # Progressive blocks
        resolutions = [8, 16, 32, 64, 128, 256, 512]
        for res in resolutions:
            model.add(self.upsample_block(res))
        
        return model

    def upsample_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(filters, 3, padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2)
        ])

    def build_discriminator(self):
        # Similar structure to generator, but in reverse order
        pass

pggan = PGGAN()
```

Slide 6: CycleGAN for Image-to-Image Translation

CycleGAN is an advanced GAN architecture designed for unpaired image-to-image translation. It learns to translate images from one domain to another without paired training data. CycleGAN uses two generator-discriminator pairs and introduces a cycle consistency loss to ensure the translated images can be mapped back to their original domain.

```python
import tensorflow as tf

class CycleGAN(tf.keras.Model):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()

    def build_generator(self):
        # Implement generator architecture
        pass

    def build_discriminator(self):
        # Implement discriminator architecture
        pass

    def cycle_loss(self, real_image, cycled_image):
        return tf.reduce_mean(tf.abs(real_image - cycled_image))

    def generator_loss(self, generated_output):
        return tf.reduce_mean(tf.square(generated_output - 1))

    def discriminator_loss(self, real_output, generated_output):
        real_loss = tf.reduce_mean(tf.square(real_output - 1))
        generated_loss = tf.reduce_mean(tf.square(generated_output))
        return (real_loss + generated_loss) * 0.5

cycle_gan = CycleGAN()
```

Slide 7: StyleGAN for High-Quality Image Synthesis

StyleGAN is a state-of-the-art GAN architecture that introduces a style-based generator. It separates high-level attributes from stochastic variation, allowing for fine-grained control over the generated images. StyleGAN uses adaptive instance normalization (AdaIN) to inject the latent code at each convolution layer, enabling multi-scale control over the image generation process.

```python
import tensorflow as tf

class StyleGAN(tf.keras.Model):
    def __init__(self):
        super(StyleGAN, self).__init__()
        self.mapping_network = self.build_mapping_network()
        self.synthesis_network = self.build_synthesis_network()

    def build_mapping_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512)
        ])
        return model

    def build_synthesis_network(self):
        # Implement synthesis network with AdaIN layers
        pass

    def generate_image(self, latent_vector):
        w = self.mapping_network(latent_vector)
        return self.synthesis_network(w)

style_gan = StyleGAN()
```

Slide 8: BigGAN for Large-Scale Image Generation

BigGAN is designed for generating high-fidelity, diverse images at scale. It incorporates several techniques to improve training stability and output quality, including self-attention mechanisms, spectral normalization, and a truncation trick for sampling. BigGAN has demonstrated impressive results in generating photorealistic images across a wide range of categories.

```python
import tensorflow as tf

class BigGAN(tf.keras.Model):
    def __init__(self, num_classes):
        super(BigGAN, self).__init__()
        self.generator = self.build_generator(num_classes)
        self.discriminator = self.build_discriminator(num_classes)

    def build_generator(self, num_classes):
        latent_dim = 128
        noise_input = tf.keras.Input(shape=(latent_dim,))
        class_input = tf.keras.Input(shape=(num_classes,))
        
        x = tf.keras.layers.Concatenate()([noise_input, class_input])
        x = tf.keras.layers.Dense(4 * 4 * 1024, use_bias=False)(x)
        x = tf.keras.layers.Reshape((4, 4, 1024))(x)
        
        # Add upsampling blocks with self-attention
        x = self.upsample_block(x, 512)
        x = self.self_attention_block(x)
        x = self.upsample_block(x, 256)
        x = self.upsample_block(x, 128)
        x = self.upsample_block(x, 64)
        
        x = tf.keras.layers.Conv2D(3, 3, padding='same', activation='tanh')(x)
        
        return tf.keras.Model(inputs=[noise_input, class_input], outputs=x)

    def upsample_block(self, x, filters):
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    def self_attention_block(self, x):
        # Implement self-attention mechanism
        pass

    def build_discriminator(self, num_classes):
        # Implement discriminator with spectral normalization
        pass

big_gan = BigGAN(num_classes=1000)
```

Slide 9: Pix2Pix for Paired Image-to-Image Translation

Pix2Pix is a conditional GAN designed for paired image-to-image translation tasks. It learns a mapping from input images to output images, given a dataset of paired examples. Pix2Pix uses a U-Net architecture for the generator and a PatchGAN discriminator, which classifies patches of the image as real or fake, promoting local consistency in the generated images.

```python
import tensorflow as tf

class Pix2Pix(tf.keras.Model):
    def __init__(self):
        super(Pix2Pix, self).__init__()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])
        
        # Encoder (downsampling)
        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),
            self.downsample(128, 4),
            self.downsample(256, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
        ]
        
        # Decoder (upsampling)
        up_stack = [
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4),
            self.upsample(256, 4),
            self.upsample(128, 4),
            self.upsample(64, 4),
        ]
        
        # Apply downsampling
        x = inputs
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        
        # Apply upsampling and concatenate with skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])
        
        last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')
        x = last(x)
        
        return tf.keras.Model(inputs=inputs, outputs=x)

    def downsample(self, filters, size, apply_batchnorm=True):
        # Implement downsampling block
        pass

    def upsample(self, filters, size, apply_dropout=False):
        # Implement upsampling block
        pass

    def build_discriminator(self):
        # Implement PatchGAN discriminator
        pass

pix2pix = Pix2Pix()
```

Slide 10: SRGAN for Super-Resolution

SRGAN (Super-Resolution GAN) is designed to upscale low-resolution images to high-resolution versions. It uses a deep residual network as the generator and introduces perceptual loss functions to achieve photo-realistic super-resolution. SRGAN can produce sharper and more detailed high-resolution images compared to traditional methods.

```python
import tensorflow as tf

class SRGAN(tf.keras.Model):
    def __init__(self):
        super(SRGAN, self).__init__()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        inputs = tf.keras.layers.Input(shape=[64, 64, 3])
        
        # Initial convolution block
        x = tf.keras.layers.Conv2D(64, 9, padding='same')(inputs)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        
        # Residual blocks
        for _ in range(16):
            x = self.residual_block(x)
        
        # Upsampling blocks
        x = self.upsample_block(x)
        x = self.upsample_block(x)
        
        # Final convolution
        x = tf.keras.layers.Conv2D(3, 9, padding='same', activation='tanh')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=x)

    def residual_block(self, x):
        # Implement residual block
        pass

    def upsample_block(self, x):
        # Implement upsampling block
        pass

    def build_discriminator(self):
        # Implement discriminator
        pass

srgan = SRGAN()
```

Slide 11: Real-life Example: Image Colorization with GANs

GANs have found practical applications in image colorization, transforming grayscale images into vibrant, colored versions. This process involves training a GAN where the generator learns to add realistic colors to grayscale input images, while the discriminator attempts to distinguish between real colored images and those produced by the generator.

```python
import tensorflow as tf

class ColorizeGAN(tf.keras.Model):
    def __init__(self):
        super(ColorizeGAN, self).__init__()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        inputs = tf.keras.layers.Input(shape=[256, 256, 1])
        
        # Encoder
        x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same')(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)
        
        # Residual blocks
        for _ in range(6):
            x = self.residual_block(x)
        
        # Decoder
        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)
        
        outputs = tf.keras.layers.Conv2D(2, 3, padding='same', activation='tanh')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def residual_block(self, x):
        # Implement residual block
        pass

    def build_discriminator(self):
        # Implement discriminator
        pass

colorize_gan = ColorizeGAN()
```

Slide 12: Real-life Example: Text-to-Image Synthesis

Text-to-image synthesis is an exciting application of GANs where the model generates images based on textual descriptions. This involves training a GAN with a text encoder that processes the input description and conditions the generator to produce corresponding images. The discriminator evaluates both the image quality and its relevance to the given text.

```python
import tensorflow as tf

class TextToImageGAN(tf.keras.Model):
    def __init__(self, vocab_size, max_text_length):
        super(TextToImageGAN, self).__init__()
        self.text_encoder = self.build_text_encoder(vocab_size, max_text_length)
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_text_encoder(self, vocab_size, max_text_length):
        inputs = tf.keras.layers.Input(shape=(max_text_length,))
        x = tf.keras.layers.Embedding(vocab_size, 256)(inputs)
        x = tf.keras.layers.LSTM(256)(x)
        x = tf.keras.layers.Dense(1024)(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    def build_generator(self):
        text_features = tf.keras.layers.Input(shape=(1024,))
        noise = tf.keras.layers.Input(shape=(100,))
        
        x = tf.keras.layers.Concatenate()([text_features, noise])
        x = tf.keras.layers.Dense(4 * 4 * 512, use_bias=False)(x)
        x = tf.keras.layers.Reshape((4, 4, 512))(x)
        
        # Implement upsampling blocks
        # ...
        
        outputs = tf.keras.layers.Conv2D(3, 3, padding='same', activation='tanh')(x)
        
        return tf.keras.Model(inputs=[text_features, noise], outputs=outputs)

    def build_discriminator(self):
        # Implement discriminator
        pass

text_to_image_gan = TextToImageGAN(vocab_size=10000, max_text_length=100)
```

Slide 13: Challenges and Future Directions in GAN Research

While GANs have shown remarkable progress, several challenges remain. These include mode collapse, training instability, and difficulty in generating coherent large-scale images. Ongoing research focuses on improving training dynamics, developing more robust evaluation metrics, and exploring novel architectures. Future directions include combining GANs with other AI techniques, exploring unsupervised and semi-supervised learning approaches, and extending GANs to new domains such as video synthesis and 3D object generation.

```python
import tensorflow as tf
import matplotlib.pyplot as plt

def visualize_gan_progress(gan_model, epochs):
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    noise = tf.random.normal([10, 100])
    
    for epoch in range(epochs):
        generated_images = gan_model.generator(noise, training=False)
        
        for i in range(10):
            ax = axs[i // 5, i % 5]
            ax.imshow(generated_images[i, :, :, 0], cmap='gray')
            ax.axis('off')
        
        plt.suptitle(f'Epoch {epoch + 1}')
        plt.pause(0.1)
    
    plt.show()

# Usage example:
# visualize_gan_progress(my_gan_model, epochs=10)
```

Slide 14: Additional Resources

For those interested in diving deeper into advanced GAN techniques, the following resources provide comprehensive information and latest research:

1. "Generative Adversarial Networks: An Overview" by Antonia Creswell et al. (2018) ArXiv: [https://arxiv.org/abs/1710.07035](https://arxiv.org/abs/1710.07035)
2. "A Survey on Generative Adversarial Networks: Variants, Applications, and Training" by Zhengwei Wang et al. (2021) ArXiv: [https://arxiv.org/abs/2006.05132](https://arxiv.org/abs/2006.05132)
3. "Improved Techniques for Training GANs" by Tim Salimans et al. (2016) ArXiv: [https://arxiv.org/abs/1606.03498](https://arxiv.org/abs/1606.03498)
4. "Progressive Growing of GANs for Improved Quality, Stability, and Variation" by Tero Karras et al. (2017) ArXiv: [https://arxiv.org/abs/1710.10196](https://arxiv.org/abs/1710.10196)

These papers provide in-depth explanations of various GAN architectures, training techniques, and applications, serving as excellent starting points for further exploration in the field of generative adversarial networks.

