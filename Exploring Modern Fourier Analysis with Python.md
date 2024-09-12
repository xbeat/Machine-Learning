## Exploring Modern Fourier Analysis with Python

Slide 1: Introduction to Modern Fourier Analysis

Modern Fourier Analysis is a powerful mathematical technique used to decompose complex signals into simpler, periodic components. It has applications in signal processing, image compression, and data analysis.

```python
import numpy as np
import matplotlib.pyplot as plt

def fourier_series(t, coefficients):
    return sum(a * np.cos(2 * np.pi * n * t) + b * np.sin(2 * np.pi * n * t)
               for n, (a, b) in enumerate(coefficients))

t = np.linspace(0, 1, 1000)
coefficients = [(1, 0), (0.5, 0), (0.25, 0)]
signal = fourier_series(t, coefficients)

plt.plot(t, signal)
plt.title("Simple Fourier Series")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()
```

Slide 2: Discrete Fourier Transform (DFT)

The Discrete Fourier Transform is a fundamental tool in digital signal processing. It transforms a finite sequence of equally-spaced samples into a same-length sequence of complex numbers in the frequency domain.

```python
import numpy as np

def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, x)

# Example usage
signal = np.array([1, 2, 3, 4])
frequencies = dft(signal)
print("DFT result:", frequencies)
```

Slide 3: Fast Fourier Transform (FFT)

The Fast Fourier Transform is an efficient algorithm to compute the DFT. It reduces the complexity from O(N^2) to O(N log N), making it practical for large datasets.

```python
import numpy as np
import time

# Compare DFT and FFT performance
N = 1024
x = np.random.random(N)

start = time.time()
np.fft.fft(x)
fft_time = time.time() - start

start = time.time()
dft(x)
dft_time = time.time() - start

print(f"FFT time: {fft_time:.6f} seconds")
print(f"DFT time: {dft_time:.6f} seconds")
print(f"Speedup: {dft_time / fft_time:.2f}x")
```

Slide 4: Windowing Functions

Windowing functions are used to reduce spectral leakage when analyzing finite-length signals. They taper the signal at the beginning and end of the sampled interval.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_window(window_func, N=1000):
    window = window_func(N)
    plt.plot(window)
    plt.title(f"{window_func.__name__} Window")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.show()

plot_window(np.hanning)
plot_window(np.hamming)
plot_window(np.blackman)
```

Slide 5: Short-Time Fourier Transform (STFT)

The Short-Time Fourier Transform analyzes how frequency content changes over time by applying the Fourier transform to short segments of a signal.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Generate a chirp signal
t = np.linspace(0, 10, 1000)
w = signal.chirp(t, f0=1, f1=10, t1=10, method='linear')

# Compute and plot the STFT
f, t, Zxx = signal.stft(w, fs=100, nperseg=256)
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Magnitude')
plt.show()
```

Slide 6: Fourier Analysis in Image Processing

Fourier transforms are widely used in image processing for tasks such as filtering, compression, and feature extraction.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift

# Load an image
image = plt.imread('your_image.png')[:,:,0]  # Grayscale

# Compute 2D FFT
f = fft2(image)
fshift = fftshift(f)

# Plot original and frequency domain image
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.axis('off')
plt.subplot(122), plt.imshow(np.log(1+np.abs(fshift)), cmap='gray')
plt.title('Frequency Domain'), plt.axis('off')
plt.show()
```

Slide 7: Filtering in the Frequency Domain

Fourier analysis allows for efficient filtering by manipulating the frequency components of a signal.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

def lowpass_filter(image, cutoff):
    rows, cols = image.shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols))
    mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 1
    return mask

# Apply lowpass filter
f = fft2(image)
fshift = fftshift(f)
mask = lowpass_filter(image, 30)
fshift_filtered = fshift * mask
f_filtered = ifftshift(fshift_filtered)
img_filtered = np.real(ifft2(f_filtered))

plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.axis('off')
plt.subplot(122), plt.imshow(img_filtered, cmap='gray')
plt.title('Filtered Image'), plt.axis('off')
plt.show()
```

Slide 8: Fourier Analysis in Audio Processing

Fourier analysis is crucial in audio processing for tasks such as frequency analysis, noise reduction, and audio compression.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft

# Load a wav file
sample_rate, audio = wavfile.read('your_audio.wav')

# Compute FFT
n = len(audio)
fft_result = fft(audio)
freq = np.fft.fftfreq(n, d=1/sample_rate)

# Plot spectrum
plt.plot(freq[:n//2], np.abs(fft_result[:n//2]))
plt.title('Audio Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()
```

Slide 9: Wavelet Transform

The Wavelet Transform provides time-frequency representation with better resolution than STFT, especially for signals with rapid changes.

```python
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Generate a signal with two different frequencies
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)

# Perform continuous wavelet transform
scales = np.arange(1, 128)
coeffs, freqs = pywt.cwt(signal, scales, 'morl')

# Plot scalogram
plt.imshow(np.abs(coeffs), extent=[0, 1, 1, 128], aspect='auto',
           cmap='jet', interpolation='bilinear')
plt.title('Wavelet Transform')
plt.ylabel('Scale')
plt.xlabel('Time')
plt.colorbar(label='Magnitude')
plt.show()
```

Slide 10: Fourier Analysis in Numerical Methods

Fourier methods are used in numerical analysis for solving differential equations and implementing fast algorithms for various mathematical operations.

```python
import numpy as np
import matplotlib.pyplot as plt

def solve_heat_equation(u0, t, alpha=0.01):
    n = len(u0)
    x = np.linspace(0, 1, n)
    k = 2 * np.pi * np.fft.fftfreq(n)
    u_hat = np.fft.fft(u0)
    u_hat *= np.exp(-alpha * k**2 * t)
    return np.real(np.fft.ifft(u_hat))

# Initial condition
n = 256
u0 = np.sin(2 * np.pi * np.linspace(0, 1, n))

# Solve and plot
t_values = [0, 0.01, 0.05, 0.1]
for t in t_values:
    u = solve_heat_equation(u0, t)
    plt.plot(np.linspace(0, 1, n), u, label=f't = {t}')

plt.legend()
plt.title('Heat Equation Solution')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.show()
```

Slide 11: Fourier Analysis in Machine Learning

Fourier techniques are increasingly used in machine learning for feature extraction, data augmentation, and efficient neural network architectures.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Define a simple 1D Fourier layer
class FourierLayer(layers.Layer):
    def __init__(self, units):
        super(FourierLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True)

    def call(self, inputs):
        # Perform FFT, apply learnable weights, then IFFT
        fft = tf.signal.fft(tf.cast(inputs, tf.complex64))
        weighted = tf.matmul(fft, tf.cast(self.kernel, tf.complex64))
        return tf.math.real(tf.signal.ifft(weighted))

# Example usage in a model
model = tf.keras.Sequential([
    layers.Input(shape=(128,)),
    FourierLayer(64),
    layers.Activation('relu'),
    layers.Dense(10)
])

model.summary()
```

Slide 12: Real-life Example: MRI Image Reconstruction

Fourier analysis is crucial in medical imaging, particularly in MRI. It's used to reconstruct images from raw scanner data.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate MRI k-space data (2D Fourier transform of an image)
image = plt.imread('mri_image.png')[:,:,0]
k_space = np.fft.fftshift(np.fft.fft2(image))

# Reconstruct image from k-space
reconstructed = np.abs(np.fft.ifft2(np.fft.ifftshift(k_space)))

plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Original'), plt.axis('off')
plt.subplot(132), plt.imshow(np.log(np.abs(k_space)), cmap='gray')
plt.title('K-space'), plt.axis('off')
plt.subplot(133), plt.imshow(reconstructed, cmap='gray')
plt.title('Reconstructed'), plt.axis('off')
plt.show()
```

Slide 13: Real-life Example: Noise Reduction in Audio

Fourier analysis allows for effective noise reduction in audio signals by manipulating frequency components.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

# Load noisy audio
sample_rate, audio = wavfile.read('noisy_audio.wav')

# Design and apply lowpass filter
nyquist = 0.5 * sample_rate
cutoff = 1000  # Adjust based on noise characteristics
b, a = butter(4, cutoff / nyquist, btype='low', analog=False)
filtered_audio = filtfilt(b, a, audio)

# Plot original and filtered signals
t = np.arange(len(audio)) / sample_rate
plt.figure(figsize=(12, 6))
plt.plot(t, audio, label='Noisy')
plt.plot(t, filtered_audio, label='Filtered')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Noise Reduction using Fourier Analysis')
plt.show()

# Save filtered audio
wavfile.write('filtered_audio.wav', sample_rate, filtered_audio.astype(np.int16))
```

Slide 14: Additional Resources

For further exploration of Modern Fourier Analysis, consider these resources:

1. "A Tutorial on Fourier Analysis for the Beginners" - ArXiv:1805.03929 URL: [https://arxiv.org/abs/1805.03929](https://arxiv.org/abs/1805.03929)
2. "Fourier Analysis and Applications in Computational Mathematics" - ArXiv:2106.07971 URL: [https://arxiv.org/abs/2106.07971](https://arxiv.org/abs/2106.07971)
3. "Fourier and Wavelet Signal Processing" - Free online book by M. Vetterli, J. Kovačević, and V. K. Goyal URL: [https://www.fourierandwavelets.org/](https://www.fourierandwavelets.org/)

These resources provide in-depth explanations and advanced applications of Fourier analysis techniques.

