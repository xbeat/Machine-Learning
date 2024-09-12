## Introduction to Fourier Transforms in Python

Slide 1: 

Introduction to Fourier Transforms

The Fourier transform is a powerful mathematical tool that decomposes a signal into its constituent frequencies. It is widely used in various fields, including signal processing, image analysis, and data analysis. In Python, the Fourier transform is implemented in the NumPy and SciPy libraries.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a simple signal
t = np.linspace(0, 2*np.pi, 1000)
signal = np.sin(5*t) + np.cos(10*t)

# Plot the signal
plt.figure(figsize=(8, 4))
plt.plot(t, signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Original Signal')
plt.show()
```

Slide 2: 

Discrete Fourier Transform (DFT)

The Discrete Fourier Transform (DFT) is used to transform a finite sequence of equally-spaced samples of a function into a sequence of coefficients representing the amplitude and phase of the constituent frequencies.

```python
import numpy as np

# Define the signal
signal = [1, 2, 3, 4, 3, 2, 1, 0]

# Compute the DFT
dft = np.fft.fft(signal)

# Print the DFT coefficients
print("DFT coefficients:", dft)
```

Slide 3: 

Fast Fourier Transform (FFT)

The Fast Fourier Transform (FFT) is an efficient algorithm for computing the Discrete Fourier Transform (DFT). It significantly reduces the computational complexity of the DFT, making it practical for larger data sets.

```python
import numpy as np

# Define the signal
signal = np.random.randn(1000)

# Compute the FFT
fft_signal = np.fft.fft(signal)

# Compute the power spectrum
power_spectrum = np.abs(fft_signal)**2

# Plot the power spectrum
plt.figure(figsize=(8, 4))
plt.plot(power_spectrum)
plt.xlabel('Frequency Bin')
plt.ylabel('Power')
plt.title('Power Spectrum')
plt.show()
```

Slide 4: 

Inverse Fourier Transform

The inverse Fourier transform is used to recover the original signal from its Fourier coefficients. It is the inverse operation of the Fourier transform.

```python
import numpy as np

# Define the Fourier coefficients
coefficients = [1, 2, 3, 4, 3, 2, 1, 0]

# Compute the inverse DFT
signal = np.fft.ifft(coefficients)

# Print the reconstructed signal
print("Reconstructed signal:", signal)
```

Slide 5: 

Frequency Domain Filtering

The Fourier transform can be used to filter signals in the frequency domain. This is useful for tasks such as noise removal, signal enhancement, and feature extraction.

```python
import numpy as np

# Define the noisy signal
noisy_signal = np.sin(2*np.pi*10*t) + np.random.randn(len(t))

# Compute the FFT
fft_signal = np.fft.fft(noisy_signal)

# Apply a low-pass filter
cutoff_freq = 20
fft_filtered = fft_signal.copy()
fft_filtered[np.abs(freqs) > cutoff_freq] = 0

# Compute the inverse FFT
filtered_signal = np.fft.ifft(fft_filtered)

# Plot the original and filtered signals
plt.figure(figsize=(8, 4))
plt.plot(t, noisy_signal, label='Noisy Signal')
plt.plot(t, filtered_signal, label='Filtered Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Frequency Domain Filtering')
plt.legend()
plt.show()
```

Slide 6: 

Convolution and Fourier Transforms

The Fourier transform can be used to efficiently compute the convolution of two signals, which is a fundamental operation in signal processing.

```python
import numpy as np

# Define two signals
signal1 = np.random.randn(100)
signal2 = np.random.randn(50)

# Compute the convolution using Fourier transforms
fft_signal1 = np.fft.fft(signal1)
fft_signal2 = np.fft.fft(signal2, len(signal1))
convolved = np.fft.ifft(fft_signal1 * fft_signal2)

# Print the convolved signal
print("Convolved signal:", convolved)
```

Slide 7: 

2D Fourier Transform

The Fourier transform can be extended to two dimensions for image processing applications, such as image filtering, denoising, and feature extraction.

```python
import numpy as np
import matplotlib.pyplot as plt

# Load an image
from skimage import data
image = data.camera()

# Compute the 2D FFT
fft_image = np.fft.fft2(image)

# Shift the zero-frequency component to the center
fft_shifted = np.fft.fftshift(fft_image)

# Visualize the FFT
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(np.log(1 + np.abs(fft_shifted)), cmap='gray')
plt.title('FFT Magnitude')
plt.show()
```

Slide 8: 

Spectral Leakage and Windowing

Spectral leakage occurs when a signal is not periodic over the sampled interval, leading to spreading of the signal's energy across multiple frequency bins. Windowing functions can be applied to mitigate this effect.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a non-periodic signal
t = np.linspace(0, 2*np.pi, 1000)
signal = np.sin(10*t) + np.cos(20*t)

# Compute the FFT without windowing
fft_signal = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal), t[1] - t[0])
power_spectrum = np.abs(fft_signal)**2

# Apply a window function
window = np.hamming(len(signal))
windowed_signal = signal * window
fft_windowed = np.fft.fft(windowed_signal)
windowed_power_spectrum = np.abs(fft_windowed)**2

# Plot the power spectra
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(freqs, power_spectrum)
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('Power Spectrum without Windowing')

plt.subplot(2, 1, 2)
plt.plot(freqs, windowed_power_spectrum)
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('Power Spectrum with Hamming Window')
plt.tight_layout()
plt.show()
```

Slide 9: 

Zero-Padding and Interpolation

Zero-padding a signal before computing the Fourier transform can increase the frequency resolution, allowing for better interpolation of the frequency components.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a signal
t = np.linspace(0, 2*np.pi, 1000)
signal = np.sin(10*t) + np.cos(20*t)

# Compute the FFT without zero-padding
fft_signal = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal), t[1] - t[0])

# Zero-pad the signal
padded_signal = np.pad(signal, (0, 9000), mode='constant')
fft_padded = np.fft.fft(padded_signal)
padded_freqs = np.fft.fftfreq(len(padded_signal), t[1] - t[0])

# Plot the power spectra
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(freqs, np.abs(fft_signal)**2)
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('Power Spectrum without Zero-Padding')

plt.subplot(2, 1, 2)
plt.plot(padded_freqs, np.abs(fft_padded)**2)
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('Power Spectrum with Zero-Padding')
plt.tight_layout()
plt.show()
```

Slide 10: 

Discrete Cosine Transform (DCT)

The Discrete Cosine Transform (DCT) is a variant of the Fourier transform that operates on real-valued signals. It is widely used in image and audio compression algorithms, such as JPEG and MP3.

```python
import numpy as np
from scipy.fft import dct, idct

# Define a signal
signal = np.random.randn(100)

# Compute the DCT
dct_signal = dct(signal)

# Reconstruct the signal from the DCT coefficients
reconstructed_signal = idct(dct_signal)

# Print the reconstructed signal
print("Reconstructed signal:", reconstructed_signal)
```

Slide 11: 

Short-Time Fourier Transform (STFT)

The Short-Time Fourier Transform (STFT) is a technique for analyzing non-stationary signals by computing the Fourier transform over short, overlapping segments of the signal. This provides a time-frequency representation of the signal.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a non-stationary signal
t = np.linspace(0, 10, 1000)
signal = np.sin(2*np.pi*5*t) + np.sin(2*np.pi*10*t * (1 + 0.1*t))

# Compute the STFT
from scipy.signal import stft
f, t, Zxx = stft(signal, fs=100, nperseg=256, noverlap=128)

# Plot the STFT
plt.figure(figsize=(8, 6))
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.abs(Zxx).max(), shading='gouraud')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Short-Time Fourier Transform')
plt.colorbar()
plt.show()
```

Slide 12: 

Applications of Fourier Transforms

Fourier transforms have numerous applications in various fields, including:

* Signal processing: filtering, denoising, compression, and feature extraction
* Image processing: filtering, denoising, and feature extraction
* Audio processing: analysis, compression, and equalization
* Spectroscopy: analyzing the composition of materials
* Partial differential equations: solving certain types of PDEs
* Electrical engineering: analyzing and designing filters and control systems

```python
# This slide does not require code
```

With this series of slides, you should have a solid understanding of Fourier transforms in Python, including their implementation, properties, and applications. Feel free to modify or expand the content as needed.

