## Matrix Factorization and Tensor Methods

Slide 1: Nonnegative Matrix Factorization (NMF)

NMF is a powerful technique for decomposing a nonnegative matrix V into two nonnegative matrices W and H, such that V ≈ WH. This method is widely used in dimensionality reduction, feature extraction, and pattern recognition.

```python
from sklearn.decomposition import NMF

# Generate a random nonnegative matrix
V = np.abs(np.random.randn(10, 5))

# Initialize NMF model
model = NMF(n_components=3, init='random', random_state=0)

# Fit the model and transform V
W = model.fit_transform(V)
H = model.components_

# Reconstruct the original matrix
V_approx = np.dot(W, H)

print("Original matrix shape:", V.shape)
print("W matrix shape:", W.shape)
print("H matrix shape:", H.shape)
print("Reconstructed matrix shape:", V_approx.shape)
```

Slide 2: NMF Real-Life Example: Image Decomposition

NMF can be used to decompose images into basic components, which is useful in facial recognition and image processing.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_olivetti_faces

# Load Olivetti faces dataset
faces = fetch_olivetti_faces().data
n_samples, n_features = faces.shape

# Apply NMF
n_components = 10
model = NMF(n_components=n_components, init='random', random_state=0)
W = model.fit_transform(faces)
H = model.components_

# Plot original and reconstructed faces
fig, axes = plt.subplots(4, 5, figsize=(12, 8))
for i, ax in enumerate(axes.flatten()):
    if i < n_components:
        ax.imshow(H[i].reshape(64, 64), cmap=plt.cm.gray)
        ax.set_title(f'Component {i+1}')
    elif i < 2 * n_components:
        ax.imshow(faces[i-n_components].reshape(64, 64), cmap=plt.cm.gray)
        ax.set_title(f'Original {i-n_components+1}')
    else:
        reconstructed = np.dot(W[i-2*n_components], H)
        ax.imshow(reconstructed.reshape(64, 64), cmap=plt.cm.gray)
        ax.set_title(f'Reconstructed {i-2*n_components+1}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 3: Tensor Methods

Tensor methods extend matrix operations to higher-dimensional arrays, allowing for more complex data analysis and representation. These methods are crucial in fields such as signal processing, computer vision, and machine learning.

```python
import tensorly as tl
from tensorly.decomposition import parafac

# Create a 3D tensor
tensor = np.random.rand(4, 5, 3)

# Perform CANDECOMP/PARAFAC (CP) decomposition
rank = 2
factors = parafac(tensor, rank=rank)

# Reconstruct the tensor
reconstructed_tensor = tl.kruskal_to_tensor(factors)

print("Original tensor shape:", tensor.shape)
print("Reconstructed tensor shape:", reconstructed_tensor.shape)
print("Reconstruction error:", np.linalg.norm(tensor - reconstructed_tensor))
```

Slide 4: Tensor Methods: Tucker Decomposition

Tucker decomposition is another popular tensor factorization method, generalizing SVD to higher-order tensors.

```python
import tensorly as tl
from tensorly.decomposition import tucker

# Create a 3D tensor
tensor = np.random.rand(4, 5, 3)

# Perform Tucker decomposition
core, factors = tucker(tensor, ranks=[2, 2, 2])

# Reconstruct the tensor
reconstructed_tensor = tl.tucker_to_tensor((core, factors))

print("Original tensor shape:", tensor.shape)
print("Core tensor shape:", core.shape)
print("Factor matrices shapes:", [f.shape for f in factors])
print("Reconstruction error:", np.linalg.norm(tensor - reconstructed_tensor))
```

Slide 5: Sparse Recovery

Sparse recovery aims to reconstruct sparse signals from a small number of linear measurements. This technique is fundamental in compressed sensing and signal processing.

```python
from sklearn.linear_model import Lasso

# Generate a sparse signal
n = 1000
k = 10
x = np.zeros(n)
x[np.random.choice(n, k, replace=False)] = np.random.randn(k)

# Create measurement matrix
m = 100
A = np.random.randn(m, n)

# Generate measurements
y = np.dot(A, x)

# Solve using Lasso (L1-regularized least squares)
lasso = Lasso(alpha=0.1)
x_recovered = lasso.fit(A, y).coef_

print("Original signal sparsity:", np.sum(x != 0))
print("Recovered signal sparsity:", np.sum(x_recovered != 0))
print("Recovery error:", np.linalg.norm(x - x_recovered) / np.linalg.norm(x))
```

Slide 6: Sparse Recovery: Orthogonal Matching Pursuit

Orthogonal Matching Pursuit (OMP) is a greedy algorithm for sparse recovery, often used in compressed sensing applications.

```python
from sklearn.linear_model import OrthogonalMatchingPursuit

# Generate a sparse signal
n = 1000
k = 10
x = np.zeros(n)
x[np.random.choice(n, k, replace=False)] = np.random.randn(k)

# Create measurement matrix
m = 100
A = np.random.randn(m, n)

# Generate measurements
y = np.dot(A, x)

# Solve using Orthogonal Matching Pursuit
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k)
x_recovered = omp.fit(A, y).coef_

print("Original signal sparsity:", np.sum(x != 0))
print("Recovered signal sparsity:", np.sum(x_recovered != 0))
print("Recovery error:", np.linalg.norm(x - x_recovered) / np.linalg.norm(x))
```

Slide 7: Dictionary Learning

Dictionary learning involves finding a sparse representation of input data in terms of a learned dictionary. This technique is useful in image processing, feature extraction, and compression.

```python
from sklearn.decomposition import DictionaryLearning
import matplotlib.pyplot as plt

# Generate random patches
n_samples, n_features = 1000, 64
data = np.random.randn(n_samples, n_features)

# Learn the dictionary
n_components = 100
dl = DictionaryLearning(n_components=n_components, alpha=1, max_iter=1000)
dictionary = dl.fit(data).components_

# Plot some dictionary atoms
fig, axes = plt.subplots(10, 10, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(dictionary[i].reshape(8, 8), cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 8: Dictionary Learning: Image Denoising

Dictionary learning can be applied to image denoising by learning a dictionary from clean image patches and using it to reconstruct noisy images.

```python
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import DictionaryLearning
from skimage import data, util
from skimage.restoration import denoise_dictionary_learning
import matplotlib.pyplot as plt

# Load and add noise to the image
image = util.img_as_float(data.camera())
noisy = image + 0.1 * np.random.randn(*image.shape)

# Extract patches and learn dictionary
patch_size = (8, 8)
patches = extract_patches_2d(noisy, patch_size)
dico = DictionaryLearning(n_components=100, alpha=1, max_iter=1000)
V = dico.fit(patches.reshape(len(patches), -1)).components_

# Denoise the image
denoised = denoise_dictionary_learning(noisy, dictionary=V.reshape((100, 8, 8)),
                                       patch_size=patch_size, alpha=0.1)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(noisy, cmap='gray')
axes[1].set_title('Noisy')
axes[2].imshow(denoised, cmap='gray')
axes[2].set_title('Denoised')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 9: Gaussian Mixture Models (GMM)

Gaussian Mixture Models are probabilistic models that assume data points are generated from a mixture of a finite number of Gaussian distributions. They are widely used for clustering and density estimation.

```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
n_samples = 300
X = np.concatenate([
    np.random.normal(0, 1, (n_samples, 2)),
    np.random.normal(3, 1.5, (n_samples, 2)),
    np.random.normal(-2, 1, (n_samples, 2))
])

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(X)

# Plot results
x = np.linspace(-6, 6, 200)
y = np.linspace(-6, 6, 200)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(X.shape)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c='white', s=10, alpha=0.5)
plt.title('Gaussian Mixture Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Negative log-likelihood')
plt.show()
```

Slide 10: Gaussian Mixture Models: Speaker Identification

GMMs can be used for speaker identification by modeling the distribution of acoustic features extracted from speech signals.

```python
from sklearn.mixture import GaussianMixture
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Simulated function to extract MFCC features from audio
def extract_mfcc(audio, n_mfcc=13):
    return np.random.randn(len(audio) // 1000, n_mfcc)

# Simulated function to load audio file
def load_audio(filename):
    return np.random.randn(44100 * 5)  # 5 seconds of audio at 44.1kHz

# Load and extract features from training data
speakers = ['speaker1', 'speaker2', 'speaker3']
models = {}

for speaker in speakers:
    audio = load_audio(f'{speaker}.wav')
    mfcc = extract_mfcc(audio)
    gmm = GaussianMixture(n_components=16, covariance_type='diag')
    models[speaker] = gmm.fit(mfcc)

# Test on new audio
test_audio = load_audio('test.wav')
test_mfcc = extract_mfcc(test_audio)

# Compute log-likelihood for each speaker
scores = {speaker: model.score(test_mfcc) for speaker, model in models.items()}

# Identify the speaker
identified_speaker = max(scores, key=scores.get)

print(f"Identified speaker: {identified_speaker}")
print("Log-likelihoods:")
for speaker, score in scores.items():
    print(f"{speaker}: {score}")
```

Slide 11: Matrix Completion

Matrix completion is the task of filling in missing entries of a partially observed matrix. It has applications in recommender systems, image inpainting, and collaborative filtering.

```python
from sklearn.impute import SimpleImputer

# Create a matrix with missing values
n, m = 10, 8
true_rank = 3
U = np.random.randn(n, true_rank)
V = np.random.randn(true_rank, m)
X = np.dot(U, V)

# Randomly mask some entries
mask = np.random.rand(n, m) < 0.3
X_incomplete = np.where(mask, np.nan, X)

# Perform matrix completion using simple mean imputation
imputer = SimpleImputer(strategy='mean')
X_completed = imputer.fit_transform(X_incomplete)

# Compute error
mse = np.mean((X - X_completed)**2)
print(f"Mean Squared Error: {mse}")

# Visualize results
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(X, cmap='viridis')
ax1.set_title('Original Matrix')
ax2.imshow(X_incomplete, cmap='viridis')
ax2.set_title('Incomplete Matrix')
ax3.imshow(X_completed, cmap='viridis')
ax3.set_title('Completed Matrix')
plt.show()
```

Slide 12: Matrix Completion: Collaborative Filtering

Matrix completion is commonly used in collaborative filtering for recommender systems, such as movie rating prediction.

```python
from scipy.sparse.linalg import svds

# Create a user-item rating matrix with missing values
n_users, n_items = 100, 50
true_rank = 5
U = np.random.randn(n_users, true_rank)
V = np.random.randn(true_rank, n_items)
R = np.dot(U, V)

# Add some noise and mask random entries
R += 0.1 * np.random.randn(n_users, n_items)
mask = np.random.rand(n_users, n_items) < 0.8
R_observed = np.where(mask, R, 0)

# Perform matrix completion using SVD
U, s, Vt = svds(R_observed, k=true_rank)
S = np.diag(s)
R_completed = np.dot(np.dot(U, S), Vt)

# Compute error on observed entries
mse = np.mean((R[mask] - R_completed[mask])**2)
print(f"Mean Squared Error on observed entries: {mse}")

# Visualize results
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(R, cmap='viridis')
ax1.set_title('True Ratings')
ax2.imshow(R_observed, cmap='viridis')
ax2.set_title('Observed Ratings')
ax3.imshow(R_completed, cmap='viridis')
ax3.set_title('Predicted Ratings')
plt.show()
```

Slide 13: Additional Resources

For further exploration of the topics covered in this presentation, consider the following resources:

1. "Matrix Factorization Techniques for Recommender Systems" by Koren et al. (2009) ArXiv: [https://arxiv.org/abs/0908.5614](https://arxiv.org/abs/0908.5614)
2. "Tensor Decompositions and Applications" by Kolda and Bader (2009) ArXiv: [https://arxiv.org/abs/0904.4505](https://arxiv.org/abs/0904.4505)
3. "Compressed Sensing" by Candès and Wakin (2008) ArXiv: [https://arxiv.org/abs/0801.2986](https://arxiv.org/abs/0801.2986)
4. "Dictionary Learning Algorithms for Sparse Representation" by Tosic and Frossard (2011) ArXiv: [https://arxiv.org/abs/1009.2374](https://arxiv.org/abs/1009.2374)
5. "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition" by Rabiner (1989) Available at: [https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf)

These resources provide in-depth coverage of the topics discussed in this presentation and can serve as excellent starting points for further study and research in these areas.

Slide 14: Conclusion

This presentation has covered several important topics in machine learning and signal processing:

1. Nonnegative Matrix Factorization (NMF)
2. Tensor Methods
3. Sparse Recovery
4. Dictionary Learning
5. Gaussian Mixture Models (GMM)
6. Matrix Completion

These techniques form the foundation for many advanced applications in data analysis, pattern recognition, and signal processing. By understanding and applying these methods, researchers and practitioners can develop powerful tools for extracting meaningful information from complex datasets.

As the field of machine learning continues to evolve, these techniques are likely to play increasingly important roles in solving real-world problems across various domains, including computer vision, natural language processing, recommender systems, and many others.

We encourage you to explore these topics further using the provided resources and to experiment with implementing these algorithms in your own projects. Remember that the code examples provided in this presentation are meant to illustrate the basic concepts, and real-world applications may require more sophisticated implementations and optimizations.


