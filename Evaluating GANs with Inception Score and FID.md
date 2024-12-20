## Evaluating GANs with Inception Score and FID
Slide 1: Introduction to GAN Evaluation Metrics

The evaluation of Generative Adversarial Networks requires quantitative metrics to assess the quality and diversity of generated samples. Two fundamental metrics emerged as standard: Inception Score (IS) and Fréchet Inception Distance (FID), both leveraging the Inception-v3 network's learned representations.

```python
import torch
import torchvision.models as models
import numpy as np
from scipy import linalg

# Load pre-trained Inception v3 model
def load_inception_model():
    model = models.inception_v3(pretrained=True)
    model.eval()
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')
```

Slide 2: Inception Score Implementation

Inception Score measures both quality and diversity by computing the KL divergence between the conditional class distributions and marginal class distributions. Higher scores indicate better quality and diversity of generated samples.

```python
def calculate_inception_score(images, model, n_split=10, batch_size=32):
    preds = []
    n_batches = len(images) // batch_size
    
    with torch.no_grad():
        for i in range(n_batches):
            batch = images[i * batch_size:(i + 1) * batch_size]
            pred = model(batch)[0]
            preds.append(pred.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    scores = []
    
    for k in range(n_split):
        part = preds[k * (len(preds) // n_split): (k + 1) * (len(preds) // n_split), :]
        py = np.mean(part, axis=0)
        scores.append(np.exp(np.mean(np.sum(part * (np.log(part) - np.log(py)), axis=1))))
    
    return np.mean(scores), np.std(scores)
```

Slide 3: Fréchet Inception Distance (FID)

FID measures the Wasserstein-2 distance between two multivariate Gaussians fitted to the Inception-v3 feature representations of real and generated images. Lower FID scores indicate better quality and similarity to real data distribution.

```python
def calculate_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
```

Slide 4: Feature Extraction Pipeline

The feature extraction process involves preprocessing images and passing them through the Inception-v3 model to obtain meaningful representations for both real and generated samples.

```python
def extract_features(images, model, batch_size=32):
    features = []
    n_batches = len(images) // batch_size
    
    for i in range(n_batches):
        batch = images[i * batch_size:(i + 1) * batch_size]
        batch = preprocess_images(batch)
        
        with torch.no_grad():
            feat = model(batch)
            features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)
```

Slide 5: Image Preprocessing Implementation

Image preprocessing is crucial for GAN evaluation, ensuring consistent input format for the Inception model. This includes resizing, normalizing, and converting images to the appropriate format.

```python
def preprocess_images(images, target_size=(299, 299)):
    preprocessed = []
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    for img in images:
        preprocessed.append(transform(img))
    
    return torch.stack(preprocessed)
```

Slide 6: Kernel Inception Distance (KID)

The Kernel Inception Distance provides an unbiased alternative to FID, using a polynomial kernel to compare real and generated image features from the Inception network.

```python
def calculate_kid(real_features, fake_features, subset_size=1000):
    n_samples = min(real_features.shape[0], fake_features.shape[0], subset_size)
    
    real_subset = real_features[np.random.choice(real_features.shape[0], n_samples, replace=False)]
    fake_subset = fake_features[np.random.choice(fake_features.shape[0], n_samples, replace=False)]
    
    # Polynomial kernel with degree=3
    kernel = lambda x, y: (np.dot(x, y.T) / x.shape[1] + 1) ** 3
    
    k_xx = kernel(real_subset, real_subset)
    k_yy = kernel(fake_subset, fake_subset)
    k_xy = kernel(real_subset, fake_subset)
    
    kid = np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)
    return kid
```

Slide 7: Real-world Implementation: CIFAR-10 Evaluation

Implementing a complete evaluation pipeline for a GAN trained on CIFAR-10 dataset, demonstrating the practical application of multiple metrics for model assessment.

```python
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def evaluate_gan_cifar10(generator, batch_size=64):
    # Load CIFAR-10
    dataset = datasets.CIFAR10(root='./data', download=True, train=True)
    real_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Generate fake images
    z = torch.randn(10000, generator.z_dim).cuda()
    fake_images = generator(z).cpu()
    
    # Initialize inception model
    inception_model = load_inception_model()
    
    # Calculate metrics
    is_score = calculate_inception_score(fake_images, inception_model)
    real_features = extract_features(real_loader, inception_model)
    fake_features = extract_features(fake_images, inception_model)
    fid_score = calculate_fid(real_features, fake_features)
    kid_score = calculate_kid(real_features, fake_features)
    
    return is_score, fid_score, kid_score
```

Slide 8: Mode Collapse Detection

Mode collapse, a common GAN failure mode, can be detected through density estimation in the feature space. This implementation uses clustering analysis to identify potential mode collapse in generated samples.

```python
from sklearn.cluster import KMeans
from scipy.stats import entropy

def detect_mode_collapse(features, n_clusters=10):
    # Perform clustering on feature representations
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    
    # Calculate cluster distribution
    cluster_distribution = np.bincount(cluster_labels, minlength=n_clusters)
    cluster_distribution = cluster_distribution / len(cluster_labels)
    
    # Calculate entropy of cluster distribution
    cluster_entropy = entropy(cluster_distribution)
    
    # Calculate distance between cluster centers
    cluster_distances = pdist(kmeans.cluster_centers_)
    
    return {
        'entropy': cluster_entropy,
        'mean_cluster_distance': np.mean(cluster_distances),
        'cluster_distribution': cluster_distribution
    }
```

Slide 9: Perceptual Path Length

Perceptual Path Length measures the smoothness of the GAN's latent space by computing the average perceptual difference between adjacent points on interpolation paths.

```python
def calculate_perceptual_path_length(generator, inception_model, n_samples=10000, epsilon=1e-4):
    def slerp(a, b, t):
        omega = np.arccos(np.clip(np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b)), -1, 1))
        so = np.sin(omega)
        return np.sin((1.0-t)*omega) / so * a + np.sin(t*omega) / so * b
    
    # Generate pairs of latent vectors
    z1 = torch.randn(n_samples, generator.z_dim)
    z2 = torch.randn(n_samples, generator.z_dim)
    
    # Interpolate and generate images
    t = torch.rand(n_samples, 1) * (1 - epsilon) + epsilon
    z_interp1 = torch.tensor(slerp(z1.numpy(), z2.numpy(), t.numpy() - epsilon))
    z_interp2 = torch.tensor(slerp(z1.numpy(), z2.numpy(), t.numpy() + epsilon))
    
    # Generate and compute features
    with torch.no_grad():
        img1 = generator(z_interp1.cuda())
        img2 = generator(z_interp2.cuda())
        feat1 = inception_model(img1)
        feat2 = inception_model(img2)
    
    # Calculate distances
    distances = torch.nn.functional.pairwise_distance(feat1, feat2)
    return distances.mean().item()
```

Slide 10: Real-time GAN Monitoring System

A comprehensive monitoring system that tracks multiple metrics during GAN training, enabling early detection of training issues and performance assessment.

```python
class GANMonitor:
    def __init__(self, generator, inception_model, real_dataset, log_dir='./logs'):
        self.generator = generator
        self.inception_model = inception_model
        self.real_dataset = real_dataset
        self.metrics_history = {
            'inception_score': [],
            'fid': [],
            'kid': [],
            'ppl': [],
            'mode_collapse': []
        }
        self.writer = SummaryWriter(log_dir)
    
    def evaluate_step(self, step):
        # Generate samples
        z = torch.randn(1000, self.generator.z_dim).cuda()
        fake_images = self.generator(z)
        
        # Calculate metrics
        is_score = calculate_inception_score(fake_images, self.inception_model)
        fid = calculate_fid(self.real_features, extract_features(fake_images))
        kid = calculate_kid(self.real_features, extract_features(fake_images))
        ppl = calculate_perceptual_path_length(self.generator, self.inception_model)
        mode_stats = detect_mode_collapse(extract_features(fake_images))
        
        # Log metrics
        self.log_metrics(step, is_score, fid, kid, ppl, mode_stats)
        
    def log_metrics(self, step, is_score, fid, kid, ppl, mode_stats):
        self.writer.add_scalar('Inception_Score', is_score, step)
        self.writer.add_scalar('FID', fid, step)
        self.writer.add_scalar('KID', kid, step)
        self.writer.add_scalar('PPL', ppl, step)
        self.writer.add_scalar('Mode_Collapse/Entropy', mode_stats['entropy'], step)
```

Slide 11: Advanced Data Preprocessing Pipeline

A robust preprocessing pipeline specifically designed for GAN evaluation, incorporating advanced augmentation techniques and ensuring consistent input formatting.

```python
class GANPreprocessor:
    def __init__(self, target_size=(299, 299), normalize=True):
        self.target_size = target_size
        self.normalize = normalize
        self.transform = self._build_transform()
    
    def _build_transform(self):
        transforms_list = [
            transforms.Resize(self.target_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        
        if self.normalize:
            transforms_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        return transforms.Compose(transforms_list)
    
    def preprocess_batch(self, images):
        processed = []
        for img in images:
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            processed.append(self.transform(img))
        return torch.stack(processed)
    
    def denormalize(self, tensor):
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        return inv_normalize(tensor)
```

Slide 12: Complete Evaluation Results Visualization

Implementation of a comprehensive visualization system for GAN evaluation metrics, including distribution plots and temporal analysis.

```python
import matplotlib.pyplot as plt
import seaborn as sns

class GANEvaluationVisualizer:
    def __init__(self, metrics_history):
        self.metrics_history = metrics_history
        
    def plot_metric_evolution(self, save_path='./results'):
        plt.figure(figsize=(15, 10))
        
        # Plot FID scores
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics_history['fid'])
        plt.title('FID Score Evolution')
        plt.xlabel('Training Steps')
        plt.ylabel('FID Score')
        
        # Plot Inception scores
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics_history['inception_score'])
        plt.title('Inception Score Evolution')
        plt.xlabel('Training Steps')
        plt.ylabel('IS Score')
        
        # Plot KID scores
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics_history['kid'])
        plt.title('KID Score Evolution')
        plt.xlabel('Training Steps')
        plt.ylabel('KID Score')
        
        # Plot Mode Collapse metrics
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics_history['mode_collapse'])
        plt.title('Mode Collapse Metric')
        plt.xlabel('Training Steps')
        plt.ylabel('Entropy')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/metrics_evolution.png')
        plt.close()
```

Slide 13: Additional Resources

*   "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
    *   Search on Google Scholar: "TTUR GAN convergence"
*   "A Note on the Inception Score"
    *   [https://arxiv.org/abs/1801.01973](https://arxiv.org/abs/1801.01973)
*   "Improved Precision and Recall Metric for Assessing Generative Models"
    *   [https://arxiv.org/abs/1904.06991](https://arxiv.org/abs/1904.06991)
*   "The Unusual Effectiveness of Averaging in GAN Training"
    *   Search on Google Scholar: "GAN training averaging effectiveness"
*   "On the Properties of the FID and Other GAN Evaluation Metrics"
    *   Search: "FID GAN evaluation properties analysis"

