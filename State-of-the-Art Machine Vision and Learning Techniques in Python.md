## State-of-the-Art Machine Vision and Learning Techniques in Python
Slide 1: NeRF: Neural Radiance Fields

NeRF: Neural Radiance Fields

Neural Radiance Fields (NeRF) is a technique for synthesizing novel views of complex scenes by optimizing an underlying continuous scene function using a neural network. NeRF represents a scene as a 5D vector-valued function that outputs an RGB color and a volume density value at any given 3D coordinate and viewing direction. By optimizing this function to match the provided input views, NeRF can generate photorealistic renderings of the scene from arbitrary novel viewpoints.

```python
import torch
import numpy as np

# Define the NeRF model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=4, skips=[4]):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips

        # Positional encoding layers
        self.pe = PositionalEncoding(self.W)

        # Construct the main NeRF network
        self.layers = nn.ModuleList([
            nn.Linear(self.input_ch + self.W * 2 * 3, W)
        ])
        for i in range(D - 1):
            if i in skips:
                self.layers.append(nn.Linear(W + self.input_ch + self.W * 2 * 3, W))
            else:
                self.layers.append(nn.Linear(W, W))
        self.fc_final = nn.Linear(W, self.output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.W * 2 * 3], dim=-1)
        input_pts_flat = input_pts.reshape(-1, self.input_ch)
        input_views_flat = input_views.reshape(-1, self.W * 2 * 3)

        h = self.pe(input_pts_flat, input_views_flat)
        for i, l in enumerate(self.layers):
            h = self.layers[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts_flat, input_views_flat, h], -1)

        alpha = self.fc_final(h)
        return alpha
```

Slide 2: V-JEPA: Video Joint-Embedding Predictive Architecture

V-JEPA: Video Joint-Embedding Predictive Architecture

V-JEPA (Video Joint-Embedding Predictive Architecture) is a self-supervised learning approach for learning representations from unlabeled videos. It learns a joint embedding space for video frames and their corresponding future representations, enabling the model to predict future frames based on the current input. This approach leverages the natural temporal coherence of videos to learn useful representations without requiring manual annotations.

```python
import torch
import torch.nn as nn

class VJEPA(nn.Module):
    def __init__(self, encoder, decoder, future_predictor):
        super(VJEPA, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.future_predictor = future_predictor

    def forward(self, x, future_frames):
        # Encode the input frames
        encoded = self.encoder(x)

        # Predict the future representations
        future_preds = self.future_predictor(encoded)

        # Encode the future frames
        future_encoded = self.encoder(future_frames)

        # Compute the joint embedding loss
        loss = self.joint_embedding_loss(future_preds, future_encoded)

        return loss

    def joint_embedding_loss(self, future_preds, future_encoded):
        # Compute the loss between predicted and actual future representations
        # (e.g., using a contrastive loss or a regression loss)
        pass
```

Slide 3: LoRA: Low-Rank Adaptation

LoRA: Low-Rank Adaptation

Low-Rank Adaptation (LoRA) is a technique for fine-tuning large pre-trained language models in a parameter-efficient manner. Instead of updating all the parameters of the model during fine-tuning, LoRA introduces a small set of trainable parameters that modify the model's behavior for a specific task. This approach significantly reduces the memory and computational requirements for fine-tuning, making it feasible to adapt large models to various downstream tasks with limited resources.

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.rank = rank
        self.weight_A = nn.Parameter(torch.randn(rank, in_features))
        self.weight_B = nn.Parameter(torch.randn(out_features, rank))
        self.weight_C = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x):
        return x + torch.matmul(torch.matmul(self.weight_B, self.weight_A.transpose(-1, -2)), x) + self.weight_C
```

Slide 4: DoRA: Weight-Decomposed Low-Rank Adaptation

DoRA: Weight-Decomposed Low-Rank Adaptation

Weight-Decomposed Low-Rank Adaptation (DoRA) is an extension of the LoRA technique for efficient fine-tuning of large language models. DoRA decomposes the weight matrices of the pre-trained model into a low-rank component and a residual component. During fine-tuning, only the low-rank component is updated, while the residual component remains fixed. This approach further reduces the number of trainable parameters compared to LoRA, enabling even more efficient adaptation of large models.

```python
import torch
import torch.nn as nn

class DoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.rank = rank
        self.weight_A = nn.Parameter(torch.randn(rank, in_features))
        self.weight_B = nn.Parameter(torch.randn(out_features, rank))
        self.weight_C = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_D = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x):
        return x + torch.matmul(torch.matmul(self.weight_B, self.weight_A.transpose(-1, -2)), x) + self.weight_C + self.weight_D
```

Slide 5: DINO

DINO: Vision Transformer for Self-Supervised Learning

DINO (Vision Transformer for Self-Supervised Learning) is a self-supervised learning approach that leverages a Vision Transformer (ViT) architecture and a novel self-distillation technique. It trains the ViT model to match the output of a teacher model initialized with the same weights but with an additional data augmentation and momentum encoder. This approach allows the model to learn meaningful representations from unlabeled data, which can be further fine-tuned on downstream tasks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True):
        super(DINOHead, self).__init__()
        nlayers = 3
        hidden_dim = 2048

        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())

        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, out_dim))
        if norm_last_layer:
            layers.append(nn.BatchNorm1d(out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class DINOModel(nn.Module):
    def __init__(self, vit_model, projection_head):
        super(DINOModel, self).__init__()
        self.vit_model = vit_model
        self.projection_head = projection_head

    def forward(self, x):
        features = self.vit_model(x)
        projections = self.projection_head(features)
        return projections
```

Slide 6: 3D Gaussian Splatting

3D Gaussian Splatting

3D Gaussian Splatting is a technique used for rendering 3D point clouds by splatting Gaussian kernels at each point's location. This approach creates smooth surfaces from the discrete point cloud data, enabling realistic rendering of 3D scenes. It involves projecting the Gaussian kernels onto the image plane and accumulating their contributions, effectively reconstructing a continuous surface representation from the sparse point cloud.

```python
import torch
import torch.nn.functional as F

def gaussian_splatting(points, weights, grid_size, sigma):
    # Convert points to grid coordinates
    grid_coords = (points * grid_size).long()

    # Compute Gaussian kernel values
    x_coords = torch.arange(grid_size, device=points.device)
    y_coords = torch.arange(grid_size, device=points.device)
    z_coords = torch.arange(grid_size, device=points.device)
    xx, yy, zz = torch.meshgrid(x_coords, y_coords, z_coords)
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))

    # Splat Gaussian kernels onto the grid
    grid = torch.zeros(grid_size, grid_size, grid_size, device=points.device)
    for i in range(points.shape[0]):
        grid[grid_coords[i, 0], grid_coords[i, 1], grid_coords[i, 2]] += weights[i] * kernel

    return grid
```

Slide 7: CLIP

CLIP: Contrastive Language-Image Pre-training

CLIP (Contrastive Language-Image Pre-training) is a neural network architecture that learns visual representations from natural language supervision. It is trained on a large dataset of image-text pairs, learning to associate images with their corresponding text descriptions. CLIP can be used for various vision tasks, such as image classification, retrieval, and captioning, by leveraging the learned cross-modal representations.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder):
        super(CLIP, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, texts):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)

        logits = (image_features @ text_features.t()) * self.logit_scale

        return logits

# Load pre-trained CLIP models
image_encoder = models.clip.create_vision_encoder()
text_encoder = models.clip.create_text_encoder()

# Create the CLIP model
clip_model = CLIP(image_encoder, text_encoder)
```

Slide 8: BYOL: Bootstrap Your Own Latent

BYOL: Bootstrap Your Own Latent

Bootstrap Your Own Latent (BYOL) is a self-supervised learning approach for training neural networks without requiring labeled data. It works by training an online network to predict the representations of a target network, which is a slowly-moving average of the online network. This bootstrapping process allows the model to learn meaningful representations from unlabeled data, which can be used for various downstream tasks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BYOL(nn.Module):
    def __init__(self, online_network, target_network, projector, predictor):
        super(BYOL, self).__init__()
        self.online_network = online_network
        self.target_network = target_network
        self.projector = projector
        self.predictor = predictor

    def forward(self, x1, x2):
        # Compute online and target representations
        online_rep = self.online_network(x1)
        target_rep = self.target_network(x2)

        # Project representations
        online_proj = self.projector(online_rep)
        target_proj = self.projector(target_rep)

        # Predict target representation
        online_pred = self.predictor(online_proj)

        # Compute loss
        loss = F.mse_loss(online_pred, target_proj.detach())

        return loss
```

Slide 9: Masked Autoencoders

Masked Autoencoders

Masked Autoencoders are a self-supervised learning approach that trains neural networks to reconstruct corrupted or masked versions of input data. The network learns to predict the missing or corrupted parts of the input, effectively capturing the underlying structure and patterns in the data. This technique has been successfully applied to various domains, including computer vision and natural language processing.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(MaskedAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, mask):
        # Mask the input
        masked_x = x * mask

        # Encode the masked input
        encoded = self.encoder(masked_x)

        # Reconstruct the original input
        reconstructed = self.decoder(encoded)

        # Compute the reconstruction loss
        loss = F.mse_loss(reconstructed * mask, x * mask)

        return loss
```

Slide 10: The Change You Want to See

The Change You Want to See

"The Change You Want to See" is a method for generating counterfactual explanations for machine learning models. It aims to understand how a model's prediction would change if specific input features were altered. This technique involves iteratively updating the input features while minimizing the change required to achieve a desired output, providing insights into the model's decision-making process and feature importance.

```python
import torch
import torch.nn as nn
import torch.optim as optim

def generate_counterfactual(model, input_data, target_output, lambda_reg=0.1):
    # Define the counterfactual optimization objective
    def counterfactual_loss(input_data_cf):
        output = model(input_data_cf)
        reconstruction_loss = torch.mean((input_data_cf - input_data) ** 2)
        target_loss = nn.functional.cross_entropy(output, target_output)
        loss = lambda_reg * reconstruction_loss + target_loss
        return loss

    # Initialize the counterfactual input data
    input_data_cf = input_data.clone().detach().requires_grad_(True)

    # Optimize the counterfactual input data
    optimizer = optim.Adam([input_data_cf], lr=0.01)
    for i in range(100):
        optimizer.zero_grad()
        loss = counterfactual_loss(input_data_cf)
        loss.backward()
        optimizer.step()

    # Return the counterfactual input data
    return input_data_cf.detach()
```

Slide 11: The Forward-Forward Algorithm

The Forward-Forward Algorithm

The Forward-Forward Algorithm is a method for interpreting the behavior of deep neural networks by identifying the input features that are most important for a particular prediction. It works by iteratively removing or masking input features and observing the changes in the model's output. The algorithm ranks the input features based on their impact on the output, providing insights into the model's decision-making process and feature importance.

```python
import torch
import torch.nn as nn

def forward_forward_algorithm(model, input_data, target_output):
    # Get the initial model output
    initial_output = model(input_data)

    # Initialize a list to store feature importances
    feature_importances = []

    # Iterate over input features
    for i in range(input_data.shape[1]):
        # Create a masked input by zeroing out the current feature
        masked_input = input_data.clone()
        masked_input[:, i] = 0

        # Get the output with the masked input
        masked_output = model(masked_input)

        # Compute the difference between the initial and masked outputs
        output_diff = torch.sum(torch.abs(initial_output - masked_output))

        # Append the output difference as the feature importance
        feature_importances.append(output_diff.item())

    # Normalize the feature importances
    feature_importances = torch.tensor(feature_importances) / torch.sum(torch.tensor(feature_importances))

    return feature_importances
```

Slide 12: Transformers For Image Recognition At Scale

Transformers For Image Recognition At Scale

Transformers have shown remarkable performance in various natural language processing tasks, and their application has been extended to computer vision tasks, including image recognition. Transformers for Image Recognition at Scale is an approach that leverages the self-attention mechanism of transformers to capture long-range dependencies in images, enabling more effective representation learning. This approach has demonstrated state-of-the-art performance on various image recognition benchmarks.

```python
import torch
import torch.nn as nn
from einops import rearrange

class ImageTransformer(nn.Module):
    def __init__(self, patch_size, num_patches, dim, num_heads, num_layers):
        super(ImageTransformer, self).__init__()
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, num_heads, dim * 4, dropout=0.1),
            num_layers,
            nn.LayerNorm(dim)
        )
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embedding(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        # Add positional embedding
        x += self.positional_embedding

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Classification
        x = x[:, 0]  # Use the [CLS] token representation
        x = self.classifier(x)

        return x
```

Slide 13: SimCLR

SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

SimCLR (A Simple Framework for Contrastive Learning of Visual Representations) is a self-supervised learning approach for training visual representations by maximizing the agreement between differently augmented views of the same image while contrasting against other images in the mini-batch. This contrastive learning approach allows the model to learn meaningful representations from unlabeled data, which can be used for various downstream tasks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    def __init__(self, encoder, projection_head):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x1, x2):
        # Encode the augmented views
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        # Apply the projection head
        p1 = self.projection_head(z1)
        p2 = self.projection_head(z2)

        # Compute the contrastive loss
        loss = contrastive_loss(p1, p2)

        return loss

def contrastive_loss(p1, p2):
    # Compute the similarity matrix
    sim_matrix = torch.matmul(p1, p2.T)

    # Compute the negative logits
    neg_logits = torch.cat((sim_matrix.diag()[:, None], sim_matrix.diag()[None, :]), dim=1)
    neg_logits = torch.logsumexp(neg_logits, dim=1, keepdim=True)

    # Compute the contrastive loss
    loss = -torch.mean(sim_matrix.diag() - neg_logits)

    return loss
```

Slide 14: Backpropagation and the brain

Backpropagation and the brain

Backpropagation is a fundamental algorithm in machine learning for training neural networks by computing the gradients of the loss function with respect to the network's weights. While backpropagation has been incredibly successful in training artificial neural networks, there has been an ongoing debate about its biological plausibility and whether it accurately models the learning process in the human brain. Recent research has explored alternative algorithms and mechanisms that could explain how the brain learns and adapts, potentially shedding light on the relationship between backpropagation and biological neural networks.

```python
# Pseudocode for backpropagation in a simple feedforward neural network

# Forward pass
input = X
for layer in network:
    weights = layer.weights
    biases = layer.biases
    output = activation(dot(input, weights) + biases)
    input = output

loss = compute_loss(output, target)

# Backward pass
grad_output = derivative_loss(output, target)
for layer in reversed(network):
    weights = layer.weights
    grad_input = grad_output * derivative_activation(layer.output)
    grad_weights = dot(layer.input.T, grad_input)
    grad_biases = sum(grad_input, axis=0)
    layer.weights -= learning_rate * grad_weights
    layer.biases -= learning_rate * grad_biases
    grad_output = dot(grad_input, weights.T)
```

Slide 15: Additional Resources

Additional Resources

For those interested in exploring these topics further, here are some additional resources from reputable sources like arXiv.org:

1. NeRF: Neural Radiance Fields
   * [https://arxiv.org/abs/2003.08934](https://arxiv.org/abs/2003.08934)
2. Video Joint-Embedding Predictive Architecture (V-JEPA)
   * [https://arxiv.org/abs/2104.13292](https://arxiv.org/abs/2104.13292)
3. Low-Rank Adaptation of Large Language Models (LoRA)
   * [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
4. Weight-Decomposed Low-Rank Adaptation (DoRA)
   * [https://arxiv.org/abs/2211.13379](https://arxiv.org/abs/2211.13379)
5. DINO: Vision Transformer for Self-Supervised Learning
   * [https://arxiv.org/abs/2104.14294](https://arxiv.org/abs/2104.14294)
6. 3D Gaussian Splatting
   * [https://arxiv.org/abs/2109.12898](https://arxiv.org/abs/2109.12898)
7. CLIP: Contrastive Language-Image Pre-training
   * [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
8. Bootstrap Your Own Latent (BYOL)
   * [https://arxiv.org/abs/2006.07733](https://arxiv.org/abs/2006.07733)
9. Masked Autoencoders
   * [https://arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377)
10. "The Change You Want to See" (Counterfactual Explanations)
  * [https://arxiv.org/abs/2109.01473](https://arxiv.org/abs/2109.01473)
11. The Forward-Forward Algorithm
  * [https://arxiv.org/abs/2108.10829](https://arxiv.org/abs/2108.10829)
12. Transformers For Image Recognition At Scale
  * [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
13. SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
  * [https://arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709)
14. Backpropagation and the brain
  * [https://arxiv.org/abs/2105.03956](https://arxiv.org/abs/2105.03956)

