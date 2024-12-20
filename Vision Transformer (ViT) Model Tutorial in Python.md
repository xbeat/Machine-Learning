## Vision Transformer (ViT) Model Tutorial in Python
Slide 1: Introduction to Vision Transformer (ViT)

The Vision Transformer (ViT) is a groundbreaking model that applies the Transformer architecture, originally designed for natural language processing, to computer vision tasks. It treats images as sequences of patches, enabling the model to capture global dependencies and achieve state-of-the-art performance on various image classification benchmarks.

```python
import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positions = nn.Parameter(torch.randn((image_size // patch_size) ** 2 + 1, embed_dim))
        
    def forward(self, x):
        x = self.projection(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positions
        return x
```

Slide 2: ViT Architecture Overview

The ViT architecture consists of three main components: Patch Embedding, Transformer Encoder, and Classification Head. The Patch Embedding layer divides the input image into fixed-size patches and linearly projects them. The Transformer Encoder processes these embedded patches using self-attention mechanisms. Finally, the Classification Head uses the output of the Transformer Encoder to make predictions.

```python
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim),
            num_layers=depth
        )
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.patch_embed(img)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.to_latent(x)
        return self.mlp_head(x)
```

Slide 3: Patch Embedding

The Patch Embedding layer is responsible for converting the input image into a sequence of embedded patches. It uses a convolutional layer to project each patch into a lower-dimensional embedding space. Additionally, it adds a learnable classification token and positional embeddings to provide spatial information to the model.

```python
def visualize_patch_embedding(image, patch_size):
    import matplotlib.pyplot as plt
    from torchvision.transforms import functional as F

    # Convert image to tensor and add batch dimension
    img_tensor = F.to_tensor(image).unsqueeze(0)

    # Create patch embedding layer
    patch_embed = PatchEmbedding(image.size[0], patch_size, 3, 64)

    # Apply patch embedding
    embedded_patches = patch_embed(img_tensor)

    # Visualize original image and embedded patches
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(embedded_patches[0, 1:].reshape(-1, patch_size, patch_size, 3).sum(axis=-1))
    ax2.set_title(f"Embedded Patches (sum of channels)")
    ax2.axis('off')

    plt.show()

# Example usage:
# from PIL import Image
# image = Image.open("example_image.jpg")
# visualize_patch_embedding(image, patch_size=16)
```

Slide 4: Self-Attention Mechanism

The core of the Transformer architecture is the self-attention mechanism. It allows the model to weigh the importance of different parts of the input sequence when processing each element. In ViT, this enables the model to capture relationships between different image patches, regardless of their spatial distance.

```python
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Visualize attention weights
def visualize_attention(model, image):
    import matplotlib.pyplot as plt

    # Process image through the model
    with torch.no_grad():
        output = model(image.unsqueeze(0))

    # Extract attention weights from the last layer
    attn_weights = model.transformer.layers[-1].self_attn.attn_output_weights[0]

    # Visualize attention weights
    plt.figure(figsize=(10, 10))
    plt.imshow(attn_weights.mean(0).cpu(), cmap='viridis')
    plt.title("Attention Weights")
    plt.colorbar()
    plt.show()

# Example usage:
# model = VisionTransformer(...)
# image = torch.randn(3, 224, 224)
# visualize_attention(model, image)
```

Slide 5: Transformer Encoder

The Transformer Encoder consists of multiple layers of self-attention and feed-forward neural networks. Each layer applies self-attention to its input, followed by layer normalization and a feed-forward network. This structure allows the model to progressively refine its representations of the image patches.

```python
class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                SelfAttention(dim, heads=heads),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Linear(mlp_dim, dim)
                )
            ]))

    def forward(self, x):
        for norm1, attn, norm2, ffn in self.layers:
            x = x + attn(norm1(x))
            x = x + ffn(norm2(x))
        return x
```

Slide 6: Classification Head

The Classification Head is the final component of the ViT model. It takes the output of the Transformer Encoder, typically focusing on the representation of the classification token, and applies a multi-layer perceptron to produce class probabilities.

```python
class ClassificationHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Use only the classification token
        x = x[:, 0]
        x = self.norm(x)
        return self.fc(x)

# Visualize class predictions
def visualize_predictions(model, image, class_names):
    import matplotlib.pyplot as plt

    # Process image through the model
    with torch.no_grad():
        output = model(image.unsqueeze(0))

    # Get top 5 predictions
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Visualize predictions
    plt.figure(figsize=(10, 5))
    plt.bar(range(5), top5_prob.cpu())
    plt.xticks(range(5), [class_names[idx] for idx in top5_catid])
    plt.title("Top 5 Predictions")
    plt.show()

# Example usage:
# model = VisionTransformer(...)
# image = torch.randn(3, 224, 224)
# class_names = ["cat", "dog", "bird", ...]
# visualize_predictions(model, image, class_names)
```

Slide 7: Training the ViT Model

Training a ViT model involves defining the loss function, optimizer, and training loop. We use cross-entropy loss for classification tasks and typically employ the AdamW optimizer with weight decay for regularization.

```python
def train_vit(model, train_loader, val_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Val Accuracy: {100*correct/total:.2f}%")

# Example usage:
# model = VisionTransformer(...)
# train_loader = ...
# val_loader = ...
# train_vit(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4)
```

Slide 8: Data Augmentation for ViT

Data augmentation is crucial for improving the generalization of ViT models. Common techniques include random cropping, horizontal flipping, color jittering, and random erasing. These augmentations help the model learn invariances to various transformations.

```python
from torchvision import transforms

def get_train_transforms(image_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomErasing(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(image_size):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Visualize augmentations
def visualize_augmentations(image, transform):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(10):
        aug_image = transform(image)
        axes[i].imshow(aug_image.permute(1, 2, 0))
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage:
# from PIL import Image
# image = Image.open("example_image.jpg")
# train_transform = get_train_transforms(224)
# visualize_augmentations(image, train_transform)
```

Slide 9: Fine-tuning ViT on Custom Datasets

Fine-tuning a pre-trained ViT model on a custom dataset allows leveraging transfer learning. We typically replace the classification head with a new one suitable for the target task and fine-tune the entire model or just the later layers.

```python
def fine_tune_vit(pretrained_model, num_classes, train_loader, val_loader, num_epochs, learning_rate):
    # Replace classification head
    pretrained_model.mlp_head = nn.Sequential(
        nn.LayerNorm(pretrained_model.mlp_head[0].normalized_shape[0]),
        nn.Linear(pretrained_model.mlp_head[0].normalized_shape[0], num_classes)
    )

    # Freeze early layers
    for param in pretrained_model.patch_embed.parameters():
        param.requires_grad = False
    for layer in pretrained_model.transformer.layers[:8]:
        for param in layer.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, pretrained_model.parameters()), 
                                  lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # Training and validation code (similar to train_vit function)
        pass

# Example usage:
# pretrained_model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
# num_classes = 10  # Number of classes in your custom dataset
# fine_tune_vit(pretrained_model, num_classes, train_loader, val_loader, num_epochs=5, learning_rate=1e-5)
```

Slide 10: Interpreting ViT Predictions

Interpreting the predictions of ViT models is crucial for understanding their behavior and building trust. Techniques like attention visualization and Grad-CAM can provide insights into which parts of the image the model focuses on when making predictions.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_attention(model, image, head_fusion="mean", discard_ratio=0.9):
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Get the attention weights
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        attentions = model.get_last_selfattention(image.unsqueeze(0))
    
    # Reshape and process attention weights
    nh = attentions.shape[1]  # number of heads
    attentions = attentions[:, :, 0, 1:].reshape(nh, -1)
    
    if head_fusion == "mean":
        attentions = attentions.mean(0)
    elif head_fusion == "max":
        attentions = attentions.max(0)[0]
    elif head_fusion == "min":
        attentions = attentions.min(0)[0]
    else:
        raise ValueError(f"Incorrect head_fusion: {head_fusion}")
    
    # Reshape attention weights to match image dimensions
    w_featmap = image.shape[-2] // model.patch_embed.patch_size[0]
    h_featmap = image.shape[-1] // model.patch_embed.patch_size[1]
    attentions = attentions.reshape(w_featmap, h_featmap)
    
    # Apply discard ratio
    if discard_ratio > 0:
        top_k = int(attentions.numel() * (1 - discard_ratio))
        attentions = torch.topk(attentions.flatten(), top_k, sorted=False)[0]
        attentions = attentions.reshape(w_featmap, h_featmap)
    
    # Upsample to match original image size
    attentions = torch.nn.functional.interpolate(
        attentions.unsqueeze(0).unsqueeze(0),
        size=image.shape[-2:],
        mode='bicubic',
        align_corners=False
    ).squeeze().numpy()
    
    # Visualize the attention map
    plt.imshow(attentions)
    plt.title("Attention Map")
    plt.colorbar()
    plt.show()

# Example usage:
# model = VisionTransformer(...)
# image = torch.randn(3, 224, 224)
# visualize_attention(model, image)
```

Slide 11: ViT for Object Detection

While originally designed for image classification, ViT can be adapted for object detection tasks. One approach is to combine ViT with a detection head, such as DETR (DEtection TRansformer), to perform end-to-end object detection.

```python
class ViTDETR(nn.Module):
    def __init__(self, vit_model, num_classes, num_queries):
        super().__init__()
        self.vit = vit_model
        self.query_embed = nn.Embedding(num_queries, vit_model.dim)
        self.class_embed = nn.Linear(vit_model.dim, num_classes + 1)  # +1 for background
        self.bbox_embed = MLP(vit_model.dim, vit_model.dim, 4, 3)

    def forward(self, x):
        features = self.vit(x)
        queries = self.query_embed.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
        hs = self.transformer_decoder(queries, features)
        
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        return outputs_class, outputs_coord

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# Example usage:
# vit_model = VisionTransformer(...)
# detr_model = ViTDETR(vit_model, num_classes=80, num_queries=100)
# image = torch.randn(1, 3, 224, 224)
# class_pred, bbox_pred = detr_model(image)
```

Slide 12: ViT for Semantic Segmentation

ViT can be adapted for semantic segmentation tasks by modifying the architecture to produce pixel-wise predictions. One approach is to use a ViT as the encoder and combine it with a decoder network to generate high-resolution segmentation maps.

```python
class ViTSegmentation(nn.Module):
    def __init__(self, vit_model, num_classes):
        super().__init__()
        self.vit = vit_model
        self.decoder = SegmentationDecoder(vit_model.dim, num_classes)

    def forward(self, x):
        features = self.vit(x)
        return self.decoder(features)

class SegmentationDecoder(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 256, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2).view(x.size(0), -1, 14, 14)  # Assuming 14x14 feature map
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.conv4(x)

# Example usage:
# vit_model = VisionTransformer(...)
# seg_model = ViTSegmentation(vit_model, num_classes=21)
# image = torch.randn(1, 3, 224, 224)
# segmentation_map = seg_model(image)
```

Slide 13: Real-life Example: Image Classification

Let's consider a real-life example of using ViT for classifying images of different types of vehicles. This could be useful for traffic monitoring systems or autonomous driving applications.

```python
import torch
from torchvision import transforms
from PIL import Image

# Define the classes
vehicle_classes = ['bicycle', 'bus', 'car', 'motorcycle', 'truck']

# Load a pre-trained ViT model
model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
model.head = torch.nn.Linear(model.head.in_features, len(vehicle_classes))

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_vehicle(image_path):
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)

    return vehicle_classes[top_catid[0]], top_prob[0].item()

# Example usage:
# image_path = 'path/to/vehicle/image.jpg'
# predicted_class, confidence = classify_vehicle(image_path)
# print(f'Predicted class: {predicted_class}, Confidence: {confidence:.2f}')
```

Slide 14: Real-life Example: Medical Image Segmentation

In this example, we'll use a ViT-based model for medical image segmentation, specifically for segmenting brain tumors in MRI scans. This application can assist radiologists in identifying and measuring tumor regions.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class ViTUNet(nn.Module):
    def __init__(self, vit_model, num_classes):
        super().__init__()
        self.vit = vit_model
        self.decoder = UNetDecoder(vit_model.dim, num_classes)

    def forward(self, x):
        features = self.vit(x)
        return self.decoder(features)

class UNetDecoder(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, kernel_size=2, stride=2)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2).view(x.size(0), -1, 14, 14)  # Assuming 14x14 feature map
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return self.conv5(x)

def segment_brain_tumor(model, image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    # Apply softmax and get the tumor segmentation mask
    segmentation_mask = F.softmax(output, dim=1)[0, 1].cpu().numpy()

    return segmentation_mask

# Example usage:
# vit_model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
# segmentation_model = ViTUNet(vit_model, num_classes=2)  # 2 classes: background and tumor
# image_path = 'path/to/brain/mri.jpg'
# tumor_mask = segment_brain_tumor(segmentation_model, image_path)

# Visualize the result
# import matplotlib.pyplot as plt
# plt.imshow(tumor_mask, cmap='hot')
# plt.title('Brain Tumor Segmentation')
# plt.colorbar()
# plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into Vision Transformers and their applications, here are some valuable resources:

1. Original ViT paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2020) ArXiv link: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
2. DeiT: Data-efficient Image Transformers ArXiv link: [https://arxiv.org/abs/2012.12877](https://arxiv.org/abs/2012.12877)
3. ViT for Object Detection: "End-to-End Object Detection with Transformers" (DETR) ArXiv link: [https://arxiv.org/abs/2005.12872](https://arxiv.org/abs/2005.12872)
4. ViT for Semantic Segmentation: "Segmenter: Transformer for Semantic Segmentation" ArXiv link: [https://arxiv.org/abs/2105.05633](https://arxiv.org/abs/2105.05633)
5. Swin Transformer: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" ArXiv link: [https://arxiv.org/abs/2103.14030](https://arxiv.org/abs/2103.14030)

These resources provide in-depth explanations, architectural details, and performance comparisons for various Vision Transformer models and their applications in computer vision tasks.

