## 3D Packing for Self-Supervised Monocular Depth Estimation
Slide 1: Introduction to 3D Packing for Self-Supervised Monocular Depth Estimation

3D packing is a crucial technique in self-supervised monocular depth estimation, which aims to predict depth from a single image without ground truth depth annotations. This approach leverages the geometry of multiple views of a scene to learn depth estimation in an unsupervised manner.

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Define a basic encoder-decoder architecture for depth estimation
class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth

# Create an instance of the model
model = DepthEstimationModel()

# Example input
input_image = torch.randn(1, 3, 256, 256)

# Predict depth
predicted_depth = model(input_image)
print(f"Predicted depth shape: {predicted_depth.shape}")
```

Slide 2: Principle of 3D Packing

3D packing in self-supervised monocular depth estimation involves reconstructing a 3D scene from multiple 2D views. The core idea is to use the predicted depth and camera motion to warp one image to another, creating a geometric consistency loss that guides the learning process.

```python
import torch
import torch.nn.functional as F

def warp_image(image, depth, pose):
    # Generate a grid of pixel coordinates
    batch_size, _, height, width = image.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    grid = torch.stack((grid_x, grid_y)).float().to(image.device)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Convert depth to 3D points
    z = depth.squeeze(1)
    xyz = torch.cat((grid * z.unsqueeze(1), z.unsqueeze(1)), dim=1)

    # Apply pose transformation
    R, t = pose[:, :3, :3], pose[:, :3, 3]
    xyz_transformed = torch.bmm(xyz.view(batch_size, 3, -1).transpose(1, 2), R.transpose(1, 2)) + t.unsqueeze(1)

    # Project back to 2D
    xy_projected = xyz_transformed[:, :, :2] / xyz_transformed[:, :, 2:3]
    xy_projected = xy_projected.view(batch_size, height, width, 2)

    # Warp the image
    warped_image = F.grid_sample(image, xy_projected, padding_mode='border')
    return warped_image

# Example usage
image = torch.randn(1, 3, 256, 256)
depth = torch.rand(1, 1, 256, 256)
pose = torch.eye(4).unsqueeze(0)
pose[0, 0, 3] = 0.1  # Add a small translation

warped_image = warp_image(image, depth, pose)
print(f"Warped image shape: {warped_image.shape}")
```

Slide 3: Photometric Loss

The photometric loss is a key component in self-supervised depth estimation. It measures the difference between the original image and the reconstructed image obtained through 3D packing and warping.

```python
import torch
import torch.nn.functional as F

def photometric_loss(original_image, reconstructed_image):
    # Calculate the absolute difference
    diff = torch.abs(original_image - reconstructed_image)

    # Apply SSIM (Structural Similarity Index)
    ssim_loss = 1 - ssim(original_image, reconstructed_image)

    # Combine L1 and SSIM losses
    alpha = 0.85
    loss = alpha * ssim_loss + (1 - alpha) * diff.mean()

    return loss

def ssim(x, y, window_size=11, size_average=True):
    # Simplified SSIM implementation
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size//2)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size//2)

    sigma_x = F.avg_pool2d(x ** 2, window_size, stride=1, padding=window_size//2) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, window_size, stride=1, padding=window_size//2) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size//2) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# Example usage
original_image = torch.rand(1, 3, 256, 256)
reconstructed_image = torch.rand(1, 3, 256, 256)

loss = photometric_loss(original_image, reconstructed_image)
print(f"Photometric loss: {loss.item()}")
```

Slide 4: Camera Pose Estimation

Accurate camera pose estimation is crucial for 3D packing. We can use a pose network to predict the relative camera motion between two frames, which is essential for warping images in the self-supervised learning process.

```python
import torch
import torch.nn as nn

class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 16 * 16, 6)  # 6 DoF pose (3 for rotation, 3 for translation)

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], dim=1)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        pose = self.fc(x)
        return pose

# Example usage
pose_net = PoseNet()
img1 = torch.rand(1, 3, 256, 256)
img2 = torch.rand(1, 3, 256, 256)

predicted_pose = pose_net(img1, img2)
print(f"Predicted pose: {predicted_pose}")

# Convert pose to transformation matrix
def pose_vec2mat(pose_vec):
    translation = pose_vec[:, :3].unsqueeze(-1)
    rot = pose_vec[:, 3:]
    rot_mat = torch.zeros(rot.shape[0], 3, 3, device=pose_vec.device)
    rot_mat[:, 0, 0] = torch.cos(rot[:, 1]) * torch.cos(rot[:, 2])
    rot_mat[:, 0, 1] = -torch.cos(rot[:, 1]) * torch.sin(rot[:, 2])
    rot_mat[:, 0, 2] = torch.sin(rot[:, 1])
    rot_mat[:, 1, 0] = torch.sin(rot[:, 0]) * torch.sin(rot[:, 1]) * torch.cos(rot[:, 2]) + torch.cos(rot[:, 0]) * torch.sin(rot[:, 2])
    rot_mat[:, 1, 1] = -torch.sin(rot[:, 0]) * torch.sin(rot[:, 1]) * torch.sin(rot[:, 2]) + torch.cos(rot[:, 0]) * torch.cos(rot[:, 2])
    rot_mat[:, 1, 2] = -torch.sin(rot[:, 0]) * torch.cos(rot[:, 1])
    rot_mat[:, 2, 0] = -torch.cos(rot[:, 0]) * torch.sin(rot[:, 1]) * torch.cos(rot[:, 2]) + torch.sin(rot[:, 0]) * torch.sin(rot[:, 2])
    rot_mat[:, 2, 1] = torch.cos(rot[:, 0]) * torch.sin(rot[:, 1]) * torch.sin(rot[:, 2]) + torch.sin(rot[:, 0]) * torch.cos(rot[:, 2])
    rot_mat[:, 2, 2] = torch.cos(rot[:, 0]) * torch.cos(rot[:, 1])
    transform_mat = torch.cat([rot_mat, translation], dim=2)
    return transform_mat

transform_matrix = pose_vec2mat(predicted_pose)
print(f"Transformation matrix:\n{transform_matrix}")
```

Slide 5: Multi-Scale Estimation

Multi-scale estimation improves the depth prediction accuracy by considering features at different scales. This approach helps capture both fine-grained details and global context.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleDepthNet(nn.Module):
    def __init__(self):
        super(MultiScaleDepthNet, self).__init__()
        self.encoder = nn.Sequential(
            self.conv_block(3, 32, 7),
            self.conv_block(32, 64, 5),
            self.conv_block(64, 128, 3),
            self.conv_block(128, 256, 3),
            self.conv_block(256, 512, 3),
        )
        
        self.decoder4 = self.upconv(512, 256)
        self.decoder3 = self.upconv(256, 128)
        self.decoder2 = self.upconv(128, 64)
        self.decoder1 = self.upconv(64, 32)
        
        self.predict_depth4 = nn.Conv2d(256, 1, 3, padding=1)
        self.predict_depth3 = nn.Conv2d(128, 1, 3, padding=1)
        self.predict_depth2 = nn.Conv2d(64, 1, 3, padding=1)
        self.predict_depth1 = nn.Conv2d(32, 1, 3, padding=1)

    def conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x5 = self.encoder[4](x4)

        # Decoder
        upfeat4 = self.decoder4(x5)
        depth4 = self.predict_depth4(upfeat4)
        
        upfeat3 = self.decoder3(upfeat4)
        depth3 = self.predict_depth3(upfeat3)
        
        upfeat2 = self.decoder2(upfeat3)
        depth2 = self.predict_depth2(upfeat2)
        
        upfeat1 = self.decoder1(upfeat2)
        depth1 = self.predict_depth1(upfeat1)

        return [depth1, depth2, depth3, depth4]

# Example usage
model = MultiScaleDepthNet()
input_image = torch.randn(1, 3, 256, 256)
depth_predictions = model(input_image)

for i, depth in enumerate(depth_predictions):
    print(f"Depth prediction at scale {i+1}: {depth.shape}")
```

Slide 6: Smoothness Loss

Smoothness loss encourages the predicted depth to be locally smooth, which helps in producing more realistic depth maps. This loss is particularly useful in areas with low texture or uniform color.

```python
import torch
import torch.nn.functional as F

def smoothness_loss(depth, image):
    def gradient(x):
        # Compute gradients in x and y directions
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    depth_dx, depth_dy = gradient(depth)
    image_dx, image_dy = gradient(image)

    # Normalize image gradients
    image_dx = image_dx.mean(dim=1, keepdim=True)
    image_dy = image_dy.mean(dim=1, keepdim=True)

    # Edge-aware weighting
    weights_x = torch.exp(-torch.abs(image_dx))
    weights_y = torch.exp(-torch.abs(image_dy))

    # Apply edge-aware weights to depth gradients
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y

    return smoothness_x.abs().mean() + smoothness_y.abs().mean()

# Example usage
depth = torch.rand(1, 1, 256, 256)  # Predicted depth
image = torch.rand(1, 3, 256, 256)  # Input image

loss = smoothness_loss(depth, image)
print(f"Smoothness loss: {loss.item()}")
```

Slide 7: Occlusion Handling

Occlusion handling is crucial in 3D packing for self-supervised depth estimation. It addresses areas in the scene that are visible in one view but occluded in another, improving the robustness of the learning process.

```python
import torch

def compute_visibility_mask(depth, warped_depth, pose):
    # Project depth to target view
    batch_size, _, height, width = depth.shape
    pixel_coords = torch.stack(torch.meshgrid(torch.arange(width), torch.arange(height)), dim=-1)
    pixel_coords = pixel_coords.unsqueeze(0).expand(batch_size, -1, -1, -1).float()

    cam_coords = pixel_coords * depth.permute(0, 2, 3, 1)
    proj_coords = torch.matmul(cam_coords, pose[:, :3, :3].transpose(1, 2)) + pose[:, :3, 3].unsqueeze(1).unsqueeze(1)

    proj_depth = proj_coords[..., 2]

    # Compare projected depth with warped depth
    visibility_mask = (proj_depth > 0) & (proj_depth < warped_depth + 1e-3)

    return visibility_mask

# Example usage
depth = torch.rand(1, 1, 256, 256)
warped_depth = torch.rand(1, 1, 256, 256)
pose = torch.eye(4).unsqueeze(0)  # Identity pose for simplicity

visibility_mask = compute_visibility_mask(depth, warped_depth, pose)
print(f"Visibility mask shape: {visibility_mask.shape}")
print(f"Percentage of visible pixels: {visibility_mask.float().mean().item() * 100:.2f}%")
```

Slide 8: Self-Supervised Training Loop

The self-supervised training loop combines all the components we've discussed to train the depth estimation model without ground truth depth data.

```python
import torch
import torch.optim as optim

def train_step(depth_net, pose_net, images, optimizer):
    optimizer.zero_grad()

    # Predict depth for the target image
    depth = depth_net(images[:, 0])

    # Predict relative pose between target and source images
    pose = pose_net(images[:, 0], images[:, 1])

    # Warp source image to target frame
    warped_image = warp_image(images[:, 1], depth, pose)

    # Compute losses
    photo_loss = photometric_loss(images[:, 0], warped_image)
    smooth_loss = smoothness_loss(depth, images[:, 0])

    # Compute visibility mask
    visibility_mask = compute_visibility_mask(depth, warp_image(depth, depth, pose), pose)

    # Apply visibility mask to photometric loss
    masked_photo_loss = (photo_loss * visibility_mask.float()).mean()

    # Combine losses
    total_loss = masked_photo_loss + 0.1 * smooth_loss

    # Backpropagate and optimize
    total_loss.backward()
    optimizer.step()

    return total_loss.item()

# Example usage
depth_net = DepthEstimationModel()
pose_net = PoseNet()
optimizer = optim.Adam(list(depth_net.parameters()) + list(pose_net.parameters()), lr=1e-4)

# Simulated training loop
for epoch in range(10):
    batch_images = torch.rand(4, 2, 3, 256, 256)  # Batch of image pairs
    loss = train_step(depth_net, pose_net, batch_images, optimizer)
    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
```

Slide 9: Depth Refinement with CRFs

Conditional Random Fields (CRFs) can be used to refine the predicted depth maps, enforcing spatial consistency and improving edge preservation.

```python
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

def refine_depth_with_crf(depth, image):
    # Convert depth to labels (for simplicity, we quantize depth into 256 levels)
    depth_labels = (depth * 255).astype(np.uint8)
    
    # Create CRF model
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 256)
    
    # Set unary potentials
    unary = unary_from_labels(depth_labels, 256, gt_prob=0.7)
    d.setUnaryEnergy(unary)
    
    # Set pairwise potentials
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
    
    # Perform inference
    Q = d.inference(5)
    refined_depth = np.argmax(Q, axis=0).reshape(image.shape[:2])
    
    return refined_depth / 255.0  # Normalize back to [0, 1]

# Example usage (assuming depth and image are numpy arrays)
depth = np.random.rand(256, 256)
image = np.random.randint(0, 256, (256, 256, 3)).astype(np.uint8)

refined_depth = refine_depth_with_crf(depth, image)
print(f"Refined depth shape: {refined_depth.shape}")
```

Slide 10: Real-life Example: Indoor Scene Reconstruction

3D packing for self-supervised monocular depth estimation can be applied to indoor scene reconstruction, enabling applications like virtual room planning or augmented reality experiences.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def reconstruct_indoor_scene(depth_map, camera_intrinsics):
    height, width = depth_map.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert pixel coordinates to 3D points
    z = depth_map.reshape(-1)
    x = (x.reshape(-1) - camera_intrinsics[0, 2]) * z / camera_intrinsics[0, 0]
    y = (y.reshape(-1) - camera_intrinsics[1, 2]) * z / camera_intrinsics[1, 1]
    
    # Create point cloud
    points = np.column_stack((x, y, z))
    
    return points

# Simulate depth prediction
depth_net = DepthEstimationModel()
input_image = torch.rand(1, 3, 256, 256)
predicted_depth = depth_net(input_image).squeeze().detach().numpy()

# Simulate camera intrinsics
camera_intrinsics = np.array([
    [500, 0, 128],
    [0, 500, 128],
    [0, 0, 1]
])

# Reconstruct 3D scene
point_cloud = reconstruct_indoor_scene(predicted_depth, camera_intrinsics)

# Visualize point cloud
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c=point_cloud[:, 2], cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Reconstructed Indoor Scene')
plt.show()
```

Slide 11: Real-life Example: Autonomous Navigation

Self-supervised monocular depth estimation can be crucial for autonomous navigation in robotics or self-driving cars, providing depth information from a single camera.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_obstacle_map(depth_map, threshold=1.0):
    obstacle_map = depth_map < threshold
    return obstacle_map

def plan_path(obstacle_map, start, goal):
    # Simple A* path planning (pseudo-code)
    # In a real implementation, you would use a proper A* algorithm
    path = [start]
    current = start
    while current != goal:
        # Find the next safe step towards the goal
        next_step = find_safe_step(current, goal, obstacle_map)
        if next_step is None:
            print("No safe path found")
            break
        path.append(next_step)
        current = next_step
    return path

# Simulate depth prediction
depth_net = DepthEstimationModel()
input_image = torch.rand(1, 3, 256, 256)
predicted_depth = depth_net(input_image).squeeze().detach().numpy()

# Generate obstacle map
obstacle_map = generate_obstacle_map(predicted_depth)

# Plan a path
start = (0, 0)
goal = (255, 255)
path = plan_path(obstacle_map, start, goal)

# Visualize
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(predicted_depth, cmap='viridis')
plt.title('Predicted Depth')
plt.colorbar()

plt.subplot(122)
plt.imshow(obstacle_map, cmap='gray')
plt.plot([p[1] for p in path], [p[0] for p in path], 'r-')
plt.title('Obstacle Map and Planned Path')
plt.show()
```

Slide 12: Challenges and Future Directions

While 3D packing for self-supervised monocular depth estimation has shown promising results, several challenges remain:

1. Handling dynamic objects in the scene
2. Improving performance in low-light conditions
3. Dealing with reflective or transparent surfaces
4. Enhancing depth estimation at object boundaries

Slide 13: Challenges and Future Directions

Future research directions may include:

1. Incorporating semantic information for better depth estimation
2. Exploring self-supervised learning in more complex environments
3. Combining with other sensors for multi-modal depth estimation
4. Developing more efficient network architectures for real-time applications

Slide 14: Challenges and Future Directions

```python
import torch
import torch.nn as nn

class SemanticGuidedDepthNet(nn.Module):
    def __init__(self, num_classes):
        super(SemanticGuidedDepthNet, self).__init__()
        self.encoder = nn.Sequential(
            # Encoder layers
        )
        self.depth_decoder = nn.Sequential(
            # Depth decoder layers
        )
        self.semantic_decoder = nn.Sequential(
            # Semantic decoder layers
        )
        self.fusion_layer = nn.Conv2d(num_classes + 1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.encoder(x)
        depth = self.depth_decoder(features)
        semantics = self.semantic_decoder(features)
        fused_depth = self.fusion_layer(torch.cat([depth, semantics], dim=1))
        return fused_depth, semantics

# Example usage
model = SemanticGuidedDepthNet(num_classes=10)
input_image = torch.randn(1, 3, 256, 256)
depth, semantics = model(input_image)
print(f"Depth shape: {depth.shape}, Semantics shape: {semantics.shape}")
```

Slide 15: Conclusion and Additional Resources

3D packing for self-supervised monocular depth estimation is a powerful technique that enables depth prediction from a single image without the need for ground truth depth data. By leveraging geometric consistency across multiple views, this approach has shown remarkable results in various applications, from 3D reconstruction to autonomous navigation.

As the field continues to evolve, we can expect to see further improvements in accuracy, efficiency, and robustness, making monocular depth estimation an increasingly valuable tool in computer vision and robotics.

Slide 16: Additional Resources

1. Godard, C., Mac Aodha, O., & Brostow, G. J. (2017). Unsupervised Monocular Depth Estimation with Left-Right Consistency. CVPR 2017. ArXiv: [https://arxiv.org/abs/1609.03677](https://arxiv.org/abs/1609.03677)
2. Zhou, T., Brown, M., Snavely, N., & Lowe, D. G. (2017). Unsupervised Learning of Depth and Ego-Motion from Video. CVPR 2017. ArXiv: [https://arxiv.org/abs/1704.07813](https://arxiv.org/abs/1704.07813)
3. Godard, C., Mac Aodha, O., Firman, M., & Brostow, G. J. (2019). Digging Into Self-Supervised Monocular Depth Estimation. ICCV 2019. ArXiv: [https://arxiv.org/abs/1806.01260](https://arxiv.org/abs/1806.01260)

These papers provide in-depth explanations of the core concepts and advanced techniques in self-supervised monocular depth estimation.

