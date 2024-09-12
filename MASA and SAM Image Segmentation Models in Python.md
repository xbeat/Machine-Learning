## MASA and SAM Image Segmentation Models in Python
Slide 1: Introduction to MASA and SAM

MASA (Matching Anything by Segmenting Anything) and SAM (Segment Anything Model) are advanced computer vision models designed to perform image segmentation tasks. These models can identify and outline specific objects or regions within an image, making them useful for various applications in image processing and analysis.

```python
import torch
from transformers import SamModel, SamProcessor

# Load pre-trained SAM model
model = SamModel.from_pretrained("facebook/sam-vit-huge")
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# Example image path
image_path = "path/to/your/image.jpg"

# Process the image
inputs = processor(Image.open(image_path), return_tensors="pt")

# Generate masks
outputs = model(**inputs)

# Extract masks
masks = outputs.pred_masks.squeeze().cpu().numpy()
```

Slide 2: MASA: Matching Anything by Segmenting Anything

MASA is an extension of the SAM model that combines segmentation capabilities with matching algorithms. It can identify and match similar objects across different images or within the same image. This makes MASA particularly useful for tasks such as object tracking, image retrieval, and instance segmentation.

```python
import torch
from torchvision.ops import box_iou

def match_objects(masks1, masks2, iou_threshold=0.5):
    # Convert masks to bounding boxes
    boxes1 = masks_to_boxes(masks1)
    boxes2 = masks_to_boxes(masks2)

    # Compute IoU between all pairs of boxes
    iou_matrix = box_iou(boxes1, boxes2)

    # Find matches based on IoU threshold
    matches = torch.nonzero(iou_matrix > iou_threshold)

    return matches

def masks_to_boxes(masks):
    # Convert binary masks to bounding boxes
    n = masks.shape[0]
    boxes = torch.zeros((n, 4), dtype=torch.float32)
    for i in range(n):
        y, x = torch.where(masks[i] > 0)
        boxes[i] = torch.tensor([x.min(), y.min(), x.max(), y.max()])
    return boxes
```

Slide 3: SAM: Segment Anything Model

SAM is a powerful and versatile image segmentation model developed by Meta AI. It can segment any object in an image based on various types of prompts, such as points, boxes, or text descriptions. SAM's architecture allows it to generalize well to new objects and scenes, making it highly adaptable for various segmentation tasks.

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def visualize_mask(image, mask):
    # Overlay mask on image
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask] = [255, 0, 0, 128]  # Red with 50% opacity

    # Combine image and overlay
    result = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(overlay))

    # Display result
    plt.imshow(result)
    plt.axis('off')
    plt.show()

# Example usage
image = Image.open("path/to/your/image.jpg")
mask = generate_mask(image)  # Assuming you have a function to generate the mask
visualize_mask(image, mask)
```

Slide 4: MASA: Object Matching Across Images

One of the key features of MASA is its ability to match objects across different images. This can be particularly useful in applications such as image retrieval, where you want to find similar objects in a large dataset of images.

```python
import torch
from torchvision.ops import box_iou

def match_objects_across_images(masks1, masks2, iou_threshold=0.5):
    boxes1 = masks_to_boxes(masks1)
    boxes2 = masks_to_boxes(masks2)

    iou_matrix = box_iou(boxes1, boxes2)
    matches = torch.nonzero(iou_matrix > iou_threshold)

    return matches

# Example usage
masks_image1 = generate_masks(image1)  # Assume we have these functions
masks_image2 = generate_masks(image2)

matches = match_objects_across_images(masks_image1, masks_image2)

print(f"Found {len(matches)} matching objects between the two images.")
```

Slide 5: SAM: Prompt-Based Segmentation

SAM's unique feature is its ability to segment objects based on various types of prompts. This includes point prompts, box prompts, and even text prompts. This flexibility allows users to guide the segmentation process in intuitive ways.

```python
from transformers import SamModel, SamProcessor

def segment_with_prompt(image_path, prompt, prompt_type="point"):
    model = SamModel.from_pretrained("facebook/sam-vit-huge")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    image = Image.open(image_path)
    
    if prompt_type == "point":
        inputs = processor(image, input_points=[prompt], return_tensors="pt")
    elif prompt_type == "box":
        inputs = processor(image, input_boxes=[prompt], return_tensors="pt")
    else:
        raise ValueError("Unsupported prompt type")

    outputs = model(**inputs)
    masks = outputs.pred_masks.squeeze().cpu().numpy()

    return masks

# Example usage
image_path = "path/to/your/image.jpg"
point_prompt = [100, 100]  # x, y coordinates
mask = segment_with_prompt(image_path, point_prompt, "point")
```

Slide 6: MASA: Instance Segmentation

MASA extends SAM's capabilities to perform instance segmentation, which involves identifying and segmenting individual instances of objects within an image. This is particularly useful in scenarios where multiple instances of the same object class are present.

```python
import numpy as np
from skimage.measure import label

def perform_instance_segmentation(image_path):
    # Assume we have functions to generate masks and match objects
    masks = generate_masks(image_path)
    
    # Perform connected component labeling to separate instances
    instance_map = np.zeros_like(masks[0], dtype=int)
    for i, mask in enumerate(masks):
        labeled_mask = label(mask)
        instance_map[labeled_mask > 0] = i + 1

    return instance_map

# Example usage
image_path = "path/to/your/image.jpg"
instance_map = perform_instance_segmentation(image_path)

plt.imshow(instance_map, cmap='tab20')
plt.title("Instance Segmentation Result")
plt.axis('off')
plt.show()
```

Slide 7: SAM: Zero-Shot Performance

One of SAM's most impressive features is its zero-shot performance. It can segment objects it has never seen before during training, making it highly versatile for real-world applications where new object types may be encountered.

```python
import random

def zero_shot_segmentation(image_path, num_points=5):
    model = SamModel.from_pretrained("facebook/sam-vit-huge")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    image = Image.open(image_path)
    width, height = image.size

    # Generate random points as prompts
    random_points = [(random.randint(0, width), random.randint(0, height)) 
                     for _ in range(num_points)]

    inputs = processor(image, input_points=random_points, return_tensors="pt")
    outputs = model(**inputs)
    masks = outputs.pred_masks.squeeze().cpu().numpy()

    return masks

# Example usage
image_path = "path/to/your/image.jpg"
zero_shot_masks = zero_shot_segmentation(image_path)

# Visualize results
fig, axs = plt.subplots(1, len(zero_shot_masks), figsize=(15, 5))
for i, mask in enumerate(zero_shot_masks):
    axs[i].imshow(mask)
    axs[i].set_title(f"Segment {i+1}")
    axs[i].axis('off')
plt.show()
```

Slide 8: MASA: Object Tracking

MASA's ability to match objects across images makes it well-suited for object tracking in video sequences. By segmenting objects in each frame and matching them across consecutive frames, we can track the movement and changes of objects over time.

```python
import cv2

def track_objects(video_path, initial_mask):
    cap = cv2.VideoCapture(video_path)
    
    tracked_masks = [initial_mask]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Generate masks for current frame
        current_masks = generate_masks(frame)
        
        # Match with previous frame's mask
        matches = match_objects_across_images(tracked_masks[-1], current_masks)
        
        if len(matches) > 0:
            # Assuming we're tracking a single object, take the first match
            tracked_masks.append(current_masks[matches[0][1]])
        else:
            # If no match found, use the previous mask
            tracked_masks.append(tracked_masks[-1])
    
    cap.release()
    return tracked_masks

# Example usage
video_path = "path/to/your/video.mp4"
initial_mask = generate_initial_mask(video_path)  # Assume we have this function
tracked_masks = track_objects(video_path, initial_mask)

# Visualize tracking results (e.g., show every 10th frame)
for i in range(0, len(tracked_masks), 10):
    plt.imshow(tracked_masks[i])
    plt.title(f"Frame {i}")
    plt.axis('off')
    plt.show()
```

Slide 9: SAM: Text-to-Mask Generation

SAM can generate masks based on text descriptions, allowing for more intuitive and flexible segmentation tasks. This feature bridges the gap between natural language processing and computer vision.

```python
from transformers import SamModel, SamProcessor, CLIPProcessor, CLIPModel

def text_to_mask(image_path, text_prompt):
    # Load SAM and CLIP models
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge")
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Process image and text with CLIP
    image = Image.open(image_path)
    clip_inputs = clip_processor(text=[text_prompt], images=[image], return_tensors="pt", padding=True)
    clip_outputs = clip_model(**clip_inputs)

    # Use CLIP's image features as input to SAM
    sam_inputs = sam_processor(image, return_tensors="pt")
    sam_inputs["pixel_values"] = clip_outputs.image_embeds

    # Generate mask with SAM
    sam_outputs = sam_model(**sam_inputs)
    masks = sam_outputs.pred_masks.squeeze().cpu().numpy()

    return masks

# Example usage
image_path = "path/to/your/image.jpg"
text_prompt = "a cat sitting on a couch"
masks = text_to_mask(image_path, text_prompt)

# Visualize the result
plt.imshow(masks[0])
plt.title(f"Mask for: {text_prompt}")
plt.axis('off')
plt.show()
```

Slide 10: MASA: Semantic Segmentation

While SAM focuses on instance segmentation, MASA can be extended to perform semantic segmentation, where each pixel is classified into a predefined set of categories. This is achieved by combining SAM's segmentation capabilities with a classification model.

```python
from torchvision.models import resnet50
from torchvision.transforms import functional as F

def semantic_segmentation(image_path, categories):
    # Load pre-trained models
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge")
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    classifier = resnet50(pretrained=True)

    # Generate masks with SAM
    image = Image.open(image_path)
    sam_inputs = sam_processor(image, return_tensors="pt")
    sam_outputs = sam_model(**sam_inputs)
    masks = sam_outputs.pred_masks.squeeze().cpu().numpy()

    # Classify each masked region
    semantic_map = np.zeros_like(masks[0], dtype=int)
    for i, mask in enumerate(masks):
        masked_image = image * mask[:, :, np.newaxis]
        tensor_image = F.to_tensor(masked_image).unsqueeze(0)
        with torch.no_grad():
            prediction = classifier(tensor_image)
        class_id = prediction.argmax().item()
        semantic_map[mask > 0] = class_id

    return semantic_map, categories[class_id]

# Example usage
image_path = "path/to/your/image.jpg"
categories = ['background', 'person', 'animal', 'vehicle', 'furniture']
semantic_map, detected_category = semantic_segmentation(image_path, categories)

plt.imshow(semantic_map, cmap='tab20')
plt.title(f"Semantic Segmentation: {detected_category}")
plt.axis('off')
plt.show()
```

Slide 11: Real-Life Example: Medical Image Analysis

MASA and SAM can be powerful tools in medical image analysis, helping to identify and segment various anatomical structures or abnormalities in medical scans.

```python
import numpy as np
from skimage import measure

def analyze_medical_image(image_path):
    # Load and preprocess the medical image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_array = np.array(image)

    # Generate masks using SAM (assuming we have this function)
    masks = generate_masks(image_path)

    # Analyze each segmented region
    for i, mask in enumerate(masks):
        # Calculate area and perimeter of the segmented region
        region_props = measure.regionprops(mask.astype(int))[0]
        area = region_props.area
        perimeter = region_props.perimeter

        # Calculate mean intensity in the region
        mean_intensity = np.mean(image_array[mask > 0])

        print(f"Region {i+1}:")
        print(f"  Area: {area} pixels")
        print(f"  Perimeter: {perimeter:.2f} pixels")
        print(f"  Mean Intensity: {mean_intensity:.2f}")

# Example usage
medical_image_path = "path/to/medical/image.jpg"
analyze_medical_image(medical_image_path)
```

Slide 12: Real-Life Example: Autonomous Driving

In autonomous driving systems, MASA and SAM can be used for real-time object detection and segmentation, helping vehicles identify and track other vehicles, pedestrians, and road features.

```python
import cv2
import numpy as np

def process_driving_scene(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Generate masks for the current frame
        masks = generate_masks(frame)

        # Classify and color-code different objects
        colored_mask = np.zeros_like(frame)
        for i, mask in enumerate(masks):
            object_type = classify_object(frame, mask)  # Assume we have this function
            if object_type == 'vehicle':
                colored_mask[mask > 0] = [255, 0, 0]  # Red for vehicles
            elif object_type == 'pedestrian':
                colored_mask[mask > 0] = [0, 255, 0]  # Green for pedestrians
            elif object_type == 'road':
                colored_mask[mask > 0] = [0, 0, 255]  # Blue for road

        # Overlay the colored mask on the original frame
        result = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

        # Display the result
        cv2.imshow('Autonomous Driving Scene Analysis', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = "path/to/driving/video.mp4"
process_driving_scene(video_path)
```

Slide 13: MASA: Fine-grained Object Matching

MASA's ability to match objects can be extended to perform fine-grained object matching, which is useful in applications like visual search or product recognition.

```python
import torch
from torchvision import models, transforms
from PIL import Image

def extract_features(image, model):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        features = model(input_batch)
    return features.squeeze()

def fine_grained_matching(query_image, database_images):
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()  # Remove the final fully connected layer

    query_features = extract_features(query_image, model)
    
    similarities = []
    for db_image in database_images:
        db_features = extract_features(db_image, model)
        similarity = torch.cosine_similarity(query_features, db_features, dim=0)
        similarities.append(similarity.item())

    return similarities

# Example usage
query_image = Image.open("path/to/query/image.jpg")
database_images = [Image.open(f"path/to/database/image_{i}.jpg") for i in range(10)]
similarities = fine_grained_matching(query_image, database_images)

# Print the similarities
for i, sim in enumerate(similarities):
    print(f"Similarity with database image {i}: {sim:.4f}")
```

Slide 14: SAM: Interactive Segmentation

SAM's ability to work with different types of prompts makes it ideal for interactive segmentation tasks, where users can refine the segmentation results in real-time.

```python
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class InteractiveSegmentation:
    def __init__(self, image_path):
        self.image = Image.open(image_path)
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)
        self.points = []
        self.mask = None

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.button_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.button = Button(self.button_ax, 'Segment')
        self.button.on_clicked(self.segment)

    def on_click(self, event):
        if event.inaxes == self.ax:
            self.points.append((event.xdata, event.ydata))
            self.ax.plot(event.xdata, event.ydata, 'ro')
            self.fig.canvas.draw()

    def segment(self, event):
        if self.points:
            # Assume we have a function to segment using SAM
            self.mask = segment_with_points(self.image, self.points)
            self.ax.imshow(self.mask, alpha=0.5)
            self.fig.canvas.draw()

# Example usage
segmenter = InteractiveSegmentation("path/to/your/image.jpg")
plt.show()
```

Slide 15: Additional Resources

For more information on MASA and SAM, refer to the following resources:

1. SAM: Segment Anything (ArXiv paper): [https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)
2. Segment Anything Model (SAM) GitHub repository: [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
3. MASA: Matching Anything by Segmenting Anything (ArXiv paper): [https://arxiv.org/abs/2303.14170](https://arxiv.org/abs/2303.14170)

These resources provide in-depth information about the models' architectures, training procedures, and potential applications. They also offer code implementations and pre-trained models for researchers and developers to experiment with.

