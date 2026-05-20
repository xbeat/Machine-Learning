## Exploring Image Transformations with OpenCV in Python
Slide 1: Introduction to Image Transformations with OpenCV

Image transformations are powerful techniques used to modify digital images. OpenCV, a popular computer vision library, provides a wide range of tools for performing these transformations in Python. We'll explore various methods to manipulate images, from basic operations to more advanced techniques.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('input_image.jpg')

# Display the original image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()
```

Slide 2: Resizing Images

Resizing is a fundamental transformation that changes the dimensions of an image. It's commonly used to reduce file size, prepare images for specific displays, or as a preprocessing step for machine learning models.

```python
# Resize the image to 50% of its original size
resized_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

# Display the resized image
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.title('Resized Image (50%)')
plt.axis('off')
plt.show()

print(f"Original size: {image.shape}")
print(f"Resized image: {resized_image.shape}")
```

Slide 3: Rotating Images

Rotation is useful for correcting image orientation or creating artistic effects. OpenCV allows us to rotate images by any angle, with options to adjust the output size and interpolation method.

```python
# Get the image dimensions
rows, cols = image.shape[:2]

# Calculate the rotation matrix
M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)

# Apply the rotation
rotated_image = cv2.warpAffine(image, M, (cols, rows))

# Display the rotated image
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.title('Rotated Image (45 degrees)')
plt.axis('off')
plt.show()
```

Slide 4: Flipping Images

Flipping is a simple yet effective transformation that can be used to augment datasets or correct mirror images. OpenCV provides an easy way to flip images horizontally, vertically, or both.

```python
# Flip the image horizontally
flipped_h = cv2.flip(image, 1)

# Flip the image vertically
flipped_v = cv2.flip(image, 0)

# Display the flipped images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(cv2.cvtColor(flipped_h, cv2.COLOR_BGR2RGB))
ax1.set_title('Horizontally Flipped')
ax1.axis('off')
ax2.imshow(cv2.cvtColor(flipped_v, cv2.COLOR_BGR2RGB))
ax2.set_title('Vertically Flipped')
ax2.axis('off')
plt.show()
```

Slide 5: Cropping Images

Cropping allows us to extract a specific region of interest from an image. This technique is useful for focusing on particular areas or removing unwanted parts of an image.

```python
# Define the region of interest (ROI)
x, y, w, h = 100, 100, 300, 300  # Example coordinates

# Crop the image
cropped_image = image[y:y+h, x:x+w]

# Display the cropped image
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title('Cropped Image')
plt.axis('off')
plt.show()
```

Slide 6: Applying Filters: Blurring

Blurring is a common operation used to reduce noise, smooth images, or create artistic effects. OpenCV offers several blurring techniques, including Gaussian blur, which we'll demonstrate here.

```python
# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

# Display the original and blurred images side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image')
ax1.axis('off')
ax2.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
ax2.set_title('Blurred Image')
ax2.axis('off')
plt.show()
```

Slide 7: Edge Detection

Edge detection is crucial in image processing and computer vision tasks. It helps identify boundaries of objects within images. The Canny edge detector is a popular algorithm for this purpose.

```python
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 100, 200)

# Display the original image and the edge-detected image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image')
ax1.axis('off')
ax2.imshow(edges, cmap='gray')
ax2.set_title('Edge Detected Image')
ax2.axis('off')
plt.show()
```

Slide 8: Thresholding

Thresholding is a technique used to create binary images, which can be useful for segmentation tasks. It separates pixels into two categories based on their intensity values.

```python
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Display the original grayscale and thresholded images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(gray, cmap='gray')
ax1.set_title('Grayscale Image')
ax1.axis('off')
ax2.imshow(binary, cmap='gray')
ax2.set_title('Binary Thresholded Image')
ax2.axis('off')
plt.show()
```

Slide 9: Color Space Conversion

OpenCV supports various color space conversions, which can be useful for different image processing tasks. Here, we'll demonstrate converting an image from BGR (OpenCV's default) to HSV (Hue, Saturation, Value) color space.

```python
# Convert BGR to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Display the original and HSV images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image (RGB)')
ax1.axis('off')
ax2.imshow(hsv_image)
ax2.set_title('HSV Image')
ax2.axis('off')
plt.show()
```

Slide 10: Perspective Transform

Perspective transformation is used to correct distorted images or to change the viewpoint of an image. It's particularly useful in scenarios like document scanning or correcting skewed images.

```python
# Define source and destination points
src_pts = np.float32([[0, 0], [image.shape[1]-1, 0], 
                      [0, image.shape[0]-1], [image.shape[1]-1, image.shape[0]-1]])
dst_pts = np.float32([[50, 50], [image.shape[1]-100, 100], 
                      [100, image.shape[0]-100], [image.shape[1]-50, image.shape[0]-50]])

# Calculate perspective transform matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply perspective transform
warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

# Display the original and warped images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image')
ax1.axis('off')
ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
ax2.set_title('Warped Image')
ax2.axis('off')
plt.show()
```

Slide 11: Morphological Operations

Morphological operations are powerful tools for image processing, particularly useful for noise removal, image segmentation, and feature extraction. We'll demonstrate dilation and erosion, two fundamental morphological operations.

```python
# Create a kernel for morphological operations
kernel = np.ones((5,5), np.uint8)

# Apply dilation
dilated = cv2.dilate(image, kernel, iterations=1)

# Apply erosion
eroded = cv2.erode(image, kernel, iterations=1)

# Display the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Original')
ax1.axis('off')
ax2.imshow(cv2.cvtColor(dilated, cv2.COLOR_BGR2RGB))
ax2.set_title('Dilated')
ax2.axis('off')
ax3.imshow(cv2.cvtColor(eroded, cv2.COLOR_BGR2RGB))
ax3.set_title('Eroded')
ax3.axis('off')
plt.show()
```

Slide 12: Real-Life Example: Document Scanner

Let's apply our knowledge to create a simple document scanner. We'll use edge detection, perspective transform, and thresholding to simulate scanning a document from an image.

```python
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# Load image and preprocess
image = cv2.imread('document.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# Find contours and apply perspective transform
cnts, _ = cv2.findContours(edged.(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

warped = four_point_transform(image, screenCnt.reshape(4, 2))

# Apply adaptive thresholding
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Display results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image')
ax1.axis('off')
ax2.imshow(T, cmap='gray')
ax2.set_title('Scanned Document')
ax2.axis('off')
plt.show()
```

Slide 13: Real-Life Example: Object Tracking

Object tracking is a crucial application in computer vision. We'll demonstrate a simple color-based object tracking using HSV color space and contour detection.

```python
import cv2
import numpy as np

def track_object(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range of blue color in HSV
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(c)
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return frame

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply tracking
    result = track_object(frame)
    
    # Display result
    cv2.imshow('Object Tracking', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Slide 14: Additional Resources

For those interested in diving deeper into image processing and computer vision with OpenCV, here are some valuable resources:

1. OpenCV Documentation: [https://docs.opencv.org/](https://docs.opencv.org/) This is the official documentation for OpenCV, providing detailed explanations of all functions and modules.
2. "Computer Vision: Algorithms and Applications" by Richard Szeliski A comprehensive textbook covering various aspects of computer vision.
3. ArXiv paper: "A Survey of Deep Learning Techniques for Image Processing" by Khan et al. (2020) ArXiv: [https://arxiv.org/abs/2009.08992](https://arxiv.org/abs/2009.08992) This survey provides an overview of deep learning techniques applied to image processing tasks.
4. PyImageSearch blog by Adrian Rosebrock Offers practical tutorials and projects related to computer vision and OpenCV.
5. OpenCV-Python Tutorials on the official OpenCV website Provides step-by-step guides for various image processing tasks using OpenCV and Python.

These resources offer a mix of theoretical knowledge and practical applications, suitable for beginners and intermediate learners alike. Remember to practice regularly and experiment with different image processing techniques to solidify your understanding.

