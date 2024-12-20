## Object Counting in Videos with Python
Slide 1: Introduction to Object Counting in Videos

Object counting in videos is a crucial task in computer vision that involves automatically identifying and quantifying specific objects or entities within video sequences. This process has numerous applications, from traffic monitoring to wildlife population studies. In this presentation, we'll explore how to implement object counting using machine learning techniques in Python.

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained object detection model
model = load_model('object_detection_model.h5')

def count_objects(video_path):
    cap = cv2.VideoCapture(video_path)
    object_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame and perform object detection
        preprocessed_frame = preprocess_frame(frame)
        detections = model.predict(preprocessed_frame)
        
        # Count detected objects
        object_count += len(detections)
    
    cap.release()
    return object_count

# Example usage
video_path = 'sample_video.mp4'
total_objects = count_objects(video_path)
print(f"Total objects counted: {total_objects}")
```

Slide 2: Video Processing Fundamentals

Before diving into object counting, it's essential to understand how to process videos in Python. We'll use the OpenCV library to read video frames and perform basic operations.

```python
import cv2

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        # Display the processed frame
        cv2.imshow('Processed Frame', blurred_frame)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")

# Example usage
video_path = 'sample_video.mp4'
process_video(video_path)
```

Slide 3: Object Detection with YOLO

YOLO (You Only Look Once) is a popular real-time object detection system. We'll use the YOLOv3 model to detect objects in video frames.

```python
import cv2
import numpy as np

def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

def detect_objects(frame, net, output_layers):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids

# Load YOLO
net, classes, output_layers = load_yolo()

# Example usage (assuming you have a video file)
video_path = 'sample_video.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    boxes, confidences, class_ids = detect_objects(frame, net, output_layers)
    
    # Draw bounding boxes and labels
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Object Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Slide 4: Tracking Objects Across Frames

To accurately count objects in videos, we need to track them across multiple frames. We'll use a simple centroid tracking algorithm to accomplish this.

```python
import numpy as np
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

# Example usage
ct = CentroidTracker()
rects = [(10, 10, 20, 20), (30, 30, 40, 40)]  # Example bounding boxes
objects = ct.update(rects)
print(f"Tracked objects: {objects}")
```

Slide 5: Counting Algorithm

Now that we can detect and track objects, let's implement a counting algorithm that keeps track of objects entering and leaving a predefined area.

```python
import cv2
import numpy as np
from centroid_tracker import CentroidTracker

def count_objects(video_path, counting_line_y):
    cap = cv2.VideoCapture(video_path)
    ct = CentroidTracker()
    object_count = 0
    counted_objects = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection (assuming we have a detect_objects function)
        boxes = detect_objects(frame)
        
        # Update object tracking
        objects = ct.update(boxes)

        # Count objects crossing the line
        for (objectID, centroid) in objects.items():
            cy = centroid[1]
            if objectID not in counted_objects and cy < counting_line_y:
                object_count += 1
                counted_objects.add(objectID)

        # Draw counting line
        cv2.line(frame, (0, counting_line_y), (frame.shape[1], counting_line_y), (0, 255, 0), 2)

        # Display count
        cv2.putText(frame, f"Count: {object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Object Counting", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return object_count

# Example usage
video_path = 'sample_video.mp4'
counting_line_y = 300  # Adjust this value based on your video
total_count = count_objects(video_path, counting_line_y)
print(f"Total objects counted: {total_count}")
```

Slide 6: Handling Different Object Classes

In real-world scenarios, we often need to count specific types of objects. Let's modify our counting algorithm to handle multiple object classes.

```python
import cv2
import numpy as np
from centroid_tracker import CentroidTracker

class MultiClassObjectCounter:
    def __init__(self, classes):
        self.classes = classes
        self.trackers = {cls: CentroidTracker() for cls in classes}
        self.counts = {cls: 0 for cls in classes}
        self.counted_objects = {cls: set() for cls in classes}

    def update(self, detections, counting_line_y):
        for cls in self.classes:
            class_detections = [d for d in detections if d['class'] == cls]
            objects = self.trackers[cls].update([d['box'] for d in class_detections])

            for (objectID, centroid) in objects.items():
                cy = centroid[1]
                if objectID not in self.counted_objects[cls] and cy < counting_line_y:
                    self.counts[cls] += 1
                    self.counted_objects[cls].add(objectID)

        return self.counts

def count_multi_class_objects(video_path, counting_line_y, classes):
    cap = cv2.VideoCapture(video_path)
    counter = MultiClassObjectCounter(classes)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection (assuming we have a detect_objects function)
        detections = detect_objects(frame)
        
        # Update object counting
        counts = counter.update(detections, counting_line_y)

        # Draw counting line
        cv2.line(frame, (0, counting_line_y), (frame.shape[1], counting_line_y), (0, 255, 0), 2)

        # Display counts
        y = 30
        for cls, count in counts.items():
            cv2.putText(frame, f"{cls}: {count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y += 30

        cv2.imshow("Multi-class Object Counting", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return counts

# Example usage
video_path = 'sample_video.mp4'
counting_line_y = 300  # Adjust this value based on your video
classes = ['person', 'car', 'bicycle']
final_counts = count_multi_class_objects(video_path, counting_line_y, classes)
print("Final counts:", final_counts)
```

Slide 7: Handling Occlusions

Occlusions pose a significant challenge in object counting. Let's implement a simple method to handle partial occlusions using object persistence.

```python
import numpy as np
from collections import deque

class OcclusionHandler:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.object_history = {}

    def update(self, objects):
        current_objects = set(objects.keys())
        
        for obj_id in list(self.disappeared.keys()):
            if obj_id not in current_objects:
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.disappeared[obj_id]
                    del self.object_history[obj_id]
            else:
                del self.disappeared[obj_id]

        for obj_id, centroid in objects.items():
            if obj_id not in self.object_history:
                self.object_history[obj_id] = deque(maxlen=30)
                self.disappeared[obj_id] = 0
            
            self.object_history[obj_id].append(centroid)

        for obj_id in list(self.disappeared.keys()):
            if obj_id not in current_objects:
                last_known_position = self.object_history[obj_id][-1]
                for current_id, current_centroid in objects.items():
                    distance = np.linalg.norm(np.array(last_known_position) - np.array(current_centroid))
                    if distance < self.max_distance:
                        del self.disappeared[obj_id]
                        del objects[current_id]
                        objects[obj_id] = current_centroid
                        break

        return objects

# Example usage
occlusion_handler = OcclusionHandler()
tracked_objects = {1: (100, 100), 2: (200, 200)}
updated_objects = occlusion_handler.update(tracked_objects)
print("Updated objects after occlusion handling:", updated_objects)
```

Slide 8: Real-time Visualization

Visualizing the counting process in real-time can provide valuable insights. Let's create a simple visualization tool using OpenCV.

```python
import cv2
import numpy as np

def draw_objects(frame, objects, counts):
    for obj_id, centroid in objects.items():
        cv2.circle(frame, tuple(centroid), 4, (0, 255, 0), -1)
        cv2.putText(frame, str(obj_id), (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Total Count: {sum(counts.values())}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    y = 60
    for cls, count in counts.items():
        cv2.putText(frame, f"{cls}: {count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y += 30

    return frame

def visualize_counting(video_path, detector, tracker, counter):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)
        counts = counter.update(tracked_objects)

        visualized_frame = draw_objects(frame, tracked_objects, counts)

        cv2.imshow("Object Counting", visualized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'sample_video.mp4'
detector = ObjectDetector()  # Assume we have this class
tracker = CentroidTracker()  # From previous slides
counter = ObjectCounter()    # Assume we have this class
visualize_counting(video_path, detector, tracker, counter)
```

Slide 9: Handling Multiple Camera Views

In real-world applications, we often need to count objects across multiple camera views. Let's create a system to handle this scenario.

```python
import cv2
import numpy as np
from multiprocessing import Process, Queue

class MultiCameraCounter:
    def __init__(self, camera_urls):
        self.camera_urls = camera_urls
        self.queues = [Queue() for _ in camera_urls]
        self.processes = []

    def process_camera(self, camera_url, queue):
        cap = cv2.VideoCapture(camera_url)
        detector = ObjectDetector()  # Assume we have this class
        tracker = CentroidTracker()  # From previous slides
        counter = ObjectCounter()    # Assume we have this class

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            tracked_objects = tracker.update(detections)
            counts = counter.update(tracked_objects)

            queue.put(counts)

        cap.release()

    def start(self):
        for i, url in enumerate(self.camera_urls):
            p = Process(target=self.process_camera, args=(url, self.queues[i]))
            self.processes.append(p)
            p.start()

    def get_total_counts(self):
        total_counts = {}
        for queue in self.queues:
            counts = queue.get()
            for cls, count in counts.items():
                total_counts[cls] = total_counts.get(cls, 0) + count
        return total_counts

    def stop(self):
        for p in self.processes:
            p.terminate()

# Example usage
camera_urls = [
    'rtsp://camera1_url',
    'rtsp://camera2_url',
    'rtsp://camera3_url'
]

multi_camera_counter = MultiCameraCounter(camera_urls)
multi_camera_counter.start()

try:
    while True:
        total_counts = multi_camera_counter.get_total_counts()
        print("Total counts across all cameras:", total_counts)
except KeyboardInterrupt:
    multi_camera_counter.stop()
```

Slide 10: Dealing with Low-Light Conditions

Low-light conditions can significantly impact object detection and counting accuracy. Let's implement a simple preprocessing step to enhance low-light video frames.

```python
import cv2
import numpy as np

def enhance_low_light(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_bgr

def process_low_light_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        enhanced_frame = enhance_low_light(frame)

        # Perform object detection and counting on the enhanced frame
        # (Assume we have detect_and_count function)
        counts = detect_and_count(enhanced_frame)

        cv2.imshow("Original", frame)
        cv2.imshow("Enhanced", enhanced_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'low_light_video.mp4'
process_low_light_video(video_path)
```

Slide 11: Handling Fast-Moving Objects

Fast-moving objects can be challenging to track and count accurately. Let's implement a motion-based detection method to improve counting for rapid movements.

```python
import cv2
import numpy as np

class FastObjectDetector:
    def __init__(self, min_area=500):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.min_area = min_area

    def detect(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)[1]
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append((x, y, x+w, y+h))
        
        return detections

def process_fast_moving_objects(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = FastObjectDetector()
    tracker = CentroidTracker()  # Assume we have this from previous slides

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        objects = tracker.update(detections)

        # Draw bounding boxes and IDs
        for (objectID, centroid) in objects.items():
            text = f"ID {objectID}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        cv2.imshow("Fast Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'fast_moving_objects.mp4'
process_fast_moving_objects(video_path)
```

Slide 12: Dealing with Crowded Scenes

Crowded scenes present unique challenges for object counting. Let's implement a density-based approach to estimate object counts in highly populated areas.

```python
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

class DensityEstimator:
    def __init__(self, model_path):
        self.model = cv2.dnn.readNetFromCaffe(model_path + '.prototxt', model_path + '.caffemodel')

    def estimate_density(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255, (224, 224), (0, 0, 0), swapRB=False, crop=False)
        self.model.setInput(blob)
        density_map = self.model.forward()
        density_map = density_map.squeeze()
        density_map = gaussian_filter(density_map, sigma=3)
        return density_map

def count_crowded_scene(image_path, model_path):
    image = cv2.imread(image_path)
    density_estimator = DensityEstimator(model_path)
    
    density_map = density_estimator.estimate_density(image)
    count = np.sum(density_map)
    
    # Visualize density map
    density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min())
    density_map = (density_map * 255).astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
    
    # Overlay density map on original image
    overlay = cv2.addWeighted(image, 0.7, density_map, 0.3, 0)
    
    cv2.putText(overlay, f"Estimated Count: {int(count)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Crowd Density Estimation", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'crowded_scene.jpg'
model_path = 'density_estimation_model'
count_crowded_scene(image_path, model_path)
```

Slide 13: Performance Optimization

To handle real-time video processing, we need to optimize our object counting pipeline. Let's implement some performance improvements using multiprocessing and GPU acceleration.

```python
import cv2
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

class OptimizedObjectCounter:
    def __init__(self, num_processes=4):
        self.num_processes = num_processes
        self.detector = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.pool = ThreadPoolExecutor(max_workers=num_processes)

    def detect_objects(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
        self.detector.setInput(blob)
        output_layers_names = self.detector.getUnconnectedOutLayersNames()
        layerOutputs = self.detector.forward(output_layers_names)
        return self.process_detections(layerOutputs, frame.shape[:2])

    def process_detections(self, layerOutputs, frame_shape):
        boxes, confidences, class_ids = [], [], []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y = int(detection[0] * frame_shape[1]), int(detection[1] * frame_shape[0])
                    w, h = int(detection[2] * frame_shape[1]), int(detection[3] * frame_shape[0])
                    x, y = int(center_x - w/2), int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        return boxes, confidences, class_ids

    def count_objects(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count, total_objects = 0, 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % self.num_processes == 0:
                futures = [self.pool.submit(self.detect_objects, frame) for _ in range(self.num_processes)]
                results = [future.result() for future in futures]
                for boxes, _, _ in results:
                    total_objects += len(boxes)

        cap.release()
        return total_objects

# Example usage
counter = OptimizedObjectCounter()
video_path = 'sample_video.mp4'
total_count = counter.count_objects(video_path)
print(f"Total objects counted: {total_count}")
```

Slide 14: Real-life Example: Pedestrian Counting

Let's apply our object counting techniques to a real-world scenario: counting pedestrians in a busy street.

```python
import cv2
import numpy as np
from optimized_object_counter import OptimizedObjectCounter

def count_pedestrians(video_path, roi_points):
    counter = OptimizedObjectCounter()
    cap = cv2.VideoCapture(video_path)
    pedestrian_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create a mask for the region of interest (ROI)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(roi_points, dtype=np.int32)], 255)

        # Apply the mask to the frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Detect and count pedestrians in the ROI
        boxes, confidences, class_ids = counter.detect_objects(masked_frame)
        pedestrians = [box for box, class_id in zip(boxes, class_ids) if class_id == 0]  # Assuming class_id 0 is pedestrian
        pedestrian_count += len(pedestrians)

        # Visualize the results
        for box in pedestrians:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"Pedestrian Count: {pedestrian_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Pedestrian Counting", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return pedestrian_count

# Example usage
video_path = 'busy_street.mp4'
roi_points = [(100, 100), (500, 100), (500, 400), (100, 400)]  # Define your ROI
total_pedestrians = count_pedestrians(video_path, roi_points)
print(f"Total pedestrians counted: {total_pedestrians}")
```

Slide 15: Additional Resources

For further exploration of object counting in videos using machine learning and Python, consider the following resources:

1. ArXiv paper: "CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes" by Y. Li et al. (2018) URL: [https://arxiv.org/abs/1802.10062](https://arxiv.org/abs/1802.10062)
2. ArXiv paper: "Scale-Aware Attention Network for Crowd Counting" by Y. Liu et al. (2019) URL: [https://arxiv.org/abs/1901.06026](https://arxiv.org/abs/1901.06026)
3. OpenCV Documentation: [https://docs.opencv.org/](https://docs.opencv.org/)
4. TensorFlow Object Detection API: [https://github.com/tensorflow/models/tree/master/research/object\_detection](https://github.com/tensorflow/models/tree/master/research/object_detection)
5. PyTorch Vision Library: [https://pytorch.org/vision/stable/index.html](https://pytorch.org/vision/stable/index.html)

These resources provide in-depth information on advanced techniques, state-of-the-art models, and practical implementations for object counting in various scenarios.

