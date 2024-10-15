## Interactive Air Canvas with Gesture Control using Python
Slide 1: Introduction to Interactive Air Canvas

Interactive Air Canvas is a technology that allows users to draw or interact with digital content using hand gestures in the air. This system combines computer vision, gesture recognition, and Python programming to create an intuitive and engaging user experience.

```python
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Create a blank canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame with MediaPipe Hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        # Draw on canvas based on hand position
        # (Code for drawing will be added in later slides)
        pass
    
    # Display the canvas
    cv2.imshow('Air Canvas', canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Slide 2: Setting Up the Environment

To create an Interactive Air Canvas, we need to set up our Python environment with the necessary libraries. The main libraries we'll use are OpenCV for image processing and video capture, NumPy for numerical operations, and MediaPipe for hand tracking.

```python
# Install required libraries
!pip install opencv-python numpy mediapipe

# Import libraries
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Create a blank canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

print("Environment setup complete!")
```

Slide 3: Capturing Video Input

The first step in creating our Interactive Air Canvas is to capture video input from the camera. We'll use OpenCV's VideoCapture class to access the camera and read frames in real-time.

```python
import cv2

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Display the frame
    cv2.imshow('Video Feed', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Slide 4: Hand Detection with MediaPipe

MediaPipe provides powerful hand detection and tracking capabilities. We'll use it to detect and track the user's hand in real-time, which will serve as our "brush" for the air canvas.

```python
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
```

Slide 5: Creating the Drawing Canvas

Now that we can detect hands, let's create a blank canvas where we'll draw. We'll use NumPy to create a black image that will serve as our canvas.

```python
import numpy as np
import cv2

# Create a black canvas
canvas_width, canvas_height = 640, 480
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Function to display the canvas
def show_canvas():
    cv2.imshow('Air Canvas', canvas)
    cv2.waitKey(1)

# Example: Draw a white circle on the canvas
cv2.circle(canvas, (320, 240), 50, (255, 255, 255), -1)
show_canvas()

# Wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Slide 6: Implementing Drawing Functionality

Let's implement the core drawing functionality. We'll track the position of the index finger and use it to draw on our canvas.

```python
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_point = None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the position of the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * 640), int(index_finger_tip.y * 480)
            
            if prev_point is not None:
                cv2.line(canvas, prev_point, (x, y), (255, 255, 255), 2)
            
            prev_point = (x, y)
    else:
        prev_point = None
    
    cv2.imshow('Air Canvas', canvas)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Slide 7: Adding Color Selection

To make our Air Canvas more interactive, let's add the ability to change colors. We'll define color regions on the screen that the user can point to for selecting different colors.

```python
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
color = (255, 255, 255)  # Default color: white

def get_color(x, y):
    if y < 50:
        if x < 213:
            return (0, 0, 255)  # Red
        elif x < 426:
            return (0, 255, 0)  # Green
        else:
            return (255, 0, 0)  # Blue
    return color

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * 640), int(index_finger_tip.y * 480)
            
            color = get_color(x, y)
            cv2.circle(canvas, (x, y), 5, color, -1)
    
    # Draw color selection bars
    cv2.rectangle(image, (0, 0), (213, 50), (0, 0, 255), -1)
    cv2.rectangle(image, (213, 0), (426, 50), (0, 255, 0), -1)
    cv2.rectangle(image, (426, 0), (640, 50), (255, 0, 0), -1)
    
    cv2.imshow('Air Canvas', cv2.add(image, canvas))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Slide 8: Implementing Eraser Functionality

Let's add an eraser function to our Air Canvas. We'll use a specific hand gesture to activate the eraser mode.

```python
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
color = (255, 255, 255)
eraser_mode = False

def is_eraser_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return abs(thumb_tip.y - index_tip.y) < 0.02

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            eraser_mode = is_eraser_gesture(hand_landmarks)
            
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * 640), int(index_finger_tip.y * 480)
            
            if eraser_mode:
                cv2.circle(canvas, (x, y), 20, (0, 0, 0), -1)
            else:
                cv2.circle(canvas, (x, y), 5, color, -1)
    
    cv2.putText(image, "Eraser: " + ("On" if eraser_mode else "Off"), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Air Canvas', cv2.add(image, canvas))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Slide 9: Saving and Loading Drawings

Let's add functionality to save our drawings and load them later. We'll use OpenCV's image reading and writing functions for this purpose.

```python
import cv2
import numpy as np

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

def save_drawing():
    filename = "air_canvas_drawing.png"
    cv2.imwrite(filename, canvas)
    print(f"Drawing saved as {filename}")

def load_drawing():
    global canvas
    filename = "air_canvas_drawing.png"
    loaded_canvas = cv2.imread(filename)
    if loaded_canvas is not None:
        canvas = loaded_canvas
        print(f"Drawing loaded from {filename}")
    else:
        print(f"Failed to load drawing from {filename}")

# Example usage in the main loop
while True:
    # ... (other code)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        save_drawing()
    elif key == ord('l'):
        load_drawing()
    elif key == 27:  # Esc key
        break

    cv2.imshow('Air Canvas', canvas)

cv2.destroyAllWindows()
```

Slide 10: Implementing Brush Size Control

Let's add the ability to change brush size using hand gestures. We'll use the distance between the thumb and index finger to control the brush size.

```python
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
color = (255, 255, 255)
brush_size = 5

def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Calculate brush size based on thumb-index distance
            distance = get_distance(thumb_tip, index_tip)
            brush_size = int(distance * 100)  # Adjust multiplier as needed
            brush_size = max(1, min(brush_size, 50))  # Limit brush size between 1 and 50
            
            x, y = int(index_tip.x * 640), int(index_tip.y * 480)
            cv2.circle(canvas, (x, y), brush_size, color, -1)
    
    cv2.putText(image, f"Brush Size: {brush_size}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Air Canvas', cv2.add(image, canvas))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Slide 11: Adding Text to the Canvas

Let's implement a feature to add text to our Air Canvas using hand gestures. We'll use a specific gesture to enter "text mode" and then track hand movement to write text.

```python
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
color = (255, 255, 255)
text_mode = False
text = ""

def is_text_mode_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    return (thumb_tip.y < index_tip.y) and (thumb_tip.y < middle_tip.y)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if is_text_mode_gesture(hand_landmarks):
                text_mode = not text_mode
                if text_mode:
                    text = ""
            
            if text_mode:
                # Implement text input based on hand movement
                # This is a simplified version; you may want to develop a more sophisticated system
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_tip.x * 640), int(index_tip.y * 480)
                if x < 213:
                    text += "A"
                elif x < 426:
                    text += "B"
                else:
                    text += "C"
    
    if text_mode:
        cv2.putText(image, f"Text: {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.putText(canvas, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        text = ""
    
    cv2.imshow('Air Canvas', cv2.add(image, canvas))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Slide 12: Implementing Shape Drawing

Let's add functionality to draw basic shapes like circles and rectangles using hand gestures.

```python
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
color = (255, 255, 255)
shape_mode = None
start_point = None

def detect_shape_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    if thumb_tip.y < index_tip.y and middle_tip.y < index_tip.y:
        return "circle"
    elif thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y:
        return "rectangle"
    return None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            shape_mode = detect_shape_gesture(hand_landmarks)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_tip.x * 640), int(index_tip.y * 480)
            
            if shape_mode:
                if start_point is None:
                    start_point = (x, y)
                else:
                    if shape_mode == "circle":
                        radius = int(np.sqrt((x - start_point[0])**2 + (y - start_point[1])**2))
                        cv2.circle(canvas, start_point, radius, color, 2)
                    elif shape_mode == "rectangle":
                        cv2.rectangle(canvas, start_point, (x, y), color, 2)
                    start_point = None
                    shape_mode = None
    
    cv2.putText(image, f"Shape: {shape_mode if shape_mode else 'None'}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Air Canvas', cv2.add(image, canvas))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Slide 13: Real-life Example: Virtual Whiteboard for Remote Teaching

The Interactive Air Canvas can be used as a virtual whiteboard for remote teaching scenarios. Teachers can use hand gestures to write, draw diagrams, and highlight important points during online classes.

```python
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
color = (255, 255, 255)
brush_size = 5

def draw_on_canvas(image, hand_landmarks):
    global canvas, color, brush_size
    
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    x, y = int(index_tip.x * 640), int(index_tip.y * 480)
    
    cv2.circle(canvas, (x, y), brush_size, color, -1)
    
    # Add text for demonstration
    cv2.putText(canvas, "Math Lesson", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(canvas, "2 + 2 = 4", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_on_canvas(image, hand_landmarks)
    
    cv2.imshow('Virtual Whiteboard', cv2.add(image, canvas))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Slide 14: Real-life Example: Interactive Art Installation

The Interactive Air Canvas can be used to create an engaging art installation where visitors can collaboratively create digital artwork using hand gestures.

```python
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
colors = [(255, 0, 0), (0, 255, 0)]  # Blue for first hand, Green for second hand

def draw_artistic_pattern(hand_landmarks, color):
    global canvas
    
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start_point = hand_landmarks.landmark[start_idx]
        end_point = hand_landmarks.landmark[end_idx]
        
        x1, y1 = int(start_point.x * 640), int(start_point.y * 480)
        x2, y2 = int(end_point.x * 640), int(end_point.y * 480)
        
        cv2.line(canvas, (x1, y1), (x2, y2), color, 2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i < len(colors):
                draw_artistic_pattern(hand_landmarks, colors[i])
    
    # Apply some artistic effects
    canvas = cv2.GaussianBlur(canvas, (5, 5), 0)
    canvas = cv2.addWeighted(canvas, 0.9, np.zeros_like(canvas), 0.1, 0)
    
    cv2.imshow('Interactive Art Installation', cv2.add(image, canvas))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Slide 15: Additional Resources

For those interested in diving deeper into the technologies used in this Interactive Air Canvas project, here are some valuable resources:

1. MediaPipe documentation: [https://google.github.io/mediapipe/](https://google.github.io/mediapipe/)
2. OpenCV Python tutorials: [https://docs.opencv.org/master/d6/d00/tutorial\_py\_root.html](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
3. NumPy documentation: [https://numpy.org/doc/](https://numpy.org/doc/)

For academic papers related to gesture recognition and computer vision, you can explore:

1. "Survey on Hand Gesture Recognition in Computer Vision" by Rautaray and Agrawal (2015). ArXiv:1512.07603
2. "Vision-Based Hand Gesture Recognition for Human-Computer Interaction: A Survey" by Mitra and Acharya (2007). DOI: 10.1109/TSMCC.2007.893280

Remember to verify these sources and look for the most up-to-date information, as the field of computer vision and gesture recognition is rapidly evolving.

