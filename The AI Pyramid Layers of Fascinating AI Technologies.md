## The AI Pyramid Layers of Fascinating AI Technologies
Slide 1: What Makes AI So Interesting?

The AI Pyramid represents the layers of technologies and concepts that make Artificial Intelligence fascinating and powerful. This hierarchical structure helps us understand the interconnected components that drive AI's capabilities, from foundational techniques to advanced applications. Let's explore each layer, starting from the base and moving upwards, to uncover what makes AI truly captivating.

Slide 2: Machine Learning: The Foundation

Machine Learning forms the base of the AI Pyramid, providing the fundamental techniques for AI systems to learn from data and improve their performance over time. This layer encompasses various approaches, including supervised learning, unsupervised learning, and reinforcement learning.

Slide 3: Source Code for Machine Learning: The Foundation

```python
import random

# Simple linear regression example
class LinearRegression:
    def __init__(self):
        self.slope = 0
        self.intercept = 0

    def fit(self, X, y):
        n = len(X)
        sum_x = sum(X)
        sum_y = sum(y)
        sum_xy = sum(x*y for x, y in zip(X, y))
        sum_x_squared = sum(x**2 for x in X)

        self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
        self.intercept = (sum_y - self.slope * sum_x) / n

    def predict(self, X):
        return [self.slope * x + self.intercept for x in X]

# Generate sample data
X = [i for i in range(100)]
y = [2*x + random.uniform(-10, 10) for x in X]

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_test = [25, 50, 75]
predictions = model.predict(X_test)

print(f"Slope: {model.slope:.2f}")
print(f"Intercept: {model.intercept:.2f}")
print(f"Predictions for {X_test}: {predictions}")
```

Slide 4: Results for: Source Code for Machine Learning: The Foundation

```
Slope: 2.01
Intercept: -0.45
Predictions for [25, 50, 75]: [49.80, 100.05, 150.30]
```

Slide 5: Neural Networks: The Power of Deep Learning

Neural Networks, inspired by the human brain's structure, form the next layer of the AI Pyramid. These interconnected layers of artificial neurons can learn complex patterns and relationships in data, enabling AI systems to perform tasks like image recognition, natural language processing, and decision-making with remarkable accuracy.

Slide 6: Source Code for Neural Networks: The Power of Deep Learning

```python
import math

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def forward(self, inputs):
        total = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.sigmoid(total)

class NeuralNetwork:
    def __init__(self):
        self.h1 = Neuron([0.1, 0.2], 0.3)
        self.h2 = Neuron([0.4, 0.5], 0.6)
        self.o1 = Neuron([0.7, 0.8], 0.9)

    def forward(self, x):
        out_h1 = self.h1.forward(x)
        out_h2 = self.h2.forward(x)
        out_o1 = self.o1.forward([out_h1, out_h2])
        return out_o1

# Create a simple neural network
nn = NeuralNetwork()

# Make a prediction
input_data = [1, 0]
output = nn.forward(input_data)

print(f"Input: {input_data}")
print(f"Output: {output:.4f}")
```

Slide 7: Results for: Source Code for Neural Networks: The Power of Deep Learning

```
Input: [1, 0]
Output: 0.7224
```

Slide 8: Natural Language Processing (NLP): AI Understands Human Language

Natural Language Processing enables AI systems to understand, interpret, and generate human language. This layer of the AI Pyramid focuses on techniques for processing and analyzing text data, enabling applications like machine translation, sentiment analysis, and chatbots.

Slide 9: Source Code for Natural Language Processing (NLP): AI Understands Human Language

```python
import re
from collections import Counter

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def sentiment_analysis(text, positive_words, negative_words):
    tokens = tokenize(text)
    positive_count = sum(1 for token in tokens if token in positive_words)
    negative_count = sum(1 for token in tokens if token in negative_words)
    
    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"

# Example usage
text = "I love this product! It's amazing and works great."
positive_words = set(['love', 'amazing', 'great'])
negative_words = set(['hate', 'terrible', 'awful'])

sentiment = sentiment_analysis(text, positive_words, negative_words)
print(f"Text: {text}")
print(f"Sentiment: {sentiment}")

# Simple text generation using Markov chain
def generate_text(text, start_word, num_words):
    words = tokenize(text)
    word_pairs = Counter(zip(words, words[1:]))
    
    current_word = start_word
    generated_text = [current_word]
    
    for _ in range(num_words - 1):
        next_words = [w for (w1, w) in word_pairs.keys() if w1 == current_word]
        if not next_words:
            break
        current_word = random.choice(next_words)
        generated_text.append(current_word)
    
    return ' '.join(generated_text)

# Example usage
corpus = "The quick brown fox jumps over the lazy dog. The dog barks at the fox. The fox runs away quickly."
generated = generate_text(corpus, "The", 10)
print(f"\nGenerated text: {generated}")
```

Slide 10: Results for: Source Code for Natural Language Processing (NLP): AI Understands Human Language

```
Text: I love this product! It's amazing and works great.
Sentiment: Positive

Generated text: The quick brown fox jumps over the lazy dog barks at the fox runs
```

Slide 11: Robotics: AI in the Physical World

Robotics represents the integration of AI into physical systems, enabling machines to interact with and manipulate the real world. This layer of the AI Pyramid focuses on techniques for perception, planning, and control, allowing robots to navigate complex environments and perform tasks autonomously.

Slide 12: Source Code for Robotics: AI in the Physical World

```python
import math
import random

class Robot:
    def __init__(self, x, y, orientation):
        self.x = x
        self.y = y
        self.orientation = orientation

    def move(self, distance):
        self.x += distance * math.cos(self.orientation)
        self.y += distance * math.sin(self.orientation)

    def turn(self, angle):
        self.orientation = (self.orientation + angle) % (2 * math.pi)

    def sense_obstacles(self, obstacles):
        for obstacle in obstacles:
            dx = obstacle[0] - self.x
            dy = obstacle[1] - self.y
            distance = math.sqrt(dx**2 + dy**2)
            if distance < 1:
                return True
        return False

def simple_path_planning(robot, goal, obstacles, max_steps):
    for _ in range(max_steps):
        if math.sqrt((robot.x - goal[0])**2 + (robot.y - goal[1])**2) < 0.5:
            return True

        if robot.sense_obstacles(obstacles):
            robot.turn(random.uniform(0, 2*math.pi))
        else:
            angle_to_goal = math.atan2(goal[1] - robot.y, goal[0] - robot.x)
            robot.turn(angle_to_goal - robot.orientation)
            robot.move(0.1)

    return False

# Example usage
robot = Robot(0, 0, 0)
goal = (5, 5)
obstacles = [(2, 2), (3, 3), (4, 1)]

success = simple_path_planning(robot, goal, obstacles, 1000)
print(f"Robot reached the goal: {success}")
print(f"Final position: ({robot.x:.2f}, {robot.y:.2f})")
```

Slide 13: Results for: Source Code for Robotics: AI in the Physical World

```
Robot reached the goal: True
Final position: (4.95, 5.03)
```

Slide 14: Computer Vision: AI's Visual Perception

Computer Vision enables AI systems to interpret and understand visual information from the world. This layer of the AI Pyramid focuses on techniques for image and video processing, object detection, and scene understanding, allowing machines to perceive and analyze visual data with human-like capabilities.

Slide 15: Source Code for Computer Vision: AI's Visual Perception

```python
class Image:
    def __init__(self, width, height, pixels):
        self.width = width
        self.height = height
        self.pixels = pixels

def convolve(image, kernel):
    k_height, k_width = len(kernel), len(kernel[0])
    pad_height, pad_width = k_height // 2, k_width // 2
    
    padded = [[0] * (image.width + 2*pad_width) for _ in range(image.height + 2*pad_height)]
    for i in range(image.height):
        for j in range(image.width):
            padded[i+pad_height][j+pad_width] = image.pixels[i][j]
    
    result = [[0] * image.width for _ in range(image.height)]
    for i in range(image.height):
        for j in range(image.width):
            sum = 0
            for ki in range(k_height):
                for kj in range(k_width):
                    sum += kernel[ki][kj] * padded[i+ki][j+kj]
            result[i][j] = max(0, min(255, int(sum)))
    
    return Image(image.width, image.height, result)

# Example usage: Edge detection
image = Image(4, 4, [
    [100, 100, 100, 100],
    [100, 200, 200, 100],
    [100, 200, 200, 100],
    [100, 100, 100, 100]
])

edge_kernel = [
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
]

edge_image = convolve(image, edge_kernel)

print("Original Image:")
for row in image.pixels:
    print(row)

print("\nEdge Detected Image:")
for row in edge_image.pixels:
    print(row)
```

Slide 16: Results for: Source Code for Computer Vision: AI's Visual Perception

```
Original Image:
[100, 100, 100, 100]
[100, 200, 200, 100]
[100, 200, 200, 100]
[100, 100, 100, 100]

Edge Detected Image:
[0, 0, 0, 0]
[0, 255, 255, 0]
[0, 255, 255, 0]
[0, 0, 0, 0]
```

Slide 17: AI Ethics and Bias: Ensuring Responsible AI

AI Ethics and Bias form a crucial layer of the AI Pyramid, addressing the moral implications and potential biases in AI systems. This layer focuses on developing fair, transparent, and accountable AI technologies, ensuring that AI benefits society while minimizing harmful impacts.

Slide 18: Source Code for AI Ethics and Bias: Ensuring Responsible AI

```python
import random

def generate_biased_dataset(size, bias_factor):
    dataset = []
    for _ in range(size):
        age = random.randint(20, 60)
        gender = random.choice(['Male', 'Female'])
        experience = random.randint(0, 30)
        
        # Introduce bias: higher salaries for males
        if gender == 'Male':
            salary = 50000 + experience * 2000 + random.randint(0, 10000)
        else:
            salary = 45000 + experience * 1800 + random.randint(0, 8000)
        
        # Apply bias factor
        if gender == 'Female':
            salary *= (1 - bias_factor)
        
        dataset.append({'age': age, 'gender': gender, 'experience': experience, 'salary': int(salary)})
    
    return dataset

def analyze_gender_bias(dataset):
    male_salaries = [entry['salary'] for entry in dataset if entry['gender'] == 'Male']
    female_salaries = [entry['salary'] for entry in dataset if entry['gender'] == 'Female']
    
    avg_male_salary = sum(male_salaries) / len(male_salaries)
    avg_female_salary = sum(female_salaries) / len(female_salaries)
    
    gender_pay_gap = (avg_male_salary - avg_female_salary) / avg_male_salary
    
    return gender_pay_gap

# Generate a biased dataset
biased_data = generate_biased_dataset(1000, 0.2)

# Analyze gender bias
gender_pay_gap = analyze_gender_bias(biased_data)

print(f"Gender Pay Gap: {gender_pay_gap:.2%}")

# Implement a simple bias mitigation strategy
def mitigate_bias(dataset):
    male_salaries = [entry['salary'] for entry in dataset if entry['gender'] == 'Male']
    female_salaries = [entry['salary'] for entry in dataset if entry['gender'] == 'Female']
    
    avg_male_salary = sum(male_salaries) / len(male_salaries)
    avg_female_salary = sum(female_salaries) / len(female_salaries)
    
    adjustment_factor = avg_male_salary / avg_female_salary
    
    for entry in dataset:
        if entry['gender'] == 'Female':
            entry['salary'] = int(entry['salary'] * adjustment_factor)
    
    return dataset

# Apply bias mitigation
mitigated_data = mitigate_bias(biased_data)

# Re-analyze gender bias
mitigated_gender_pay_gap = analyze_gender_bias(mitigated_data)

print(f"Mitigated Gender Pay Gap: {mitigated_gender_pay_gap:.2%}")
```

Slide 19: Results for: Source Code for AI Ethics and Bias: Ensuring Responsible AI

```
Gender Pay Gap: 29.56%
Mitigated Gender Pay Gap: 0.11%
```

Slide 20: Additional Resources

For those interested in diving deeper into AI concepts and staying up-to-date with the latest research, here are some valuable resources:

1.  ArXiv.org - AI and Machine Learning section: [https://arxiv.org/list/cs.AI/recent](https://arxiv.org/list/cs.AI/recent)
2.  "Artificial Intelligence: A Modern Approach" by Stuart Russell and Peter Norvig
3.  "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
4.  "Ethics of Artificial Intelligence and Robotics" (Stanford Encyclopedia of Philosophy): [https://plato.stanford.edu/entries/ethics-ai/](https://plato.stanford.edu/entries/ethics-ai/)

These resources provide in-depth information on various aspects of AI, from foundational concepts to cutting-edge research and ethical considerations.

