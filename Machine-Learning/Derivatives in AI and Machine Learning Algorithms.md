## Derivatives in AI and Machine Learning Algorithms
Slide 1: Derivatives in AI and Machine Learning

Derivatives are indeed fundamental to many AI and Machine Learning algorithms. This mathematical concept from calculus plays a crucial role in optimizing models and improving their performance. Let's explore how derivatives shape innovation across various industries.

```python
def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

# Example function
def f(x):
    return x**2

# Calculate derivative at x=3
x = 3
df_dx = derivative(f, x)
print(f"The derivative of f(x) = x^2 at x={x} is approximately {df_dx:.4f}")
```

Slide 2: Gradient Descent in Healthcare

In medical imaging, AI models use derivatives during training to minimize errors. Gradient descent, a method relying on derivatives, helps these models learn to detect anomalies like tumors with increasing precision, leading to earlier diagnoses and better patient outcomes.

```python
def gradient_descent(f, df, x0, learning_rate=0.01, num_iterations=100):
    x = x0
    for _ in range(num_iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
    return x

# Example: Minimizing a simple function
def f(x):
    return x**2 + 2*x + 1

def df(x):
    return 2*x + 2

minimum = gradient_descent(f, df, x0=0)
print(f"The minimum of f(x) = x^2 + 2x + 1 is at x ≈ {minimum:.4f}")
```

Slide 3: Backpropagation in Neural Networks

Backpropagation, a key algorithm in training neural networks, relies heavily on derivatives. It calculates the gradient of the loss function with respect to each weight by applying the chain rule from calculus.

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward_propagation(inputs, weights):
    hidden = sigmoid(sum(i * w for i, w in zip(inputs, weights[0])))
    output = sigmoid(sum([hidden * weights[1][0]]))
    return hidden, output

def backpropagation(inputs, target, weights, learning_rate):
    hidden, output = forward_propagation(inputs, weights)
    
    output_error = target - output
    output_delta = output_error * sigmoid_derivative(output)
    
    hidden_error = output_delta * weights[1][0]
    hidden_delta = hidden_error * sigmoid_derivative(hidden)
    
    weights[1][0] += learning_rate * output_delta * hidden
    for i in range(len(inputs)):
        weights[0][i] += learning_rate * hidden_delta * inputs[i]
    
    return weights

# Example usage
inputs = [0.5, 0.1]
target = 0.7
weights = [[0.2, 0.3], [0.4]]
learning_rate = 0.1

for _ in range(1000):
    weights = backpropagation(inputs, target, weights, learning_rate)

_, output = forward_propagation(inputs, weights)
print(f"Final output: {output:.4f}")
```

Slide 4: Convolutional Neural Networks in Image Recognition

Convolutional Neural Networks (CNNs) use derivatives in their training process to adjust filters for feature extraction. This is crucial in tasks like object recognition for autonomous vehicles.

```python
def convolution2d(image, kernel):
    m, n = len(image), len(image[0])
    k = len(kernel)
    result = [[0 for _ in range(n-k+1)] for _ in range(m-k+1)]
    
    for i in range(m-k+1):
        for j in range(n-k+1):
            sum = 0
            for di in range(k):
                for dj in range(k):
                    sum += image[i+di][j+dj] * kernel[di][dj]
            result[i][j] = sum
    
    return result

# Example usage
image = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

kernel = [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
]

result = convolution2d(image, kernel)
for row in result:
    print(row)
```

Slide 5: Predictive Maintenance in Manufacturing

Manufacturers use AI/ML models to predict equipment failures. By calculating derivatives of sensor data, these models can detect subtle changes in machine performance, enabling proactive maintenance and reducing downtime.

```python
import random

def generate_sensor_data(n, trend=0.1, noise=0.5):
    data = []
    value = 0
    for _ in range(n):
        value += trend + random.uniform(-noise, noise)
        data.append(value)
    return data

def calculate_rate_of_change(data):
    return [data[i+1] - data[i] for i in range(len(data)-1)]

# Generate sample sensor data
sensor_data = generate_sensor_data(100)

# Calculate rate of change
rate_of_change = calculate_rate_of_change(sensor_data)

# Detect anomalies (simplified)
threshold = 2 * sum(abs(r) for r in rate_of_change) / len(rate_of_change)
anomalies = [i for i, r in enumerate(rate_of_change) if abs(r) > threshold]

print(f"Detected anomalies at time steps: {anomalies}")
```

Slide 6: Real-time Decision Making in Autonomous Vehicles

Autonomous vehicles rely on real-time decision-making. Derivatives are used in training neural networks for tasks like object recognition and path planning, allowing self-driving cars to adjust to changing conditions instantly.

```python
import math

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def path_planning(start, goal, obstacles):
    path = [start]
    current = start
    while distance(current, goal) > 1:
        best_next = None
        best_distance = float('inf')
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                next_point = (current[0] + dx, current[1] + dy)
                if next_point not in obstacles:
                    dist = distance(next_point, goal)
                    if dist < best_distance:
                        best_distance = dist
                        best_next = next_point
        if best_next is None:
            return None  # No path found
        path.append(best_next)
        current = best_next
    return path

# Example usage
start = (0, 0)
goal = (5, 5)
obstacles = [(2, 2), (2, 3), (3, 2), (3, 3)]

path = path_planning(start, goal, obstacles)
print("Path found:", path)
```

Slide 7: Demand Forecasting in Retail

Retailers use AI/ML to forecast product demand. By analyzing historical sales data and computing derivatives, these models can detect trends and seasonal patterns, allowing businesses to optimize inventory levels and reduce waste.

```python
import math

def exponential_smoothing(data, alpha):
    result = [data[0]]
    for t in range(1, len(data)):
        result.append(alpha * data[t] + (1 - alpha) * result[t-1])
    return result

def forecast(smoothed_data, periods):
    return [smoothed_data[-1]] * periods

# Example usage
sales_data = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
alpha = 0.3
smoothed_data = exponential_smoothing(sales_data, alpha)
forecast_periods = 3
forecast_result = forecast(smoothed_data, forecast_periods)

print("Original data:", sales_data)
print("Smoothed data:", [round(x, 2) for x in smoothed_data])
print(f"Forecast for next {forecast_periods} periods:", forecast_result)
```

Slide 8: Precision Farming in Agriculture

Farmers leverage AI to increase crop yields. By applying derivatives to environmental data like soil moisture and temperature, AI models can recommend optimal planting and harvesting times, improving efficiency and sustainability.

```python
def calculate_growth_rate(temperature, moisture, nutrients):
    # Simplified growth rate model
    return 0.1 * temperature + 0.2 * moisture + 0.3 * nutrients

def optimize_planting_time(weather_forecast, soil_data):
    best_time = 0
    max_growth_rate = 0
    
    for day in range(len(weather_forecast)):
        temperature = weather_forecast[day]['temperature']
        moisture = weather_forecast[day]['moisture']
        nutrients = soil_data['nutrients']
        
        growth_rate = calculate_growth_rate(temperature, moisture, nutrients)
        
        if growth_rate > max_growth_rate:
            max_growth_rate = growth_rate
            best_time = day
    
    return best_time

# Example usage
weather_forecast = [
    {'temperature': 25, 'moisture': 0.6},
    {'temperature': 27, 'moisture': 0.5},
    {'temperature': 26, 'moisture': 0.7},
    {'temperature': 28, 'moisture': 0.6},
    {'temperature': 29, 'moisture': 0.5}
]
soil_data = {'nutrients': 0.8}

optimal_planting_day = optimize_planting_time(weather_forecast, soil_data)
print(f"Optimal planting day: Day {optimal_planting_day}")
```

Slide 9: Natural Language Processing

Derivatives play a crucial role in training language models for tasks like sentiment analysis and machine translation. Let's look at a simple example of sentiment analysis using a basic neural network.

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def train_sentiment_model(texts, labels, epochs=1000, learning_rate=0.1):
    vocab = set(word for text in texts for word in text.split())
    word_to_index = {word: i for i, word in enumerate(vocab)}
    weights = [0.0] * len(vocab)
    
    for _ in range(epochs):
        for text, label in zip(texts, labels):
            prediction = sigmoid(sum(weights[word_to_index[word]] for word in text.split() if word in word_to_index))
            error = label - prediction
            for word in text.split():
                if word in word_to_index:
                    weights[word_to_index[word]] += learning_rate * error * prediction * (1 - prediction)
    
    return weights, word_to_index

def predict_sentiment(text, weights, word_to_index):
    score = sum(weights[word_to_index[word]] for word in text.split() if word in word_to_index)
    return sigmoid(score)

# Example usage
texts = ["I love this movie", "This film is terrible", "Great acting and plot"]
labels = [1, 0, 1]  # 1 for positive, 0 for negative

weights, word_to_index = train_sentiment_model(texts, labels)

test_text = "Amazing performance by the actors"
sentiment = predict_sentiment(test_text, weights, word_to_index)
print(f"Sentiment score for '{test_text}': {sentiment:.2f}")
```

Slide 10: Computer Vision in Medical Imaging

Derivatives are essential in training models for medical image analysis. Here's a simple example of edge detection, a fundamental operation in image processing, using a Sobel operator.

```python
def sobel_edge_detection(image):
    height, width = len(image), len(image[0])
    
    def convolve(kernel):
        result = [[0 for _ in range(width)] for _ in range(height)]
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                value = sum(kernel[di+1][dj+1] * image[i+di][j+dj]
                            for di in [-1, 0, 1] for dj in [-1, 0, 1])
                result[i][j] = max(0, min(255, value))
        return result
    
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    
    gx = convolve(sobel_x)
    gy = convolve(sobel_y)
    
    edges = [[int((gx[i][j]**2 + gy[i][j]**2)**0.5) for j in range(width)]
             for i in range(height)]
    
    return edges

# Example usage (simplified grayscale image)
image = [
    [50, 50, 50, 50, 50],
    [50, 100, 100, 100, 50],
    [50, 100, 200, 100, 50],
    [50, 100, 100, 100, 50],
    [50, 50, 50, 50, 50]
]

edges = sobel_edge_detection(image)
for row in edges:
    print(' '.join(f"{pixel:3d}" for pixel in row))
```

Slide 11: Reinforcement Learning in Robotics

Derivatives are crucial in reinforcement learning algorithms used to train robots. Here's a simplified Q-learning example for a robot navigating a grid world.

```python
import random

def q_learning(grid, episodes=1000, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
    rows, cols = len(grid), len(grid[0])
    q_table = {(i, j): [0, 0, 0, 0] for i in range(rows) for j in range(cols)}
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    
    for _ in range(episodes):
        state = (0, 0)
        while state != (rows-1, cols-1):
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = q_table[state].index(max(q_table[state]))
            
            next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
            if 0 <= next_state[0] < rows and 0 <= next_state[1] < cols:
                reward = grid[next_state[0]][next_state[1]]
                old_value = q_table[state][action]
                next_max = max(q_table[next_state])
                
                new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
                q_table[state][action] = new_value
                state = next_state
    
    return q_table

# Example usage
grid = [
    [0, 0, 0, 1],
    [0, -1, 0, -1],
    [0, 0, 0, 2]
]

q_table = q_learning(grid)
print("Q-learning completed. Q-table generated for the grid world.")
```

Slide 12: Optimization in Supply Chain Management

Derivatives play a key role in optimizing supply chain operations. Here's a simple example of using gradient descent to minimize transportation costs.

```python
def transportation_cost(x, distances, demands):
    return sum(d * x[i] for i, d in enumerate(distances)) + sum(abs(sum(x) - demand) for demand in demands)

def gradient_descent(cost_func, initial_x, learning_rate=0.01, num_iterations=1000):
    x = initial_x.copy()
    for _ in range(num_iterations):
        gradient = [0] * len(x)
        for i in range(len(x)):
            h = 1e-5
            x_plus_h = x.copy()
            x_plus_h[i] += h
            gradient[i] = (cost_func(x_plus_h) - cost_func(x)) / h
        x = [xi - learning_rate * gi for xi, gi in zip(x, gradient)]
    return x

# Example usage
distances = [10, 20, 30]  # distances from warehouses to destination
demands = [100, 150]  # demands at two time periods
initial_allocation = [50, 50, 50]  # initial guess for allocation from each warehouse

optimal_allocation = gradient_descent(lambda x: transportation_cost(x, distances, demands), initial_allocation)
print("Optimal allocation:", [round(x, 2) for x in optimal_allocation])
print("Minimum cost:", round(transportation_cost(optimal_allocation, distances, demands), 2))
```

Slide 13: Natural Language Processing: Word Embeddings

Word embeddings, crucial for many NLP tasks, rely on derivatives during training. Here's a simplified example of creating word vectors using a basic skip-gram model.

```python
import random
import math

def create_corpus():
    return "the quick brown fox jumps over the lazy dog".split()

def create_context_target_pairs(corpus, window_size=2):
    pairs = []
    for i, word in enumerate(corpus):
        for j in range(max(0, i - window_size), min(len(corpus), i + window_size + 1)):
            if i != j:
                pairs.append((word, corpus[j]))
    return pairs

def initialize_vectors(vocab, vector_size):
    return {word: [random.uniform(-1, 1) for _ in range(vector_size)] for word in vocab}

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def train_skip_gram(corpus, vector_size=10, learning_rate=0.01, epochs=1000):
    vocab = set(corpus)
    word_vectors = initialize_vectors(vocab, vector_size)
    context_vectors = initialize_vectors(vocab, vector_size)
    pairs = create_context_target_pairs(corpus)

    for _ in range(epochs):
        for target, context in pairs:
            z = sum(w * c for w, c in zip(word_vectors[target], context_vectors[context]))
            y_pred = sigmoid(z)
            
            error = 1 - y_pred
            
            for i in range(vector_size):
                word_vectors[target][i] += learning_rate * error * context_vectors[context][i]
                context_vectors[context][i] += learning_rate * error * word_vectors[target][i]

    return word_vectors

# Example usage
corpus = create_corpus()
word_vectors = train_skip_gram(corpus)
print("Word vectors created for the corpus.")
```

Slide 14: Generative Adversarial Networks (GANs)

GANs, used in various applications including image generation, heavily rely on derivatives for training both the generator and discriminator networks. Here's a simplified pseudocode representation of a GAN training process:

```python
# Pseudocode for GAN training
def train_gan(real_data, num_epochs):
    generator = initialize_generator()
    discriminator = initialize_discriminator()
    
    for epoch in range(num_epochs):
        # Train Discriminator
        for _ in range(discriminator_steps):
            real_samples = sample_real_data(real_data)
            fake_samples = generator.generate(noise())
            
            real_loss = discriminator.train(real_samples, labels=1)
            fake_loss = discriminator.train(fake_samples, labels=0)
            
            discriminator_loss = real_loss + fake_loss
        
        # Train Generator
        for _ in range(generator_steps):
            noise = generate_noise()
            fake_samples = generator.generate(noise)
            
            generator_loss = generator.train(fake_samples, discriminator)
        
        if epoch % log_interval == 0:
            print(f"Epoch {epoch}: Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}")

# Example usage (not runnable)
real_data = load_real_data()
train_gan(real_data, num_epochs=1000)
```

Slide 15: Additional Resources

For those interested in diving deeper into the mathematics behind AI and machine learning algorithms, here are some valuable resources:

1.  ArXiv.org: A repository of electronic preprints of scientific papers in various fields, including AI and ML. URL: [https://arxiv.org/list/cs.AI/recent](https://arxiv.org/list/cs.AI/recent)
2.  "Calculus on Computational Graphs: Backpropagation" by Christopher Olah ArXiv ID: 1502.05767
3.  "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville Available online at: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
4.  "Neural Networks and Deep Learning" by Michael Nielsen Available online at: [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)

These resources provide in-depth explanations of the mathematical concepts, including derivatives, that underpin modern AI and ML algorithms.

