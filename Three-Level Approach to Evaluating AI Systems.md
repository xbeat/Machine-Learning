## Three-Level Approach to Evaluating AI Systems
Slide 1: Three-Level Approach to AI Evaluation

AI evaluation is crucial for ensuring the quality and effectiveness of AI systems. This approach provides a structured method to assess AI performance at different stages of development and deployment. We'll explore each level in detail, along with code examples and best practices.

```python
# Visualization of the Three-Level Approach
import matplotlib.pyplot as plt

levels = ['Unit Tests', 'Model & Human Evaluation', 'A/B Testing']
complexity = [1, 2, 3]
frequency = [3, 2, 1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(levels, complexity, label='Complexity')
ax.bar(levels, frequency, bottom=complexity, label='Frequency')

ax.set_ylabel('Scale')
ax.set_title('Three-Level Approach to AI Evaluation')
ax.legend()

plt.show()
```

Slide 2: Level 1: Unit Tests

Unit tests are the foundation of AI evaluation. They are fast, inexpensive, and run with every code change. These tests focus on specific components or functions of the AI system to ensure they work as expected.

```python
import unittest
from ai_model import RealEstateLLM

class TestRealEstateLLM(unittest.TestCase):
    def setUp(self):
        self.llm = RealEstateLLM()

    def test_fetch_homes_under_2m(self):
        homes = self.llm.fetch_homes(max_price=2000000)
        for home in homes:
            self.assertLess(home.price, 2000000)

if __name__ == '__main__':
    unittest.main()
```

Slide 3: Implementing Continuous Integration for Unit Tests

To ensure unit tests are run consistently, we can set up a continuous integration (CI) pipeline using tools like GitHub Actions. This automates the testing process and provides immediate feedback on code changes.

```yaml
# .github/workflows/unit_tests.yml
name: Run Unit Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.x'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: python -m unittest discover tests
```

Slide 4: Level 2: Model & Human Evaluation

This level involves logging and evaluating AI's interaction with users. It combines automated metrics with human judgment to assess the quality of AI outputs. Tools like LangSmith can help track model behavior for adjustments.

```python
import langsmith

class AIEvaluator:
    def __init__(self):
        self.client = langsmith.Client()

    def log_interaction(self, user_input, ai_output, human_rating):
        run = self.client.create_run(
            name="User Interaction",
            inputs={"user_input": user_input},
            outputs={"ai_output": ai_output},
            extra={"human_rating": human_rating}
        )
        return run.id

    def get_average_rating(self):
        runs = self.client.list_runs(filter="name = 'User Interaction'")
        ratings = [run.extra["human_rating"] for run in runs]
        return sum(ratings) / len(ratings) if ratings else 0

evaluator = AIEvaluator()
run_id = evaluator.log_interaction("How's the weather?", "It's sunny today!", 4)
avg_rating = evaluator.get_average_rating()
print(f"Average rating: {avg_rating}")
```

Slide 5: Implementing a Human Review Interface

To facilitate human evaluation, we can create a simple web interface using Flask. This allows reviewers to easily rate AI outputs and provide feedback.

```python
from flask import Flask, render_template, request
from ai_model import AIModel
from evaluator import AIEvaluator

app = Flask(__name__)
ai_model = AIModel()
evaluator = AIEvaluator()

@app.route('/', methods=['GET', 'POST'])
def review_interface():
    if request.method == 'POST':
        user_input = request.form['user_input']
        ai_output = ai_model.generate_response(user_input)
        rating = int(request.form['rating'])
        evaluator.log_interaction(user_input, ai_output, rating)
        return render_template('thank_you.html')
    return render_template('review_form.html')

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 6: Level 3: A/B Testing

A/B testing is the most complex and costly level of AI evaluation, but it's crucial for measuring real user engagement. This method involves comparing two versions of the AI system to determine which performs better in real-world scenarios.

```python
import random
from sklearn.metrics import accuracy_score

class ABTester:
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
        self.results_a = []
        self.results_b = []

    def run_test(self, user_input):
        if random.random() < 0.5:
            output = self.model_a.generate_response(user_input)
            self.results_a.append((user_input, output))
            return output, 'A'
        else:
            output = self.model_b.generate_response(user_input)
            self.results_b.append((user_input, output))
            return output, 'B'

    def evaluate_results(self, ground_truth):
        accuracy_a = accuracy_score([gt for _, gt in ground_truth], 
                                    [self.model_a.generate_response(input) for input, _ in ground_truth])
        accuracy_b = accuracy_score([gt for _, gt in ground_truth], 
                                    [self.model_b.generate_response(input) for input, _ in ground_truth])
        return accuracy_a, accuracy_b

# Usage
tester = ABTester(ModelA(), ModelB())
for _ in range(1000):
    user_input = get_user_input()
    output, version = tester.run_test(user_input)
    present_to_user(output)

ground_truth = get_ground_truth()
accuracy_a, accuracy_b = tester.evaluate_results(ground_truth)
print(f"Model A accuracy: {accuracy_a}, Model B accuracy: {accuracy_b}")
```

Slide 7: Visualizing A/B Test Results

To make informed decisions based on A/B test results, it's important to visualize the data effectively. Here's an example of how to create a comparative bar chart using matplotlib.

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_ab_test(metric_a, metric_b, metric_name):
    labels = ['Model A', 'Model B']
    metrics = [metric_a, metric_b]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects = ax.bar(x, metrics, width)

    ax.set_ylabel(metric_name)
    ax.set_title(f'A/B Test Results: {metric_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.bar_label(rects, padding=3)

    fig.tight_layout()
    plt.show()

# Example usage
accuracy_a, accuracy_b = 0.85, 0.89
visualize_ab_test(accuracy_a, accuracy_b, 'Accuracy')
```

Slide 8: Lessons Learned: Fast Iterations

One key lesson in AI development is the importance of fast iterations. Rapid testing and refinement cycles lead to better AI performance and faster improvements.

```python
import time

class AIModel:
    def __init__(self):
        self.version = 1
        self.performance = 0.5

    def train(self):
        time.sleep(1)  # Simulate training time
        self.performance += 0.1
        self.version += 1

def rapid_iteration(model, iterations):
    start_time = time.time()
    for _ in range(iterations):
        model.train()
        print(f"Version {model.version}: Performance = {model.performance:.2f}")
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

model = AIModel()
rapid_iteration(model, 5)
```

Slide 9: Balancing Model Tweaking and Testing

Many teams focus too much on tweaking models without sufficient testing. It's crucial to strike a balance between model improvements and comprehensive testing.

```python
import random

class AIProject:
    def __init__(self):
        self.model_quality = 50
        self.test_coverage = 50

    def tweak_model(self):
        self.model_quality += random.randint(1, 5)
        self.test_coverage -= random.randint(1, 3)

    def improve_testing(self):
        self.test_coverage += random.randint(1, 5)
        self.model_quality += random.randint(0, 2)

    def overall_score(self):
        return (self.model_quality + self.test_coverage) / 2

project = AIProject()
for _ in range(10):
    if random.random() < 0.7:  # 70% chance to tweak model
        project.tweak_model()
    else:
        project.improve_testing()
    print(f"Model Quality: {project.model_quality}, Test Coverage: {project.test_coverage}")
    print(f"Overall Score: {project.overall_score()}")
```

Slide 10: Best Practices: Start Simple

When implementing AI evaluation, it's best to start with simple tools and gradually increase complexity. Here's an example of using GitHub Actions for basic test tracking.

```yaml
# .github/workflows/test_tracker.yml
name: Test Tracker

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  track_tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.x'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-cov
    - name: Run tests with coverage
      run: pytest --cov=./ --cov-report=xml
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v1
```

Slide 11: Incorporating Human Review

AI evaluation should always include human review, as AI alone can't replace human judgment. Here's a simple script to facilitate human review of AI outputs.

```python
import random

def ai_generate_response(prompt):
    responses = [
        "The weather is sunny today.",
        "I recommend trying the new Italian restaurant downtown.",
        "The capital of France is Paris.",
    ]
    return random.choice(responses)

def human_review():
    prompt = input("Enter a prompt for the AI: ")
    ai_response = ai_generate_response(prompt)
    print(f"AI Response: {ai_response}")
    
    rating = int(input("Rate the response (1-5): "))
    feedback = input("Provide any additional feedback: ")
    
    return {
        "prompt": prompt,
        "ai_response": ai_response,
        "rating": rating,
        "feedback": feedback
    }

# Collect multiple reviews
reviews = [human_review() for _ in range(3)]

# Analyze results
average_rating = sum(review["rating"] for review in reviews) / len(reviews)
print(f"Average rating: {average_rating:.2f}")
```

Slide 12: Data-Driven Decision Making

Let test results guide your decisions in AI development. Here's an example of how to use test data to make informed choices about model improvements.

```python
import numpy as np
import matplotlib.pyplot as plt

class ModelPerformance:
    def __init__(self, name):
        self.name = name
        self.accuracy_history = []

    def add_result(self, accuracy):
        self.accuracy_history.append(accuracy)

    def get_average_accuracy(self):
        return np.mean(self.accuracy_history)

def visualize_performance(models):
    for model in models:
        plt.plot(model.accuracy_history, label=model.name)
    
    plt.xlabel('Test Iteration')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Over Time')
    plt.legend()
    plt.show()

# Simulate performance data
model_a = ModelPerformance("Model A")
model_b = ModelPerformance("Model B")

for _ in range(10):
    model_a.add_result(np.random.normal(0.8, 0.05))
    model_b.add_result(np.random.normal(0.75, 0.05))

visualize_performance([model_a, model_b])

print(f"Model A average accuracy: {model_a.get_average_accuracy():.2f}")
print(f"Model B average accuracy: {model_b.get_average_accuracy():.2f}")
```

Slide 13: Real-Life Example: Image Classification Evaluation

Let's consider an image classification model for identifying different types of vehicles. We'll implement a simple evaluation pipeline using the three-level approach.

```python
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class VehicleClassifier:
    def predict(self, images):
        # Simulated predictions
        return np.random.choice(['car', 'truck', 'motorcycle', 'bicycle'], size=len(images))

# Level 1: Unit Test
def test_classifier_output(classifier, test_images):
    predictions = classifier.predict(test_images)
    assert all(pred in ['car', 'truck', 'motorcycle', 'bicycle'] for pred in predictions)

# Level 2: Model Evaluation
def evaluate_model(classifier, test_images, true_labels):
    predictions = classifier.predict(test_images)
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Level 3: A/B Testing (simplified)
def ab_test(classifier_a, classifier_b, test_images, true_labels):
    predictions_a = classifier_a.predict(test_images)
    predictions_b = classifier_b.predict(test_images)
    
    accuracy_a = accuracy_score(true_labels, predictions_a)
    accuracy_b = accuracy_score(true_labels, predictions_b)
    
    print(f"Classifier A Accuracy: {accuracy_a:.2f}")
    print(f"Classifier B Accuracy: {accuracy_b:.2f}")

# Simulate data and run evaluation
classifier = VehicleClassifier()
test_images = np.random.rand(100, 224, 224, 3)  # 100 random images
true_labels = np.random.choice(['car', 'truck', 'motorcycle', 'bicycle'], size=100)

test_classifier_output(classifier, test_images)
evaluate_model(classifier, test_images, true_labels)
ab_test(VehicleClassifier(), VehicleClassifier(), test_images, true_labels)
```

Slide 14: Real-Life Example: Natural Language Processing Evaluation

Consider a sentiment analysis model for customer reviews. We'll implement an evaluation pipeline using the three-level approach for a natural language processing task.

```python
import random
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class SentimentAnalyzer:
    def analyze(self, texts):
        # Simulate sentiment analysis
        return [random.choice(['positive', 'negative', 'neutral']) for _ in texts]

# Level 1: Unit Test
def test_sentiment_analyzer(analyzer):
    test_texts = ["Great product!", "Terrible service", "It was okay"]
    results = analyzer.analyze(test_texts)
    assert all(sentiment in ['positive', 'negative', 'neutral'] for sentiment in results)

# Level 2: Model Evaluation
def evaluate_model(analyzer, test_texts, true_sentiments):
    predictions = analyzer.analyze(test_texts)
    accuracy = accuracy_score(true_sentiments, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")
    print(classification_report(true_sentiments, predictions))

# Level 3: A/B Testing
def ab_test(analyzer_a, analyzer_b, test_texts, true_sentiments):
    predictions_a = analyzer_a.analyze(test_texts)
    predictions_b = analyzer_b.analyze(test_texts)
    
    accuracy_a = accuracy_score(true_sentiments, predictions_a)
    accuracy_b = accuracy_score(true_sentiments, predictions_b)
    
    plt.bar(['Analyzer A', 'Analyzer B'], [accuracy_a, accuracy_b])
    plt.title('A/B Test Results')
    plt.ylabel('Accuracy')
    plt.show()

# Run evaluation
analyzer = SentimentAnalyzer()
test_texts = ["I love this!", "Not good at all", "It's fine", "Amazing experience"]
true_sentiments = ['positive', 'negative', 'neutral', 'positive']

test_sentiment_analyzer(analyzer)
evaluate_model(analyzer, test_texts, true_sentiments)
ab_test(SentimentAnalyzer(), SentimentAnalyzer(), test_texts, true_sentiments)
```

Slide 15: Additional Resources

For those interested in diving deeper into AI evaluation techniques and best practices, here are some valuable resources:

1. "Evaluating Machine Learning Models" by Alice Zheng (O'Reilly)
2. "Human-in-the-Loop Machine Learning" by Robert Munro (Manning Publications)
3. "Practical Evaluation of Machine Learning Models" (arXiv:2101.00173)
4. "A Survey on Evaluation Methods for Machine Learning" (arXiv:2008.05281)

These resources provide comprehensive insights into various aspects of AI evaluation, from theoretical foundations to practical implementations.

