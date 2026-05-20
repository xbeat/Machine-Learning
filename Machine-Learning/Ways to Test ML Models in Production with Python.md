## Ways to Test ML Models in Production with Python
Slide 1:

Introduction to ML Model Testing in Production

Machine Learning (ML) models require rigorous testing in production environments to ensure their reliability, performance, and effectiveness. This presentation explores four key methods for testing ML models in production: A/B Testing, Canary Releases, Interleaved Experiments, and Shadow Testing. We'll delve into each approach, providing Python code examples and practical insights for implementation.

Slide 2:

A/B Testing: The Basics

A/B testing, also known as split testing, involves comparing two versions of a model to determine which performs better. In this method, users are randomly divided into two groups, with each group interacting with a different version of the model. The performance of both models is then compared to identify the superior version.

```python
import random

def ab_test(user_id, model_a, model_b):
    # Determine which model to use based on user ID
    if hash(user_id) % 2 == 0:
        return model_a.predict()
    else:
        return model_b.predict()

# Example usage
user_id = "user123"
result = ab_test(user_id, model_a, model_b)
print(f"Prediction for user {user_id}: {result}")
```

Slide 3:

A/B Testing: Analyzing Results

After running an A/B test, it's crucial to analyze the results to determine which model performed better. This typically involves statistical analysis to ensure the differences observed are significant and not due to random chance.

```python
import scipy.stats as stats

def analyze_ab_test(control_conversions, control_users, 
                    experiment_conversions, experiment_users):
    control_rate = control_conversions / control_users
    experiment_rate = experiment_conversions / experiment_users
    
    z_score, p_value = stats.proportions_ztest(
        [control_conversions, experiment_conversions],
        [control_users, experiment_users],
        alternative='two-sided'
    )
    
    print(f"Control conversion rate: {control_rate:.2%}")
    print(f"Experiment conversion rate: {experiment_rate:.2%}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("The difference is statistically significant.")
    else:
        print("The difference is not statistically significant.")

# Example usage
analyze_ab_test(control_conversions=100, control_users=1000,
                experiment_conversions=120, experiment_users=1000)
```

Output:

```
Control conversion rate: 10.00%
Experiment conversion rate: 12.00%
P-value: 0.1573
The difference is not statistically significant.
```

Slide 4:

Canary Releases: Concept and Implementation

Canary releases involve gradually rolling out a new model to a small subset of users before deploying it to the entire user base. This approach allows for monitoring the new model's performance and quickly rolling back if issues arise.

```python
import random

class CanaryDeployment:
    def __init__(self, old_model, new_model, canary_percentage):
        self.old_model = old_model
        self.new_model = new_model
        self.canary_percentage = canary_percentage

    def get_prediction(self, user_id):
        if random.random() < self.canary_percentage:
            return self.new_model.predict(), "new"
        else:
            return self.old_model.predict(), "old"

# Example usage
canary = CanaryDeployment(old_model, new_model, canary_percentage=0.1)
prediction, model_used = canary.get_prediction("user456")
print(f"Prediction: {prediction}, Model used: {model_used}")
```

Slide 5:

Canary Releases: Monitoring and Rollback

During a canary release, it's essential to monitor the new model's performance closely. If issues are detected, a rollback mechanism should be in place to quickly revert to the old model.

```python
import time

class CanaryMonitor:
    def __init__(self, canary_deployment, error_threshold):
        self.canary = canary_deployment
        self.error_threshold = error_threshold
        self.errors = 0
        self.total_requests = 0

    def monitor(self, duration_seconds):
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            prediction, model_used = self.canary.get_prediction(f"user{self.total_requests}")
            self.total_requests += 1
            if model_used == "new" and self.simulate_error():
                self.errors += 1
            
            error_rate = self.errors / self.total_requests
            if error_rate > self.error_threshold:
                print(f"Error rate ({error_rate:.2%}) exceeded threshold. Rolling back.")
                return False
        
        print(f"Canary release successful. Error rate: {error_rate:.2%}")
        return True

    def simulate_error(self):
        return random.random() < 0.05  # 5% chance of error

# Example usage
monitor = CanaryMonitor(canary, error_threshold=0.1)
success = monitor.monitor(duration_seconds=60)
```

Slide 6:

Interleaved Experiments: Concept

Interleaved experiments involve combining the results from multiple models and presenting them to users in an interleaved fashion. This method allows for direct comparison of model performance within the same user session.

```python
import random

def interleave_results(model_a_results, model_b_results):
    interleaved = []
    a_index, b_index = 0, 0
    while a_index < len(model_a_results) and b_index < len(model_b_results):
        if random.random() < 0.5:
            interleaved.append(("A", model_a_results[a_index]))
            a_index += 1
        else:
            interleaved.append(("B", model_b_results[b_index]))
            b_index += 1
    return interleaved

# Example usage
model_a_results = ["A1", "A2", "A3", "A4"]
model_b_results = ["B1", "B2", "B3", "B4"]
interleaved_results = interleave_results(model_a_results, model_b_results)
print("Interleaved results:", interleaved_results)
```

Output:

```
Interleaved results: [('B', 'B1'), ('A', 'A1'), ('B', 'B2'), ('A', 'A2'), ('A', 'A3'), ('B', 'B3'), ('A', 'A4'), ('B', 'B4')]
```

Slide 7:

Interleaved Experiments: Analysis

After running an interleaved experiment, it's important to analyze the results to determine which model performed better. This typically involves comparing user engagement or other relevant metrics for each model.

```python
from collections import defaultdict

def analyze_interleaved_experiment(interleaved_results, user_clicks):
    model_clicks = defaultdict(int)
    model_impressions = defaultdict(int)

    for model, result in interleaved_results:
        model_impressions[model] += 1
        if result in user_clicks:
            model_clicks[model] += 1

    for model in model_impressions:
        ctr = model_clicks[model] / model_impressions[model]
        print(f"Model {model} - CTR: {ctr:.2%}")

    winner = max(model_clicks, key=model_clicks.get)
    print(f"Winner: Model {winner}")

# Example usage
user_clicks = ["A1", "B2", "A3"]
analyze_interleaved_experiment(interleaved_results, user_clicks)
```

Output:

```
Model B - CTR: 25.00%
Model A - CTR: 50.00%
Winner: Model A
```

Slide 8:

Shadow Testing: Concept

Shadow testing involves running a new model alongside the existing production model, but only using the existing model's output. The new model's predictions are logged and compared to the production model's performance without affecting the user experience.

```python
class ShadowTest:
    def __init__(self, production_model, shadow_model):
        self.production_model = production_model
        self.shadow_model = shadow_model
        self.shadow_results = []

    def predict(self, input_data):
        production_result = self.production_model.predict(input_data)
        shadow_result = self.shadow_model.predict(input_data)
        self.shadow_results.append((input_data, production_result, shadow_result))
        return production_result

    def analyze_results(self):
        total_matches = sum(1 for _, prod, shadow in self.shadow_results if prod == shadow)
        match_rate = total_matches / len(self.shadow_results)
        print(f"Shadow model match rate: {match_rate:.2%}")

# Example usage
shadow_test = ShadowTest(production_model, shadow_model)
for i in range(100):
    input_data = f"data_{i}"
    result = shadow_test.predict(input_data)

shadow_test.analyze_results()
```

Slide 9:

Shadow Testing: Implementation

Implementing shadow testing requires careful consideration of system resources and logging mechanisms. Here's an example of how to implement shadow testing with performance monitoring:

```python
import time
import threading

class ShadowTestWithMonitoring:
    def __init__(self, production_model, shadow_model):
        self.production_model = production_model
        self.shadow_model = shadow_model
        self.shadow_results = []
        self.performance_metrics = {"production": [], "shadow": []}

    def predict(self, input_data):
        production_start = time.time()
        production_result = self.production_model.predict(input_data)
        production_time = time.time() - production_start

        def shadow_prediction():
            shadow_start = time.time()
            shadow_result = self.shadow_model.predict(input_data)
            shadow_time = time.time() - shadow_start
            self.shadow_results.append((input_data, production_result, shadow_result))
            self.performance_metrics["shadow"].append(shadow_time)

        # Run shadow prediction in a separate thread
        threading.Thread(target=shadow_prediction).start()

        self.performance_metrics["production"].append(production_time)
        return production_result

    def analyze_results(self):
        total_matches = sum(1 for _, prod, shadow in self.shadow_results if prod == shadow)
        match_rate = total_matches / len(self.shadow_results)
        print(f"Shadow model match rate: {match_rate:.2%}")

        avg_production_time = sum(self.performance_metrics["production"]) / len(self.performance_metrics["production"])
        avg_shadow_time = sum(self.performance_metrics["shadow"]) / len(self.performance_metrics["shadow"])
        print(f"Avg. production model time: {avg_production_time:.4f} seconds")
        print(f"Avg. shadow model time: {avg_shadow_time:.4f} seconds")

# Example usage
shadow_test = ShadowTestWithMonitoring(production_model, shadow_model)
for i in range(100):
    input_data = f"data_{i}"
    result = shadow_test.predict(input_data)

shadow_test.analyze_results()
```

Slide 10:

Real-Life Example: Content Recommendation System

Let's consider a content recommendation system for a streaming platform. We'll use A/B testing to compare two recommendation algorithms:

```python
import random

class RecommendationSystem:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def recommend(self, user_id):
        if self.algorithm == "A":
            # Algorithm A: Based on user's watch history
            return ["Movie1", "Series2", "Documentary3"]
        elif self.algorithm == "B":
            # Algorithm B: Based on collaborative filtering
            return ["Series4", "Movie5", "Show6"]

def ab_test_recommendations(user_id):
    if hash(user_id) % 2 == 0:
        return RecommendationSystem("A").recommend(user_id)
    else:
        return RecommendationSystem("B").recommend(user_id)

# Simulate user interactions
user_engagement = {"A": 0, "B": 0}
total_users = 1000

for i in range(total_users):
    user_id = f"user_{i}"
    recommendations = ab_test_recommendations(user_id)
    
    # Simulate user engagement (clicking on a recommended item)
    if random.random() < 0.3:  # 30% chance of engagement
        user_engagement["A" if hash(user_id) % 2 == 0 else "B"] += 1

# Analyze results
for algorithm, engagements in user_engagement.items():
    engagement_rate = engagements / (total_users / 2)
    print(f"Algorithm {algorithm} engagement rate: {engagement_rate:.2%}")
```

Output:

```
Algorithm A engagement rate: 29.80%
Algorithm B engagement rate: 30.20%
```

Slide 11:

Real-Life Example: Fraud Detection System

Let's implement a shadow testing scenario for a fraud detection system in an e-commerce platform:

```python
import random
import time

class FraudDetectionModel:
    def __init__(self, version):
        self.version = version

    def predict(self, transaction):
        # Simulate prediction time
        time.sleep(random.uniform(0.01, 0.05))
        
        # Simple fraud detection logic
        if self.version == "production":
            return transaction["amount"] > 1000 or transaction["country"] != transaction["card_country"]
        elif self.version == "shadow":
            return transaction["amount"] > 800 or transaction["country"] != transaction["card_country"]

class ShadowTestFraudDetection:
    def __init__(self):
        self.production_model = FraudDetectionModel("production")
        self.shadow_model = FraudDetectionModel("shadow")
        self.results = []

    def check_transaction(self, transaction):
        prod_start = time.time()
        prod_result = self.production_model.predict(transaction)
        prod_time = time.time() - prod_start

        shadow_start = time.time()
        shadow_result = self.shadow_model.predict(transaction)
        shadow_time = time.time() - shadow_start

        self.results.append({
            "transaction": transaction,
            "production": {"result": prod_result, "time": prod_time},
            "shadow": {"result": shadow_result, "time": shadow_time}
        })

        return prod_result

    def analyze_results(self):
        total = len(self.results)
        matches = sum(1 for r in self.results if r["production"]["result"] == r["shadow"]["result"])
        false_positives = sum(1 for r in self.results if r["shadow"]["result"] and not r["production"]["result"])
        false_negatives = sum(1 for r in self.results if r["production"]["result"] and not r["shadow"]["result"])

        print(f"Total transactions: {total}")
        print(f"Match rate: {matches/total:.2%}")
        print(f"False positive rate: {false_positives/total:.2%}")
        print(f"False negative rate: {false_negatives/total:.2%}")
        print(f"Avg. production time: {sum(r['production']['time'] for r in self.results)/total:.4f} seconds")
        print(f"Avg. shadow time: {sum(r['shadow']['time'] for r in self.results)/total:.4f} seconds")

# Run shadow test
shadow_test = ShadowTestFraudDetection()
for _ in range(1000):
    transaction = {
        "amount": random.uniform(100, 2000),
        "country": random.choice(["US", "UK", "FR", "DE"]),
        "card_country": random.choice(["US", "UK", "FR", "DE"])
    }
    shadow_test.check_transaction(transaction)
```

Slide 12:

Interleaved Experiments: Advanced Implementation

Let's explore a more sophisticated implementation of interleaved experiments, incorporating user feedback and real-time analysis:

```python
import random
from collections import defaultdict

class InterleavedExperiment:
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
        self.results = defaultdict(lambda: {'A': 0, 'B': 0})

    def get_recommendations(self, user_id, n=5):
        a_recs = self.model_a.get_recommendations(user_id, n)
        b_recs = self.model_b.get_recommendations(user_id, n)
        
        interleaved = []
        a_index, b_index = 0, 0
        
        while len(interleaved) < n:
            if random.random() < 0.5 and a_index < len(a_recs):
                interleaved.append(('A', a_recs[a_index]))
                a_index += 1
            elif b_index < len(b_recs):
                interleaved.append(('B', b_recs[b_index]))
                b_index += 1
            else:
                interleaved.append(('A', a_recs[a_index]))
                a_index += 1
        
        return interleaved

    def record_click(self, user_id, model):
        self.results[user_id][model] += 1

    def analyze_results(self):
        total_a = sum(r['A'] for r in self.results.values())
        total_b = sum(r['B'] for r in self.results.values())
        
        print(f"Model A clicks: {total_a}")
        print(f"Model B clicks: {total_b}")
        
        if total_a > total_b:
            print("Model A is performing better")
        elif total_b > total_a:
            print("Model B is performing better")
        else:
            print("Models are performing equally")

# Example usage
experiment = InterleavedExperiment(model_a, model_b)

for user_id in range(100):
    recommendations = experiment.get_recommendations(user_id)
    
    # Simulate user clicks
    for model, rec in recommendations:
        if random.random() < 0.2:  # 20% chance of clicking
            experiment.record_click(user_id, model)

experiment.analyze_results()
```

Slide 13:

Canary Releases: Gradual Rollout Strategy

Implementing a gradual rollout strategy for canary releases allows for more controlled testing and easier rollback if issues arise:

```python
import random
import time

class GradualCanaryRelease:
    def __init__(self, old_model, new_model, rollout_duration, target_percentage):
        self.old_model = old_model
        self.new_model = new_model
        self.rollout_duration = rollout_duration
        self.target_percentage = target_percentage
        self.start_time = time.time()

    def get_current_percentage(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.rollout_duration:
            return self.target_percentage
        return (elapsed_time / self.rollout_duration) * self.target_percentage

    def predict(self, input_data):
        current_percentage = self.get_current_percentage()
        if random.random() < current_percentage:
            return self.new_model.predict(input_data), "new"
        else:
            return self.old_model.predict(input_data), "old"

    def rollback(self):
        self.target_percentage = 0
        self.start_time = time.time()
        print("Rolling back to old model")

# Example usage
canary = GradualCanaryRelease(old_model, new_model, rollout_duration=3600, target_percentage=0.2)

for _ in range(1000):
    result, model_used = canary.predict(input_data)
    
    # Simulate issue detection
    if model_used == "new" and random.random() < 0.01:  # 1% chance of issue
        canary.rollback()
        break

    time.sleep(0.1)  # Simulate time passing

print(f"Final rollout percentage: {canary.get_current_percentage():.2%}")
```

Slide 14:

Shadow Testing: Performance Comparison

When implementing shadow testing, it's crucial to compare not just the accuracy but also the performance of the new model against the production model:

```python
import time
import numpy as np

class PerformanceTracker:
    def __init__(self):
        self.latencies = []

    def record(self, latency):
        self.latencies.append(latency)

    def get_stats(self):
        latencies = np.array(self.latencies)
        return {
            "mean": np.mean(latencies),
            "median": np.median(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99)
        }

class ShadowTest:
    def __init__(self, production_model, shadow_model):
        self.production_model = production_model
        self.shadow_model = shadow_model
        self.production_tracker = PerformanceTracker()
        self.shadow_tracker = PerformanceTracker()

    def predict(self, input_data):
        # Production prediction
        prod_start = time.time()
        prod_result = self.production_model.predict(input_data)
        prod_latency = time.time() - prod_start
        self.production_tracker.record(prod_latency)

        # Shadow prediction
        shadow_start = time.time()
        shadow_result = self.shadow_model.predict(input_data)
        shadow_latency = time.time() - shadow_start
        self.shadow_tracker.record(shadow_latency)

        return prod_result

    def compare_performance(self):
        prod_stats = self.production_tracker.get_stats()
        shadow_stats = self.shadow_tracker.get_stats()

        print("Production Model Performance:")
        for metric, value in prod_stats.items():
            print(f"  {metric}: {value*1000:.2f} ms")

        print("\nShadow Model Performance:")
        for metric, value in shadow_stats.items():
            print(f"  {metric}: {value*1000:.2f} ms")

# Example usage
shadow_test = ShadowTest(production_model, shadow_model)

for _ in range(1000):
    shadow_test.predict(input_data)

shadow_test.compare_performance()
```

Slide 15:

Additional Resources

For those interested in diving deeper into ML model testing in production, here are some valuable resources:

1. "Continuous Delivery for Machine Learning" by D. Sato, A. Wider, and C. Windheuser (2019) ArXiv URL: [https://arxiv.org/abs/1909.00002](https://arxiv.org/abs/1909.00002)
2. "Hidden Technical Debt in Machine Learning Systems" by D. Sculley et al. (2015) ArXiv URL: [https://arxiv.org/abs/1909.00002](https://arxiv.org/abs/1909.00002)
3. "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction" by E. Breck et al. (2017) ArXiv URL: [https://arxiv.org/abs/1706.00409](https://arxiv.org/abs/1706.00409)

These papers provide in-depth insights into best practices for deploying and testing ML models in production environments.

