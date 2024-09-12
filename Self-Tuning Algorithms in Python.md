## Self-Tuning Algorithms in Python
Slide 1: Introduction to Self-Tuning in Python

Self-tuning is an advanced technique in machine learning and optimization where algorithms automatically adjust their parameters or hyperparameters to improve performance. This process is crucial for creating adaptive systems that can handle varying data distributions and evolving environments. In Python, we can implement self-tuning algorithms using various libraries and custom code.

```python
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Example of a simple self-tuning Random Forest using RandomizedSearchCV
param_dist = {
    'n_estimators': np.arange(10, 200),
    'max_depth': np.arange(1, 20),
    'min_samples_split': np.arange(2, 11)
}

rf = RandomForestClassifier()
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=100, cv=5)

# Assuming X and y are your feature matrix and target vector
random_search.fit(X, y)

print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)
```

Slide 2: Self-Tuning vs. Manual Tuning

Self-tuning offers several advantages over manual tuning. It saves time, reduces human bias, and can explore a wider range of parameter combinations. However, it may require more computational resources and can be less interpretable. The choice between self-tuning and manual tuning depends on the specific problem and available resources.

```python
import time
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Manual tuning example
def manual_tune(X, y):
    best_score = 0
    best_params = {}
    for C in [0.1, 1, 10]:
        for kernel in ['linear', 'rbf']:
            svm = SVC(C=C, kernel=kernel)
            svm.fit(X, y)
            score = svm.score(X, y)
            if score > best_score:
                best_score = score
                best_params = {'C': C, 'kernel': kernel}
    return best_params, best_score

# Self-tuning example
def self_tune(X, y):
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.best_score_

# Compare execution time and results
start_time = time.time()
manual_params, manual_score = manual_tune(X, y)
manual_time = time.time() - start_time

start_time = time.time()
auto_params, auto_score = self_tune(X, y)
auto_time = time.time() - start_time

print(f"Manual tuning: Time={manual_time:.2f}s, Score={manual_score:.4f}, Params={manual_params}")
print(f"Self-tuning: Time={auto_time:.2f}s, Score={auto_score:.4f}, Params={auto_params}")
```

Slide 3: Bayesian Optimization for Self-Tuning

Bayesian optimization is a powerful technique for self-tuning that uses probabilistic models to guide the search for optimal parameters. It's particularly useful when the objective function is expensive to evaluate, such as training deep neural networks.

```python
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer

# Define the objective function (e.g., model training and evaluation)
def objective(params):
    learning_rate, num_layers, activation = params
    # Create and train your model using these parameters
    # Return the negative of the performance metric (for minimization)
    return -model_performance

# Define the search space
space = [
    Real(1e-4, 1e-2, name='learning_rate', prior='log-uniform'),
    Integer(1, 5, name='num_layers'),
    Categorical(['relu', 'tanh'], name='activation')
]

# Perform Bayesian optimization
result = gp_minimize(objective, space, n_calls=50, random_state=0)

print("Best parameters:", result.x)
print("Best score:", -result.fun)

# Visualize the optimization process
from skopt.plots import plot_convergence
plot_convergence(result)
```

Slide 4: Online Learning and Adaptive Algorithms

Online learning algorithms update their parameters incrementally as new data arrives, making them naturally self-tuning. These algorithms are particularly useful for streaming data and non-stationary environments.

```python
from river import linear_model, metrics

# Create an online learning model
model = linear_model.PARegressor()

# Initialize a metric to track performance
mae = metrics.MAE()

# Simulate a data stream
for t in range(1000):
    # Generate a sample (in practice, this would come from your data stream)
    x = {'feature1': np.random.rand(), 'feature2': np.random.rand()}
    y = 2 * x['feature1'] + 3 * x['feature2'] + np.random.normal(0, 0.1)
    
    # Make a prediction
    y_pred = model.predict_one(x)
    
    # Update the model
    model.learn_one(x, y)
    
    # Update the metric
    mae.update(y, y_pred)
    
    if t % 100 == 0:
        print(f"Step {t}, MAE: {mae.get():.4f}")

print("Final model parameters:", model.weights)
```

Slide 5: Hyperparameter Optimization with Genetic Algorithms

Genetic algorithms provide a nature-inspired approach to self-tuning, mimicking the process of natural selection to evolve optimal parameter sets. They're particularly useful for complex, non-convex optimization problems.

```python
import random
from deap import base, creator, tools, algorithms

# Define the fitness function (to be maximized)
def evaluate(individual):
    # Decode individual into model parameters
    params = decode_parameters(individual)
    # Train and evaluate model with these parameters
    return model_performance(params),

# Set up the genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run the genetic algorithm
population = toolbox.population(n=50)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=True)

best_ind = tools.selBest(population, 1)[0]
print("Best individual:", best_ind)
print("Best fitness:", best_ind.fitness.values)
```

Slide 6: Reinforcement Learning for Self-Tuning

Reinforcement learning (RL) can be used for self-tuning by treating the tuning process as a sequential decision-making problem. The RL agent learns to adjust parameters based on the performance feedback it receives.

```python
import gym
import numpy as np
from stable_baselines3 import PPO

# Custom environment for hyperparameter tuning
class TuningEnv(gym.Env):
    def __init__(self):
        super(TuningEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(3,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
        
    def step(self, action):
        # Decode action into hyperparameters
        learning_rate = 10**(-5 + 3*action[0])
        batch_size = int(32 + 224*action[1])
        n_epochs = int(1 + 9*action[2])
        
        # Train model with these hyperparameters and get performance
        performance = train_and_evaluate(learning_rate, batch_size, n_epochs)
        
        # Construct observation (could include current performance, iteration, etc.)
        obs = np.array([performance, learning_rate, batch_size, n_epochs, self.steps])
        
        self.steps += 1
        done = self.steps >= 100
        
        return obs, performance, done, {}
    
    def reset(self):
        self.steps = 0
        return np.zeros(5)

# Create and train the RL agent
env = TuningEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Use the trained agent for tuning
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    if done:
        break

print("Final hyperparameters:", action)
print("Final performance:", reward)
```

Slide 7: Automated Feature Selection

Automated feature selection is a crucial aspect of self-tuning in machine learning pipelines. It helps identify the most relevant features, reducing dimensionality and potentially improving model performance.

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, 
                           n_redundant=5, n_repeated=0, n_classes=2, 
                           random_state=42)

# Create a base model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a selector
selector = SelectFromModel(rf, prefit=False)

# Fit the selector
selector.fit(X, y)

# Transform the data
X_selected = selector.transform(X)

print("Original number of features:", X.shape[1])
print("Number of features after selection:", X_selected.shape[1])

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)
print("Selected feature indices:", selected_feature_indices)

# Train a model on the selected features
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_selected, y)
print("Model accuracy on selected features:", rf_selected.score(X_selected, y))
```

Slide 8: Self-Tuning Neural Networks

Neural networks can incorporate self-tuning mechanisms to adapt their architecture or learning process dynamically. This can include techniques like neural architecture search or adaptive learning rates.

```python
import tensorflow as tf

class SelfTuningDense(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SelfTuningDense, self).__init__(**kwargs)
        self.units = units
        self.adaptive_lr = tf.Variable(0.01, trainable=False)
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
    def adapt_lr(self, loss):
        # Simple adaptive learning rate based on loss change
        if hasattr(self, 'prev_loss'):
            if loss < self.prev_loss:
                self.adaptive_lr.assign(self.adaptive_lr * 1.05)
            else:
                self.adaptive_lr.assign(self.adaptive_lr * 0.95)
        self.prev_loss = loss

# Use the self-tuning layer in a model
model = tf.keras.Sequential([
    SelfTuningDense(64, activation='relu', input_shape=(10,)),
    SelfTuningDense(32, activation='relu'),
    SelfTuningDense(1)
])

optimizer = tf.keras.optimizers.SGD()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = tf.keras.losses.mean_squared_error(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Apply adaptive learning rates
    for layer in model.layers:
        if isinstance(layer, SelfTuningDense):
            layer.adapt_lr(loss)
            optimizer.learning_rate.assign(layer.adaptive_lr)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop (assuming x_train and y_train are your data)
for epoch in range(100):
    loss = train_step(x_train, y_train)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")
```

Slide 9: Ensemble Methods with Self-Tuning

Ensemble methods can incorporate self-tuning mechanisms to dynamically adjust the combination of models or their individual parameters. This approach can lead to robust and adaptive predictive systems.

```python
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

class SelfTuningEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.estimators = [RandomForestRegressor() for _ in range(n_estimators)]
        self.weights = np.ones(n_estimators) / n_estimators
        self.meta_model = LinearRegression()
    
    def fit(self, X, y):
        # Train base models
        for estimator in self.estimators:
            estimator.fit(X, y)
        
        # Get predictions from base models
        base_predictions = np.column_stack([e.predict(X) for e in self.estimators])
        
        # Train meta-model to learn optimal combination
        self.meta_model.fit(base_predictions, y)
        
        # Update weights based on meta-model coefficients
        self.weights = np.abs(self.meta_model.coef_)
        self.weights /= np.sum(self.weights)
        
        return self
    
    def predict(self, X):
        base_predictions = np.column_stack([e.predict(X) for e in self.estimators])
        return self.meta_model.predict(base_predictions)

# Usage
ensemble = SelfTuningEnsemble(n_estimators=5)
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)

print("Ensemble weights:", ensemble.weights)
```

Slide 10: Real-Life Example: Self-Tuning Recommender System

Recommender systems often need to adapt to changing user preferences and new items. A self-tuning recommender system can automatically adjust its parameters to maintain or improve recommendation quality over time.

```python
import numpy as np
from sklearn.metrics import mean_squared_error

class SelfTuningRecommender:
    def __init__(self, n_factors=100, learning_rate=0.005, reg_param=0.02):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.user_factors = {}
        self.item_factors = {}
    
    def fit(self, ratings, epochs=10):
        for _ in range(epochs):
            for user, item, rating in ratings:
                if user not in self.user_factors:
                    self.user_factors[user] = np.random.normal(0, 0.1, self.n_factors)
                if item not in self.item_factors:
                    self.item_factors[item] = np.random.normal(0, 0.1, self.n_factors)
                
                prediction = np.dot(self.user_factors[user], self.item_factors[item])
                error = rating - prediction
                
                self.user_factors[user] += self.learning_rate * (error * self.item_factors[item] - self.reg_param * self.user_factors[user])
                self.item_factors[item] += self.learning_rate * (error * self.user_factors[user] - self.reg_param * self.item_factors[item])
            
            self._adjust_parameters(ratings)
    
    def _adjust_parameters(self, ratings):
        predictions = [np.dot(self.user_factors[user], self.item_factors[item]) for user, item, _ in ratings]
        true_ratings = [rating for _, _, rating in ratings]
        mse = mean_squared_error(true_ratings, predictions)
        
        if mse < 0.5:
            self.learning_rate *= 0.9
        else:
            self.learning_rate *= 1.1
        
        self.learning_rate = max(0.001, min(0.1, self.learning_rate))
    
    def predict(self, user, item):
        if user in self.user_factors and item in self.item_factors:
            return np.dot(self.user_factors[user], self.item_factors[item])
        return None

# Usage
recommender = SelfTuningRecommender()
ratings = [(1, 1, 5), (1, 2, 3), (2, 1, 4), (2, 2, 2)]
recommender.fit(ratings)
print(recommender.predict(1, 2))
```

Slide 11: Real-Life Example: Self-Tuning Image Classifier

Image classification models often need to adapt to new types of images or changes in image quality. A self-tuning image classifier can automatically adjust its architecture or hyperparameters to maintain high accuracy across different datasets.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

class SelfTuningImageClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._create_model()
        self.learning_rate = 0.001
    
    def _create_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def fit(self, x_train, y_train, epochs=10, batch_size=32):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        
        for epoch in range(epochs):
            history = self.model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
            accuracy = history.history['accuracy'][0]
            
            if accuracy > 0.95:
                self.learning_rate *= 0.9
            elif accuracy < 0.8:
                self.learning_rate *= 1.1
            
            self.learning_rate = max(0.0001, min(0.01, self.learning_rate))
            self.model.optimizer.learning_rate.assign(self.learning_rate)
            
            print(f"Epoch {epoch+1}/{epochs}, Accuracy: {accuracy:.4f}, Learning Rate: {self.learning_rate:.6f}")
    
    def predict(self, x):
        return self.model.predict(x)

# Usage (assuming you have your image data prepared)
classifier = SelfTuningImageClassifier(input_shape=(28, 28, 1), num_classes=10)
classifier.fit(x_train, y_train, epochs=20)
predictions = classifier.predict(x_test)
```

Slide 12: Challenges and Limitations of Self-Tuning

While self-tuning algorithms offer numerous advantages, they also face several challenges:

Computational Cost: Self-tuning often requires multiple iterations or parallel model training, which can be computationally expensive.

Overfitting: Aggressive self-tuning might lead to overfitting on the validation set, reducing generalization performance.

Complexity: Self-tuning algorithms can be more complex to implement and maintain compared to static models.

Interpretability: The dynamic nature of self-tuning models can make them less interpretable, which is crucial in certain domains.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SelfTuningModelWithSafeguards:
    def __init__(self, base_model):
        self.base_model = base_model
        self.best_model = None
        self.best_score = 0
        self.iterations_without_improvement = 0
    
    def fit(self, X, y, max_iterations=100, early_stopping_rounds=10):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        for i in range(max_iterations):
            # Train the model
            self.base_model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = self.base_model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            
            if score > self.best_score:
                self.best_model = self.base_model
                self.best_score = score
                self.iterations_without_improvement = 0
            else:
                self.iterations_without_improvement += 1
            
            # Early stopping
            if self.iterations_without_improvement >= early_stopping_rounds:
                print(f"Early stopping at iteration {i}")
                break
            
            # Self-tuning logic (simplified example)
            if hasattr(self.base_model, 'learning_rate'):
                self.base_model.learning_rate *= np.random.choice([0.9, 1.1])
        
        return self.best_model

# Usage
base_model = YourBaseModel()  # Replace with your actual model
self_tuning_model = SelfTuningModelWithSafeguards(base_model)
best_model = self_tuning_model.fit(X, y)
```

Slide 13: Future Directions in Self-Tuning

The field of self-tuning algorithms continues to evolve, with several exciting directions for future research and development:

1. Meta-Learning: Developing algorithms that can learn how to tune themselves across multiple tasks or datasets.
2. Adaptive Self-Tuning: Creating systems that can adjust their tuning strategy based on the characteristics of the data or task at hand.
3. Explainable Self-Tuning: Developing methods to make self-tuning processes more interpretable and transparent.
4. Continuous Learning: Designing self-tuning systems that can adapt to changing data distributions in real-time without catastrophic forgetting.
5. Energy-Efficient Self-Tuning: Creating algorithms that can balance performance improvements with computational costs, especially for edge devices.

```python
# Pseudocode for a meta-learning self-tuning system
class MetaLearningTuner:
    def __init__(self):
        self.meta_model = create_meta_model()
        self.task_database = []
    
    def tune(self, task, model):
        task_features = extract_task_features(task)
        similar_tasks = find_similar_tasks(task_features, self.task_database)
        
        if similar_tasks:
            initial_hyperparameters = self.meta_model.predict(task_features)
        else:
            initial_hyperparameters = default_hyperparameters()
        
        optimized_model = optimize_model(model, initial_hyperparameters, task)
        
        self.update_task_database(task, optimized_model)
        self.update_meta_model()
        
        return optimized_model

# Note: This is a high-level pseudocode and would require significant 
# implementation details for a working system.
```

Slide 14: Additional Resources

For those interested in diving deeper into self-tuning algorithms and related topics, here are some valuable resources:

1. "Automated Machine Learning: Methods, Systems, Challenges" (Springer, 2019) - Available on arXiv: [https://arxiv.org/abs/1904.12054](https://arxiv.org/abs/1904.12054)
2. "Neural Architecture Search: A Survey" by Thomas Elsken, Jan Hendrik Metzen, Frank Hutter (2018) - arXiv:1808.05377
3. "Hyperparameter Optimization: A Spectral Approach" by Elad Hazan, Adam Klivans, Yang Yuan (2017) - arXiv:1706.00764
4. "Learning to Learn by Gradient Descent by Gradient Descent" by Marcin Andrychowicz et al. (2016) - arXiv:1606.04474
5. "Auto-WEKA: Combined Selection and Hyperparameter Optimization of Classification Algorithms" by Chris Thornton et al. (2013) - Available at: [https://www.cs.ubc.ca/~nando/papers/autoweka.pdf](https://www.cs.ubc.ca/~nando/papers/autoweka.pdf)

These resources provide a mix of theoretical foundations and practical implementations of self-tuning algorithms and automated machine learning techniques.

