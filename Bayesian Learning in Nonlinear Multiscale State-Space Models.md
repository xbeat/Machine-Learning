## Bayesian Learning in Nonlinear Multiscale State-Space Models
Slide 1: Introduction to Bayesian Learning in Nonlinear Multiscale State-Space Models

Bayesian learning in nonlinear multiscale state-space models combines probabilistic inference with complex dynamical systems. This approach allows us to estimate hidden states and parameters in systems that evolve across multiple scales of time and space, while accounting for uncertainties in our observations and model.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Simple example of a nonlinear state-space model
def nonlinear_state_transition(x, a=0.5, b=25, c=8):
    return a * x + b * x / (1 + x**2) + c * np.cos(1.2 * x)

# Generate a sequence of states
x = np.zeros(100)
x[0] = 0.1
for t in range(1, 100):
    x[t] = nonlinear_state_transition(x[t-1]) + np.random.normal(0, 0.1)

plt.plot(x)
plt.title('Nonlinear State-Space Model Trajectory')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.show()
```

Slide 2: State-Space Models: The Foundation

State-space models describe the evolution of a system's hidden states over time and how these states relate to observable measurements. In nonlinear multiscale models, the state transitions and observation processes can be complex, involving multiple timescales and nonlinear relationships.

Slide 3: State-Space Models: The Foundation 

```python
import numpy as np

class NonlinearMultiscaleStateSpaceModel:
    def __init__(self, initial_state, observation_noise, transition_noise):
        self.state = initial_state
        self.observation_noise = observation_noise
        self.transition_noise = transition_noise
    
    def transition(self):
        # Nonlinear state transition with multiple timescales
        fast_component = 0.9 * self.state + 0.1 * np.sin(self.state)
        slow_component = 0.1 * np.tanh(self.state)
        self.state = fast_component + slow_component + np.random.normal(0, self.transition_noise)
        
    def observe(self):
        # Nonlinear observation process
        return np.exp(self.state) + np.random.normal(0, self.observation_noise)

# Simulate the model
model = NonlinearMultiscaleStateSpaceModel(initial_state=0, observation_noise=0.1, transition_noise=0.05)
states, observations = [], []

for _ in range(100):
    model.transition()
    states.append(model.state)
    observations.append(model.observe())

plt.plot(states, label='True State')
plt.plot(observations, label='Observations')
plt.legend()
plt.title('Nonlinear Multiscale State-Space Model Simulation')
plt.show()
```

Slide 3: Bayesian Inference: The Core of Learning

Bayesian inference forms the backbone of learning in these models. It allows us to update our beliefs about the system's states and parameters as we observe new data. The key is to compute the posterior distribution of the states given the observations.

Slide 4: Bayesian Inference: The Core of Learning

```python
import numpy as np
from scipy.stats import norm

def bayesian_update(prior_mean, prior_var, observation, obs_var):
    # Compute the posterior distribution parameters
    likelihood_precision = 1 / obs_var
    posterior_precision = 1 / prior_var + likelihood_precision
    posterior_mean = (prior_mean / prior_var + observation * likelihood_precision) / posterior_precision
    posterior_var = 1 / posterior_precision
    
    return posterior_mean, posterior_var

# Example usage
prior_mean, prior_var = 0, 1
observation, obs_var = 2, 0.5

posterior_mean, posterior_var = bayesian_update(prior_mean, prior_var, observation, obs_var)

x = np.linspace(-4, 6, 1000)
plt.plot(x, norm.pdf(x, prior_mean, np.sqrt(prior_var)), label='Prior')
plt.plot(x, norm.pdf(x, observation, np.sqrt(obs_var)), label='Likelihood')
plt.plot(x, norm.pdf(x, posterior_mean, np.sqrt(posterior_var)), label='Posterior')
plt.legend()
plt.title('Bayesian Update Example')
plt.show()

print(f"Posterior mean: {posterior_mean:.2f}, Posterior variance: {posterior_var:.2f}")
```

Slide 5: Particle Filtering: Handling Nonlinearity

Particle filtering is a powerful technique for performing Bayesian inference in nonlinear state-space models. It represents the posterior distribution using a set of weighted particles, which are updated as new observations arrive.

Slide 6: Particle Filtering: Handling Nonlinearity

```python
import numpy as np
import matplotlib.pyplot as plt

def particle_filter(observations, num_particles=1000):
    particles = np.random.normal(0, 1, num_particles)
    weights = np.ones(num_particles) / num_particles
    
    filtered_states = []
    
    for obs in observations:
        # Predict
        particles = nonlinear_state_transition(particles) + np.random.normal(0, 0.1, num_particles)
        
        # Update weights
        weights *= norm.pdf(obs, np.exp(particles), 0.1)
        weights /= np.sum(weights)
        
        # Resample if effective sample size is too low
        if 1 / np.sum(weights**2) < num_particles / 2:
            indices = np.random.choice(num_particles, num_particles, p=weights)
            particles = particles[indices]
            weights = np.ones(num_particles) / num_particles
        
        filtered_states.append(np.average(particles, weights=weights))
    
    return np.array(filtered_states)

# Use the particle filter on our simulated data
filtered_states = particle_filter(observations)

plt.plot(states, label='True State')
plt.plot(observations, label='Observations')
plt.plot(filtered_states, label='Filtered State')
plt.legend()
plt.title('Particle Filtering Results')
plt.show()
```

Slide 7: Handling Multiple Scales: Hierarchical Models

Multiscale state-space models often involve hierarchical structures, where processes at different scales interact. Bayesian learning in these models requires careful consideration of how information flows between scales.

Slide 8: Handling Multiple Scales: Hierarchical Models

```python
import numpy as np
import matplotlib.pyplot as plt

class HierarchicalMultiscaleModel:
    def __init__(self, num_scales=3):
        self.num_scales = num_scales
        self.states = np.zeros(num_scales)
    
    def transition(self):
        for i in range(self.num_scales):
            # Slower scales influence faster ones
            if i > 0:
                self.states[i] += 0.1 * self.states[i-1]
            
            # Each scale has its own dynamics
            self.states[i] = 0.9 * self.states[i] + 0.1 * np.sin(self.states[i])
            self.states[i] += np.random.normal(0, 0.01 * (i + 1))
    
    def observe(self):
        # Observation is a combination of all scales
        return np.sum(self.states) + np.random.normal(0, 0.1)

# Simulate the hierarchical model
model = HierarchicalMultiscaleModel()
states_history = []
observations = []

for _ in range(1000):
    model.transition()
    states_history.append(model.states.())
    observations.append(model.observe())

states_history = np.array(states_history)

plt.figure(figsize=(10, 6))
for i in range(model.num_scales):
    plt.plot(states_history[:, i], label=f'Scale {i+1}')
plt.plot(observations, label='Observations', alpha=0.5)
plt.legend()
plt.title('Hierarchical Multiscale State-Space Model')
plt.show()
```

Slide 9: Parameter Estimation: Learning the Model Dynamics

In addition to inferring hidden states, Bayesian learning allows us to estimate the parameters governing the system's dynamics. This is crucial for understanding and predicting the behavior of complex multiscale systems.

Slide 10: Parameter Estimation: Learning the Model Dynamics

```python
import numpy as np
import pymc3 as pm

# Generate some example data
true_params = {'a': 0.8, 'b': 0.1, 'noise': 0.05}
T = 100
x = np.zeros(T)
y = np.zeros(T)
x[0] = 0.5
for t in range(1, T):
    x[t] = true_params['a'] * x[t-1] + true_params['b'] * np.sin(x[t-1]) + np.random.normal(0, true_params['noise'])
    y[t] = x[t] + np.random.normal(0, 0.1)

# Bayesian parameter estimation
with pm.Model() as model:
    # Priors
    a = pm.Uniform('a', 0, 1)
    b = pm.Normal('b', 0, 1)
    noise = pm.HalfNormal('noise', 0.1)
    
    # Likelihood
    x = pm.AR('x', a, sigma=noise, observed=y)
    
    # Inference
    trace = pm.sample(2000, tune=1000, return_inferencedata=False)

# Plot results
pm.plot_posterior(trace, var_names=['a', 'b', 'noise'])
plt.show()

print("True parameters:", true_params)
print("Estimated parameters:")
print(pm.summary(trace, var_names=['a', 'b', 'noise']))
```

Slide 11: Real-Life Example: Climate Modeling

Climate systems exhibit complex multiscale dynamics, making them an ideal application for Bayesian learning in nonlinear multiscale state-space models. Let's consider a simplified model of global temperature anomalies.

Slide 12: Real-Life Example: Climate Modeling

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class ClimateModel:
    def __init__(self, solar_cycle_length=11, volcanic_activity_rate=0.05):
        self.temperature = 0
        self.solar_cycle = 0
        self.volcanic_activity = 0
        self.solar_cycle_length = solar_cycle_length
        self.volcanic_activity_rate = volcanic_activity_rate
    
    def transition(self):
        # Solar cycle (faster scale)
        self.solar_cycle += 2 * np.pi / self.solar_cycle_length
        solar_effect = 0.1 * np.sin(self.solar_cycle)
        
        # Volcanic activity (slower scale)
        if np.random.random() < self.volcanic_activity_rate:
            self.volcanic_activity = -np.random.exponential(0.5)
        else:
            self.volcanic_activity *= 0.9
        
        # Temperature dynamics
        self.temperature = 0.9 * self.temperature + solar_effect + 0.1 * self.volcanic_activity + np.random.normal(0, 0.05)
    
    def observe(self):
        return self.temperature + np.random.normal(0, 0.1)

# Simulate the climate model
model = ClimateModel()
temperatures = []
observations = []

for _ in range(500):
    model.transition()
    temperatures.append(model.temperature)
    observations.append(model.observe())

plt.figure(figsize=(12, 6))
plt.plot(temperatures, label='True Temperature')
plt.plot(observations, label='Observed Temperature', alpha=0.5)
plt.legend()
plt.title('Multiscale Climate Model: Temperature Anomalies')
plt.xlabel('Time (months)')
plt.ylabel('Temperature Anomaly (°C)')
plt.show()
```

Slide 13: Kalman Filtering: Linear Subcase

While our focus is on nonlinear models, it's instructive to understand the Kalman filter, which provides an optimal solution for linear Gaussian state-space models. This serves as a foundation for more advanced nonlinear techniques.

Slide 14: Real-Life Example: Climate Modeling

```python
import numpy as np
import matplotlib.pyplot as plt

def kalman_filter(y, A, C, Q, R, x0, P0):
    n = len(y)
    x = np.zeros((n, x0.shape[0]))
    P = np.zeros((n, P0.shape[0], P0.shape[1]))
    
    x[0] = x0
    P[0] = P0
    
    for t in range(1, n):
        # Predict
        x_pred = A @ x[t-1]
        P_pred = A @ P[t-1] @ A.T + Q
        
        # Update
        K = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + R)
        x[t] = x_pred + K @ (y[t] - C @ x_pred)
        P[t] = (np.eye(x0.shape[0]) - K @ C) @ P_pred
    
    return x, P

# Generate some linear data
A = np.array([[0.9, 0.1], [-0.1, 0.9]])
C = np.array([[1.0, 0.0]])
Q = np.eye(2) * 0.01
R = np.array([[0.1]])

x_true = np.zeros((100, 2))
y = np.zeros((100, 1))

for t in range(1, 100):
    x_true[t] = A @ x_true[t-1] + np.random.multivariate_normal(np.zeros(2), Q)
    y[t] = C @ x_true[t] + np.random.normal(0, np.sqrt(R[0,0]))

# Apply Kalman filter
x0 = np.zeros(2)
P0 = np.eye(2)
x_filtered, P_filtered = kalman_filter(y, A, C, Q, R, x0, P0)

plt.figure(figsize=(12, 6))
plt.plot(x_true[:, 0], label='True State 1')
plt.plot(x_true[:, 1], label='True State 2')
plt.plot(x_filtered[:, 0], label='Filtered State 1')
plt.plot(x_filtered[:, 1], label='Filtered State 2')
plt.legend()
plt.title('Kalman Filter Results')
plt.show()
```

Slide 15: Extended Kalman Filter: Handling Mild Nonlinearity

The Extended Kalman Filter (EKF) extends the Kalman filter to mildly nonlinear systems by linearizing the state transition and observation models around the current state estimate.

Slide 16: Extended Kalman Filter: Handling Mild Nonlinearity

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.array([x[0] + 0.1 * np.sin(x[1]), x[1] + 0.1 * np.cos(x[0])])

def h(x):
    return np.array([np.sqrt(x[0]**2 + x[1]**2)])

def F(x):
    return np.array([[1, 0.1 * np.cos(x[1])], [-0.1 * np.sin(x[0]), 1]])

def H(x):
    r = np.sqrt(x[0]**2 + x[1]**2)
    return np.array([[x[0] / r, x[1] / r]])

def extended_kalman_filter(y, f, h, F, H, Q, R, x0, P0):
    n = len(y)
    x = np.zeros((n, x0.shape[0]))
    P = np.zeros((n, P0.shape[0], P0.shape[1]))
    
    x[0] = x0
    P[0] = P0
    
    for t in range(1, n):
        # Predict
        x_pred = f(x[t-1])
        F_t = F(x[t-1])
        P_pred = F_t @ P[t-1] @ F_t.T + Q
        
        # Update
        H_t = H(x_pred)
        K = P_pred @ H_t.T @ np.linalg.inv(H_t @ P_pred @ H_t.T + R)
        x[t] = x_pred + K @ (y[t] - h(x_pred))
        P[t] = (np.eye(x0.shape[0]) - K @ H_t) @ P_pred
    
    return x, P

# Generate nonlinear data and apply EKF
Q = np.eye(2) * 0.01
R = np.array([[0.1]])
x_true = np.zeros((100, 2))
y = np.zeros((100, 1))

for t in range(1, 100):
    x_true[t] = f(x_true[t-1]) + np.random.multivariate_normal(np.zeros(2), Q)
    y[t] = h(x_true[t]) + np.random.normal(0, np.sqrt(R[0,0]))

x0 = np.zeros(2)
P0 = np.eye(2)
x_filtered, P_filtered = extended_kalman_filter(y, f, h, F, H, Q, R, x0, P0)

plt.figure(figsize=(12, 6))
plt.plot(x_true[:, 0], x_true[:, 1], label='True State')
plt.plot(x_filtered[:, 0], x_filtered[:, 1], label='Filtered State')
plt.legend()
plt.title('Extended Kalman Filter Results')
plt.show()
```

Slide 17: Unscented Kalman Filter: Improved Nonlinear Estimation

The Unscented Kalman Filter (UKF) offers better performance for highly nonlinear systems compared to the EKF. It uses a deterministic sampling technique to propagate the state distribution through the nonlinear functions.

Slide 18: Unscented Kalman Filter: Improved Nonlinear Estimation

```python
import numpy as np
from scipy.linalg import cholesky

def unscented_transform(f, x, P, Q):
    n = x.shape[0]
    alpha, beta, kappa = 1e-3, 2, 0
    lambda_ = alpha**2 * (n + kappa) - n
    
    # Generate sigma points
    sigma_points = np.zeros((2*n + 1, n))
    sigma_points[0] = x
    L = cholesky((n + lambda_) * P)
    for i in range(n):
        sigma_points[i+1] = x + L[i]
        sigma_points[n+i+1] = x - L[i]
    
    # Weights for mean and covariance
    Wm = np.full(2*n + 1, 1 / (2*(n + lambda_)))
    Wm[0] = lambda_ / (n + lambda_)
    Wc = Wm.()
    Wc[0] += (1 - alpha**2 + beta)
    
    # Propagate sigma points
    Y = np.array([f(sigma) for sigma in sigma_points])
    
    # Compute transformed mean and covariance
    y_mean = np.sum(Wm[:, np.newaxis] * Y, axis=0)
    y_cov = np.sum(Wc[:, np.newaxis, np.newaxis] * 
                   (Y - y_mean)[..., np.newaxis] @ 
                   (Y - y_mean)[:, np.newaxis, :], axis=0) + Q
    
    return y_mean, y_cov, Y

def unscented_kalman_filter(y, f, h, Q, R, x0, P0):
    n = len(y)
    x = np.zeros((n, x0.shape[0]))
    P = np.zeros((n, P0.shape[0], P0.shape[1]))
    
    x[0] = x0
    P[0] = P0
    
    for t in range(1, n):
        # Predict
        x_pred, P_pred, X = unscented_transform(f, x[t-1], P[t-1], Q)
        
        # Update
        y_pred, Pyy, Y = unscented_transform(h, x_pred, P_pred, R)
        Pxy = np.sum(Wc[:, np.newaxis, np.newaxis] * 
                     (X - x_pred)[..., np.newaxis] @ 
                     (Y - y_pred)[:, np.newaxis, :], axis=0)
        
        K = Pxy @ np.linalg.inv(Pyy)
        x[t] = x_pred + K @ (y[t] - y_pred)
        P[t] = P_pred - K @ Pyy @ K.T
    
    return x, P

# Apply UKF to the same nonlinear system as in the EKF example
# ... (use the same f, h, Q, R, x_true, y as before)

x_filtered_ukf, P_filtered_ukf = unscented_kalman_filter(y, f, h, Q, R, x0, P0)

plt.figure(figsize=(12, 6))
plt.plot(x_true[:, 0], x_true[:, 1], label='True State')
plt.plot(x_filtered_ukf[:, 0], x_filtered_ukf[:, 1], label='UKF Filtered State')
plt.legend()
plt.title('Unscented Kalman Filter Results')
plt.show()
```

Slide 19: Real-Life Example: Robot Localization

Robot localization is a practical application of Bayesian learning in nonlinear multiscale state-space models. Here, we simulate a robot moving in a 2D environment with noisy sensor measurements.

Slide 20: Real-Life Example: Robot Localization

```python
import numpy as np
import matplotlib.pyplot as plt

class Robot:
    def __init__(self, x, y, theta):
        self.true_state = np.array([x, y, theta])
        self.estimated_state = self.true_state + np.random.normal(0, 0.1, 3)
        self.covariance = np.eye(3) * 0.1
    
    def move(self, v, omega, dt):
        # True motion (nonlinear)
        theta = self.true_state[2]
        self.true_state += np.array([
            v * np.cos(theta) * dt,
            v * np.sin(theta) * dt,
            omega * dt
        ]) + np.random.normal(0, 0.05, 3)
        
        # Estimated motion (linearized)
        theta_est = self.estimated_state[2]
        F = np.array([
            [1, 0, -v * np.sin(theta_est) * dt],
            [0, 1, v * np.cos(theta_est) * dt],
            [0, 0, 1]
        ])
        self.estimated_state = F @ self.estimated_state + np.array([
            v * np.cos(theta_est) * dt,
            v * np.sin(theta_est) * dt,
            omega * dt
        ])
        self.covariance = F @ self.covariance @ F.T + np.eye(3) * 0.01
    
    def measure(self, landmarks):
        measurements = []
        for lm in landmarks:
            dx = lm[0] - self.true_state[0]
            dy = lm[1] - self.true_state[1]
            dist = np.sqrt(dx**2 + dy**2) + np.random.normal(0, 0.1)
            angle = np.arctan2(dy, dx) - self.true_state[2] + np.random.normal(0, 0.05)
            measurements.append([dist, angle])
        return np.array(measurements)
    
    def update(self, measurements, landmarks):
        for z, lm in zip(measurements, landmarks):
            dx = lm[0] - self.estimated_state[0]
            dy = lm[1] - self.estimated_state[1]
            q = dx**2 + dy**2
            z_pred = np.array([np.sqrt(q), np.arctan2(dy, dx) - self.estimated_state[2]])
            
            H = np.array([
                [-dx/np.sqrt(q), -dy/np.sqrt(q), 0],
                [dy/q, -dx/q, -1]
            ])
            
            S = H @ self.covariance @ H.T + np.diag([0.01, 0.0025])
            K = self.covariance @ H.T @ np.linalg.inv(S)
            
            self.estimated_state += K @ (z - z_pred)
            self.covariance = (np.eye(3) - K @ H) @ self.covariance

# Simulation
robot = Robot(0, 0, 0)
landmarks = np.array([[5, 5], [-5, 5], [-5, -5], [5, -5]])
true_path = []
estimated_path = []

for t in range(100):
    robot.move(0.1, 0.05, 1)
    measurements = robot.measure(landmarks)
    robot.update(measurements, landmarks)
    
    true_path.append(robot.true_state[:2])
    estimated_path.append(robot.estimated_state[:2])

true_path = np.array(true_path)
estimated_path = np.array(estimated_path)

plt.figure(figsize=(10, 10))
plt.plot(true_path[:, 0], true_path[:, 1], label='True Path')
plt.plot(estimated_path[:, 0], estimated_path[:, 1], label='Estimated Path')
plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='^', label='Landmarks')
plt.legend()
plt.title('Robot Localization using EKF')
plt.axis('equal')
plt.show()
```

Slide 21: Particle Filters for Highly Nonlinear Systems

Particle filters, also known as Sequential Monte Carlo methods, are particularly well-suited for highly nonlinear and non-Gaussian state-space models. They represent the posterior distribution using a set of weighted particles.

Slide 22: Particle Filters for Highly Nonlinear Systems

```python
import numpy as np
import matplotlib.pyplot as plt

def particle_filter(y, f, h, Q, R, num_particles=1000):
    n = len(y)
    d = Q.shape[0]
    
    particles = np.random.multivariate_normal(np.zeros(d), np.eye(d), num_particles)
    weights = np.ones(num_particles) / num_particles
    
    x_est = np.zeros((n, d))
    
    for t in range(n):
        # Predict
        particles = np.array([f(p) for p in particles]) + np.random.multivariate_normal(np.zeros(d), Q, num_particles)
        
        # Update
        weights *= np.array([np.exp(-0.5 * (y[t] - h(p)).T @ np.linalg.inv(R) @ (y[t] - h(p))) for p in particles])
        weights /= np.sum(weights)
        
        # Resample if effective sample size is too low
        if 1 / np.sum(weights**2) < num_particles / 2:
            indices = np.random.choice(num_particles, num_particles, p=weights)
            particles = particles[indices]
            weights = np.ones(num_particles) / num_particles
        
        x_est[t] = np.average(particles, axis=0, weights=weights)
    
    return x_est

# Highly nonlinear system
def f(x):
    return np.array([x[0]/2 + 25*x[0]/(1+x[0]**2) + 8*np.cos(1.2*x[1]), x[1]/2 + 25*x[1]/(1+x[1]**2) + 8*np.cos(1.2*x[0])])

def h(x):
    return np.array([np.arctan(x[1]/x[0])])

Q = np.eye(2) * 0.1
R = np.array([[0.1]])

# Generate true states and observations
n = 100
x_true = np.zeros((n, 2))
y = np.zeros((n, 1))

for t in range(1, n):
    x_true[t] = f(x_true[t-1]) + np.random.multivariate_normal(np.zeros(2), Q)
    y[t] = h(x_true[t]) + np.random.multivariate_normal(np.zeros(1), R)

# Apply particle filter
x_est = particle_filter(y, f, h, Q, R)

plt.figure(figsize=(12, 6))
plt.plot(x_true[:, 0], x_true[:, 1], label='True State')
plt.plot(x_est[:, 0], x_est[:, 1], label='Particle Filter Estimate')
plt.legend()
plt.title('Particle Filter Results for Highly Nonlinear System')
plt.show()
```

Slide 23: Challenges and Future Directions

Bayesian learning in nonlinear multiscale state-space models faces several challenges:

1. Computational complexity: As the dimensionality of the state space increases, the computational cost of inference grows rapidly.
2. Model selection: Choosing the right level of complexity for the model is crucial but often difficult.
3. Handling unknown parameters: Joint estimation of states and parameters can be challenging, especially in multiscale systems.
4. Non-Gaussian noise: Many real-world systems exhibit non-Gaussian noise, requiring more sophisticated inference techniques.

Slide 24: Challenges and Future Directions

Future research directions include:

1. Developing more efficient inference algorithms for high-dimensional systems.
2. Incorporating machine learning techniques for automatic model selection and adaptation.
3. Exploring hybrid approaches that combine the strengths of different inference methods.
4. Applying these techniques to emerging fields such as quantum computing and neuroscience.

Slide 25: Additional Resources

For those interested in delving deeper into Bayesian learning in nonlinear multiscale state-space models, here are some valuable resources:

1. Särkkä, S. (2013). Bayesian Filtering and Smoothing. Cambridge University Press.
2. Doucet, A., de Freitas, N., & Gordon, N. (2001). Sequential Monte Carlo Methods in Practice. Springer.
3. Kantas, N., Doucet, A., Singh, S. S., Maciejowski, J., & Chopin, N. (2015). On Particle Methods for Parameter Estimation in State-Space Models. Statistical Science, 30(3), 328-351. arXiv:1412.8695

Slide 26: Additional Resources

4. Ghahramani, Z., & Roweis, S. T. (1999). Learning Nonlinear Dynamical Systems Using an EM Algorithm. Advances in Neural Information Processing Systems, 11, 431-437.
5. Wan, E. A., & Van Der Merwe, R. (2000). The Unscented Kalman Filter for Nonlinear Estimation. Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium, 153-158.

These resources provide a comprehensive overview of the theoretical foundations and practical applications of Bayesian learning in nonlinear multiscale state-space models. They cover various techniques including Kalman filtering, particle filtering, and expectation-maximization algorithms for parameter estimation.

