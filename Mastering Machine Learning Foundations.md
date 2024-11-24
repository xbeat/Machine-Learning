## Mastering Machine Learning Foundations
Slide 1: Linear Algebra Foundations for Machine Learning

Linear algebra forms the mathematical backbone of machine learning algorithms. Understanding vector spaces, transformations, and matrix operations is crucial for implementing efficient ML solutions from scratch. We'll start with vector operations and gradually build complexity.

```python
import numpy as np

class VectorSpace:
    def __init__(self, vectors):
        self.vectors = np.array(vectors)
    
    def linear_combination(self, coefficients):
        # Compute linear combination of vectors
        return np.dot(coefficients, self.vectors)
    
    def is_linearly_independent(self):
        # Check if vectors are linearly independent
        rank = np.linalg.matrix_rank(self.vectors)
        return rank == len(self.vectors)

# Example usage
vectors = [[1, 0], [0, 1]]  # Standard basis vectors
vs = VectorSpace(vectors)
print(f"Linear combination: {vs.linear_combination([2, 3])}")
print(f"Linearly independent: {vs.is_linearly_independent()}")
```

Slide 2: Matrix Transformations Implementation

Matrix transformations are essential for understanding how neural networks modify input data through layers. This implementation demonstrates basic matrix operations and their geometric interpretations in machine learning contexts.

```python
class MatrixTransformation:
    def __init__(self, matrix):
        self.matrix = np.array(matrix)
    
    def transform(self, vector):
        return np.dot(self.matrix, vector)
    
    def compose(self, other_transform):
        return np.dot(self.matrix, other_transform.matrix)
    
    def get_determinant(self):
        return np.linalg.det(self.matrix)

# Example: Rotation matrix (45 degrees)
theta = np.pi/4
rotation = MatrixTransformation([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

vector = np.array([1, 0])
print(f"Transformed vector: {rotation.transform(vector)}")
```

Slide 3: Probability Fundamentals in ML

Probability theory provides the framework for understanding uncertainty in machine learning models. This implementation focuses on fundamental probability calculations essential for building probabilistic models.

```python
class ProbabilityDistribution:
    def __init__(self, data):
        self.data = np.array(data)
        
    def gaussian_pdf(self, x, mu, sigma):
        return (1/(sigma * np.sqrt(2*np.pi))) * \
               np.exp(-0.5 * ((x-mu)/sigma)**2)
    
    def estimate_parameters(self):
        mu = np.mean(self.data)
        sigma = np.std(self.data)
        return mu, sigma
    
    def likelihood(self, x):
        mu, sigma = self.estimate_parameters()
        return self.gaussian_pdf(x, mu, sigma)

# Example usage
data = np.random.normal(0, 1, 1000)
pd = ProbabilityDistribution(data)
mu, sigma = pd.estimate_parameters()
print(f"Estimated μ: {mu:.2f}, σ: {sigma:.2f}")
```

Slide 4: Statistical Inference Implementation

Statistical inference allows us to draw conclusions from data, crucial for model evaluation and hypothesis testing in machine learning applications. This implementation covers key statistical concepts.

```python
class StatisticalInference:
    def __init__(self, sample_data):
        self.data = np.array(sample_data)
        
    def confidence_interval(self, confidence=0.95):
        n = len(self.data)
        mean = np.mean(self.data)
        std_err = np.std(self.data, ddof=1) / np.sqrt(n)
        z_score = np.abs(np.percentile(np.random.standard_normal(10000),
                                     (1 - confidence) * 100))
        margin = z_score * std_err
        return mean - margin, mean + margin
    
    def hypothesis_test(self, null_value, alpha=0.05):
        t_stat = (np.mean(self.data) - null_value) / \
                (np.std(self.data, ddof=1) / np.sqrt(len(self.data)))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(self.data)-1))
        return p_value < alpha

sample = np.random.normal(10, 2, 100)
si = StatisticalInference(sample)
ci = si.confidence_interval()
print(f"95% Confidence Interval: [{ci[0]:.2f}, {ci[1]:.2f}]")
```

Slide 5: Calculus Foundations in Neural Networks

Understanding derivatives and gradients is fundamental for implementing backpropagation in neural networks. This implementation demonstrates automatic differentiation concepts used in deep learning frameworks.

```python
class AutoDifferentiation:
    def __init__(self, value, derivative=1.0):
        self.value = value
        self.derivative = derivative
        
    def __mul__(self, other):
        value = self.value * other.value
        derivative = self.value * other.derivative + other.value * self.derivative
        return AutoDifferentiation(value, derivative)
    
    def __add__(self, other):
        return AutoDifferentiation(
            self.value + other.value, 
            self.derivative + other.derivative
        )

# Example: Computing gradient of f(x) = x^2 + 2x
x = AutoDifferentiation(3.0, 1.0)
f = x * x + AutoDifferentiation(2.0) * x
print(f"Value at x=3: {f.value}")
print(f"Derivative at x=3: {f.derivative}")
```

Slide 6: Optimization Algorithms From Scratch

Optimization algorithms are crucial for training machine learning models. This implementation shows gradient descent variants commonly used in deep learning applications.

```python
class Optimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None
        
    def gradient_descent(self, params, gradients):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
            
        self.velocity = self.momentum * self.velocity - self.lr * gradients
        return params + self.velocity
    
    def adam(self, params, gradients, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if not hasattr(self, 'm'):
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        self.m = beta1 * self.m + (1 - beta1) * gradients
        self.v = beta2 * self.v + (1 - beta2) * np.square(gradients)
        
        m_hat = self.m / (1 - beta1**t)
        v_hat = self.v / (1 - beta2**t)
        
        return params - self.lr * m_hat / (np.sqrt(v_hat) + epsilon)

# Example usage
params = np.array([1.0, 2.0])
gradients = np.array([0.1, 0.2])
optimizer = Optimizer()
updated_params = optimizer.gradient_descent(params, gradients)
print(f"Updated parameters: {updated_params}")
```

Slide 7: Vector Calculus for Deep Learning

Vector calculus is essential for understanding gradient flow in neural networks. This implementation demonstrates key concepts of multivariable calculus used in deep learning.

```python
class VectorCalculus:
    def __init__(self, dimensions):
        self.dims = dimensions
        
    def jacobian(self, func, x, epsilon=1e-7):
        jac = np.zeros((len(func(x)), len(x)))
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon
            x_minus = x.copy()
            x_minus[i] -= epsilon
            jac[:, i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
        return jac
    
    def hessian(self, func, x, epsilon=1e-5):
        def gradient(x):
            return self.jacobian(func, x, epsilon).flatten()
        return self.jacobian(gradient, x, epsilon)

# Example: Computing Jacobian for a simple function
def f(x):
    return np.array([x[0]**2 + x[1], x[0] * x[1]])

vc = VectorCalculus(2)
x = np.array([1.0, 2.0])
print(f"Jacobian at x=[1,2]:\n{vc.jacobian(f, x)}")
```

Slide 8: Statistical Learning Theory

Statistical learning theory provides the theoretical foundation for machine learning algorithms. This implementation demonstrates key concepts like empirical risk minimization.

```python
class StatisticalLearning:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        
    def empirical_risk(self, weights, loss_func):
        predictions = np.dot(self.X, weights)
        return np.mean(loss_func(predictions, self.y))
    
    def cross_validation_risk(self, weights, loss_func, k_folds=5):
        fold_size = len(self.X) // k_folds
        risks = []
        
        for k in range(k_folds):
            start_idx = k * fold_size
            end_idx = (k + 1) * fold_size
            
            X_val = self.X[start_idx:end_idx]
            y_val = self.y[start_idx:end_idx]
            
            predictions = np.dot(X_val, weights)
            risk = np.mean(loss_func(predictions, y_val))
            risks.append(risk)
            
        return np.mean(risks), np.std(risks)

# Example usage
def squared_loss(pred, true):
    return (pred - true) ** 2

X = np.random.randn(100, 3)
y = np.random.randn(100)
sl = StatisticalLearning(X, y)
weights = np.random.randn(3)
risk, risk_std = sl.cross_validation_risk(weights, squared_loss)
print(f"Cross-validation risk: {risk:.4f} ± {risk_std:.4f}")
```

Slide 9: Information Theory in Machine Learning

Information theory concepts are crucial for understanding model complexity and optimization. This implementation covers entropy, KL divergence, and mutual information calculations essential for deep learning.

```python
class InformationTheory:
    def __init__(self):
        self.epsilon = 1e-10
        
    def entropy(self, p):
        p = np.clip(p, self.epsilon, 1)
        return -np.sum(p * np.log2(p))
    
    def kl_divergence(self, p, q):
        p = np.clip(p, self.epsilon, 1)
        q = np.clip(q, self.epsilon, 1)
        return np.sum(p * np.log2(p/q))
    
    def mutual_information(self, joint_prob, marginal_x, marginal_y):
        mutual_info = 0
        for i, px in enumerate(marginal_x):
            for j, py in enumerate(marginal_y):
                pxy = joint_prob[i][j]
                if pxy > 0:
                    mutual_info += pxy * np.log2(pxy / (px * py))
        return mutual_info

# Example usage
it = InformationTheory()
p = np.array([0.3, 0.7])
q = np.array([0.5, 0.5])
print(f"Entropy of p: {it.entropy(p):.4f}")
print(f"KL divergence: {it.kl_divergence(p, q):.4f}")
```

Slide 10: Matrix Decompositions for Dimensionality Reduction

Matrix decomposition techniques are fundamental for feature extraction and dimensionality reduction in ML. This implementation shows SVD and eigendecomposition from scratch.

```python
class MatrixDecomposition:
    def __init__(self, matrix):
        self.matrix = np.array(matrix)
        
    def power_iteration(self, num_iterations=100):
        n = self.matrix.shape[0]
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)
        
        for _ in range(num_iterations):
            v_new = np.dot(self.matrix, v)
            v_new = v_new / np.linalg.norm(v_new)
            v = v_new
            
        eigenvalue = np.dot(np.dot(v, self.matrix), v)
        return eigenvalue, v
    
    def svd(self, k=None):
        if k is None:
            k = min(self.matrix.shape)
            
        # Compute eigendecomposition of M^T M
        MTM = np.dot(self.matrix.T, self.matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(MTM)
        
        # Sort eigenvalues and eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute singular values and right singular vectors
        singular_values = np.sqrt(eigenvalues[:k])
        V = eigenvectors[:, :k]
        
        # Compute left singular vectors
        U = np.zeros((self.matrix.shape[0], k))
        for i in range(k):
            if singular_values[i] > 1e-10:
                U[:, i] = np.dot(self.matrix, V[:, i]) / singular_values[i]
                
        return U, singular_values, V.T

# Example usage
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
md = MatrixDecomposition(matrix)
U, s, VT = md.svd(k=2)
print(f"Singular values: {s}")
```

Slide 11: Practical Implementation of Regularization Techniques

Regularization is crucial for preventing overfitting in machine learning models. This implementation demonstrates various regularization methods and their effects on model training.

```python
class Regularization:
    def __init__(self, model_params):
        self.params = model_params
        
    def l1_penalty(self, lambda_param=0.01):
        """Lasso regularization"""
        penalty = lambda_param * np.sum(np.abs(self.params))
        gradient = lambda_param * np.sign(self.params)
        return penalty, gradient
    
    def l2_penalty(self, lambda_param=0.01):
        """Ridge regularization"""
        penalty = 0.5 * lambda_param * np.sum(self.params ** 2)
        gradient = lambda_param * self.params
        return penalty, gradient
    
    def elastic_net(self, lambda_param=0.01, l1_ratio=0.5):
        """Elastic Net regularization"""
        l1_pen, l1_grad = self.l1_penalty(lambda_param * l1_ratio)
        l2_pen, l2_grad = self.l2_penalty(lambda_param * (1 - l1_ratio))
        return l1_pen + l2_pen, l1_grad + l2_grad
    
    def dropout(self, layer_output, dropout_rate=0.5, training=True):
        if not training:
            return layer_output
        
        mask = np.random.binomial(1, 1-dropout_rate, size=layer_output.shape)
        return layer_output * mask / (1 - dropout_rate)

# Example usage
params = np.array([0.5, -0.3, 0.8, -0.1])
reg = Regularization(params)
penalty, gradient = reg.elastic_net()
print(f"Elastic Net penalty: {penalty:.4f}")
print(f"Elastic Net gradient: {gradient}")
```

Slide 12: Advanced Optimization Techniques

Advanced optimization methods are essential for training deep neural networks efficiently. This implementation covers momentum-based optimizers and adaptive learning rate methods.

```python
class AdvancedOptimizer:
    def __init__(self):
        self.states = {}
        
    def rmsprop(self, params, grads, learning_rate=0.001, decay_rate=0.9):
        if 'cache' not in self.states:
            self.states['cache'] = np.zeros_like(params)
            
        cache = self.states['cache']
        cache = decay_rate * cache + (1 - decay_rate) * grads**2
        self.states['cache'] = cache
        
        update = -learning_rate * grads / (np.sqrt(cache) + 1e-8)
        return params + update
        
    def adamw(self, params, grads, learning_rate=0.001, beta1=0.9, 
             beta2=0.999, weight_decay=0.01, t=1):
        if 'm' not in self.states:
            self.states['m'] = np.zeros_like(params)
            self.states['v'] = np.zeros_like(params)
            
        m, v = self.states['m'], self.states['v']
        
        # Weight decay
        params = params * (1 - weight_decay * learning_rate)
        
        # Momentum and RMSprop updates
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * grads**2
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        self.states['m'], self.states['v'] = m, v
        return params - learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

# Example usage
optimizer = AdvancedOptimizer()
params = np.array([0.1, -0.2, 0.3])
grads = np.array([0.01, -0.02, 0.03])

updated_params = optimizer.adamw(params, grads, t=1)
print(f"Original params: {params}")
print(f"Updated params: {updated_params}")
```

Slide 13: Statistical Tests for Model Evaluation

Understanding statistical significance in model performance is crucial for reliable ML systems. This implementation covers essential statistical tests for model comparison.

```python
class ModelEvaluation:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        
    def mcnemar_test(self, model1_predictions, model2_predictions, true_labels):
        """McNemar's test for comparing two ML models"""
        n01 = np.sum((model1_predictions != true_labels) & 
                     (model2_predictions == true_labels))
        n10 = np.sum((model1_predictions == true_labels) & 
                     (model2_predictions != true_labels))
        
        statistic = (abs(n01 - n10) - 1)**2 / (n01 + n10)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        return statistic, p_value, p_value < self.alpha
    
    def bootstrap_confidence_interval(self, metric_func, predictions, 
                                    true_labels, n_bootstrap=1000):
        n_samples = len(true_labels)
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            indices = np.random.randint(0, n_samples, size=n_samples)
            score = metric_func(predictions[indices], true_labels[indices])
            bootstrap_scores.append(score)
            
        confidence_interval = np.percentile(bootstrap_scores, [2.5, 97.5])
        return confidence_interval

# Example usage
evaluator = ModelEvaluation()
model1_preds = np.array([1, 0, 1, 1, 0])
model2_preds = np.array([1, 0, 0, 1, 1])
true_labels = np.array([1, 0, 1, 1, 1])

statistic, p_value, is_significant = evaluator.mcnemar_test(
    model1_preds, model2_preds, true_labels
)
print(f"McNemar's test p-value: {p_value:.4f}")
print(f"Statistically significant: {is_significant}")
```

Slide 14: Additional Resources

*   Advanced Neural Networks and Deep Learning
    *   [https://arxiv.org/abs/1404.7828](https://arxiv.org/abs/1404.7828) "Deep Learning in Neural Networks: An Overview"
    *   [https://arxiv.org/abs/1506.02078](https://arxiv.org/abs/1506.02078) "Batch Normalization: Accelerating Deep Network Training"
    *   [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980) "Adam: A Method for Stochastic Optimization"
*   Statistical Learning and Optimization
    *   [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747) "An Overview of Statistical Learning Theory"
    *   [https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) "Adaptive Subgradient Methods"
*   Practical Machine Learning
    *   [https://scikit-learn.org/stable/tutorial/](https://scikit-learn.org/stable/tutorial/)
    *   [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
    *   [https://www.tensorflow.org/tutorials](https://www.tensorflow.org/tutorials)

