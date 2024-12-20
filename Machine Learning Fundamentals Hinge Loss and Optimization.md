x## Machine Learning Fundamentals Hinge Loss and Optimization
Slide 1: Understanding Hinge Loss Implementation

Hinge loss is a fundamental loss function used in Support Vector Machines and other classification tasks. It penalizes predictions based on their distance from the decision boundary, promoting better margin separation between classes. The mathematical formulation enforces a margin while allowing for soft classification boundaries.

```python
import numpy as np
import matplotlib.pyplot as plt

def hinge_loss(predictions, true_labels):
    """
    Compute hinge loss for binary classification
    Args:
        predictions: Model predictions
        true_labels: Ground truth labels (-1 or 1)
    Returns:
        loss: Average hinge loss
    """
    # Calculate max(0, 1 - y * f(x))
    losses = np.maximum(0, 1 - true_labels * predictions)
    return np.mean(losses)

# Example usage
X = np.random.randn(100, 2)  # Features
y = 2 * (X[:, 0] + X[:, 1] > 0) - 1  # Labels: -1 or 1
predictions = X[:, 0] + X[:, 1]  # Simple linear prediction

loss = hinge_loss(predictions, y)
print(f"Hinge Loss: {loss:.4f}")

# Visualize hinge loss
margins = np.linspace(-2, 2, 100)
losses = [hinge_loss(np.array([m]), np.array([1])) for m in margins]

plt.plot(margins, losses)
plt.xlabel('Margin')
plt.ylabel('Loss')
plt.title('Hinge Loss Function')
plt.grid(True)
```

Slide 2: Constituency Parsing with NLTK

Constituency parsing reveals the hierarchical structure of sentences by decomposing them into nested constituents. This implementation demonstrates basic constituency parsing using NLTK's built-in parser and visualization tools, essential for understanding syntactic relationships.

```python
import nltk
from nltk import Tree
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def constituency_parse(sentence):
    """
    Perform constituency parsing on input sentence
    Args:
        sentence: Input text string
    Returns:
        parse_tree: NLTK Tree object
    """
    # Tokenize and POS tag
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    
    # Define a simple grammar
    grammar = nltk.CFG.fromstring("""
        S -> NP VP
        NP -> DT NN | DT NNS | PRP
        VP -> VB NP | VBZ NP
        DT -> 'the' | 'a'
        NN -> 'cat' | 'dog'
        NNS -> 'cats' | 'dogs'
        VB -> 'chase'
        VBZ -> 'chases'
        PRP -> 'they'
    """)
    
    # Create parser
    parser = nltk.ChartParser(grammar)
    
    # Get first parse tree
    trees = list(parser.parse(tokens))
    if trees:
        return trees[0]
    return None

# Example usage
sentence = "the cat chases the dog"
tree = constituency_parse(sentence)

# Visualize parse tree
if tree:
    tree.draw()
```

Slide 3: Gradient-based Convergence Analysis

Convergence in machine learning refers to the optimization process reaching a stable minimum in the loss landscape. This implementation visualizes the convergence behavior of gradient descent, showing how different learning rates affect the optimization trajectory and stability.

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent_visualization(learning_rates=[0.1, 0.01, 0.001]):
    """
    Visualize convergence for different learning rates
    Args:
        learning_rates: List of learning rates to test
    """
    # Define simple quadratic function
    def f(x): return x**2
    def df(x): return 2*x
    
    x_start = 5
    n_iterations = 50
    
    plt.figure(figsize=(12, 6))
    x = np.linspace(-5, 5, 100)
    plt.plot(x, f(x), 'k-', label='Loss Surface')
    
    for lr in learning_rates:
        x_history = [x_start]
        current_x = x_start
        
        for _ in range(n_iterations):
            current_x = current_x - lr * df(current_x)
            x_history.append(current_x)
        
        plt.plot(x_history, [f(x) for x in x_history], 
                'o-', label=f'LR = {lr}')
    
    plt.xlabel('Parameter Value')
    plt.ylabel('Loss')
    plt.title('Gradient Descent Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()
    
gradient_descent_visualization()
```

Slide 4: Implementing Custom SVM with Hinge Loss

Support Vector Machine implementation from scratch using hinge loss optimization. This implementation demonstrates core concepts including margin maximization, gradient computation, and iterative optimization for binary classification tasks.

```python
import numpy as np
from sklearn.datasets import make_classification

class CustomSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]
                    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)

# Generate synthetic dataset
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, 
                         n_clusters_per_class=1, n_redundant=0)
y = np.where(y <= 0, -1, 1)

# Train model
svm = CustomSVM()
svm.fit(X, y)

# Make predictions
predictions = svm.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 5: Advanced Constituency Parsing with Deep Learning

Natural language processing has evolved to incorporate deep learning for more accurate constituency parsing. This implementation uses a simplified neural network approach to learn syntactic structures from annotated sentences.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ConstituentParser(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output

# Example parameters
VOCAB_SIZE = 1000
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2

# Initialize model
model = ConstituentParser(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)

# Example training loop
def train_step(model, optimizer, input_seq, target_seq):
    optimizer.zero_grad()
    output = model(input_seq)
    loss = nn.CrossEntropyLoss()(output.view(-1, VOCAB_SIZE), target_seq.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

# Example usage
optimizer = optim.Adam(model.parameters())
dummy_input = torch.randint(0, VOCAB_SIZE, (1, 10))  # Batch size 1, sequence length 10
dummy_target = torch.randint(0, VOCAB_SIZE, (1, 10))
loss = train_step(model, optimizer, dummy_input, dummy_target)
print(f"Training Loss: {loss:.4f}")
```

Slide 6: Convergence Analysis with Learning Rate Scheduling

Learning rate scheduling is crucial for optimal convergence in deep learning models. This implementation demonstrates various scheduling techniques and their impact on model convergence through visualization and performance metrics.

```python
import numpy as np
import matplotlib.pyplot as plt

class LRScheduler:
    def __init__(self, initial_lr):
        self.initial_lr = initial_lr
        
    def step_decay(self, epoch, drop=0.5, epochs_drop=10):
        """Step decay learning rate schedule"""
        return self.initial_lr * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    
    def exponential_decay(self, epoch, decay_rate=0.95):
        """Exponential decay learning rate schedule"""
        return self.initial_lr * np.power(decay_rate, epoch)
    
    def cosine_decay(self, epoch, total_epochs):
        """Cosine annealing learning rate schedule"""
        return self.initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

def plot_lr_schedules():
    epochs = np.arange(0, 100)
    scheduler = LRScheduler(initial_lr=0.1)
    
    # Calculate learning rates for each schedule
    step_lrs = [scheduler.step_decay(epoch) for epoch in epochs]
    exp_lrs = [scheduler.exponential_decay(epoch) for epoch in epochs]
    cos_lrs = [scheduler.cosine_decay(epoch, 100) for epoch in epochs]
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, step_lrs, label='Step Decay')
    plt.plot(epochs, exp_lrs, label='Exponential Decay')
    plt.plot(epochs, cos_lrs, label='Cosine Annealing')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Scheduling Methods')
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualize different learning rate schedules
plot_lr_schedules()
```

Slide 7: Real-world Application - Sentiment Analysis with SVM

This implementation demonstrates a complete sentiment analysis pipeline using SVM with hinge loss, including text preprocessing, feature extraction, and model evaluation on real-world movie review data.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SentimentSVM:
    def __init__(self, lr=0.001, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.vectorizer = TfidfVectorizer(max_features=5000)
        
    def preprocess_data(self, texts, labels):
        # Convert text to TF-IDF features
        X = self.vectorizer.fit_transform(texts).toarray()
        y = np.array(labels)
        return X, y
    
    def train(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.w) - self.b
            # Compute gradients using hinge loss
            mask = y * y_pred < 1
            dw = self.w - self.lr * np.dot(X[mask].T, -y[mask])
            db = self.lr * np.sum(y[mask])
            
            self.w -= dw
            self.b -= db
            
    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)

# Example usage with sample data
reviews = [
    "This movie was fantastic and entertaining",
    "Terrible waste of time, very disappointing",
    "Great performances by all actors",
    "Boring plot and poor acting"
]
labels = [1, -1, 1, -1]  # 1 for positive, -1 for negative

# Split data
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2)

# Train model
svm = SentimentSVM()
X_train_feat, y_train = svm.preprocess_data(X_train, y_train)
X_test_feat = svm.vectorizer.transform(X_test).toarray()

svm.train(X_train_feat, y_train)
predictions = svm.predict(X_test_feat)

# Print results
print(classification_report(y_test, predictions))
```

Slide 8: Constituency Parsing Evaluation Metrics

Understanding parsing accuracy requires specific evaluation metrics. This implementation showcases various metrics including bracketing F1 score, crossing brackets, and labeled attachment score for constituency parsing evaluation.

```python
import numpy as np
from collections import defaultdict

class ParsingEvaluator:
    def __init__(self):
        self.metrics = defaultdict(float)
        
    def compute_bracketing_statistics(self, gold_tree, pred_tree):
        """
        Compute bracketing precision, recall, and F1 score
        Args:
            gold_tree: List of gold standard brackets
            pred_tree: List of predicted brackets
        Returns:
            dict: Evaluation metrics
        """
        gold_spans = set(self._get_spans(gold_tree))
        pred_spans = set(self._get_spans(pred_tree))
        
        correct = len(gold_spans.intersection(pred_spans))
        total_gold = len(gold_spans)
        total_pred = len(pred_spans)
        
        precision = correct / total_pred if total_pred > 0 else 0
        recall = correct / total_gold if total_gold > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _get_spans(self, tree):
        """Extract spans from a tree structure"""
        spans = []
        def extract_spans(node, start):
            if isinstance(node, tuple):
                label, children = node
                current_pos = start
                child_spans = []
                for child in children:
                    span, end = extract_spans(child, current_pos)
                    child_spans.extend(span)
                    current_pos = end
                spans.append((start, current_pos, label))
                return spans, current_pos
            else:
                return [], start + 1
        extract_spans(tree, 0)
        return spans

# Example usage
gold_tree = ('S', [('NP', ['The', 'cat']), ('VP', ['chased', ('NP', ['the', 'mouse'])])])
pred_tree = ('S', [('NP', ['The', 'cat']), ('VP', ['chased', 'the', 'mouse'])])

evaluator = ParsingEvaluator()
metrics = evaluator.compute_bracketing_statistics(gold_tree, pred_tree)

for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")
```

Slide 9: Convergence Monitoring System

This implementation creates a comprehensive convergence monitoring system that tracks various metrics during model training, helping identify convergence issues and optimize training parameters.

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class ConvergenceMonitor:
    def __init__(self, window_size=50, tolerance=1e-5):
        self.window_size = window_size
        self.tolerance = tolerance
        self.loss_history = []
        self.grad_norm_history = []
        self.rolling_mean = deque(maxlen=window_size)
        
    def update(self, loss, gradients):
        """
        Update monitoring metrics
        Args:
            loss: Current loss value
            gradients: Current parameter gradients
        Returns:
            bool: True if converged
        """
        self.loss_history.append(loss)
        grad_norm = np.linalg.norm(gradients)
        self.grad_norm_history.append(grad_norm)
        self.rolling_mean.append(loss)
        
        if len(self.rolling_mean) == self.window_size:
            std = np.std(list(self.rolling_mean))
            return std < self.tolerance
        return False
    
    def plot_metrics(self):
        """Visualize convergence metrics"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot loss history
        ax1.plot(self.loss_history)
        ax1.set_title('Loss History')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot gradient norm history
        ax2.plot(self.grad_norm_history)
        ax2.set_title('Gradient Norm History')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Norm')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
monitor = ConvergenceMonitor()

# Simulate training loop
for i in range(1000):
    fake_loss = 1.0 / (i + 1) + np.random.normal(0, 0.01)
    fake_gradients = np.random.randn(10) / (i + 1)
    
    if monitor.update(fake_loss, fake_gradients):
        print(f"Converged at iteration {i}")
        break

monitor.plot_metrics()
```

Slide 10: Mathematical Foundations of Hinge Loss

This implementation focuses on the mathematical derivation and visualization of hinge loss, including its gradient computation and geometric interpretation in the context of SVM optimization.

```python
import numpy as np
import matplotlib.pyplot as plt

class HingeLossMath:
    def __init__(self):
        """
        Mathematical formulation of hinge loss:
        Code block containing LaTeX formulas (not rendered):
        
        $$L(y, f(x)) = \max(0, 1 - y f(x))$$
        $$\frac{\partial L}{\partial f(x)} = \begin{cases} 
        -y & \text{if } y f(x) < 1 \\
        0 & \text{otherwise}
        \end{cases}$$
        """
        pass
    
    def compute_loss_gradient(self, y_true, y_pred):
        """Compute hinge loss and its gradient"""
        margins = y_true * y_pred
        loss = np.maximum(0, 1 - margins)
        gradients = np.where(margins < 1, -y_true, 0)
        return loss, gradients
    
    def visualize_loss_surface(self):
        """Visualize hinge loss surface"""
        y_pred = np.linspace(-3, 3, 100)
        y_true = np.array([1, -1])  # Consider both positive and negative cases
        
        plt.figure(figsize=(10, 6))
        for yt in y_true:
            loss, _ = self.compute_loss_gradient(yt, y_pred)
            plt.plot(y_pred, loss, label=f'y_true = {yt}')
        
        plt.xlabel('Prediction')
        plt.ylabel('Loss')
        plt.title('Hinge Loss Surface')
        plt.legend()
        plt.grid(True)
        plt.show()

# Demonstration
hlm = HingeLossMath()
hlm.visualize_loss_surface()

# Example with specific values
y_true = np.array([1, -1, 1])
y_pred = np.array([0.5, -0.8, 1.2])
loss, grad = hlm.compute_loss_gradient(y_true, y_pred)
print(f"Loss values: {loss}")
print(f"Gradients: {grad}")
```

Slide 11: Advanced Constituency Tree Visualization

This implementation creates an advanced visualization system for constituency parse trees, incorporating color coding, node probabilities, and interactive features for better analysis of parsing results.

```python
import numpy as np
import graphviz
from IPython.display import display

class TreeVisualizer:
    def __init__(self):
        self.dot = graphviz.Digraph()
        self.node_count = 0
        
    def create_tree_visualization(self, tree, probabilities=None):
        """
        Create interactive visualization of parse tree
        Args:
            tree: Parsed tree structure
            probabilities: Optional node probabilities
        """
        self.dot = graphviz.Digraph(comment='Constituency Parse Tree')
        self.dot.attr(rankdir='TB')
        self.node_count = 0
        
        def add_node(node, parent_id=None):
            node_id = str(self.node_count)
            self.node_count += 1
            
            # Node styling
            if isinstance(node, tuple):
                label, children = node
                prob = probabilities.get(label, 1.0) if probabilities else 1.0
                color = self._get_probability_color(prob)
                self.dot.node(node_id, label, style='filled', fillcolor=color)
                
                if parent_id is not None:
                    self.dot.edge(parent_id, node_id)
                
                for child in children:
                    add_node(child, node_id)
            else:
                self.dot.node(node_id, str(node), shape='box')
                if parent_id is not None:
                    self.dot.edge(parent_id, node_id)
        
        add_node(tree)
        return self.dot
    
    def _get_probability_color(self, prob):
        """Generate color based on probability"""
        r = int(255 * (1 - prob))
        g = int(255 * prob)
        return f"#{r:02x}{g:02x}ff"

# Example usage
example_tree = (
    'S', [
        ('NP', [('DET', ['The']), ('NN', ['cat'])]),
        ('VP', [('VB', ['chases']), ('NP', [('DET', ['the']), ('NN', ['mouse'])])])
    ]
)

# Sample probabilities for nodes
probabilities = {
    'S': 0.95,
    'NP': 0.85,
    'VP': 0.90,
    'DET': 0.99,
    'NN': 0.95,
    'VB': 0.92
}

visualizer = TreeVisualizer()
tree_viz = visualizer.create_tree_visualization(example_tree, probabilities)
display(tree_viz)
```

Slide 12: Convergence Analysis with Information Theory

This implementation explores convergence through the lens of information theory, measuring entropy and mutual information during model training to provide deeper insights into the learning process.

```python
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

class InformationTheoreticConvergence:
    def __init__(self):
        self.entropy_history = []
        self.mi_history = []
        
    def compute_entropy(self, probabilities):
        """Compute Shannon entropy of probability distribution"""
        return entropy(probabilities)
    
    def compute_mutual_information(self, joint_prob, margin_x, margin_y):
        """Compute mutual information between variables"""
        mi = 0
        for i in range(len(margin_x)):
            for j in range(len(margin_y)):
                if joint_prob[i][j] > 0:
                    mi += joint_prob[i][j] * np.log2(
                        joint_prob[i][j] / (margin_x[i] * margin_y[j])
                    )
        return mi
    
    def analyze_convergence(self, predictions, targets, n_epochs):
        """Analyze convergence using information theory metrics"""
        for epoch in range(n_epochs):
            # Compute probability distributions
            pred_prob = np.abs(predictions) / np.sum(np.abs(predictions))
            target_prob = np.abs(targets) / np.sum(np.abs(targets))
            
            # Compute joint probability matrix
            joint_prob = np.outer(pred_prob, target_prob)
            
            # Calculate metrics
            h_pred = self.compute_entropy(pred_prob)
            mi = self.compute_mutual_information(joint_prob, pred_prob, target_prob)
            
            self.entropy_history.append(h_pred)
            self.mi_history.append(mi)
            
            # Update predictions (simulated training)
            predictions = predictions * 0.95 + targets * 0.05
        
        self.plot_metrics()
    
    def plot_metrics(self):
        """Plot information theoretic metrics"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        epochs = range(len(self.entropy_history))
        
        ax1.plot(epochs, self.entropy_history)
        ax1.set_title('Prediction Entropy Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Entropy')
        ax1.grid(True)
        
        ax2.plot(epochs, self.mi_history)
        ax2.set_title('Mutual Information Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mutual Information')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
analyzer = InformationTheoreticConvergence()
initial_predictions = np.random.randn(10)
targets = np.random.randn(10)
analyzer.analyze_convergence(initial_predictions, targets, n_epochs=100)
```

Slide 13: Real-world Application - Named Entity Recognition with Parse Trees

This implementation demonstrates a practical application of constituency parsing for named entity recognition, combining syntactic analysis with statistical methods for entity detection and classification.

```python
import numpy as np
from collections import defaultdict

class NERParser:
    def __init__(self):
        self.entity_patterns = defaultdict(list)
        self.constituent_probs = defaultdict(float)
        
    def train(self, parsed_sentences, entity_annotations):
        """
        Train NER model using constituency parse trees
        Args:
            parsed_sentences: List of constituency parsed trees
            entity_annotations: Corresponding entity labels
        """
        for tree, entities in zip(parsed_sentences, entity_annotations):
            self._extract_patterns(tree, entities)
            
    def _extract_patterns(self, tree, entities):
        """Extract syntactic patterns for entity detection"""
        def traverse(node, path):
            if isinstance(node, tuple):
                label, children = node
                # Update constituent probabilities
                if any(entity in ' '.join(self._get_leaves(node)) for entity in entities):
                    self.constituent_probs[label] += 1
                    self.entity_patterns[label].append(path)
                
                for i, child in enumerate(children):
                    traverse(child, path + [label])
                    
        traverse(tree, [])
        
    def _get_leaves(self, node):
        """Extract leaf nodes (words) from tree"""
        if isinstance(node, tuple):
            _, children = node
            leaves = []
            for child in children:
                leaves.extend(self._get_leaves(child))
            return leaves
        return [node]
    
    def predict(self, parsed_sentence):
        """
        Predict named entities in new sentence
        Args:
            parsed_sentence: Constituency parsed tree
        Returns:
            list: Predicted entities with their types
        """
        entities = []
        
        def find_entities(node, path):
            if isinstance(node, tuple):
                label, children = node
                # Check if current subtree matches entity patterns
                if label in self.entity_patterns:
                    for pattern in self.entity_patterns[label]:
                        if self._matches_pattern(path, pattern):
                            text = ' '.join(self._get_leaves(node))
                            prob = self.constituent_probs[label]
                            entities.append((text, label, prob))
                
                for child in children:
                    find_entities(child, path + [label])
                    
        find_entities(parsed_sentence, [])
        return entities
    
    def _matches_pattern(self, path1, path2):
        """Check if two paths match"""
        return len(path1) >= len(path2) and path1[:len(path2)] == path2

# Example usage
example_trees = [
    ('S', [
        ('NP', [('NNP', ['John']), ('NNP', ['Smith'])]),
        ('VP', [('VBZ', ['works']), ('PP', [('IN', ['at']), 
                ('NP', [('NNP', ['Google'])])])])
    ])
]

example_entities = [['John Smith', 'Google']]

# Train model
ner = NERParser()
ner.train(example_trees, example_entities)

# Make prediction
new_sentence = ('S', [
    ('NP', [('NNP', ['Mary']), ('NNP', ['Johnson'])]),
    ('VP', [('VBZ', ['visited']), ('NP', [('NNP', ['Microsoft'])])])
])

predictions = ner.predict(new_sentence)
for entity, label, prob in predictions:
    print(f"Entity: {entity}")
    print(f"Type: {label}")
    print(f"Confidence: {prob:.2f}\n")
```

Slide 14: Additional Resources

*   Theoretical Foundations of Support Vector Machines and Hinge Loss
    *   [https://arxiv.org/abs/1707.02061](https://arxiv.org/abs/1707.02061) "Understanding Support Vector Machine Classification using Hinge Loss Optimization"
*   Recent Advances in Constituency Parsing
    *   [https://arxiv.org/abs/2003.08907](https://arxiv.org/abs/2003.08907) "Neural Constituency Parsing: A Survey"
*   Convergence Analysis in Deep Learning
    *   [https://arxiv.org/abs/1908.08293](https://arxiv.org/abs/1908.08293) "On the Convergence of Deep Learning with Dynamic Training"
*   Recommended searches:
    *   "constituency parsing transformers"
    *   "hinge loss optimization techniques"
    *   "convergence analysis neural networks"
    *   "named entity recognition with parse trees"

