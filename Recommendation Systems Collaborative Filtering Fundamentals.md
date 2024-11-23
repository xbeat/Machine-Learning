## Recommendation Systems Collaborative Filtering Fundamentals
Slide 1: Collaborative Filtering Fundamentals

Collaborative filtering is the most widely used recommendation system approach, leveraging user-item interactions to identify patterns and similarities. It assumes that users who agreed in their evaluation of certain items are likely to agree again in the future.

```python
import numpy as np
from typing import Dict, List

class CollaborativeFiltering:
    def __init__(self, ratings_matrix: Dict[int, Dict[int, float]]):
        self.ratings = ratings_matrix
        self.n_users = max(ratings_matrix.keys()) + 1
        self.n_items = max(max(d.keys()) for d in ratings_matrix.values()) + 1
        
    def compute_similarity(self, user1: int, user2: int) -> float:
        # Get items rated by both users
        common_items = set(self.ratings[user1].keys()) & set(self.ratings[user2].keys())
        if not common_items:
            return 0.0
        
        # Calculate Pearson correlation
        ratings1 = [self.ratings[user1][item] for item in common_items]
        ratings2 = [self.ratings[user2][item] for item in common_items]
        
        return np.corrcoef(ratings1, ratings2)[0, 1]

# Example usage
ratings = {
    0: {0: 5, 1: 3, 2: 4},
    1: {0: 3, 1: 1, 2: 2},
    2: {1: 4, 2: 4, 3: 5}
}

cf = CollaborativeFiltering(ratings)
similarity = cf.compute_similarity(0, 1)
print(f"Similarity between user 0 and 1: {similarity:.2f}")
```

Slide 2: Matrix Factorization for Recommendations

Matrix factorization decomposes the user-item interaction matrix into lower dimensional latent factor matrices, capturing hidden relationships between users and items through these learned representations.

```python
import numpy as np

class MatrixFactorization:
    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01, regularization: float = 0.02):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
    
    def train(self, ratings: np.ndarray, n_epochs: int = 20):
        n_users, n_items = ratings.shape
        
        # Initialize latent factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        mask = (ratings > 0).astype(float)
        
        for epoch in range(n_epochs):
            error = mask * (ratings - self.user_factors @ self.item_factors.T)
            
            # Update factors
            user_grads = -error @ self.item_factors + self.reg * self.user_factors
            item_grads = -error.T @ self.user_factors + self.reg * self.item_factors
            
            self.user_factors -= self.lr * user_grads
            self.item_factors -= self.lr * item_grads
            
            rmse = np.sqrt(np.sum(error ** 2) / np.sum(mask))
            print(f"Epoch {epoch + 1}, RMSE: {rmse:.4f}")
    
    def predict(self) -> np.ndarray:
        return self.user_factors @ self.item_factors.T

# Example usage
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4]
])

mf = MatrixFactorization(n_factors=3)
mf.train(ratings)
predictions = mf.predict()
print("\nPredicted ratings:\n", predictions.round(2))
```

Slide 3: Content-Based Filtering Implementation

Content-based filtering analyzes item attributes to recommend similar items. This approach creates item profiles using features and matches them with user preferences, making it particularly effective for domains with rich item metadata.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ContentBasedRecommender:
    def __init__(self, items_df: pd.DataFrame):
        self.items_df = items_df
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # Create item profiles using TF-IDF
        self.item_profiles = self.tfidf.fit_transform(items_df['description'])
        self.similarity_matrix = cosine_similarity(self.item_profiles)
        
    def recommend(self, item_id: int, n_recommendations: int = 5) -> pd.DataFrame:
        # Get similarity scores for the item
        similarities = self.similarity_matrix[item_id]
        
        # Get top similar items (excluding itself)
        similar_indices = similarities.argsort()[::-1][1:n_recommendations+1]
        similar_items = self.items_df.iloc[similar_indices].copy()
        similar_items['similarity_score'] = similarities[similar_indices]
        
        return similar_items[['title', 'similarity_score']]

# Example usage
items_data = {
    'title': ['Movie 1', 'Movie 2', 'Movie 3', 'Movie 4'],
    'description': [
        'Action movie with supernatural elements',
        'Romantic comedy about modern dating',
        'Action-packed superhero adventure',
        'Drama about family relationships'
    ]
}

items_df = pd.DataFrame(items_data)
recommender = ContentBasedRecommender(items_df)
recommendations = recommender.recommend(0)
print("Recommendations for Movie 1:\n", recommendations)
```

Slide 4: Deep Learning-Based Recommender System

Neural networks have revolutionized recommendation systems by capturing complex non-linear relationships between users and items. This implementation demonstrates a basic deep learning recommender using PyTorch with embedding layers.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DeepRecommender(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int = 50):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        
        self.layers = nn.Sequential(
            nn.Linear(n_factors * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        x = torch.cat([user_embeds, item_embeds], dim=1)
        return self.layers(x).squeeze()

# Example usage
n_users, n_items = 1000, 500
model = DeepRecommender(n_users, n_items)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Training example
user_ids = torch.randint(0, n_users, (64,))
item_ids = torch.randint(0, n_items, (64,))
labels = torch.randint(0, 2, (64,)).float()

# Single training step
optimizer.zero_grad()
predictions = model(user_ids, item_ids)
loss = criterion(predictions, labels)
loss.backward()
optimizer.step()

print(f"Training loss: {loss.item():.4f}")
```

Slide 5: Time-Aware Collaborative Filtering

Time-aware recommendations incorporate temporal dynamics into the recommendation process, recognizing that user preferences and item relevance change over time. This implementation uses time decay factors.

```python
import numpy as np
from datetime import datetime
from typing import Dict, Tuple

class TimeAwareCollaborativeFiltering:
    def __init__(self, time_decay_factor: float = 0.1):
        self.decay_factor = time_decay_factor
        self.ratings: Dict[Tuple[int, int], Tuple[float, datetime]] = {}
        
    def add_rating(self, user_id: int, item_id: int, rating: float, 
                  timestamp: datetime) -> None:
        self.ratings[(user_id, item_id)] = (rating, timestamp)
    
    def compute_time_weight(self, timestamp: datetime, 
                          current_time: datetime) -> float:
        time_diff = (current_time - timestamp).days
        return np.exp(-self.decay_factor * time_diff)
    
    def predict_rating(self, target_user: int, target_item: int, 
                      current_time: datetime) -> float:
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for (user_id, item_id), (rating, timestamp) in self.ratings.items():
            if item_id == target_item and user_id != target_user:
                time_weight = self.compute_time_weight(timestamp, current_time)
                weighted_sum += rating * time_weight
                weight_sum += time_weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

# Example usage
recommender = TimeAwareCollaborativeFiltering()

# Add some ratings with timestamps
current_time = datetime(2024, 1, 1)
recommender.add_rating(1, 1, 5.0, datetime(2023, 12, 1))
recommender.add_rating(2, 1, 4.0, datetime(2023, 12, 15))
recommender.add_rating(3, 1, 3.0, datetime(2023, 11, 1))

# Predict rating for a new user
predicted_rating = recommender.predict_rating(4, 1, current_time)
print(f"Predicted rating: {predicted_rating:.2f}")
```

Slide 6: Hybrid Recommendation System

A hybrid approach combines multiple recommendation techniques to leverage their complementary strengths. This implementation merges collaborative filtering and content-based recommendations using a weighted scheme.

```python
import numpy as np
from sklearn.preprocessing import normalize
from typing import Dict, List, Tuple

class HybridRecommender:
    def __init__(self, cf_weight: float = 0.7, cb_weight: float = 0.3):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.user_ratings: Dict[int, Dict[int, float]] = {}
        self.item_features: Dict[int, np.ndarray] = {}
    
    def add_user_rating(self, user_id: int, item_id: int, 
                       rating: float) -> None:
        if user_id not in self.user_ratings:
            self.user_ratings[user_id] = {}
        self.user_ratings[user_id][item_id] = rating
    
    def add_item_features(self, item_id: int, features: np.ndarray) -> None:
        self.item_features[item_id] = normalize(features.reshape(1, -1))[0]
    
    def get_cf_score(self, user_id: int, item_id: int) -> float:
        if not self.user_ratings.get(user_id):
            return 0.0
            
        similar_ratings = []
        for rated_item, rating in self.user_ratings[user_id].items():
            if rated_item in self.item_features:
                similarity = np.dot(self.item_features[rated_item], 
                                 self.item_features[item_id])
                similar_ratings.append((similarity, rating))
        
        if not similar_ratings:
            return 0.0
            
        weights_sum = sum(sim for sim, _ in similar_ratings)
        if weights_sum == 0:
            return 0.0
            
        weighted_ratings = sum(sim * rating for sim, rating in similar_ratings)
        return weighted_ratings / weights_sum
    
    def get_cb_score(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_ratings or item_id not in self.item_features:
            return 0.0
            
        user_profile = np.zeros_like(self.item_features[item_id])
        n_ratings = 0
        
        for rated_item, rating in self.user_ratings[user_id].items():
            if rated_item in self.item_features:
                user_profile += self.item_features[rated_item] * rating
                n_ratings += 1
        
        if n_ratings == 0:
            return 0.0
            
        user_profile /= n_ratings
        return np.dot(user_profile, self.item_features[item_id])
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        cf_score = self.get_cf_score(user_id, item_id)
        cb_score = self.get_cb_score(user_id, item_id)
        
        return (self.cf_weight * cf_score + 
                self.cb_weight * cb_score)

# Example usage
recommender = HybridRecommender()

# Add some user ratings
recommender.add_user_rating(1, 1, 5.0)
recommender.add_user_rating(1, 2, 3.0)

# Add item features (example with 3 features)
recommender.add_item_features(1, np.array([1.0, 0.5, 0.2]))
recommender.add_item_features(2, np.array([0.8, 0.4, 0.3]))
recommender.add_item_features(3, np.array([0.9, 0.4, 0.2]))

# Predict rating
predicted_rating = recommender.predict_rating(1, 3)
print(f"Predicted rating: {predicted_rating:.2f}")
```

Slide 7: Factorization Machines Implementation

Factorization Machines extend matrix factorization by modeling feature interactions, making them particularly effective for sparse datasets and cold-start scenarios. This implementation showcases a basic FM model using numpy.

```python
import numpy as np

class FactorizationMachine:
    def __init__(self, n_features: int, n_factors: int = 10, 
                 learning_rate: float = 0.01, regularization: float = 0.01):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        
        # Initialize parameters
        self.w0 = 0.0
        self.w = np.zeros(n_features)
        self.V = np.random.normal(0, 0.1, (n_features, n_factors))
        
    def predict(self, x: np.ndarray) -> float:
        # Linear terms
        linear = self.w0 + np.dot(x, self.w)
        
        # Interaction terms
        interactions = 0.0
        for f in range(self.n_factors):
            sum_square = np.sum(self.V[:, f] * x) ** 2
            square_sum = np.sum((self.V[:, f] * x) ** 2)
            interactions += 0.5 * (sum_square - square_sum)
            
        return linear + interactions
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                x = X[i]
                pred = self.predict(x)
                error = pred - y[i]
                total_loss += error ** 2
                
                # Update parameters
                self.w0 -= self.lr * (error + 2 * self.reg * self.w0)
                self.w -= self.lr * (error * x + 2 * self.reg * self.w)
                
                for f in range(self.n_factors):
                    term = x * (np.sum(self.V[:, f] * x))
                    self.V[:, f] -= self.lr * (error * term + 2 * self.reg * self.V[:, f])
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {total_loss/len(X):.4f}")

# Example usage
# Create synthetic data
np.random.seed(42)
n_samples, n_features = 1000, 10
X = np.random.randn(n_samples, n_features)
w_true = np.random.randn(n_features)
y = np.dot(X, w_true) + np.random.randn(n_samples) * 0.1

# Train FM model
fm = FactorizationMachine(n_features=n_features)
fm.fit(X, y)

# Make predictions
test_sample = np.random.randn(n_features)
prediction = fm.predict(test_sample)
print(f"\nPrediction for test sample: {prediction:.4f}")
```

Slide 8: Sequential Recommendation with Recurrent Neural Networks

Sequential recommendation systems model the temporal dynamics of user behavior patterns. This implementation uses a GRU-based neural network to capture sequential dependencies in user interactions.

```python
import torch
import torch.nn as nn
from typing import List, Tuple

class SequentialRecommender(nn.Module):
    def __init__(self, n_items: int, embedding_dim: int = 50, 
                 hidden_dim: int = 100):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, 
                                         padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_items)
        
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence shape: (batch_size, sequence_length)
        embedded = self.item_embedding(sequence)
        output, _ = self.gru(embedded)
        predictions = self.output_layer(output[:, -1, :])
        return torch.softmax(predictions, dim=-1)
    
    def train_step(self, optimizer: torch.optim.Optimizer, 
                  sequences: torch.Tensor, targets: torch.Tensor) -> float:
        self.train()
        optimizer.zero_grad()
        
        predictions = self(sequences)
        loss = nn.CrossEntropyLoss()(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()

# Example usage
def prepare_sequences(interactions: List[List[int]], 
                     seq_length: int) -> List[Tuple[List[int], int]]:
    sequences = []
    for user_seq in interactions:
        for i in range(len(user_seq) - seq_length):
            sequence = user_seq[i:i + seq_length]
            target = user_seq[i + seq_length]
            sequences.append((sequence, target))
    return sequences

# Create synthetic data
n_items = 1000
seq_length = 5
batch_size = 32

# Example interaction sequences
interactions = [
    [1, 4, 2, 7, 3, 9],
    [2, 5, 1, 8, 4, 6],
    [3, 2, 7, 4, 5, 8]
]

sequences = prepare_sequences(interactions, seq_length)
model = SequentialRecommender(n_items)
optimizer = torch.optim.Adam(model.parameters())

# Training example
sequence_tensor = torch.tensor([seq for seq, _ in sequences])
target_tensor = torch.tensor([target for _, target in sequences])

loss = model.train_step(optimizer, sequence_tensor, target_tensor)
print(f"Training loss: {loss:.4f}")

# Generate recommendations
with torch.no_grad():
    test_sequence = torch.tensor([[1, 4, 2, 7, 3]])
    predictions = model(test_sequence)
    top_items = torch.topk(predictions[0], k=5).indices
    print("\nTop 5 recommended items:", top_items.tolist())
```

Slide 9: Multi-Armed Bandit Recommendations

Multi-armed bandit algorithms balance exploration and exploitation in recommendation systems, continuously learning from user feedback while optimizing for engagement. This implementation uses Thompson Sampling.

```python
import numpy as np
from scipy.stats import beta
from typing import Dict, List, Tuple

class BanditRecommender:
    def __init__(self, n_items: int):
        self.n_items = n_items
        # Initialize alpha and beta parameters for Beta distribution
        self.alphas = np.ones(n_items)
        self.betas = np.ones(n_items)
        
    def recommend(self, n_recommendations: int = 1) -> List[int]:
        # Sample from Beta distribution for each item
        samples = [beta.rvs(a, b) for a, b in zip(self.alphas, self.betas)]
        # Return top-n items
        return np.argsort(samples)[-n_recommendations:][::-1]
    
    def update(self, item_id: int, reward: int) -> None:
        # Update Beta distribution parameters based on reward
        self.alphas[item_id] += reward
        self.betas[item_id] += (1 - reward)
        
    def get_item_statistics(self, item_id: int) -> Dict[str, float]:
        total_trials = self.alphas[item_id] + self.betas[item_id] - 2
        success_rate = self.alphas[item_id] - 1 / total_trials if total_trials > 0 else 0
        return {
            'trials': total_trials,
            'success_rate': success_rate,
            'expected_value': self.alphas[item_id] / 
                            (self.alphas[item_id] + self.betas[item_id])
        }

# Example usage
class RecommendationSimulator:
    def __init__(self, n_items: int):
        self.true_item_probs = np.random.beta(2, 2, n_items)
        
    def get_reward(self, item_id: int) -> int:
        return np.random.binomial(1, self.true_item_probs[item_id])

# Run simulation
n_items = 100
n_iterations = 1000
recommender = BanditRecommender(n_items)
simulator = RecommendationSimulator(n_items)

cumulative_reward = 0
rewards_history = []

for i in range(n_iterations):
    # Get recommendation
    recommended_items = recommender.recommend(1)
    item_id = recommended_items[0]
    
    # Simulate user interaction
    reward = simulator.get_reward(item_id)
    cumulative_reward += reward
    
    # Update recommender
    recommender.update(item_id, reward)
    
    # Track performance
    if (i + 1) % 100 == 0:
        avg_reward = cumulative_reward / (i + 1)
        rewards_history.append(avg_reward)
        print(f"Iteration {i+1}, Average Reward: {avg_reward:.4f}")

# Print final statistics for top performing items
top_items = recommender.recommend(5)
print("\nTop 5 items statistics:")
for item_id in top_items:
    stats = recommender.get_item_statistics(item_id)
    print(f"Item {item_id}:")
    print(f"  Trials: {stats['trials']}")
    print(f"  Success rate: {stats['success_rate']:.4f}")
    print(f"  Expected value: {stats['expected_value']:.4f}")
```

Slide 10: Self-Attention Based Recommendation

Self-attention mechanisms have revolutionized sequential recommendation by capturing long-range dependencies and item relationships. This implementation uses a transformer-based architecture.

```python
import torch
import torch.nn as nn
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, v), attention_weights
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, 
                                                           torch.Tensor]:
        batch_size = q.size(0)
        
        q = self.query(q).view(batch_size, -1, self.n_heads, self.d_k)
        k = self.key(k).view(batch_size, -1, self.n_heads, self.d_k)
        v = self.value(v).view(batch_size, -1, self.n_heads, self.d_k)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        x, attention = self.attention(q, k, v, mask)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(x), attention

class SelfAttentionRecommender(nn.Module):
    def __init__(self, n_items: int, d_model: int = 256, n_heads: int = 8):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(1000, d_model)
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.layer_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.output_layer = nn.Linear(d_model, n_items)
        
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(sequence.size(1), device=sequence.device)
        positions = positions.unsqueeze(0).expand(sequence.size(0), -1)
        
        x = self.item_embedding(sequence)
        x = x + self.position_embedding(positions)
        
        # Self-attention
        attended, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attended)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm(x + ff_output)
        
        # Final prediction
        return torch.softmax(self.output_layer(x[:, -1, :]), dim=-1)

# Example usage
n_items = 1000
model = SelfAttentionRecommender(n_items)
optimizer = torch.optim.Adam(model.parameters())

# Example sequence
sequence = torch.randint(0, n_items, (32, 50))  # batch_size=32, seq_length=50
predictions = model(sequence)

print(f"Prediction shape: {predictions.shape}")
print(f"Top-5 recommended items for first sequence: {torch.topk(predictions[0], 5).indices.tolist()}")
```

Slide 11: Context-Aware Recommendation System

Context-aware recommendation systems incorporate situational information such as time, location, and user context to improve recommendation accuracy. This implementation demonstrates a contextual bandit approach.

```python
import numpy as np
from typing import List, Dict, Tuple

class ContextualRecommender:
    def __init__(self, n_items: int, n_context_features: int, 
                 learning_rate: float = 0.01):
        self.n_items = n_items
        self.n_features = n_context_features
        self.lr = learning_rate
        
        # Initialize weights for linear model per item
        self.weights = np.zeros((n_items, n_context_features))
        self.covariance = np.array([np.eye(n_context_features) 
                                  for _ in range(n_items)])
        
    def get_item_score(self, context: np.ndarray, item_id: int) -> Tuple[float, float]:
        mean = context.dot(self.weights[item_id])
        std = np.sqrt(context.dot(self.covariance[item_id]).dot(context))
        return mean, std
    
    def recommend(self, context: np.ndarray, 
                 exploration_weight: float = 1.0) -> List[int]:
        scores = []
        for item_id in range(self.n_items):
            mean, std = self.get_item_score(context, item_id)
            ucb_score = mean + exploration_weight * std
            scores.append(ucb_score)
        
        return np.argsort(scores)[::-1]
    
    def update(self, context: np.ndarray, item_id: int, reward: float) -> None:
        # Update covariance matrix
        self.covariance[item_id] = np.linalg.inv(
            np.linalg.inv(self.covariance[item_id]) + 
            np.outer(context, context)
        )
        
        # Update weights
        error = reward - context.dot(self.weights[item_id])
        self.weights[item_id] += self.lr * self.covariance[item_id].dot(context) * error

class ContextSimulator:
    def __init__(self, n_items: int, n_context_features: int):
        self.true_weights = np.random.randn(n_items, n_context_features)
        self.noise_std = 0.1
        
    def get_reward(self, context: np.ndarray, item_id: int) -> float:
        true_score = context.dot(self.true_weights[item_id])
        return true_score + np.random.normal(0, self.noise_std)

# Example usage
n_items = 100
n_context_features = 5
n_iterations = 1000

recommender = ContextualRecommender(n_items, n_context_features)
simulator = ContextSimulator(n_items, n_context_features)

cumulative_regret = 0
regret_history = []

for i in range(n_iterations):
    # Generate random context
    context = np.random.randn(n_context_features)
    context /= np.linalg.norm(context)  # Normalize context
    
    # Get recommendation
    recommended_items = recommender.recommend(context)
    selected_item = recommended_items[0]
    
    # Simulate reward
    reward = simulator.get_reward(context, selected_item)
    
    # Update model
    recommender.update(context, selected_item, reward)
    
    # Track performance
    best_item_reward = max([simulator.get_reward(context, item) 
                           for item in range(n_items)])
    regret = best_item_reward - reward
    cumulative_regret += regret
    
    if (i + 1) % 100 == 0:
        avg_regret = cumulative_regret / (i + 1)
        regret_history.append(avg_regret)
        print(f"Iteration {i+1}, Average Regret: {avg_regret:.4f}")

# Print final performance statistics
print("\nFinal Performance Statistics:")
print(f"Final Average Regret: {regret_history[-1]:.4f}")
```

Slide 12: Real-world Implementation - E-commerce Recommendation System

This implementation demonstrates a production-ready recommendation system for e-commerce, combining multiple approaches and handling real-world challenges like cold-start and scalability.

```python
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

class EcommerceRecommender:
    def __init__(self, config: Dict):
        self.item_embeddings: Dict[int, np.ndarray] = {}
        self.user_profiles: Dict[int, Dict] = {}
        self.item_metadata: Dict[int, Dict] = {}
        self.popularity_scores: Dict[int, float] = {}
        
        # Configuration
        self.embedding_dim = config.get('embedding_dim', 128)
        self.decay_factor = config.get('time_decay', 0.1)
        self.cold_start_window = config.get('cold_start_items', 50)
        
    def initialize_item(self, item_id: int, 
                       metadata: Dict, 
                       embedding: Optional[np.ndarray] = None) -> None:
        if embedding is None:
            embedding = np.random.randn(self.embedding_dim)
            embedding /= np.linalg.norm(embedding)
            
        self.item_embeddings[item_id] = embedding
        self.item_metadata[item_id] = metadata
        self.popularity_scores[item_id] = 0.0
        
    def update_user_profile(self, user_id: int, 
                          item_id: int, 
                          interaction_type: str,
                          timestamp: datetime) -> None:
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'interactions': [],
                'categories': {},
                'embedding': np.zeros(self.embedding_dim)
            }
        
        # Update interaction history
        self.user_profiles[user_id]['interactions'].append({
            'item_id': item_id,
            'type': interaction_type,
            'timestamp': timestamp
        })
        
        # Update category preferences
        category = self.item_metadata[item_id].get('category')
        if category:
            self.user_profiles[user_id]['categories'][category] = \
                self.user_profiles[user_id]['categories'].get(category, 0) + 1
                
        # Update user embedding
        interaction_weight = 1.0 if interaction_type == 'purchase' else 0.2
        self.user_profiles[user_id]['embedding'] += \
            self.item_embeddings[item_id] * interaction_weight
            
        # Update item popularity
        time_weight = np.exp(-self.decay_factor * 
                           (datetime.now() - timestamp).days)
        self.popularity_scores[item_id] += interaction_weight * time_weight
        
    def get_recommendations(self, user_id: int, 
                          n_recommendations: int = 10,
                          context: Optional[Dict] = None) -> List[Dict]:
        if user_id not in self.user_profiles:
            # Cold-start user: return popular items
            return self._get_popular_recommendations(n_recommendations)
            
        # Combine different signals for ranking
        item_scores = {}
        user_profile = self.user_profiles[user_id]
        
        for item_id in self.item_embeddings:
            # Skip recently interacted items
            if self._is_recent_interaction(user_id, item_id):
                continue
                
            score = self._calculate_item_score(
                item_id, user_profile, context
            )
            item_scores[item_id] = score
            
        # Get top items
        recommended_items = sorted(
            item_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n_recommendations]
        
        return [self._format_recommendation(item_id, score) 
                for item_id, score in recommended_items]
    
    def _calculate_item_score(self, item_id: int, 
                            user_profile: Dict, 
                            context: Optional[Dict]) -> float:
        # Collaborative filtering score
        cf_score = np.dot(
            user_profile['embedding'],
            self.item_embeddings[item_id]
        )
        
        # Category preference score
        category = self.item_metadata[item_id].get('category')
        category_score = user_profile['categories'].get(category, 0)
        
        # Popularity score
        popularity_score = self.popularity_scores[item_id]
        
        # Combine scores
        final_score = (
            0.6 * cf_score +
            0.2 * category_score +
            0.2 * popularity_score
        )
        
        # Apply context boosting if available
        if context:
            final_score *= self._get_context_multiplier(item_id, context)
            
        return final_score
    
    def _format_recommendation(self, item_id: int, score: float) -> Dict:
        return {
            'item_id': item_id,
            'score': score,
            'metadata': self.item_metadata[item_id],
            'popularity': self.popularity_scores[item_id]
        }

# Example usage
config = {
    'embedding_dim': 128,
    'time_decay': 0.1,
    'cold_start_items': 50
}

recommender = EcommerceRecommender(config)

# Initialize some items
for i in range(100):
    recommender.initialize_item(
        item_id=i,
        metadata={
            'category': f'category_{i % 5}',
            'price': np.random.uniform(10, 1000)
        }
    )

# Simulate user interactions
user_id = 1
for i in range(10):
    recommender.update_user_profile(
        user_id=user_id,
        item_id=np.random.randint(100),
        interaction_type='purchase' if np.random.random() > 0.7 else 'view',
        timestamp=datetime.now()
    )

# Get recommendations
recommendations = recommender.get_recommendations(
    user_id=user_id,
    n_recommendations=5,
    context={'time_of_day': 'evening', 'device': 'mobile'}
)

print("\nTop 5 Recommendations:")
for rank, rec in enumerate(recommendations, 1):
    print(f"{rank}. Item {rec['item_id']}")
    print(f"   Score: {rec['score']:.4f}")
    print(f"   Category: {rec['metadata']['category']}")
    print(f"   Popularity: {rec['popularity']:.4f}")
```

Slide 13: Cross-Domain Recommendation System

Cross-domain recommendation systems leverage knowledge from multiple domains to improve recommendation quality, particularly useful for addressing the cold-start problem and enhancing recommendation diversity.

```python
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict

class CrossDomainRecommender:
    def __init__(self, domains: List[str], embedding_dim: int = 100):
        self.domains = domains
        self.embedding_dim = embedding_dim
        
        # Initialize domain-specific components
        self.domain_items: Dict[str, Set[int]] = {d: set() for d in domains}
        self.item_embeddings: Dict[str, Dict[int, np.ndarray]] = {
            d: {} for d in domains
        }
        self.user_embeddings: Dict[str, Dict[int, np.ndarray]] = {
            d: {} for d in domains
        }
        
        # Cross-domain mapping matrices
        self.domain_mappings: Dict[Tuple[str, str], np.ndarray] = {}
        self._initialize_domain_mappings()
        
    def _initialize_domain_mappings(self) -> None:
        for i, source_domain in enumerate(self.domains):
            for target_domain in self.domains[i+1:]:
                mapping = np.random.randn(self.embedding_dim, self.embedding_dim)
                mapping = mapping / np.linalg.norm(mapping, axis=1)[:, np.newaxis]
                self.domain_mappings[(source_domain, target_domain)] = mapping
                self.domain_mappings[(target_domain, source_domain)] = mapping.T
    
    def add_item(self, domain: str, item_id: int, 
                 features: np.ndarray) -> None:
        embedding = features / np.linalg.norm(features)
        self.domain_items[domain].add(item_id)
        self.item_embeddings[domain][item_id] = embedding
    
    def update_user_profile(self, user_id: int, domain: str, 
                          interactions: List[Tuple[int, float]]) -> None:
        # Compute weighted average of item embeddings
        total_weight = 0
        user_embedding = np.zeros(self.embedding_dim)
        
        for item_id, weight in interactions:
            if item_id in self.item_embeddings[domain]:
                user_embedding += (
                    self.item_embeddings[domain][item_id] * weight
                )
                total_weight += weight
        
        if total_weight > 0:
            user_embedding /= total_weight
            self.user_embeddings[domain][user_id] = user_embedding
    
    def transfer_knowledge(self, user_id: int, 
                         source_domain: str, 
                         target_domain: str) -> np.ndarray:
        if user_id not in self.user_embeddings[source_domain]:
            return np.zeros(self.embedding_dim)
            
        source_embedding = self.user_embeddings[source_domain][user_id]
        mapping = self.domain_mappings[(source_domain, target_domain)]
        return source_embedding @ mapping
    
    def get_recommendations(self, user_id: int, domain: str, 
                          n_recommendations: int = 10) -> List[Tuple[int, float]]:
        user_embedding = np.zeros(self.embedding_dim)
        domain_weights = {d: 0.0 for d in self.domains}
        
        # Collect user embeddings from all domains
        for d in self.domains:
            if user_id in self.user_embeddings[d]:
                transferred = self.transfer_knowledge(user_id, d, domain)
                weight = 1.0 if d == domain else 0.3
                user_embedding += transferred * weight
                domain_weights[d] = weight
        
        total_weight = sum(domain_weights.values())
        if total_weight > 0:
            user_embedding /= total_weight
        
        # Compute scores for all items in target domain
        scores = []
        for item_id in self.domain_items[domain]:
            item_embedding = self.item_embeddings[domain][item_id]
            score = np.dot(user_embedding, item_embedding)
            scores.append((item_id, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:n_recommendations]

# Example usage
domains = ['books', 'movies', 'music']
recommender = CrossDomainRecommender(domains)

# Add items to different domains
np.random.seed(42)
for domain in domains:
    for item_id in range(100):
        features = np.random.randn(100)
        recommender.add_item(domain, item_id, features)

# Simulate user interactions across domains
user_id = 1
interactions = {
    'books': [(i, np.random.random()) for i in range(10)],
    'movies': [(i, np.random.random()) for i in range(5)],
    'music': [(i, np.random.random()) for i in range(3)]
}

for domain, domain_interactions in interactions.items():
    recommender.update_user_profile(user_id, domain, domain_interactions)

# Get recommendations for each domain
print("Cross-Domain Recommendations:")
for domain in domains:
    print(f"\n{domain.capitalize()} Recommendations:")
    recommendations = recommender.get_recommendations(user_id, domain, 5)
    for rank, (item_id, score) in enumerate(recommendations, 1):
        print(f"{rank}. Item {item_id}: {score:.4f}")
```

Slide 14: Additional Resources

*   "Deep Neural Networks for YouTube Recommendations" [https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)
*   "Neural Collaborative Filtering" [https://arxiv.org/abs/1708.05031](https://arxiv.org/abs/1708.05031)
*   "Self-Attentive Sequential Recommendation" [https://arxiv.org/abs/1808.09781](https://arxiv.org/abs/1808.09781)
*   "Wide & Deep Learning for Recommender Systems" [https://arxiv.org/abs/1606.07792](https://arxiv.org/abs/1606.07792)
*   "AutoRec: Autoencoders Meet Collaborative Filtering" Search on Google Scholar for the latest version
*   "Learning to Rank for Information Retrieval and Recommendation" Recommended search on academic databases
*   "Matrix Factorization Techniques for Recommender Systems" Available in IEEE Xplore Digital Library

For practical implementations and industry insights:

*   Google Research Blog
*   Netflix Technology Blog
*   Spotify Engineering Blog
*   Amazon Science Blog

