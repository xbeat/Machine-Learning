## Recommender Systems Evaluation Metrics and Loss Functions in Python
Slide 1: Introduction to Recommender Systems

Recommender systems are algorithms designed to suggest relevant items to users based on their preferences and behavior. These systems are widely used in various applications, from e-commerce to content streaming platforms. In this presentation, we'll explore the evaluation metrics and loss functions used to measure and improve the performance of recommender systems.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulating user-item interactions
users = ['User A', 'User B', 'User C', 'User D']
items = ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5']

# Creating a sample interaction matrix (1 for interaction, 0 for no interaction)
interactions = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0],
    [1, 1, 0, 0, 1],
    [0, 0, 1, 1, 0]
])

plt.imshow(interactions, cmap='Blues')
plt.xticks(range(len(items)), items)
plt.yticks(range(len(users)), users)
plt.title('User-Item Interaction Matrix')
plt.colorbar(label='Interaction')
plt.show()
```

Slide 2: Accuracy Metrics - Precision and Recall

Precision and recall are fundamental metrics used to evaluate the accuracy of recommender systems. Precision measures the proportion of relevant items among the recommended items, while recall measures the proportion of relevant items that were successfully recommended.

```python
def precision_recall(recommendations, actual):
    true_positives = len(set(recommendations) & set(actual))
    precision = true_positives / len(recommendations)
    recall = true_positives / len(actual)
    return precision, recall

# Example
recommendations = ['Item 1', 'Item 2', 'Item 3', 'Item 4']
actual_relevant = ['Item 1', 'Item 3', 'Item 5']

precision, recall = precision_recall(recommendations, actual_relevant)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
```

Slide 3: F1 Score

The F1 score is the harmonic mean of precision and recall, providing a single metric that balances both. It's particularly useful when you want to find an optimal balance between precision and recall.

```python
def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

# Using the previous example
f1 = f1_score(precision, recall)
print(f"F1 Score: {f1:.2f}")

# Visualizing the relationship between Precision, Recall, and F1 Score
p = np.linspace(0.01, 1, 100)
r = np.linspace(0.01, 1, 100)
P, R = np.meshgrid(p, r)
F1 = 2 * (P * R) / (P + R)

plt.figure(figsize=(10, 8))
plt.contourf(P, R, F1, levels=20, cmap='viridis')
plt.colorbar(label='F1 Score')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('F1 Score as a Function of Precision and Recall')
plt.show()
```

Slide 4: Mean Average Precision (MAP)

Mean Average Precision is a metric that considers the order of recommendations. It calculates the average precision at each relevant item in the recommendation list and then takes the mean across all users.

```python
def average_precision(recommendations, actual):
    ap = 0
    relevant_count = 0
    
    for i, item in enumerate(recommendations, 1):
        if item in actual:
            relevant_count += 1
            ap += relevant_count / i
    
    return ap / len(actual) if len(actual) > 0 else 0

def mean_average_precision(all_recommendations, all_actual):
    aps = [average_precision(rec, act) for rec, act in zip(all_recommendations, all_actual)]
    return sum(aps) / len(aps)

# Example
user1_recs = ['Item 1', 'Item 2', 'Item 3', 'Item 4']
user1_actual = ['Item 1', 'Item 3', 'Item 5']

user2_recs = ['Item 2', 'Item 1', 'Item 5', 'Item 3']
user2_actual = ['Item 1', 'Item 2', 'Item 5']

map_score = mean_average_precision([user1_recs, user2_recs], [user1_actual, user2_actual])
print(f"Mean Average Precision: {map_score:.3f}")
```

Slide 5: Normalized Discounted Cumulative Gain (NDCG)

NDCG is a ranking metric that takes into account both the relevance and the position of recommended items. It emphasizes the importance of highly relevant items appearing earlier in the recommendation list.

```python
import numpy as np

def dcg(relevances, k=None):
    if k is None:
        k = len(relevances)
    return np.sum(relevances[:k] / np.log2(np.arange(2, k + 2)))

def ndcg(recommendations, actual, k=None):
    relevances = [1 if item in actual else 0 for item in recommendations]
    ideal_relevances = sorted([1] * len(actual) + [0] * (len(recommendations) - len(actual)), reverse=True)
    
    dcg_score = dcg(relevances, k)
    idcg_score = dcg(ideal_relevances, k)
    
    return dcg_score / idcg_score if idcg_score > 0 else 0

# Example
recommendations = ['Item 1', 'Item 4', 'Item 2', 'Item 3', 'Item 5']
actual_relevant = ['Item 1', 'Item 2', 'Item 5']

ndcg_score = ndcg(recommendations, actual_relevant)
print(f"NDCG Score: {ndcg_score:.3f}")
```

Slide 6: Hit Rate

Hit Rate is a simple metric that measures the proportion of users for whom the recommender system successfully recommended at least one relevant item within the top-N recommendations.

```python
def hit_rate(all_recommendations, all_actual, N=5):
    hits = sum(1 for rec, act in zip(all_recommendations, all_actual) if any(item in act for item in rec[:N]))
    return hits / len(all_recommendations)

# Example
user1_recs = ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5']
user1_actual = ['Item 1', 'Item 3', 'Item 5']

user2_recs = ['Item 2', 'Item 1', 'Item 5', 'Item 3', 'Item 4']
user2_actual = ['Item 1', 'Item 2', 'Item 5']

user3_recs = ['Item 4', 'Item 3', 'Item 2', 'Item 1', 'Item 5']
user3_actual = ['Item 6', 'Item 7', 'Item 8']

hr = hit_rate([user1_recs, user2_recs, user3_recs], [user1_actual, user2_actual, user3_actual])
print(f"Hit Rate: {hr:.2f}")
```

Slide 7: Coverage

Coverage measures the proportion of items in the catalog that the recommender system is able to recommend. It's important to ensure that the system can suggest a wide range of items and not just the most popular ones.

```python
def catalog_coverage(all_recommendations, catalog):
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)
    return len(recommended_items) / len(catalog)

# Example
catalog = set(['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5', 'Item 6', 'Item 7', 'Item 8'])

user1_recs = ['Item 1', 'Item 2', 'Item 3']
user2_recs = ['Item 2', 'Item 4', 'Item 5']
user3_recs = ['Item 1', 'Item 3', 'Item 6']

coverage = catalog_coverage([user1_recs, user2_recs, user3_recs], catalog)
print(f"Catalog Coverage: {coverage:.2f}")
```

Slide 8: Diversity

Diversity measures how different the recommended items are from each other. It's important to provide diverse recommendations to avoid monotony and increase user satisfaction.

```python
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def diversity(recommendations, item_features):
    pairs = [(i, j) for i in range(len(recommendations)) for j in range(i+1, len(recommendations))]
    similarities = [cosine_similarity(item_features[recommendations[i]], item_features[recommendations[j]]) for i, j in pairs]
    return 1 - np.mean(similarities)

# Example: Item features (simplified as binary vectors)
item_features = {
    'Item 1': [1, 0, 1, 0],
    'Item 2': [0, 1, 1, 0],
    'Item 3': [1, 1, 0, 1],
    'Item 4': [0, 0, 1, 1],
    'Item 5': [1, 0, 0, 1]
}

recommendations = ['Item 1', 'Item 2', 'Item 3', 'Item 4']
div_score = diversity(recommendations, item_features)
print(f"Diversity Score: {div_score:.3f}")
```

Slide 9: Serendipity

Serendipity measures how surprising and relevant the recommendations are. It's about recommending items that a user might not have discovered on their own but would enjoy.

```python
def serendipity(recommendations, user_profile, item_features, unexpectedness_threshold=0.5):
    unexpected_items = [item for item in recommendations if cosine_similarity(user_profile, item_features[item]) < unexpectedness_threshold]
    return len(unexpected_items) / len(recommendations)

# Example
user_profile = [1, 0, 1, 0]  # Simplified user profile
recommendations = ['Item 1', 'Item 2', 'Item 3', 'Item 4']

serendipity_score = serendipity(recommendations, user_profile, item_features)
print(f"Serendipity Score: {serendipity_score:.3f}")
```

Slide 10: Mean Squared Error (MSE)

Mean Squared Error is a common loss function used in collaborative filtering algorithms. It measures the average squared difference between predicted and actual ratings.

```python
import numpy as np

def mean_squared_error(predictions, actual):
    return np.mean((np.array(predictions) - np.array(actual)) ** 2)

# Example
predicted_ratings = [4.2, 3.8, 2.7, 4.5, 3.2]
actual_ratings = [4, 4, 3, 5, 3]

mse = mean_squared_error(predicted_ratings, actual_ratings)
print(f"Mean Squared Error: {mse:.3f}")

# Visualizing predictions vs actual ratings
plt.figure(figsize=(10, 6))
plt.scatter(actual_ratings, predicted_ratings, alpha=0.5)
plt.plot([min(actual_ratings), max(actual_ratings)], [min(actual_ratings), max(actual_ratings)], 'r--')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Predicted vs Actual Ratings')
plt.show()
```

Slide 11: Binary Cross-Entropy Loss

Binary Cross-Entropy Loss is commonly used in implicit feedback scenarios, where the goal is to predict whether a user will interact with an item or not.

```python
import numpy as np

def binary_cross_entropy(predictions, actual):
    epsilon = 1e-15  # Small value to avoid log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(actual * np.log(predictions) + (1 - actual) * np.log(1 - predictions))

# Example
predicted_probs = [0.8, 0.3, 0.6, 0.2, 0.9]
actual_interactions = [1, 0, 1, 0, 1]

bce = binary_cross_entropy(predicted_probs, actual_interactions)
print(f"Binary Cross-Entropy Loss: {bce:.3f}")

# Visualizing the binary cross-entropy loss
p = np.linspace(0.01, 0.99, 100)
bce_0 = -np.log(1 - p)
bce_1 = -np.log(p)

plt.figure(figsize=(10, 6))
plt.plot(p, bce_0, label='Actual = 0')
plt.plot(p, bce_1, label='Actual = 1')
plt.xlabel('Predicted Probability')
plt.ylabel('Binary Cross-Entropy Loss')
plt.title('Binary Cross-Entropy Loss vs Predicted Probability')
plt.legend()
plt.show()
```

Slide 12: Bayesian Personalized Ranking (BPR) Loss

BPR Loss is designed for implicit feedback scenarios and aims to rank relevant items higher than irrelevant ones for each user.

```python
import numpy as np

def bpr_loss(positive_scores, negative_scores):
    return -np.mean(np.log(1 / (1 + np.exp(-(positive_scores - negative_scores)))))

# Example
positive_scores = [0.8, 0.7, 0.9]
negative_scores = [0.3, 0.2, 0.5]

bpr = bpr_loss(positive_scores, negative_scores)
print(f"BPR Loss: {bpr:.3f}")

# Visualizing BPR loss
score_diff = np.linspace(-5, 5, 100)
bpr_loss_values = -np.log(1 / (1 + np.exp(-score_diff)))

plt.figure(figsize=(10, 6))
plt.plot(score_diff, bpr_loss_values)
plt.xlabel('Score Difference (Positive - Negative)')
plt.ylabel('BPR Loss')
plt.title('BPR Loss vs Score Difference')
plt.show()
```

Slide 13: Real-life Example: Movie Recommendation System

Let's consider a movie recommendation system for a streaming platform. We'll use a simplified version to demonstrate how these metrics and loss functions can be applied in practice.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-movie interaction matrix
user_movie_matrix = np.array([
    [5, 3, 0, 1, 0],
    [4, 0, 0, 1, 2],
    [1, 1, 0, 5, 0],
    [0, 0, 4, 0, 2],
    [0, 1, 5, 4, 0]
])

# Calculate item-item similarity
item_similarity = cosine_similarity(user_movie_matrix.T)

def recommend_movies(user_id, n_recommendations=3):
    user_ratings = user_movie_matrix[user_id]
    weighted_sum = np.dot(item_similarity, user_ratings)
    already_watched = user_ratings > 0
    weighted_sum[already_watched] = -np.inf
    return np.argsort(weighted_sum)[-n_recommendations:][::-1]

# Example recommendation
user_id = 2
recommendations = recommend_movies(user_id)
print(f"Recommended movies for user {user_id}: {recommendations}")

# Evaluate recommendations
actual_ratings = user_movie_matrix[user_id]
predicted_ratings = item_similarity[recommendations, user_id]

mse = np.mean((actual_ratings[actual_ratings > 0] - predicted_ratings[actual_ratings > 0])**2)
print(f"Mean Squared Error: {mse:.3f}")

# Calculate hit rate
hit_rate = np.mean(actual_ratings[recommendations] > 0)
print(f"Hit Rate: {hit_rate:.2f}")
```

Slide 14: Real-life Example: Content-Based Recommendation for a News Website

Consider a news recommendation system that suggests articles to users based on their reading history and article content. This example demonstrates how content-based filtering can be implemented and evaluated.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample articles and their content
articles = [
    "Climate change impacts global economy",
    "New AI breakthrough in natural language processing",
    "Sports team wins championship after 50 years",
    "Advancements in renewable energy technology",
    "Political tensions rise in international relations"
]

# User reading history (1 for read, 0 for unread)
user_history = [1, 1, 0, 1, 0]

# Create TF-IDF vectors for articles
vectorizer = TfidfVectorizer()
article_vectors = vectorizer.fit_transform(articles)

# Calculate similarity between articles
article_similarity = cosine_similarity(article_vectors)

def recommend_articles(user_history, n_recommendations=2):
    user_profile = np.dot(user_history, article_vectors)
    scores = cosine_similarity(user_profile, article_vectors)[0]
    already_read = user_history == 1
    scores[already_read] = -np.inf
    return np.argsort(scores)[-n_recommendations:][::-1]

# Get recommendations
recommendations = recommend_articles(user_history)
print(f"Recommended articles: {recommendations}")

# Evaluate diversity
def diversity_score(recommendations):
    pairs = [(recommendations[i], recommendations[j]) 
             for i in range(len(recommendations)) 
             for j in range(i+1, len(recommendations))]
    return 1 - np.mean([article_similarity[i, j] for i, j in pairs])

diversity = diversity_score(recommendations)
print(f"Diversity score: {diversity:.3f}")
```

Slide 15: Additional Resources

For those interested in diving deeper into recommender systems, evaluation metrics, and loss functions, here are some valuable resources:

1. "Evaluating Recommendation Systems" by Guy Shani and Asela Gunawardana ArXiv URL: [https://arxiv.org/abs/1109.2646](https://arxiv.org/abs/1109.2646)
2. "Deep Learning based Recommender System: A Survey and New Perspectives" by Shuai Zhang et al. ArXiv URL: [https://arxiv.org/abs/1707.07435](https://arxiv.org/abs/1707.07435)
3. "Neural Collaborative Filtering" by Xiangnan He et al. ArXiv URL: [https://arxiv.org/abs/1708.05031](https://arxiv.org/abs/1708.05031)
4. "Collaborative Filtering for Implicit Feedback Datasets" by Yifan Hu et al. (This paper introduces the concept of implicit feedback in recommender systems)
5. "BPR: Bayesian Personalized Ranking from Implicit Feedback" by Steffen Rendle et al. (This paper introduces the BPR loss function)

These resources provide in-depth discussions on various aspects of recommender systems, including advanced evaluation metrics, novel loss functions, and state-of-the-art approaches using deep learning techniques.

