## Collaborative Filtering Algorithms in Python

Slide 1: Introduction to Collaborative Filtering

Collaborative Filtering is a popular technique used in recommendation systems to predict user preferences based on historical interactions. It leverages the idea that users with similar tastes in the past will have similar preferences in the future. This approach is widely used by companies like Netflix, Amazon, and Spotify to suggest products, movies, or songs to their users.

```python
# Simple representation of user-item interactions
user_item_matrix = {
    'Alice': {'Item1': 5, 'Item2': 3, 'Item3': 4},
    'Bob': {'Item1': 3, 'Item2': 1, 'Item3': 2},
    'Charlie': {'Item1': 4, 'Item2': 3, 'Item3': 5}
}
```

Slide 2: Types of Collaborative Filtering

There are two main types of collaborative filtering: User-Based and Item-Based. User-Based CF finds similar users and recommends items they liked. Item-Based CF finds similar items based on user ratings and recommends them. Both approaches aim to predict ratings for items a user hasn't interacted with yet.

```python
def user_based_cf(user, item, matrix):
    similar_users = find_similar_users(user, matrix)
    return predict_rating(similar_users, item, matrix)

def item_based_cf(user, item, matrix):
    similar_items = find_similar_items(item, matrix)
    return predict_rating(user, similar_items, matrix)
```

Slide 3: User-Based Collaborative Filtering

User-Based CF identifies users with similar preferences to the target user. It then uses their ratings to predict how the target user would rate an item they haven't interacted with yet. This method is effective for systems with a stable item set but can be computationally expensive for large user bases.

```python
def find_similar_users(target_user, matrix):
    similarities = {}
    for user in matrix:
        if user != target_user:
            similarity = calculate_similarity(matrix[target_user], matrix[user])
            similarities[user] = similarity
    return sorted(similarities, key=similarities.get, reverse=True)[:5]

def calculate_similarity(user1, user2):
    common_items = set(user1.keys()) & set(user2.keys())
    if not common_items:
        return 0
    sum_of_squares = sum((user1[item] - user2[item])**2 for item in common_items)
    return 1 / (1 + sum_of_squares)  # Simple similarity measure
```

Slide 4: Item-Based Collaborative Filtering

Item-Based CF focuses on finding items similar to those the user has liked in the past. This approach is often more scalable than user-based CF, especially for systems with many users but a relatively stable item set. It's less affected by new users joining the system.

```python
def find_similar_items(target_item, matrix):
    similarities = {}
    for item in set(i for user in matrix.values() for i in user):
        if item != target_item:
            similarity = calculate_item_similarity(target_item, item, matrix)
            similarities[item] = similarity
    return sorted(similarities, key=similarities.get, reverse=True)[:5]

def calculate_item_similarity(item1, item2, matrix):
    users_rated_both = [user for user in matrix if item1 in matrix[user] and item2 in matrix[user]]
    if not users_rated_both:
        return 0
    sum_of_squares = sum((matrix[user][item1] - matrix[user][item2])**2 for user in users_rated_both)
    return 1 / (1 + sum_of_squares)  # Simple similarity measure
```

Slide 5: Matrix Factorization

Matrix Factorization is a powerful algorithm commonly used for collaborative filtering. It works by decomposing the user-item interaction matrix into two lower-dimensional matrices: one representing user factors and another representing item factors. These factors capture latent features that explain the observed ratings.

```python
import numpy as np

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:], Q[:,j])
                    for k in range(K):
                        P[i][k] += alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] += alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T

# Example usage
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

N = len(R)
M = len(R[0])
K = 2

P = np.random.rand(N,K)
Q = np.random.rand(M,K)

nP, nQ = matrix_factorization(R, P, Q, K)
nR = np.dot(nP, nQ.T)

print("Original Matrix:")
print(R)
print("\nPredicted Matrix:")
print(nR)
```

Slide 6: Singular Value Decomposition (SVD)

SVD is another popular matrix factorization technique used in collaborative filtering. It decomposes the user-item matrix into three matrices: U (user features), Σ (singular values), and V^T (item features). SVD can help reduce noise in the data and capture the most important factors influencing user preferences.

```python
import numpy as np

def svd_recommender(R, k):
    # Perform SVD
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    
    # Truncate to k dimensions
    s_k = np.diag(s[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    
    # Reconstruct the matrix
    R_k = np.dot(np.dot(U_k, s_k), Vt_k)
    
    return R_k

# Example usage
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

k = 2  # Number of latent factors
R_predicted = svd_recommender(R, k)

print("Original Matrix:")
print(R)
print("\nPredicted Matrix:")
print(R_predicted)
```

Slide 7: Alternating Least Squares (ALS)

ALS is an iterative optimization algorithm used in collaborative filtering, especially for large-scale recommendation systems. It alternates between fixing the user factors and item factors, solving a least squares problem in each step. This approach is particularly useful for implicit feedback datasets.

```python
import numpy as np

def als(R, k, num_iterations=10, lambda_reg=0.1):
    m, n = R.shape
    X = np.random.rand(m, k)
    Y = np.random.rand(n, k)

    for _ in range(num_iterations):
        # Fix Y and solve for X
        for i in range(m):
            Y_i = Y[R[i, :] > 0]
            X[i] = np.linalg.solve(Y_i.T @ Y_i + lambda_reg * np.eye(k), Y_i.T @ R[i, R[i, :] > 0])

        # Fix X and solve for Y
        for j in range(n):
            X_j = X[R[:, j] > 0]
            Y[j] = np.linalg.solve(X_j.T @ X_j + lambda_reg * np.eye(k), X_j.T @ R[R[:, j] > 0, j])

    return X, Y

# Example usage
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

k = 2  # Number of latent factors
X, Y = als(R, k)
R_predicted = X @ Y.T

print("Original Matrix:")
print(R)
print("\nPredicted Matrix:")
print(R_predicted)
```

Slide 8: Neighborhood-Based Collaborative Filtering

Neighborhood-based methods are intuitive and easy to implement. They rely on finding similar users or items based on their rating patterns. These methods can be effective for smaller datasets and provide explainable recommendations.

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def neighborhood_cf(R, user_id, item_id, k=3):
    user_similarities = [cosine_similarity(R[user_id], R[i]) for i in range(len(R))]
    similar_users = np.argsort(user_similarities)[-k-1:-1][::-1]
    
    numerator = sum(R[u][item_id] * user_similarities[u] for u in similar_users if R[u][item_id] != 0)
    denominator = sum(user_similarities[u] for u in similar_users if R[u][item_id] != 0)
    
    if denominator == 0:
        return 0
    return numerator / denominator

# Example usage
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

user_id = 0
item_id = 2
predicted_rating = neighborhood_cf(R, user_id, item_id)

print(f"Predicted rating for user {user_id} on item {item_id}: {predicted_rating}")
```

Slide 9: Handling Cold Start Problem

The cold start problem occurs when the system lacks sufficient information about new users or items. Hybrid approaches combining collaborative filtering with content-based methods can help mitigate this issue. Here's a simple example of a hybrid recommender:

```python
import numpy as np

def content_based_similarity(item_features):
    return np.dot(item_features, item_features.T) / (np.linalg.norm(item_features, axis=1)[:, np.newaxis] * np.linalg.norm(item_features, axis=1))

def hybrid_recommender(R, item_features, user_id, alpha=0.5):
    cf_scores = R[user_id]
    cb_scores = np.dot(R[user_id], content_based_similarity(item_features))
    hybrid_scores = alpha * cf_scores + (1 - alpha) * cb_scores
    return hybrid_scores

# Example usage
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

item_features = np.array([
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 1, 1],
])

user_id = 0
recommendations = hybrid_recommender(R, item_features, user_id)

print(f"Hybrid recommendations for user {user_id}:")
print(recommendations)
```

Slide 10: Evaluation Metrics

Evaluating collaborative filtering algorithms is crucial for assessing their performance. Common metrics include Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Normalized Discounted Cumulative Gain (NDCG). Here's an implementation of MAE and RMSE:

```python
import numpy as np

def calculate_mae(true_ratings, predicted_ratings):
    return np.mean(np.abs(true_ratings - predicted_ratings))

def calculate_rmse(true_ratings, predicted_ratings):
    return np.sqrt(np.mean((true_ratings - predicted_ratings)**2))

# Example usage
true_ratings = np.array([4, 3, 5, 2, 1])
predicted_ratings = np.array([3.8, 3.2, 4.7, 2.3, 1.5])

mae = calculate_mae(true_ratings, predicted_ratings)
rmse = calculate_rmse(true_ratings, predicted_ratings)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Square Error: {rmse}")
```

Slide 11: Real-Life Example: Movie Recommendation System

Let's implement a simple movie recommendation system using collaborative filtering. This example demonstrates how to recommend movies to users based on their ratings and the ratings of similar users.

```python
import numpy as np

# Sample movie ratings (users x movies)
ratings = np.array([
    [4, 3, 0, 5, 0],
    [5, 0, 4, 0, 2],
    [3, 1, 2, 4, 1],
    [0, 0, 0, 2, 5],
    [1, 5, 3, 4, 0],
])

movies = ["The Matrix", "Inception", "Interstellar", "Pulp Fiction", "The Godfather"]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recommend_movies(user_id, num_recommendations=2):
    user_ratings = ratings[user_id]
    similarities = [cosine_similarity(user_ratings, other_ratings) for other_ratings in ratings]
    similar_users = np.argsort(similarities)[-2:][::-1]  # Top 2 similar users
    
    recommendations = []
    for movie_id in range(len(movies)):
        if user_ratings[movie_id] == 0:  # User hasn't rated this movie
            predicted_rating = np.mean([ratings[u][movie_id] for u in similar_users if ratings[u][movie_id] > 0])
            if not np.isnan(predicted_rating):
                recommendations.append((movies[movie_id], predicted_rating))
    
    return sorted(recommendations, key=lambda x: x[1], reverse=True)[:num_recommendations]

# Example usage
user_id = 0
recommendations = recommend_movies(user_id)
print(f"Recommended movies for user {user_id}:")
for movie, rating in recommendations:
    print(f"{movie}: Predicted rating {rating:.2f}")
```

Slide 12: Real-Life Example: Music Playlist Generation

Here's an example of how collaborative filtering can be used to generate personalized music playlists based on user listening history and song similarities.

```python
import numpy as np

# Sample listening history (users x songs)
listening_history = np.array([
    [10
```

Slide 12: Real-Life Example: Music Playlist Generation

Let's implement a simple music playlist generator using collaborative filtering. This example demonstrates how to create personalized playlists based on user listening history and song similarities.

```python
import numpy as np

# Sample listening history (users x songs)
listening_history = np.array([
    [10, 5, 0, 8, 3],
    [8, 0, 7, 2, 9],
    [5, 2, 4, 0, 6],
    [0, 10, 6, 4, 1],
    [2, 4, 0, 5, 8],
])

songs = ["Song A", "Song B", "Song C", "Song D", "Song E"]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_playlist(user_id, num_songs=3):
    user_history = listening_history[user_id]
    similarities = [cosine_similarity(user_history, other_history) for other_history in listening_history]
    similar_users = np.argsort(similarities)[-2:][::-1]  # Top 2 similar users
    
    playlist = []
    for song_id in range(len(songs)):
        if user_history[song_id] == 0:  # User hasn't listened to this song
            predicted_listens = np.mean([listening_history[u][song_id] for u in similar_users])
            if not np.isnan(predicted_listens):
                playlist.append((songs[song_id], predicted_listens))
    
    return sorted(playlist, key=lambda x: x[1], reverse=True)[:num_songs]

# Example usage
user_id = 0
playlist = generate_playlist(user_id)
print(f"Recommended playlist for user {user_id}:")
for song, score in playlist:
    print(f"{song}: Recommendation score {score:.2f}")
```

Slide 13: Challenges in Collaborative Filtering

Collaborative filtering faces several challenges in real-world applications. These include:

1.  Sparsity: User-item interaction matrices are often sparse, with many users rating only a small fraction of items.
2.  Scalability: As the number of users and items grows, computational complexity increases.
3.  Cold start: Difficulty in making recommendations for new users or items with no interaction history.
4.  Diversity: Tendency to recommend popular items, potentially creating a "filter bubble."
5.  Privacy concerns: Collaborative filtering relies on user data, which raises privacy issues.

To address these challenges, researchers and practitioners often combine collaborative filtering with other techniques, such as content-based filtering, deep learning, and context-aware recommendations.

Slide 14: Future Directions in Collaborative Filtering

The field of collaborative filtering continues to evolve, with several promising directions:

1.  Deep learning approaches: Leveraging neural networks to capture complex patterns in user-item interactions.
2.  Temporal dynamics: Incorporating time-based factors to model changing user preferences.
3.  Cross-domain recommendations: Utilizing information from multiple domains to improve recommendation quality.
4.  Explainable AI: Developing methods to provide interpretable recommendations to users.
5.  Federated learning: Enabling collaborative filtering while preserving user privacy by keeping data on local devices.

As these areas develop, we can expect more sophisticated and effective recommendation systems that balance accuracy, scalability, and user privacy.

Slide 15: Additional Resources

For those interested in diving deeper into collaborative filtering and recommendation systems, here are some valuable resources:

1.  "Matrix Factorization Techniques for Recommender Systems" by Koren et al. (2009) ArXiv: [https://arxiv.org/abs/1802.07222](https://arxiv.org/abs/1802.07222)
2.  "Collaborative Filtering for Implicit Feedback Datasets" by Hu et al. (2008) ArXiv: [https://arxiv.org/abs/1801.01973](https://arxiv.org/abs/1801.01973)
3.  "Deep Learning based Recommender System: A Survey and New Perspectives" by Zhang et al. (2019) ArXiv: [https://arxiv.org/abs/1707.07435](https://arxiv.org/abs/1707.07435)

These papers provide in-depth discussions of various collaborative filtering techniques and their applications in recommendation systems.

