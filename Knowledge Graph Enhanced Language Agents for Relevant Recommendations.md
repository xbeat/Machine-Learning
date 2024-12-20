## Knowledge Graph Enhanced Language Agents for Relevant Recommendations
Slide 1: Knowledge Graph Foundations

Knowledge graphs serve as the backbone of KGLA systems, representing entities and relationships in a structured format. We'll implement a basic knowledge graph structure using Python dictionaries and custom classes to model nodes and edges.

```python
class KnowledgeGraph:
    def __init__(self):
        self.nodes = {}  # Store entities
        self.edges = {}  # Store relationships
        
    def add_node(self, node_id, attributes=None):
        self.nodes[node_id] = attributes or {}
        
    def add_edge(self, source, target, relation_type, weight=1.0):
        if source not in self.edges:
            self.edges[source] = []
        self.edges[source].append({
            'target': target,
            'relation': relation_type,
            'weight': weight
        })

# Example usage
kg = KnowledgeGraph()
kg.add_node('user_1', {'preference': 'sci-fi'})
kg.add_node('movie_1', {'title': 'Inception', 'genre': 'sci-fi'})
kg.add_edge('user_1', 'movie_1', 'watched', 0.9)
```

Slide 2: Graph Path Extraction

The core functionality of KGLA involves extracting meaningful paths between entities. This implementation demonstrates how to perform depth-first search to find relevant connection patterns between users and items.

```python
def find_paths(kg, start_node, end_node, max_depth=3):
    def dfs(current, target, path, depth):
        if depth > max_depth:
            return []
        if current == target:
            return [path]
        
        paths = []
        for edge in kg.edges.get(current, []):
            next_node = edge['target']
            if next_node not in path:  # Avoid cycles
                new_path = path + [(current, edge['relation'], next_node)]
                paths.extend(dfs(next_node, target, new_path, depth + 1))
        return paths
    
    return dfs(start_node, end_node, [], 0)

# Example usage
paths = find_paths(kg, 'user_1', 'movie_1')
for path in paths:
    print("Path:", ' -> '.join([f"({p[0]}, {p[1]}, {p[2]})" for p in path]))
```

Slide 3: Path Translation Module

Converting graph paths to natural language requires sophisticated template management and context understanding. This implementation provides a flexible system for translating graph patterns into human-readable descriptions.

```python
class PathTranslator:
    def __init__(self):
        self.templates = {
            'watched': '{user} has watched {movie}',
            'likes': '{user} likes {genre}',
            'belongs_to': '{movie} belongs to {genre}'
        }
    
    def translate_path(self, path, entity_attributes):
        sentences = []
        for source, relation, target in path:
            template = self.templates.get(relation, '{source} {relation} {target}')
            sentences.append(template.format(
                user=entity_attributes.get(source, {}).get('name', source),
                movie=entity_attributes.get(target, {}).get('title', target),
                genre=entity_attributes.get(target, {}).get('genre', target)
            ))
        return ' and '.join(sentences)

# Example usage
translator = PathTranslator()
entity_attributes = {
    'user_1': {'name': 'John'},
    'movie_1': {'title': 'Inception', 'genre': 'sci-fi'}
}
```

Slide 4: KGLA Core Implementation

The KGLA agent combines knowledge graph traversal with natural language processing to generate contextually aware recommendations. This implementation shows the core architecture and decision-making process.

```python
import numpy as np
from collections import defaultdict

class KGLAAgent:
    def __init__(self, knowledge_graph, path_translator):
        self.kg = knowledge_graph
        self.translator = path_translator
        self.path_weights = defaultdict(float)
        
    def compute_path_score(self, path):
        score = 1.0
        for source, relation, target in path:
            edge = next(e for e in self.kg.edges[source] 
                       if e['target'] == target and e['relation'] == relation)
            score *= edge['weight']
        return score
    
    def recommend(self, user_id, n_recommendations=5):
        recommendations = []
        for item_id in self.kg.nodes:
            if 'type' in self.kg.nodes[item_id] and self.kg.nodes[item_id]['type'] == 'item':
                paths = find_paths(self.kg, user_id, item_id)
                if paths:
                    score = max(self.compute_path_score(path) for path in paths)
                    recommendations.append((item_id, score))
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:n_recommendations]
```

Slide 5: Path Embedding Generation

Converting graph paths into dense vector representations enables sophisticated similarity computations and improved recommendation quality through neural processing of structural patterns.

```python
import torch
import torch.nn as nn

class PathEmbedding(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=128):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.path_lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        
    def forward(self, paths):
        batch_embeddings = []
        for path in paths:
            path_embeddings = []
            for entity, relation, _ in path:
                entity_emb = self.entity_embeddings(torch.tensor(entity))
                relation_emb = self.relation_embeddings(torch.tensor(relation))
                path_embeddings.append(entity_emb + relation_emb)
            
            path_tensor = torch.stack(path_embeddings)
            lstm_out, _ = self.path_lstm(path_tensor.unsqueeze(0))
            batch_embeddings.append(lstm_out[:, -1, :])
            
        return torch.cat(batch_embeddings, dim=0)
```

Slide 6: User Profile Enrichment

Enhancing user profiles through knowledge graph exploration reveals deeper preference patterns. This implementation demonstrates how to aggregate and weight information from multiple path-based sources.

```python
class UserProfileEnricher:
    def __init__(self, kg, embedding_model):
        self.kg = kg
        self.embedding_model = embedding_model
        
    def extract_preferences(self, user_id, max_depth=3):
        preferences = defaultdict(float)
        
        # Extract direct preferences
        for edge in self.kg.edges.get(user_id, []):
            if edge['relation'] == 'likes':
                preferences[edge['target']] += edge['weight']
        
        # Extract indirect preferences through paths
        for node_id in self.kg.nodes:
            paths = find_paths(self.kg, user_id, node_id, max_depth)
            if paths:
                path_embeddings = self.embedding_model(paths)
                preference_score = torch.mean(path_embeddings, dim=0).item()
                preferences[node_id] += preference_score
                
        return dict(sorted(preferences.items(), key=lambda x: x[1], reverse=True))

# Example usage
enricher = UserProfileEnricher(kg, PathEmbedding(1000, 50))
user_preferences = enricher.extract_preferences('user_1')
```

Slide 7: Relevance Scoring

The mathematical foundation for computing recommendation relevance combines path-based features with embedding similarities. This implementation shows the core scoring mechanism.

```python
class RelevanceScorer:
    def __init__(self, alpha=0.7, beta=0.3):
        self.alpha = alpha  # Path-based weight
        self.beta = beta    # Embedding-based weight
        
    def compute_score(self, path_score, embedding_similarity):
        """
        Computes final relevance score using the formula:
        $$score = \alpha \cdot path\_score + \beta \cdot embedding\_similarity$$
        """
        return self.alpha * path_score + self.beta * embedding_similarity
    
    def batch_score(self, candidate_items, user_profile, path_scores, embeddings):
        scores = {}
        user_embedding = embeddings[user_profile]
        
        for item in candidate_items:
            path_score = path_scores.get(item, 0)
            item_embedding = embeddings[item]
            embedding_similarity = torch.cosine_similarity(
                user_embedding, item_embedding, dim=0
            ).item()
            
            scores[item] = self.compute_score(path_score, embedding_similarity)
            
        return scores

# Example usage
scorer = RelevanceScorer()
relevance_scores = scorer.batch_score(
    candidate_items=['item_1', 'item_2'],
    user_profile='user_1',
    path_scores={'item_1': 0.8, 'item_2': 0.6},
    embeddings={'user_1': torch.randn(128), 'item_1': torch.randn(128), 'item_2': torch.randn(128)}
)
```

Slide 8: Performance Evaluation Metrics

Implementing comprehensive evaluation metrics for KGLA systems requires consideration of both path relevance and recommendation accuracy. This code demonstrates key metric calculations specific to knowledge graph-enhanced recommendations.

```python
class KGLAEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def compute_path_diversity(self, recommended_paths):
        unique_relations = set()
        for path in recommended_paths:
            relations = [edge[1] for edge in path]
            unique_relations.update(relations)
        return len(unique_relations)
    
    def evaluate(self, predictions, ground_truth, recommended_paths):
        # Precision at K
        k = len(predictions)
        hits = sum(1 for item in predictions if item in ground_truth)
        precision = hits / k
        
        # Path-aware NDCG
        dcg = sum((1 / np.log2(i + 2)) * (item in ground_truth)
                 for i, item in enumerate(predictions))
        idcg = sum(1 / np.log2(i + 2) 
                  for i in range(min(k, len(ground_truth))))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        # Path diversity score
        diversity = self.compute_path_diversity(recommended_paths)
        
        self.metrics = {
            'precision@k': precision,
            'ndcg': ndcg,
            'path_diversity': diversity
        }
        return self.metrics

# Example usage
evaluator = KGLAEvaluator()
metrics = evaluator.evaluate(
    predictions=['item_1', 'item_2', 'item_3'],
    ground_truth=['item_1', 'item_3', 'item_4'],
    recommended_paths=[
        [('user_1', 'likes', 'genre_1'), ('genre_1', 'contains', 'item_1')],
        [('user_1', 'watched', 'item_2')]
    ]
)
```

Slide 9: Real-world Implementation: Movie Recommendation System

This implementation demonstrates a complete movie recommendation system using KGLA, including data preprocessing, model training, and recommendation generation with the MovieLens dataset.

```python
class MovieKGLA:
    def __init__(self, embedding_dim=128):
        self.kg = KnowledgeGraph()
        self.entity_map = {}
        self.relation_map = {}
        self.embedding_model = None
        self.embedding_dim = embedding_dim
        
    def preprocess_data(self, ratings_df, movies_df):
        # Map entities to indices
        users = ratings_df['userId'].unique()
        movies = movies_df['movieId'].unique()
        genres = movies_df['genres'].str.split('|').explode().unique()
        
        for idx, entity in enumerate(users):
            self.entity_map[f'user_{entity}'] = idx
        for idx, entity in enumerate(movies, len(users)):
            self.entity_map[f'movie_{entity}'] = idx
        for idx, entity in enumerate(genres, len(users) + len(movies)):
            self.entity_map[f'genre_{entity}'] = idx
            
        # Build knowledge graph
        for _, row in ratings_df.iterrows():
            user_node = f'user_{row["userId"]}'
            movie_node = f'movie_{row["movieId"]}'
            self.kg.add_edge(user_node, movie_node, 'rated', row['rating'] / 5.0)
            
        for _, row in movies_df.iterrows():
            movie_node = f'movie_{row["movieId"]}'
            for genre in row['genres'].split('|'):
                genre_node = f'genre_{genre}'
                self.kg.add_edge(movie_node, genre_node, 'belongs_to', 1.0)
                
    def train(self):
        num_entities = len(self.entity_map)
        num_relations = len(set(edge['relation'] for edges in self.kg.edges.values() 
                                                for edge in edges))
        self.embedding_model = PathEmbedding(num_entities, num_relations, 
                                           self.embedding_dim)
```

Slide 10: Source Code for Movie Recommendation System (Continued)

```python
    def recommend_movies(self, user_id, n_recommendations=5):
        user_node = f'user_{user_id}'
        recommendations = []
        
        # Get all movie nodes
        movie_nodes = [node for node in self.kg.nodes 
                      if node.startswith('movie_')]
        
        for movie_node in movie_nodes:
            # Find all paths between user and movie
            paths = find_paths(self.kg, user_node, movie_node)
            if paths:
                # Generate path embeddings
                path_embeddings = self.embedding_model(paths)
                
                # Compute relevance score
                relevance = torch.mean(path_embeddings).item()
                
                # Get movie metadata
                movie_id = movie_node.split('_')[1]
                recommendations.append((movie_id, relevance))
                
        # Sort and return top N recommendations
        return sorted(recommendations, key=lambda x: x[1], 
                     reverse=True)[:n_recommendations]

# Example usage with MovieLens data
import pandas as pd

ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv')

movie_kgla = MovieKGLA()
movie_kgla.preprocess_data(ratings_df, movies_df)
movie_kgla.train()

# Get recommendations for user 1
recommendations = movie_kgla.recommend_movies(1)
```

Slide 11: Results Analysis for Movie Recommendation System

The following implementation provides comprehensive analysis tools for evaluating the performance of the KGLA movie recommender system, including metrics visualization and path explanation generation.

```python
class ResultsAnalyzer:
    def __init__(self, kg, movie_data):
        self.kg = kg
        self.movie_data = movie_data
        self.metrics_history = defaultdict(list)
        
    def analyze_recommendation_paths(self, user_id, recommendations):
        path_analysis = {}
        for movie_id, score in recommendations:
            paths = find_paths(self.kg, f'user_{user_id}', f'movie_{movie_id}')
            path_analysis[movie_id] = {
                'num_paths': len(paths),
                'avg_length': np.mean([len(p) for p in paths]),
                'relevance_score': score,
                'movie_title': self.movie_data[
                    self.movie_data['movieId'] == int(movie_id)
                ]['title'].values[0]
            }
        return path_analysis

    def calculate_metrics_over_time(self, recommendations, ground_truth):
        precision = len(set(r[0] for r in recommendations) & set(ground_truth)) / len(recommendations)
        coverage = len(set(r[0] for r in recommendations)) / len(self.movie_data)
        
        self.metrics_history['precision'].append(precision)
        self.metrics_history['coverage'].append(coverage)
        
        return {
            'precision': precision,
            'coverage': coverage,
            'avg_relevance': np.mean([r[1] for r in recommendations])
        }

# Example usage
analyzer = ResultsAnalyzer(movie_kgla.kg, movies_df)
results = analyzer.analyze_recommendation_paths(1, recommendations)
metrics = analyzer.calculate_metrics_over_time(recommendations, ground_truth_movies)

for movie_id, analysis in results.items():
    print(f"Movie: {analysis['movie_title']}")
    print(f"Number of paths: {analysis['num_paths']}")
    print(f"Average path length: {analysis['avg_length']:.2f}")
    print(f"Relevance score: {analysis['relevance_score']:.3f}\n")
```

Slide 12: Real-world Implementation: E-commerce Product Recommendations

This implementation showcases KGLA application in e-commerce, incorporating product categories, user browsing history, and purchase patterns into the knowledge graph structure.

```python
class EcommerceKGLA:
    def __init__(self, embedding_dim=128):
        self.kg = KnowledgeGraph()
        self.path_translator = PathTranslator()
        self.product_embeddings = {}
        
    def build_product_graph(self, products_df, interactions_df):
        # Add product nodes with hierarchical categories
        for _, product in products_df.iterrows():
            self.kg.add_node(f'product_{product["id"]}', {
                'title': product['title'],
                'category': product['category'],
                'subcategory': product['subcategory']
            })
            
            # Add category relationships
            self.kg.add_edge(
                f'product_{product["id"]}',
                f'category_{product["category"]}',
                'belongs_to'
            )
            self.kg.add_edge(
                f'category_{product["category"]}',
                f'subcategory_{product["subcategory"]}',
                'contains'
            )
        
        # Add user interaction edges
        for _, interaction in interactions_df.iterrows():
            self.kg.add_edge(
                f'user_{interaction["user_id"]}',
                f'product_{interaction["product_id"]}',
                interaction['interaction_type'],
                interaction['interaction_strength']
            )
            
    def generate_product_embeddings(self):
        paths = []
        for product_id in self.product_nodes:
            category_paths = find_paths(
                self.kg, 
                product_id, 
                f'category_{self.kg.nodes[product_id]["category"]}'
            )
            paths.extend(category_paths)
        
        path_embedder = PathEmbedding(
            num_entities=len(self.kg.nodes),
            num_relations=len(self.relation_types)
        )
        self.product_embeddings = path_embedder(paths)
```

Slide 13: Source Code for E-commerce Product Recommendations (Continued)

```python
    def recommend_products(self, user_id, n_recommendations=5):
        def calculate_similarity_score(product_embedding, user_history_embedding):
            return torch.cosine_similarity(
                product_embedding, 
                user_history_embedding, 
                dim=0
            ).item()
        
        # Get user history embeddings
        user_history = self.get_user_history(user_id)
        user_history_embedding = torch.mean(
            torch.stack([
                self.product_embeddings[p_id] 
                for p_id in user_history
            ]), 
            dim=0
        )
        
        recommendations = []
        for product_id, embedding in self.product_embeddings.items():
            if product_id not in user_history:
                similarity = calculate_similarity_score(
                    embedding, 
                    user_history_embedding
                )
                paths = find_paths(
                    self.kg, 
                    f'user_{user_id}', 
                    f'product_{product_id}'
                )
                path_score = max([self.compute_path_score(p) for p in paths])
                final_score = 0.7 * similarity + 0.3 * path_score
                recommendations.append((product_id, final_score))
                
        return sorted(recommendations, key=lambda x: x[1], 
                     reverse=True)[:n_recommendations]
    
    def explain_recommendation(self, user_id, product_id):
        paths = find_paths(self.kg, f'user_{user_id}', f'product_{product_id}')
        explanations = []
        
        for path in paths:
            explanation = self.path_translator.translate_path(
                path,
                self.kg.nodes
            )
            relevance = self.compute_path_score(path)
            explanations.append({
                'explanation': explanation,
                'relevance': relevance
            })
            
        return sorted(explanations, key=lambda x: x['relevance'], reverse=True)

# Example usage
ecommerce_kgla = EcommerceKGLA()
ecommerce_kgla.build_product_graph(products_df, interactions_df)
ecommerce_kgla.generate_product_embeddings()

recommendations = ecommerce_kgla.recommend_products(user_id=123)
for product_id, score in recommendations:
    explanations = ecommerce_kgla.explain_recommendation(123, product_id)
    print(f"\nProduct {product_id}: Score {score:.3f}")
    print(f"Top explanation: {explanations[0]['explanation']}")
```

Slide 14: Performance Visualization and Analysis

This implementation provides tools for visualizing KGLA performance metrics and analyzing the effectiveness of path-based recommendations compared to traditional methods.

```python
class KGLAVisualizer:
    def __init__(self, metrics_history):
        self.metrics_history = metrics_history
        
    def plot_metrics_comparison(self, baseline_metrics):
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot precision comparison
        epochs = range(len(self.metrics_history['precision']))
        ax1.plot(epochs, self.metrics_history['precision'], 
                label='KGLA', marker='o')
        ax1.plot(epochs, baseline_metrics['precision'], 
                label='Baseline', marker='s')
        ax1.set_title('Precision@K Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Precision')
        ax1.legend()
        
        # Plot path diversity
        ax2.plot(epochs, self.metrics_history['path_diversity'], 
                label='Path Diversity', color='green', marker='o')
        ax2.set_title('Path Diversity Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Unique Paths')
        
        plt.tight_layout()
        return fig

    def analyze_path_patterns(self):
        path_patterns = defaultdict(int)
        total_paths = 0
        
        for epoch_paths in self.metrics_history['paths']:
            for path in epoch_paths:
                pattern = tuple(edge[1] for edge in path)  # Get relation types
                path_patterns[pattern] += 1
                total_paths += 1
        
        # Calculate pattern frequencies
        pattern_frequencies = {
            pattern: count/total_paths 
            for pattern, count in path_patterns.items()
        }
        
        return pattern_frequencies

# Example usage
visualizer = KGLAVisualizer(analyzer.metrics_history)
pattern_frequencies = visualizer.analyze_path_patterns()

print("\nMost Common Path Patterns:")
for pattern, freq in sorted(
    pattern_frequencies.items(), 
    key=lambda x: x[1], 
    reverse=True
)[:5]:
    print(f"Pattern: {' -> '.join(pattern)}")
    print(f"Frequency: {freq:.3f}")
```

Slide 15: Additional Resources

*   Knowledge Graph Enhanced Language Agents for Personalized Recommendations: [https://arxiv.org/abs/2310.12541](https://arxiv.org/abs/2310.12541)
*   Graph Neural Networks for Knowledge-Enhanced Language Agents: [https://arxiv.org/abs/2305.15317](https://arxiv.org/abs/2305.15317)
*   Path-Based Reasoning in Knowledge Graphs for Recommendation: [https://arxiv.org/abs/2309.09508](https://arxiv.org/abs/2309.09508)
*   Explaining Knowledge Graph-Based Recommendations: A Survey: [https://arxiv.org/abs/2308.14321](https://arxiv.org/abs/2308.14321)
*   Neural Path Reasoning for Knowledge Graph Completion: [https://arxiv.org/abs/2312.05532](https://arxiv.org/abs/2312.05532)

