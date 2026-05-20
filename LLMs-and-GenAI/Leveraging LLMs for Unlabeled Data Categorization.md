## Leveraging LLMs for Unlabeled Data Categorization
Slide 1: Zero-Shot Classification with LLMs

Zero-shot learning enables classification of unlabeled data without prior training by leveraging pre-trained language models through carefully crafted prompts. This approach utilizes the semantic understanding inherent in large language models to categorize text into arbitrary classes.

```python
import openai
import numpy as np
from typing import List, Dict

def zero_shot_classify(texts: List[str], categories: List[str]) -> Dict[str, str]:
    classifications = {}
    prompt_template = """Text: {text}
    Categories: {categories}
    Choose the most appropriate category for the text above:"""
    
    for text in texts:
        prompt = prompt_template.format(
            text=text,
            categories=", ".join(categories)
        )
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50,
            temperature=0.3
        )
        classifications[text] = response.choices[0].text.strip()
    
    return classifications

# Example usage
texts = ["The stock market crashed today", "New ML model achieves SOTA results"]
categories = ["Finance", "Technology", "Politics"]
results = zero_shot_classify(texts, categories)
```

Slide 2: Embedding-Based Clustering

Transforming text into dense vector representations allows for unsupervised clustering of unlabeled data. This method leverages semantic similarities captured by embeddings to group similar content without predefined categories.

```python
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer

def cluster_embeddings(texts: List[str], n_clusters: int):
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    embeddings = model.encode(texts)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Group texts by cluster
    clustered_texts = {}
    for text, cluster in zip(texts, clusters):
        if cluster not in clustered_texts:
            clustered_texts[cluster] = []
        clustered_texts[cluster].append(text)
    
    return clustered_texts
```

Slide 3: Self-Training with LLM Confidence Scores

This advanced technique leverages LLM confidence scores to iteratively label high-confidence examples, which are then used to train a lighter classifier. This approach combines the power of LLMs with traditional ML methods.

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def self_training_classifier(labeled_data, unlabeled_data, confidence_threshold=0.8):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(labeled_data['features'], labeled_data['labels'])
    
    while len(unlabeled_data) > 0:
        # Get predictions and confidence scores
        probs = model.predict_proba(unlabeled_data['features'])
        confidence_scores = np.max(probs, axis=1)
        
        # Select high confidence predictions
        high_conf_idx = confidence_scores >= confidence_threshold
        new_labeled = {
            'features': unlabeled_data['features'][high_conf_idx],
            'labels': model.predict(unlabeled_data['features'][high_conf_idx])
        }
        
        # Update datasets
        labeled_data['features'] = np.vstack([labeled_data['features'], 
                                            new_labeled['features']])
        labeled_data['labels'] = np.concatenate([labeled_data['labels'], 
                                               new_labeled['labels']])
        
        # Remove labeled examples from unlabeled set
        unlabeled_data['features'] = unlabeled_data['features'][~high_conf_idx]
    
    return model
```

Slide 4: Contrastive Learning for Representation

Contrastive learning creates robust representations by learning to distinguish between similar and dissimilar examples without explicit labels. This unsupervised approach helps capture meaningful patterns in the data structure.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return F.normalize(self.encoder(x), dim=1)

def contrastive_loss(anchor, positive, temperature=0.07):
    # Compute similarities
    similarity = torch.matmul(anchor, positive.T) / temperature
    
    # Labels are diagonal (each anchor matches with its positive)
    labels = torch.arange(anchor.size(0)).to(anchor.device)
    
    return F.cross_entropy(similarity, labels)
```

Slide 5: Dynamic Few-Shot Learning Pipeline

This approach dynamically selects the most relevant examples from a small labeled set to guide classification of new instances. The pipeline combines semantic similarity with prototype learning for improved accuracy on unlabeled data.

```python
from scipy.spatial.distance import cosine
import numpy as np

class DynamicFewShotLearner:
    def __init__(self, n_shots=5):
        self.n_shots = n_shots
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.support_set = {}
        
    def add_to_support(self, texts, labels):
        embeddings = self.model.encode(texts)
        for embed, label in zip(embeddings, labels):
            if label not in self.support_set:
                self.support_set[label] = []
            self.support_set[label].append(embed)
    
    def predict(self, query_text):
        query_embed = self.model.encode(query_text)
        best_score = float('-inf')
        best_label = None
        
        for label, supports in self.support_set.items():
            similarities = [1 - cosine(query_embed, support) 
                          for support in supports[:self.n_shots]]
            score = np.mean(similarities)
            if score > best_score:
                best_score = score
                best_label = label
                
        return best_label, best_score
```

Slide 6: Iterative Label Propagation

Label propagation extends initial labels to unlabeled data points by exploiting the manifold structure of the data. This method iteratively updates pseudo-labels based on neighborhood relationships in the embedding space.

```python
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

class IterativeLabelPropagation:
    def __init__(self, n_neighbors=5, max_iter=1000, tol=1e-3):
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.tol = tol
        
    def fit_predict(self, embeddings, initial_labels):
        # Build similarity matrix
        W = rbf_kernel(embeddings)
        np.fill_diagonal(W, 0)
        
        # Normalize similarity matrix
        D = np.diag(W.sum(axis=1))
        T = np.linalg.inv(D) @ W
        
        # Initialize label matrix
        Y = np.zeros((len(embeddings), len(np.unique(initial_labels))))
        for idx, label in enumerate(initial_labels):
            if label != -1:  # -1 indicates unlabeled
                Y[idx, label] = 1
                
        Y_prev = Y.copy()
        
        # Iterate until convergence
        for _ in range(self.max_iter):
            Y = T @ Y
            Y[initial_labels != -1] = Y_prev[initial_labels != -1]
            
            if np.abs(Y - Y_prev).max() < self.tol:
                break
                
            Y_prev = Y.copy()
            
        return Y.argmax(axis=1)
```

Slide 7: Hierarchical Topic Discovery

This technique automatically discovers hierarchical relationships in unlabeled text data by combining embedding clustering with tree-based topic modeling. The approach reveals natural category structures without predefined labels.

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class HierarchicalTopicDiscovery:
    def __init__(self, max_depth=3, min_cluster_size=5):
        self.max_depth = max_depth
        self.min_cluster_size = min_cluster_size
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
    def build_topic_tree(self, texts, depth=0):
        if (depth >= self.max_depth or 
            len(texts) < self.min_cluster_size * 2):
            return {'texts': texts}
            
        # Create document vectors
        vectors = self.vectorizer.fit_transform(texts).toarray()
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(n_clusters=2)
        labels = clustering.fit_predict(vectors)
        
        # Split texts by cluster
        clusters = {
            'left': self.build_topic_tree(
                [t for i, t in enumerate(texts) if labels[i] == 0], 
                depth + 1
            ),
            'right': self.build_topic_tree(
                [t for i, t in enumerate(texts) if labels[i] == 1], 
                depth + 1
            )
        }
        
        return clusters
```

Slide 8: Semi-Supervised Embedding Alignment

This advanced technique aligns embeddings from different sources using a small set of anchor points, enabling transfer of labels across domains while preserving the semantic structure of unlabeled data.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class EmbeddingAligner(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.alignment_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, source_embeds):
        return self.alignment_network(source_embeds)

def train_aligner(source_embeds, target_embeds, anchor_pairs, epochs=100):
    model = EmbeddingAligner(source_embeds.shape[1], 256)
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        # Alignment loss for anchor pairs
        source_anchors = source_embeds[anchor_pairs[:, 0]]
        target_anchors = target_embeds[anchor_pairs[:, 1]]
        
        aligned = model(source_anchors)
        alignment_loss = F.mse_loss(aligned, target_anchors)
        
        # Structure preservation loss
        random_source = source_embeds[torch.randint(
            len(source_embeds), (256,))]
        aligned_random = model(random_source)
        structure_loss = preserve_structure_loss(
            random_source, aligned_random)
        
        total_loss = alignment_loss + 0.1 * structure_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    return model

def preserve_structure_loss(original, transformed):
    orig_dist = torch.pdist(original)
    trans_dist = torch.pdist(transformed)
    return F.mse_loss(trans_dist, orig_dist)
```

Slide 9: Neural Topic Modeling with Autoencoders

Neural topic modeling combines the interpretability of traditional topic models with the expressiveness of neural networks. This approach learns latent topic representations while maintaining document reconstruction capability.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralTopicModel(nn.Module):
    def __init__(self, vocab_size, n_topics, hidden_size=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_topics*2)  # Mean and log variance
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(n_topics, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
            nn.Softmax(dim=1)
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        # Encode
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        
        # Sample topic distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decoder(z)
        
        return reconstructed, mu, logvar
```

Slide 10: Probabilistic Label Refinement

This methodology refines initial noisy labels through an iterative probabilistic framework that combines model confidence scores with structural constraints from the data manifold.

```python
import numpy as np
from scipy.special import softmax

class ProbabilisticLabelRefiner:
    def __init__(self, alpha=0.9, max_iter=50):
        self.alpha = alpha
        self.max_iter = max_iter
        
    def refine_labels(self, embeddings, initial_probs):
        n_samples = len(embeddings)
        n_classes = initial_probs.shape[1]
        
        # Compute similarity matrix
        sim_matrix = self._compute_similarities(embeddings)
        
        # Initialize label distributions
        label_dist = initial_probs.copy()
        
        for _ in range(self.max_iter):
            prev_dist = label_dist.copy()
            
            # Propagate labels through similarity graph
            propagated = sim_matrix @ label_dist
            
            # Combine with initial probabilities
            label_dist = (self.alpha * propagated + 
                        (1 - self.alpha) * initial_probs)
            
            # Check convergence
            if np.abs(label_dist - prev_dist).max() < 1e-6:
                break
                
        return label_dist
    
    def _compute_similarities(self, embeddings):
        # Compute pairwise cosine similarities
        dot_product = embeddings @ embeddings.T
        norms = np.linalg.norm(embeddings, axis=1)
        similarities = dot_product / np.outer(norms, norms)
        
        # Apply temperature scaling and normalize
        temp_scaled = similarities / 0.1
        return softmax(temp_scaled, axis=1)
```

Slide 11: Source Code for Cross-Domain Label Transfer

The cross-domain label transfer system enables knowledge transfer between different domains by learning domain-invariant representations and adapting the classification boundary using unlabeled target domain data.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainAdapter(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        
        # Feature extractors for both domains
        self.source_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.target_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Domain discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2)
        )
        
    def forward(self, x, domain='source'):
        if domain == 'source':
            features = self.source_encoder(x)
        else:
            features = self.target_encoder(x)
            
        domain_pred = self.discriminator(features)
        class_pred = self.classifier(features)
        
        return features, class_pred, domain_pred
    
    def transfer_loss(self, source_x, target_x):
        # Get embeddings
        source_f, source_c, source_d = self(source_x, 'source')
        target_f, target_c, target_d = self(target_x, 'target')
        
        # Domain alignment loss (adversarial)
        domain_loss = (F.binary_cross_entropy_with_logits(source_d, 
                      torch.ones_like(source_d)) +
                      F.binary_cross_entropy_with_logits(target_d, 
                      torch.zeros_like(target_d)))
        
        # Feature similarity loss
        similarity_loss = F.mse_loss(
            source_f.mean(0), target_f.mean(0))
        
        return domain_loss + similarity_loss
```

Slide 12: Weak Supervision Framework

This framework combines multiple weak supervision sources to generate high-quality training labels. It uses a generative model to estimate source accuracies and correlations, creating a robust labeling function ensemble.

```python
import numpy as np
from sklearn.mixture import GaussianMixture

class WeakSupervisionAggregator:
    def __init__(self, n_sources, n_classes):
        self.n_sources = n_sources
        self.n_classes = n_classes
        self.source_accuracies = None
        
    def fit(self, weak_labels):
        # Initialize source accuracies
        self.source_accuracies = np.ones(self.n_sources) / self.n_sources
        
        # Estimate source reliabilities using EM
        for _ in range(50):  # Max iterations
            # E-step: Estimate true labels
            predicted_labels = self._aggregate_labels(weak_labels)
            
            # M-step: Update source accuracies
            for s in range(self.n_sources):
                matches = (weak_labels[:, s] == predicted_labels)
                self.source_accuracies[s] = np.mean(matches)
                
    def _aggregate_labels(self, weak_labels):
        weighted_votes = np.zeros((len(weak_labels), self.n_classes))
        
        for s in range(self.n_sources):
            for c in range(self.n_classes):
                mask = (weak_labels[:, s] == c)
                weighted_votes[mask, c] += self.source_accuracies[s]
                
        return np.argmax(weighted_votes, axis=1)
    
    def predict(self, weak_labels):
        if self.source_accuracies is None:
            self.fit(weak_labels)
        return self._aggregate_labels(weak_labels)
```

Slide 13: Real-World Implementation: News Article Categorization

This implementation demonstrates a complete pipeline for categorizing unlabeled news articles using a combination of embedding-based clustering and zero-shot classification with confidence thresholding.

```python
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

class NewsClassifier:
    def __init__(self, confidence_threshold=0.8):
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.zero_shot = pipeline("zero-shot-classification")
        self.confidence_threshold = confidence_threshold
        
    def process_articles(self, articles, candidate_labels):
        # Generate embeddings
        embeddings = self.st_model.encode(articles)
        
        # Cluster similar articles
        clusters = DBSCAN(eps=0.3, min_samples=3).fit(embeddings)
        
        # Classify confident examples
        classifications = []
        for article in articles:
            result = self.zero_shot(article, candidate_labels)
            if result['scores'][0] > self.confidence_threshold:
                classifications.append({
                    'text': article,
                    'label': result['labels'][0],
                    'confidence': result['scores'][0]
                })
            
        # Propagate labels within clusters
        cluster_labels = clusters.labels_
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_mask = cluster_labels == cluster_id
            cluster_articles = np.array(articles)[cluster_mask]
            
            # Find most confident prediction in cluster
            confidences = [self.zero_shot(art, candidate_labels)
                         for art in cluster_articles]
            best_conf = max(c['scores'][0] for c in confidences)
            best_label = confidences[np.argmax(
                [c['scores'][0] for c in confidences])]['labels'][0]
            
            # Assign cluster label
            for art in cluster_articles:
                classifications.append({
                    'text': art,
                    'label': best_label,
                    'confidence': best_conf
                })
                
        return classifications

# Example usage
articles = [
    "Bitcoin price surges to new all-time high",
    "New AI model achieves human-level performance",
    "Global climate summit concludes with new agreements"
]
labels = ["Finance", "Technology", "Environment"]
classifier = NewsClassifier()
results = classifier.process_articles(articles, labels)
```

Slide 14: Additional Resources

*   Understanding Zero-shot Learning and LLMs
    *   [https://arxiv.org/abs/2302.03004](https://arxiv.org/abs/2302.03004)
    *   Search: "Zero-shot learning survey large language models"
*   Neural Topic Modeling and Representation Learning
    *   [https://arxiv.org/abs/2004.03974](https://arxiv.org/abs/2004.03974)
    *   [https://arxiv.org/abs/1901.04553](https://arxiv.org/abs/1901.04553)
*   Weak Supervision and Label Generation
    *   [https://arxiv.org/abs/2202.02382](https://arxiv.org/abs/2202.02382)
    *   Search: "Programmatic weak supervision deep learning"
*   Cross-domain Adaptation for Text Classification
    *   [https://arxiv.org/abs/2010.12352](https://arxiv.org/abs/2010.12352)
    *   Search: "Domain adaptation natural language processing"
*   Label Propagation and Semi-supervised Learning
    *   [https://arxiv.org/abs/2103.14902](https://arxiv.org/abs/2103.14902)
    *   Search: "Graph-based semi-supervised learning text classification"

