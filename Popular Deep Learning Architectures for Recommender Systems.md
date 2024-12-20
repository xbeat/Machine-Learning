## Popular Deep Learning Architectures for Recommender Systems
Slide 1: Wide and Deep Architecture (2016)

The Wide and Deep architecture combines a wide linear model with a deep neural network to capture both memorization and generalization in recommender systems. This approach allows the model to learn both broad and specific feature interactions.

```python
import tensorflow as tf

# Wide component
wide_inputs = tf.keras.layers.Input(shape=(10,))
wide_output = tf.keras.layers.Dense(1)(wide_inputs)

# Deep component
deep_inputs = tf.keras.layers.Input(shape=(20,))
x = tf.keras.layers.Dense(64, activation='relu')(deep_inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)
deep_output = tf.keras.layers.Dense(1)(x)

# Combine wide and deep outputs
combined_output = tf.keras.layers.Add()([wide_output, deep_output])

model = tf.keras.Model(inputs=[wide_inputs, deep_inputs], outputs=combined_output)
model.compile(optimizer='adam', loss='mse')

# Example usage
import numpy as np
wide_features = np.random.rand(100, 10)
deep_features = np.random.rand(100, 20)
targets = np.random.rand(100, 1)

model.fit([wide_features, deep_features], targets, epochs=5, batch_size=32)
```

Slide 2: Cross Features in Wide and Deep

Cross features in the Wide and Deep model allow for explicit feature interactions. They help capture specific correlations in the data that might be overlooked by the deep component alone.

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Sample data
data = pd.DataFrame({
    'user_id': [1, 2, 3, 4],
    'item_id': ['A', 'B', 'C', 'A'],
    'category': ['Electronics', 'Books', 'Electronics', 'Books']
})

# Create cross feature
data['user_category'] = data['user_id'].astype(str) + '_' + data['category']

# One-hot encode cross feature
encoder = OneHotEncoder(sparse=False)
cross_feature_encoded = encoder.fit_transform(data[['user_category']])

print("Original data:")
print(data)
print("\nCross feature encoded:")
print(cross_feature_encoded)
```

Slide 3: Deep Factorization Machine (DeepFM, 2017)

DeepFM combines factorization machines with deep neural networks. It models low-order feature interactions like FM and learns high-order feature interactions like DNN.

```python
import tensorflow as tf

class DeepFM(tf.keras.Model):
    def __init__(self, feature_size, field_size, embedding_size):
        super(DeepFM, self).__init__()
        self.embedding = tf.keras.layers.Embedding(feature_size, embedding_size)
        self.fm = FactorizationMachine()
        self.dnn = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        embeddings = self.embedding(inputs)
        fm_output = self.fm(embeddings)
        dnn_output = self.dnn(tf.reshape(embeddings, (-1, embeddings.shape[1] * embeddings.shape[2])))
        return fm_output + dnn_output

class FactorizationMachine(tf.keras.layers.Layer):
    def __init__(self):
        super(FactorizationMachine, self).__init__()

    def call(self, inputs):
        square_of_sum = tf.square(tf.reduce_sum(inputs, axis=1))
        sum_of_square = tf.reduce_sum(tf.square(inputs), axis=1)
        return 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=1, keepdims=True)

# Example usage
model = DeepFM(1000, 10, 8)
sample_input = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
output = model(sample_input)
print(f"Output shape: {output.shape}")
```

Slide 4: Neural Collaborative Filtering (NCF, 2017)

NCF leverages the power of neural networks for collaborative filtering. It learns the user-item interaction function using neural architectures.

```python
import tensorflow as tf

class NCF(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size):
        super(NCF, self).__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_size)
        self.fc_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        user_input, item_input = inputs
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        vector = tf.concat([user_embedded, item_embedded], axis=-1)
        return self.fc_layers(vector)

# Example usage
num_users, num_items = 1000, 500
model = NCF(num_users, num_items, embedding_size=32)
user_ids = tf.constant([1, 2, 3])
item_ids = tf.constant([10, 20, 30])
predictions = model([user_ids, item_ids])
print(f"Predictions shape: {predictions.shape}")
```

Slide 5: Deep and Cross Network (DCN, 2017)

DCN introduces a novel cross network that efficiently captures feature interactions of bounded degrees. It combines a cross network with a deep network.

```python
import tensorflow as tf

class CrossLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        self.w = self.add_weight(shape=(input_dim, 1), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(input_dim, 1), initializer="zeros", trainable=True)

    def call(self, x0, x):
        x0 = tf.expand_dims(x0, axis=-1)
        x = tf.expand_dims(x, axis=-1)
        cross = tf.matmul(tf.matmul(x0, tf.transpose(x, [0, 2, 1])), self.w) + self.b
        return tf.squeeze(cross, axis=-1) + x

class DCN(tf.keras.Model):
    def __init__(self, input_dim, num_cross_layers, num_deep_layers):
        super(DCN, self).__init__()
        self.cross_layers = [CrossLayer(input_dim) for _ in range(num_cross_layers)]
        self.deep_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu') for _ in range(num_deep_layers)
        ] + [tf.keras.layers.Dense(1)])

    def call(self, inputs):
        cross_output = inputs
        for cross_layer in self.cross_layers:
            cross_output = cross_layer(inputs, cross_output)
        deep_output = self.deep_layers(inputs)
        return tf.keras.layers.Add()([cross_output, deep_output])

# Example usage
model = DCN(input_dim=10, num_cross_layers=3, num_deep_layers=2)
sample_input = tf.random.normal((32, 10))
output = model(sample_input)
print(f"Output shape: {output.shape}")
```

Slide 6: AutoInt (2019)

AutoInt uses a multi-head self-attention mechanism to model feature interactions automatically. It captures both low-order and high-order feature interactions efficiently.

```python
import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        return self.dense(output)

class AutoInt(tf.keras.Model):
    def __init__(self, feature_size, embedding_size, num_heads, num_layers):
        super(AutoInt, self).__init__()
        self.embedding = tf.keras.layers.Embedding(feature_size, embedding_size)
        self.attention_layers = [MultiHeadAttention(embedding_size, num_heads) for _ in range(num_layers)]
        self.output_layer = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        x = self.embedding(inputs)
        for attention_layer in self.attention_layers:
            x = attention_layer(x, x, x)
        return self.output_layer(tf.reduce_sum(x, axis=1))

# Example usage
model = AutoInt(feature_size=1000, embedding_size=16, num_heads=2, num_layers=3)
sample_input = tf.constant([[1, 2, 3, 4, 5]])
output = model(sample_input)
print(f"Output shape: {output.shape}")
```

Slide 7: Deep Learning Recommendation Model (DLRM, 2019)

DLRM combines embedding layers for categorical features with MLPs for continuous features. It uses dot product interactions between embeddings to capture feature interactions.

```python
import tensorflow as tf

class DLRM(tf.keras.Model):
    def __init__(self, num_categorical_features, embedding_dim, num_dense_features, bottom_stack, top_stack):
        super(DLRM, self).__init__()
        self.embeddings = [tf.keras.layers.Embedding(num_categorical_features, embedding_dim) for _ in range(num_categorical_features)]
        self.bottom_stack = tf.keras.Sequential(bottom_stack)
        self.top_stack = tf.keras.Sequential(top_stack)
        
    def call(self, inputs):
        dense_features, sparse_features = inputs
        
        # Process dense features
        dense_output = self.bottom_stack(dense_features)
        
        # Process sparse features
        embeddings = [embedding(sparse_features[:, i]) for i, embedding in enumerate(self.embeddings)]
        
        # Compute dot product interactions
        interactions = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                interactions.append(tf.reduce_sum(embeddings[i] * embeddings[j], axis=-1, keepdims=True))
        
        # Concatenate all features
        all_features = tf.concat([dense_output] + embeddings + interactions, axis=1)
        
        # Final MLP
        return self.top_stack(all_features)

# Example usage
num_categorical_features = 5
embedding_dim = 16
num_dense_features = 13

bottom_stack = [
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(embedding_dim)
]

top_stack = [
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
]

model = DLRM(num_categorical_features, embedding_dim, num_dense_features, bottom_stack, top_stack)

# Sample input
dense_input = tf.random.normal((32, num_dense_features))
sparse_input = tf.random.uniform((32, num_categorical_features), maxval=100, dtype=tf.int32)

output = model([dense_input, sparse_input])
print(f"Output shape: {output.shape}")
```

Slide 8: DCN V2 (2020)

DCN V2 enhances the original Deep & Cross Network by introducing a more efficient cross network that allows for learning feature interactions more effectively.

```python
import tensorflow as tf

class CrossNetV2(tf.keras.layers.Layer):
    def __init__(self, layer_num, input_dim):
        super(CrossNetV2, self).__init__()
        self.layer_num = layer_num
        self.kernels = [self.add_weight(name=f'kernel_{i}',
                                        shape=(input_dim, input_dim),
                                        initializer='glorot_uniform',
                                        trainable=True) for i in range(self.layer_num)]
        self.biases = [self.add_weight(name=f'bias_{i}',
                                       shape=(input_dim, 1),
                                       initializer='zeros',
                                       trainable=True) for i in range(self.layer_num)]
    
    def call(self, inputs):
        x0 = tf.expand_dims(inputs, axis=-1)
        xl = x0
        for i in range(self.layer_num):
            xl_w = tf.matmul(self.kernels[i], xl)
            xl = tf.matmul(x0, xl_w) + self.biases[i] + xl
        return tf.squeeze(xl, axis=-1)

class DCNV2(tf.keras.Model):
    def __init__(self, input_dim, cross_layers, deep_layers):
        super(DCNV2, self).__init__()
        self.cross_net = CrossNetV2(cross_layers, input_dim)
        self.deep_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu') for _ in range(deep_layers)
        ] + [tf.keras.layers.Dense(1)])
    
    def call(self, inputs):
        cross_output = self.cross_net(inputs)
        deep_output = self.deep_net(inputs)
        return tf.keras.layers.Add()([cross_output, deep_output])

# Example usage
input_dim = 16
model = DCNV2(input_dim, cross_layers=3, deep_layers=2)
sample_input = tf.random.normal((32, input_dim))
output = model(sample_input)
print(f"Output shape: {output.shape}")
```

Slide 9: Deep Hierarchical Embedding Network (DHEN, 2022)

DHEN introduces a hierarchical embedding structure to capture both coarse-grained and fine-grained feature interactions in recommender systems.

```python
import tensorflow as tf

class HierarchicalEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_categories, embedding_dim, levels):
        super(HierarchicalEmbedding, self).__init__()
        self.levels = levels
        self.embeddings = [tf.keras.layers.Embedding(num_categories, embedding_dim // (2**i)) 
                           for i in range(levels)]
    
    def call(self, inputs):
        embeddings = [embedding(inputs) for embedding in self.embeddings]
        return tf.concat(embeddings, axis=-1)

class DHEN(tf.keras.Model):
    def __init__(self, num_categories, embedding_dim, levels, num_dense_features):
        super(DHEN, self).__init__()
        self.hierarchical_embedding = HierarchicalEmbedding(num_categories, embedding_dim, levels)
        self.dense_layer = tf.keras.layers.Dense(embedding_dim)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    
    def call(self, inputs):
        categorical_inputs, dense_inputs = inputs
        cat_embeddings = self.hierarchical_embedding(categorical_inputs)
        dense_embeddings = self.dense_layer(dense_inputs)
        combined = tf.concat([cat_embeddings, dense_embeddings], axis=1)
        return self.mlp(combined)

# Example usage
num_categories = 1000
embedding_dim = 32
levels = 3
num_dense_features = 10

model = DHEN(num_categories, embedding_dim, levels, num_dense_features)

categorical_input = tf.random.uniform((64, 5), maxval=num_categories, dtype=tf.int32)
dense_input = tf.random.normal((64, num_dense_features))

output = model([categorical_input, dense_input])
print(f"Output shape: {output.shape}")
```

Slide 10: Graph Deep Collaborative Network (GDCN, 2023)

GDCN leverages graph neural networks to capture complex interactions between users and items in recommender systems.

```python
import tensorflow as tf
import tensorflow_gnn as tfgnn

class GDCN(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim):
        super(GDCN, self).__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)
        self.graph_conv = tfgnn.keras.layers.GraphConvolution(
            units=embedding_dim,
            activation='relu',
            use_bias=True
        )
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    
    def call(self, inputs):
        graph, user_ids, item_ids = inputs
        
        # Embed users and items
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Apply graph convolution
        graph_output = self.graph_conv(graph)
        
        # Combine embeddings and graph output
        combined = tf.concat([user_emb, item_emb, graph_output], axis=1)
        
        # Final prediction
        return self.mlp(combined)

# Example usage (Note: This is a simplified example and requires setting up a proper graph structure)
num_users, num_items = 1000, 5000
embedding_dim = 32

model = GDCN(num_users, num_items, embedding_dim)

# Assuming we have a graph, user_ids, and item_ids
graph = tfgnn.Graph(...)  # This needs to be properly constructed
user_ids = tf.constant([1, 2, 3])
item_ids = tf.constant([10, 20, 30])

output = model([graph, user_ids, item_ids])
print(f"Output shape: {output.shape}")
```

Slide 11: Graph Neural Networks in RecSys

Graph Neural Networks (GNNs) have become increasingly popular in recommender systems due to their ability to capture complex relationships between users and items.

```python
import tensorflow as tf
import tensorflow_gnn as tfgnn

class GNNLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(GNNLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(units, activation='relu')
    
    def call(self, graph):
        # Aggregate messages from neighbors
        messages = tfgnn.broadcast_node_to_edges(graph, 'user', 'rating', 'item')
        aggregated = tfgnn.pool_edges_to_node(graph, 'item', 'rating', 'mean')
        
        # Update node representations
        updated = self.dense(aggregated)
        return graph.replace_features(nodes={'item': updated})

class GNNRecommender(tf.keras.Model):
    def __init__(self, num_layers):
        super(GNNRecommender, self).__init__()
        self.gnn_layers = [GNNLayer(64) for _ in range(num_layers)]
        self.final_layer = tf.keras.layers.Dense(1)
    
    def call(self, graph):
        for layer in self.gnn_layers:
            graph = layer(graph)
        
        # Get final item embeddings
        item_embeddings = graph.node_sets['item']['hidden_state']
        
        # Make predictions
        return self.final_layer(item_embeddings)

# Example usage (Note: This is a simplified example)
model = GNNRecommender(num_layers=3)

# Assuming we have a properly constructed graph
graph = tfgnn.Graph(...)  # This needs to be properly constructed

predictions = model(graph)
print(f"Predictions shape: {predictions.shape}")
```

Slide 12: Two-Tower Model in RecSys

The Two-Tower model is a popular architecture for large-scale retrieval tasks in recommender systems. It separately encodes user and item features, allowing for efficient retrieval.

```python
import tensorflow as tf

class TwoTowerModel(tf.keras.Model):
    def __init__(self, user_vocab_size, item_vocab_size, embedding_dim):
        super(TwoTowerModel, self).__init__()
        
        # User tower
        self.user_embedding = tf.keras.layers.Embedding(user_vocab_size, embedding_dim)
        self.user_dense = tf.keras.layers.Dense(64, activation='relu')
        
        # Item tower
        self.item_embedding = tf.keras.layers.Embedding(item_vocab_size, embedding_dim)
        self.item_dense = tf.keras.layers.Dense(64, activation='relu')
    
    def call(self, inputs):
        user_input, item_input = inputs
        
        # User tower
        user_embedded = self.user_embedding(user_input)
        user_vector = self.user_dense(user_embedded)
        
        # Item tower
        item_embedded = self.item_embedding(item_input)
        item_vector = self.item_dense(item_embedded)
        
        # Compute dot product similarity
        return tf.reduce_sum(user_vector * item_vector, axis=1)

# Example usage
user_vocab_size = 10000
item_vocab_size = 50000
embedding_dim = 32

model = TwoTowerModel(user_vocab_size, item_vocab_size, embedding_dim)

# Sample batch
batch_size = 64
user_ids = tf.random.uniform((batch_size,), maxval=user_vocab_size, dtype=tf.int32)
item_ids = tf.random.uniform((batch_size,), maxval=item_vocab_size, dtype=tf.int32)

similarities = model([user_ids, item_ids])
print(f"Similarities shape: {similarities.shape}")
```

Slide 13: Real-life Example: Movie Recommendation

Let's implement a simple collaborative filtering model for movie recommendations using the Neural Collaborative Filtering (NCF) architecture.

```python
import tensorflow as tf
import numpy as np

class MovieRecommender(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_size):
        super(MovieRecommender, self).__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)
        self.movie_embedding = tf.keras.layers.Embedding(num_movies, embedding_size)
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        user_id, movie_id = inputs
        user_embedded = self.user_embedding(user_id)
        movie_embedded = self.movie_embedding(movie_id)
        concatenated = tf.concat([user_embedded, movie_embedded], axis=-1)
        return self.dense_layers(concatenated)

# Example usage
num_users = 1000
num_movies = 5000
embedding_size = 32

model = MovieRecommender(num_users, num_movies, embedding_size)

# Sample data
user_ids = np.array([1, 2, 3, 4, 5])
movie_ids = np.array([10, 20, 30, 40, 50])
ratings = np.array([4.5, 3.0, 5.0, 2.0, 4.0])

# Train the model
model.compile(optimizer='adam', loss='mse')
model.fit([user_ids, movie_ids], ratings, epochs=10, verbose=0)

# Make predictions
new_user_ids = np.array([1, 2, 3])
new_movie_ids = np.array([15, 25, 35])
predictions = model.predict([new_user_ids, new_movie_ids])

print("Predicted ratings:")
for user, movie, rating in zip(new_user_ids, new_movie_ids, predictions):
    print(f"User {user}, Movie {movie}: Predicted rating {rating[0]:.2f}")
```

Slide 14: Real-life Example: Product Recommendation

Let's implement a simple Deep & Cross Network (DCN) for product recommendations in an e-commerce setting.

```python
import tensorflow as tf
import numpy as np

class ProductRecommender(tf.keras.Model):
    def __init__(self, num_products, num_categories, embedding_dim):
        super(ProductRecommender, self).__init__()
        self.product_embedding = tf.keras.layers.Embedding(num_products, embedding_dim)
        self.category_embedding = tf.keras.layers.Embedding(num_categories, embedding_dim)
        self.cross_layer = tf.keras.layers.Dense(embedding_dim * 2)
        self.deep_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        product_id, category_id, user_features = inputs
        product_emb = self.product_embedding(product_id)
        category_emb = self.category_embedding(category_id)
        combined = tf.concat([product_emb, category_emb, user_features], axis=-1)
        
        # Cross network
        cross_output = self.cross_layer(combined)
        
        # Deep network
        deep_output = self.deep_layers(combined)
        
        return tf.keras.layers.Add()([cross_output, deep_output])

# Example usage
num_products = 10000
num_categories = 100
embedding_dim = 16
user_feature_dim = 10

model = ProductRecommender(num_products, num_categories, embedding_dim)

# Sample data
batch_size = 32
product_ids = np.random.randint(0, num_products, batch_size)
category_ids = np.random.randint(0, num_categories, batch_size)
user_features = np.random.random((batch_size, user_feature_dim))
purchase_probability = np.random.random(batch_size)

# Train the model
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit([product_ids, category_ids, user_features], purchase_probability, epochs=5, verbose=0)

# Make predictions
new_product_ids = np.array([5, 15, 25])
new_category_ids = np.array([1, 2, 3])
new_user_features = np.random.random((3, user_feature_dim))

predictions = model.predict([new_product_ids, new_category_ids, new_user_features])

print("Predicted purchase probabilities:")
for product, category, prob in zip(new_product_ids, new_category_ids, predictions):
    print(f"Product {product}, Category {category}: Purchase probability {prob[0]:.2f}")
```

Slide 15: Additional Resources

For those interested in diving deeper into recommender system architectures, here are some valuable resources:

1. Wide & Deep Learning for Recommender Systems ArXiv: [https://arxiv.org/abs/1606.07792](https://arxiv.org/abs/1606.07792)
2. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction ArXiv: [https://arxiv.org/abs/1703.04247](https://arxiv.org/abs/1703.04247)
3. Neural Collaborative Filtering ArXiv: [https://arxiv.org/abs/1708.05031](https://arxiv.org/abs/1708.05031)
4. Deep & Cross Network for Ad Click Predictions ArXiv: [https://arxiv.org/abs/1708.05123](https://arxiv.org/abs/1708.05123)
5. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks ArXiv: [https://arxiv.org/abs/1810.11921](https://arxiv.org/abs/1810.11921)
6. Deep Learning Recommendation Model for Personalization and Recommendation Systems ArXiv: [https://arxiv.org/abs/1906.00091](https://arxiv.org/abs/1906.00091)
7. DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems ArXiv: [https://arxiv.org/abs/2008.13535](https://arxiv.org/abs/2008.13535)

These papers provide in-depth explanations of the architectures discussed in this slideshow and offer insights into their implementation and performance in real-world scenarios.

