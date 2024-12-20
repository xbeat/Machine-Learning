## Siamese Network Architecture in Python
Slide 1: Introduction to Siamese Networks

Siamese networks are neural network architectures designed to compare two inputs and determine their similarity. They are particularly useful for tasks like face recognition, signature verification, and image similarity.

```python
import tensorflow as tf

def siamese_network(input_shape):
    input_a = tf.keras.layers.Input(shape=input_shape)
    input_b = tf.keras.layers.Input(shape=input_shape)
    
    base_network = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu')
    ])
    
    feat_a = base_network(input_a)
    feat_b = base_network(input_b)
    
    distance = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([feat_a, feat_b])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(distance)
    
    return tf.keras.Model(inputs=[input_a, input_b], outputs=output)
```

Slide 2: Architecture of Siamese Networks

Siamese networks consist of two identical subnetworks that share weights. These subnetworks process two different inputs and produce feature vectors, which are then compared to determine similarity.

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_siamese_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw subnetworks
    ax.add_patch(plt.Rectangle((0.1, 0.1), 0.3, 0.8, fill=False))
    ax.add_patch(plt.Rectangle((0.6, 0.1), 0.3, 0.8, fill=False))
    
    # Draw inputs
    ax.text(0.25, 0.95, 'Input A', ha='center')
    ax.text(0.75, 0.95, 'Input B', ha='center')
    
    # Draw feature vectors
    ax.arrow(0.4, 0.5, 0.1, 0, head_width=0.05, head_length=0.05)
    ax.arrow(0.6, 0.5, -0.1, 0, head_width=0.05, head_length=0.05)
    
    # Draw comparison
    ax.add_patch(plt.Circle((0.5, 0.5), 0.1, fill=False))
    ax.text(0.5, 0.5, 'Compare', ha='center', va='center')
    
    # Draw output
    ax.arrow(0.5, 0.4, 0, -0.2, head_width=0.05, head_length=0.05)
    ax.text(0.5, 0.1, 'Similarity Score', ha='center')
    
    ax.axis('off')
    plt.tight_layout()
    plt.show()

visualize_siamese_architecture()
```

Slide 3: Shared Weights Concept

The key feature of Siamese networks is weight sharing between subnetworks. This ensures that similar inputs produce similar feature representations, regardless of which subnetwork processes them.

```python
import tensorflow as tf

def create_shared_network():
    shared_network = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu')
    ])
    
    input_a = tf.keras.layers.Input(shape=(64, 64, 3))
    input_b = tf.keras.layers.Input(shape=(64, 64, 3))
    
    output_a = shared_network(input_a)
    output_b = shared_network(input_b)
    
    return tf.keras.Model(inputs=[input_a, input_b], outputs=[output_a, output_b])

model = create_shared_network()
print(model.summary())
```

Slide 4: Distance Metrics

Siamese networks use various distance metrics to compare feature vectors. Common choices include Euclidean distance, cosine similarity, and Manhattan distance.

```python
import tensorflow as tf

def euclidean_distance(vects):
    x, y = vects
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

def cosine_similarity(vects):
    x, y = vects
    x_norm = tf.nn.l2_normalize(x, axis=1)
    y_norm = tf.nn.l2_normalize(y, axis=1)
    return tf.reduce_sum(x_norm * y_norm, axis=1, keepdims=True)

def manhattan_distance(vects):
    x, y = vects
    return tf.reduce_sum(tf.abs(x - y), axis=1, keepdims=True)

# Usage example
vector1 = tf.constant([[1.0, 2.0, 3.0]])
vector2 = tf.constant([[4.0, 5.0, 6.0]])

print("Euclidean distance:", euclidean_distance([vector1, vector2]).numpy())
print("Cosine similarity:", cosine_similarity([vector1, vector2]).numpy())
print("Manhattan distance:", manhattan_distance([vector1, vector2]).numpy())
```

Slide 5: Loss Functions for Siamese Networks

Siamese networks often use contrastive loss or triplet loss to train the model. These loss functions aim to minimize the distance between similar pairs and maximize the distance between dissimilar pairs.

```python
import tensorflow as tf

def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    return tf.reduce_mean(tf.maximum(basic_loss, 0.0))

# Example usage
y_true = tf.constant([[1], [0]])  # 1 for similar pair, 0 for dissimilar pair
y_pred = tf.constant([[0.2], [0.7]])  # Predicted distances

print("Contrastive loss:", contrastive_loss(y_true, y_pred).numpy())

# For triplet loss
y_true_triplet = tf.constant([[0]])  # Dummy label, not used in triplet loss
y_pred_triplet = tf.constant([[0.1, 0.2, 0.7]])  # [anchor, positive, negative]

print("Triplet loss:", triplet_loss(y_true_triplet, y_pred_triplet).numpy())
```

Slide 6: Data Preparation for Siamese Networks

Preparing data for Siamese networks involves creating pairs or triplets of samples. For pair-based training, we create positive (similar) and negative (dissimilar) pairs.

```python
import numpy as np
import tensorflow as tf

def create_pairs(x, y):
    pairs = []
    labels = []
    n_classes = max(y) + 1

    for d in range(n_classes):
        inds = np.where(y == d)[0]
        for i in range(len(inds)):
            for j in range(i + 1, len(inds)):
                pairs.append([x[inds[i]], x[inds[j]]])
                labels.append(1)
            
            for _ in range(len(inds)):
                inc = np.random.randint(1, n_classes)
                dn = (d + inc) % n_classes
                idx = np.random.randint(len(np.where(y == dn)[0]))
                pairs.append([x[inds[i]], x[np.where(y == dn)[0][idx]]])
                labels.append(0)
    
    return np.array(pairs), np.array(labels)

# Example usage
x = np.random.rand(100, 28, 28, 1)  # 100 images of size 28x28 with 1 channel
y = np.random.randint(0, 10, 100)  # 10 classes

pairs, labels = create_pairs(x, y)
print("Number of pairs:", len(pairs))
print("Number of positive pairs:", np.sum(labels))
print("Number of negative pairs:", len(labels) - np.sum(labels))
```

Slide 7: Training a Siamese Network

Training a Siamese network involves feeding pairs of inputs through the network and updating weights based on the similarity predictions and true labels.

```python
import tensorflow as tf

def train_siamese_network(model, train_data, train_labels, epochs=10, batch_size=32):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = contrastive_loss

    @tf.function
    def train_step(batch_data, batch_labels):
        with tf.GradientTape() as tape:
            predictions = model(batch_data)
            loss = loss_fn(batch_labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        progbar = tf.keras.utils.Progbar(len(train_data))
        
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            loss = train_step(batch_data, batch_labels)
            progbar.add(len(batch_data), values=[("loss", loss)])

# Assuming 'model' is your Siamese network and 'pairs' and 'labels' are your training data
# train_siamese_network(model, pairs, labels)
```

Slide 8: Inference with Siamese Networks

After training, Siamese networks can be used for various tasks such as verification, one-shot learning, or similarity ranking.

```python
import tensorflow as tf
import numpy as np

def siamese_inference(model, image1, image2):
    # Ensure images are in the correct shape
    image1 = np.expand_dims(image1, axis=0)
    image2 = np.expand_dims(image2, axis=0)
    
    # Get the similarity score
    similarity = model.predict([image1, image2])[0][0]
    
    return similarity

def verify_identity(model, known_image, test_image, threshold=0.5):
    similarity = siamese_inference(model, known_image, test_image)
    return similarity > threshold, similarity

# Example usage
# Assuming 'model' is your trained Siamese network
# known_image = ... # Load a known image
# test_image = ... # Load a test image

# is_same, confidence = verify_identity(model, known_image, test_image)
# print(f"Same identity: {is_same}, Confidence: {confidence:.2f}")
```

Slide 9: One-Shot Learning with Siamese Networks

Siamese networks excel at one-shot learning, where they can classify new instances based on just one example per class.

```python
import numpy as np
import tensorflow as tf

def one_shot_classification(model, support_set, support_labels, query_image):
    similarities = []
    
    for support_image in support_set:
        similarity = siamese_inference(model, support_image, query_image)
        similarities.append(similarity)
    
    most_similar_idx = np.argmax(similarities)
    predicted_label = support_labels[most_similar_idx]
    confidence = similarities[most_similar_idx]
    
    return predicted_label, confidence

# Example usage
# Assuming 'model' is your trained Siamese network
# support_set = [...] # List of example images, one per class
# support_labels = [...] # Corresponding labels for the support set
# query_image = ... # The image to classify

# predicted_label, confidence = one_shot_classification(model, support_set, support_labels, query_image)
# print(f"Predicted class: {predicted_label}, Confidence: {confidence:.2f}")
```

Slide 10: Face Verification Example

One common application of Siamese networks is face verification, where the network determines if two face images belong to the same person.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_face_verification_model(input_shape):
    base_network = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu')
    ])
    
    input_a = tf.keras.layers.Input(shape=input_shape)
    input_b = tf.keras.layers.Input(shape=input_shape)
    
    vector_a = base_network(input_a)
    vector_b = base_network(input_b)
    
    distance = tf.keras.layers.Lambda(euclidean_distance)([vector_a, vector_b])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(distance)
    
    return tf.keras.Model(inputs=[input_a, input_b], outputs=output)

# Create and compile the model
input_shape = (96, 96, 3)  # Example input shape for face images
model = create_face_verification_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Example usage (assuming you have training data)
# history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_labels, 
#                     validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
#                     epochs=10, batch_size=32)

# Visualize training history
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
```

Slide 11: Signature Verification Example

Another practical application of Siamese networks is signature verification, where the network determines if a given signature matches a known authentic signature.

```python
def create_signature_verification_model(input_shape):
    base_network = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu')
    ])
    
    input_a = tf.keras.layers.Input(shape=input_shape)
    input_b = tf.keras.layers.Input(shape=input_shape)
    
    vector_a = base_network(input_a)
    vector_b = base_network(input_b)
    
    distance = tf.keras.layers.Lambda(euclidean_distance)([vector_a, vector_b])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(distance)
    
    return tf.keras.Model(inputs=[input_a, input_b], outputs=output)

# Usage example
input_shape = (100, 200, 1)  # Example input shape for signature images
model = create_signature_verification_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.fit([train_signatures_a, train_signatures_b], train_labels, epochs=10)
```

Slide 12: Image Similarity Search

Siamese networks can be used for image similarity search, allowing you to find similar images in a large dataset.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def create_image_embeddings(model, images):
    # Assuming the model's base network can be accessed as model.get_layer('base_network')
    base_network = model.get_layer('base_network')
    return base_network.predict(images)

def find_similar_images(query_embedding, database_embeddings, k=5):
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(database_embeddings)
    distances, indices = nn.kneighbors([query_embedding])
    return distances[0], indices[0]

# Usage example
# database_images = ...  # Load your image database
# database_embeddings = create_image_embeddings(model, database_images)

# query_image = ...  # Load a query image
# query_embedding = create_image_embeddings(model, np.expand_dims(query_image, axis=0))[0]

# distances, indices = find_similar_images(query_embedding, database_embeddings)
# similar_images = database_images[indices]
```

Slide 13: Fine-tuning Siamese Networks

Fine-tuning allows adapting a pre-trained Siamese network to a new, related task with limited data.

```python
def fine_tune_siamese_network(base_model, new_input_shape, num_classes):
    # Freeze the base network
    base_model.trainable = False
    
    # Create a new model with the pre-trained base
    input_a = tf.keras.layers.Input(shape=new_input_shape)
    input_b = tf.keras.layers.Input(shape=new_input_shape)
    
    vector_a = base_model(input_a)
    vector_b = base_model(input_b)
    
    # Add new layers for fine-tuning
    x = tf.keras.layers.Concatenate()([vector_a, vector_b])
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    new_model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
    
    return new_model

# Usage example
# original_model = ...  # Your pre-trained Siamese network
# new_input_shape = (64, 64, 3)  # New input shape
# num_classes = 10  # Number of classes in the new task

# fine_tuned_model = fine_tune_siamese_network(original_model, new_input_shape, num_classes)
# fine_tuned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fine_tuned_model.fit([new_train_data_a, new_train_data_b], new_train_labels, epochs=5)
```

Slide 14: Visualizing Siamese Network Embeddings

Visualizing the embeddings produced by a Siamese network can help understand how well it separates different classes.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(model, images, labels):
    embeddings = create_image_embeddings(model, images)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of Siamese network embeddings')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.show()

# Usage example
# test_images = ...  # Load test images
# test_labels = ...  # Load corresponding labels
# visualize_embeddings(model, test_images, test_labels)
```

Slide 15: Additional Resources

For more information on Siamese networks and their applications, consider exploring these resources:

1. "Siamese Neural Networks for One-shot Image Recognition" by Koch et al. (2015) ArXiv: [https://arxiv.org/abs/1504.03641](https://arxiv.org/abs/1504.03641)
2. "FaceNet: A Unified Embedding for Face Recognition and Clustering" by Schroff et al. (2015) ArXiv: [https://arxiv.org/abs/1503.03832](https://arxiv.org/abs/1503.03832)
3. "Learning a Similarity Metric Discriminatively, with Application to Face Verification" by Chopra et al. (2005) Available in IEEE Xplore (not on ArXiv)

These papers provide in-depth explanations of Siamese networks and their applications in various domains.

