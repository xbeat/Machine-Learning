## Restricted Boltzmann Machines for Speech Recognition in Python
Slide 1: Introduction to Restricted Boltzmann Machines (RBMs)

Restricted Boltzmann Machines are a type of neural network used for unsupervised learning. They consist of visible and hidden layers, with connections between but not within layers. RBMs are particularly useful for feature extraction and dimensionality reduction in speech recognition tasks.

```python
import numpy as np
import tensorflow as tf

# Define a simple RBM structure
n_visible = 6
n_hidden = 2

# Initialize weights and biases
W = tf.Variable(tf.random.normal([n_visible, n_hidden]))
bv = tf.Variable(tf.zeros([n_visible]))
bh = tf.Variable(tf.zeros([n_hidden]))

# Function to compute hidden layer probabilities
def compute_hidden_probs(visible):
    return tf.nn.sigmoid(tf.matmul(visible, W) + bh)

# Example visible layer input
visible_input = tf.constant([[1, 0, 1, 0, 1, 1]], dtype=tf.float32)

# Compute hidden layer probabilities
hidden_probs = compute_hidden_probs(visible_input)
print("Hidden layer probabilities:", hidden_probs.numpy())
```

Slide 2: RBM Architecture for Speech Recognition

In speech recognition, RBMs can be used to model the probability distribution of speech features. The visible layer represents the input speech features (e.g., mel-frequency cepstral coefficients), while the hidden layer captures higher-level representations.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate speech features (MFCC-like)
n_frames = 50
n_features = 13
speech_features = np.random.randn(n_frames, n_features)

# Plot the simulated speech features
plt.figure(figsize=(10, 6))
plt.imshow(speech_features.T, aspect='auto', cmap='viridis')
plt.title('Simulated Speech Features')
plt.xlabel('Time Frames')
plt.ylabel('Feature Dimensions')
plt.colorbar(label='Feature Value')
plt.show()
```

Slide 3: Training RBMs with Contrastive Divergence

Contrastive Divergence (CD) is a common method for training RBMs. It approximates the gradient of the log-likelihood and is computationally efficient. The CD-k algorithm performs k steps of Gibbs sampling to estimate the model's distribution.

```python
def contrastive_divergence(visible, k=1):
    # Positive phase
    pos_hidden_probs = compute_hidden_probs(visible)
    pos_hidden_states = tf.nn.relu(tf.sign(pos_hidden_probs - tf.random.uniform(tf.shape(pos_hidden_probs))))
    pos_associations = tf.matmul(tf.transpose(visible), pos_hidden_states)
    
    # Negative phase
    hidden_states = pos_hidden_states
    for _ in range(k):
        visible_probs = tf.nn.sigmoid(tf.matmul(hidden_states, tf.transpose(W)) + bv)
        visible_states = tf.nn.relu(tf.sign(visible_probs - tf.random.uniform(tf.shape(visible_probs))))
        hidden_probs = compute_hidden_probs(visible_states)
        hidden_states = tf.nn.relu(tf.sign(hidden_probs - tf.random.uniform(tf.shape(hidden_probs))))
    
    neg_associations = tf.matmul(tf.transpose(visible_states), hidden_states)
    
    # Update weights and biases
    lr = 0.01  # Learning rate
    W.assign_add(lr * (pos_associations - neg_associations))
    bv.assign_add(lr * tf.reduce_mean(visible - visible_states, 0))
    bh.assign_add(lr * tf.reduce_mean(pos_hidden_states - hidden_states, 0))

# Train the RBM
n_epochs = 1000
batch_size = 10
for epoch in range(n_epochs):
    batch = tf.random.normal([batch_size, n_visible])
    contrastive_divergence(batch)

print("Training completed")
```

Slide 4: Feature Extraction for Speech Recognition

RBMs can be used to extract meaningful features from raw speech data. These features can then be used as input to other machine learning models for speech recognition tasks.

```python
import librosa
import numpy as np

# Load a sample audio file (replace with your own audio file)
audio_file = 'path/to/your/audio/file.wav'
y, sr = librosa.load(audio_file)

# Extract MFCC features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Normalize the features
mfccs_normalized = (mfccs - np.mean(mfccs)) / np.std(mfccs)

# Use the trained RBM to extract higher-level features
input_features = tf.constant(mfccs_normalized.T, dtype=tf.float32)
higher_level_features = compute_hidden_probs(input_features)

print("Shape of higher-level features:", higher_level_features.shape)
```

Slide 5: Stacking RBMs for Deep Learning

Multiple RBMs can be stacked to form a Deep Belief Network (DBN). Each RBM layer learns increasingly abstract representations of the input data, which can improve speech recognition performance.

```python
class DeepBeliefNetwork:
    def __init__(self, layer_sizes):
        self.rbm_layers = []
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(layer_sizes[i], layer_sizes[i+1])
            self.rbm_layers.append(rbm)
    
    def pretrain(self, data, epochs=10):
        input_data = data
        for rbm in self.rbm_layers:
            print(f"Pretraining RBM with {rbm.n_visible} visible and {rbm.n_hidden} hidden units")
            rbm.train(input_data, epochs)
            input_data = rbm.transform(input_data)
    
    def transform(self, data):
        transformed_data = data
        for rbm in self.rbm_layers:
            transformed_data = rbm.transform(transformed_data)
        return transformed_data

# Example usage
layer_sizes = [n_visible, 100, 50, 20]
dbn = DeepBeliefNetwork(layer_sizes)
dbn.pretrain(mfccs_normalized.T)

# Transform input data using the pretrained DBN
transformed_features = dbn.transform(mfccs_normalized.T)
print("Shape of transformed features:", transformed_features.shape)
```

Slide 6: Fine-tuning for Speech Recognition

After pretraining the DBN, we can add a classification layer and fine-tune the entire network for speech recognition tasks using backpropagation.

```python
import tensorflow as tf

class SpeechRecognizer(tf.keras.Model):
    def __init__(self, dbn, n_classes):
        super(SpeechRecognizer, self).__init__()
        self.dbn = dbn
        self.classifier = tf.keras.layers.Dense(n_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.dbn.transform(inputs)
        return self.classifier(x)

# Assuming we have labeled data (X_train, y_train)
n_classes = 10  # Number of speech classes
recognizer = SpeechRecognizer(dbn, n_classes)

# Compile and train the model
recognizer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = recognizer.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

Slide 7: Handling Variable-Length Speech Inputs

Speech inputs often have variable lengths. We can use padding and masking to handle this variability in our RBM-based speech recognition system.

```python
def pad_sequences(sequences, max_len=None):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        padded_seq = np.zeros((max_len, seq.shape[1]))
        padded_seq[:len(seq)] = seq
        padded_sequences.append(padded_seq)
    
    return np.array(padded_sequences)

# Example usage
speech_samples = [np.random.randn(np.random.randint(50, 100), 13) for _ in range(5)]
padded_samples = pad_sequences(speech_samples)

print("Shape of padded samples:", padded_samples.shape)

# Create a mask for valid time steps
mask = tf.sequence_mask(tf.constant([len(seq) for seq in speech_samples]), maxlen=padded_samples.shape[1])

# Apply the mask in the model
masked_output = tf.where(mask[:, :, tf.newaxis], padded_samples, 0.0)
```

Slide 8: Real-life Example: Speech Emotion Recognition

RBMs can be used for speech emotion recognition, which has applications in customer service, mental health monitoring, and human-computer interaction.

```python
import librosa
import numpy as np
import tensorflow as tf

def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_normalized = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    return mfccs_normalized.T

# Load and preprocess emotion labeled audio files
emotions = ['angry', 'happy', 'sad', 'neutral']
X = []
y = []

for i, emotion in enumerate(emotions):
    audio_files = glob.glob(f'path/to/{emotion}_audio_files/*.wav')
    for audio_file in audio_files:
        features = extract_features(audio_file)
        X.append(features)
        y.append(i)

X_padded = pad_sequences(X)
y = np.array(y)

# Train RBM-based emotion recognizer
dbn = DeepBeliefNetwork([X_padded.shape[2], 100, 50, 20])
dbn.pretrain(X_padded.reshape(-1, X_padded.shape[2]))

emotion_recognizer = SpeechRecognizer(dbn, len(emotions))
emotion_recognizer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
emotion_recognizer.fit(X_padded, y, epochs=50, validation_split=0.2)

# Test the emotion recognizer
test_audio = 'path/to/test_audio.wav'
test_features = extract_features(test_audio)
test_features_padded = pad_sequences([test_features])
predicted_emotion = emotions[np.argmax(emotion_recognizer.predict(test_features_padded))]
print(f"Predicted emotion: {predicted_emotion}")
```

Slide 9: Real-life Example: Speech-to-Text Transcription

RBMs can be used as part of a speech-to-text system, which has applications in transcription services, closed captioning, and voice assistants.

```python
import numpy as np
import tensorflow as tf
import librosa

# Simplified character-level language model
class LanguageModel:
    def __init__(self, vocab_size):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 64),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dense(vocab_size, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    def train(self, text_data, epochs=10):
        char_to_idx = {char: idx for idx, char in enumerate(sorted(set(''.join(text_data))))}
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        X = [[char_to_idx[char] for char in text] for text in text_data]
        y = [[char_to_idx[char] for char in text[1:]] + [char_to_idx['\n']] for text in text_data]
        
        self.model.fit(tf.keras.preprocessing.sequence.pad_sequences(X, padding='post'),
                       tf.keras.preprocessing.sequence.pad_sequences(y, padding='post'),
                       epochs=epochs)
        
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
    
    def predict_next_char(self, prefix):
        x = [self.char_to_idx[char] for char in prefix]
        y_pred = self.model.predict(tf.keras.preprocessing.sequence.pad_sequences([x], padding='post'))
        return self.idx_to_char[np.argmax(y_pred[0, -1])]

# Speech-to-text system
class SpeechToText:
    def __init__(self, dbn, language_model):
        self.dbn = dbn
        self.language_model = language_model
        self.char_classifier = tf.keras.layers.Dense(len(language_model.char_to_idx), activation='softmax')
    
    def transcribe(self, audio_path):
        features = extract_features(audio_path)
        transformed_features = self.dbn.transform(features)
        char_probs = self.char_classifier(transformed_features)
        
        transcription = ''
        for probs in char_probs:
            char = self.language_model.idx_to_char[np.argmax(probs)]
            transcription += char
            next_char = self.language_model.predict_next_char(transcription)
            transcription += next_char
        
        return transcription

# Train language model
text_data = ["hello world", "speech recognition", "machine learning"]
lm = LanguageModel(len(set(''.join(text_data))))
lm.train(text_data)

# Initialize speech-to-text system
dbn = DeepBeliefNetwork([13, 100, 50, 20])  # Assuming 13 MFCC features
stt = SpeechToText(dbn, lm)

# Transcribe audio
audio_path = 'path/to/audio.wav'
transcription = stt.transcribe(audio_path)
print(f"Transcription: {transcription}")
```

Slide 10: Visualizing RBM Weights

Visualizing the weights of an RBM can provide insights into the features it has learned. This can be particularly useful for understanding what patterns the RBM is detecting in speech data.

```python
import matplotlib.pyplot as plt

def visualize_rbm_weights(rbm, fig_size=(10, 5)):
    weights = rbm.W.numpy()
    plt.figure(figsize=fig_size)
    plt.imshow(weights, cmap='viridis', aspect='auto')
    plt.title('RBM Weights Visualization')
    plt.xlabel('Hidden Units')
    plt.ylabel('Visible Units')
    plt.colorbar(label='Weight Value')
    plt.show()

# Assuming we have a trained RBM
visualize_rbm_weights(rbm_layers[0])
```

Slide 11: Hyperparameter Tuning for RBMs

Hyperparameter tuning is crucial for optimizing RBM performance in speech recognition tasks. We can use techniques like grid search or random search to find the best hyperparameters.

```python
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

class RBMWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_hidden=10, learning_rate=0.01, n_epochs=10):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.rbm = None

    def fit(self, X, y=None):
        n_visible = X.shape[1]
        self.rbm = RBM(n_visible, self.n_hidden)
        self.rbm.train(X, learning_rate=self.learning_rate, n_epochs=self.n_epochs)
        return self

    def transform(self, X):
        return self.rbm.transform(X)

# Assuming X_train is our training data
param_dist = {
    'n_hidden': np.arange(10, 100, 10),
    'learning_rate': np.logspace(-3, 0, 10),
    'n_epochs': [10, 20, 50, 100]
}

rbm_wrapper = RBMWrapper()
random_search = RandomizedSearchCV(rbm_wrapper, param_distributions=param_dist, n_iter=20, cv=3)
random_search.fit(X_train)

print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)
```

Slide 12: Dealing with Overfitting in RBMs

Overfitting can be a challenge when training RBMs for speech recognition. We can use techniques like weight decay and dropout to mitigate this issue.

```python
class RegularizedRBM(RBM):
    def __init__(self, n_visible, n_hidden, weight_decay=0.0001):
        super().__init__(n_visible, n_hidden)
        self.weight_decay = weight_decay

    def update_weights(self, pos_associations, neg_associations, lr):
        weight_update = lr * ((pos_associations - neg_associations) / self.batch_size - self.weight_decay * self.W)
        self.W += weight_update

# Training loop with dropout
def train_with_dropout(rbm, data, dropout_rate=0.5, n_epochs=10, lr=0.01):
    for epoch in range(n_epochs):
        for batch in create_batches(data):
            # Apply dropout to visible units
            dropout_mask = np.random.binomial(1, 1 - dropout_rate, batch.shape)
            dropped_batch = batch * dropout_mask
            
            pos_hidden_probs = rbm.compute_hidden(dropped_batch)
            pos_associations = np.dot(dropped_batch.T, pos_hidden_probs)
            
            neg_visible = rbm.compute_visible(rbm.compute_hidden(dropped_batch))
            neg_hidden_probs = rbm.compute_hidden(neg_visible)
            neg_associations = np.dot(neg_visible.T, neg_hidden_probs)
            
            rbm.update_weights(pos_associations, neg_associations, lr)

# Usage
regularized_rbm = RegularizedRBM(n_visible, n_hidden)
train_with_dropout(regularized_rbm, X_train)
```

Slide 13: Incorporating RBMs into Hybrid Speech Recognition Systems

RBMs can be combined with other techniques like Hidden Markov Models (HMMs) or Deep Neural Networks (DNNs) to create hybrid systems for improved speech recognition performance.

```python
import numpy as np
from hmmlearn import hmm

class HybridRBM_HMM:
    def __init__(self, n_visible, n_hidden, n_components):
        self.rbm = RBM(n_visible, n_hidden)
        self.hmm = hmm.GaussianHMM(n_components=n_components)

    def fit(self, X, lengths):
        # Train RBM
        self.rbm.train(X)
        
        # Transform data using RBM
        X_transformed = self.rbm.transform(X)
        
        # Train HMM on transformed data
        self.hmm.fit(X_transformed, lengths)

    def decode(self, X):
        X_transformed = self.rbm.transform(X)
        return self.hmm.decode(X_transformed)[1]

# Usage
n_visible = 13  # MFCC features
n_hidden = 50
n_components = 10  # Number of HMM states

hybrid_model = HybridRBM_HMM(n_visible, n_hidden, n_components)
hybrid_model.fit(X_train, lengths_train)

# Decode test data
predicted_states = hybrid_model.decode(X_test)
```

Slide 14: Evaluating RBM-based Speech Recognition Systems

To assess the performance of our RBM-based speech recognition system, we can use metrics such as Word Error Rate (WER) and Phoneme Error Rate (PER).

```python
def calculate_wer(reference, hypothesis):
    # Implement Levenshtein distance calculation
    # and word error rate computation
    pass

def calculate_per(reference, hypothesis):
    # Implement phoneme error rate computation
    pass

# Assuming we have reference transcriptions and model predictions
reference_transcriptions = ["hello world", "speech recognition"]
predicted_transcriptions = ["hello word", "speech recogniton"]

wer = calculate_wer(reference_transcriptions, predicted_transcriptions)
per = calculate_per(reference_transcriptions, predicted_transcriptions)

print(f"Word Error Rate: {wer:.2f}")
print(f"Phoneme Error Rate: {per:.2f}")

# Visualize confusion matrix for phoneme recognition
from sklearn.metrics import confusion_matrix
import seaborn as sns

phoneme_true = ["ae", "th", "iy", "s", "p", "ch"]
phoneme_pred = ["ae", "s", "iy", "s", "b", "ch"]

cm = confusion_matrix(phoneme_true, phoneme_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Phoneme Confusion Matrix')
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into Restricted Boltzmann Machines for Speech Recognition, here are some valuable resources:

1. "Deep Learning for Speech Recognition" by Dong Yu and Li Deng (ArXiv:1504.01482) URL: [https://arxiv.org/abs/1504.01482](https://arxiv.org/abs/1504.01482)
2. "Speech Recognition with Deep Recurrent Neural Networks" by Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton (ArXiv:1303.5778) URL: [https://arxiv.org/abs/1303.5778](https://arxiv.org/abs/1303.5778)
3. "Applying Convolutional Neural Networks Concepts to Hybrid NN-HMM Model for Speech Recognition" by Ossama Abdel-Hamid et al. (ArXiv:1204.5673) URL: [https://arxiv.org/abs/1204.5673](https://arxiv.org/abs/1204.5673)

These papers provide in-depth discussions on advanced techniques and applications of deep learning models, including RBMs, in speech recognition tasks.

