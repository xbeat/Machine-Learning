## Credit Card Fraud Detection with Autoencoders in Python
Slide 1: Introduction to Credit Card Fraud Detection

Credit card fraud is a significant problem for financial institutions and consumers alike. Machine learning techniques, particularly autoencoders, can be powerful tools for detecting fraudulent transactions. This presentation will explore how to implement an autoencoder using Python to detect credit card fraud.

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
data = pd.read_csv('credit_card_transactions.csv')
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Slide 2: What is an Autoencoder?

An autoencoder is a type of neural network that learns to compress data into a lower-dimensional representation and then reconstruct it. It consists of an encoder that compresses the input and a decoder that attempts to recreate the original input from the compressed representation.

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_autoencoder(input_dim, encoding_dim):
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    
    # Decoder
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    # Autoencoder model
    autoencoder = Model(input_layer, decoded)
    
    # Encoder model
    encoder = Model(input_layer, encoded)
    
    return autoencoder, encoder

input_dim = X_train_scaled.shape[1]
encoding_dim = 14  # Adjust based on your needs

autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
```

Slide 3: Training the Autoencoder

We train the autoencoder to reconstruct normal transactions. The model learns to compress and decompress the data, minimizing the reconstruction error for legitimate transactions.

```python
# Train the autoencoder
history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_split=0.2,
    verbose=1
)

# Plot the training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

Slide 4: Detecting Anomalies

To detect fraud, we compare the reconstruction error of each transaction. Fraudulent transactions should have higher reconstruction errors as the autoencoder struggles to reconstruct them accurately.

```python
def get_reconstruction_error(autoencoder, data):
    predictions = autoencoder.predict(data)
    mse = np.mean(np.power(data - predictions, 2), axis=1)
    return mse

# Calculate reconstruction errors
train_errors = get_reconstruction_error(autoencoder, X_train_scaled)
test_errors = get_reconstruction_error(autoencoder, X_test_scaled)

# Plot the distribution of reconstruction errors
plt.figure(figsize=(10, 5))
plt.hist(train_errors, bins=50, alpha=0.5, label='Training Data')
plt.hist(test_errors, bins=50, alpha=0.5, label='Test Data')
plt.title('Distribution of Reconstruction Errors')
plt.xlabel('Reconstruction Error')
plt.ylabel('Count')
plt.legend()
plt.show()
```

Slide 5: Setting a Threshold

We need to set a threshold to classify transactions as fraudulent or legitimate based on their reconstruction error. One approach is to use a percentile of the training data's reconstruction errors.

```python
import numpy as np

# Set threshold as the 95th percentile of training errors
threshold = np.percentile(train_errors, 95)

# Classify transactions
y_pred = (test_errors > threshold).astype(int)

# Calculate performance metrics
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

Slide 6: Visualizing Results

To better understand our model's performance, we can visualize the reconstruction errors for both legitimate and fraudulent transactions.

```python
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), test_errors, c=y_test, cmap='coolwarm')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error for Test Data')
plt.xlabel('Data Point')
plt.ylabel('Reconstruction Error')
plt.colorbar(label='Actual Class')
plt.legend()
plt.show()
```

Slide 7: Feature Importance

We can analyze which features contribute most to the reconstruction error, potentially revealing which transaction characteristics are most indicative of fraud.

```python
def get_feature_importance(autoencoder, data):
    predictions = autoencoder.predict(data)
    mse_per_feature = np.mean(np.power(data - predictions, 2), axis=0)
    return mse_per_feature

feature_importance = get_feature_importance(autoencoder, X_test_scaled)
feature_names = X.columns

plt.figure(figsize=(12, 6))
plt.bar(feature_names, feature_importance)
plt.title('Feature Importance Based on Reconstruction Error')
plt.xlabel('Features')
plt.ylabel('Mean Squared Error')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```

Slide 8: Handling Imbalanced Data

Credit card fraud datasets are often highly imbalanced. We can address this by using techniques like oversampling or undersampling.

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Scale the balanced data
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)

# Train the autoencoder on balanced data
autoencoder_balanced, _ = build_autoencoder(input_dim, encoding_dim)
autoencoder_balanced.compile(optimizer='adam', loss='mean_squared_error')
autoencoder_balanced.fit(
    X_train_balanced_scaled, X_train_balanced_scaled,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_split=0.2,
    verbose=1
)
```

Slide 9: Real-time Fraud Detection

In a real-world scenario, we need to process transactions in real-time. Here's how we might implement a simple real-time fraud detection system:

```python
def detect_fraud_realtime(transaction, autoencoder, scaler, threshold):
    # Preprocess the transaction
    transaction_scaled = scaler.transform([transaction])
    
    # Get reconstruction error
    error = get_reconstruction_error(autoencoder, transaction_scaled)[0]
    
    # Classify the transaction
    is_fraudulent = error > threshold
    
    return is_fraudulent, error

# Example usage
new_transaction = [0, 1, 2, 3, 4, 5, ...]  # Replace with actual transaction data
is_fraud, error = detect_fraud_realtime(new_transaction, autoencoder, scaler, threshold)

print(f"Transaction {'is' if is_fraud else 'is not'} flagged as fraudulent.")
print(f"Reconstruction error: {error:.4f}")
```

Slide 10: Ensemble Methods

To improve performance, we can combine multiple models in an ensemble. Here's an example using a combination of autoencoder and isolation forest:

```python
from sklearn.ensemble import IsolationForest

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X_train_scaled)

# Combine predictions
def ensemble_predict(X, autoencoder, iso_forest, ae_threshold):
    ae_errors = get_reconstruction_error(autoencoder, X)
    ae_predictions = (ae_errors > ae_threshold).astype(int)
    
    if_predictions = (iso_forest.predict(X) == -1).astype(int)
    
    # Combine predictions (flagged as fraud if either model predicts fraud)
    ensemble_predictions = np.logical_or(ae_predictions, if_predictions).astype(int)
    
    return ensemble_predictions

ensemble_predictions = ensemble_predict(X_test_scaled, autoencoder, iso_forest, threshold)
print(classification_report(y_test, ensemble_predictions))
```

Slide 11: Model Interpretability

Understanding why a model flags a transaction as fraudulent is crucial. We can use techniques like SHAP (SHapley Additive exPlanations) values to interpret our model's decisions.

```python
import shap

# Create an explainer
explainer = shap.KernelExplainer(autoencoder.predict, shap.sample(X_train_scaled, 100))

# Calculate SHAP values for a single instance
instance = X_test_scaled[0:1]
shap_values = explainer.shap_values(instance)

# Visualize the SHAP values
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, instance, feature_names=X.columns)
```

Slide 12: Monitoring and Updating

Fraud patterns change over time, so it's essential to continuously monitor and update our model. Here's a simple example of how we might implement a sliding window approach:

```python
from collections import deque

class FraudDetectionSystem:
    def __init__(self, window_size=1000, update_frequency=100):
        self.window = deque(maxlen=window_size)
        self.update_frequency = update_frequency
        self.transaction_count = 0
        
    def process_transaction(self, transaction):
        self.window.append(transaction)
        self.transaction_count += 1
        
        if self.transaction_count % self.update_frequency == 0:
            self.update_model()
    
    def update_model(self):
        # Retrain the model using the current window of transactions
        X_window = np.array(self.window)
        X_window_scaled = scaler.fit_transform(X_window)
        
        autoencoder, _ = build_autoencoder(X_window.shape[1], encoding_dim)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(X_window_scaled, X_window_scaled, epochs=10, batch_size=32, verbose=0)
        
        print("Model updated with latest transactions.")

# Usage
fraud_system = FraudDetectionSystem()
for transaction in new_transactions:
    fraud_system.process_transaction(transaction)
```

Slide 13: Real-life Example: E-commerce Fraud Detection

In an e-commerce setting, we can use our autoencoder to detect unusual purchasing patterns. Features might include:

1. Time since last purchase
2. Number of items in cart
3. Total purchase amount
4. Shipping address distance from billing address
5. Device type used for purchase

Our model would learn the typical patterns of legitimate purchases and flag transactions that deviate significantly from these patterns.

```python
# Example e-commerce transaction data
ecommerce_data = pd.DataFrame({
    'time_since_last_purchase': [2, 5, 1, 10, 3],
    'num_items': [3, 1, 5, 2, 4],
    'total_amount': [150, 50, 300, 100, 200],
    'address_distance': [0, 10, 5, 100, 2],
    'device_type': [0, 1, 0, 2, 1]  # 0: PC, 1: Mobile, 2: Tablet
})

# Scale the data and detect anomalies
ecommerce_data_scaled = scaler.fit_transform(ecommerce_data)
anomaly_scores = get_reconstruction_error(autoencoder, ecommerce_data_scaled)

print("Anomaly scores for e-commerce transactions:")
print(anomaly_scores)
```

Slide 14: Real-life Example: Network Intrusion Detection

We can adapt our autoencoder for network intrusion detection. Features might include:

1. Packet size
2. Protocol type
3. Source and destination IP addresses
4. Connection duration
5. Number of failed login attempts

The autoencoder would learn normal network traffic patterns and flag unusual activities that could indicate a cyber attack.

```python
# Example network traffic data
network_data = pd.DataFrame({
    'packet_size': [1000, 500, 2000, 1500, 3000],
    'protocol_type': [0, 1, 0, 2, 1],  # 0: TCP, 1: UDP, 2: ICMP
    'src_ip': [192168001001, 192168001002, 192168001003, 192168001004, 192168001005],
    'dst_ip': [192168002001, 192168002002, 192168002003, 192168002004, 192168002005],
    'duration': [10, 5, 15, 8, 20],
    'failed_logins': [0, 0, 1, 0, 3]
})

# Scale the data and detect anomalies
network_data_scaled = scaler.fit_transform(network_data)
anomaly_scores = get_reconstruction_error(autoencoder, network_data_scaled)

print("Anomaly scores for network traffic:")
print(anomaly_scores)
```

Slide 15: Additional Resources

For those interested in diving deeper into autoencoders and anomaly detection, here are some valuable resources:

1. "Anomaly Detection with Autoencoders Made Easy" by François Chollet ([https://arxiv.org/abs/1802.03903](https://arxiv.org/abs/1802.03903))
2. "A Survey of Deep Learning Techniques for Anomaly Detection" by Chalapathy and Chawla ([https://arxiv.org/abs/1901.03407](https://arxiv.org/abs/1901.03407))
3. "Deep Learning for Anomaly Detection: A Survey" by Pang et al. ([https://arxiv.org/abs/2007.02500](https://arxiv.org/abs/2007.02500))

These papers provide comprehensive overviews of various techniques and applications in anomaly detection using deep learning methods, including autoencoders.

