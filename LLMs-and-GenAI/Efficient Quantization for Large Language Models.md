## Efficient Quantization for Large Language Models
Slide 1: Understanding Quantization Basics

Quantization is a technique to reduce model size by mapping floating-point values to lower-precision formats. This process involves determining the optimal mapping between continuous values and their discrete representations while minimizing information loss.

```python
import numpy as np

def basic_quantization(weights, num_bits=8):
    # Calculate the range of values
    min_val, max_val = weights.min(), weights.max()
    
    # Calculate step size for quantization
    step_size = (max_val - min_val) / (2**num_bits - 1)
    
    # Quantize the weights
    quantized = np.round((weights - min_val) / step_size)
    
    # Dequantize to get approximate values
    dequantized = quantized * step_size + min_val
    
    return quantized, dequantized

# Example usage
original = np.random.normal(0, 1, 1000)
quantized, reconstructed = basic_quantization(original)
print(f"Original size: {original.nbytes} bytes")
print(f"Quantized size: {quantized.nbytes} bytes")
```

Slide 2: Linear Quantization Implementation

Linear quantization maps continuous values to discrete levels using uniform intervals. This implementation demonstrates the process of converting 32-bit floating-point numbers to 8-bit integers while preserving the relative relationships between values.

```python
def linear_quantize(tensor, bits=8):
    # Calculate scaling factor
    scale = (2**bits - 1) / (tensor.max() - tensor.min())
    
    # Zero point calculation
    zero_point = -(tensor.min() * scale)
    
    # Quantize to integers
    quant = np.clip(np.round(tensor * scale + zero_point), 0, 2**bits - 1)
    
    # Dequantize
    dequant = (quant - zero_point) / scale
    
    return quant.astype(np.uint8), dequant, scale, zero_point

# Example with sample tensor
tensor = np.array([0.3, -1.2, 0.8, -0.4, 1.5])
q, dq, scale, zp = linear_quantize(tensor)
print(f"Original: {tensor}")
print(f"Quantized: {q}")
print(f"Dequantized: {dq}")
```

Slide 3: Symmetric Quantization

Symmetric quantization provides a more efficient representation for neural networks by assuming a symmetric distribution around zero. This approach is particularly effective for weight matrices in deep learning models.

```python
def symmetric_quantize(weights, bits=8):
    # Determine maximum absolute value
    abs_max = np.max(np.abs(weights))
    
    # Calculate scale factor
    scale = (2**(bits-1) - 1) / abs_max
    
    # Quantize weights
    quantized = np.clip(np.round(weights * scale), 
                       -(2**(bits-1)), 
                       2**(bits-1) - 1)
    
    # Dequantize for comparison
    dequantized = quantized / scale
    
    return quantized, dequantized, scale

# Example usage
weights = np.random.normal(0, 1, (100,))
q, dq, s = symmetric_quantize(weights)
print(f"Compression ratio: {weights.nbytes/q.nbytes:.2f}x")
```

Slide 4: Per-Channel Quantization

This advanced quantization technique applies different scaling factors for each channel or layer, providing better accuracy than global quantization. It's particularly effective for convolutional neural networks.

```python
def per_channel_quantize(tensor, bits=8, axis=0):
    shape = tensor.shape
    scales = []
    quantized = np.zeros_like(tensor)
    
    # Iterate over channels
    for i in range(shape[axis]):
        # Select channel
        if axis == 0:
            channel = tensor[i]
        else:
            channel = tensor[:, i]
            
        # Calculate scale for this channel
        max_abs = np.max(np.abs(channel))
        scale = (2**(bits-1) - 1) / max_abs
        scales.append(scale)
        
        # Quantize channel
        quant = np.clip(np.round(channel * scale),
                       -(2**(bits-1)),
                       2**(bits-1) - 1)
                       
        # Store quantized values
        if axis == 0:
            quantized[i] = quant
        else:
            quantized[:, i] = quant
            
    return quantized, np.array(scales)

# Example with 2D tensor
tensor = np.random.normal(0, 1, (3, 4))
q, scales = per_channel_quantize(tensor)
print(f"Channel scales: {scales}")
```

Slide 5: Dynamic Range Quantization

Dynamic range quantization adapts the quantization parameters based on the actual distribution of values in each tensor, making it particularly effective for activations in neural networks that can have varying ranges during inference.

```python
def dynamic_range_quantize(tensor, bits=8, percentile=99.9):
    # Calculate dynamic range using percentiles
    pos_threshold = np.percentile(tensor, percentile)
    neg_threshold = np.percentile(tensor, 100-percentile)
    
    # Compute symmetric range
    abs_max = max(abs(pos_threshold), abs(neg_threshold))
    scale = (2**(bits-1) - 1) / abs_max
    
    # Quantize using dynamic range
    quantized = np.clip(np.round(tensor * scale),
                       -(2**(bits-1)),
                       2**(bits-1) - 1)
    
    # Dequantize for verification
    dequantized = quantized / scale
    
    return quantized, dequantized, scale

# Example with skewed distribution
data = np.concatenate([
    np.random.normal(0, 1, 1000),
    np.random.normal(5, 0.1, 100)  # Outliers
])
q, dq, s = dynamic_range_quantize(data)
print(f"Scale factor: {s}")
print(f"Max quantization error: {np.max(np.abs(data - dq))}")
```

Slide 6: Weight Clustering for Quantization

Weight clustering reduces the number of unique values in the model by grouping similar weights together. This technique combines well with quantization to achieve higher compression rates while maintaining model performance.

```python
import sklearn.cluster as cluster

def cluster_weights(weights, num_clusters=256):
    # Reshape weights to 1D array
    flat_weights = weights.reshape(-1)
    
    # Perform k-means clustering
    kmeans = cluster.KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(flat_weights.reshape(-1, 1))
    
    # Get cluster centers and assignments
    centroids = kmeans.cluster_centers_.flatten()
    assignments = kmeans.labels_
    
    # Create quantized weights using cluster centers
    quantized = centroids[assignments].reshape(weights.shape)
    
    return quantized, centroids, assignments

# Example usage
weights = np.random.normal(0, 1, (1000, 1000))
q_weights, centroids, _ = cluster_weights(weights)

print(f"Original unique values: {len(np.unique(weights))}")
print(f"Quantized unique values: {len(np.unique(q_weights))}")
print(f"Compression ratio: {weights.nbytes/q_weights.nbytes:.2f}x")
```

Slide 7: Mixed-Precision Quantization

Mixed-precision quantization applies different bit widths to different parts of the model based on their sensitivity to quantization. This implementation demonstrates how to selectively quantize layers while maintaining accuracy.

```python
def mixed_precision_quantize(layers_dict):
    quantized_layers = {}
    
    for layer_name, config in layers_dict.items():
        weights = config['weights']
        bits = config['bits']
        
        if bits == 32:
            # Keep full precision
            quantized_layers[layer_name] = weights
        else:
            # Apply appropriate quantization
            if bits <= 4:
                q, _, _ = cluster_weights(weights, num_clusters=2**bits)
            else:
                q, _, _ = symmetric_quantize(weights, bits=bits)
            quantized_layers[layer_name] = q
    
    return quantized_layers

# Example usage
layers = {
    'embedding': {
        'weights': np.random.normal(0, 1, (1000, 64)),
        'bits': 8
    },
    'attention': {
        'weights': np.random.normal(0, 1, (64, 64)),
        'bits': 4
    },
    'output': {
        'weights': np.random.normal(0, 1, (64, 10)),
        'bits': 32
    }
}

quantized = mixed_precision_quantize(layers)
for layer, weights in quantized.items():
    print(f"{layer} shape: {weights.shape}, dtype: {weights.dtype}")
```

Slide 8: Calibration for Quantization

Calibration is crucial for determining optimal quantization parameters. This implementation shows how to collect statistics from actual data to improve quantization accuracy.

```python
def calibrate_quantization(model_outputs, num_samples=1000):
    # Initialize statistics collectors
    min_vals = []
    max_vals = []
    distributions = []
    
    # Collect statistics from model outputs
    for output in model_outputs[:num_samples]:
        min_vals.append(np.min(output))
        max_vals.append(np.max(output))
        
        # Calculate histogram for distribution analysis
        hist, _ = np.histogram(output, bins=100)
        distributions.append(hist)
    
    # Calculate stable quantization parameters
    stable_min = np.percentile(min_vals, 1)  # Remove outliers
    stable_max = np.percentile(max_vals, 99)
    
    # Calculate optimal scale based on distribution
    avg_dist = np.mean(distributions, axis=0)
    scale = (stable_max - stable_min) / (np.sum(avg_dist > 0))
    
    return stable_min, stable_max, scale

# Example with synthetic model outputs
outputs = [np.random.normal(0, i/10, 1000) for i in range(100)]
min_val, max_val, scale = calibrate_quantization(outputs)

print(f"Calibrated range: [{min_val:.3f}, {max_val:.3f}]")
print(f"Optimal scale: {scale:.3f}")
```

Slide 9: Post-Training Quantization Implementation

Post-training quantization (PTQ) is applied after model training and requires careful handling of activation statistics. This implementation shows how to apply PTQ to a pre-trained model while monitoring accuracy degradation.

```python
def post_training_quantize(weights, activations, bits=8):
    # Collect activation statistics
    act_mean = np.mean(activations, axis=0)
    act_std = np.std(activations, axis=0)
    
    # Calculate optimal quantization parameters
    weight_scale = (2**(bits-1) - 1) / np.max(np.abs(weights))
    act_scale = (2**(bits-1) - 1) / (2 * act_std)
    
    # Quantize weights
    w_quantized = np.clip(np.round(weights * weight_scale),
                         -(2**(bits-1)),
                         2**(bits-1) - 1)
    
    # Quantize activations considering distribution
    a_quantized = np.clip(np.round((activations - act_mean) * act_scale),
                         -(2**(bits-1)),
                         2**(bits-1) - 1)
    
    # Dequantize for inference
    w_dequantized = w_quantized / weight_scale
    a_dequantized = (a_quantized / act_scale) + act_mean
    
    return {
        'weights': {'quantized': w_quantized, 'scale': weight_scale},
        'activations': {'quantized': a_quantized, 'scale': act_scale, 'mean': act_mean}
    }

# Example usage
weights = np.random.normal(0, 0.1, (256, 256))
activations = np.random.normal(0.5, 0.2, (1000, 256))
quantized_model = post_training_quantize(weights, activations)

print("Quantization Stats:")
print(f"Weight scale: {quantized_model['weights']['scale']:.4f}")
print(f"Activation scale: {quantized_model['activations']['scale']:.4f}")
```

Slide 10: Hardware-Aware Quantization

This implementation considers hardware constraints by aligning quantization parameters with specific hardware requirements, ensuring efficient deployment on target devices.

```python
def hardware_aware_quantize(tensor, hw_config):
    # Hardware-specific parameters
    supported_bits = hw_config['supported_bits']
    alignment = hw_config['alignment']
    max_groups = hw_config['max_groups']
    
    # Find closest supported bit width
    bits = min(supported_bits, key=lambda x: abs(x - hw_config['target_bits']))
    
    # Align tensor dimensions
    pad_size = (alignment - (tensor.shape[-1] % alignment)) % alignment
    padded_tensor = np.pad(tensor, ((0, 0), (0, pad_size)))
    
    # Group channels for hardware efficiency
    n_channels = padded_tensor.shape[-1]
    group_size = max(1, min(n_channels // max_groups, alignment))
    
    quantized_groups = []
    scales = []
    
    # Quantize by groups
    for i in range(0, n_channels, group_size):
        group = padded_tensor[..., i:i+group_size]
        scale = (2**(bits-1) - 1) / np.max(np.abs(group))
        quantized = np.clip(np.round(group * scale),
                          -(2**(bits-1)),
                          2**(bits-1) - 1)
        quantized_groups.append(quantized)
        scales.append(scale)
    
    return np.concatenate(quantized_groups, axis=-1), np.array(scales)

# Example hardware configuration
hw_config = {
    'supported_bits': [2, 4, 8, 16],
    'alignment': 32,
    'max_groups': 4,
    'target_bits': 6
}

tensor = np.random.normal(0, 1, (100, 100))
q_tensor, scales = hardware_aware_quantize(tensor, hw_config)

print(f"Quantized shape: {q_tensor.shape}")
print(f"Number of quantization groups: {len(scales)}")
```

Slide 11: Handling Outliers in Quantization

Outlier handling is crucial for maintaining model accuracy during quantization. This implementation demonstrates techniques to identify and handle outliers while preserving the model's statistical properties.

```python
def outlier_aware_quantization(tensor, bits=8, percentile=99.9):
    # Identify outliers using statistical methods
    upper_bound = np.percentile(tensor, percentile)
    lower_bound = np.percentile(tensor, 100 - percentile)
    
    # Create outlier mask
    outlier_mask = (tensor > upper_bound) | (tensor < lower_bound)
    
    # Separate main distribution and outliers
    main_dist = tensor[~outlier_mask]
    outliers = tensor[outlier_mask]
    
    # Quantize main distribution
    main_scale = (2**(bits-1) - 1) / max(abs(np.min(main_dist)), abs(np.max(main_dist)))
    main_quantized = np.clip(np.round(main_dist * main_scale),
                           -(2**(bits-1)),
                           2**(bits-1) - 1)
    
    # Special handling for outliers (using full precision)
    result = tensor.copy()
    result[~outlier_mask] = main_quantized / main_scale
    
    return result, main_scale, outlier_mask

# Example usage
data = np.concatenate([
    np.random.normal(0, 1, 10000),  # Main distribution
    np.random.normal(10, 0.1, 100)  # Outliers
])

quantized, scale, outliers = outlier_aware_quantization(data)
print(f"Percentage of outliers: {np.mean(outliers)*100:.2f}%")
print(f"Main distribution scale: {scale:.4f}")
```

Slide 12: Quantization-Aware Metrics

Implementing specific metrics to evaluate quantization quality helps in fine-tuning the quantization process and maintaining model performance.

```python
def quantization_metrics(original, quantized, bits):
    # Mean Squared Error
    mse = np.mean((original - quantized) ** 2)
    
    # Signal-to-Quantization-Noise Ratio (SQNR)
    signal_power = np.mean(original ** 2)
    noise_power = mse
    sqnr = 10 * np.log10(signal_power / noise_power)
    
    # Cosine similarity
    cos_sim = np.dot(original.flatten(), quantized.flatten()) / (
        np.linalg.norm(original) * np.linalg.norm(quantized))
    
    # Effective bit utilization
    unique_values = len(np.unique(quantized))
    bit_utilization = np.log2(unique_values) / bits
    
    return {
        'mse': mse,
        'sqnr': sqnr,
        'cosine_similarity': cos_sim,
        'bit_utilization': bit_utilization
    }

# Example evaluation
original_weights = np.random.normal(0, 1, (1000,))
quantized_weights = symmetric_quantize(original_weights, bits=8)[0]
metrics = quantization_metrics(original_weights, quantized_weights, bits=8)

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

Slide 13: Additional Resources

*   "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
    *   [https://arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877)
*   "Post-Training 4-bit Quantization of Convolutional Networks for Rapid-Deployment"
    *   [https://arxiv.org/abs/1810.05723](https://arxiv.org/abs/1810.05723)
*   "A Survey of Quantization Methods for Efficient Neural Network Inference"
    *   [https://arxiv.org/abs/2103.13630](https://arxiv.org/abs/2103.13630)
*   "ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers"
    *   [https://arxiv.org/abs/2206.01861](https://arxiv.org/abs/2206.01861)
*   "Understanding and Improving Knowledge Distillation"
    *   For more information about model compression techniques, search on Google Scholar

