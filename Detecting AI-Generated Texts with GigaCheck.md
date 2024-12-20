## Detecting AI-Generated Texts with GigaCheck
Slide 1: Text Classification with Mistral-7B

Text classification task implementation using the Mistral-7B model for AI-generated content detection. This approach utilizes fine-tuning techniques on a pre-trained model to achieve state-of-the-art performance in distinguishing between human and AI-generated text.

```python
import torch
from transformers import MistralForSequenceClassification, AutoTokenizer
import numpy as np

class TextClassifier:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = MistralForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            torch_dtype=torch.float16
        )
        
    def preprocess(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
    def predict(self, text):
        inputs = self.preprocess(text)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        return probs.detach().numpy()

# Example usage
classifier = TextClassifier()
text = "Sample text to analyze for AI detection"
prediction = classifier.predict(text)
print(f"Human probability: {prediction[0][0]:.3f}")
print(f"AI probability: {prediction[0][1]:.3f}")
```

Slide 2: Feature Extraction Pipeline

The feature extraction process is crucial for the DETR-based interval detection system. This implementation creates a robust pipeline to extract meaningful features from text using the fine-tuned Mistral model's hidden states.

```python
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, model_name="mistralai/Mistral-7B-v0.3"):
        super().__init__()
        self.base_model = MistralForSequenceClassification.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        self.feature_size = self.base_model.config.hidden_size
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Extract features from the last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        # Apply attention pooling
        attention_weights = attention_mask.unsqueeze(-1).float()
        weighted_states = last_hidden_state * attention_weights
        features = weighted_states.sum(dim=1) / attention_weights.sum(dim=1)
        return features

# Example usage
extractor = FeatureExtractor()
inputs = tokenizer("Example text", return_tensors="pt")
features = extractor(inputs['input_ids'], inputs['attention_mask'])
print(f"Feature shape: {features.shape}")
```

Slide 3: DN-DAB-DETR Model Architecture

Implementation of the Detection Transformer architecture adapted for text analysis. This model identifies specific intervals of AI-generated content within a given text using a modified version of the DN-DAB-DETR approach from computer vision.

```python
class TextDETR(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_decoder_layers=6):
        super().__init__()
        self.d_model = d_model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers
        )
        self.position_embedding = nn.Embedding(512, d_model)
        self.query_embed = nn.Embedding(100, d_model)
        self.linear_class = nn.Linear(d_model, 2)  # Binary classification
        self.linear_bbox = nn.Linear(d_model, 2)   # Start and end positions
        
    def forward(self, features):
        bs = features.shape[0]
        # Generate positional embeddings
        pos = torch.arange(features.size(1), device=features.device)
        pos_emb = self.position_embedding(pos)
        # Transform features
        memory = self.transformer.encoder(
            features + pos_emb.unsqueeze(0).repeat(bs, 1, 1)
        )
        # Generate queries
        tgt = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        # Decode
        hs = self.transformer.decoder(tgt, memory)
        # Predict classes and boundaries
        outputs_class = self.linear_class(hs)
        outputs_coord = self.linear_bbox(hs).sigmoid()
        return outputs_class, outputs_coord
```

Slide 4: Training Configuration

Comprehensive training setup for the AI detection system, including loss functions, optimizers, and training loop implementation. This configuration ensures effective model convergence and robust performance.

```python
class TrainingConfig:
    def __init__(self):
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.num_epochs = 100
        self.batch_size = 16
        
    def setup_training(self, model):
        criterion = {
            'cls': nn.CrossEntropyLoss(),
            'bbox': nn.L1Loss()
        }
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs
        )
        return criterion, optimizer, scheduler

# Training loop implementation
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        outputs_class, outputs_coord = model(batch['features'])
        
        loss_cls = criterion['cls'](
            outputs_class.transpose(1, 2),
            batch['labels']
        )
        loss_bbox = criterion['bbox'](
            outputs_coord,
            batch['boxes']
        )
        loss = loss_cls + 5 * loss_bbox
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

Slide 5: Data Preprocessing Pipeline

Advanced preprocessing pipeline designed specifically for AI-generated text detection. This implementation handles text tokenization, feature normalization, and dataset preparation for both classification and interval detection tasks.

```python
class DataPreprocessor:
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
        
    def prepare_dataset(self, texts, labels=None, intervals=None):
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create interval masks if provided
        if intervals:
            interval_masks = torch.zeros(
                (len(texts), self.max_length),
                dtype=torch.float32
            )
            for idx, interval_list in enumerate(intervals):
                for start, end in interval_list:
                    interval_masks[idx, start:end] = 1.0
                    
        # Prepare dataset dictionary
        dataset = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels) if labels else None,
            'interval_masks': interval_masks if intervals else None
        }
        
        return dataset

# Example usage
preprocessor = DataPreprocessor()
texts = [
    "This is a human-written text.",
    "This is an AI-generated text sample."
]
labels = [0, 1]  # 0: human, 1: AI
intervals = [[], [(0, 15)]]  # Example intervals

dataset = preprocessor.prepare_dataset(texts, labels, intervals)
print(f"Dataset shapes:")
print(f"Input IDs: {dataset['input_ids'].shape}")
print(f"Attention Mask: {dataset['attention_mask'].shape}")
print(f"Interval Masks: {dataset['interval_masks'].shape}")
```

Slide 6: Model Evaluation Metrics

Implementation of comprehensive evaluation metrics for both text classification and interval detection tasks. This includes accuracy, precision, recall, F1-score, and specialized metrics for interval boundary detection.

```python
class EvaluationMetrics:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.total_samples = 0
        self.interval_iou_scores = []
        
    def update(self, predictions, targets, intervals_pred=None, intervals_true=None):
        # Classification metrics
        self.true_positives += ((predictions == 1) & (targets == 1)).sum().item()
        self.false_positives += ((predictions == 1) & (targets == 0)).sum().item()
        self.false_negatives += ((predictions == 0) & (targets == 1)).sum().item()
        self.total_samples += len(predictions)
        
        # Interval detection metrics
        if intervals_pred and intervals_true:
            for pred, true in zip(intervals_pred, intervals_true):
                iou = self.calculate_interval_iou(pred, true)
                self.interval_iou_scores.append(iou)
    
    def calculate_interval_iou(self, pred_intervals, true_intervals):
        intersection = 0
        union = 0
        
        # Convert intervals to sets of positions
        pred_positions = set()
        true_positions = set()
        
        for start, end in pred_intervals:
            pred_positions.update(range(start, end))
        for start, end in true_intervals:
            true_positions.update(range(start, end))
            
        intersection = len(pred_positions & true_positions)
        union = len(pred_positions | true_positions)
        
        return intersection / union if union > 0 else 0.0
    
    def get_metrics(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = self.true_positives / self.total_samples
        mean_iou = np.mean(self.interval_iou_scores) if self.interval_iou_scores else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'mean_iou': mean_iou
        }

# Example usage
evaluator = EvaluationMetrics()
predictions = torch.tensor([1, 0, 1, 1, 0])
targets = torch.tensor([1, 0, 1, 0, 1])
intervals_pred = [[(0, 10), (20, 30)]]
intervals_true = [[(5, 15), (25, 35)]]

evaluator.update(predictions, targets, intervals_pred, intervals_true)
metrics = evaluator.get_metrics()
print("Evaluation Results:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

Slide 7: Paraphrasing Attack Defense

Implementation of defensive mechanisms against paraphrasing attacks in AI text detection. This system analyzes semantic similarity and structural patterns to maintain detection accuracy even when texts are paraphrased.

```python
class ParaphraseDefense:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = MistralForSequenceClassification.from_pretrained(model_name)
        self.semantic_threshold = 0.85
        
    def extract_semantic_features(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        # Get embeddings from the last hidden state
        embeddings = outputs.hidden_states[-1].mean(dim=1)
        return embeddings
    
    def calculate_similarity(self, text1, text2):
        emb1 = self.extract_semantic_features(text1)
        emb2 = self.extract_semantic_features(text2)
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
        return similarity.item()
    
    def detect_paraphrase_attack(self, original_text, suspicious_text):
        similarity = self.calculate_similarity(original_text, suspicious_text)
        
        # Structural analysis
        orig_stats = self.get_text_statistics(original_text)
        susp_stats = self.get_text_statistics(suspicious_text)
        
        structure_match = all(
            abs(orig_stats[k] - susp_stats[k]) < 0.2 
            for k in orig_stats
        )
        
        return {
            'similarity_score': similarity,
            'structure_match': structure_match,
            'is_potential_attack': similarity > self.semantic_threshold and structure_match
        }
    
    def get_text_statistics(self, text):
        sentences = text.split('.')
        return {
            'avg_sentence_length': sum(len(s.split()) for s in sentences) / len(sentences),
            'vocabulary_richness': len(set(text.split())) / len(text.split()),
            'punctuation_ratio': sum(c in '.,!?' for c in text) / len(text)
        }

# Example usage
defender = ParaphraseDefense()
original = "The quick brown fox jumps over the lazy dog."
paraphrased = "A swift brown fox leaps across the inactive canine."

result = defender.detect_paraphrase_attack(original, paraphrased)
print("Paraphrase Attack Analysis:")
print(f"Similarity Score: {result['similarity_score']:.3f}")
print(f"Structure Match: {result['structure_match']}")
print(f"Potential Attack: {result['is_potential_attack']}")
```

Slide 8: Out-of-Distribution Detection

Implementation of robust out-of-distribution (OOD) detection mechanism to identify text samples that differ significantly from the training distribution, enhancing the model's reliability on unseen data.

```python
class OODDetector:
    def __init__(self, feature_extractor, training_features=None):
        self.feature_extractor = feature_extractor
        self.training_distribution = None
        if training_features is not None:
            self.fit_distribution(training_features)
            
    def fit_distribution(self, features):
        # Calculate mean and covariance of training features
        self.mean = torch.mean(features, dim=0)
        self.cov = torch.cov(features.T)
        
        # Calculate Mahalanobis distances for training set
        distances = self.calculate_mahalanobis(features)
        self.threshold = torch.quantile(distances, 0.95)  # 95th percentile
        
    def calculate_mahalanobis(self, features):
        diff = features - self.mean.unsqueeze(0)
        inv_covmat = torch.linalg.inv(self.cov)
        left = torch.mm(diff, inv_covmat)
        mahalanobis = torch.sum(left * diff, dim=1)
        return torch.sqrt(mahalanobis)
    
    def detect(self, text, return_score=False):
        # Extract features
        features = self.feature_extractor.extract(text)
        
        # Calculate Mahalanobis distance
        distance = self.calculate_mahalanobis(features)
        
        # Determine if sample is OOD
        is_ood = distance > self.threshold
        
        if return_score:
            return is_ood, distance
        return is_ood
    
    def get_confidence_score(self, distance):
        # Convert distance to confidence score (0 to 1)
        return torch.exp(-distance / self.threshold)

# Example usage
def create_sample_distribution():
    # Create synthetic features for demonstration
    return torch.randn(1000, 768)  # 1000 samples, 768 features

training_features = create_sample_distribution()
detector = OODDetector(feature_extractor, training_features)

# Test sample
test_text = "This is a test sample that might be out of distribution."
is_ood, distance = detector.detect(test_text, return_score=True)
confidence = detector.get_confidence_score(distance)

print(f"OOD Detection Results:")
print(f"Is Out-of-Distribution: {is_ood}")
print(f"Confidence Score: {confidence:.3f}")
```

Slide 9: Real-time Analysis Pipeline

Implementation of a real-time analysis system that processes streaming text input and provides immediate AI detection results. This pipeline combines classification and interval detection with efficient batch processing.

```python
class RealTimeAnalyzer:
    def __init__(self, classifier, interval_detector, batch_size=8):
        self.classifier = classifier
        self.interval_detector = interval_detector
        self.batch_size = batch_size
        self.buffer = []
        
    async def process_stream(self, text_stream):
        async for text in text_stream:
            self.buffer.append(text)
            
            if len(self.buffer) >= self.batch_size:
                results = await self.process_batch()
                yield results
                self.buffer = []
    
    async def process_batch(self):
        # Prepare batch
        features = self.prepare_features(self.buffer)
        
        # Run classification
        cls_results = await self.classify_batch(features)
        
        # Run interval detection for AI-flagged texts
        interval_results = await self.detect_intervals(
            features[cls_results['is_ai']]
        )
        
        return {
            'classifications': cls_results,
            'intervals': interval_results,
            'processing_time': time.time()
        }
    
    async def classify_batch(self, features):
        with torch.no_grad():
            outputs = self.classifier(features)
            probs = torch.softmax(outputs, dim=1)
            is_ai = probs[:, 1] > 0.5
            
        return {
            'is_ai': is_ai,
            'confidence': probs.max(dim=1).values
        }
    
    async def detect_intervals(self, features):
        with torch.no_grad():
            outputs = self.interval_detector(features)
            
        return self.post_process_intervals(outputs)
    
    def post_process_intervals(self, detector_outputs):
        boxes = detector_outputs['pred_boxes']
        scores = detector_outputs['pred_logits'].softmax(-1)
        
        # Filter by confidence
        mask = scores > 0.7
        filtered_boxes = boxes[mask]
        
        # Non-maximum suppression
        kept_indices = nms(filtered_boxes, scores[mask], iou_threshold=0.5)
        
        return filtered_boxes[kept_indices]

# Example usage
async def main():
    analyzer = RealTimeAnalyzer(classifier, interval_detector)
    
    async def text_generator():
        texts = [
            "This is a human written text.",
            "This is an AI generated text sample.",
            # ... more texts
        ]
        for text in texts:
            yield text
            await asyncio.sleep(0.1)  # Simulate stream delay
    
    async for results in analyzer.process_stream(text_generator()):
        print("\nBatch Results:")
        print(f"Processed at: {results['processing_time']}")
        print(f"AI Detected: {results['classifications']['is_ai'].sum().item()}")
        print(f"Intervals Found: {len(results['intervals'])}")

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

Slide 10: Performance Optimization Module

Implementation of optimization techniques to improve model inference speed and memory efficiency. This module includes quantization, pruning, and caching mechanisms for production deployment.

```python
class PerformanceOptimizer:
    def __init__(self, model, quantization_config=None):
        self.model = model
        self.cache = {}
        self.quantized_model = None
        self.pruned_model = None
        
    def quantize_model(self, calibration_data=None):
        """Applies dynamic quantization to the model"""
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
        
        if calibration_data is not None:
            self._calibrate_model(calibration_data)
        
        return self.quantized_model
    
    def prune_model(self, importance_scores=None):
        """Applies magnitude-based pruning"""
        parameters_to_prune = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        
        if importance_scores is None:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=0.2,  # Prune 20% of connections
            )
        else:
            for (module, name), score in zip(parameters_to_prune, importance_scores):
                prune.l1_unstructured(module, name, amount=score)
        
        self.pruned_model = self.model
        return self.pruned_model
    
    def cache_predictions(self, text_hash, prediction):
        """Caches prediction results"""
        self.cache[text_hash] = {
            'prediction': prediction,
            'timestamp': time.time()
        }
        
    def get_cached_prediction(self, text_hash, max_age=3600):
        """Retrieves cached prediction if available and not expired"""
        if text_hash in self.cache:
            cache_entry = self.cache[text_hash]
            if time.time() - cache_entry['timestamp'] < max_age:
                return cache_entry['prediction']
        return None
    
    def benchmark_performance(self, test_data):
        """Measures inference speed and memory usage"""
        results = {
            'original': self._measure_performance(self.model, test_data),
            'quantized': self._measure_performance(self.quantized_model, test_data)
            if self.quantized_model else None,
            'pruned': self._measure_performance(self.pruned_model, test_data)
            if self.pruned_model else None
        }
        return results
    
    def _measure_performance(self, model, test_data):
        start_time = time.time()
        memory_start = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            for batch in test_data:
                _ = model(batch)
        
        return {
            'inference_time': time.time() - start_time,
            'memory_usage': torch.cuda.memory_allocated() - memory_start
        }

# Example usage
optimizer = PerformanceOptimizer(model)

# Quantize model
quantized_model = optimizer.quantize_model()

# Prune model
pruned_model = optimizer.prune_model()

# Benchmark performance
test_data = [(torch.randn(32, 512), torch.randn(32, 2)) for _ in range(10)]
performance_metrics = optimizer.benchmark_performance(test_data)

print("Performance Metrics:")
for model_type, metrics in performance_metrics.items():
    if metrics:
        print(f"\n{model_type.title()} Model:")
        print(f"Inference Time: {metrics['inference_time']:.3f} seconds")
        print(f"Memory Usage: {metrics['memory_usage']/1e6:.2f} MB")
```

Slide 11: Multi-Model Ensemble Detection

Implementation of an ensemble system that combines predictions from multiple detection models. This approach improves robustness and accuracy by leveraging diverse model architectures and training strategies.

```python
class EnsembleDetector:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.normalize_weights()
        
    def normalize_weights(self):
        total = sum(self.weights)
        self.weights = [w/total for w in self.weights]
        
    def predict(self, text, return_confidence=False):
        predictions = []
        confidences = []
        
        for model, weight in zip(self.models, self.weights):
            # Get prediction and confidence from each model
            pred, conf = model.predict(text, return_confidence=True)
            predictions.append(pred * weight)
            confidences.append(conf * weight)
            
        # Weighted ensemble prediction
        ensemble_pred = torch.stack(predictions).sum(dim=0)
        ensemble_conf = torch.stack(confidences).sum(dim=0)
        
        # Apply threshold
        final_pred = (ensemble_pred > 0.5).float()
        
        if return_confidence:
            return final_pred, ensemble_conf
        return final_pred
    
    def detect_intervals(self, text):
        interval_predictions = []
        
        for model, weight in zip(self.models, self.weights):
            intervals = model.detect_intervals(text)
            interval_predictions.append((intervals, weight))
            
        return self.merge_intervals(interval_predictions)
    
    def merge_intervals(self, interval_predictions):
        all_intervals = []
        
        # Collect all intervals with their weights
        for intervals, weight in interval_predictions:
            for start, end, conf in intervals:
                all_intervals.append({
                    'start': start,
                    'end': end,
                    'score': conf * weight
                })
        
        # Merge overlapping intervals
        merged = self.merge_overlapping_intervals(all_intervals)
        return merged
    
    def merge_overlapping_intervals(self, intervals):
        if not intervals:
            return []
            
        # Sort by start position
        sorted_intervals = sorted(intervals, key=lambda x: x['start'])
        merged = [sorted_intervals[0]]
        
        for current in sorted_intervals[1:]:
            previous = merged[-1]
            
            # Check for overlap
            if current['start'] <= previous['end']:
                # Update end position and score
                previous['end'] = max(previous['end'], current['end'])
                previous['score'] = max(previous['score'], current['score'])
            else:
                merged.append(current)
                
        return merged

# Example usage
def create_sample_models():
    return [
        TextClassifier("model1"),
        TextClassifier("model2"),
        TextClassifier("model3")
    ]

# Initialize ensemble with different weights
models = create_sample_models()
weights = [0.4, 0.3, 0.3]  # Weights based on model performance
ensemble = EnsembleDetector(models, weights)

# Test prediction
test_text = "This is a sample text to analyze."
prediction, confidence = ensemble.predict(test_text, return_confidence=True)

print("Ensemble Detection Results:")
print(f"Prediction: {'AI' if prediction else 'Human'}")
print(f"Confidence: {confidence.item():.3f}")

# Test interval detection
intervals = ensemble.detect_intervals(test_text)
print("\nDetected Intervals:")
for interval in intervals:
    print(f"Start: {interval['start']}, End: {interval['end']}, "
          f"Score: {interval['score']:.3f}")
```

Slide 12: Cross-Domain Adaptation

Implementation of domain adaptation techniques to improve model performance across different text domains and styles. This module handles domain shift and maintains detection accuracy across various text sources.

```python
class DomainAdapter:
    def __init__(self, base_model, adaptation_rate=0.01):
        self.base_model = base_model
        self.adaptation_rate = adaptation_rate
        self.domain_specific_layers = nn.ModuleDict()
        self.domain_statistics = {}
        
    def adapt_to_domain(self, domain_name, domain_samples):
        # Create domain-specific adaptation layer if not exists
        if domain_name not in self.domain_specific_layers:
            self.domain_specific_layers[domain_name] = self.create_adaptation_layer()
        
        # Calculate domain statistics
        domain_stats = self.calculate_domain_statistics(domain_samples)
        self.domain_statistics[domain_name] = domain_stats
        
        # Adapt model to domain
        return self.adapt_model(domain_name, domain_samples)
    
    def create_adaptation_layer(self):
        return nn.Sequential(
            nn.Linear(768, 768),  # Match base model dimensions
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def calculate_domain_statistics(self, samples):
        features = []
        with torch.no_grad():
            for text in samples:
                feat = self.base_model.extract_features(text)
                features.append(feat)
        
        features = torch.stack(features)
        return {
            'mean': features.mean(dim=0),
            'std': features.std(dim=0),
            'feature_distribution': features
        }
    
    def adapt_model(self, domain_name, samples):
        adaptation_layer = self.domain_specific_layers[domain_name]
        optimizer = torch.optim.Adam(adaptation_layer.parameters(), 
                                   lr=self.adaptation_rate)
        
        for epoch in range(5):  # Quick adaptation
            for text in samples:
                features = self.base_model.extract_features(text)
                adapted_features = adaptation_layer(features)
                
                # Calculate adaptation loss
                loss = self.calculate_adaptation_loss(
                    adapted_features,
                    self.domain_statistics[domain_name]
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        return adaptation_layer
    
    def calculate_adaptation_loss(self, features, domain_stats):
        # MMD (Maximum Mean Discrepancy) loss
        source_features = domain_stats['feature_distribution']
        target_features = features.unsqueeze(0)
        
        mmd_loss = self.compute_mmd(source_features, target_features)
        return mmd_loss
    
    def compute_mmd(self, source, target, kernel='rbf'):
        diff = source.unsqueeze(1) - target.unsqueeze(0)
        if kernel == 'rbf':
            bandwidth = torch.median(torch.pdist(source))
            diff_norm = torch.sum(diff ** 2, dim=-1)
            kernel_val = torch.exp(-diff_norm / (2 * bandwidth ** 2))
        else:
            kernel_val = torch.sum(diff ** 2, dim=-1)
        
        return torch.mean(kernel_val)

# Example usage
adapter = DomainAdapter(base_model)

# Adapt to new domain
domain_samples = [
    "Technical documentation example text.",
    "Scientific paper abstract sample.",
    "Academic writing instance."
]

adapted_layer = adapter.adapt_to_domain("academic", domain_samples)

# Use adapted model for prediction
test_text = "This is a technical research paper introduction."
features = base_model.extract_features(test_text)
adapted_features = adapted_layer(features)

print("Domain Adaptation Results:")
print(f"Original feature norm: {features.norm().item():.3f}")
print(f"Adapted feature norm: {adapted_features.norm().item():.3f}")
```

Slide 13: Results and Performance Analysis

Comprehensive implementation of a performance analysis system that evaluates the detection model across multiple metrics and datasets. This module generates detailed reports on model accuracy, efficiency, and reliability.

```python
class PerformanceAnalyzer:
    def __init__(self):
        self.metrics = {
            'classification': defaultdict(list),
            'interval_detection': defaultdict(list),
            'runtime': defaultdict(list)
        }
        
    def analyze_model_performance(self, model, test_data, detailed=True):
        results = {
            'classification_metrics': self.evaluate_classification(model, test_data),
            'interval_metrics': self.evaluate_intervals(model, test_data),
            'runtime_metrics': self.measure_runtime(model, test_data),
            'robustness_score': self.evaluate_robustness(model, test_data)
        }
        
        if detailed:
            results.update({
                'confusion_matrix': self.generate_confusion_matrix(model, test_data),
                'roc_curve': self.generate_roc_curve(model, test_data),
                'error_analysis': self.analyze_errors(model, test_data)
            })
        
        return results
    
    def evaluate_classification(self, model, test_data):
        predictions = []
        labels = []
        
        for text, label in test_data:
            pred = model.predict(text)
            predictions.append(pred)
            labels.append(label)
            
        predictions = torch.stack(predictions)
        labels = torch.stack(labels)
        
        return {
            'accuracy': (predictions == labels).float().mean().item(),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1': f1_score(labels, predictions),
            'auc_roc': roc_auc_score(labels, predictions)
        }
    
    def evaluate_intervals(self, model, test_data):
        iou_scores = []
        boundary_accuracy = []
        
        for text, intervals in test_data:
            pred_intervals = model.detect_intervals(text)
            iou = self.calculate_interval_iou(pred_intervals, intervals)
            boundary_acc = self.evaluate_boundary_accuracy(pred_intervals, intervals)
            
            iou_scores.append(iou)
            boundary_accuracy.append(boundary_acc)
            
        return {
            'mean_iou': np.mean(iou_scores),
            'boundary_accuracy': np.mean(boundary_accuracy),
            'interval_precision': self.calculate_interval_precision(iou_scores)
        }
    
    def measure_runtime(self, model, test_data):
        processing_times = []
        memory_usage = []
        
        for text, _ in test_data:
            start_time = time.time()
            start_memory = torch.cuda.memory_allocated()
            
            _ = model.predict(text)
            
            processing_times.append(time.time() - start_time)
            memory_usage.append(torch.cuda.memory_allocated() - start_memory)
            
        return {
            'avg_processing_time': np.mean(processing_times),
            'std_processing_time': np.std(processing_times),
            'peak_memory_usage': max(memory_usage) / 1e6,  # MB
            'avg_memory_usage': np.mean(memory_usage) / 1e6  # MB
        }
    
    def evaluate_robustness(self, model, test_data):
        perturbation_scores = []
        
        for text, label in test_data:
            perturbed_texts = self.generate_perturbations(text)
            stability_score = self.calculate_stability(model, text, perturbed_texts)
            perturbation_scores.append(stability_score)
            
        return np.mean(perturbation_scores)
    
    @staticmethod
    def generate_perturbations(text, num_perturbations=5):
        perturbations = []
        words = text.split()
        
        for _ in range(num_perturbations):
            perturbed = words.copy()
            # Apply random perturbations
            if len(perturbed) > 1:
                idx1, idx2 = random.sample(range(len(perturbed)), 2)
                perturbed[idx1], perturbed[idx2] = perturbed[idx2], perturbed[idx1]
            perturbations.append(' '.join(perturbed))
            
        return perturbations

# Example usage
analyzer = PerformanceAnalyzer()
test_dataset = [
    ("This is a human-written text.", 0),
    ("This is an AI-generated text.", 1),
    # ... more test samples
]

results = analyzer.analyze_model_performance(model, test_dataset)

print("Performance Analysis Results:")
print("\nClassification Metrics:")
for metric, value in results['classification_metrics'].items():
    print(f"{metric}: {value:.3f}")

print("\nInterval Detection Metrics:")
for metric, value in results['interval_metrics'].items():
    print(f"{metric}: {value:.3f}")

print("\nRuntime Metrics:")
for metric, value in results['runtime_metrics'].items():
    print(f"{metric}: {value:.3f}")

print(f"\nRobustness Score: {results['robustness_score']:.3f}")
```

Slide 14: Additional Resources

*   Technical paper on GigaCheck approach: [https://arxiv.org/abs/2401.01871](https://arxiv.org/abs/2401.01871)
*   Detection Transformer fundamentals: [https://arxiv.org/abs/2005.12872](https://arxiv.org/abs/2005.12872)
*   Recent advances in AI text detection: [https://arxiv.org/abs/2310.05023](https://arxiv.org/abs/2310.05023)
*   Broader survey on AI content detection: [https://www.sciencedirect.com/science/article/pii/S0306457323001528](https://www.sciencedirect.com/science/article/pii/S0306457323001528)
*   Search suggestions:
    *   "Detection Transformer for text analysis"
    *   "AI generated text detection techniques"
    *   "GigaCheck implementation details"
    *   "Out-of-distribution detection in NLP"

