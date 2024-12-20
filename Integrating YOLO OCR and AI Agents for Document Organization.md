## Integrating YOLO OCR and AI Agents for Document Organization

Slide 1: Understanding YOLO Architecture for Document Detection

The YOLO (You Only Look Once) architecture revolutionizes document detection by treating it as a regression problem, directly predicting bounding boxes and class probabilities through a single neural network evaluation, enabling real-time processing of document images with high accuracy.

```python
import torch
import torch.nn as nn

class DocumentYOLO(nn.Module):
    def __init__(self, num_classes=4):  # Common document classes: text, table, figure, heading
        super(DocumentYOLO, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.grid_size = 7
        self.num_boxes = 2
        self.num_classes = num_classes
        
        # Output tensor shape: (batch_size, grid_size, grid_size, (num_boxes * 5 + num_classes))
        self.output_size = self.num_boxes * 5 + num_classes
        
    def forward(self, x):
        x = self.features(x)
        return x.view(-1, self.grid_size, self.grid_size, self.output_size)

# Example usage
model = DocumentYOLO()
dummy_input = torch.randn(1, 3, 448, 448)
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # Expected: torch.Size([1, 7, 7, 14])
```

Slide 2: Advanced OCR Pipeline Implementation

Modern OCR systems combine traditional image processing with deep learning techniques to achieve superior text recognition accuracy. This implementation showcases a complete pipeline including preprocessing, text detection, and recognition stages.

```python
import cv2
import numpy as np
from PIL import Image
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class AdvancedOCR:
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
        
    def preprocess_image(self, image):
        # Convert to grayscale and apply adaptive thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        return thresh
    
    def extract_text(self, image):
        preprocessed = self.preprocess_image(image)
        
        # Convert image for transformer input
        pil_image = Image.fromarray(preprocessed)
        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
        
        # Generate text
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text

# Example usage
ocr = AdvancedOCR()
image = cv2.imread('document.png')
text = ocr.extract_text(image)
print(f"Extracted text: {text}")
```

Slide 3: Language Model Integration for Document Understanding

The integration of Large Language Models with document processing enables sophisticated text analysis and understanding. This implementation demonstrates how to combine OCR output with LLM-based text processing for enhanced document comprehension.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class DocumentUnderstanding:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=4  # Document categories
        )
        
    def analyze_document(self, text):
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Get model predictions
        outputs = self.model(**inputs)
        predictions = F.softmax(outputs.logits, dim=1)
        
        # Document structure analysis
        doc_structure = {
            'sections': self._identify_sections(text),
            'key_points': self._extract_key_points(text),
            'classification': predictions.argmax().item()
        }
        return doc_structure
    
    def _identify_sections(self, text):
        # Simple section identification based on newlines and headers
        sections = []
        current_section = ""
        for line in text.split('\n'):
            if line.isupper() or line.endswith(':'):
                if current_section:
                    sections.append(current_section.strip())
                current_section = line
            else:
                current_section += f"\n{line}"
        sections.append(current_section.strip())
        return sections
    
    def _extract_key_points(self, text):
        # Extract important points using basic heuristics
        sentences = text.split('.')
        key_points = [s.strip() for s in sentences if any(kw in s.lower() 
                     for kw in ['important', 'key', 'significant', 'must', 'should'])]
        return key_points

# Example usage
doc_analyzer = DocumentUnderstanding()
sample_text = """
EXECUTIVE SUMMARY:
This document outlines important findings from our research.
Key results indicate significant improvements in performance.
"""
analysis = doc_analyzer.analyze_document(sample_text)
print(f"Document analysis: {analysis}")
```

Slide 4: Intelligent Agent Framework for Document Organization

The Intelligent Agent framework implements a multi-agent system for autonomous document organization, utilizing reinforcement learning for decision-making and collaborative document processing between agents, each specialized in different aspects of document handling.

```python
import numpy as np
from typing import List, Dict
import random

class DocumentAgent:
    def __init__(self, specialization: str, learning_rate: float = 0.1):
        self.specialization = specialization
        self.learning_rate = learning_rate
        self.q_table = {}
        self.state_history = []
        
    def get_state(self, document: Dict) -> str:
        # Create a state representation based on document features
        features = [
            document.get('type', 'unknown'),
            document.get('size', 0) > 1000000,
            bool(document.get('text_content')),
            bool(document.get('images'))
        ]
        return '_'.join(map(str, features))
    
    def choose_action(self, state: str, epsilon: float = 0.1) -> str:
        if state not in self.q_table:
            self.q_table[state] = {
                'process': 0.0,
                'delegate': 0.0,
                'archive': 0.0
            }
        
        if random.random() < epsilon:
            return random.choice(list(self.q_table[state].keys()))
        return max(self.q_table[state].items(), key=lambda x: x[1])[0]
    
    def update_q_value(self, state: str, action: str, reward: float):
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0}
        
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.learning_rate * (reward - current_q)

class DocumentOrganizer:
    def __init__(self):
        self.agents = {
            'classifier': DocumentAgent('classification'),
            'ocr': DocumentAgent('text_extraction'),
            'archiver': DocumentAgent('archiving')
        }
        
    def process_document(self, document: Dict) -> Dict:
        results = {}
        for agent_name, agent in self.agents.items():
            state = agent.get_state(document)
            action = agent.choose_action(state)
            
            # Simulate processing and get reward
            reward = self._execute_action(action, document)
            agent.update_q_value(state, action, reward)
            
            results[agent_name] = {
                'action': action,
                'reward': reward
            }
        
        return results
    
    def _execute_action(self, action: str, document: Dict) -> float:
        # Simulate action execution and return reward
        if action == 'process':
            return 1.0 if document.get('size', 0) < 1000000 else 0.5
        elif action == 'delegate':
            return 0.7
        else:  # archive
            return 0.3

# Example usage
organizer = DocumentOrganizer()
sample_document = {
    'type': 'pdf',
    'size': 500000,
    'text_content': True,
    'images': True
}

results = organizer.process_document(sample_document)
print(f"Processing results: {results}")
```

Slide 5: Custom YOLO Loss Function for Document Detection

A specialized loss function for document detection combines boundary box regression with class prediction penalties, incorporating IoU (Intersection over Union) calculations and confidence scores specifically tailored for document layout analysis.

```python
import torch
import torch.nn as nn

class DocumentYOLOLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super(DocumentYOLOLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def compute_iou(self, box1, box2):
        # Calculate intersection over union
        x1 = torch.max(box1[..., 0], box2[..., 0])
        y1 = torch.max(box1[..., 1], box2[..., 1])
        x2 = torch.min(box1[..., 2], box2[..., 2])
        y2 = torch.min(box1[..., 3], box2[..., 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        
        union = box1_area + box2_area - intersection
        return intersection / (union + 1e-6)
    
    def forward(self, predictions, targets):
        # Predictions shape: (batch_size, grid_size, grid_size, num_boxes * 5 + num_classes)
        batch_size = predictions.size(0)
        grid_size = predictions.size(1)
        
        # Separate coordinates, confidence, and class predictions
        pred_coords = predictions[..., :4]
        pred_conf = predictions[..., 4]
        pred_classes = predictions[..., 5:]
        
        target_coords = targets[..., :4]
        target_conf = targets[..., 4]
        target_classes = targets[..., 5:]
        
        # Compute coordinate loss (only for cells with objects)
        coord_mask = target_conf.unsqueeze(-1).expand_as(pred_coords)
        coord_loss = self.lambda_coord * torch.sum(
            coord_mask * torch.pow(pred_coords - target_coords, 2)
        )
        
        # Compute confidence loss
        iou = self.compute_iou(pred_coords, target_coords)
        conf_loss = torch.sum(
            target_conf * torch.pow(pred_conf - iou, 2) +
            self.lambda_noobj * (1 - target_conf) * torch.pow(pred_conf, 2)
        )
        
        # Compute classification loss
        class_loss = torch.sum(
            target_conf.unsqueeze(-1) * torch.pow(pred_classes - target_classes, 2)
        )
        
        total_loss = (coord_loss + conf_loss + class_loss) / batch_size
        return total_loss

# Example usage
loss_fn = DocumentYOLOLoss()
pred = torch.randn(4, 7, 7, 9)  # Batch of 4, 7x7 grid, 4 coords + 1 conf + 4 classes
target = torch.zeros(4, 7, 7, 9)
loss = loss_fn(pred, target)
print(f"Total loss: {loss.item()}")
```

Slide 6: Advanced Document Text Extraction Pipeline

This implementation demonstrates a sophisticated text extraction pipeline combining traditional OCR with deep learning-based post-processing for improved accuracy. The system handles various document layouts and text styles through a multi-stage approach.

```python
import cv2
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class TextRegion:
    bbox: Tuple[int, int, int, int]
    text: str
    confidence: float

class DocumentTextExtractor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.correction_model = AutoModelForSeq2SeqGeneration.from_pretrained("t5-small")
        
    def extract_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        # Apply EAST text detector
        scaled = cv2.resize(image, (320, 320))
        blob = cv2.dnn.blobFromImage(scaled, 1.0, (320, 320),
                                   (123.68, 116.78, 103.94), True, False)
        
        # Simulated EAST network output
        scores = self._detect_text_regions(blob)
        boxes = self._get_geometry(scores)
        
        regions = []
        for box in boxes:
            roi = self._extract_roi(image, box)
            text = self._ocr_with_correction(roi)
            confidence = self._calculate_confidence(text)
            regions.append(TextRegion(box, text, confidence))
            
        return regions
    
    def _detect_text_regions(self, blob: np.ndarray) -> np.ndarray:
        # Simplified text detection
        return np.random.rand(1, 1, 80, 80)
    
    def _get_geometry(self, scores: np.ndarray) -> List[Tuple[int, int, int, int]]:
        # Convert scores to bounding boxes
        boxes = []
        thresh = 0.5
        for y in range(scores.shape[2]):
            for x in range(scores.shape[3]):
                if scores[0, 0, y, x] > thresh:
                    boxes.append((x*4, y*4, (x+1)*4, (y+1)*4))
        return boxes
    
    def _extract_roi(self, image: np.ndarray, 
                    bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]
    
    def _ocr_with_correction(self, roi: np.ndarray) -> str:
        # Simulate OCR
        raw_text = "smaple text with erors"
        
        # Correct OCR errors using transformer
        inputs = self.tokenizer(f"correct: {raw_text}", 
                              return_tensors="pt", 
                              max_length=128, 
                              truncation=True)
        
        outputs = self.correction_model.generate(**inputs)
        corrected_text = self.tokenizer.decode(outputs[0], 
                                             skip_special_tokens=True)
        
        return corrected_text
    
    def _calculate_confidence(self, text: str) -> float:
        # Simple confidence calculation based on text length and characteristics
        if not text:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Adjust based on text properties
        if len(text) > 3:
            confidence += 0.2
        if any(c.isupper() for c in text):
            confidence += 0.1
        if any(c.isdigit() for c in text):
            confidence += 0.1
            
        return min(confidence, 1.0)

# Example usage
extractor = DocumentTextExtractor()
image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
regions = extractor.extract_text_regions(image)

print("Extracted Text Regions:")
for region in regions:
    print(f"Box: {region.bbox}")
    print(f"Text: {region.text}")
    print(f"Confidence: {region.confidence:.2f}")
    print("---")
```

Slide 7: Intelligent Document Classification Model

```python
import torch
import torch.nn as nn
from torchvision.models import resnet50
from typing import Dict, List

class DocumentClassifier(nn.Module):
    def __init__(self, num_classes: int = 5):
        super(DocumentClassifier, self).__init__()
        self.backbone = resnet50(pretrained=True)
        
        # Modify final layer for document classification
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Additional layers for layout analysis
        self.layout_branch = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features from backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        features = self.backbone.layer4(x)
        
        # Global classification branch
        global_pool = self.backbone.avgpool(features)
        global_pool = torch.flatten(global_pool, 1)
        classification = self.backbone.fc(global_pool)
        
        # Layout analysis branch
        layout = self.layout_branch(features)
        
        return {
            'classification': classification,
            'layout': layout
        }

class DocumentClassificationTrainer:
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        images = batch['image'].to(self.device)
        class_labels = batch['class_label'].to(self.device)
        layout_labels = batch['layout_label'].to(self.device)
        
        outputs = self.model(images)
        
        # Calculate losses
        class_loss = self.criterion(outputs['classification'], class_labels)
        layout_loss = self.criterion(outputs['layout'], layout_labels)
        total_loss = class_loss + 0.5 * layout_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'class_loss': class_loss.item(),
            'layout_loss': layout_loss.item()
        }
        
    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        correct_class = 0
        correct_layout = 0
        total = 0
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            class_labels = batch['class_label'].to(self.device)
            layout_labels = batch['layout_label'].to(self.device)
            
            outputs = self.model(images)
            
            # Calculate accuracy
            class_pred = outputs['classification'].argmax(dim=1)
            layout_pred = outputs['layout'].argmax(dim=1)
            
            correct_class += (class_pred == class_labels).sum().item()
            correct_layout += (layout_pred == layout_labels).sum().item()
            total += images.size(0)
        
        return {
            'class_accuracy': correct_class / total,
            'layout_accuracy': correct_layout / total
        }

# Example usage
model = DocumentClassifier()
trainer = DocumentClassificationTrainer(model)

# Simulate batch
batch = {
    'image': torch.randn(4, 3, 224, 224),
    'class_label': torch.randint(0, 5, (4,)),
    'layout_label': torch.randint(0, 5, (4,))
}

losses = trainer.train_step(batch)
print(f"Training losses: {losses}")
```

Slide 8: Document Layout Analysis Using Transformer Architecture

A specialized transformer-based architecture for document layout analysis that processes both visual and spatial information, incorporating attention mechanisms to capture relationships between different document regions.

```python
import torch
import torch.nn as nn
import math

class LayoutTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6):
        super(LayoutTransformer, self).__init__()
        
        # Positional encoding for layout information
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Layout features embedding
        self.layout_embedding = nn.Sequential(
            nn.Linear(5, d_model),  # x, y, width, height, class_id
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, 4)  # 4 layout classes
        
    def forward(self, layout_features, masks=None):
        # layout_features: (batch_size, seq_len, 5)
        batch_size, seq_len, _ = layout_features.shape
        
        # Embed layout features
        x = self.layout_embedding(layout_features)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transpose for transformer: (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Transform masks if provided
        if masks is not None:
            masks = masks.bool()
        
        # Pass through transformer
        encoded = self.transformer_encoder(x, src_key_padding_mask=masks)
        
        # Transform back: (batch_size, seq_len, d_model)
        encoded = encoded.transpose(0, 1)
        
        # Project to output space
        output = self.output_proj(encoded)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DocumentLayoutAnalyzer:
    def __init__(self, model_dim=512):
        self.model = LayoutTransformer(d_model=model_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_batch(self, features, labels, masks=None):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(features, masks)
        loss = self.criterion(outputs.view(-1, 4), labels.view(-1))
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def analyze_layout(self, document_features):
        self.model.eval()
        outputs = self.model(document_features)
        predictions = torch.argmax(outputs, dim=-1)
        return predictions

# Example usage
layout_analyzer = DocumentLayoutAnalyzer()

# Simulate batch of document layout features
batch_size = 4
seq_len = 10
features = torch.randn(batch_size, seq_len, 5)  # 5 features per region
labels = torch.randint(0, 4, (batch_size, seq_len))  # 4 layout classes
masks = torch.zeros(batch_size, seq_len, dtype=torch.bool)

# Train on batch
loss = layout_analyzer.train_batch(features, labels, masks)
print(f"Training loss: {loss:.4f}")

# Analyze new document
new_document = torch.randn(1, seq_len, 5)
layout_predictions = layout_analyzer.analyze_layout(new_document)
print(f"Layout predictions: {layout_predictions}")
```

Slide 9: Multi-Agent Document Processing System

```python
from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import asyncio

class AgentType(Enum):
    CLASSIFIER = "classifier"
    EXTRACTOR = "extractor"
    ANALYZER = "analyzer"
    COORDINATOR = "coordinator"

@dataclass
class DocumentTask:
    task_id: str
    task_type: str
    priority: int
    data: Dict[str, Any]
    status: str = "pending"

class DocumentAgent:
    def __init__(self, agent_type: AgentType, processing_capacity: int = 5):
        self.agent_type = agent_type
        self.processing_capacity = processing_capacity
        self.task_queue = asyncio.Queue()
        self.processing = False
        
    async def process_task(self, task: DocumentTask) -> Dict[str, Any]:
        if self.agent_type == AgentType.CLASSIFIER:
            return await self._classify_document(task)
        elif self.agent_type == AgentType.EXTRACTOR:
            return await self._extract_content(task)
        elif self.agent_type == AgentType.ANALYZER:
            return await self._analyze_document(task)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")
    
    async def _classify_document(self, task: DocumentTask) -> Dict[str, Any]:
        # Simulate document classification
        await asyncio.sleep(0.5)  # Simulate processing time
        return {
            'task_id': task.task_id,
            'document_type': np.random.choice(['invoice', 'contract', 'report']),
            'confidence': np.random.uniform(0.8, 1.0)
        }
    
    async def _extract_content(self, task: DocumentTask) -> Dict[str, Any]:
        await asyncio.sleep(0.7)
        return {
            'task_id': task.task_id,
            'extracted_text': f"Sample text for document {task.task_id}",
            'metadata': {'pages': np.random.randint(1, 10)}
        }
    
    async def _analyze_document(self, task: DocumentTask) -> Dict[str, Any]:
        await asyncio.sleep(0.6)
        return {
            'task_id': task.task_id,
            'analysis': {
                'sentiment': np.random.choice(['positive', 'neutral', 'negative']),
                'key_topics': ['topic1', 'topic2']
            }
        }

class DocumentProcessingSystem:
    def __init__(self):
        self.agents = {
            AgentType.CLASSIFIER: DocumentAgent(AgentType.CLASSIFIER),
            AgentType.EXTRACTOR: DocumentAgent(AgentType.EXTRACTOR),
            AgentType.ANALYZER: DocumentAgent(AgentType.ANALYZER)
        }
        self.results = {}
        
    async def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        task_id = f"task_{len(self.results)}"
        task = DocumentTask(task_id, "process", 1, document)
        
        # Execute pipeline
        classification = await self.agents[AgentType.CLASSIFIER].process_task(task)
        extraction = await self.agents[AgentType.EXTRACTOR].process_task(task)
        analysis = await self.agents[AgentType.ANALYZER].process_task(task)
        
        # Combine results
        result = {
            'task_id': task_id,
            'classification': classification,
            'extraction': extraction,
            'analysis': analysis
        }
        
        self.results[task_id] = result
        return result

# Example usage
async def main():
    system = DocumentProcessingSystem()
    
    # Process sample document
    document = {
        'content': 'Sample document content',
        'metadata': {'source': 'email'}
    }
    
    result = await system.process_document(document)
    print("Processing Results:")
    print(f"Classification: {result['classification']}")
    print(f"Extraction: {result['extraction']}")
    print(f"Analysis: {result['analysis']}")

# Run the example
asyncio.run(main())
```

Slide 10: Advanced OCR Post-Processing with NLP

Implementing sophisticated post-processing techniques to improve OCR accuracy by leveraging natural language processing models for context-aware error correction and text normalization.

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import Levenshtein
import re
from typing import List, Tuple, Dict
import numpy as np

class OCRPostProcessor:
    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
        self.common_corrections = self._load_common_corrections()
        
    def _load_common_corrections(self) -> Dict[str, str]:
        return {
            'tho': 'the',
            'thai': 'that',
            'ls': 'is',
            '0': 'O',
            '1': 'I',
            'I0': 'IO',
            'II': 'H'
        }
        
    def process_text(self, text: str, confidence_scores: List[float]) -> str:
        # Split text into words and align with confidence scores
        words = text.split()
        word_confidences = self._align_confidence_scores(words, confidence_scores)
        
        # Process each word
        corrected_words = []
        for word, conf in zip(words, word_confidences):
            if conf < self.confidence_threshold:
                corrected = self._correct_word(word)
            else:
                corrected = word
            corrected_words.append(corrected)
            
        return ' '.join(corrected_words)
    
    def _align_confidence_scores(self, 
                               words: List[str], 
                               scores: List[float]) -> List[float]:
        # Ensure scores align with words using linear interpolation
        word_scores = []
        score_idx = 0
        for word in words:
            word_len = len(word)
            word_score = np.mean(scores[score_idx:score_idx + word_len])
            word_scores.append(word_score)
            score_idx += word_len
        return word_scores
    
    def _correct_word(self, word: str) -> str:
        # Check common corrections first
        if word.lower() in self.common_corrections:
            return self.common_corrections[word.lower()]
        
        # Use BERT for context-aware correction
        context = self._get_context_window(word)
        masked_text = context.replace(word, self.tokenizer.mask_token)
        
        inputs = self.tokenizer(masked_text, return_tensors='pt')
        outputs = self.model(**inputs)
        
        # Get top predictions
        mask_idx = torch.where(inputs['input_ids'][0] == self.tokenizer.mask_token_id)[0]
        predictions = outputs.logits[0, mask_idx].softmax(dim=-1)
        top_tokens = torch.topk(predictions, k=5, dim=-1)
        
        candidates = [
            self.tokenizer.decode([token_id]) 
            for token_id in top_tokens.indices[0]
        ]
        
        # Choose best candidate based on edit distance
        best_candidate = min(candidates, 
                           key=lambda x: Levenshtein.distance(word.lower(), x.lower()))
        
        return best_candidate
    
    def _get_context_window(self, word: str, window_size: int = 5) -> str:
        # Placeholder for context window extraction
        return f"This is {word} in context"
    
    def normalize_text(self, text: str) -> str:
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize punctuation
        text = re.sub(r'[.,!?]+(?=[.,!?])', '', text)
        
        # Fix common OCR artifacts
        text = text.replace('|', 'I')
        text = text.replace('{', '(')
        text = text.replace('}', ')')
        
        return text.strip()

class DocumentTextProcessor:
    def __init__(self):
        self.post_processor = OCRPostProcessor()
        
    def process_document(self, ocr_output: Dict[str, Any]) -> Dict[str, Any]:
        text = ocr_output['text']
        confidence_scores = ocr_output['confidence_scores']
        
        # Post-process text
        corrected_text = self.post_processor.process_text(text, confidence_scores)
        normalized_text = self.post_processor.normalize_text(corrected_text)
        
        return {
            'original_text': text,
            'corrected_text': corrected_text,
            'normalized_text': normalized_text,
            'quality_metrics': self._calculate_quality_metrics(text, corrected_text)
        }
    
    def _calculate_quality_metrics(self, 
                                 original: str, 
                                 corrected: str) -> Dict[str, float]:
        return {
            'character_change_ratio': self._char_change_ratio(original, corrected),
            'word_change_ratio': self._word_change_ratio(original, corrected),
            'correction_confidence': self._calculate_correction_confidence(original, 
                                                                        corrected)
        }
    
    def _char_change_ratio(self, original: str, corrected: str) -> float:
        return Levenshtein.distance(original, corrected) / len(original)
    
    def _word_change_ratio(self, original: str, corrected: str) -> float:
        original_words = set(original.split())
        corrected_words = set(corrected.split())
        changes = len(original_words.symmetric_difference(corrected_words))
        return changes / len(original_words)
    
    def _calculate_correction_confidence(self, 
                                      original: str, 
                                      corrected: str) -> float:
        # Simple confidence calculation based on edit distance
        max_distance = max(len(original), len(corrected))
        distance = Levenshtein.distance(original, corrected)
        return 1 - (distance / max_distance)

# Example usage
processor = DocumentTextProcessor()
sample_output = {
    'text': 'Th1s ls a samp1e 0CR output w1th errors.',
    'confidence_scores': [0.8, 0.7, 0.9, 0.85, 0.6, 0.95, 0.75, 0.8]
}

result = processor.process_document(sample_output)
print("Processing Results:")
print(f"Original: {result['original_text']}")
print(f"Corrected: {result['corrected_text']}")
print(f"Quality Metrics: {result['quality_metrics']}")
```

Slide 11: YOLO-Based Document Structure Analysis

The YOLO-based document structure analyzer integrates region proposal with semantic segmentation to identify and classify document components while maintaining spatial relationships and hierarchical structure.

```python
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple
import numpy as np

class DocumentYOLOFeatureExtractor(nn.Module):
    def __init__(self, num_classes: int = 5):
        super(DocumentYOLOFeatureExtractor, self).__init__()
        
        # Use ResNet50 as backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Feature Pyramid Network
        self.fpn_layers = nn.ModuleDict({
            'p5': nn.Conv2d(2048, 256, 1),
            'p4': nn.Conv2d(1024, 256, 1),
            'p3': nn.Conv2d(512, 256, 1)
        })
        
        # Prediction heads
        self.pred_heads = nn.ModuleDict({
            'bbox': nn.Conv2d(256, 4, 3, padding=1),
            'conf': nn.Conv2d(256, 1, 3, padding=1),
            'class': nn.Conv2d(256, num_classes, 3, padding=1)
        })
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features through backbone
        c3 = self.backbone[:6](x)
        c4 = self.backbone[6:7](c3)
        c5 = self.backbone[7:](c4)
        
        # FPN forward pass
        p5 = self.fpn_layers['p5'](c5)
        p4 = self.fpn_layers['p4'](c4) + nn.functional.interpolate(p5, scale_factor=2)
        p3 = self.fpn_layers['p3'](c3) + nn.functional.interpolate(p4, scale_factor=2)
        
        # Generate predictions
        features = [p3, p4, p5]
        predictions = []
        
        for feature in features:
            pred = {
                'bbox': self.pred_heads['bbox'](feature),
                'conf': self.pred_heads['conf'](feature),
                'class': self.pred_heads['class'](feature)
            }
            predictions.append(pred)
            
        return predictions

class DocumentStructureAnalyzer:
    def __init__(self, conf_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.model = DocumentYOLOFeatureExtractor()
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
    def analyze_structure(self, image: torch.Tensor) -> Dict[str, List]:
        # Get predictions
        predictions = self.model(image)
        
        # Process predictions
        boxes, scores, classes = self._process_predictions(predictions)
        
        # Apply NMS
        kept_indices = self._non_max_suppression(boxes, scores)
        
        # Extract final predictions
        final_boxes = boxes[kept_indices]
        final_scores = scores[kept_indices]
        final_classes = classes[kept_indices]
        
        # Build hierarchical structure
        structure = self._build_hierarchy(final_boxes, final_classes)
        
        return {
            'boxes': final_boxes.tolist(),
            'scores': final_scores.tolist(),
            'classes': final_classes.tolist(),
            'hierarchy': structure
        }
    
    def _process_predictions(self, 
                           predictions: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, 
                                                                             torch.Tensor, 
                                                                             torch.Tensor]:
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for pred in predictions:
            # Convert predictions to bounding boxes
            boxes = self._decode_boxes(pred['bbox'])
            scores = torch.sigmoid(pred['conf']).squeeze(-1)
            classes = torch.argmax(pred['class'], dim=1)
            
            # Filter by confidence
            mask = scores > self.conf_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            classes = classes[mask]
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)
            
        return (torch.cat(all_boxes), 
                torch.cat(all_scores), 
                torch.cat(all_classes))
    
    def _decode_boxes(self, box_preds: torch.Tensor) -> torch.Tensor:
        # Convert center/width/height to corner coordinates
        xy_center = box_preds[:, :2]
        wh = box_preds[:, 2:]
        
        xy_min = xy_center - wh/2
        xy_max = xy_center + wh/2
        
        return torch.cat([xy_min, xy_max], dim=1)
    
    def _non_max_suppression(self, 
                           boxes: torch.Tensor, 
                           scores: torch.Tensor) -> torch.Tensor:
        return torch.ops.torchvision.nms(boxes, scores, self.nms_threshold)
    
    def _build_hierarchy(self, 
                        boxes: torch.Tensor, 
                        classes: torch.Tensor) -> Dict:
        structure = {'type': 'document', 'children': []}
        
        # Sort boxes by area (larger boxes first)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_indices = torch.argsort(areas, descending=True)
        
        boxes = boxes[sorted_indices]
        classes = classes[sorted_indices]
        
        # Build tree structure
        used_boxes = set()
        for i, (box, cls) in enumerate(zip(boxes, classes)):
            if i in used_boxes:
                continue
                
            node = {
                'type': self._get_class_name(cls.item()),
                'box': box.tolist(),
                'children': []
            }
            
            # Find children
            for j, (child_box, child_cls) in enumerate(zip(boxes[i+1:], classes[i+1:])):
                if self._is_contained(box, child_box):
                    node['children'].append({
                        'type': self._get_class_name(child_cls.item()),
                        'box': child_box.tolist()
                    })
                    used_boxes.add(i + j + 1)
                    
            structure['children'].append(node)
            
        return structure
    
    def _get_class_name(self, class_id: int) -> str:
        classes = ['text', 'title', 'table', 'figure', 'list']
        return classes[class_id]
    
    def _is_contained(self, parent_box: torch.Tensor, child_box: torch.Tensor) -> bool:
        return (child_box[0] >= parent_box[0] and
                child_box[1] >= parent_box[1] and
                child_box[2] <= parent_box[2] and
                child_box[3] <= parent_box[3])

# Example usage
analyzer = DocumentStructureAnalyzer()
sample_image = torch.randn(1, 3, 640, 640)
structure = analyzer.analyze_structure(sample_image)

print("Document Structure Analysis:")
print(f"Number of detected regions: {len(structure['boxes'])}")
print(f"Hierarchy: {structure['hierarchy']}")
```

Slide 12: Intelligent Document Error Recovery System

A robust error recovery system that combines multiple strategies to handle OCR errors, missing data, and structural anomalies in document processing through machine learning-based correction mechanisms.

```python
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import difflib
import numpy as np
from dataclasses import dataclass

@dataclass
class ErrorContext:
    error_type: str
    confidence: float
    original_content: str
    suggested_correction: Optional[str] = None
    context: Optional[Dict] = None

class ErrorRecoveryModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 256):
        super(ErrorRecoveryModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_size=512, 
                           num_layers=2, 
                           bidirectional=True,
                           dropout=0.3)
        
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8)
        self.fc = nn.Linear(1024, vocab_size)
        
    def forward(self, x: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embed input sequence
        embedded = self.embedding(x)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Apply attention if context is provided
        if context is not None:
            context_embedded = self.embedding(context)
            attended, _ = self.attention(lstm_out, context_embedded, context_embedded)
            lstm_out = lstm_out + attended
            
        # Project to vocabulary space
        output = self.fc(lstm_out)
        return output

class DocumentErrorRecovery:
    def __init__(self, vocab_size: int = 10000):
        self.error_model = ErrorRecoveryModel(vocab_size)
        self.correction_threshold = 0.85
        self.error_patterns = self._compile_error_patterns()
        
    def _compile_error_patterns(self) -> Dict[str, Dict]:
        return {
            'ocr_errors': {
                'patterns': [
                    (r'\b\d+[A-Za-z]\b', 'mixed_digits_letters'),
                    (r'[A-Z]{5,}', 'all_caps_long'),
                    (r'[^a-zA-Z0-9\s.,!?-]', 'special_chars')
                ],
                'corrections': self._load_correction_dictionary()
            },
            'structural_errors': {
                'patterns': [
                    ('missing_header', self._check_missing_header),
                    ('incomplete_table', self._check_incomplete_table),
                    ('misaligned_content', self._check_misalignment)
                ]
            }
        }
        
    def _load_correction_dictionary(self) -> Dict[str, str]:
        # Simulated correction dictionary
        return {
            'common_errors': {
                'teh': 'the',
                'recieved': 'received',
                'accomodation': 'accommodation'
            },
            'domain_specific': {
                'inv0ice': 'invoice',
                'sh1pment': 'shipment',
                'P0': 'PO'
            }
        }
        
    def process_document(self, 
                        document: Dict[str, Any]) -> Tuple[Dict[str, Any], 
                                                         List[ErrorContext]]:
        errors = []
        corrected_document = document.copy()
        
        # Check for OCR errors
        ocr_errors = self._detect_ocr_errors(document['content'])
        errors.extend(ocr_errors)
        
        # Check for structural errors
        structural_errors = self._detect_structural_errors(document)
        errors.extend(structural_errors)
        
        # Apply corrections
        if errors:
            corrected_document = self._apply_corrections(document, errors)
            
        return corrected_document, errors
    
    def _detect_ocr_errors(self, content: str) -> List[ErrorContext]:
        errors = []
        
        # Check against error patterns
        for pattern, error_type in self.error_patterns['ocr_errors']['patterns']:
            matches = re.finditer(pattern, content)
            for match in matches:
                error = ErrorContext(
                    error_type='ocr',
                    confidence=0.9,
                    original_content=match.group(),
                    context={'pattern': error_type}
                )
                errors.append(error)
                
        # Check against known corrections
        words = content.split()
        for word in words:
            lower_word = word.lower()
            if lower_word in self.error_patterns['ocr_errors']['corrections']['common_errors']:
                error = ErrorContext(
                    error_type='known_error',
                    confidence=1.0,
                    original_content=word,
                    suggested_correction=self.error_patterns['ocr_errors']
                                        ['corrections']['common_errors'][lower_word]
                )
                errors.append(error)
                
        return errors
    
    def _detect_structural_errors(self, document: Dict) -> List[ErrorContext]:
        errors = []
        
        # Check each structural error pattern
        for error_type, check_func in self.error_patterns['structural_errors']['patterns']:
            if error := check_func(document):
                errors.append(error)
                
        return errors
    
    def _check_missing_header(self, document: Dict) -> Optional[ErrorContext]:
        if 'header' not in document or not document['header']:
            return ErrorContext(
                error_type='structural',
                confidence=0.95,
                original_content='',
                context={'missing_element': 'header'}
            )
        return None
    
    def _check_incomplete_table(self, document: Dict) -> Optional[ErrorContext]:
        if 'tables' in document:
            for table in document['tables']:
                if self._is_table_incomplete(table):
                    return ErrorContext(
                        error_type='structural',
                        confidence=0.85,
                        original_content=str(table),
                        context={'table_id': table.get('id')}
                    )
        return None
    
    def _is_table_incomplete(self, table: Dict) -> bool:
        # Check for missing cells or inconsistent rows
        if not table.get('rows'):
            return True
            
        row_lengths = [len(row) for row in table['rows']]
        return len(set(row_lengths)) > 1
    
    def _check_misalignment(self, document: Dict) -> Optional[ErrorContext]:
        if 'layout' in document:
            # Check for misaligned elements
            elements = document['layout']['elements']
            for i, elem in enumerate(elements[:-1]):
                if self._check_alignment(elem, elements[i+1]):
                    return ErrorContext(
                        error_type='structural',
                        confidence=0.8,
                        original_content=str(elem),
                        context={'element_id': elem.get('id')}
                    )
        return None
    
    def _check_alignment(self, elem1: Dict, elem2: Dict) -> bool:
        # Check if elements are properly aligned
        threshold = 5  # pixels
        return abs(elem1['x'] - elem2['x']) > threshold
    
    def _apply_corrections(self, 
                          document: Dict, 
                          errors: List[ErrorContext]) -> Dict:
        corrected = document.copy()
        
        for error in errors:
            if error.error_type == 'ocr':
                corrected['content'] = self._fix_ocr_error(
                    corrected['content'], 
                    error
                )
            elif error.error_type == 'structural':
                corrected = self._fix_structural_error(corrected, error)
                
        return corrected
    
    def _fix_ocr_error(self, content: str, error: ErrorContext) -> str:
        if error.suggested_correction:
            return content.replace(error.original_content, 
                                 error.suggested_correction)
            
        # Use error recovery model for unknown errors
        correction = self._predict_correction(error)
        if correction:
            return content.replace(error.original_content, correction)
            
        return content
    
    def _predict_correction(self, error: ErrorContext) -> Optional[str]:
        # Convert to tensor and get model prediction
        input_tensor = self._prepare_input(error.original_content)
        with torch.no_grad():
            output = self.error_model(input_tensor)
            
        # Get most likely correction
        prediction = torch.argmax(output, dim=-1)
        correction = self._tensor_to_text(prediction)
        
        # Return correction if confidence is high enough
        confidence = torch.max(torch.softmax(output, dim=-1)).item()
        if confidence > self.correction_threshold:
            return correction
            
        return None

# Example usage
recovery_system = DocumentErrorRecovery()
sample_document = {
    'content': 'This is a samp1e document w1th some err0rs.',
    'header': None,
    'tables': [
        {
            'id': 1,
            'rows': [
                ['A', 'B', 'C'],
                ['D', 'E']  # Incomplete row
            ]
        }
    ]
}

corrected_document, errors = recovery_system.process_document(sample_document)
print("Found Errors:")
for error in errors:
    print(f"Type: {error.error_type}")
    print(f"Confidence: {error.confidence}")
    print(f"Original: {error.original_content}")
    print(f"Suggested Correction: {error.suggested_correction}")
    print("---")
```

Slide 13: Real-Time Document Stream Processing System

An advanced system for processing continuous streams of document data, implementing parallel processing pipelines and intelligent queue management to handle high-throughput document analysis.

```python
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import time
import uuid
from collections import deque
import numpy as np

@dataclass
class DocumentChunk:
    id: str
    content: bytes
    metadata: Dict[str, Any]
    timestamp: float
    priority: int = 0

class DocumentStreamProcessor:
    def __init__(self, max_workers: int = 4, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.max_workers = max_workers
        self.input_queue = asyncio.Queue(maxsize=buffer_size)
        self.processing_queue = deque(maxlen=buffer_size)
        self.output_queue = asyncio.Queue(maxsize=buffer_size)
        self.workers = []
        self.running = False
        self.stats = {
            'processed_documents': 0,
            'errors': 0,
            'avg_processing_time': 0
        }
        
    async def start(self):
        self.running = True
        self.workers = [
            asyncio.create_task(self._process_worker(i))
            for i in range(self.max_workers)
        ]
        
        # Start monitoring task
        asyncio.create_task(self._monitor_performance())
        
    async def stop(self):
        self.running = False
        await self._drain_queues()
        for worker in self.workers:
            worker.cancel()
            
    async def ingest_document(self, document: DocumentChunk):
        if self.running:
            await self.input_queue.put(document)
            
    async def get_processed_document(self) -> Optional[Dict]:
        if not self.output_queue.empty():
            return await self.output_queue.get()
        return None
        
    async def _process_worker(self, worker_id: int):
        while self.running:
            try:
                # Get document from input queue
                document = await self.input_queue.get()
                
                # Process document
                start_time = time.time()
                processed_doc = await self._process_document(document)
                processing_time = time.time() - start_time
                
                # Update statistics
                self._update_stats(processing_time)
                
                # Put processed document in output queue
                await self.output_queue.put(processed_doc)
                
                self.input_queue.task_done()
                
            except Exception as e:
                self.stats['errors'] += 1
                print(f"Worker {worker_id} error: {str(e)}")
                
    async def _process_document(self, document: DocumentChunk) -> Dict:
        # Simulate document processing stages
        processed = {}
        
        # 1. Document preprocessing
        processed['content'] = await self._preprocess_chunk(document.content)
        
        # 2. Feature extraction
        features = await self._extract_features(processed['content'])
        processed['features'] = features
        
        # 3. Analysis and classification
        analysis = await self._analyze_document(features)
        processed['analysis'] = analysis
        
        # 4. Metadata enrichment
        processed['metadata'] = await self._enrich_metadata(
            document.metadata,
            analysis
        )
        
        return processed
        
    async def _preprocess_chunk(self, content: bytes) -> bytes:
        # Simulate preprocessing operations
        await asyncio.sleep(0.01)
        return content
        
    async def _extract_features(self, content: bytes) -> Dict:
        # Simulate feature extraction
        await asyncio.sleep(0.02)
        return {
            'length': len(content),
            'type': 'text/plain',
            'features': np.random.rand(128).tolist()
        }
        
    async def _analyze_document(self, features: Dict) -> Dict:
        # Simulate document analysis
        await asyncio.sleep(0.03)
        return {
            'classification': np.random.choice(['invoice', 'report', 'contract']),
            'confidence': np.random.uniform(0.8, 1.0),
            'topics': ['topic1', 'topic2']
        }
        
    async def _enrich_metadata(self, 
                             original_metadata: Dict, 
                             analysis: Dict) -> Dict:
        # Combine original metadata with analysis results
        return {
            **original_metadata,
            'processed_timestamp': time.time(),
            'analysis_results': analysis
        }
        
    async def _monitor_performance(self):
        while self.running:
            await asyncio.sleep(5)  # Monitor every 5 seconds
            print(f"Performance Statistics:")
            print(f"Processed Documents: {self.stats['processed_documents']}")
            print(f"Average Processing Time: {self.stats['avg_processing_time']:.3f}s")
            print(f"Errors: {self.stats['errors']}")
            print(f"Queue Sizes - Input: {self.input_queue.qsize()}, "
                  f"Output: {self.output_queue.qsize()}")
            
    def _update_stats(self, processing_time: float):
        self.stats['processed_documents'] += 1
        # Update moving average
        alpha = 0.1
        self.stats['avg_processing_time'] = (
            alpha * processing_time +
            (1 - alpha) * self.stats['avg_processing_time']
        )
        
    async def _drain_queues(self):
        # Process remaining documents in queues
        while not self.input_queue.empty():
            document = await self.input_queue.get()
            processed = await self._process_document(document)
            await self.output_queue.put(processed)
            self.input_queue.task_done()

# Example usage
async def main():
    processor = DocumentStreamProcessor(max_workers=4)
    await processor.start()
    
    # Simulate document stream
    for i in range(10):
        document = DocumentChunk(
            id=str(uuid.uuid4()),
            content=f"Document {i} content".encode(),
            metadata={'source': 'test', 'index': i},
            timestamp=time.time(),
            priority=1
        )
        await processor.ingest_document(document)
        
        # Process output
        if processed_doc := await processor.get_processed_document():
            print(f"Processed document {i}:")
            print(f"Analysis: {processed_doc['analysis']}")
            print("---")
            
    await asyncio.sleep(2)  # Allow processing to complete
    await processor.stop()

# Run the example
asyncio.run(main())
```

Slide 14: Document Entity Relationship Extraction

A sophisticated system that identifies and maps relationships between different entities within documents using advanced NLP techniques and graph-based relationship modeling.

```python
import spacy
import networkx as nx
from typing import List, Dict, Tuple, Set
import numpy as np
from dataclasses import dataclass

@dataclass
class DocumentEntity:
    text: str
    label: str
    start: int
    end: int
    confidence: float

@dataclass
class EntityRelation:
    source: DocumentEntity
    target: DocumentEntity
    relation_type: str
    confidence: float

class EntityRelationshipExtractor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.relation_patterns = self._load_relation_patterns()
        self.entity_graph = nx.DiGraph()
        
    def _load_relation_patterns(self) -> Dict[str, List[Dict]]:
        return {
            'organizational': [
                {'pattern': [{'POS': 'PROPN'}, {'LOWER': 'works'}, {'LOWER': 'for'}]},
                {'pattern': [{'POS': 'PROPN'}, {'LOWER': 'is'}, {'LOWER': 'part'}, {'LOWER': 'of'}]}
            ],
            'temporal': [
                {'pattern': [{'POS': 'PROPN'}, {'LOWER': 'before'}]},
                {'pattern': [{'POS': 'PROPN'}, {'LOWER': 'after'}]}
            ],
            'causal': [
                {'pattern': [{'LOWER': 'because'}, {'LOWER': 'of'}]},
                {'pattern': [{'LOWER': 'leads'}, {'LOWER': 'to'}]}
            ]
        }
        
    def extract_relationships(self, document: str) -> Tuple[List[DocumentEntity], 
                                                          List[EntityRelation]]:
        # Process document with spaCy
        doc = self.nlp(document)
        
        # Extract entities
        entities = self._extract_entities(doc)
        
        # Extract relationships
        relationships = self._extract_relations(doc, entities)
        
        # Build entity graph
        self._build_entity_graph(entities, relationships)
        
        return entities, relationships
    
    def _extract_entities(self, doc) -> List[DocumentEntity]:
        entities = []
        for ent in doc.ents:
            entity = DocumentEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=self._calculate_entity_confidence(ent)
            )
            entities.append(entity)
        return entities
    
    def _calculate_entity_confidence(self, entity) -> float:
        # Calculate confidence based on various factors
        confidence = 0.5
        
        # Adjust based on entity length
        if len(entity.text.split()) > 1:
            confidence += 0.1
            
        # Adjust based on capitalization
        if entity.text[0].isupper():
            confidence += 0.1
            
        # Adjust based on entity label
        common_labels = {'PERSON', 'ORG', 'DATE', 'GPE'}
        if entity.label_ in common_labels:
            confidence += 0.2
            
        return min(confidence, 1.0)
    
    def _extract_relations(self, 
                          doc, 
                          entities: List[DocumentEntity]) -> List[EntityRelation]:
        relations = []
        entity_spans = {(e.start, e.end): e for e in entities}
        
        for sent in doc.sents:
            # Extract relations within each sentence
            sent_relations = self._extract_sentence_relations(sent, entity_spans)
            relations.extend(sent_relations)
            
        return relations
    
    def _extract_sentence_relations(self, 
                                  sent, 
                                  entity_spans: Dict) -> List[EntityRelation]:
        relations = []
        
        # Get entity pairs in sentence
        entity_pairs = self._get_entity_pairs(sent, entity_spans)
        
        for source, target in entity_pairs:
            # Find relation type between entities
            relation_type = self._identify_relation(sent, source, target)
            if relation_type:
                relation = EntityRelation(
                    source=entity_spans[(source.start_char, source.end_char)],
                    target=entity_spans[(target.start_char, target.end_char)],
                    relation_type=relation_type,
                    confidence=self._calculate_relation_confidence(sent, source, target)
                )
                relations.append(relation)
                
        return relations
    
    def _get_entity_pairs(self, sent, entity_spans: Dict) -> List[Tuple]:
        sent_entities = []
        for start, end in entity_spans:
            if sent.start_char <= start and end <= sent.end_char:
                span = sent.doc[sent.start:sent.end].char_span(
                    start - sent.start_char,
                    end - sent.start_char
                )
                if span:
                    sent_entities.append(span)
                    
        # Generate all possible pairs
        return [(e1, e2) for i, e1 in enumerate(sent_entities)
                for e2 in sent_entities[i+1:]]
    
    def _identify_relation(self, sent, source, target) -> str:
        # Check each relation pattern
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                # Create matcher for pattern
                if self._matches_pattern(sent, source, target, pattern):
                    return relation_type
        return None
    
    def _matches_pattern(self, sent, source, target, pattern: Dict) -> bool:
        # Simple pattern matching
        text = sent.text.lower()
        return all(token['LOWER'] in text for token in pattern['pattern'])
    
    def _calculate_relation_confidence(self, sent, source, target) -> float:
        # Base confidence
        confidence = 0.5
        
        # Adjust based on distance between entities
        token_distance = abs(source.start - target.start)
        if token_distance < 5:
            confidence += 0.2
        elif token_distance < 10:
            confidence += 0.1
            
        # Adjust based on dependency path
        if self._has_direct_dependency_path(source.root, target.root):
            confidence += 0.2
            
        return min(confidence, 1.0)
    
    def _has_direct_dependency_path(self, source_token, target_token) -> bool:
        # Check for direct dependency path between tokens
        return any(token in target_token.ancestors for token in source_token.ancestors)
    
    def _build_entity_graph(self, 
                           entities: List[DocumentEntity],
                           relations: List[EntityRelation]):
        # Clear existing graph
        self.entity_graph.clear()
        
        # Add entities as nodes
        for entity in entities:
            self.entity_graph.add_node(entity.text, 
                                     label=entity.label,
                                     confidence=entity.confidence)
            
        # Add relations as edges
        for relation in relations:
            self.entity_graph.add_edge(relation.source.text,
                                     relation.target.text,
                                     type=relation.relation_type,
                                     confidence=relation.confidence)
    
    def get_entity_relationships(self, entity_text: str) -> Dict[str, List[Dict]]:
        if entity_text not in self.entity_graph:
            return {}
            
        relationships = {
            'outgoing': [],
            'incoming': []
        }
        
        # Get outgoing relationships
        for _, target, data in self.entity_graph.edges(entity_text, data=True):
            relationships['outgoing'].append({
                'target': target,
                'relation': data['type'],
                'confidence': data['confidence']
            })
            
        # Get incoming relationships
        for source, _, data in self.entity_graph.in_edges(entity_text, data=True):
            relationships['incoming'].append({
                'source': source,
                'relation': data['type'],
                'confidence': data['confidence']
            })
            
        return relationships

# Example usage
extractor = EntityRelationshipExtractor()
sample_text = """
John Smith works for Acme Corporation. The company is part of Global Industries.
Because of recent changes, Sarah Johnson leads the development team after
the reorganization.
"""

entities, relations = extractor.extract_relationships(sample_text)

print("Extracted Entities:")
for entity in entities:
    print(f"{entity.text} ({entity.label}): {entity.confidence:.2f}")

print("\nExtracted Relations:")
for relation in relations:
    print(f"{relation.source.text} -> {relation.relation_type} -> {relation.target.text}")

print("\nRelationships for 'John Smith':")
relationships = extractor.get_entity_relationships("John Smith")
print(relationships)
```

Slide 15: Additional Resources

*   "YOLOv4: Optimal Speed and Accuracy of Object Detection" [https://arxiv.org/abs/2004.10934](https://arxiv.org/abs/2004.10934)
*   "LayoutLM: Pre-training of Text and Layout for Document Image Understanding" [https://arxiv.org/abs/1912.13318](https://arxiv.org/abs/1912.13318)
*   "DocFormer: End-to-End Transformer for Document Understanding" [https://arxiv.org/abs/2106.11539](https://arxiv.org/abs/2106.11539)
*   "SPADE: Spatial Dependency Parsing for Semi-Structured Document Information Extraction" [https://arxiv.org/abs/2005.00642](https://arxiv.org/abs/2005.00642)
*   "Graph-based Neural Networks for Document Layout Analysis" [https://arxiv.org/abs/2104.12837](https://arxiv.org/abs/2104.12837)
*   "Neural Networks for Information Extraction from Visually Rich Documents" [https://arxiv.org/abs/2102.11838](https://arxiv.org/abs/2102.11838)
*   "Transformers for Document Image Understanding: A Survey" [https://arxiv.org/abs/2203.15935](https://arxiv.org/abs/2203.15935)
*   "Document Intelligence: Past, Present and Future" [https://arxiv.org/abs/2210.12249](https://arxiv.org/abs/2210.12249)
*   "End-to-End Document Image Recognition: A Survey" [https://arxiv.org/abs/2210.12279](https://arxiv.org/abs/2210.12279)

