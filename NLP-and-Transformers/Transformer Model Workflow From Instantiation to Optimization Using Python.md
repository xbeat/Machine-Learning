## Transformer Model Workflow From Instantiation to Optimization Using Python
Slide 1: Model Instantiation

Understanding how to instantiate a Transformer model is the first step in working with these powerful architectures. We'll use the Hugging Face Transformers library to load a pre-trained BERT model.

```python
from transformers import BertModel, BertConfig

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

# Or create a model with a custom configuration
config = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12)
custom_model = BertModel(config)
```

Slide 2: Tokenization

Tokenization is crucial for preparing text data for input into Transformer models. We'll use BERT's tokenizer to convert text into token IDs.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "Hello, how are you?"
encoded = tokenizer(text, return_tensors='pt')
print(encoded)
```

Slide 3: Forward Pass

After tokenization, we can perform a forward pass through the model to obtain contextual embeddings.

```python
import torch

# Assuming 'model' and 'encoded' from previous slides
with torch.no_grad():
    outputs = model(**encoded)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)
```

Slide 4: Fine-tuning Setup

Fine-tuning adapts a pre-trained model to a specific task. We'll set up a classification head on top of BERT for sentiment analysis.

```python
from transformers import BertForSequenceClassification

num_labels = 2  # binary classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Prepare dataset (simplified example)
texts = ["I love this movie!", "This film was terrible."]
labels = torch.tensor([1, 0])  # 1 for positive, 0 for negative
encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
```

Slide 5: Training Loop

Implementing a basic training loop is essential for fine-tuning. We'll use PyTorch's optimizer and loss function.

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(**encoded, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

Slide 6: Evaluation

After fine-tuning, it's crucial to evaluate the model's performance on a separate test set.

```python
from sklearn.metrics import accuracy_score, classification_report

model.eval()
test_texts = ["A great film!", "I didn't enjoy it."]
test_labels = [1, 0]

with torch.no_grad():
    inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")
print(classification_report(test_labels, predictions))
```

Slide 7: Optimization Techniques

Various optimization techniques can improve training efficiency and model performance. We'll explore gradient accumulation and mixed precision training.

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
accumulation_steps = 4

for epoch in range(3):
    for i, batch in enumerate(dataloader):
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

Slide 8: Interpretability - Attention Visualization

Visualizing attention weights can provide insights into how the model processes input. We'll create a simple attention heatmap.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs, output_attentions=True)
    
    attn_weights = outputs.attentions[-1].squeeze().mean(dim=0)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights.detach(), xticklabels=tokens, yticklabels=tokens)
    plt.title("Attention Weights Heatmap")
    plt.show()

visualize_attention(model, tokenizer, "The cat sat on the mat.")
```

Slide 9: Model Pruning

Pruning reduces model size by removing less important weights. We'll implement a simple magnitude-based pruning technique.

```python
def prune_model(model, pruning_threshold):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight_mask = (module.weight.abs() > pruning_threshold).float()
            module.weight.data *= weight_mask

pruning_threshold = 0.1
prune_model(model, pruning_threshold)
```

Slide 10: Model Compression - Quantization

Quantization reduces model size and inference time by representing weights with fewer bits.

```python
import torch.quantization

def quantize_model(model):
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    return model

quantized_model = quantize_model(model.cpu())  # Quantization requires CPU
```

Slide 11: Knowledge Distillation

Knowledge distillation transfers knowledge from a larger model (teacher) to a smaller model (student).

```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0):
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
    distillation = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
    
    ce_loss = F.cross_entropy(student_logits, labels)
    return ce_loss + distillation

# Training loop (simplified)
for batch in dataloader:
    student_outputs = student_model(**batch)
    with torch.no_grad():
        teacher_outputs = teacher_model(**batch)
    
    loss = distillation_loss(student_outputs.logits, teacher_outputs.logits, batch['labels'])
    loss.backward()
    optimizer.step()
```

Slide 12: Model Analysis - Probe Tasks

Probe tasks help understand what linguistic knowledge is captured in different layers of the model.

```python
class ProbeClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

def train_probe(model, tokenizer, texts, labels, layer_num):
    model.eval()
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encodings, output_hidden_states=True)
    
    layer_output = outputs.hidden_states[layer_num][:, 0, :]  # CLS token
    probe = ProbeClassifier(layer_output.shape[1], num_classes=len(set(labels)))
    
    # Train probe (simplified)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters())
    
    for epoch in range(10):
        logits = probe(layer_output)
        loss = criterion(logits, torch.tensor(labels))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return probe

probe = train_probe(model, tokenizer, texts, labels, layer_num=6)
```

Slide 13: Deployment and Inference Optimization

Optimizing models for deployment involves techniques like model quantization and TorchScript conversion for faster inference.

```python
import torch

def optimize_for_inference(model):
    # Convert to TorchScript
    example_input = tokenizer("Example input", return_tensors='pt')
    traced_model = torch.jit.trace(model, example_input['input_ids'])
    
    # Quantize (assuming CPU deployment)
    quantized_model = torch.quantization.quantize_dynamic(
        traced_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    return quantized_model

optimized_model = optimize_for_inference(model)

# Inference
text = "New input for inference"
inputs = tokenizer(text, return_tensors='pt')
with torch.no_grad():
    outputs = optimized_model(inputs['input_ids'])
```

Slide 14: Additional Resources

For further exploration of Transformer models and advanced techniques:

1. "Attention Is All You Need" - Original Transformer paper ([https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762))
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" ([https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805))
3. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" ([https://arxiv.org/abs/1803.03635](https://arxiv.org/abs/1803.03635))
4. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter" ([https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108))
5. "Probing Neural Network Comprehension of Natural Language Arguments" ([https://arxiv.org/abs/1907.07355](https://arxiv.org/abs/1907.07355))

