## Fine-Tuning LLMs with LoRA in Python
Slide 1: Fine-Tuning LLMs using LoRA

Fine-tuning Large Language Models (LLMs) is a crucial technique for adapting pre-trained models to specific tasks or domains. Low-Rank Adaptation (LoRA) is an efficient method that significantly reduces the number of trainable parameters while maintaining performance. This approach is particularly useful for those with limited computational resources.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(model, peft_config)
print(f"Trainable parameters: {model.print_trainable_parameters()}")
```

Slide 2: Understanding LoRA

LoRA works by adding pairs of rank decomposition matrices to the weights of the original model. These low-rank matrices are trained on the target task, allowing the model to adapt without changing all parameters. This results in a significant reduction in the number of trainable parameters and memory usage.

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = 1 / torch.sqrt(torch.tensor(rank).float())

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling

# Example usage
lora = LoRALayer(768, 768, rank=8)
input_tensor = torch.randn(1, 768)
output = lora(input_tensor)
print(f"Output shape: {output.shape}")
```

Slide 3: Preparing Data for Fine-Tuning

Before fine-tuning, we need to prepare our dataset. This involves tokenizing the text, creating attention masks, and formatting the data for training. Here's an example of how to prepare a simple dataset for fine-tuning:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load a sample dataset
dataset = load_dataset("imdb", split="train[:1000]")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Prepare the dataset for training
train_dataset = tokenized_dataset.remove_columns(["text", "label"])
train_dataset = train_dataset.rename_column("attention_mask", "attention_mask")
train_dataset.set_format("torch")

print(f"Number of samples: {len(train_dataset)}")
print(f"Sample features: {train_dataset[0].keys()}")
```

Slide 4: Configuring LoRA for Fine-Tuning

To use LoRA for fine-tuning, we need to configure which layers of the model to adapt. Typically, we apply LoRA to the attention layers of the transformer model. Here's how to set up LoRA configuration for a GPT-2 model:

```python
from peft import LoraConfig, get_peft_model, TaskType

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"]
)

model = AutoModelForCausalLM.from_pretrained("gpt2")
peft_model = get_peft_model(model, peft_config)

print(f"Trainable parameters: {peft_model.print_trainable_parameters()}")
```

Slide 5: Training Loop Setup

Now that we have our model and data prepared, we can set up the training loop. This involves creating a training function, setting up an optimizer, and defining the training parameters.

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
)

# Start training
trainer.train()
```

Slide 6: Monitoring Training Progress

During training, it's important to monitor the progress to ensure the model is learning effectively. We can use TensorBoard to visualize the training metrics:

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter(log_dir="./logs")

# Simulating training progress
for epoch in range(3):
    for step in range(100):
        loss = np.random.rand()
        writer.add_scalar("Loss/train", loss, epoch * 100 + step)

    # Add other metrics as needed
    writer.add_scalar("Accuracy/train", np.random.rand(), epoch)

writer.close()

print("Training visualization data has been logged. Use TensorBoard to view.")
```

Slide 7: Saving and Loading LoRA Weights

After fine-tuning, it's crucial to save the LoRA weights separately from the base model. This allows for easy adaptation and sharing of the fine-tuned model without distributing the entire model weights.

```python
from peft import PeftModel, PeftConfig

# Save LoRA weights
peft_model_path = "./peft_model"
peft_model.save_pretrained(peft_model_path)

# Load LoRA weights
config = PeftConfig.from_pretrained(peft_model_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
peft_model = PeftModel.from_pretrained(model, peft_model_path)

print(f"LoRA weights loaded successfully. Trainable parameters: {peft_model.print_trainable_parameters()}")
```

Slide 8: Inference with Fine-Tuned Model

Once the model is fine-tuned, we can use it for inference. Here's how to generate text using the adapted model:

```python
from transformers import AutoTokenizer, TextGenerationPipeline

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
pipeline = TextGenerationPipeline(model=peft_model, tokenizer=tokenizer)

prompt = "In this essay, I will argue that"
generated_text = pipeline(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

print(f"Generated text:\n{generated_text}")
```

Slide 9: Evaluating the Fine-Tuned Model

To assess the performance of our fine-tuned model, we need to evaluate it on a test set. Here's an example of how to calculate perplexity, a common metric for language models:

```python
import torch
from tqdm import tqdm

def calculate_perplexity(model, tokenizer, test_data):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(test_data):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

test_data = ["This is a test sentence.", "Another example for evaluation."]
perplexity = calculate_perplexity(peft_model, tokenizer, test_data)
print(f"Model perplexity: {perplexity:.2f}")
```

Slide 10: Real-Life Example: Sentiment Analysis

Let's fine-tune a model for sentiment analysis using LoRA. We'll use a small dataset of movie reviews:

```python
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

# Load dataset
dataset = load_dataset("imdb", split="train[:1000]")

# Prepare model and tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA
peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Train
training_args = TrainingArguments(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=8)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
trainer.train()

print("Fine-tuning completed.")
```

Slide 11: Real-Life Example: Text Generation for Recipe Creation

In this example, we'll fine-tune a model to generate cooking recipes using LoRA:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType

# Prepare model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Configure LoRA
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)

# Sample recipe data (in practice, you'd use a larger dataset)
recipes = [
    "To make pancakes, mix flour, eggs, milk, and sugar. Cook on a griddle until golden brown.",
    "For a quick pasta dish, boil spaghetti and toss with olive oil, garlic, and parmesan cheese."
]

# Tokenize and format data
def prepare_data(texts):
    return tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

inputs = prepare_data(recipes)

# Train (simplified for demonstration)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(10):
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print("Fine-tuning completed. The model can now generate recipe-like text.")
```

Slide 12: Hyperparameter Tuning for LoRA

Fine-tuning the hyperparameters of LoRA can significantly impact the model's performance. Here's an example of how to perform a simple grid search for LoRA hyperparameters:

```python
from sklearn.model_selection import ParameterGrid
import numpy as np

def evaluate_model(model, eval_data):
    # Placeholder for evaluation metric (e.g., perplexity)
    return np.random.rand()

param_grid = {
    'r': [4, 8, 16],
    'lora_alpha': [16, 32, 64],
    'lora_dropout': [0.05, 0.1, 0.2]
}

best_score = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=params['r'],
        lora_alpha=params['lora_alpha'],
        lora_dropout=params['lora_dropout']
    )
    
    model = get_peft_model(base_model, peft_config)
    # Train model here
    
    score = evaluate_model(model, eval_data)
    
    if score < best_score:
        best_score = score
        best_params = params

print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
```

Slide 13: Comparing LoRA to Full Fine-Tuning

To understand the benefits of LoRA, it's useful to compare it with full fine-tuning in terms of performance and efficiency:

```python
import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType

model_name = "gpt2"
base_model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# LoRA fine-tuning
start_time = time.time()
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1)
lora_model = get_peft_model(base_model, peft_config)
lora_trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
lora_time = time.time() - start_time

# Full fine-tuning
start_time = time.time()
full_model = GPT2LMHeadModel.from_pretrained(model_name)
full_trainable_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
full_time = time.time() - start_time

print(f"LoRA trainable parameters: {lora_trainable_params}")
print(f"Full fine-tuning trainable parameters: {full_trainable_params}")
print(f"LoRA setup time: {lora_time:.2f} seconds")
print(f"Full fine-tuning setup time: {full_time:.2f} seconds")
```

Slide 14: Additional Resources

For those interested in diving deeper into LoRA and fine-tuning LLMs, here are some valuable resources:

1. LoRA: Low-Rank Adaptation of Large Language Models (arXiv:2106.09685) URL: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
2. PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods URL: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
3. Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning (arXiv:2303.15647) URL: [https://arxiv.org/abs/2303.15647](https://arxiv.org/abs/2303.15647)
4. The HuggingFace documentation on fine-tuning transformer models URL: [https://huggingface.co/docs/transformers/training](https://huggingface.co/docs/transformers/training)

These resources provide in-depth explanations, implementations, and best practices for using LoRA and other efficient fine-tuning techniques for large language models.

