## Fundamental Topics of LLM Fine-tuning
Slide 1: Introduction to LLM Fine-tuning

Large Language Models (LLMs) have revolutionized natural language processing, but they often require fine-tuning for specific tasks. Fine-tuning adapts a pre-trained model to perform better on targeted applications. This process involves updating the model's parameters using task-specific data.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pre-trained LLM
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare your dataset
train_data = [
    "Fine-tuning adapts LLMs to specific tasks.",
    "The process involves updating model parameters.",
    "Task-specific data is crucial for effective fine-tuning."
]

# Tokenize the data
encoded_data = tokenizer(train_data, padding=True, truncation=True, return_tensors="pt")

# Fine-tune the model (simplified example)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for input_ids, attention_mask in zip(encoded_data["input_ids"], encoded_data["attention_mask"]):
        outputs = model(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), labels=input_ids.unsqueeze(0))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print("Fine-tuning complete!")
```

Slide 2: Reinforcement Learning from Human Feedback (RLHF)

RLHF is a powerful technique for fine-tuning LLMs using human preferences. It involves training a reward model based on human feedback and then using reinforcement learning to optimize the LLM's behavior. This approach helps align the model's outputs with human values and preferences.

```python
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Simulated human feedback function
def get_human_feedback(generated_text):
    # In a real scenario, this would involve actual human evaluation
    return np.random.rand()  # Simulated score between 0 and 1

# RLHF training loop (simplified)
for epoch in range(10):
    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Get human feedback
    reward = get_human_feedback(generated_text)
    
    # Update model (this is a simplified version, actual RLHF is more complex)
    loss = -reward * model(input_ids, labels=output).loss
    loss.backward()
    
    print(f"Epoch {epoch + 1}, Reward: {reward:.4f}")

print("RLHF training complete!")
```

Slide 3: Challenges of Fine-tuning an LLM

Fine-tuning LLMs presents several challenges, including computational requirements, overfitting, and catastrophic forgetting. Large models demand significant computational resources, while overfitting can occur when the model becomes too specialized on the fine-tuning data. Catastrophic forgetting happens when the model loses its general knowledge while adapting to a specific task.

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Prepare dataset (example using a text file)
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="path_to_your_data.txt",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Fine-tune the model
trainer.train()

# Evaluate on a test prompt
test_prompt = "The challenge of fine-tuning is"
input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"Generated text: {generated_text}")

# Note: This example doesn't directly address overfitting or catastrophic forgetting,
# but it demonstrates the basic setup for fine-tuning, which can lead to these issues.
```

Slide 4: Fundamentals of Low-Rank Adaptation (LoRA)

Low-Rank Adaptation (LoRA) is an efficient fine-tuning technique that reduces the number of trainable parameters. LoRA works by adding low-rank decomposition matrices to the weights of the transformer layers. This approach significantly decreases the computational and memory requirements while maintaining performance.

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = 0.01

    def forward(self, x):
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

# Load a pre-trained model
model = AutoModel.from_pretrained("bert-base-uncased")

# Apply LoRA to the query and value matrices of self-attention layers
for layer in model.encoder.layer:
    attention = layer.attention.self
    
    # Create LoRA layers
    attention.query_lora = LoRALayer(768, 768)
    attention.value_lora = LoRALayer(768, 768)
    
    # Modify forward pass
    original_forward = attention.forward
    def lora_forward(self, *args, **kwargs):
        output = original_forward(*args, **kwargs)
        hidden_states = output[0]
        hidden_states += self.query_lora(hidden_states)
        hidden_states += self.value_lora(hidden_states)
        return (hidden_states,) + output[1:]
    
    attention.forward = lora_forward.__get__(attention)

print("LoRA applied to the model")

# Now only the LoRA parameters need to be trained
trainable_params = [p for n, p in model.named_parameters() if "lora" in n]
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

# Training loop would go here
```

Slide 5: Guiding Principles for Setting Hyperparameters during Fine-tuning

Setting hyperparameters is crucial for successful fine-tuning. Key principles include starting with a lower learning rate than used in pre-training, using a smaller batch size to prevent overfitting, and employing techniques like learning rate scheduling and early stopping. It's also important to monitor validation performance and adjust hyperparameters accordingly.

```python
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load and preprocess dataset
dataset = load_dataset("glue", "mrpc")
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments with carefully chosen hyperparameters
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,  # Lower learning rate for fine-tuning
    per_device_train_batch_size=16,  # Smaller batch size
    num_train_epochs=3,
    weight_decay=0.01,  # L2 regularization to prevent overfitting
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",  # Evaluate after each epoch
    save_strategy="epoch",  # Save after each epoch
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train the model
trainer.train()

print("Training complete. Best model saved.")
```

Slide 6: Hyperparameters Specific to LoRA-powered LLM Fine-tuning

When fine-tuning LLMs using LoRA, specific hyperparameters need attention. These include the rank of the low-rank matrices, the scaling factor, and which layers to apply LoRA to. The rank determines the expressiveness of the adaptation, while the scaling factor controls the magnitude of the LoRA updates.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

# Load base model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define LoRA configuration
lora_config = LoraConfig(
    r=8,  # rank of the update matrices
    lora_alpha=32,  # scaling factor
    target_modules=["c_attn"],  # layers to apply LoRA to
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora_gpt2",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=True,
)

# Initialize Trainer (assuming you have prepared your dataset)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Your prepared dataset
    eval_dataset=eval_dataset,  # Your evaluation dataset
)

# Start training
trainer.train()

print("LoRA fine-tuning complete!")
```

Slide 7: Preparing a Regular Instruction Fine-tuning Dataset

Creating a high-quality dataset is crucial for instruction fine-tuning. The dataset should consist of input-output pairs that represent the desired behavior of the model. It's important to ensure diversity in the instructions and to cover a wide range of tasks relevant to your use case.

```python
import json
from datasets import Dataset

# Sample instruction-response pairs
instruction_data = [
    {"instruction": "Summarize the following text:", "input": "The Industrial Revolution was a period of major industrialization and innovation during the late 18th and early 19th centuries. The Industrial Revolution began in Great Britain and quickly spread throughout Western Europe and North America.", "output": "The Industrial Revolution was a time of significant technological and economic change in the late 18th and early 19th centuries, starting in Britain and spreading to Western Europe and North America."},
    {"instruction": "Translate the following English text to French:", "input": "Hello, how are you?", "output": "Bonjour, comment allez-vous ?"},
    {"instruction": "Generate a short story based on the following prompt:", "input": "A mysterious package arrives at your doorstep.", "output": "As Sarah opened her front door, she was surprised to find a small, unmarked package sitting on her welcome mat. Curiosity piqued, she picked it up, noticing its unexpected weight. With trembling hands, she carefully unwrapped the brown paper, revealing a ornate wooden box. Inside, nestled on velvet, lay an antique key with a note that read: 'Your adventure begins now.'"}
]

# Convert to Dataset
dataset = Dataset.from_pandas(pd.DataFrame(instruction_data))

# Save dataset to disk
dataset.save_to_disk("instruction_dataset")

# Load dataset (for demonstration)
loaded_dataset = Dataset.load_from_disk("instruction_dataset")

# Print a sample
print(loaded_dataset[0])

# Tokenize the dataset (assuming you have a tokenizer)
def tokenize_function(examples):
    prompt = f"Instruction: {examples['instruction']}\nInput: {examples['input']}\nOutput: "
    target = examples['output']
    
    tokenized_prompt = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized_target = tokenizer(target, truncation=True, padding="max_length", max_length=512)
    
    tokenized_prompt["labels"] = tokenized_target["input_ids"]
    
    return tokenized_prompt

tokenized_dataset = loaded_dataset.map(tokenize_function, batched=True)

print("Dataset prepared and tokenized for instruction fine-tuning!")
```

Slide 8: Preparing a DPO (Direct Preference Optimization) Dataset

DPO is an alternative to RLHF that directly optimizes the model based on human preferences. Preparing a DPO dataset involves collecting pairs of model outputs along with human preferences between them. This approach can be more efficient than RLHF as it doesn't require training a separate reward model.

```python
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Sample DPO data
dpo_data = [
    {
        "prompt": "Write a tweet about climate change:",
        "chosen": "Climate change is a global challenge that requires immediate action. We must reduce emissions and invest in renewable energy for a sustainable future. #ClimateAction",
        "rejected": "The Earth is getting hotter, but it's not a big deal. We'll figure it out eventually. #WhateverHappensHappens"
    },
    {
        "prompt": "Explain the concept of gravity to a 5-year-old:",
        "chosen": "Gravity is like a big invisible hug that the Earth gives to everything. It's why things fall down when you drop them and why we don't float away!",
        "rejected": "Gravity is a fundamental force described by Einstein's theory of general relativity, causing spacetime curvature proportional to mass-energy density."
    }
]

# Convert to Dataset
dataset = Dataset.from_pandas(pd.DataFrame(dpo_data))

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Tokenize function
def tokenize_function(examples):
    chosen_inputs = tokenizer(examples["prompt"] + examples["chosen"], truncation=True, padding="max_length", max_length=512)
    rejected_inputs = tokenizer(examples["prompt"] + examples["rejected"], truncation=True, padding="max_length", max_length=512)
    
    return {
        "chosen_input_ids": chosen_inputs["input_ids"],
        "chosen_attention_mask": chosen_inputs["attention_mask"],
        "rejected_input_ids": rejected_inputs["input_ids"],
        "rejected_attention_mask": rejected_inputs["attention_mask"],
    }

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print("DPO dataset prepared and tokenized!")
print(f"Number of examples: {len(tokenized_dataset)}")
print(f"Features: {tokenized_dataset.features}")
```

Slide 9: Preparing an ORPO (Odds Ratio Preference Optimization) Dataset

ORPO is a variant of DPO that uses odds ratios instead of direct preferences. The dataset preparation is similar to DPO, but it includes an additional step of calculating odds ratios based on the preference data. This method can provide more nuanced training signals.

```python
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Sample preference data
preference_data = [
    {"prompt": "Summarize climate change:", "response_A": "Global warming caused by human activities.", "response_B": "Natural climate cycle, not human-caused.", "preference": "A"},
    {"prompt": "Describe healthy eating:", "response_A": "Balanced diet with variety of nutrients.", "response_B": "Extreme restriction and fad diets.", "preference": "A"},
]

# Convert preferences to odds ratios
def preference_to_odds_ratio(preference):
    if preference == "A":
        return np.log(0.9 / 0.1)  # Strong preference for A
    elif preference == "B":
        return np.log(0.1 / 0.9)  # Strong preference for B
    else:
        return 0  # No preference

for item in preference_data:
    item["odds_ratio"] = preference_to_odds_ratio(item["preference"])

# Create dataset
dataset = Dataset.from_pandas(pd.DataFrame(preference_data))

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Tokenize function
def tokenize_function(examples):
    input_A = tokenizer(examples["prompt"] + examples["response_A"], truncation=True, padding="max_length", max_length=512)
    input_B = tokenizer(examples["prompt"] + examples["response_B"], truncation=True, padding="max_length", max_length=512)
    
    return {
        "input_ids_A": input_A["input_ids"],
        "attention_mask_A": input_A["attention_mask"],
        "input_ids_B": input_B["input_ids"],
        "attention_mask_B": input_B["attention_mask"],
        "odds_ratio": examples["odds_ratio"]
    }

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print("ORPO dataset prepared and tokenized!")
print(f"Number of examples: {len(tokenized_dataset)}")
print(f"Features: {tokenized_dataset.features}")
```

Slide 10: Real-life Example: Sentiment Analysis Fine-tuning

In this example, we'll fine-tune a pre-trained model for sentiment analysis of product reviews. This task is common in e-commerce and customer feedback analysis.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load dataset
dataset = load_dataset("amazon_reviews_multi", "en")

# Preprocess function
def preprocess_function(examples):
    return tokenizer(examples["review_body"], truncation=True, padding="max_length")

# Load model and tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize datasets
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

print("Fine-tuning complete!")
```

Slide 11: Real-life Example: Question Answering System

In this example, we'll fine-tune a model for a question answering system, which is useful in various applications such as customer support chatbots or information retrieval systems.

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset("squad")

# Load model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocess function
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Tokenize datasets
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

print("Question Answering model fine-tuned successfully!")
```

Slide 12: Monitoring and Evaluating Fine-tuned Models

Monitoring and evaluating fine-tuned models is crucial to ensure they perform well on the target task and generalize to unseen data. This process involves tracking various metrics during training and performing thorough evaluations on held-out test sets.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt

# Load dataset (using IMDB for sentiment analysis as an example)
dataset = load_dataset("imdb")

# Preprocess function
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

# Load model and tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize datasets
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments with evaluation logging
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# Fine-tune the model
train_result = trainer.train()

# Evaluate the model
eval_result = trainer.evaluate()

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(train_result.history['train_loss'])
plt.title('Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()

# Print evaluation results
print("Evaluation results:", eval_result)

# Perform error analysis
test_pred = trainer.predict(tokenized_datasets["test"])
incorrect_predictions = np.where(test_pred.predictions.argmax(-1) != test_pred.label_ids)[0]

print(f"Number of incorrect predictions: {len(incorrect_predictions)}")
print("Sample of incorrect predictions:")
for i in incorrect_predictions[:5]:
    print(f"True label: {dataset['test'][i]['label']}, Predicted: {test_pred.predictions[i].argmax()}")
    print(f"Text: {dataset['test'][i]['text'][:100]}...\n")

print("Fine-tuning and evaluation complete!")
```

Slide 13: Best Practices for LLM Fine-tuning

When fine-tuning LLMs, follow these best practices to ensure optimal results:

1. Use a representative and diverse dataset
2. Start with a pre-trained model close to your target domain
3. Employ techniques like gradient accumulation for larger effective batch sizes
4. Use learning rate scheduling and early stopping
5. Regularly evaluate on a held-out validation set
6. Perform thorough error analysis
7. Consider using techniques like LoRA for efficient fine-tuning
8. Be mindful of ethical considerations and potential biases

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# Load dataset
dataset = load_dataset("glue", "mrpc")

# Load model and tokenizer
model_name = "roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "key", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(model, lora_config)

# Preprocess function
def preprocess_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")

# Tokenize datasets
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Training arguments with best practices
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,  # Effective batch size of 64
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Perform error analysis (pseudocode)
# test_results = trainer.predict(tokenized_datasets["test"])
# analyze_errors(test_results.predictions, test_results.label_ids, dataset["test"])

print("Fine-tuning complete with best practices applied!")
```

Slide 14: Ethical Considerations in LLM Fine-tuning

When fine-tuning LLMs, it's crucial to consider ethical implications:

1. Data bias: Ensure your training data is diverse and representative
2. Model bias: Regularly audit your model's outputs for unfair biases
3. Privacy: Use anonymized data and respect data protection regulations
4. Dual use: Consider potential misuse of your fine-tuned model
5. Environmental impact: Use efficient fine-tuning methods to reduce computational resources
6. Transparency: Document your fine-tuning process and model limitations

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Pseudocode for ethical data preparation
def prepare_ethical_dataset(data_path):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    
    # Ensure balanced representation
    df = balance_demographics(df)
    
    # Anonymize sensitive information
    df = anonymize_data(df)
    
    # Split data
    train_data, test_data = train_test_split(df, test_size=0.2, stratify=df['label'])
    
    return train_data, test_data

# Pseudocode for model fine-tuning with ethical considerations
def ethical_fine_tuning(model_name, train_data, test_data):
    # Load pre-trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize data
    train_encodings = tokenizer(train_data['text'].tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test_data['text'].tolist(), truncation=True, padding=True)
    
    # Set up training arguments (with reduced computational resources)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        load_best_model_at_end=True,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encodings,
        eval_dataset=test_encodings,
    )
    
    # Fine-tune the model
    trainer.train()
    
    # Evaluate the model
    evaluation_results = trainer.evaluate()
    
    # Audit model for biases
    audit_results = audit_model_biases(model, test_data)
    
    # Document process and results
    document_fine_tuning_process(model, training_args, evaluation_results, audit_results)
    
    return model, evaluation_results, audit_results

# Main execution
train_data, test_data = prepare_ethical_dataset('path_to_your_data.csv')
model, eval_results, audit_results = ethical_fine_tuning('bert-base-uncased', train_data, test_data)

print("Ethical fine-tuning complete. Check documentation for process details and audit results.")
```

Slide 15: Additional Resources

For those interested in diving deeper into LLM fine-tuning, here are some valuable resources:

1. "Fine-Tuning Language Models from Human Preferences" by D. Ziegler et al. (2019) ArXiv: [https://arxiv.org/abs/1909.08593](https://arxiv.org/abs/1909.08593)
2. "LoRA: Low-Rank Adaptation of Large Language Models" by E. J. Hu et al. (2021) ArXiv: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
3. "On the Opportunities and Risks of Foundation Models" by R. Bommasani et al. (2021) ArXiv: [https://arxiv.org/abs/2108.07258](https://arxiv.org/abs/2108.07258)
4. "Scaling Laws for Neural Language Models" by J. Kaplan et al. (2020) ArXiv: [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
5. "Language Models are Few-Shot Learners" by T. Brown et al. (2020) ArXiv: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

These papers provide in-depth insights into various aspects of LLM fine-tuning, from techniques like RLHF and LoRA to broader considerations about the impact and potential of large language models.

