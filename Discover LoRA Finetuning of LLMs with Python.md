## Discover LoRA Finetuning of LLMs with Python
Slide 1: Introduction to LoRA Finetuning

LoRA (Low-Rank Adaptation) is a technique for efficiently fine-tuning large language models (LLMs) with minimal computational resources. It works by adding small, trainable matrices to the model's attention layers, allowing for task-specific adaptation without modifying the entire model. This approach significantly reduces the number of trainable parameters and memory requirements.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("gpt2")
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
```

Slide 2: Understanding LoRA Architecture

LoRA introduces low-rank decomposition matrices (A and B) to the attention layers of the pre-trained model. These matrices are initialized randomly and trained on the specific task. The original weight matrix W is frozen, and the adaptation is performed through the low-rank matrices: W + AB^T. This approach allows for efficient fine-tuning with minimal parameter updates.

```python
class LoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(in_features, rank))
        self.B = torch.nn.Parameter(torch.randn(rank, out_features))
        self.W = torch.nn.Linear(in_features, out_features)
        self.W.weight.requires_grad = False  # Freeze original weights

    def forward(self, x):
        return self.W(x) + torch.matmul(torch.matmul(x, self.A), self.B)
```

Slide 3: Setting Up the Environment

To get started with LoRA finetuning, we need to set up our Python environment with the necessary libraries. We'll be using Hugging Face's Transformers library along with the PEFT (Parameter-Efficient Fine-Tuning) library, which implements LoRA.

```python
!pip install transformers peft datasets torch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
```

Slide 4: Preparing the Dataset

For this example, we'll use a simple text classification dataset. We'll load it using the Hugging Face datasets library and preprocess it for our model.

```python
dataset = load_dataset("imdb", split="train[:1000]")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenized_dataset = dataset.map(preprocess_function, batched=True)
```

Slide 5: Initializing the Model with LoRA

We'll start with a pre-trained GPT-2 model and apply LoRA to it. This involves configuring LoRA parameters and wrapping our model with the PEFT library.

```python
model = AutoModelForCausalLM.from_pretrained("gpt2")

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Slide 6: Defining Training Arguments

Before we start training, we need to set up our training arguments. These parameters control various aspects of the training process, such as learning rate, batch size, and number of epochs.

```python
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
```

Slide 7: Training the Model

Now that we have our model, dataset, and training arguments set up, we can start the finetuning process. The Trainer class handles the training loop for us.

```python
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./lora_finetuned_model")
```

Slide 8: Inference with the Finetuned Model

After training, we can use our finetuned model for inference. Here's how to load the model and generate text based on a prompt.

```python
from peft import PeftModel, PeftConfig

config = PeftConfig.from_pretrained("./lora_finetuned_model")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, "./lora_finetuned_model")

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

prompt = "This movie was"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

outputs = model.generate(input_ids=input_ids, max_length=50, num_return_sequences=1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Slide 9: Real-Life Example: Sentiment Analysis

Let's apply our LoRA-finetuned model to a real-world task: sentiment analysis. We'll use it to classify movie reviews as positive or negative.

```python
def classify_sentiment(review):
    prompt = f"Classify the sentiment of this movie review: '{review}'\nSentiment:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, max_length=len(prompt) + 10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Sentiment:")[-1].strip()

reviews = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "I was disappointed by this film. The plot was confusing and the acting was subpar."
]

for review in reviews:
    sentiment = classify_sentiment(review)
    print(f"Review: {review}\nSentiment: {sentiment}\n")
```

Slide 10: Real-Life Example: Text Summarization

Another practical application of our LoRA-finetuned model is text summarization. We can use it to generate concise summaries of longer texts.

```python
def summarize_text(text):
    prompt = f"Summarize the following text:\n{text}\n\nSummary:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, max_length=len(prompt) + 100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Summary:")[-1].strip()

long_text = """
The Internet of Things (IoT) is transforming the way we live and work. 
It refers to the interconnected network of physical devices, vehicles, 
home appliances, and other items embedded with electronics, software, 
sensors, and network connectivity, which enables these objects to collect 
and exchange data. The IoT has applications in various fields, including 
smart homes, healthcare, agriculture, and industrial automation.
"""

summary = summarize_text(long_text)
print(f"Original text:\n{long_text}\n\nSummary:\n{summary}")
```

Slide 11: Hyperparameter Tuning for LoRA

Optimizing LoRA hyperparameters can significantly impact the model's performance. Key parameters include the rank (r), alpha, and target modules. Here's an example of how to perform a simple grid search for these parameters.

```python
import itertools

def train_and_evaluate(r, alpha, target_modules):
    lora_config = LoraConfig(r=r, lora_alpha=alpha, target_modules=target_modules)
    model = get_peft_model(AutoModelForCausalLM.from_pretrained("gpt2"), lora_config)
    
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
    trainer.train()
    
    # Evaluate the model (you need to implement your own evaluation metric)
    return evaluate_model(model)

r_values = [4, 8, 16]
alpha_values = [16, 32, 64]
target_modules = [["c_attn"], ["c_attn", "c_proj"]]

best_score = float('-inf')
best_params = None

for r, alpha, modules in itertools.product(r_values, alpha_values, target_modules):
    score = train_and_evaluate(r, alpha, modules)
    if score > best_score:
        best_score = score
        best_params = (r, alpha, modules)

print(f"Best parameters: r={best_params[0]}, alpha={best_params[1]}, target_modules={best_params[2]}")
```

Slide 12: Visualizing LoRA's Impact

To better understand how LoRA affects the model, we can visualize the attention patterns before and after finetuning. This can provide insights into how the model's focus changes for specific tasks.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(model, text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    attention = outputs.attentions[-1].squeeze().detach().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, cmap="YlOrRd")
    plt.title("Attention Pattern")
    plt.xlabel("Token Position (Key)")
    plt.ylabel("Token Position (Query)")
    plt.show()

text = "The quick brown fox jumps over the lazy dog."

print("Attention pattern before LoRA finetuning:")
plot_attention(AutoModelForCausalLM.from_pretrained("gpt2"), text)

print("\nAttention pattern after LoRA finetuning:")
plot_attention(model, text)
```

Slide 13: Merging LoRA Weights

After finetuning, we can merge the LoRA weights with the base model for efficient inference. This step combines the original model weights with the learned LoRA adaptations.

```python
from peft import PeftModel

# Load the base model and LoRA weights
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
peft_model = PeftModel.from_pretrained(base_model, "./lora_finetuned_model")

# Merge weights
merged_model = peft_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("./merged_model")

# Now you can use the merged model for inference without LoRA overhead
merged_model = AutoModelForCausalLM.from_pretrained("./merged_model")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "The future of AI is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
outputs = merged_model.generate(input_ids=input_ids, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Slide 14: Additional Resources

For those interested in diving deeper into LoRA and efficient fine-tuning techniques, here are some valuable resources:

1. Original LoRA paper: "LoRA: Low-Rank Adaptation of Large Language Models" (arXiv:2106.09685)
2. Hugging Face PEFT library documentation: [https://huggingface.co/docs/peft/index](https://huggingface.co/docs/peft/index)
3. "Parameter-Efficient Transfer Learning for NLP" (arXiv:1902.00751)
4. "Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning" (arXiv:2303.15647)

These resources provide in-depth explanations of LoRA and related techniques, as well as their applications in various natural language processing tasks.

