## Fine-Tuning LLMs with PEFT and ROUGE Score
Slide 1: Introduction to Fine-Tuning Large Language Models

Fine-tuning large language models (LLMs) has become a crucial technique in natural language processing. This process involves adapting pre-trained models to specific tasks or domains, improving their performance and efficiency. In this presentation, we'll explore fine-tuning using Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically focusing on Low-Rank Adaptation (LoRA), and evaluate the results using the Rouge score.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pre-trained model
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loaded model: {model_name}")
print(f"Model parameters: {model.num_parameters():,}")
```

Slide 2: Understanding Parameter-Efficient Fine-Tuning (PEFT)

PEFT techniques aim to fine-tune large language models while updating only a small subset of the model's parameters. This approach significantly reduces computational resources and storage requirements compared to full fine-tuning. PEFT methods maintain the model's general knowledge while adapting it to specific tasks, making them ideal for resource-constrained environments.

```python
from peft import get_peft_config, PeftModel, PeftConfig, LoraConfig

# Define a LoRA configuration
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

# Apply PEFT to the model
peft_model = PeftModel.from_pretrained(model, peft_config)

print(f"Trainable parameters: {peft_model.num_parameters(train=True):,}")
print(f"Total parameters: {peft_model.num_parameters():,}")
```

Slide 3: Low-Rank Adaptation (LoRA): A Deep Dive

LoRA is a PEFT technique that adds trainable low-rank matrices to the layers of a pre-trained model. These matrices capture task-specific information while keeping most of the original model frozen. LoRA's efficacy comes from its ability to learn meaningful adaptations with a small number of parameters, typically less than 1% of the original model's size.

```python
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        
    def forward(self, x):
        return x @ (self.A @ self.B)

# Example usage
lora_layer = LoRALayer(768, 768, rank=8)
input_tensor = torch.randn(1, 768)
output = lora_layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print(f"LoRA parameters: {sum(p.numel() for p in lora_layer.parameters()):,}")
```

Slide 4: Preparing Data for Fine-Tuning

Before fine-tuning, we need to prepare our dataset. This involves tokenizing the text, creating attention masks, and formatting the data for our model. Let's use a simple example of preparing a dataset for sentiment analysis.

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load a sample dataset
dataset = load_dataset("imdb", split="train[:1000]")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=8)

print(f"Number of samples: {len(tokenized_dataset)}")
print(f"Sample features: {next(iter(dataloader)).keys()}")
```

Slide 5: Implementing LoRA Fine-Tuning

Now that we have our data prepared, let's implement LoRA fine-tuning. We'll use the Hugging Face Transformers library along with the PEFT library to fine-tune our model on the sentiment analysis task.

```python
from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Create a Trainer instance
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start fine-tuning
trainer.train()

print("Fine-tuning completed!")
print(f"Trained model saved to: {training_args.output_dir}")
```

Slide 6: Evaluating Fine-Tuned Models: Introduction to Rouge Score

The Rouge (Recall-Oriented Understudy for Gisting Evaluation) score is a set of metrics used to evaluate the quality of generated text, particularly in tasks like summarization. It compares the generated text to one or more reference texts, measuring the overlap of n-grams, word sequences, and word pairs.

```python
from rouge_score import rouge_scorer

def calculate_rouge(prediction, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(prediction, reference)
    return scores

# Example usage
prediction = "The cat sat on the mat."
reference = "A cat is sitting on the mat."
scores = calculate_rouge(prediction, reference)

for metric, score in scores.items():
    print(f"{metric}: {score.fmeasure:.4f}")
```

Slide 7: Understanding Rouge Metrics

Rouge offers several metrics, each capturing different aspects of text similarity:

* Rouge-N: Measures the overlap of n-grams between the generated and reference texts.
* Rouge-L: Calculates the longest common subsequence between the texts.
* Rouge-W: A weighted version of Rouge-L that favors consecutive matches.

Let's implement a function to calculate these metrics:

```python
def detailed_rouge_scores(prediction, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(prediction, reference)
    
    results = {}
    for metric, score in scores.items():
        results[f"{metric}_precision"] = score.precision
        results[f"{metric}_recall"] = score.recall
        results[f"{metric}_fmeasure"] = score.fmeasure
    
    return results

# Example usage
prediction = "The quick brown fox jumps over the lazy dog."
reference = "A fast brown fox leaps above a sleepy canine."
detailed_scores = detailed_rouge_scores(prediction, reference)

for metric, value in detailed_scores.items():
    print(f"{metric}: {value:.4f}")
```

Slide 8: Interpreting Rouge Scores

Rouge scores range from 0 to 1, where higher scores indicate better similarity between the generated and reference texts. However, interpreting these scores requires context and understanding of the specific task. Let's create a function to provide a qualitative interpretation of Rouge scores:

```python
def interpret_rouge_score(score):
    if score < 0.2:
        return "Poor similarity"
    elif score < 0.4:
        return "Fair similarity"
    elif score < 0.6:
        return "Moderate similarity"
    elif score < 0.8:
        return "Good similarity"
    else:
        return "Excellent similarity"

# Example usage
rouge_l_score = 0.65
interpretation = interpret_rouge_score(rouge_l_score)
print(f"Rouge-L score: {rouge_l_score:.2f}")
print(f"Interpretation: {interpretation}")
```

Slide 9: Real-Life Example: Fine-Tuning for Text Summarization

Let's apply our knowledge to a real-world scenario: fine-tuning a model for text summarization. We'll use a small dataset of news articles and their summaries to demonstrate the process.

```python
from datasets import load_dataset

# Load a summarization dataset
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:100]")

# Prepare the data
def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    
    labels = tokenizer(examples["highlights"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Fine-tune the model
trainer = Trainer(
    model=peft_model,
    args=TrainingArguments(output_dir="./summarization_model", num_train_epochs=3),
    train_dataset=tokenized_dataset,
)

trainer.train()
print("Fine-tuning for summarization completed!")
```

Slide 10: Evaluating the Summarization Model

Now that we've fine-tuned our model for summarization, let's evaluate its performance using the Rouge score. We'll generate summaries for a few test articles and compare them to the reference summaries.

```python
from transformers import pipeline

# Load the fine-tuned model
summarizer = pipeline("summarization", model=peft_model, tokenizer=tokenizer)

# Test data
test_article = """
The United Nations has warned that the world is facing its largest humanitarian crisis since 1945. 
More than 20 million people in four countries are at risk of starvation and famine. 
The UN humanitarian chief Stephen O'Brien has called for an urgent mobilization of funds to 
prevent a catastrophe. The countries most at risk are Yemen, South Sudan, Somalia and Nigeria. 
Ongoing conflicts in these regions have exacerbated food shortages and economic crises.
"""

# Generate summary
generated_summary = summarizer(test_article, max_length=50, min_length=10, do_sample=False)[0]['summary_text']

# Reference summary (human-written)
reference_summary = "UN warns of largest humanitarian crisis since 1945 with over 20 million at risk of starvation in four countries due to conflicts and economic issues."

# Calculate Rouge scores
scores = calculate_rouge(generated_summary, reference_summary)

print("Generated Summary:", generated_summary)
print("\nReference Summary:", reference_summary)
print("\nRouge Scores:")
for metric, score in scores.items():
    print(f"{metric}: {score.fmeasure:.4f}")
```

Slide 11: Real-Life Example: Sentiment Analysis Fine-Tuning

Let's explore another real-world application: fine-tuning a model for sentiment analysis of product reviews. We'll use a dataset of Amazon product reviews to demonstrate the process.

```python
from datasets import load_dataset

# Load a sentiment analysis dataset
dataset = load_dataset("amazon_reviews_multi", "en", split="train[:1000]")

# Prepare the data
def preprocess_function(examples):
    return tokenizer(examples["review_body"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("stars", "labels")
tokenized_dataset = tokenized_dataset.remove_columns(["review_id", "product_id", "reviewer_id", "review_title", "product_category", "review_body", "language"])

# Fine-tune the model
trainer = Trainer(
    model=peft_model,
    args=TrainingArguments(output_dir="./sentiment_model", num_train_epochs=3),
    train_dataset=tokenized_dataset,
)

trainer.train()
print("Fine-tuning for sentiment analysis completed!")
```

Slide 12: Evaluating the Sentiment Analysis Model

Now that we've fine-tuned our model for sentiment analysis, let's evaluate its performance on some test reviews. We'll use the fine-tuned model to predict sentiments and compare them with the actual ratings.

```python
from transformers import pipeline

# Load the fine-tuned model
sentiment_analyzer = pipeline("sentiment-analysis", model=peft_model, tokenizer=tokenizer)

# Test data
test_reviews = [
    "This product exceeded my expectations. It's durable and works perfectly!",
    "I'm disappointed with the quality. It broke after just a week of use.",
    "Average product. Does the job, but nothing special."
]

# Predict sentiments
for review in test_reviews:
    result = sentiment_analyzer(review)[0]
    sentiment = "Positive" if result['label'] == "LABEL_1" else "Negative"
    confidence = result['score']
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.2f})")
    print()

# Note: In a real scenario, you would compare these predictions with actual ratings
# and calculate metrics like accuracy, precision, recall, and F1-score.
```

Slide 13: Challenges and Considerations in Fine-Tuning

While fine-tuning LLMs with PEFT techniques like LoRA can be powerful, it's important to be aware of potential challenges:

1. Overfitting: Fine-tuned models may perform well on the training data but fail to generalize to new, unseen data.
2. Catastrophic forgetting: The model might lose some of its general knowledge while adapting to a specific task.
3. Bias amplification: Fine-tuning on biased datasets can exacerbate existing biases in the model.

To address these challenges, consider the following:

```python
from transformers import EarlyStoppingCallback

# Example: Using early stopping to prevent overfitting
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

trainer = Trainer(
    model=peft_model,
    args=TrainingArguments(
        output_dir="./robust_model",
        num_train_epochs=10,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
    ),
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset.select(range(100)),  # Small validation set
    callbacks=[early_stopping_callback],
)

trainer.train()

print("Robust fine-tuning completed!")
```

Slide 14: Conclusion and Future Directions

Fine-tuning large language models using PEFT techniques like LoRA offers a powerful way to adapt pre-trained models to specific tasks while maintaining efficiency. The Rouge score provides a valuable metric for evaluating the quality of generated text, particularly in tasks like summarization.

As the field of NLP continues to evolve, we can expect to see:

1. More advanced PEFT techniques that further reduce the computational requirements of fine-tuning.
2. Improved evaluation metrics that capture more nuanced aspects of text quality and relevance.
3. Techniques to address challenges like bias and ensure the ethical use of fine-tuned models.

By combining these advancements with careful consideration of the challenges and best practices, we can continue to push the boundaries of what's possible with large language models.

Slide 15: Additional Resources

For those interested in diving deeper into the topics covered in this presentation, here are some valuable resources:

1. LoRA: Low-Rank Adaptation of Large Language Models ArXiv: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
2. ROUGE: A Package for Automatic Evaluation of Summaries ArXiv: [https://arxiv.org/abs/1803.01937](https://arxiv.org/abs/1803.01937)
3. Parameter-Efficient Transfer Learning for NLP ArXiv: [https://arxiv.org/abs/1902.00751](https://arxiv.org/abs/1902.00751)
4. Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning ArXiv: [https://arxiv](https://arxiv).

