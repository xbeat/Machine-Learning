## Enhancing Open Source LLMs with Attached Heads
Slide 1: Introduction to Attaching Heads to Open Source LLMs

Attaching heads to open source Large Language Models (LLMs) is a powerful technique for enhancing their capabilities. This process involves adding specialized neural network layers on top of pre-trained LLMs to perform specific tasks. We'll explore various methods including linear probes, multi-task fine-tuning, and LLM regression.

```python
import torch
import transformers

# Load a pre-trained LLM
model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

# Define a simple head (linear layer)
head = torch.nn.Linear(model.config.n_embd, 2)  # Binary classification

# Combine the LLM and the head
class ModelWithHead(torch.nn.Module):
    def __init__(self, base_model, head):
        super().__init__()
        self.base_model = base_model
        self.head = head

    def forward(self, input_ids):
        outputs = self.base_model(input_ids)
        return self.head(outputs.last_hidden_state[:, -1, :])

# Create the combined model
model_with_head = ModelWithHead(model, head)
```

Slide 2: Linear Probes

Linear probes are simple yet effective tools for analyzing LLMs. They involve attaching a linear layer to the LLM and training only this layer while keeping the base model frozen. This technique helps us understand what information is captured in the LLM's representations.

```python
import torch.optim as optim

# Freeze the base model parameters
for param in model_with_head.base_model.parameters():
    param.requires_grad = False

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model_with_head.head.parameters(), lr=0.001)

# Training loop (pseudo-code)
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model_with_head(batch.input_ids)
        loss = criterion(outputs, batch.labels)
        loss.backward()
        optimizer.step()

# Result: The linear probe is now trained to perform the task
```

Slide 3: Multi-task Fine-tuning

Multi-task fine-tuning involves training an LLM on multiple tasks simultaneously. This approach can lead to improved performance across various tasks and better generalization. We'll attach multiple heads to the LLM, each responsible for a different task.

```python
class MultiTaskModel(torch.nn.Module):
    def __init__(self, base_model, num_tasks, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.task_heads = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, 2) for _ in range(num_tasks)
        ])

    def forward(self, input_ids, task_id):
        outputs = self.base_model(input_ids)
        return self.task_heads[task_id](outputs.last_hidden_state[:, -1, :])

# Create a multi-task model
num_tasks = 3
multi_task_model = MultiTaskModel(model, num_tasks, model.config.n_embd)

# Training loop (pseudo-code)
for epoch in range(num_epochs):
    for batch in multi_task_dataloader:
        optimizer.zero_grad()
        outputs = multi_task_model(batch.input_ids, batch.task_id)
        loss = criterion(outputs, batch.labels)
        loss.backward()
        optimizer.step()

# Result: The model is now fine-tuned for multiple tasks
```

Slide 4: LLM Regression

LLM regression involves using an LLM to predict continuous values rather than discrete classes. This technique is useful for tasks such as sentiment analysis on a continuous scale or predicting numerical values based on text input.

```python
class RegressionHead(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, 64)
        self.linear2 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)

# Create a model for regression
regression_model = ModelWithHead(model, RegressionHead(model.config.n_embd))

# Define loss function and optimizer for regression
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(regression_model.parameters(), lr=0.001)

# Training loop (pseudo-code)
for epoch in range(num_epochs):
    for batch in regression_dataloader:
        optimizer.zero_grad()
        outputs = regression_model(batch.input_ids)
        loss = criterion(outputs, batch.values)
        loss.backward()
        optimizer.step()

# Result: The model can now predict continuous values
```

Slide 5: Attention Visualization

Visualizing attention patterns can provide insights into how the LLM processes input. We'll create a simple attention visualization tool to help understand the model's focus during inference.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    
    attn = outputs.attentions[-1].mean(dim=1).mean(dim=1).squeeze()
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn.detach().numpy().reshape(1, -1), annot=True, xticklabels=tokens)
    plt.title("Attention Visualization")
    plt.show()

# Example usage
text = "The quick brown fox jumps over the lazy dog."
visualize_attention(model, tokenizer, text)

# Result: A heatmap showing attention weights for each token
```

Slide 6: Real-life Example: Sentiment Analysis

Let's apply our knowledge to a real-world task: sentiment analysis. We'll attach a classification head to an LLM to predict sentiment from movie reviews.

```python
from datasets import load_dataset

# Load the IMDB dataset
dataset = load_dataset("imdb")

# Tokenize the dataset
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)

# Create a sentiment analysis model
sentiment_model = ModelWithHead(model, torch.nn.Linear(model.config.n_embd, 2))

# Training loop (pseudo-code)
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = sentiment_model(batch["input_ids"])
        loss = criterion(outputs, batch["label"])
        loss.backward()
        optimizer.step()

# Example prediction
text = "This movie was absolutely fantastic! I loved every minute of it."
inputs = tokenizer(text, return_tensors="pt")
outputs = sentiment_model(inputs.input_ids)
sentiment = "Positive" if outputs.argmax().item() == 1 else "Negative"
print(f"Predicted sentiment: {sentiment}")

# Result: The model can now predict sentiment from movie reviews
```

Slide 7: Feature Extraction and Transfer Learning

Feature extraction involves using the representations learned by an LLM for downstream tasks without fine-tuning the entire model. This technique is particularly useful when working with limited computational resources or small datasets.

```python
class FeatureExtractor(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids):
        with torch.no_grad():
            outputs = self.base_model(input_ids)
        return outputs.last_hidden_state[:, -1, :]

# Create a feature extractor
feature_extractor = FeatureExtractor(model)

# Define a simple classifier
classifier = torch.nn.Sequential(
    torch.nn.Linear(model.config.n_embd, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 2)
)

# Training loop (pseudo-code)
for epoch in range(num_epochs):
    for batch in dataloader:
        features = feature_extractor(batch.input_ids)
        outputs = classifier(features)
        loss = criterion(outputs, batch.labels)
        loss.backward()
        optimizer.step()

# Result: A classifier trained on features extracted from the LLM
```

Slide 8: Prompt Engineering with Attached Heads

Prompt engineering is a technique to guide LLMs towards desired outputs by crafting specific input prompts. We can combine this with attached heads for more targeted responses.

```python
class PromptModel(torch.nn.Module):
    def __init__(self, base_model, head):
        super().__init__()
        self.base_model = base_model
        self.head = head

    def forward(self, input_ids, prompt_ids):
        combined_input = torch.cat([prompt_ids, input_ids], dim=1)
        outputs = self.base_model(combined_input)
        return self.head(outputs.last_hidden_state[:, -1, :])

# Create a prompt-based model
prompt_model = PromptModel(model, torch.nn.Linear(model.config.n_embd, 2))

# Example usage
prompt = "Classify the following text as positive or negative: "
text = "I had a great day at the park!"
prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = tokenizer(text, return_tensors="pt").input_ids

outputs = prompt_model(input_ids, prompt_ids)
sentiment = "Positive" if outputs.argmax().item() == 1 else "Negative"
print(f"Predicted sentiment: {sentiment}")

# Result: The model uses both the prompt and input for prediction
```

Slide 9: Ensemble of LLM Heads

Ensemble methods combine predictions from multiple models to improve overall performance. We can create an ensemble of different heads attached to the same LLM base.

```python
class EnsembleModel(torch.nn.Module):
    def __init__(self, base_model, num_heads):
        super().__init__()
        self.base_model = base_model
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(base_model.config.n_embd, 2) for _ in range(num_heads)
        ])

    def forward(self, input_ids):
        outputs = self.base_model(input_ids)
        head_outputs = [head(outputs.last_hidden_state[:, -1, :]) for head in self.heads]
        return torch.stack(head_outputs).mean(dim=0)

# Create an ensemble model
ensemble_model = EnsembleModel(model, num_heads=5)

# Training loop (pseudo-code)
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = ensemble_model(batch.input_ids)
        loss = criterion(outputs, batch.labels)
        loss.backward()
        optimizer.step()

# Result: An ensemble of heads for improved prediction
```

Slide 10: Gradient-based Interpretation

Understanding how LLMs make decisions is crucial. We can use gradient-based methods to interpret the importance of input tokens for the model's predictions.

```python
def interpret_prediction(model, tokenizer, text, target_class):
    model.zero_grad()
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_embeds = model.base_model.transformer.wte(input_ids)
    input_embeds.requires_grad_()
    
    outputs = model(inputs_embeds=input_embeds)
    loss = outputs[:, target_class].sum()
    loss.backward()
    
    gradients = input_embeds.grad.sum(dim=2).squeeze()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    for token, grad in zip(tokens, gradients):
        print(f"{token}: {grad.item():.4f}")

# Example usage
text = "The food at this restaurant was delicious and the service was excellent."
interpret_prediction(sentiment_model, tokenizer, text, target_class=1)  # 1 for positive sentiment

# Result: Token-wise importance scores for the prediction
```

Slide 11: Continual Learning with LLM Heads

Continual learning allows models to learn new tasks without forgetting previously learned ones. We'll implement a simple replay buffer to maintain performance on old tasks while learning new ones.

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, sample):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(sample)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

# Create a replay buffer
replay_buffer = ReplayBuffer(capacity=10000)

# Continual learning loop (pseudo-code)
for task in tasks:
    for epoch in range(num_epochs):
        for batch in task_dataloader:
            # Train on current task
            optimizer.zero_grad()
            outputs = model_with_head(batch.input_ids)
            loss = criterion(outputs, batch.labels)
            loss.backward()
            optimizer.step()

            # Add samples to replay buffer
            replay_buffer.add((batch.input_ids, batch.labels))

        # Replay old samples
        old_samples = replay_buffer.sample(batch_size)
        optimizer.zero_grad()
        outputs = model_with_head(old_samples.input_ids)
        loss = criterion(outputs, old_samples.labels)
        loss.backward()
        optimizer.step()

# Result: A model that can learn new tasks while maintaining performance on old ones
```

Slide 12: Few-shot Learning with LLM Heads

Few-shot learning enables models to adapt to new tasks with limited examples. We'll implement a simple few-shot learning approach using attached heads.

```python
def few_shot_train(model, support_set, query_set, num_iterations=100):
    optimizer = optim.Adam(model.head.parameters(), lr=0.01)
    
    for _ in range(num_iterations):
        # Train on support set
        for inputs, labels in support_set:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate on query set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in query_set:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

# Example usage
support_set = [(torch.tensor([1, 2, 3]), torch.tensor([1])), (torch.tensor([4, 5, 6]), torch.tensor([0]))]
query_set = [(torch.tensor([2, 3, 4]), torch.tensor([1])), (torch.tensor([5, 6, 7]), torch.tensor([0]))]

accuracy = few_shot_train(model_with_head, support_set, query_set)
print(f"Few-shot learning accuracy: {accuracy:.2f}")

# Result: A model that can adapt to new tasks with limited examples
```

Slide 13: Explainable AI with LLM Heads

Explainable AI techniques help us understand the decision-making process of our models. We'll implement a simple explanation method using input gradients to highlight important words in the input text.

```python
def explain_prediction(model, tokenizer, text, target_class):
    model.eval()
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_embed = model.base_model.get_input_embeddings()(input_ids)
    input_embed.requires_grad_()

    outputs = model(inputs_embeds=input_embed)
    target_score = outputs[0][0][target_class]
    
    model.zero_grad()
    target_score.backward()
    
    word_importance = input_embed.grad.abs().sum(dim=2).squeeze()
    words = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    for word, importance in zip(words, word_importance):
        print(f"{word}: {importance.item():.4f}")

# Example usage
text = "The movie was fantastic and the acting was superb."
explain_prediction(sentiment_model, tokenizer, text, target_class=1)  # 1 for positive sentiment

# Result: Word importance scores for the model's prediction
```

Slide 14: Real-life Example: Named Entity Recognition

Let's apply our knowledge to another real-world task: Named Entity Recognition (NER). We'll attach a sequence labeling head to an LLM to identify and classify named entities in text.

```python
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

# Load a NER dataset
dataset = load_dataset("conll2003")

# Tokenize the dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Create a NER model
class NERHead(torch.nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.linear(x)

ner_model = ModelWithHead(model, NERHead(model.config.n_embd, num_labels=9))  # 9 NER tags in CoNLL-2003

# Training loop (pseudo-code)
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = ner_model(batch["input_ids"])
        loss = criterion(outputs.view(-1, 9), batch["labels"].view(-1))
        loss.backward()
        optimizer.step()

# Example prediction
text = "John Smith works at Google in New York."
inputs = tokenizer(text, return_tensors="pt")
outputs = ner_model(inputs.input_ids)
predictions = outputs.argmax(dim=2)

# Decode predictions
ner_tags = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
for token, pred in zip(tokens, predictions[0]):
    print(f"{token}: {ner_tags[pred]}")

# Result: The model can now identify named entities in text
```

Slide 15: Additional Resources

For those interested in diving deeper into the topics covered in this presentation, here are some valuable resources:

1. "Attention Is All You Need" - The paper that introduced the Transformer architecture, which is the foundation of many modern LLMs. ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" - Introduces BERT, a milestone in transfer learning for NLP. ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Language Models are Few-Shot Learners" - Discusses GPT-3 and its capabilities in few-shot learning. ArXiv: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
4. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" - Introduces T5, a versatile model for various NLP tasks. ArXiv: [https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)
5. "A Survey of Deep Learning Techniques for Neural Machine Translation" - Provides an overview of various techniques used in NMT, many of which are applicable to other NLP tasks. ArXiv: [https://arxiv.org/abs/2002.07526](https://arxiv.org/abs/2002.07526)

These papers provide in-depth information on the underlying technologies and methodologies used in modern NLP and LLMs. They can help you understand the theoretical foundations behind the practical techniques we've discussed in this presentation.

