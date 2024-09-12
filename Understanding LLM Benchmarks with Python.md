## Understanding LLM Benchmarks with Python
Slide 1: Understanding LLM Benchmarks using Python

Large Language Models (LLMs) have revolutionized natural language processing. To evaluate their performance, we use benchmarks. This presentation explores various LLM benchmarks and demonstrates how to implement them using Python.

```python
import transformers
import datasets

# Load a pre-trained model
model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

# Load a benchmark dataset
dataset = datasets.load_dataset("glue", "mnli")

# Example of tokenizing and processing data
inputs = tokenizer(dataset["train"][0]["premise"], return_tensors="pt")
outputs = model(**inputs)

print(f"Input: {dataset['train'][0]['premise']}")
print(f"Output logits shape: {outputs.logits.shape}")
```

Slide 2: GLUE Benchmark

The General Language Understanding Evaluation (GLUE) benchmark is a collection of tasks designed to evaluate natural language understanding systems. It includes tasks like sentiment analysis, question answering, and textual entailment.

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load GLUE dataset
dataset = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()
```

Slide 3: SuperGLUE Benchmark

SuperGLUE is an extension of GLUE, featuring more challenging tasks. It includes tasks like reading comprehension with commonsense reasoning and word sense disambiguation.

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer

# Load SuperGLUE dataset (e.g., MultiRC)
dataset = load_dataset("super_glue", "multirc")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    passages = [p.strip() for p in examples["paragraph"]]
    first_sentences = [[p] * 2 for p in passages]
    second_sentences = [[q + " " + examples["answer1"][i], q + " " + examples["answer2"][i]] for i, q in enumerate(questions)]
    
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, padding="max_length")
    tokenized_examples["label"] = examples["label"]
    return tokenized_examples

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

model = AutoModelForMultipleChoice.from_pretrained("roberta-base")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()
```

Slide 4: LAMBADA Benchmark

LAMBADA (LAnguage Modeling Broadened to Account for Discourse Aspects) is a dataset designed to evaluate the capabilities of computational models for text understanding by means of a word prediction task.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

def evaluate_lambada(text):
    inputs = tokenizer.encode(text, return_tensors='pt')
    target = inputs[:, -1].unsqueeze(0)
    inputs = inputs[:, :-1]

    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
    predicted_token_id = torch.argmax(probs).item()
    predicted_token = tokenizer.decode([predicted_token_id])
    actual_token = tokenizer.decode(target[0])
    
    print(f"Context: {tokenizer.decode(inputs[0])}")
    print(f"Predicted next word: {predicted_token}")
    print(f"Actual next word: {actual_token}")
    print(f"Correct: {predicted_token == actual_token}")

# Example LAMBADA-style text
text = "The chef put the strawberries in a bowl and sprinkled them with sugar. He then added a splash of balsamic vinegar to enhance the"

evaluate_lambada(text)
```

Slide 5: SQuAD Benchmark

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text from the corresponding reading passage.

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
    
    return answer

context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world."

question = "Who is the Eiffel Tower named after?"

answer = answer_question(question, context)
print(f"Question: {question}")
print(f"Answer: {answer}")
```

Slide 6: BLEU Score for Machine Translation

BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. It's particularly useful for comparing the performance of different machine translation systems.

```python
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

def calculate_bleu(reference, candidate):
    reference = [word_tokenize(reference.lower())]
    candidate = word_tokenize(candidate.lower())
    score = sentence_bleu(reference, candidate)
    return score

reference = "The cat is on the mat."
candidate1 = "The cat sits on the mat."
candidate2 = "On the mat is a cat."

score1 = calculate_bleu(reference, candidate1)
score2 = calculate_bleu(reference, candidate2)

print(f"Reference: {reference}")
print(f"Candidate 1: {candidate1}")
print(f"BLEU Score 1: {score1:.4f}")
print(f"Candidate 2: {candidate2}")
print(f"BLEU Score 2: {score2:.4f}")
```

Slide 7: ROUGE Metric for Text Summarization

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used for evaluating automatic summarization and machine translation software in natural language processing. It works by comparing an automatically produced summary or translation against a set of reference summaries (typically human-produced).

```python
from rouge import Rouge

def calculate_rouge(reference, summary):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference)
    return scores[0]

reference = "The quick brown fox jumps over the lazy dog. It was a beautiful day in the forest."
summary1 = "The fox jumps over the dog. It was a nice day."
summary2 = "A brown fox and a lazy dog were in the forest."

scores1 = calculate_rouge(reference, summary1)
scores2 = calculate_rouge(reference, summary2)

print(f"Reference: {reference}")
print(f"Summary 1: {summary1}")
print(f"ROUGE Scores 1: {scores1}")
print(f"Summary 2: {summary2}")
print(f"ROUGE Scores 2: {scores2}")
```

Slide 8: Perplexity Metric

Perplexity is a measurement of how well a probability distribution predicts a sample. In the context of natural language processing and LLMs, it's often used to evaluate language models. A lower perplexity indicates better performance.

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def calculate_perplexity(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text1 = "The cat sat on the mat."
text2 = "The cat sat on the cloud."

perplexity1 = calculate_perplexity(text1, model, tokenizer)
perplexity2 = calculate_perplexity(text2, model, tokenizer)

print(f"Text 1: {text1}")
print(f"Perplexity 1: {perplexity1:.2f}")
print(f"Text 2: {text2}")
print(f"Perplexity 2: {perplexity2:.2f}")
```

Slide 9: F1 Score for Named Entity Recognition

The F1 score is the harmonic mean of precision and recall. It's particularly useful for evaluating named entity recognition (NER) systems, where we want to measure both the accuracy of the entities identified and the completeness of the identification.

```python
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

def ner_prediction(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    return predictions[0].tolist()

def calculate_f1(true_labels, predicted_labels):
    return f1_score(true_labels, predicted_labels, average='weighted')

text = "John lives in New York and works for Google."
true_labels = [1, 0, 0, 1, 1, 0, 0, 0, 1]  # 1 for entity, 0 for non-entity
predicted_labels = ner_prediction(text)

f1 = calculate_f1(true_labels, predicted_labels)

print(f"Text: {text}")
print(f"True labels: {true_labels}")
print(f"Predicted labels: {predicted_labels}")
print(f"F1 Score: {f1:.4f}")
```

Slide 10: METEOR Score for Machine Translation

METEOR (Metric for Evaluation of Translation with Explicit ORdering) is a metric for the evaluation of machine translation output. It is based on the harmonic mean of unigram precision and recall, with recall weighted higher than precision.

```python
from nltk.translate import meteor_score
from nltk import word_tokenize

def calculate_meteor(reference, hypothesis):
    reference = word_tokenize(reference)
    hypothesis = word_tokenize(hypothesis)
    return meteor_score.meteor_score([reference], hypothesis)

reference = "The cat is sitting on the mat."
hypothesis1 = "The cat is on the mat."
hypothesis2 = "A dog is sitting on a rug."

score1 = calculate_meteor(reference, hypothesis1)
score2 = calculate_meteor(reference, hypothesis2)

print(f"Reference: {reference}")
print(f"Hypothesis 1: {hypothesis1}")
print(f"METEOR Score 1: {score1:.4f}")
print(f"Hypothesis 2: {hypothesis2}")
print(f"METEOR Score 2: {score2:.4f}")
```

Slide 11: TER (Translation Edit Rate)

TER is a metric used to evaluate machine translation output. It measures the number of edits required to change a system output into one of the reference translations. A lower TER score indicates better performance.

```python
from torchtext.data.metrics import bleu_score
import sacrebleu

def calculate_ter(reference, hypothesis):
    return sacrebleu.corpus_ter([hypothesis], [[reference]]).score

reference = "The cat is sitting on the mat."
hypothesis1 = "A cat sits on the mat."
hypothesis2 = "The dog is standing near the sofa."

ter1 = calculate_ter(reference, hypothesis1)
ter2 = calculate_ter(reference, hypothesis2)

print(f"Reference: {reference}")
print(f"Hypothesis 1: {hypothesis1}")
print(f"TER Score 1: {ter1:.4f}")
print(f"Hypothesis 2: {hypothesis2}")
print(f"TER Score 2: {ter2:.4f}")
```

Slide 12: GLUE Benchmark: Real-life Example

Let's use the GLUE benchmark to evaluate a model on the MRPC (Microsoft Research Paraphrase Corpus) task, which involves determining whether two sentences are paraphrases of each other.

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# Load MRPC dataset
dataset = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()

predictions = trainer.predict(tokenized_datasets["test"])
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

accuracy = accuracy_score(labels, preds)
f1 = f1_score(labels, preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
```

Slide 13: SQuAD Benchmark: Real-life Example

In this example, we'll use a pre-trained model to perform question answering on a passage from the SQuAD dataset.

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
It is named after the engineer Gustave Eiffel, whose company designed and built the tower. 
Constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair, it was initially 
criticized by some of France's leading artists and intellectuals for its design, but it has 
become a global cultural icon of France and one of the most recognizable structures in the world.
"""

questions = [
    "Who is the Eiffel Tower named after?",
    "When was the Eiffel Tower constructed?",
    "What was the initial reaction to the Eiffel Tower's design?"
]

for question in questions:
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
    
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")
```

Slide 14: Perplexity in Language Model Evaluation

Perplexity is a crucial metric for evaluating language models. It measures how well a model predicts a sample of text. Lower perplexity indicates better performance.

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def calculate_perplexity(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is revolutionizing various industries.",
    "Xylophone zealously quizzical jumbo wafting vexed."
]

for text in texts:
    perplexity = calculate_perplexity(text, model, tokenizer)
    print(f"Text: {text}")
    print(f"Perplexity: {perplexity:.2f}\n")
```

Slide 15: Additional Resources

For those interested in diving deeper into LLM benchmarks and evaluation metrics, here are some valuable resources:

1. "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding" (Wang et al., 2018) ArXiv: [https://arxiv.org/abs/1804.07461](https://arxiv.org/abs/1804.07461)
2. "SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems" (Wang et al., 2019) ArXiv: [https://arxiv.org/abs/1905.00537](https://arxiv.org/abs/1905.00537)
3. "SQuAD: 100,000+ Questions for Machine Comprehension of Text" (Rajpurkar et al., 2016) ArXiv: [https://arxiv.org/abs/1606.05250](https://arxiv.org/abs/1606.05250)
4. "BLEU: a Method for Automatic Evaluation of Machine Translation" (Papineni et al., 2002) ArXiv: [https://arxiv.org/abs/cs/0305007](https://arxiv.org/abs/cs/0305007)

These papers provide in-depth explanations of various benchmarks and metrics used in evaluating language models.

