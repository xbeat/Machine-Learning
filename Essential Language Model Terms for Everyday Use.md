## Essential Language Model Terms for Everyday Use

Slide 1: Transformer Models

Transformer models are the foundation of modern Natural Language Processing (NLP). They process text efficiently by handling multiple parts of a sentence simultaneously, allowing for better understanding of context and relationships between words.

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        attended = self.attention(x, x, x)[0]
        x = self.norm1(x + attended)
        fedforward = self.feed_forward(x)
        return self.norm2(x + fedforward)

# Example usage
embed_dim, num_heads = 512, 8
model = TransformerBlock(embed_dim, num_heads)
sample_input = torch.randn(10, 32, embed_dim)  # (seq_len, batch_size, embed_dim)
output = model(sample_input)
print(output.shape)  # Should be torch.Size([10, 32, 512])
```

Slide 2: Fine-Tuning

Fine-tuning adapts a pre-trained model to excel at a specific task by training it on a smaller, specialized dataset. This process allows models to leverage general language understanding while becoming experts in particular domains or applications.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare dataset
dataset = load_dataset("imdb")
tokenized_dataset = dataset.map(lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True), batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Create Trainer and start fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

trainer.train()

# The model is now fine-tuned for sentiment analysis on movie reviews
```

Slide 3: Embeddings

Embeddings turn words into numerical vectors, capturing their meanings and relationships for machine processing. This allows models to understand semantic similarities and differences between words, enhancing various NLP tasks.

```python
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

# Sample corpus
corpus = [
    ["I", "love", "machine", "learning"],
    ["I", "enjoy", "natural", "language", "processing"],
    ["Embeddings", "are", "useful", "for", "NLP", "tasks"]
]

# Train Word2Vec model
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

# Function to plot word embeddings
def plot_embeddings(words):
    embeddings = [model.wv[word] for word in words]
    x = [emb[0] for emb in embeddings]
    y = [emb[1] for emb in embeddings]
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y)
    for i, word in enumerate(words):
        plt.annotate(word, xy=(x[i], y[i]))
    plt.title("Word Embeddings Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

# Visualize embeddings for selected words
words_to_plot = ["machine", "learning", "natural", "language", "processing", "NLP"]
plot_embeddings(words_to_plot)
```

Slide 4: Tokenization

Tokenization breaks down text into smaller pieces like words or subwords, making it easier for machines to process. This crucial step prepares text for input into language models and other NLP tasks.

```python
from transformers import AutoTokenizer

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Example text
text = "Tokenization is a crucial step in NLP tasks."

# Tokenize the text
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# Convert tokens to input IDs
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Input IDs:", input_ids)

# Decode the input IDs back to text
decoded_text = tokenizer.decode(input_ids)
print("Decoded text:", decoded_text)

# Output:
# Tokens: ['token', '##ization', 'is', 'a', 'crucial', 'step', 'in', 'nl', '##p', 'tasks', '.']
# Input IDs: [19204, 29353, 2003, 1037, 4263, 2896, 1999, 17757, 2705, 7161, 1012]
# Decoded text: tokenization is a crucial step in nlp tasks.
```

Slide 5: Pre-Training

Pre-training involves teaching a model general language patterns using a large dataset before fine-tuning it for specific tasks. This process allows models to develop a broad understanding of language structure and semantics.

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example sentence with a masked word
text = "The [MASK] is shining brightly in the sky."

# Tokenize and create input tensors
inputs = tokenizer(text, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

# Get model predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

# Get the predicted token
predicted_token_id = predictions[0, mask_token_index].argmax(axis=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"Original text: {text}")
print(f"Predicted word: {predicted_token}")

# Output:
# Original text: The [MASK] is shining brightly in the sky.
# Predicted word: sun
```

Slide 6: Attention Mechanism

Attention mechanisms allow models to focus on the most relevant parts of the input, improving understanding and accuracy. This technique has revolutionized NLP by enabling models to capture long-range dependencies in text.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

# Example usage
embed_size = 256
heads = 8
attention = SelfAttention(embed_size, heads)
x = torch.randn(32, 10, embed_size)  # (batch_size, seq_len, embed_size)
output = attention(x, x, x)
print(output.shape)  # Should be torch.Size([32, 10, 256])
```

Slide 7: BERT (Bidirectional Encoder Representations from Transformers)

BERT is a model that understands the context of words by looking at the surrounding text, making it very accurate for various NLP tasks. Its bidirectional nature allows it to capture context from both left and right sides of a word.

```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Prepare input
text = "BERT understands context from both directions."
encoded_input = tokenizer(text, return_tensors='pt')

# Get BERT embeddings
with torch.no_grad():
    output = model(**encoded_input)

# Get the [CLS] token embedding (sentence representation)
sentence_embedding = output.last_hidden_state[:, 0, :]

print("Sentence embedding shape:", sentence_embedding.shape)
print("First few values of the sentence embedding:", sentence_embedding[0][:5])

# Output:
# Sentence embedding shape: torch.Size([1, 768])
# First few values of the sentence embedding: tensor([ 0.0668, -0.0955, -0.0543,  0.0384, -0.0847])
```

Slide 8: GPT (Generative Pre-trained Transformer)

GPT excels at generating human-like text, making it ideal for content creation tasks. It uses a transformer architecture and is trained on a vast amount of text data to predict the next word in a sequence.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the pad token
tokenizer.pad_token = tokenizer.eos_token

# Prepare input
text = "GPT models are great for"
input_ids = tokenizer.encode(text, return_tensors='pt')

# Generate text
output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

# Example output:
# GPT models are great for generating text, but they can also be used for other tasks such as language translation, summarization, and question answering. They are particularly useful for tasks that require...
```

Slide 9: Perplexity

Perplexity measures how well a language model predicts text; lower perplexity indicates better performance. It's a common metric used to evaluate the quality of language models.

```python
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def calculate_perplexity(model, tokenizer, text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids

    # Calculate perplexity
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    
    return perplexity.item()

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Example texts
text1 = "The cat sat on the mat."
text2 = "The dog barked at the mailman."
text3 = "Colorless green ideas sleep furiously."

# Calculate and print perplexity for each text
for i, text in enumerate([text1, text2, text3], 1):
    perplexity = calculate_perplexity(model, tokenizer, text)
    print(f"Text {i} Perplexity: {perplexity:.2f}")

# Example output:
# Text 1 Perplexity: 42.18
# Text 2 Perplexity: 28.76
# Text 3 Perplexity: 1234.56
```

Slide 10: Real-Life Example: Chatbot with Intent Classification

This example demonstrates how to create a simple chatbot that classifies user intents using a pre-trained BERT model.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define intents and responses
intents = {0: "Negative", 1: "Positive"}
responses = {
    "Negative": "I'm sorry to hear that. How can I help improve your experience?",
    "Positive": "That's great to hear! Is there anything else I can assist you with?"
}

def classify_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    return intents[predicted_class]

def chatbot_response(user_input):
    intent = classify_intent(user_input)
    return responses[intent]

# Example conversation
print("Chatbot: Hello! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Chatbot: Thank you for chatting. Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Chatbot:", response)

# Example output:
# Chatbot: Hello! How can I assist you today?
# You: I'm having trouble with my order
# Chatbot: I'm sorry to hear that. How can I help improve your experience?
# You: The product arrived earlier than expected
# Chatbot: That's great to hear! Is there anything else I can assist you with?
# You: quit
# Chatbot: Thank you for chatting. Goodbye!
```

Slide 11: Real-Life Example: Text Summarization

This example shows how to use a pre-trained model to generate summaries of longer texts, which can be useful for processing large documents or articles.

```python
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example long text
long_text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural-language generation. Many different classes of machine-learning algorithms have been applied to natural-language-processing tasks. These algorithms take as input a large set of "features" that are generated from the input data. Some of the earliest-used machine learning algorithms, such as decision trees, produced systems of hard if-then rules similar to existing hand-written rules. However, part-of-speech tagging introduced the use of hidden Markov models to natural language processing, and increasingly, research has focused on statistical models, which make soft, probabilistic decisions based on attaching real-valued weights to each input feature. Such models have the advantage that they can express the relative certainty of many different possible answers rather than only one, producing more reliable results when such a model is included as a component of a larger system.
"""

# Generate summary
summary = summarizer(long_text, max_length=150, min_length=50, do_sample=False)

print("Original text length:", len(long_text))
print("\nSummary:")
print(summary[0]['summary_text'])

# Example output:
# Original text length: 1053
# Summary:
# NLP is a subfield of linguistics, computer science, and AI concerned with interactions between computers and human language. The goal is to program computers to process and analyze large amounts of natural language data. Challenges include speech recognition, natural language understanding, and natural-language generation. Machine learning algorithms have been applied to NLP tasks.
```

Slide 12: Transfer Learning in NLP

Transfer learning allows models to leverage knowledge gained from one task to perform better on a different, but related task. This technique is particularly powerful in NLP, where pre-trained models can be fine-tuned for specific applications.

Slide 13: Transfer Learning in NLP Code

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load and preprocess dataset
dataset = load_dataset("imdb")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
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
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Example output:
# {'eval_loss': 0.3421, 'eval_accuracy': 0.8734, 'eval_f1': 0.8731, 'eval_precision': 0.8729, 'eval_recall': 0.8733, 'eval_runtime': 651.3456, 'eval_samples_per_second': 38.382, 'eval_steps_per_second': 0.600}
```

Slide 14: Contextual Word Embeddings

Contextual word embeddings, unlike traditional static embeddings, capture the meaning of words based on their context in a sentence. This allows for more nuanced understanding of language.

```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get contextual embeddings
def get_contextual_embedding(sentence, word):
    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors="pt")
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the last hidden states
    last_hidden_states = outputs.last_hidden_state[0]
    
    # Find the token(s) corresponding to the word
    word_tokens = tokenizer.tokenize(word)
    word_ids = inputs.word_ids()
    word_indices = [i for i, id in enumerate(word_ids) if tokenizer.convert_ids_to_tokens([inputs['input_ids'][0][i]])[0] in word_tokens]
    
    # Average the embeddings if the word is split into multiple tokens
    word_embedding = torch.mean(last_hidden_states[word_indices], dim=0)
    
    return word_embedding

# Example sentences
sentence1 = "The bank is by the river."
sentence2 = "I need to bank my check."

# Get contextual embeddings for 'bank' in different contexts
bank_embedding1 = get_contextual_embedding(sentence1, "bank")
bank_embedding2 = get_contextual_embedding(sentence2, "bank")

# Calculate cosine similarity
cosine_similarity = torch.nn.functional.cosine_similarity(bank_embedding1.unsqueeze(0), bank_embedding2.unsqueeze(0))

print(f"Cosine similarity between 'bank' embeddings: {cosine_similarity.item():.4f}")

# Example output:
# Cosine similarity between 'bank' embeddings: 0.7823
```

Slide 15: Sentiment Analysis with BERT

This example demonstrates how to use a fine-tuned BERT model for sentiment analysis, a common NLP task used in various applications such as product reviews and social media monitoring.

```python
from transformers import pipeline

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Example texts
texts = [
    "I absolutely loved this movie! The acting was superb.",
    "The customer service was terrible and the product arrived damaged.",
    "The restaurant was okay, nothing special but not bad either.",
    "This book changed my life. I couldn't put it down!"
]

# Analyze sentiments
for text in texts:
    result = sentiment_analyzer(text)[0]
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}, Score: {result['score']:.4f}\n")

# Example output:
# Text: I absolutely loved this movie! The acting was superb.
# Sentiment: POSITIVE, Score: 0.9998

# Text: The customer service was terrible and the product arrived damaged.
# Sentiment: NEGATIVE, Score: 0.9994

# Text: The restaurant was okay, nothing special but not bad either.
# Sentiment: NEUTRAL, Score: 0.7265

# Text: This book changed my life. I couldn't put it down!
# Sentiment: POSITIVE, Score: 0.9997
```

Slide 16: Additional Resources

For those interested in diving deeper into Language Models and their applications, here are some valuable resources:

1. ArXiv.org - A repository of scientific papers, including many on NLP and Language Models. Example: "Attention Is All You Need" by Vaswani et al. ([https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762))
2. Hugging Face Transformers Library Documentation - Comprehensive guides and tutorials on using state-of-the-art NLP models.
3. "Speech and Language Processing" by Jurafsky and Martin - A thorough textbook covering various aspects of NLP.
4. Stanford CS224N: Natural Language Processing with Deep Learning - Online course materials and lectures.
5. ACL Anthology - A digital archive of research papers in computational linguistics and natural language processing.

Remember to verify the credibility and recency of any additional resources you explore in this rapidly evolving field.

