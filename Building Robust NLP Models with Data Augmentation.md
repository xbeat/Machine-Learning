## Building Robust NLP Models with Data Augmentation
Slide 1: Introduction to Data Augmentation in NLP

Data augmentation is a technique used to increase the diversity and size of training data by creating modified versions of existing data. In Natural Language Processing (NLP), this helps build more robust models that can generalize better to unseen data.

```python
import nlpaug.augmenter.word as naw

# Example of a simple augmentation
augmenter = naw.SynonymAug(aug_src='wordnet')
text = "The quick brown fox jumps over the lazy dog"
augmented_text = augmenter.augment(text)
print(augmented_text)
```

Slide 2: Why Data Augmentation?

Data augmentation helps overcome common challenges in NLP:

1. Limited labeled data
2. Imbalanced datasets
3. Overfitting
4. Improving model generalization

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load a sample dataset
df = pd.read_csv('sentiment_data.csv')
X = df['text']
y = df['sentiment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
```

Slide 3: Types of Data Augmentation in NLP

1. Lexical Substitution
2. Back-Translation
3. Text Generation
4. Noise Injection
5. Sentence Permutation

```python
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

text = "I love this movie, it's amazing!"

# Lexical Substitution
aug_synonym = naw.SynonymAug(aug_src='wordnet')
print("Synonym:", aug_synonym.augment(text))

# Back-Translation
aug_back_translation = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en')
print("Back-Translation:", aug_back_translation.augment(text))

# Sentence Permutation
aug_sentence = nas.ContextualWordEmbsForSentenceAug(model_path='distilbert-base-uncased')
print("Sentence Augmentation:", aug_sentence.augment(text))
```

Slide 4: Lexical Substitution

Lexical substitution involves replacing words with their synonyms, antonyms, or related words. This technique helps the model learn semantic relationships and improves vocabulary coverage.

```python
import nlpaug.augmenter.word as naw

text = "The cat is sleeping on the couch"

# Synonym replacement
aug_synonym = naw.SynonymAug(aug_src='wordnet')
print("Synonym:", aug_synonym.augment(text))

# Antonym replacement
aug_antonym = naw.AntonymAug()
print("Antonym:", aug_antonym.augment(text))

# Word embedding replacement
aug_w2v = naw.WordEmbsAug(model_type='word2vec', model_path='./word2vec.bin')
print("Word Embedding:", aug_w2v.augment(text))
```

Slide 5: Back-Translation

Back-translation involves translating text to another language and then back to the original language. This technique introduces diverse phrasing and sentence structures.

```python
from transformers import MarianMTModel, MarianTokenizer

def back_translate(text, source_lang="en", target_lang="fr"):
    # Load models
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    # Translate to target language
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
    
    # Translate back to source language
    model_name = f'Helsinki-NLP/opus-mt-{target_lang}-{source_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    back_translated = model.generate(**tokenizer(tgt_text, return_tensors="pt", padding=True))
    back_text = [tokenizer.decode(t, skip_special_tokens=True) for t in back_translated][0]
    
    return back_text

original_text = "The weather is beautiful today"
augmented_text = back_translate(original_text)
print(f"Original: {original_text}")
print(f"Augmented: {augmented_text}")
```

Slide 6: Text Generation

Text generation involves creating new text based on existing data. This can be done using language models or rule-based systems to expand the dataset with synthetic examples.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length=50):
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompt = "The restaurant was"
generated_text = generate_text(prompt)
print(f"Generated text: {generated_text}")
```

Slide 7: Noise Injection

Noise injection involves adding random perturbations to the text, such as spelling mistakes, character swaps, or word deletions. This technique helps create a more robust model that can handle imperfect input.

```python
import random
import string

def add_noise(text, p=0.1):
    words = text.split()
    noisy_words = []
    
    for word in words:
        if random.random() < p:
            noise_type = random.choice(['swap', 'delete', 'insert'])
            if noise_type == 'swap' and len(word) > 1:
                i, j = random.sample(range(len(word)), 2)
                word = list(word)
                word[i], word[j] = word[j], word[i]
                word = ''.join(word)
            elif noise_type == 'delete' and len(word) > 1:
                i = random.randint(0, len(word) - 1)
                word = word[:i] + word[i+1:]
            elif noise_type == 'insert':
                i = random.randint(0, len(word))
                char = random.choice(string.ascii_lowercase)
                word = word[:i] + char + word[i:]
        noisy_words.append(word)
    
    return ' '.join(noisy_words)

original_text = "The quick brown fox jumps over the lazy dog"
noisy_text = add_noise(original_text)
print(f"Original: {original_text}")
print(f"Noisy: {noisy_text}")
```

Slide 8: Sentence Permutation

Sentence permutation involves changing the order of sentences in a document or creating new combinations of sentences. This technique helps the model learn different discourse structures and improve coherence understanding.

```python
import random

def permute_sentences(text):
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    random.shuffle(sentences)
    return '. '.join(sentences) + '.'

original_text = "I went to the store. It was a sunny day. I bought some groceries. The cashier was friendly."
permuted_text = permute_sentences(original_text)
print(f"Original: {original_text}")
print(f"Permuted: {permuted_text}")
```

Slide 9: Implementing Data Augmentation in a Pipeline

Integrating data augmentation into your NLP pipeline involves applying augmentation techniques to your training data before model training.

```python
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import torch

def augment_data(texts, labels, aug_technique, aug_factor=2):
    augmented_texts, augmented_labels = [], []
    for text, label in zip(texts, labels):
        augmented_texts.append(text)
        augmented_labels.append(label)
        for _ in range(aug_factor - 1):
            aug_text = aug_technique(text)
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
    return augmented_texts, augmented_labels

# Assume we have texts and labels
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# Apply augmentation
X_train_aug, y_train_aug = augment_data(X_train, y_train, aug_technique=add_noise)

# Tokenize and create dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(X_train_aug, truncation=True, padding=True)
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(y_train_aug)
)

# Train model (simplified)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# ... (training loop)
```

Slide 10: Evaluating Augmentation Impact

It's crucial to evaluate the impact of data augmentation on your model's performance. Compare the model's performance with and without augmentation.

```python
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        inputs = tokenizer(X_test, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    return accuracy, report

# Assume we have trained two models: model_no_aug and model_with_aug

accuracy_no_aug, report_no_aug = evaluate_model(model_no_aug, X_test, y_test)
accuracy_with_aug, report_with_aug = evaluate_model(model_with_aug, X_test, y_test)

print(f"Accuracy without augmentation: {accuracy_no_aug}")
print(f"Accuracy with augmentation: {accuracy_with_aug}")
print("\nClassification Report (No Augmentation):")
print(report_no_aug)
print("\nClassification Report (With Augmentation):")
print(report_with_aug)
```

Slide 11: Real-life Example: Sentiment Analysis

Let's apply data augmentation to a sentiment analysis task using movie reviews.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import nlpaug.augmenter.word as naw
import torch

# Load data (assume we have a CSV with 'review' and 'sentiment' columns)
df = pd.read_csv('movie_reviews.csv')
X = df['review'].tolist()
y = df['sentiment'].tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Augmentation function
aug_synonym = naw.SynonymAug(aug_src='wordnet')

def augment_data(texts, labels, aug_factor=2):
    augmented_texts, augmented_labels = [], []
    for text, label in zip(texts, labels):
        augmented_texts.append(text)
        augmented_labels.append(label)
        for _ in range(aug_factor - 1):
            aug_text = aug_synonym.augment(text)
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
    return augmented_texts, augmented_labels

# Apply augmentation
X_train_aug, y_train_aug = augment_data(X_train, y_train)

# Tokenize and create datasets
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(X_train_aug, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(y_train_aug)
)

test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(test_encodings['input_ids']),
    torch.tensor(test_encodings['attention_mask']),
    torch.tensor(y_test)
)

# Train model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Evaluate
results = trainer.evaluate()
print(results)
```

Slide 12: Real-life Example: Named Entity Recognition (NER)

Let's apply data augmentation to a Named Entity Recognition task using news articles.

```python
import spacy
import random

nlp = spacy.load("en_core_web_sm")

train_data = [
    ("Apple Inc. is planning to open a new store in New York City.", {"entities": [(0, 9, "ORG"), (41, 54, "GPE")]}),
    ("Microsoft announced a partnership with OpenAI.", {"entities": [(0, 9, "ORG"), (37, 42, "ORG")]})
]

def augment_ner_data(text, entities):
    doc = nlp(text)
    augmented_text = []
    augmented_entities = []
    
    for token in doc:
        if random.random() < 0.1 and token.pos_ in ["NOUN", "VERB", "ADJ"]:
            synonyms = [syn.lower_ for syn in token._.synonyms]
            if synonyms:
                replacement = random.choice(synonyms)
                augmented_text.append(replacement)
            else:
                augmented_text.append(token.text)
        else:
            augmented_text.append(token.text)
    
    augmented_text = " ".join(augmented_text)
    
    for start, end, label in entities["entities"]:
        new_start = len(" ".join(augmented_text.split()[:start]))
        new_end = new_start + len(" ".join(augmented_text.split()[start:end]))
        augmented_entities.append((new_start, new_end, label))
    
    return augmented_text, {"entities": augmented_entities}

augmented_train_data = []
for text, annotations in train_data:
    augmented_train_data.append((text, annotations))
    for _ in range(2):  # Create 2 augmented examples for each original
        aug_text, aug_annotations = augment_ner_data(text, annotations)
        augmented_train_data.append((aug_text, aug_annotations))

print(f"Original dataset size: {len(train_data)}")
print(f"Augmented dataset size: {len(augmented_train_data)}")
print("\nSample augmented data:")
print(augmented_train_data[2])
```

Slide 13: Challenges and Considerations

When implementing data augmentation for NLP:

1. Preserve semantic meaning
2. Maintain label consistency
3. Balance augmentation techniques
4. Avoid introducing bias

```python
def check_augmentation_quality(original, augmented):
    original_doc = nlp(original)
    augmented_doc = nlp(augmented)
    
    # Check semantic similarity
    similarity = original_doc.similarity(augmented_doc)
    
    # Check label consistency (example for sentiment analysis)
    original_sentiment = original_doc.sentiment
    augmented_sentiment = augmented_doc.sentiment
    
    print(f"Semantic similarity: {similarity}")
    print(f"Original sentiment: {original_sentiment}")
    print(f"Augmented sentiment: {augmented_sentiment}")
    
    if similarity < 0.7 or abs(original_sentiment - augmented_sentiment) > 0.3:
        print("Warning: Augmentation may have altered meaning or label.")

original_text = "The movie was fantastic and I enjoyed every minute of it."
augmented_text = "The film was terrific and I relished each moment of it."

check_augmentation_quality(original_text, augmented_text)
```

Slide 14: Best Practices for NLP Data Augmentation

1. Experiment with multiple techniques
2. Use domain-specific augmentation when possible
3. Monitor impact on model performance
4. Regularly update augmentation strategies

```python
def augmentation_pipeline(text, techniques):
    augmented_texts = [text]
    for technique in techniques:
        new_text = technique(text)
        augmented_texts.append(new_text)
    return augmented_texts

# Example usage
techniques = [
    lambda x: add_noise(x, p=0.1),
    lambda x: back_translate(x, source_lang="en", target_lang="fr"),
    lambda x: aug_synonym.augment(x)
]

sample_text = "The weather is beautiful today"
augmented_samples = augmentation_pipeline(sample_text, techniques)

for i, sample in enumerate(augmented_samples):
    print(f"Sample {i}: {sample}")
```

Slide 15: Additional Resources

For further exploration of data augmentation in NLP:

1. "A Survey of Data Augmentation Approaches for NLP" (ArXiv:2105.03075) URL: [https://arxiv.org/abs/2105.03075](https://arxiv.org/abs/2105.03075)
2. "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks" (ArXiv:1901.11196) URL: [https://arxiv.org/abs/1901.11196](https://arxiv.org/abs/1901.11196)
3. "Data Augmentation Using Pre-trained Transformer Models" (ArXiv:2003.02245) URL: [https://arxiv.org/abs/2003.02245](https://arxiv.org/abs/2003.02245)

These resources provide in-depth discussions on various data augmentation techniques and their applications in NLP tasks.

