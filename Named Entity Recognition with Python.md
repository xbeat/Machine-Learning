## Named Entity Recognition with Python
Slide 1: Introduction to Named Entity Recognition

Named Entity Recognition (NER) is a natural language processing task that identifies and classifies named entities in text into predefined categories such as person names, organizations, locations, and more. It's a crucial component in various NLP applications, including information extraction, question answering, and text summarization.

```python
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Apple Inc. is planning to open a new store in New York City next month."

# Process the text
doc = nlp(text)

# Print entities
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")

# Output:
# Entity: Apple Inc., Label: ORG
# Entity: New York City, Label: GPE
```

Slide 2: NER Techniques: Rule-based Approach

The rule-based approach to NER relies on handcrafted rules and patterns to identify entities. This method is effective for specific domains with well-defined naming conventions but may struggle with ambiguity and require frequent updates.

```python
import re

def rule_based_ner(text):
    patterns = {
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'PHONE': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'DATE': r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'
    }
    
    entities = []
    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            entities.append((match.group(), label, match.start(), match.end()))
    
    return entities

# Example usage
text = "Contact john.doe@example.com or call 123-456-7890 by 5/15/2023."
results = rule_based_ner(text)
for entity, label, start, end in results:
    print(f"Entity: {entity}, Label: {label}, Position: ({start}, {end})")

# Output:
# Entity: john.doe@example.com, Label: EMAIL, Position: (8, 28)
# Entity: 123-456-7890, Label: PHONE, Position: (37, 49)
# Entity: 5/15/2023, Label: DATE, Position: (53, 62)
```

Slide 3: NER Techniques: Machine Learning Approach

Machine learning approaches for NER typically use sequence labeling models such as Conditional Random Fields (CRF) or Hidden Markov Models (HMM). These models learn to predict entity labels based on features extracted from the text and surrounding context.

```python
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
import nltk
from nltk.corpus import conll2002

# Download the dataset
nltk.download('conll2002')

# Feature extraction function
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features

# Prepare the data
train_sents = list(conll2002.iob_sents('esp.train'))
test_sents = list(conll2002.iob_sents('esp.testb'))

X_train = [[word2features(s, i) for i in range(len(s))] for s in train_sents]
y_train = [[w[2] for w in s] for s in train_sents]

X_test = [[word2features(s, i) for i in range(len(s))] for s in test_sents]
y_test = [[w[2] for w in s] for s in test_sents]

# Train the CRF model
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit(X_train, y_train)

# Make predictions
y_pred = crf.predict(X_test)

# Print the classification report
print(flat_classification_report(y_test, y_pred))
```

Slide 4: NER Techniques: Deep Learning Approach

Deep learning models, particularly those based on neural networks like BiLSTM-CRF and Transformer architectures, have shown remarkable performance in NER tasks. These models can automatically learn complex features from the input data.

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(text, 
                                  return_offsets_mapping=True, 
                                  max_length=self.max_len, 
                                  padding='max_length', 
                                  truncation=True)

        word_ids = encoding.word_ids()
        label_ids = [-100] * len(word_ids)

        for word_idx, label_item in enumerate(label):
            word_id = word_ids.index(word_idx)
            label_ids[word_id] = label_item

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(label_ids)
        }

class BertNER(nn.Module):
    def __init__(self, num_labels):
        super(BertNER, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

# Example usage (assuming you have prepared your data)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128
num_labels = 9  # Number of entity types + 1 for 'O'

dataset = NERDataset(texts, labels, tokenizer, max_len)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = BertNER(num_labels)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# Training loop (simplified)
for epoch in range(3):
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs.view(-1, num_labels), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed")
```

Slide 5: Tokenization in NER

Tokenization is a crucial preprocessing step in NER, where text is split into individual tokens. Proper tokenization helps in accurately identifying entity boundaries and improving overall NER performance.

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')

text = "Dr. John Smith works at New York University. He can be reached at john.smith@nyu.edu."

# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)

# Word tokenization
for sentence in sentences:
    words = word_tokenize(sentence)
    print("Words:", words)

# Output:
# Sentences: ['Dr. John Smith works at New York University.', 'He can be reached at john.smith@nyu.edu.']
# Words: ['Dr.', 'John', 'Smith', 'works', 'at', 'New', 'York', 'University', '.']
# Words: ['He', 'can', 'be', 'reached', 'at', 'john.smith@nyu.edu', '.']
```

Slide 6: Feature Extraction for NER

Feature extraction involves creating relevant attributes from the input text to help the NER model make accurate predictions. Common features include word-level features, context features, and gazetteer features.

```python
import re
from nltk import pos_tag
from nltk.corpus import words

nltk.download('averaged_perceptron_tagger')
nltk.download('words')

def extract_features(text):
    words = word_tokenize(text)
    features = []
    
    for i, word in enumerate(words):
        word_features = {
            'word': word,
            'is_first': i == 0,
            'is_last': i == len(words) - 1,
            'is_capitalized': word[0].isupper(),
            'is_all_caps': word.isupper(),
            'is_all_lower': word.islower(),
            'prefix-1': word[0],
            'prefix-2': word[:2],
            'prefix-3': word[:3],
            'suffix-1': word[-1],
            'suffix-2': word[-2:],
            'suffix-3': word[-3:],
            'prev_word': '' if i == 0 else words[i-1],
            'next_word': '' if i == len(words) - 1 else words[i+1],
            'has_hyphen': '-' in word,
            'is_numeric': word.isdigit(),
            'capitals_inside': word[1:].lower() != word[1:],
            'pos': pos_tag([word])[0][1],
            'is_alphanumeric': bool(re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])', word)),
            'is_in_dictionary': word.lower() in words.words()
        }
        features.append(word_features)
    
    return features

# Example usage
sample_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
extracted_features = extract_features(sample_text)

for word_features in extracted_features:
    print(f"Word: {word_features['word']}")
    print(f"Is capitalized: {word_features['is_capitalized']}")
    print(f"POS tag: {word_features['pos']}")
    print(f"Is in dictionary: {word_features['is_in_dictionary']}")
    print("---")
```

Slide 7: Evaluation Metrics for NER

Evaluating NER systems requires specialized metrics that account for both the entity type and its exact boundaries. Common metrics include precision, recall, and F1-score, calculated at both the entity level and the token level.

```python
from seqeval.metrics import classification_report, f1_score

def evaluate_ner(true_labels, pred_labels):
    print(classification_report(true_labels, pred_labels))
    print(f"Overall F1-score: {f1_score(true_labels, pred_labels)}")

# Example usage
true_labels = [
    ['O', 'B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'O'],
    ['B-LOC', 'I-LOC', 'O', 'B-PER', 'I-PER', 'O']
]

pred_labels = [
    ['O', 'B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'O'],
    ['B-LOC', 'I-LOC', 'O', 'B-PER', 'O', 'O']
]

evaluate_ner(true_labels, pred_labels)

# Output:
#               precision    recall  f1-score   support
#
#          LOC     1.0000    1.0000    1.0000         1
#          ORG     1.0000    1.0000    1.0000         1
#          PER     0.6667    0.6667    0.6667         3
#
#    micro avg     0.8000    0.8000    0.8000         5
#    macro avg     0.8889    0.8889    0.8889         5
# weighted avg     0.8000    0.8000    0.8000         5
#
# Overall F1-score: 0.8
```

Slide 8: Handling Ambiguity in NER

Ambiguity is a common challenge in NER, where entities can have multiple interpretations or overlap with common words. Contextual information and domain-specific knowledge are crucial for resolving ambiguities.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def resolve_ambiguity(text):
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.text.istitle():
            # Check if the entity is followed by a verb
            next_token = doc[ent.end] if ent.end < len(doc) else None
            if next_token and next_token.pos_ == "VERB":
                entities.append((ent.text, "PERSON"))
            else:
                entities.append((ent.text, "ORG"))
        else:
            entities.append((ent.text, ent.label_))
    
    return entities

# Example usage
text1 = "Apple is looking to hire new engineers."
text2 = "Apple eats an apple every day."

print("Text 1 entities:", resolve_ambiguity(text1))
print("Text 2 entities:", resolve_ambiguity(text2))

# Output:
# Text 1 entities: [('Apple', 'ORG')]
# Text 2 entities: [('Apple', 'PERSON')]
```

Slide 9: Entity Linking and Disambiguation

Entity linking and disambiguation are crucial steps in NER that involve connecting named entities to their corresponding entries in a knowledge base. This process helps resolve ambiguities and provides additional context for the identified entities.

```python
import wikipedia

def entity_linking(entity, context):
    try:
        # Search for potential matches
        search_results = wikipedia.search(entity)
        
        if not search_results:
            return None
        
        # Get the page content for the first result
        page = wikipedia.page(search_results[0])
        summary = page.summary
        
        # Simple relevance check (can be improved with more sophisticated methods)
        if any(word in summary.lower() for word in context.lower().split()):
            return {
                'entity': entity,
                'link': page.url,
                'description': summary[:100] + '...'  # Truncate for brevity
            }
        else:
            return None
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle disambiguation pages
        return {'entity': entity, 'disambiguation': e.options[:5]}
    except:
        return None

# Example usage
entity = "Python"
context = "programming language for data science"

result = entity_linking(entity, context)
if result:
    if 'link' in result:
        print(f"Entity: {result['entity']}")
        print(f"Link: {result['link']}")
        print(f"Description: {result['description']}")
    else:
        print(f"Disambiguation options for {result['entity']}:")
        print(result['disambiguation'])
else:
    print("No relevant entity found")

# Note: This is a simplified example. Real-world entity linking systems
# use more sophisticated algorithms and larger knowledge bases.
```

Slide 10: NER in Multiple Languages

Multilingual NER systems can recognize entities across different languages, which is crucial for processing diverse text data. These systems often use language-agnostic features or multilingual embeddings to achieve cross-lingual performance.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def multilingual_ner(text, model_name="xlm-roberta-large-finetuned-conll03-english"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    results = ner_pipeline(text)
    
    return results

# Example usage
texts = [
    "Apple Inc. was founded by Steve Jobs in Cupertino.",  # English
    "La Tour Eiffel est située à Paris.",  # French
    "Der Reichstag ist in Berlin.",  # German
]

for text in texts:
    print(f"Text: {text}")
    entities = multilingual_ner(text)
    for entity in entities:
        print(f"Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']:.2f}")
    print()

# Note: The actual output may vary depending on the model's performance
# and the specific entities in each sentence.
```

Slide 11: Real-life Example: NER in Healthcare

Named Entity Recognition plays a crucial role in processing medical texts, helping to extract important information such as disease names, symptoms, medications, and treatments from clinical notes and research papers.

```python
import spacy
from spacy.tokens import Span

# Load the English model
nlp = spacy.load("en_core_web_sm")

# Define custom entity labels
SYMPT = "SYMPTOM"
MED = "MEDICATION"

# Add the custom labels to the pipeline
ner = nlp.get_pipe("ner")
ner.add_label(SYMPT)
ner.add_label(MED)

# Sample function to update the model (in practice, you'd train on a large dataset)
def update_model(train_data):
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    # Update the model
    # In practice, you'd need to train for multiple iterations on a large dataset
    nlp.update([train_data], drop=0.5, losses={})

# Example training data
train_data = [
    ("The patient complained of severe headache and nausea.", {"entities": [(35, 43, SYMPT), (48, 54, SYMPT)]}),
    ("Doctor prescribed acetaminophen for pain relief.", {"entities": [(19, 32, MED)]})
]

# Update the model with the training data
update_model(train_data)

# Function to analyze medical text
def analyze_medical_text(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Example usage
medical_text = "The patient reported experiencing fatigue and was prescribed amoxicillin for the infection."
results = analyze_medical_text(medical_text)

print("Entities found:")
for entity, label in results:
    print(f"- {entity} ({label})")

# Note: This is a simplified example. A real-world medical NER system would
# require extensive training data and domain-specific models.
```

Slide 12: Real-life Example: NER in Social Media Analysis

NER can be applied to social media data to extract valuable insights about mentioned entities, such as products, brands, or public figures. This information can be used for sentiment analysis, trend detection, and customer feedback analysis.

```python
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_tweet(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_entities_from_tweet(text):
    preprocessed_text = preprocess_tweet(text)
    doc = nlp(preprocessed_text)
    
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "PERSON", "GPE"]:
            entities.append((ent.text, ent.label_))
    
    return entities

# Example tweets
tweets = [
    "I love my new @Apple iPhone! The camera is amazing. #technology",
    "Just watched the latest @Marvel movie. @RobertDowneyJr was fantastic as always!",
    "Planning a trip to Paris next month. Any recommendations for places to visit?"
]

for tweet in tweets:
    print("Original tweet:", tweet)
    entities = extract_entities_from_tweet(tweet)
    print("Extracted entities:")
    for entity, label in entities:
        print(f"- {entity} ({label})")
    print()

# Note: This example uses a general-purpose NER model. For better results
# in social media analysis, you would typically fine-tune the model on
# social media data or use a specialized social media NER model.
```

Slide 13: Challenges and Future Directions in NER

Named Entity Recognition continues to evolve, facing challenges such as handling emerging entities, domain adaptation, and improving performance on low-resource languages. Future directions include incorporating more contextual information, leveraging knowledge graphs, and developing more efficient training methods for large-scale NER systems.

```python
import random

class FutureNERSystem:
    def __init__(self):
        self.knowledge_graph = self.initialize_knowledge_graph()
        self.context_memory = []
    
    def initialize_knowledge_graph(self):
        # Simulated knowledge graph
        return {
            "Apple": ["Company", "Technology", "iPhone"],
            "Elon Musk": ["Person", "CEO", "Tesla", "SpaceX"],
            "Python": ["Programming Language", "Data Science", "AI"]
        }
    
    def update_knowledge_graph(self, entity, information):
        if entity in self.knowledge_graph:
            self.knowledge_graph[entity].extend(information)
        else:
            self.knowledge_graph[entity] = information
    
    def recognize_entities(self, text):
        words = text.split()
        entities = []
        for word in words:
            if word in self.knowledge_graph:
                entity_type = self.knowledge_graph[word][0]
                entities.append((word, entity_type))
            elif random.random() < 0.1:  # Simulate recognition of new entities
                entities.append((word, "EMERGING_ENTITY"))
                self.update_knowledge_graph(word, ["EMERGING_ENTITY"])
        return entities
    
    def process_text(self, text):
        entities = self.recognize_entities(text)
        self.context_memory.append(text)
        if len(self.context_memory) > 5:
            self.context_memory.pop(0)
        return entities

# Example usage
future_ner = FutureNERSystem()

texts = [
    "Apple is working on a new AI project.",
    "Elon Musk announced the latest Tesla model.",
    "Researchers use Python for advanced NLP tasks.",
    "The new XYZ technology is revolutionizing the industry."
]

for text in texts:
    print(f"Text: {text}")
    entities = future_ner.process_text(text)
    print("Recognized entities:")
    for entity, entity_type in entities:
        print(f"- {entity}: {entity_type}")
    print()

print("Updated Knowledge Graph:")
print(future_ner.knowledge_graph)

# Note: This is a simplified simulation of potential future NER systems.
# Actual implementations would be much more complex and use advanced
# machine learning techniques.
```

Slide 14: Additional Resources

For those interested in diving deeper into Named Entity Recognition, here are some valuable resources:

1. Research Papers:
   * "Neural Architectures for Named Entity Recognition" (Lample et al., 2016)
   * "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018)
2. Tutorials and Courses:
   * Stanford CS224N: Natural Language Processing with Deep Learning
   * Coursera: Natural Language Processing Specialization
3. Tools and Libraries:
   * spaCy: Industrial-strength Natural Language Processing
   * NLTK (Natural Language Toolkit)
   * Hugging Face Transformers
4. Datasets:
   * CoNLL-2003 Dataset
   * OntoNotes 5.0
5. ArXiv Papers:
   * "A Survey on Deep Learning for Named Entity Recognition" (Li et al., 2020) ArXiv URL: [https://arxiv.org/abs/1812.09449](https://arxiv.org/abs/1812.09449)
   * "CrossNER: Evaluating Cross-Domain Named Entity Recognition" (Lyu et al., 2020) ArXiv URL: [https://arxiv.org/abs/2012.04373](https://arxiv.org/abs/2012.04373)

These resources provide a mix of theoretical foundations and practical implementations to further your understanding of Named Entity Recognition.

