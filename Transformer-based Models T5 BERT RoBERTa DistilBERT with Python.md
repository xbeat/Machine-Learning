## Transformer-based Models T5, BERT, RoBERTa, DistilBERT with Python
Slide 1: Introduction to Transformer-based Models

Transformer-based models have revolutionized natural language processing. In this presentation, we'll explore T5, BERT, RoBERTa, and DistilBERT, understanding their architecture, use cases, and implementation details.

```python
import torch
from transformers import AutoModel, AutoTokenizer

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# Example usage
bert_tokenizer, bert_model = load_model('bert-base-uncased')
```

Slide 2: T5: Text-to-Text Transfer Transformer

T5 is a versatile model that frames all NLP tasks as text-to-text problems. It can handle various tasks like translation, summarization, and question-answering with a single architecture.

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

def summarize_text(text):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    
    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Example usage
text = "T5 is a powerful model that can perform various NLP tasks. It was introduced by Google in 2020."
print(summarize_text(text))
```

Slide 3: T5 Architecture and Pre-training

T5 uses a standard encoder-decoder transformer architecture. It's pre-trained on a large corpus of web text using a "masked language modeling" objective, where it learns to predict missing words in a sentence.

```python
import torch
import torch.nn as nn

class SimplifiedT5(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(SimplifiedT5, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_encoder_layers)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead), num_decoder_layers)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt):
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)
        memory = self.encoder(src_embed)
        output = self.decoder(tgt_embed, memory)
        return self.linear(output)

# Example instantiation
model = SimplifiedT5(vocab_size=30000, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
```

Slide 4: BERT: Bidirectional Encoder Representations from Transformers

BERT is a transformer-based model designed to pre-train deep bidirectional representations. It excels in tasks like sentence classification, named entity recognition, and question answering.

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

def classify_sentiment(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = "Positive" if probabilities[0][1] > probabilities[0][0] else "Negative"
    
    return sentiment

# Example usage
text = "I really enjoyed this movie. The plot was engaging and the actors were fantastic."
print(classify_sentiment(text))
```

Slide 5: BERT Pre-training Objectives

BERT uses two pre-training objectives: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). MLM randomly masks 15% of the tokens in the input, and the model learns to predict the original vocabulary id of the masked word. NSP trains the model to understand the relationship between sentences.

```python
import torch
import torch.nn as nn

class SimplifiedBERTPreTraining(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(SimplifiedBERTPreTraining, self).__init__()
        self.bert = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=12), num_layers=12)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        self.nsp_head = nn.Linear(hidden_size, 2)
    
    def forward(self, input_ids, masked_lm_labels, next_sentence_label):
        sequence_output = self.bert(input_ids)
        mlm_scores = self.mlm_head(sequence_output)
        nsp_score = self.nsp_head(sequence_output[:, 0, :])
        return mlm_scores, nsp_score

# Example instantiation
model = SimplifiedBERTPreTraining(vocab_size=30000, hidden_size=768)
```

Slide 6: RoBERTa: Robustly Optimized BERT Approach

RoBERTa is an optimized version of BERT that modifies key hyperparameters, removing the next-sentence pretraining objective and training with much larger mini-batches and learning rates.

```python
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

def predict_masked_word(text):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    
    # Replace a word with [MASK]
    text = text.replace('predict', tokenizer.mask_token)
    
    inputs = tokenizer(text, return_tensors='pt')
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    
    outputs = model(**inputs)
    logits = outputs.logits
    
    mask_token_logits = logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    
    for token in top_5_tokens:
        print(f"{tokenizer.decode([token])}: {torch.softmax(mask_token_logits, dim=1)[0, token].item():.3f}")

# Example usage
text = "The model will predict the masked word in this sentence."
predict_masked_word(text)
```

Slide 7: RoBERTa Training Improvements

RoBERTa introduces several improvements over BERT, including dynamic masking, full-sentences without NSP loss, larger mini-batches, and a larger byte-level BPE vocabulary.

```python
import torch
from torch.utils.data import DataLoader, Dataset

class DynamicMaskingDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length')
        input_ids = torch.tensor(encoding['input_ids'])
        attention_mask = torch.tensor(encoding['attention_mask'])
        
        # Dynamic masking
        mask = torch.rand(input_ids.shape) < 0.15
        mask = mask & (input_ids != self.tokenizer.cls_token_id) & (input_ids != self.tokenizer.sep_token_id)
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask] = self.tokenizer.mask_token_id
        
        return {
            'input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids
        }

# Example usage
texts = ["This is an example sentence.", "Another sentence for demonstration."]
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
dataset = DynamicMaskingDataset(texts, tokenizer, max_length=128)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

Slide 8: DistilBERT: Distilled Version of BERT

DistilBERT is a smaller, faster, cheaper, and lighter version of BERT that retains 97% of BERT's language understanding capabilities while being 40% smaller and 60% faster.

```python
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

def answer_question(context, question):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
    
    inputs = tokenizer(question, context, return_tensors='pt')
    outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    
    return answer

# Example usage
context = "DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT's performances as measured on the GLUE language understanding benchmark."
question = "What is DistilBERT?"
print(answer_question(context, question))
```

Slide 9: DistilBERT Architecture and Training

DistilBERT uses knowledge distillation during the pre-training phase to reduce the size of a BERT model. It is trained on the same data as BERT, using the output of the BERT model as soft targets.

```python
import torch
import torch.nn as nn

class SimplifiedDistilBERT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(SimplifiedDistilBERT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=12), num_layers=num_layers)
        self.classifier = nn.Linear(hidden_size, 2)  # For binary classification
    
    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        hidden_states = self.transformer(embeddings)
        return self.classifier(hidden_states[:, 0, :])  # Use [CLS] token for classification

# Example instantiation
model = SimplifiedDistilBERT(vocab_size=30000, hidden_size=768, num_layers=6)

# Simulating knowledge distillation
teacher_model = SimplifiedBERT(vocab_size=30000, hidden_size=768, num_layers=12)
distillation_loss = nn.KLDivLoss(reduction='batchmean')

def train_step(input_ids, labels):
    student_logits = model(input_ids)
    with torch.no_grad():
        teacher_logits = teacher_model(input_ids)
    
    # Compute distillation loss
    loss = distillation_loss(
        torch.log_softmax(student_logits / temperature, dim=-1),
        torch.softmax(teacher_logits / temperature, dim=-1)
    ) * (temperature ** 2)
    
    # Add task-specific loss (e.g., cross-entropy for classification)
    task_loss = nn.CrossEntropyLoss()(student_logits, labels)
    total_loss = 0.5 * loss + 0.5 * task_loss
    
    return total_loss

# Note: This is a simplified example and doesn't include the full training loop
```

Slide 10: Real-life Example: Sentiment Analysis with BERT

Let's use BERT for sentiment analysis on movie reviews, a common task in natural language processing.

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

def analyze_sentiment(review):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    inputs = tokenizer(review, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = "Positive" if probabilities[0][1] > probabilities[0][0] else "Negative"
    confidence = max(probabilities[0]).item()
    
    return sentiment, confidence

# Example usage
reviews = [
    "This movie was absolutely fantastic! The plot was engaging and the acting was superb.",
    "I was disappointed with this film. The storyline was confusing and the characters were poorly developed.",
    "An average movie. It had its moments, but overall it was nothing special."
]

for review in reviews:
    sentiment, confidence = analyze_sentiment(review)
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})\n")
```

Slide 11: Real-life Example: Text Generation with T5

Let's use T5 for text generation tasks such as translation and summarization.

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

def generate_text(input_text, task):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    
    input_text = f"{task}: {input_text}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    
    outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
text = "The quick brown fox jumps over the lazy dog. This sentence is often used as a pangram in typography."

print("Original text:", text)
print("\nTranslation to French:")
print(generate_text(text, "translate English to French"))
print("\nSummarization:")
print(generate_text(text, "summarize"))
```

Slide 12: Comparing Model Sizes and Performance

Let's compare the model sizes and relative performance of BERT, RoBERTa, and DistilBERT on the GLUE benchmark.

```python
import matplotlib.pyplot as plt
import numpy as np

models = ['BERT-base', 'RoBERTa-base', 'DistilBERT']
params = [110, 125, 66]  # Millions of parameters
glue_scores = [80.5, 83.2, 77.0]  # Average GLUE scores

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

x = np.arange(len(models))
width = 0.35

rects1 = ax1.bar(x - width/2, params, width, label='Parameters (M)', color='skyblue')
rects2 = ax2.bar(x + width/2, glue_scores, width, label='GLUE Score', color='lightgreen')

ax1.set_xlabel('Models')
ax1.set_ylabel('Parameters (Millions)')
ax2.set_ylabel('Average GLUE Score')
ax1.set_title('Model Size vs Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(models)

fig.tight_layout()
plt.show()
```

Slide 13: T5 vs BERT: Task-Specific vs General-Purpose

T5 and BERT represent different approaches to transformer-based models. T5 is designed for versatility across various NLP tasks, while BERT focuses on creating powerful pre-trained representations.

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertForSequenceClassification, BertTokenizer

def compare_t5_bert(text):
    # T5 for text generation
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
    t5_input = t5_tokenizer(f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
    t5_summary = t5_model.generate(t5_input.input_ids, max_length=150, num_beams=4, early_stopping=True)
    t5_output = t5_tokenizer.decode(t5_summary[0], skip_special_tokens=True)
    
    # BERT for sentiment analysis
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    bert_input = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    bert_output = bert_model(**bert_input)
    sentiment = "Positive" if bert_output.logits[0][1] > bert_output.logits[0][0] else "Negative"
    
    return f"T5 Summary: {t5_output}\nBERT Sentiment: {sentiment}"

# Example usage
text = "This movie was fantastic. The plot was engaging and the acting was superb. I highly recommend it."
print(compare_t5_bert(text))
```

Slide 14: Future Directions and Ongoing Research

The field of transformer-based models is rapidly evolving. Current research focuses on making models more efficient, interpretable, and capable of handling longer sequences.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulating the trend of model sizes over time
years = np.array([2018, 2019, 2020, 2021, 2022, 2023])
model_sizes = np.array([0.11, 0.34, 175, 1000, 540, 1800])  # In billions of parameters

plt.figure(figsize=(10, 6))
plt.semilogy(years, model_sizes, marker='o')
plt.title('Trend of Transformer Model Sizes')
plt.xlabel('Year')
plt.ylabel('Model Size (Billions of Parameters)')
plt.grid(True)
plt.show()

# Note: This plot shows the exponential growth in model sizes,
# highlighting the need for more efficient architectures.
```

Slide 15: Additional Resources

For more in-depth information on these models, refer to the following research papers:

1. T5: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" ([https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683))
2. BERT: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" ([https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805))
3. RoBERTa: "RoBERTa: A Robustly Optimized BERT Pretraining Approach" ([https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692))
4. DistilBERT: "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter" ([https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108))

These papers provide detailed insights into the architecture, training procedures, and performance of each model.

