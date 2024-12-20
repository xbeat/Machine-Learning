## BERT for Question Answering
Slide 1: Introduction to BERT for Question Answering

BERT (Bidirectional Encoder Representations from Transformers) is a powerful language model that has revolutionized natural language processing tasks, including question answering. This presentation will explore how BERT can be used for question answering tasks, focusing on the AutoModelForQuestionAnswering with the "bert-large-uncased-whole-word-masking-finetuned-squad" configuration.

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Model loaded: {model_name}")
print(f"Model parameters: {model.num_parameters():,}")
```

Slide 2: Understanding BERT's Architecture

BERT's architecture is based on the Transformer model, which uses self-attention mechanisms to process input sequences. The "large" version of BERT has more parameters, allowing it to capture more complex language patterns. The "uncased" aspect means it treats uppercase and lowercase letters the same, making it more robust to variations in capitalization.

```python
import torch

# Create a simple input
input_text = "BERT is a powerful language model."
inputs = tokenizer(input_text, return_tensors="pt")

# Get the hidden states
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

hidden_states = outputs.hidden_states

print(f"Number of hidden states: {len(hidden_states)}")
print(f"Shape of last hidden state: {hidden_states[-1].shape}")
```

Slide 3: Whole Word Masking

Whole word masking is a training technique used in this BERT model. Instead of masking individual subword tokens, entire words are masked during pre-training. This helps the model learn more coherent and meaningful representations of words in context.

```python
import random

def mask_whole_word(text, mask_token="[MASK]"):
    words = text.split()
    mask_index = random.randint(0, len(words) - 1)
    words[mask_index] = mask_token
    return " ".join(words)

original_text = "The quick brown fox jumps over the lazy dog"
masked_text = mask_whole_word(original_text)

print(f"Original: {original_text}")
print(f"Masked:   {masked_text}")
```

Slide 4: Fine-tuning on SQuAD

The model is fine-tuned on the Stanford Question Answering Dataset (SQuAD), which consists of questions posed on a set of Wikipedia articles. This fine-tuning process adapts the pre-trained BERT model to the specific task of question answering.

```python
def simulate_squad_fine_tuning(model, tokenizer, context, question, answer):
    inputs = tokenizer(question, context, return_tensors="pt")
    start = context.index(answer)
    end = start + len(answer)
    inputs["start_positions"] = torch.tensor([start])
    inputs["end_positions"] = torch.tensor([end])
    
    loss = model(**inputs).loss
    print(f"Fine-tuning loss: {loss.item():.4f}")

context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris."
question = "Where is the Eiffel Tower located?"
answer = "Paris"

simulate_squad_fine_tuning(model, tokenizer, context, question, answer)
```

Slide 5: Tokenization Process

Before processing text with BERT, it needs to be tokenized. The tokenizer breaks down the input text into subword tokens that the model can understand. This process is crucial for handling out-of-vocabulary words and maintaining a manageable vocabulary size.

```python
text = "The AutoModelForQuestionAnswering uses sophisticated NLP techniques."
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Original text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)
```

Slide 6: Encoding Questions and Contexts

To perform question answering, we need to encode both the question and the context (passage) together. The model uses special tokens to distinguish between the question and context parts of the input.

```python
question = "What is the capital of France?"
context = "France is a country in Western Europe. Its capital is Paris."

inputs = tokenizer(question, context, return_tensors="pt")

print("Input IDs shape:", inputs["input_ids"].shape)
print("Attention mask shape:", inputs["attention_mask"].shape)
print("Token type IDs shape:", inputs["token_type_ids"].shape)
```

Slide 7: Model Inference

During inference, the model predicts the start and end positions of the answer within the context. These positions correspond to token indices in the input sequence.

```python
outputs = model(**inputs)

start_scores = outputs.start_logits
end_scores = outputs.end_logits

start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)

answer_tokens = inputs["input_ids"][0][start_index:end_index+1]
answer = tokenizer.decode(answer_tokens)

print(f"Question: {question}")
print(f"Predicted answer: {answer}")
```

Slide 8: Handling No-Answer Scenarios

Sometimes, the question might not have an answer in the given context. The model can be configured to handle such scenarios by predicting special "no answer" tokens.

```python
def has_answer(start_scores, end_scores, threshold=0):
    no_answer_score = start_scores[0][0] + end_scores[0][0]
    best_score = torch.max(start_scores) + torch.max(end_scores)
    return best_score > no_answer_score + threshold

question = "What is the population of Mars?"
context = "Mars is the fourth planet from the Sun in our solar system."

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

if has_answer(outputs.start_logits, outputs.end_logits):
    print("The model predicts an answer exists.")
else:
    print("The model predicts no answer in the given context.")
```

Slide 9: Confidence Scores

The model's confidence in its predictions can be assessed using the logits (scores) for the start and end positions. Higher scores indicate greater confidence in the predicted answer span.

```python
def get_confidence_score(start_scores, end_scores):
    start_probs = torch.softmax(start_scores, dim=1)
    end_probs = torch.softmax(end_scores, dim=1)
    return (torch.max(start_probs) * torch.max(end_probs)).item()

question = "Who founded Microsoft?"
context = "Microsoft was founded by Bill Gates and Paul Allen in 1975."

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

confidence = get_confidence_score(outputs.start_logits, outputs.end_logits)
print(f"Model confidence: {confidence:.4f}")
```

Slide 10: Handling Long Contexts

BERT has a maximum input length limitation (typically 512 tokens). For longer contexts, we need to implement a sliding window approach to process the entire text.

```python
def answer_question_long_context(question, context, max_length=512, stride=128):
    inputs = tokenizer(question, context, return_overflowing_tokens=True,
                       max_length=max_length, stride=stride, return_tensors="pt")
    
    all_scores = []
    for i in range(len(inputs["input_ids"])):
        outputs = model(input_ids=inputs["input_ids"][i].unsqueeze(0),
                        attention_mask=inputs["attention_mask"][i].unsqueeze(0))
        all_scores.append((outputs.start_logits, outputs.end_logits))
    
    # Process scores and find the best answer (implementation details omitted for brevity)
    # ...

    return best_answer

long_context = "..." # A very long text
question = "What is the main topic of this text?"

answer = answer_question_long_context(question, long_context)
print(f"Answer: {answer}")
```

Slide 11: Real-Life Example: Customer Support Chatbot

A customer support chatbot can use BERT for question answering to provide accurate responses based on a knowledge base.

```python
knowledge_base = """
Our return policy allows customers to return items within 30 days of purchase.
Shipping is free for orders over $50. Standard shipping takes 3-5 business days.
We offer a 1-year warranty on all electronic devices.
"""

def chatbot_response(user_question):
    inputs = tokenizer(user_question, knowledge_base, return_tensors="pt")
    outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)
    answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end+1])
    
    return answer

user_question = "What is your return policy?"
response = chatbot_response(user_question)
print(f"User: {user_question}")
print(f"Chatbot: {response}")
```

Slide 12: Real-Life Example: Document Summarization

BERT can be used to generate extractive summaries by answering questions about the main points of a document.

```python
def extract_summary(document):
    summary_questions = [
        "What is the main topic of this document?",
        "What are the key points discussed?",
        "What is the conclusion of the document?"
    ]
    
    summary = []
    for question in summary_questions:
        inputs = tokenizer(question, document, return_tensors="pt")
        outputs = model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)
        answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end+1])
        summary.append(answer)
    
    return " ".join(summary)

document = "..." # A long document text
summary = extract_summary(document)
print("Document Summary:", summary)
```

Slide 13: Limitations and Considerations

While BERT is powerful for question answering, it has limitations:

1. Maximum input length restriction (typically 512 tokens)
2. Computationally intensive, requiring significant resources
3. May struggle with questions requiring external knowledge or complex reasoning
4. Performance depends on the quality and coverage of the training data

```python
def demonstrate_input_length_limitation():
    max_length = tokenizer.model_max_length
    long_text = "a" * (max_length * 2)  # Text twice as long as the maximum length
    
    inputs = tokenizer(long_text, truncation=True, return_tensors="pt")
    
    print(f"Original text length: {len(long_text)}")
    print(f"Truncated input length: {inputs['input_ids'].shape[1]}")
    print(f"Truncation occurred: {len(long_text) > inputs['input_ids'].shape[1]}")

demonstrate_input_length_limitation()
```

Slide 14: Future Directions and Improvements

Ongoing research in NLP aims to address BERT's limitations:

1. Longer context models (e.g., Longformer, BigBird)
2. More efficient architectures (e.g., ALBERT, DistilBERT)
3. Integration with external knowledge bases
4. Multi-task learning for improved generalization

```python
def simulate_longer_context_model(text, max_length=4096):
    # Simulating a model with longer context support
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    
    print(f"Original text length: {len(text)}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Each chunk length: {len(chunks[0])}")

long_text = "a" * 10000  # A very long text
simulate_longer_context_model(long_text)
```

Slide 15: Additional Resources

For further exploration of BERT and question answering:

1. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding arXiv:1810.04805 \[cs.CL\] [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
2. SQuAD: 100,000+ Questions for Machine Comprehension of Text arXiv:1606.05250 \[cs.CL\] [https://arxiv.org/abs/1606.05250](https://arxiv.org/abs/1606.05250)
3. Hugging Face Transformers Documentation [https://huggingface.co/transformers/](https://huggingface.co/transformers/)

These resources provide in-depth information about BERT's architecture, training process, and applications in question answering tasks.

