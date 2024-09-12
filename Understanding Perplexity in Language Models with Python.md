## Understanding Perplexity in Language Models with Python
Slide 1: Understanding Perplexity in Language Models

Perplexity is a crucial metric in evaluating language models, measuring how well a model predicts a sample of text. Lower perplexity indicates better prediction. We'll explore this concept using Python, demonstrating its calculation and significance in natural language processing.

```python
import math

def calculate_perplexity(probabilities):
    n = len(probabilities)
    product = 1
    for p in probabilities:
        product *= (1 / p)
    return math.pow(product, 1/n)

# Example probabilities for a sequence of words
word_probs = [0.1, 0.2, 0.05, 0.15, 0.1]
perplexity = calculate_perplexity(word_probs)
print(f"Perplexity: {perplexity:.2f}")
```

Slide 2: The Basics of Perplexity

Perplexity quantifies the uncertainty of a language model in predicting the next word. It's calculated as the exponential of the cross-entropy of the model on a given text. A lower perplexity suggests the model is more confident and accurate in its predictions.

```python
import numpy as np

def perplexity_from_cross_entropy(cross_entropy):
    return np.exp(cross_entropy)

# Example cross-entropy value
cross_entropy = 3.5
perplexity = perplexity_from_cross_entropy(cross_entropy)
print(f"Cross-entropy: {cross_entropy}")
print(f"Perplexity: {perplexity:.2f}")
```

Slide 3: Calculating Perplexity Step by Step

To calculate perplexity, we first compute the probability of each word in a sequence, then take the inverse product, and finally the nth root where n is the number of words. This process normalizes the score for different text lengths.

```python
def detailed_perplexity(text, word_probabilities):
    words = text.split()
    n = len(words)
    product = 1
    for word in words:
        prob = word_probabilities.get(word, 1e-10)  # Small probability for unknown words
        product *= (1 / prob)
    return math.pow(product, 1/n)

text = "the cat sat on the mat"
word_probs = {"the": 0.1, "cat": 0.05, "sat": 0.02, "on": 0.08, "mat": 0.01}
perplexity = detailed_perplexity(text, word_probs)
print(f"Perplexity of '{text}': {perplexity:.2f}")
```

Slide 4: Perplexity in Practice: N-gram Language Models

N-gram models are simple yet effective for demonstrating perplexity. We'll create a bigram model and calculate its perplexity on a test sentence.

```python
from collections import defaultdict
import math

def train_bigram_model(text):
    words = text.split()
    bigrams = defaultdict(lambda: defaultdict(int))
    for i in range(len(words) - 1):
        bigrams[words[i]][words[i+1]] += 1
    return bigrams

def bigram_probability(bigram_model, word1, word2):
    total = sum(bigram_model[word1].values())
    return bigram_model[word1][word2] / total if total > 0 else 1e-10

train_text = "the cat sat on the mat the dog ran on the grass"
test_text = "the cat ran on the grass"

bigram_model = train_bigram_model(train_text)
test_words = test_text.split()
probabilities = [bigram_probability(bigram_model, test_words[i-1], test_words[i]) 
                 for i in range(1, len(test_words))]

perplexity = calculate_perplexity(probabilities)
print(f"Perplexity of '{test_text}': {perplexity:.2f}")
```

Slide 5: Perplexity vs. Accuracy

While perplexity is a valuable metric, it's important to understand its relationship with accuracy. Lower perplexity doesn't always mean better real-world performance. Let's compare perplexity with simple accuracy for word prediction.

```python
def simple_accuracy(true_words, predicted_words):
    correct = sum(t == p for t, p in zip(true_words, predicted_words))
    return correct / len(true_words)

true_text = "the cat sat on the mat"
pred_text1 = "the dog sat on the floor"
pred_text2 = "a cat slept on the mat"

true_words = true_text.split()
pred_words1 = pred_text1.split()
pred_words2 = pred_text2.split()

accuracy1 = simple_accuracy(true_words, pred_words1)
accuracy2 = simple_accuracy(true_words, pred_words2)

print(f"Accuracy of prediction 1: {accuracy1:.2f}")
print(f"Accuracy of prediction 2: {accuracy2:.2f}")

# Assuming we have perplexity values for these predictions
perplexity1 = 120.5  # Example value
perplexity2 = 95.2   # Example value

print(f"Perplexity of prediction 1: {perplexity1:.2f}")
print(f"Perplexity of prediction 2: {perplexity2:.2f}")
```

Slide 6: Visualizing Perplexity Across Different Models

To better understand how perplexity varies across different language models, we can create a visualization. This helps in comparing model performance and identifying trends.

```python
import matplotlib.pyplot as plt

models = ['GPT-2', 'BERT', 'RoBERTa', 'XLNet', 'T5']
perplexities = [35.13, 29.81, 27.54, 26.02, 23.88]

plt.figure(figsize=(10, 6))
plt.bar(models, perplexities, color='skyblue')
plt.title('Perplexity Comparison Across Language Models')
plt.xlabel('Models')
plt.ylabel('Perplexity (lower is better)')
plt.ylim(0, max(perplexities) * 1.2)

for i, v in enumerate(perplexities):
    plt.text(i, v + 0.5, f'{v:.2f}', ha='center')

plt.tight_layout()
plt.show()
```

Slide 7: Perplexity in Subword Tokenization

Modern language models often use subword tokenization, which affects perplexity calculation. Let's explore how to handle perplexity with subword tokens.

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def calculate_subword_perplexity(text):
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

text = "The quick brown fox jumps over the lazy dog."
perplexity = calculate_subword_perplexity(text)
print(f"Perplexity: {perplexity:.2f}")
```

Slide 8: Perplexity in Context: A Real-life Example

Let's consider a real-life scenario where perplexity is used to evaluate the quality of machine-generated text in a content creation system.

```python
import random

def generate_text(seed, length):
    words = seed.split()
    for _ in range(length):
        words.append(random.choice(["delicious", "spicy", "savory", "sweet", "tangy"]))
    return " ".join(words)

def evaluate_recipe_quality(recipe, vocab):
    words = recipe.split()
    probs = [1/len(vocab) if word in vocab else 1e-5 for word in words]
    return calculate_perplexity(probs)

seed = "This recipe for pasta is"
generated_recipe = generate_text(seed, 5)
cooking_vocab = ["delicious", "spicy", "savory", "sweet", "tangy", "pasta", "recipe"]

perplexity = evaluate_recipe_quality(generated_recipe, cooking_vocab)
print(f"Generated recipe: {generated_recipe}")
print(f"Recipe quality (perplexity): {perplexity:.2f}")
```

Slide 9: Perplexity in Language Model Fine-tuning

When fine-tuning language models, perplexity can be used to track improvement. Let's simulate a fine-tuning process and observe perplexity changes.

```python
import numpy as np

def simulate_fine_tuning(initial_perplexity, epochs):
    perplexities = [initial_perplexity]
    for _ in range(epochs):
        improvement = np.random.uniform(0.9, 0.99)
        perplexities.append(perplexities[-1] * improvement)
    return perplexities

initial_perplexity = 100
epochs = 10
fine_tuning_progress = simulate_fine_tuning(initial_perplexity, epochs)

plt.figure(figsize=(10, 6))
plt.plot(range(epochs + 1), fine_tuning_progress, marker='o')
plt.title('Perplexity During Fine-tuning')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.grid(True)
plt.show()

print(f"Initial perplexity: {fine_tuning_progress[0]:.2f}")
print(f"Final perplexity: {fine_tuning_progress[-1]:.2f}")
```

Slide 10: Perplexity vs. BLEU Score

While perplexity measures model uncertainty, BLEU score evaluates translation quality. Let's compare these metrics for a simple translation task.

```python
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(reference, candidate):
    return sentence_bleu([reference.split()], candidate.split())

reference = "The cat is sleeping on the couch"
candidate1 = "A cat sleeps on the sofa"
candidate2 = "The feline is resting on the furniture"

bleu1 = calculate_bleu(reference, candidate1)
bleu2 = calculate_bleu(reference, candidate2)

# Assuming we have perplexity values for these translations
perplexity1 = 15.3  # Example value
perplexity2 = 18.7  # Example value

print(f"Candidate 1 - BLEU: {bleu1:.4f}, Perplexity: {perplexity1:.2f}")
print(f"Candidate 2 - BLEU: {bleu2:.4f}, Perplexity: {perplexity2:.2f}")
```

Slide 11: Perplexity in Multi-lingual Models

Multi-lingual models present unique challenges for perplexity calculation. Let's explore how to handle perplexity across different languages.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"  # Replace with a multi-lingual model name for actual multi-lingual support
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def calculate_multilingual_perplexity(texts):
    perplexities = {}
    for lang, text in texts.items():
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        perplexities[lang] = perplexity
    return perplexities

texts = {
    "English": "The weather is nice today.",
    "French": "Le temps est beau aujourd'hui.",
    "German": "Das Wetter ist heute schön."
}

multi_perplexities = calculate_multilingual_perplexity(texts)
for lang, ppl in multi_perplexities.items():
    print(f"{lang} perplexity: {ppl:.2f}")
```

Slide 12: Perplexity in Domain-Specific Language Models

Domain-specific language models often have different perplexity characteristics. Let's compare perplexity between a general and a domain-specific model.

```python
import random

def simulate_model_prediction(domain_specific, word):
    if domain_specific:
        return random.uniform(0.1, 0.5) if word in ["genome", "protein", "DNA"] else random.uniform(0.01, 0.1)
    else:
        return random.uniform(0.05, 0.2)

text = "The genome sequence contains protein coding DNA regions"
words = text.split()

general_probs = [simulate_model_prediction(False, word) for word in words]
domain_probs = [simulate_model_prediction(True, word) for word in words]

general_ppl = calculate_perplexity(general_probs)
domain_ppl = calculate_perplexity(domain_probs)

print(f"General model perplexity: {general_ppl:.2f}")
print(f"Domain-specific model perplexity: {domain_ppl:.2f}")
```

Slide 13: Perplexity in Conversational AI

In conversational AI, perplexity can help evaluate the coherence of responses. Let's simulate a dialogue system and calculate perplexity for different responses.

```python
def dialogue_perplexity(context, response, language_model):
    # Simulating language model probability assignment
    word_probs = {word: language_model.get(word, 0.01) for word in response.split()}
    return detailed_perplexity(response, word_probs)

context = "What's the weather like today?"
responses = [
    "It's sunny and warm outside.",
    "The capital of France is Paris.",
    "I like eating ice cream."
]

# Simulated language model (would be a trained model in reality)
language_model = {
    "sunny": 0.1, "warm": 0.08, "outside": 0.05,
    "capital": 0.03, "France": 0.02, "Paris": 0.02,
    "like": 0.07, "eating": 0.04, "ice": 0.03, "cream": 0.03
}

for response in responses:
    ppl = dialogue_perplexity(context, response, language_model)
    print(f"Response: '{response}'")
    print(f"Perplexity: {ppl:.2f}\n")
```

Slide 14: Limitations and Considerations of Perplexity

While perplexity is a valuable metric, it has limitations. It doesn't capture semantic coherence or factual accuracy. Let's explore a case where low perplexity doesn't guarantee quality.

```python
def generate_repetitive_text(word, length):
    return " ".join([word] * length)

def calculate_simple_perplexity(text, vocab_size):
    words = text.split()
    prob = 1 / vocab_size
    return math.pow(1/prob, 1/len(words))

normal_text = "The quick brown fox jumps over the lazy dog"
repetitive_text = generate_repetitive_text("the", 9)

vocab_size = 10000  # Assumption for a simple language model

normal_ppl = calculate_simple_perplexity(normal_text, vocab_size)
repetitive_ppl = calculate_simple_perplexity(repetitive_text, vocab_size)

print(f"Normal text: '{normal_text}'")
print(f"Perplexity: {normal_ppl:.2f}")
print(f"\nRepetitive text: '{repetitive_text}'")
print(f"Perplexity: {repetitive_ppl:.2f}")
```

Slide 15: Real-life Example: Perplexity in Voice Assistants

Voice assistants use perplexity to evaluate the quality of their language understanding. Let's simulate how perplexity might be used to choose the best interpretation of a voice command.

```python
def simulate_voice_recognition(command):
    interpretations = {
        "Set a timer for 5 minutes": 0.7,
        "Set a timer 4 5 minutes": 0.2,
        "Set a time or 5 minutes": 0.1
    }
    return interpretations

def choose_best_interpretation(interpretations):
    return max(interpretations, key=interpretations.get)

voice_command = "Set a timer for 5 minutes"
interpretations = simulate_voice_recognition(voice_command)

print("Voice command interpretations and their probabilities:")
for interp, prob in interpretations.items():
    ppl = 1 / prob  # Simplified perplexity calculation
    print(f"'{interp}': Probability = {prob:.2f}, Perplexity = {ppl:.2f}")

best_interp = choose_best_interpretation(interpretations)
print(f"\nBest interpretation: '{best_interp}'")
```

Slide 16: Additional Resources

For further exploration of perplexity in language models, consider these academic papers:

1. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
2. "Language Models are Few-Shot Learners" by Brown et al. (2020) ArXiv: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
3. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" by Raffel et al. (2019) ArXiv: [https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)

These papers provide in-depth discussions on language model architectures and evaluation metrics, including perplexity.

