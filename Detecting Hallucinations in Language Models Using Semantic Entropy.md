## Detecting Hallucinations in Language Models Using Semantic Entropy
Slide 1: Detecting Hallucinations in Large Language Models

Semantic entropy provides a novel approach to identifying hallucinations in large language models. This technique analyzes the semantic coherence and consistency of generated text, helping to distinguish between reliable outputs and potential fabrications.

```python
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def calculate_semantic_entropy(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return entropy.mean().item()

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

text = "The sky is blue and the grass is green."
entropy = calculate_semantic_entropy(text, model, tokenizer)
print(f"Semantic entropy: {entropy:.4f}")
```

Slide 2: Understanding Semantic Entropy

Semantic entropy measures the uncertainty or randomness in the distribution of predicted words. Lower entropy indicates more confident and consistent predictions, while higher entropy suggests potential hallucinations or inconsistencies.

```python
import matplotlib.pyplot as plt

def plot_entropy_distribution(entropies):
    plt.figure(figsize=(10, 6))
    plt.hist(entropies, bins=20, edgecolor='black')
    plt.title("Distribution of Semantic Entropy")
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.show()

# Generate sample entropies
np.random.seed(42)
entropies = np.random.normal(loc=2.5, scale=0.5, size=1000)
plot_entropy_distribution(entropies)
```

Slide 3: Tokenization and Embedding

Before calculating semantic entropy, we need to tokenize the input text and convert it into embeddings. This process transforms words into numerical representations that capture semantic meaning.

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "Detecting hallucinations in language models"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

embeddings = outputs.last_hidden_state
print(f"Embedding shape: {embeddings.shape}")
print(f"First token embedding: {embeddings[0][0][:5]}")
```

Slide 4: Calculating Token-level Entropy

To detect hallucinations, we first calculate the entropy for each token in the generated text. This helps identify specific words or phrases that might be inconsistent or unreliable.

```python
import torch

def token_entropy(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return entropy

# Sample logits (3 tokens, vocabulary size of 5)
logits = torch.tensor([
    [1.0, 2.0, 0.5, 1.5, 0.8],
    [0.2, 3.0, 1.0, 0.5, 1.2],
    [2.0, 0.1, 1.5, 0.8, 1.0]
])

token_entropies = token_entropy(logits)
print("Token entropies:", token_entropies)
```

Slide 5: Aggregating Entropy Scores

After calculating token-level entropies, we aggregate these scores to obtain an overall measure of semantic coherence for the entire generated text. This helps in identifying potential hallucinations at a higher level.

```python
def aggregate_entropy(token_entropies, method='mean'):
    if method == 'mean':
        return torch.mean(token_entropies)
    elif method == 'max':
        return torch.max(token_entropies)
    elif method == 'weighted':
        weights = torch.linspace(1, 0.5, len(token_entropies))
        return torch.sum(token_entropies * weights) / torch.sum(weights)

# Using the token_entropies from the previous slide
mean_entropy = aggregate_entropy(token_entropies, method='mean')
max_entropy = aggregate_entropy(token_entropies, method='max')
weighted_entropy = aggregate_entropy(token_entropies, method='weighted')

print(f"Mean entropy: {mean_entropy:.4f}")
print(f"Max entropy: {max_entropy:.4f}")
print(f"Weighted entropy: {weighted_entropy:.4f}")
```

Slide 6: Threshold-based Detection

One approach to detecting hallucinations is to set a threshold for semantic entropy. Texts with entropy scores above this threshold are flagged as potential hallucinations.

```python
def detect_hallucination(text, model, tokenizer, threshold=2.5):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    entropies = token_entropy(logits[0])
    mean_entropy = torch.mean(entropies)
    
    is_hallucination = mean_entropy > threshold
    return is_hallucination, mean_entropy

# Example usage
text1 = "The Earth orbits around the Sun."
text2 = "Quantum computers can solve all NP-complete problems in polynomial time."

hallucination1, entropy1 = detect_hallucination(text1, model, tokenizer)
hallucination2, entropy2 = detect_hallucination(text2, model, tokenizer)

print(f"Text 1 - Hallucination: {hallucination1}, Entropy: {entropy1:.4f}")
print(f"Text 2 - Hallucination: {hallucination2}, Entropy: {entropy2:.4f}")
```

Slide 7: Comparative Analysis

To improve hallucination detection, we can compare the semantic entropy of generated text with that of known reliable sources. This helps in establishing a baseline for expected entropy levels.

```python
def compare_entropy(generated_text, reference_text, model, tokenizer):
    gen_inputs = tokenizer(generated_text, return_tensors="pt")
    ref_inputs = tokenizer(reference_text, return_tensors="pt")
    
    gen_outputs = model(**gen_inputs)
    ref_outputs = model(**ref_inputs)
    
    gen_entropy = calculate_semantic_entropy(generated_text, model, tokenizer)
    ref_entropy = calculate_semantic_entropy(reference_text, model, tokenizer)
    
    entropy_diff = gen_entropy - ref_entropy
    return gen_entropy, ref_entropy, entropy_diff

generated = "AI will surpass human intelligence in all domains by 2030."
reference = "AI research continues to make progress in specific domains."

gen_entropy, ref_entropy, diff = compare_entropy(generated, reference, model, tokenizer)

print(f"Generated entropy: {gen_entropy:.4f}")
print(f"Reference entropy: {ref_entropy:.4f}")
print(f"Entropy difference: {diff:.4f}")
```

Slide 8: Contextual Entropy Analysis

Analyzing semantic entropy in context can provide more nuanced insights into potential hallucinations. We can examine how entropy changes across different parts of the generated text.

```python
def contextual_entropy(text, model, tokenizer, window_size=5):
    tokens = tokenizer.tokenize(text)
    entropies = []
    
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i+window_size]
        window_text = tokenizer.convert_tokens_to_string(window)
        entropy = calculate_semantic_entropy(window_text, model, tokenizer)
        entropies.append(entropy)
    
    return entropies

text = "The theory of relativity explains the nature of space and time. It revolutionized our understanding of the universe."
contextual_entropies = contextual_entropy(text, model, tokenizer)

plt.figure(figsize=(10, 6))
plt.plot(contextual_entropies)
plt.title("Contextual Entropy Analysis")
plt.xlabel("Token Position")
plt.ylabel("Entropy")
plt.show()
```

Slide 9: Real-life Example: Fact-checking

Semantic entropy can be used to enhance fact-checking systems by identifying potentially unreliable statements. This example demonstrates how to flag suspicious claims in news articles.

```python
def fact_check(claim, model, tokenizer, threshold=2.0):
    entropy = calculate_semantic_entropy(claim, model, tokenizer)
    is_suspicious = entropy > threshold
    return is_suspicious, entropy

claims = [
    "Water boils at 100 degrees Celsius at sea level.",
    "Scientists have discovered a new planet made entirely of diamond.",
    "The Great Wall of China is visible from the Moon.",
]

for claim in claims:
    is_suspicious, entropy = fact_check(claim, model, tokenizer)
    print(f"Claim: {claim}")
    print(f"Suspicious: {is_suspicious}, Entropy: {entropy:.4f}\n")
```

Slide 10: Real-life Example: Content Moderation

Semantic entropy can assist in content moderation by identifying potentially fabricated or misleading user-generated content on social media platforms.

```python
def moderate_content(post, model, tokenizer, threshold=2.2):
    sentences = post.split('.')
    flagged_sentences = []
    
    for sentence in sentences:
        entropy = calculate_semantic_entropy(sentence.strip(), model, tokenizer)
        if entropy > threshold:
            flagged_sentences.append((sentence, entropy))
    
    return flagged_sentences

user_post = """
The government has developed a secret technology to control the weather.
Climate change is a hoax perpetrated by scientists.
The Earth is flat, and NASA is hiding the truth from us.
"""

flagged_content = moderate_content(user_post, model, tokenizer)

print("Flagged sentences:")
for sentence, entropy in flagged_content:
    print(f"- {sentence.strip()} (Entropy: {entropy:.4f})")
```

Slide 11: Limitations and Challenges

While semantic entropy is a powerful tool for detecting hallucinations, it has limitations. This slide explores some challenges and potential improvements.

```python
def demonstrate_limitations(model, tokenizer):
    # Challenge 1: Context-dependent statements
    context_dependent = "The ball is red."
    entropy1 = calculate_semantic_entropy(context_dependent, model, tokenizer)
    
    # Challenge 2: Sarcasm and figurative language
    sarcasm = "The weather is just perfect for a picnic during a thunderstorm."
    entropy2 = calculate_semantic_entropy(sarcasm, model, tokenizer)
    
    # Challenge 3: Domain-specific knowledge
    domain_specific = "The Higgs boson was discovered at CERN in 2012."
    entropy3 = calculate_semantic_entropy(domain_specific, model, tokenizer)
    
    print(f"Context-dependent: {entropy1:.4f}")
    print(f"Sarcasm: {entropy2:.4f}")
    print(f"Domain-specific: {entropy3:.4f}")

demonstrate_limitations(model, tokenizer)
```

Slide 12: Combining Semantic Entropy with Other Techniques

To improve hallucination detection, we can combine semantic entropy with other techniques such as perplexity and attention analysis. This multi-faceted approach provides a more robust detection system.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def combined_detection(text, entropy_threshold=2.0, perplexity_threshold=50):
    # Semantic Entropy
    entropy = calculate_semantic_entropy(text, model, tokenizer)
    
    # Perplexity
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    inputs = gpt2_tokenizer(text, return_tensors='pt')
    outputs = gpt2_model(**inputs, labels=inputs['input_ids'])
    perplexity = torch.exp(outputs.loss).item()
    
    # Combined decision
    is_hallucination = entropy > entropy_threshold or perplexity > perplexity_threshold
    return is_hallucination, entropy, perplexity

text = "Quantum computers can simulate the entire universe in real-time."
is_hallucination, entropy, perplexity = combined_detection(text)

print(f"Is hallucination: {is_hallucination}")
print(f"Entropy: {entropy:.4f}")
print(f"Perplexity: {perplexity:.4f}")
```

Slide 13: Future Directions and Research Opportunities

Semantic entropy in hallucination detection opens up various research opportunities. This slide explores potential future directions for improving the technique.

```python
def simulate_research_directions():
    directions = [
        "Attention-based entropy analysis",
        "Transfer learning for domain-specific hallucination detection",
        "Temporal entropy patterns in long-form text",
        "Multi-lingual hallucination detection using cross-lingual embeddings",
        "Reinforcement learning for adaptive entropy thresholds"
    ]
    
    for i, direction in enumerate(directions, 1):
        print(f"{i}. {direction}")
        # Simulate research progress
        progress = np.random.rand()
        print(f"   Simulated progress: {progress:.2%}\n")

simulate_research_directions()
```

Slide 14: Additional Resources

For further exploration of semantic entropy and hallucination detection in large language models, consider the following resources:

1. "Hallucination Detection in Language Models: A Survey" (ArXiv:2404.12397) URL: [https://arxiv.org/abs/2404.12397](https://arxiv.org/abs/2404.12397)
2. "Entropy-based Uncertainty Quantification for Large Language Models" (ArXiv:2311.08118) URL: [https://arxiv.org/abs/2311.08118](https://arxiv.org/abs/2311.08118)
3. "Evaluating Hallucinations in Large Language Models" (ArXiv:2305.05661) URL: [https://arxiv.org/abs/2305.05661](https://arxiv.org/abs/2305.05661)

These papers provide in-depth discussions on various aspects of hallucination detection and semantic entropy in the context of large language models.

