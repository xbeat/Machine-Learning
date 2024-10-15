## Mastering Temperature and Top_p in Python Language Models
Slide 1: Understanding Temperature and Top\_p in Language Models

Temperature and top\_p are crucial parameters in language models that control the randomness and diversity of generated text. These settings affect how the model selects the next token in a sequence, influencing the creativity and coherence of the output.

```python
import torch
import torch.nn.functional as F

def sample_with_temperature(logits, temperature=1.0):
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)
    # Sample from the distribution
    return torch.multinomial(probs, num_samples=1)

# Example logits
logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
# Sample with different temperatures
low_temp = sample_with_temperature(logits, temperature=0.5)
high_temp = sample_with_temperature(logits, temperature=2.0)

print(f"Low temperature (0.5) sample: {low_temp.item()}")
print(f"High temperature (2.0) sample: {high_temp.item()}")
```

Slide 2: Temperature in Language Models

Temperature is a hyperparameter that controls the randomness of predictions by scaling the logits before applying softmax. A lower temperature makes the model more confident and deterministic, while a higher temperature increases randomness and creativity.

```python
import numpy as np
import matplotlib.pyplot as plt

def apply_temperature(logits, temperature):
    return np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))

logits = np.array([1, 2, 3, 4, 5])
temperatures = [0.1, 0.5, 1.0, 2.0]

plt.figure(figsize=(12, 6))
for temp in temperatures:
    probs = apply_temperature(logits, temp)
    plt.plot(probs, label=f"T={temp}")

plt.title("Effect of Temperature on Token Probabilities")
plt.xlabel("Token Index")
plt.ylabel("Probability")
plt.legend()
plt.show()
```

Slide 3: Implementing Temperature in Token Selection

Here's how to implement temperature-based token selection in Python. This process involves scaling the logits, converting them to probabilities, and then sampling from the resulting distribution.

```python
import numpy as np

def sample_with_temperature(logits, temperature=1.0):
    # Scale logits by temperature
    scaled_logits = logits / temperature
    # Convert to probabilities
    probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
    # Sample from the distribution
    return np.random.choice(len(logits), p=probs)

# Example usage
logits = np.array([1, 2, 3, 4, 5])
selected_token = sample_with_temperature(logits, temperature=0.5)
print(f"Selected token index: {selected_token}")
```

Slide 4: Top\_p Sampling (Nucleus Sampling)

Top\_p sampling, also known as nucleus sampling, is an alternative to temperature-based sampling. It selects from the smallest possible set of tokens whose cumulative probability exceeds the probability p. This method helps maintain diversity while avoiding low-probability tokens.

```python
import numpy as np

def top_p_sampling(logits, p=0.9):
    # Sort the logits in descending order
    sorted_logits = np.sort(logits)[::-1]
    sorted_indices = np.argsort(logits)[::-1]
    
    # Calculate cumulative probabilities
    cumulative_probs = np.cumsum(sorted_logits)
    
    # Find the index where cumulative probability exceeds p
    last_index = np.where(cumulative_probs > p)[0][0]
    
    # Create a mask for the selected tokens
    mask = np.zeros_like(logits, dtype=bool)
    mask[sorted_indices[:last_index+1]] = True
    
    # Set the probabilities of unselected tokens to 0
    logits[~mask] = float('-inf')
    
    # Convert to probabilities and sample
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return np.random.choice(len(logits), p=probs)

# Example usage
logits = np.array([1, 2, 3, 4, 5])
selected_token = top_p_sampling(logits, p=0.9)
print(f"Selected token index: {selected_token}")
```

Slide 5: Comparing Temperature and Top\_p

Both temperature and top\_p affect the diversity of generated text, but they work differently. Temperature scales the entire distribution, while top\_p truncates it. Let's compare their effects on a simple probability distribution.

```python
import numpy as np
import matplotlib.pyplot as plt

def apply_temperature(logits, temperature):
    return np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))

def apply_top_p(probs, p):
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = np.where(cumulative_probs > p)[0][0]
    probs[probs < sorted_probs[cutoff]] = 0
    return probs / np.sum(probs)

logits = np.array([1, 2, 3, 4, 5])
base_probs = np.exp(logits) / np.sum(np.exp(logits))

plt.figure(figsize=(12, 6))
plt.plot(base_probs, label="Original")
plt.plot(apply_temperature(logits, 0.5), label="Temperature=0.5")
plt.plot(apply_top_p(base_probs.(), 0.9), label="Top_p=0.9")

plt.title("Comparison of Temperature and Top_p Sampling")
plt.xlabel("Token Index")
plt.ylabel("Probability")
plt.legend()
plt.show()
```

Slide 6: Real-Life Example: Text Generation

Let's look at how temperature and top\_p affect text generation in a simple language model. We'll use a pre-trained model to generate text with different settings.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length, temperature=1.0, top_p=1.0):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=1
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompt = "The future of artificial intelligence is"
print("Temperature 0.7:")
print(generate_text(prompt, 50, temperature=0.7))
print("\nTop_p 0.9:")
print(generate_text(prompt, 50, temperature=1.0, top_p=0.9))
```

Slide 7: Visualizing Token Probabilities

To better understand how temperature and top\_p affect token selection, let's visualize the probability distribution for the next token in a sequence.

```python
import numpy as np
import matplotlib.pyplot as plt

def get_token_probs(logits, temperature=1.0, top_p=1.0):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    if top_p < 1.0:
        sorted_probs = np.sort(probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = np.where(cumulative_probs > top_p)[0][0]
        probs[probs < sorted_probs[cutoff]] = 0
        probs = probs / np.sum(probs)
    return probs

logits = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
tokens = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.figure(figsize=(12, 6))
plt.bar(tokens, get_token_probs(logits), alpha=0.5, label='Original')
plt.bar(tokens, get_token_probs(logits, temperature=0.5), alpha=0.5, label='Temperature=0.5')
plt.bar(tokens, get_token_probs(logits, top_p=0.9), alpha=0.5, label='Top_p=0.9')

plt.title("Token Probabilities with Different Sampling Methods")
plt.xlabel("Tokens")
plt.ylabel("Probability")
plt.legend()
plt.show()
```

Slide 8: Temperature and Creativity

Temperature can be used to control the creativity and randomness of generated text. Let's experiment with different temperature values to see how they affect the output of a simple text generation model.

```python
import random

def simple_text_generator(prompt, length, temperature):
    vocabulary = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    generated_text = prompt
    
    for _ in range(length):
        # Simple "logits" based on the last character
        logits = [ord(c) for c in vocabulary]
        logits = [l + (ord(generated_text[-1]) - l) * 0.1 for l in logits]
        
        # Apply temperature
        probs = [np.exp(l / temperature) for l in logits]
        probs = [p / sum(probs) for p in probs]
        
        # Sample next character
        next_char = random.choices(vocabulary, weights=probs)[0]
        generated_text += next_char
    
    return generated_text

prompt = "THE QUICK BROWN FOX "
temperatures = [0.1, 0.5, 1.0, 2.0]

for temp in temperatures:
    print(f"Temperature {temp}:")
    print(simple_text_generator(prompt, 50, temp))
    print()
```

Slide 9: Top\_p and Coherence

Top\_p sampling can help maintain coherence in generated text by focusing on the most likely tokens. Let's implement a simple top\_p sampling function and see how it affects text generation.

```python
import random

def top_p_sampling(probs, p):
    sorted_probs = sorted(probs, reverse=True)
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = next(i for i, cp in enumerate(cumulative_probs) if cp > p)
    top_p_probs = [p if p >= sorted_probs[cutoff] else 0 for p in probs]
    return [p / sum(top_p_probs) for p in top_p_probs]

def simple_text_generator_top_p(prompt, length, p):
    vocabulary = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    generated_text = prompt
    
    for _ in range(length):
        # Simple "logits" based on the last character
        logits = [ord(c) for c in vocabulary]
        logits = [l + (ord(generated_text[-1]) - l) * 0.1 for l in logits]
        
        # Convert to probabilities and apply top_p
        probs = [np.exp(l) for l in logits]
        probs = [p / sum(probs) for p in probs]
        probs = top_p_sampling(probs, p)
        
        # Sample next character
        next_char = random.choices(vocabulary, weights=probs)[0]
        generated_text += next_char
    
    return generated_text

prompt = "THE QUICK BROWN FOX "
p_values = [0.5, 0.9, 0.99]

for p in p_values:
    print(f"Top_p {p}:")
    print(simple_text_generator_top_p(prompt, 50, p))
    print()
```

Slide 10: Combining Temperature and Top\_p

In practice, it's common to use both temperature and top\_p sampling together. This combination can provide fine-grained control over the trade-off between creativity and coherence in generated text.

```python
def combined_sampling(logits, temperature, top_p):
    # Apply temperature
    scaled_logits = logits / temperature
    probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
    
    # Apply top_p
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = np.where(cumulative_probs > top_p)[0][0]
    probs[probs < sorted_probs[cutoff]] = 0
    probs = probs / np.sum(probs)
    
    return np.random.choice(len(logits), p=probs)

# Example usage
logits = np.array([1, 2, 3, 4, 5])
temperature = 0.8
top_p = 0.9

selected_token = combined_sampling(logits, temperature, top_p)
print(f"Selected token index: {selected_token}")
```

Slide 11: Real-Life Example: Chatbot Responses

Let's implement a simple chatbot that uses temperature and top\_p to generate responses. We'll use different settings to show how they affect the chatbot's personality.

```python
import random

responses = {
    "greeting": ["Hello!", "Hi there!", "Greetings!", "Hey!", "Good day!"],
    "farewell": ["Goodbye!", "See you later!", "Bye!", "Take care!", "Until next time!"],
    "unknown": ["I'm not sure about that.", "Could you rephrase that?", "Interesting question!", "I'll have to think about that."]
}

def chatbot_response(input_text, temperature, top_p):
    input_lower = input_text.lower()
    if "hello" in input_lower or "hi" in input_lower:
        category = "greeting"
    elif "bye" in input_lower or "goodbye" in input_lower:
        category = "farewell"
    else:
        category = "unknown"
    
    logits = [random.random() for _ in responses[category]]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    selected_index = combined_sampling(logits, temperature, top_p)
    return responses[category][selected_index]

print("Conservative Chatbot (Low temperature, high top_p):")
for _ in range(3):
    user_input = random.choice(["Hello", "What's the meaning of life?", "Goodbye"])
    print(f"User: {user_input}")
    print(f"Bot: {chatbot_response(user_input, temperature=0.5, top_p=0.9)}\n")

print("Creative Chatbot (High temperature, low top_p):")
for _ in range(3):
    user_input = random.choice(["Hello", "What's the meaning of life?", "Goodbye"])
    print(f"User: {user_input}")
    print(f"Bot: {chatbot_response(user_input, temperature=1.5, top_p=0.7)}\n")
```

Slide 12: Optimizing Temperature and Top\_p

Finding the right balance of temperature and top\_p can significantly improve the quality of generated text. Let's create a simple optimization function that tries different combinations and evaluates them based on a given metric.

```python
import numpy as np
from scipy.optimize import minimize

def text_quality_metric(text):
    # Placeholder metric: unique character ratio
    return len(set(text)) / len(text)

def generate_text(temperature, top_p):
    # Placeholder function to generate text
    # In practice, this would use a language model
    return "Sample generated text"

def optimize_params(initial_temp, initial_top_p):
    def objective(params):
        temp, top_p = params
        text = generate_text(temp, top_p)
        return -text_quality_metric(text)  # Negative for minimization

    result = minimize(
        objective,
        [initial_temp, initial_top_p],
        bounds=[(0.1, 2.0), (0.1, 1.0)],
        method='L-BFGS-B'
    )

    return result.x

optimal_temp, optimal_top_p = optimize_params(1.0, 0.9)
print(f"Optimal temperature: {optimal_temp:.2f}")
print(f"Optimal top_p: {optimal_top_p:.2f}")
```

Slide 13: Adaptive Temperature and Top\_p

In some scenarios, it's beneficial to adjust temperature and top\_p dynamically based on the context or the generated text so far. Let's implement a simple adaptive mechanism.

```python
def adaptive_sampling(logits, base_temperature, base_top_p, context):
    # Adjust temperature based on context length
    adaptive_temp = base_temperature * (1 + len(context) * 0.01)
    
    # Adjust top_p based on unique token ratio in context
    unique_ratio = len(set(context)) / len(context)
    adaptive_top_p = base_top_p * (1 - unique_ratio * 0.2)
    
    # Apply combined sampling with adaptive parameters
    return combined_sampling(logits, adaptive_temp, adaptive_top_p)

# Example usage
context = "The quick brown fox jumps over the lazy dog"
logits = np.array([1, 2, 3, 4, 5])
base_temperature = 0.8
base_top_p = 0.9

selected_token = adaptive_sampling(logits, base_temperature, base_top_p, context)
print(f"Selected token index: {selected_token}")
```

Slide 14: Visualizing the Impact of Temperature and Top\_p

To better understand how temperature and top\_p affect token selection, let's create a heatmap visualization of token probabilities under different parameter combinations.

```python
import numpy as np
import matplotlib.pyplot as plt

def token_probabilities(logits, temperature, top_p):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = np.where(cumulative_probs > top_p)[0][0] if top_p < 1.0 else len(probs)
    probs[probs < sorted_probs[cutoff]] = 0
    return probs / np.sum(probs)

logits = np.array([1, 2, 3, 4, 5])
temperatures = np.linspace(0.1, 2.0, 20)
top_ps = np.linspace(0.1, 1.0, 20)

heatmap_data = np.zeros((len(temperatures), len(top_ps)))

for i, temp in enumerate(temperatures):
    for j, p in enumerate(top_ps):
        probs = token_probabilities(logits, temp, p)
        heatmap_data[i, j] = np.max(probs)  # Use max probability as metric

plt.figure(figsize=(10, 8))
plt.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='lower')
plt.colorbar(label='Max Token Probability')
plt.xlabel('Top_p')
plt.ylabel('Temperature')
plt.title('Impact of Temperature and Top_p on Token Probabilities')
plt.xticks(range(0, len(top_ps), 5), [f'{p:.1f}' for p in top_ps[::5]])
plt.yticks(range(0, len(temperatures), 5), [f'{t:.1f}' for t in temperatures[::5]])
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into the topics of temperature and top\_p sampling in language models, here are some valuable resources:

1. "The Curious Case of Neural Text Degeneration" by Holtzman et al. (2019) ArXiv: [https://arxiv.org/abs/1904.09751](https://arxiv.org/abs/1904.09751)
2. "How to Sample from Language Models" by Hugging Face [https://huggingface.co/blog/how-to-generate](https://huggingface.co/blog/how-to-generate)
3. "Language Models are Few-Shot Learners" by Brown et al. (2020) ArXiv: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

These resources provide in-depth discussions on sampling strategies, their effects on text generation, and their applications in various language models.

