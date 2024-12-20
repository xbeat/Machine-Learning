## Temperature's Role in Language Model Output
Slide 1: Understanding Temperature in Language Models

Temperature is a crucial hyperparameter in language models that controls the randomness of the model's output. It affects the probability distribution of the next token prediction, influencing the creativity and diversity of generated text.

```python
import torch
import torch.nn.functional as F

# Example of applying temperature to logits
logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
temperature = 0.5

# Apply temperature scaling
scaled_logits = logits / temperature

# Convert to probabilities
probabilities = F.softmax(scaled_logits, dim=0)

print(f"Original logits: {logits}")
print(f"Scaled logits (T={temperature}): {scaled_logits}")
print(f"Probabilities: {probabilities}")
```

Slide 2: The Mathematics Behind Temperature

Temperature is applied by dividing the logits (raw output scores) by the temperature value before applying the softmax function. This operation alters the shape of the probability distribution, making it either sharper (lower temperature) or flatter (higher temperature).

```python
import numpy as np
import matplotlib.pyplot as plt

def apply_temperature(logits, temperature):
    return logits / temperature

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

logits = np.array([1, 2, 3, 4, 5])
temperatures = [0.5, 1.0, 2.0]

plt.figure(figsize=(12, 4))
for i, temp in enumerate(temperatures):
    scaled_logits = apply_temperature(logits, temp)
    probs = softmax(scaled_logits)
    plt.subplot(1, 3, i+1)
    plt.bar(range(len(probs)), probs)
    plt.title(f"Temperature = {temp}")
    plt.ylim(0, 1)

plt.tight_layout()
plt.show()
```

Slide 3: Low Temperature: Focused and Deterministic

A low temperature (< 1.0) makes the model more confident and deterministic. It amplifies the differences between token probabilities, causing the model to focus on the most likely tokens and produce more predictable output.

```python
import random

def generate_text(prompt, temperature=0.5, max_length=50):
    generated_text = prompt
    vocab = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    
    for _ in range(max_length):
        # Simulate logits (normally from a language model)
        logits = [random.random() for _ in vocab]
        
        # Apply temperature
        scaled_logits = [l / temperature for l in logits]
        
        # Convert to probabilities
        probs = softmax(scaled_logits)
        
        # Sample next word
        next_word = random.choices(vocab, weights=probs)[0]
        generated_text += " " + next_word
    
    return generated_text

low_temp_text = generate_text("The", temperature=0.2)
print("Low temperature output:", low_temp_text)
```

Slide 4: High Temperature: Creative and Diverse

A high temperature (> 1.0) makes the model more random and creative. It flattens the probability distribution, giving less likely tokens a higher chance of being selected and producing more diverse and potentially surprising output.

```python
# Using the generate_text function from the previous slide
high_temp_text = generate_text("The", temperature=2.0)
print("High temperature output:", high_temp_text)
```

Slide 5: Balancing Creativity and Coherence

Finding the right temperature is crucial for balancing creativity and coherence in generated text. Too low, and the output becomes repetitive; too high, and it loses coherence. The optimal temperature often depends on the specific task and desired outcome.

```python
def analyze_temperature_effects(prompt, temperatures):
    for temp in temperatures:
        generated = generate_text(prompt, temperature=temp)
        unique_words = len(set(generated.split()))
        print(f"Temperature {temp}:")
        print(f"Generated text: {generated}")
        print(f"Unique words: {unique_words}\n")

analyze_temperature_effects("Once upon a time", [0.2, 1.0, 2.0])
```

Slide 6: Temperature in Practice: Text Completion

Let's explore how temperature affects text completion tasks. We'll use a simple mock language model to demonstrate the impact of different temperature values on completing a sentence.

```python
import numpy as np

def mock_language_model(prompt, temperature=1.0):
    # Mock logits for next word prediction
    logits = {
        "sunny": 5.0,
        "rainy": 3.0,
        "cloudy": 2.0,
        "stormy": 1.0
    }
    
    scaled_logits = {word: score / temperature for word, score in logits.items()}
    probs = softmax(list(scaled_logits.values()))
    
    return np.random.choice(list(logits.keys()), p=probs)

prompt = "The weather today is"

for temp in [0.1, 1.0, 2.0]:
    completions = [mock_language_model(prompt, temperature=temp) for _ in range(10)]
    print(f"Temperature {temp}:")
    print(f"Completions: {completions}")
    print(f"Unique completions: {len(set(completions))}\n")
```

Slide 7: Temperature in Dialogue Systems

Temperature plays a crucial role in dialogue systems, affecting the personality and consistency of the generated responses. Let's simulate a simple dialogue system to see how temperature influences conversation flow.

```python
def dialogue_system(user_input, temperature=1.0):
    responses = {
        "hello": ["Hi there!", "Hello!", "Greetings!", "Hey!"],
        "how are you": ["I'm good, thanks!", "Doing well!", "Great!", "Fantastic!"],
        "goodbye": ["Bye!", "See you!", "Take care!", "Farewell!"]
    }
    
    if user_input.lower() in responses:
        logits = [i + 1 for i in range(len(responses[user_input.lower()]))]
        scaled_logits = [l / temperature for l in logits]
        probs = softmax(scaled_logits)
        return np.random.choice(responses[user_input.lower()], p=probs)
    else:
        return "I'm not sure how to respond to that."

conversations = [
    ["hello", "how are you", "goodbye"],
    ["hello", "how are you", "goodbye"]
]

for temp in [0.5, 2.0]:
    print(f"Temperature {temp}:")
    for conv in conversations:
        print("User: " + " -> ".join(conv))
        print("Bot: " + " -> ".join([dialogue_system(msg, temperature=temp) for msg in conv]))
    print()
```

Slide 8: Temperature and Sampling Strategies

Temperature is often used in conjunction with other sampling strategies like top-k and nucleus (top-p) sampling. These methods can help control the trade-off between diversity and quality in generated text.

```python
import torch
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    
    return logits

# Example usage
logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
temperature = 0.7
top_k = 3
top_p = 0.9

# Apply temperature
scaled_logits = logits / temperature

# Apply top-k and top-p filtering
filtered_logits = top_k_top_p_filtering(scaled_logits, top_k=top_k, top_p=top_p)

print(f"Original logits: {logits}")
print(f"Scaled logits (T={temperature}): {scaled_logits}")
print(f"Filtered logits: {filtered_logits}")
```

Slide 9: Real-life Example: Recipe Generation

Let's explore how temperature affects recipe generation. We'll use a simple mock model to demonstrate how different temperatures can lead to varying levels of creativity in recipe instructions.

```python
import random

def generate_recipe_step(temperature=1.0):
    actions = ["Mix", "Stir", "Chop", "Bake", "Fry", "Boil"]
    ingredients = ["onions", "tomatoes", "chicken", "pasta", "eggs", "cheese"]
    durations = ["for 5 minutes", "until golden brown", "thoroughly", "gently"]
    
    action = random.choices(actions, weights=softmax([i/temperature for i in range(len(actions))]))[0]
    ingredient = random.choices(ingredients, weights=softmax([i/temperature for i in range(len(ingredients))]))[0]
    duration = random.choices(durations, weights=softmax([i/temperature for i in range(len(durations))]))[0]
    
    return f"{action} the {ingredient} {duration}."

def generate_recipe(steps=5, temperature=1.0):
    return [generate_recipe_step(temperature) for _ in range(steps)]

print("Low temperature recipe (T=0.5):")
print("\n".join(generate_recipe(temperature=0.5)))

print("\nHigh temperature recipe (T=2.0):")
print("\n".join(generate_recipe(temperature=2.0)))
```

Slide 10: Temperature and Model Confidence

Temperature can be used to calibrate a model's confidence. Lower temperatures make the model more confident in its predictions, while higher temperatures make it less certain. This can be particularly useful in applications where we need to assess the model's uncertainty.

```python
import numpy as np
import matplotlib.pyplot as plt

def model_confidence(logits, temperature):
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits)
    return np.max(probs)

logits = np.array([1, 2, 3, 4, 5])
temperatures = np.linspace(0.1, 2, 100)

confidences = [model_confidence(logits, t) for t in temperatures]

plt.figure(figsize=(10, 6))
plt.plot(temperatures, confidences)
plt.title("Model Confidence vs Temperature")
plt.xlabel("Temperature")
plt.ylabel("Confidence (Max Probability)")
plt.grid(True)
plt.show()
```

Slide 11: Temperature in Multi-modal Models

Temperature is not limited to text generation; it's also used in multi-modal models that generate images or audio. Let's simulate how temperature might affect image generation using a simple color palette example.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_color_palette(temperature=1.0, size=5):
    # Simulate logits for RGB values
    logits = np.random.rand(3, 256)
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits, axis=1)
    
    # Sample colors
    colors = []
    for _ in range(size):
        r = np.random.choice(256, p=probs[0])
        g = np.random.choice(256, p=probs[1])
        b = np.random.choice(256, p=probs[2])
        colors.append((r/255, g/255, b/255))
    
    return colors

plt.figure(figsize=(15, 5))
temperatures = [0.5, 1.0, 2.0]

for i, temp in enumerate(temperatures):
    palette = generate_color_palette(temperature=temp)
    plt.subplot(1, 3, i+1)
    plt.bar(range(len(palette)), [1]*len(palette), color=palette, width=1)
    plt.title(f"Temperature = {temp}")
    plt.axis('off')

plt.tight_layout()
plt.show()
```

Slide 12: Adaptive Temperature

In some applications, it's beneficial to use adaptive temperature, where the temperature value changes based on the context or the model's confidence. This can help balance between coherence and creativity dynamically.

```python
import numpy as np

def adaptive_temperature(confidence, min_temp=0.5, max_temp=2.0):
    return min_temp + (1 - confidence) * (max_temp - min_temp)

def generate_with_adaptive_temp(prompt, confidence):
    temp = adaptive_temperature(confidence)
    return generate_text(prompt, temperature=temp)

# Simulate different confidence levels
confidences = [0.9, 0.5, 0.1]
prompt = "The future of AI is"

for conf in confidences:
    temp = adaptive_temperature(conf)
    generated = generate_with_adaptive_temp(prompt, conf)
    print(f"Confidence: {conf:.2f}, Temperature: {temp:.2f}")
    print(f"Generated: {generated}\n")
```

Slide 13: Best Practices for Temperature Tuning

1. Start with a default temperature of 1.0 and adjust based on your specific needs.
2. Use lower temperatures (0.1 - 0.5) for tasks requiring factual accuracy or consistency.
3. Use higher temperatures (1.5 - 2.0) for creative tasks or when you want more diverse outputs.
4. Experiment with different temperatures and evaluate the results for your specific use case.
5. Consider combining temperature with other sampling strategies for more fine-grained control.

```python
def temperature_experiment(prompt, temperatures):
    results = {}
    for temp in temperatures:
        generated = generate_text(prompt, temperature=temp)
        results[temp] = generated
    return results

prompt = "In the year 2050, technology will"
temps = [0.5, 1.0, 1.5]

experiment_results = temperature_experiment(prompt, temps)

for temp, result in experiment_results.items():
    print(f"Temperature {temp}:")
    print(result)
    print()
```

Slide 14: Additional Resources

1. "The Curious Case of Neural Text Degeneration" by Holtzman et al. (2020) ArXiv: [https://arxiv.org/abs/1904.09751](https://arxiv.org/abs/1904.09751)
2. "How to Generate Text: Using Different Decoding Methods for Language Generation with Transformers" by Hugging Face URL: [https://huggingface.co/blog/how-to-generate](https://huggingface.co/blog/how-to-generate)
3. "A Survey of Deep Learning Techniques for Neural Machine Translation" by Stahlberg (2020) ArXiv: [https://arxiv.org/abs/1905.13891](https://arxiv.org/abs/1905.13891)

These resources provide in-depth discussions on temperature and other text generation strategies in language models.

