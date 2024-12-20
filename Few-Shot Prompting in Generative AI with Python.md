## Few-Shot Prompting in Generative AI with Python
Slide 1: What is Few-Shot Prompting?

Few-Shot Prompting is a technique in generative AI where a model is given a small number of examples to learn from before generating new content. This approach bridges the gap between zero-shot learning (no examples) and fine-tuning (many examples).

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

few_shot_prompt = """
Translate English to French:
Hello -> Bonjour
Goodbye -> Au revoir
Good morning -> """

input_ids = tokenizer.encode(few_shot_prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Slide 2: The Anatomy of a Few-Shot Prompt

A Few-Shot Prompt typically consists of three components: the task description, a set of examples, and the new input. This structure helps the model understand the context and expected output format.

```python
def create_few_shot_prompt(task, examples, new_input):
    prompt = f"{task}\n\n"
    for input_ex, output_ex in examples:
        prompt += f"Input: {input_ex}\nOutput: {output_ex}\n\n"
    prompt += f"Input: {new_input}\nOutput:"
    return prompt

task = "Classify the sentiment of the following movie reviews as positive or negative."
examples = [
    ("This movie was fantastic!", "Positive"),
    ("I hated every minute of it.", "Negative")
]
new_input = "The acting was great, but the plot was confusing."

prompt = create_few_shot_prompt(task, examples, new_input)
print(prompt)
```

Slide 3: Advantages of Few-Shot Prompting

Few-Shot Prompting allows models to adapt quickly to new tasks without extensive fine-tuning. It's particularly useful for tasks with limited training data or when rapid prototyping is needed.

```python
import numpy as np
import matplotlib.pyplot as plt

def performance_vs_examples(num_examples):
    return 1 - np.exp(-0.5 * num_examples)

examples = np.arange(0, 10)
performance = performance_vs_examples(examples)

plt.figure(figsize=(10, 6))
plt.plot(examples, performance)
plt.title("Performance vs Number of Examples in Few-Shot Prompting")
plt.xlabel("Number of Examples")
plt.ylabel("Performance")
plt.show()
```

Slide 4: Implementing Few-Shot Prompting with OpenAI's GPT-3

Here's an example of how to implement Few-Shot Prompting using OpenAI's GPT-3 API. Note that you'll need an API key to run this code.

```python
import openai

openai.api_key = "your-api-key-here"

def generate_with_few_shot(prompt, max_tokens=50):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=max_tokens
    )
    return response.choices[0].text.strip()

few_shot_prompt = """
Correct the spelling in these sentences:
1. The cat satt on the mat. -> The cat sat on the mat.
2. I love eting pizza. -> I love eating pizza.
3. She went to the beech yesterday. ->"""

result = generate_with_few_shot(few_shot_prompt)
print(result)
```

Slide 5: Fine-tuning vs Few-Shot Prompting

While fine-tuning involves updating model weights, Few-Shot Prompting works within the existing knowledge of the model. This comparison highlights the efficiency of Few-Shot Prompting for quick adaptation.

```python
import time

def simulate_training(method, num_examples):
    if method == "fine-tuning":
        return num_examples * 0.5  # Simulated time in seconds
    elif method == "few-shot":
        return 0.1  # Constant time for few-shot

methods = ["fine-tuning", "few-shot"]
examples = [10, 100, 1000]

for method in methods:
    times = [simulate_training(method, n) for n in examples]
    plt.plot(examples, times, label=method)

plt.xlabel("Number of Examples")
plt.ylabel("Time (seconds)")
plt.title("Fine-tuning vs Few-Shot Prompting: Time Comparison")
plt.legend()
plt.show()
```

Slide 6: Few-Shot Prompting for Text Classification

Let's explore how Few-Shot Prompting can be used for text classification tasks, such as sentiment analysis or topic categorization.

```python
def text_classifier(text, examples, labels):
    prompt = "Classify the following text into one of these categories: " + ", ".join(labels) + "\n\n"
    for ex, label in zip(examples, labels):
        prompt += f"Text: {ex}\nCategory: {label}\n\n"
    prompt += f"Text: {text}\nCategory:"

    # In a real scenario, you would use an AI model here
    # For demonstration, we'll use a simple keyword matching
    for label, keywords in zip(labels, [["happy", "great"], ["sad", "terrible"], ["angry", "furious"]]):
        if any(keyword in text.lower() for keyword in keywords):
            return label
    return "Unknown"

examples = [
    "I had a great day at the park!",
    "The weather is terrible today.",
    "I'm furious about the poor service."
]
labels = ["Happy", "Sad", "Angry"]

new_text = "I'm really excited about the upcoming concert!"
result = text_classifier(new_text, examples, labels)
print(f"Classification result: {result}")
```

Slide 7: Few-Shot Prompting for Named Entity Recognition (NER)

Few-Shot Prompting can be applied to NER tasks, helping models identify and classify named entities in text.

```python
def ner_tagger(text, examples):
    prompt = "Tag named entities in the following text. Entities: PERSON, ORGANIZATION, LOCATION\n\n"
    for ex in examples:
        prompt += f"Text: {ex['text']}\nTagged: {ex['tagged']}\n\n"
    prompt += f"Text: {text}\nTagged:"

    # Simulating NER tagging (in reality, you'd use an AI model)
    words = text.split()
    tagged = []
    for word in words:
        if word[0].isupper():
            if word in ["Inc.", "Corp."]:
                tagged.append(f"{word}<ORGANIZATION>")
            elif word in ["Street", "Avenue", "Road"]:
                tagged.append(f"{word}<LOCATION>")
            else:
                tagged.append(f"{word}<PERSON>")
        else:
            tagged.append(word)
    return " ".join(tagged)

examples = [
    {"text": "John works at Apple Inc.", "tagged": "John<PERSON> works at Apple Inc.<ORGANIZATION>"},
    {"text": "Paris is beautiful in spring.", "tagged": "Paris<LOCATION> is beautiful in spring."}
]

new_text = "Sarah visited Google headquarters on Main Street."
result = ner_tagger(new_text, examples)
print(f"NER result: {result}")
```

Slide 8: Few-Shot Prompting for Text Generation

Few-Shot Prompting can guide text generation tasks, helping models produce content in specific styles or formats.

```python
def generate_text(prompt, style_examples):
    full_prompt = "Generate text in the following style:\n\n"
    for example in style_examples:
        full_prompt += f"Example: {example}\n\n"
    full_prompt += f"Now generate: {prompt}\n"

    # Simulating text generation (in reality, you'd use an AI model)
    import random
    styles = ["poetic", "technical", "humorous"]
    chosen_style = random.choice(styles)
    
    if chosen_style == "poetic":
        return "In whispers of dawn, secrets unfold, a tapestry of words yet untold."
    elif chosen_style == "technical":
        return "The algorithmic complexity of the process is O(n log n), optimizing efficiency."
    else:
        return "Why did the AI go to therapy? It had too many unresolved issues in its neural network!"

style_examples = [
    "Roses are red, violets are blue, AI is smart, and so are you.",
    "The quantum computer utilizes superposition to perform parallel computations.",
    "Why don't scientists trust atoms? Because they make up everything!"
]

prompt = "Write about the future of technology"
generated_text = generate_text(prompt, style_examples)
print(f"Generated text: {generated_text}")
```

Slide 9: Few-Shot Prompting for Code Generation

Few-Shot Prompting can be used to generate code snippets based on natural language descriptions and examples.

```python
def generate_code(description, language, examples):
    prompt = f"Generate {language} code for the following description:\n\n"
    for ex in examples:
        prompt += f"Description: {ex['description']}\nCode:\n{ex['code']}\n\n"
    prompt += f"Description: {description}\nCode:\n"

    # Simulating code generation (in reality, you'd use an AI model)
    if "sort" in description.lower():
        return "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"
    elif "fibonacci" in description.lower():
        return "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)"
    else:
        return "# Unable to generate code for the given description"

examples = [
    {
        "description": "Function to calculate the factorial of a number",
        "code": "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)"
    }
]

description = "Write a function to generate Fibonacci sequence"
generated_code = generate_code(description, "Python", examples)
print(f"Generated code:\n{generated_code}")
```

Slide 10: Few-Shot Prompting for Data Augmentation

Few-Shot Prompting can be used to generate additional training data, helping to augment existing datasets for machine learning tasks.

```python
import random

def augment_data(original_data, num_augmentations):
    augmented_data = []
    for item in original_data:
        augmented_data.append(item)  # Keep the original
        for _ in range(num_augmentations):
            # Simulating data augmentation (in reality, you'd use an AI model)
            augmented_item = item.()
            if random.random() < 0.5:
                augmented_item['text'] = augmented_item['text'].replace("good", "great")
            else:
                augmented_item['text'] = augmented_item['text'].replace("bad", "terrible")
            augmented_data.append(augmented_item)
    return augmented_data

original_data = [
    {"text": "This movie was good", "label": "positive"},
    {"text": "I had a bad experience", "label": "negative"}
]

augmented_data = augment_data(original_data, num_augmentations=2)
for item in augmented_data:
    print(f"Text: {item['text']}, Label: {item['label']}")
```

Slide 11: Few-Shot Prompting for Multi-task Learning

Few-Shot Prompting can be applied to multi-task scenarios, where a single model learns to perform multiple related tasks based on examples.

```python
def multi_task_model(task, input_text, examples):
    prompt = f"Perform the following task: {task}\n\n"
    for ex in examples:
        prompt += f"Task: {ex['task']}\nInput: {ex['input']}\nOutput: {ex['output']}\n\n"
    prompt += f"Task: {task}\nInput: {input_text}\nOutput:"

    # Simulating multi-task processing (in reality, you'd use an AI model)
    if "translate" in task.lower():
        return "Bonjour, comment ça va?"
    elif "summarize" in task.lower():
        return "Brief summary of the input text."
    elif "sentiment" in task.lower():
        return "Positive"
    else:
        return "Task not recognized"

examples = [
    {"task": "Translate to French", "input": "Hello, how are you?", "output": "Bonjour, comment allez-vous?"},
    {"task": "Summarize", "input": "Long text about AI advancements...", "output": "AI is rapidly advancing in various fields."},
    {"task": "Sentiment Analysis", "input": "I love this product!", "output": "Positive"}
]

task = "Translate to French"
input_text = "Hello, how are you?"
result = multi_task_model(task, input_text, examples)
print(f"Multi-task result: {result}")
```

Slide 12: Real-life Example: Customer Support Chatbot

A practical application of Few-Shot Prompting in customer support, where a chatbot can handle various types of inquiries based on a few examples.

```python
def customer_support_bot(query, examples):
    prompt = "Answer the following customer query based on these examples:\n\n"
    for ex in examples:
        prompt += f"Query: {ex['query']}\nResponse: {ex['response']}\n\n"
    prompt += f"Query: {query}\nResponse:"

    # Simulating bot response (in reality, you'd use an AI model)
    if "return" in query.lower():
        return "To initiate a return, please log into your account, go to 'Order History', and select 'Return Item' next to the product you wish to return. Follow the instructions to generate a return label."
    elif "shipping" in query.lower():
        return "We offer free standard shipping on orders over $50. Express shipping is available for an additional fee. Most orders are processed within 1-2 business days."
    else:
        return "I'm sorry, I couldn't understand your query. Please contact our support team at support@example.com for further assistance."

examples = [
    {"query": "How do I track my order?", "response": "You can track your order by logging into your account and clicking on 'Order Status' in the menu."},
    {"query": "What's your refund policy?", "response": "We offer full refunds for items returned within 30 days of purchase, provided they are in their original condition."}
]

customer_query = "How long does shipping take?"
bot_response = customer_support_bot(customer_query, examples)
print(f"Bot: {bot_response}")
```

Slide 13: Real-life Example: Recipe Generator

Another practical application of Few-Shot Prompting in creative tasks, such as generating recipes based on given ingredients and cuisine styles.

```python
def recipe_generator(ingredients, cuisine, examples):
    prompt = f"Generate a {cuisine} recipe using these ingredients: {', '.join(ingredients)}\n\n"
    for ex in examples:
        prompt += f"Cuisine: {ex['cuisine']}\nIngredients: {', '.join(ex['ingredients'])}\nRecipe: {ex['recipe']}\n\n"
    prompt += f"Cuisine: {cuisine}\nIngredients: {', '.join(ingredients)}\nRecipe:"

    # Simulating recipe generation (in reality, you'd use an AI model)
    if "Italian" in cuisine:
        return "1. Cook pasta al dente. 2. Sauté garlic in olive oil. 3. Add tomatoes, simmer. 4. Toss pasta with sauce, add basil."
    elif "Indian" in cuisine:
        return "1. Sauté spices in oil. 2. Add vegetables, cook until tender. 3. Stir in yogurt or coconut milk. 4. Serve with rice or naan."
    else:
        return "1. Prepare ingredients. 2. Cook main component. 3. Add seasonings. 4. Combine all elements. 5. Garnish and serve."

examples = [
    {
        "cuisine": "Mexican",
        "ingredients": ["corn tortillas", "black beans", "avocado", "lime"],
        "recipe": "1. Warm tortillas. 2. Mash beans, spread on tortillas. 3. Top with sliced avocado. 4. Squeeze lime juice over. 5. Roll and serve."
    }
]

ingredients = ["chicken", "rice", "soy sauce", "ginger"]
cuisine = "Asian"
generated_recipe = recipe_generator(ingredients, cuisine, examples)
print(f"Generated Recipe:\n{generated_recipe}")
```

Slide 14: Challenges and Limitations of Few-Shot Prompting

While powerful, Few-Shot Prompting has its limitations. It may struggle with complex tasks or produce inconsistent results. Performance depends heavily on the quality and relevance of provided examples.

```python
import random

def simulate_few_shot_performance(num_examples, task_complexity):
    base_performance = 0.5
    example_benefit = min(0.4, 0.05 * num_examples)
    complexity_penalty = task_complexity * 0.1
    
    performance = base_performance + example_benefit - complexity_penalty
    performance = max(0, min(1, performance))
    
    # Add some randomness to simulate variability
    performance += random.uniform(-0.1, 0.1)
    return max(0, min(1, performance))

num_examples_range = range(1, 11)
complexities = [0.2, 0.5, 0.8]  # Low, Medium, High

for complexity in complexities:
    performances = [simulate_few_shot_performance(n, complexity) for n in num_examples_range]
    plt.plot(num_examples_range, performances, label=f"Complexity: {complexity}")

plt.xlabel("Number of Examples")
plt.ylabel("Performance")
plt.title("Few-Shot Performance vs. Number of Examples and Task Complexity")
plt.legend()
plt.show()
```

Slide 15: Future Directions in Few-Shot Prompting

Research in Few-Shot Prompting is ongoing, with potential advancements in areas such as example selection, prompt optimization, and integration with other AI techniques.

```python
import networkx as nx

def create_research_graph():
    G = nx.Graph()
    nodes = [
        "Few-Shot Prompting",
        "Example Selection",
        "Prompt Optimization",
        "Multi-Task Learning",
        "Meta-Learning",
        "Transfer Learning"
    ]
    G.add_nodes_from(nodes)
    edges = [
        ("Few-Shot Prompting", "Example Selection"),
        ("Few-Shot Prompting", "Prompt Optimization"),
        ("Few-Shot Prompting", "Multi-Task Learning"),
        ("Few-Shot Prompting", "Meta-Learning"),
        ("Few-Shot Prompting", "Transfer Learning"),
        ("Example Selection", "Prompt Optimization"),
        ("Multi-Task Learning", "Meta-Learning"),
        ("Meta-Learning", "Transfer Learning")
    ]
    G.add_edges_from(edges)
    return G

G = create_research_graph()
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')
plt.title("Future Research Directions in Few-Shot Prompting")
plt.axis('off')
plt.tight_layout()
plt.show()
```

Slide 16: Additional Resources

For more information on Few-Shot Prompting and related topics, consider exploring these resources:

1. "Language Models are Few-Shot Learners" by Brown et al. (2020) ArXiv: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
2. "Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm" by Reynolds and McDonell (2021) ArXiv: [https://arxiv.org/abs/2102.07350](https://arxiv.org/abs/2102.07350)
3. "Multitask Prompted Training Enables Zero-Shot Task Generalization" by Sanh et al. (2022) ArXiv: [https://arxiv.org/abs/2110.08207](https://arxiv.org/abs/2110.08207)

These papers provide in-depth discussions on the theory and applications of Few-Shot Prompting in various contexts of natural language processing and machine learning.
