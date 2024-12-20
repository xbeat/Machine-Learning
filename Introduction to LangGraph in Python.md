## Exploring Conditional Probability with Python:
Slide 1: Introduction to LangGraph

LangGraph is a library for building stateful, multi-agent applications with large language models (LLMs). It extends the functionality of LangChain, providing tools for creating complex workflows and interactions between different AI agents.

```python
import langgraph as lg
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Initialize LangGraph components
graph = lg.Graph()
llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("Hello, {name}!")
```

Slide 2: Core Concepts of LangGraph

LangGraph introduces the concept of a graph-based workflow for LLM applications. It allows developers to define nodes (representing tasks or agents) and edges (representing the flow of information between nodes).

```python
# Define nodes in the graph
@graph.node
def greeting(state):
    name = state['name']
    response = llm.invoke(prompt.format(name=name))
    return {"greeting": response}

@graph.node
def farewell(state):
    return {"farewell": "Goodbye, " + state['name'] + "!"}

# Define edges
graph.add_edge('greeting', 'farewell')
```

Slide 3: State Management in LangGraph

LangGraph manages state throughout the execution of a graph. This allows for complex, multi-step interactions where information is passed between different nodes.

```python
# Initialize state
initial_state = {"name": "Alice"}

# Run the graph
final_state = graph.run(initial_state)

print(final_state)
# Output: {'name': 'Alice', 'greeting': 'Hello, Alice!', 'farewell': 'Goodbye, Alice!'}
```

Slide 4: Conditional Flows

LangGraph supports conditional flows, allowing for dynamic routing based on the current state or output of previous nodes.

```python
@graph.node
def check_mood(state):
    mood = state.get('mood', 'neutral')
    if mood == 'happy':
        return {'next': 'positive_response'}
    else:
        return {'next': 'neutral_response'}

graph.add_edge('check_mood', 'positive_response', condition=lambda x: x['next'] == 'positive_response')
graph.add_edge('check_mood', 'neutral_response', condition=lambda x: x['next'] == 'neutral_response')
```

Slide 5: Parallel Execution

LangGraph allows for parallel execution of nodes, enabling efficient processing of independent tasks.

```python
@graph.node
def task_a(state):
    return {"result_a": "Task A completed"}

@graph.node
def task_b(state):
    return {"result_b": "Task B completed"}

graph.add_edge('start', ['task_a', 'task_b'])
graph.add_edge(['task_a', 'task_b'], 'end')

result = graph.run({"start": True})
print(result)
# Output: {'start': True, 'result_a': 'Task A completed', 'result_b': 'Task B completed'}
```

Slide 6: Error Handling

LangGraph provides mechanisms for handling errors and exceptions within the graph execution.

```python
@graph.node
def risky_operation(state):
    if state.get('safe', False):
        return {"result": "Operation successful"}
    else:
        raise ValueError("Unsafe operation")

@graph.node
def error_handler(state, error):
    return {"error_message": str(error)}

graph.add_edge('risky_operation', 'success', condition=lambda x: 'result' in x)
graph.add_edge('risky_operation', 'error_handler', on_error=True)

result = graph.run({"safe": False})
print(result)
# Output: {'safe': False, 'error_message': 'Unsafe operation'}
```

Slide 7: Integrating External APIs

LangGraph can be used to integrate external APIs into your LLM workflows, allowing for rich, data-driven applications.

```python
import requests

@graph.node
def fetch_weather(state):
    city = state['city']
    api_key = "your_api_key_here"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    return {"weather": data['weather'][0]['description']}

@graph.node
def generate_weather_report(state):
    weather = state['weather']
    report = llm.invoke(f"Generate a weather report for {weather} conditions.")
    return {"report": report}

graph.add_edge('fetch_weather', 'generate_weather_report')
```

Slide 8: Real-Life Example: Customer Support Chatbot

Let's create a simple customer support chatbot using LangGraph that can handle different types of inquiries.

```python
@graph.node
def classify_inquiry(state):
    inquiry = state['user_input']
    classification = llm.invoke(f"Classify this inquiry: {inquiry}")
    return {"inquiry_type": classification}

@graph.node
def handle_product_inquiry(state):
    inquiry = state['user_input']
    response = llm.invoke(f"Provide product information for: {inquiry}")
    return {"bot_response": response}

@graph.node
def handle_shipping_inquiry(state):
    inquiry = state['user_input']
    response = llm.invoke(f"Provide shipping information for: {inquiry}")
    return {"bot_response": response}

graph.add_edge('classify_inquiry', 'handle_product_inquiry', condition=lambda x: 'product' in x['inquiry_type'].lower())
graph.add_edge('classify_inquiry', 'handle_shipping_inquiry', condition=lambda x: 'shipping' in x['inquiry_type'].lower())

result = graph.run({"user_input": "When will my order arrive?"})
print(result['bot_response'])
```

Slide 9: Persistent State Across Conversations

LangGraph allows for maintaining state across multiple interactions, enabling more context-aware conversations.

```python
class ConversationState:
    def __init__(self):
        self.history = []

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_context(self):
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history[-5:]])

conv_state = ConversationState()

@graph.node
def process_user_input(state):
    user_input = state['user_input']
    conv_state.add_message("user", user_input)
    context = conv_state.get_context()
    response = llm.invoke(f"Given this context:\n{context}\nRespond to: {user_input}")
    conv_state.add_message("assistant", response)
    return {"bot_response": response}

# Usage
for user_input in ["Hello!", "What's the weather like?", "Thank you!"]:
    result = graph.run({"user_input": user_input})
    print(f"User: {user_input}")
    print(f"Bot: {result['bot_response']}")
```

Slide 10: Dynamic Graph Modification

LangGraph supports dynamic modification of the graph structure during runtime, allowing for adaptive workflows.

```python
@graph.node
def assess_complexity(state):
    query = state['user_query']
    complexity = llm.invoke(f"Assess the complexity of this query: {query}")
    return {"complexity": complexity}

@graph.node
def simple_response(state):
    query = state['user_query']
    return {"response": llm.invoke(f"Provide a simple answer to: {query}")}

@graph.node
def complex_response(state):
    query = state['user_query']
    return {"response": llm.invoke(f"Provide a detailed, expert-level answer to: {query}")}

@graph.node
def route_query(state):
    if state['complexity'] == 'simple':
        graph.add_edge('route_query', 'simple_response')
    else:
        graph.add_edge('route_query', 'complex_response')
    return {}

graph.add_edge('assess_complexity', 'route_query')

result = graph.run({"user_query": "What is photosynthesis?"})
print(result['response'])
```

Slide 11: Multi-Agent Collaboration

LangGraph can orchestrate collaboration between multiple AI agents, each specialized in different tasks.

```python
@graph.node
def researcher(state):
    topic = state['topic']
    research = llm.invoke(f"Conduct research on: {topic}")
    return {"research": research}

@graph.node
def writer(state):
    research = state['research']
    article = llm.invoke(f"Write an article based on this research: {research}")
    return {"article": article}

@graph.node
def editor(state):
    article = state['article']
    edited_article = llm.invoke(f"Edit and improve this article: {article}")
    return {"final_article": edited_article}

graph.add_edge('researcher', 'writer')
graph.add_edge('writer', 'editor')

result = graph.run({"topic": "Artificial Intelligence in Healthcare"})
print(result['final_article'])
```

Slide 12: Real-Life Example: Recipe Generator

Let's create a recipe generator that takes dietary preferences and available ingredients into account.

```python
@graph.node
def get_preferences(state):
    preferences = state['user_preferences']
    return {"dietary_restrictions": llm.invoke(f"Extract dietary restrictions from: {preferences}")}

@graph.node
def inventory_check(state):
    ingredients = state['available_ingredients']
    return {"usable_ingredients": llm.invoke(f"List ingredients that can be used in a recipe: {ingredients}")}

@graph.node
def generate_recipe(state):
    restrictions = state['dietary_restrictions']
    ingredients = state['usable_ingredients']
    recipe = llm.invoke(f"Generate a recipe considering these restrictions: {restrictions} and using these ingredients: {ingredients}")
    return {"recipe": recipe}

graph.add_edge('get_preferences', 'inventory_check')
graph.add_edge('inventory_check', 'generate_recipe')

result = graph.run({
    "user_preferences": "vegetarian, no nuts",
    "available_ingredients": "tomatoes, pasta, garlic, olive oil, basil"
})
print(result['recipe'])
```

Slide 13: Debugging and Logging

LangGraph provides tools for debugging and logging the execution of your graph, which is crucial for complex workflows.

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@graph.node
def log_state(state):
    logger.info(f"Current state: {state}")
    return {}

@graph.node
def process_data(state):
    data = state['input_data']
    result = llm.invoke(f"Process this data: {data}")
    logger.info(f"Processed data: {result}")
    return {"processed_data": result}

graph.add_edge('log_state', 'process_data')
graph.add_edge('process_data', 'log_state')

graph.run({"input_data": "Sample input for processing"})
```

Slide 14: Additional Resources

For more information on LangGraph and its applications, consider exploring the following resources:

1. LangChain Documentation: [https://python.langchain.com/docs/get\_started/introduction](https://python.langchain.com/docs/get_started/introduction)
2. "Large Language Models and Graph-based Reasoning" (ArXiv:2307.05722): [https://arxiv.org/abs/2307.05722](https://arxiv.org/abs/2307.05722)
3. "Graph-based Reasoning with Large Language Models" (ArXiv:2305.15117): [https://arxiv.org/abs/2305.15117](https://arxiv.org/abs/2305.15117)

These resources provide deeper insights into the concepts and techniques used in LangGraph and similar graph-based approaches to LLM applications.

