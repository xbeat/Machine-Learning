## Choosing the Right LLM Framework for AI Applications LangChain, LlamaIndex, or Haystack
Slide 1: Introduction to LLM Frameworks

LLM frameworks are essential tools for building AI applications with large language models. This presentation explores three popular frameworks: LangChain, LlamaIndex, and Haystack, comparing their features, use cases, and implementation in Python.

```python
# Importing the frameworks
from langchain import LangChain
from llama_index import LlamaIndex
from haystack import Haystack

# Initialize the frameworks
langchain = LangChain()
llama_index = LlamaIndex()
haystack = Haystack()

print("Frameworks initialized and ready for comparison!")
```

Slide 2: LangChain Overview

LangChain is a framework for developing applications powered by language models. It provides a set of tools and components for building complex AI-powered applications, emphasizing composability and flexibility.

```python
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

# Define a prompt template
template = "What is a {concept} in simple terms?"
prompt = PromptTemplate(template=template, input_variables=["concept"])

# Create an LLMChain
llm = OpenAI(temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt)

# Use the chain
result = chain.run("artificial intelligence")
print(result)
```

Slide 3: LlamaIndex Overview

LlamaIndex is a data framework designed to connect custom data sources to large language models. It focuses on efficient indexing and retrieval of information, making it ideal for building AI applications that require access to specific knowledge bases.

```python
from llama_index import GPTSimpleVectorIndex, Document
from llama_index.readers import SimpleDirectoryReader

# Load documents from a directory
documents = SimpleDirectoryReader('data').load_data()

# Create an index
index = GPTSimpleVectorIndex.from_documents(documents)

# Query the index
response = index.query("What are the key features of LlamaIndex?")
print(response)
```

Slide 4: Haystack Overview

Haystack is an open-source framework for building search systems that work with large document collections. It integrates NLP models and offers various components for document retrieval, question answering, and semantic search.

```python
from haystack import Pipeline
from haystack.nodes import FARMReader, ElasticsearchRetriever

# Initialize components
retriever = ElasticsearchRetriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# Create a pipeline
pipe = Pipeline()
pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipe.add_node(component=reader, name="Reader", inputs=["Retriever"])

# Run the pipeline
results = pipe.run(query="What is Haystack used for?")
print(results)
```

Slide 5: Key Features Comparison

Let's compare the key features of LangChain, LlamaIndex, and Haystack to understand their strengths and use cases.

```python
import pandas as pd

features = {
    'Framework': ['LangChain', 'LlamaIndex', 'Haystack'],
    'Focus': ['Composability', 'Data Indexing', 'Search Systems'],
    'Integration': ['High', 'Medium', 'High'],
    'Scalability': ['Medium', 'High', 'High'],
    'Ease of Use': ['Medium', 'High', 'Medium']
}

comparison = pd.DataFrame(features)
print(comparison)
```

Slide 6: Use Case: Question Answering

Let's implement a simple question-answering system using each framework to compare their approaches and syntax.

```python
# LangChain
from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)
response = conversation.predict(input="What is the capital of France?")
print("LangChain:", response)

# LlamaIndex
from llama_index import GPTSimpleVectorIndex, Document

documents = [Document("Paris is the capital of France.")]
index = GPTSimpleVectorIndex.from_documents(documents)
response = index.query("What is the capital of France?")
print("LlamaIndex:", response)

# Haystack
from haystack import Pipeline
from haystack.nodes import FARMReader, TfidfRetriever

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
retriever = TfidfRetriever(document_store=document_store)
pipe = Pipeline()
pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipe.add_node(component=reader, name="Reader", inputs=["Retriever"])
result = pipe.run(query="What is the capital of France?")
print("Haystack:", result['answers'][0].answer)
```

Slide 7: Performance Comparison

Let's compare the performance of these frameworks for a simple task like text classification.

```python
import time

def measure_performance(framework, task):
    start_time = time.time()
    # Simulating task execution
    time.sleep(1)  # Replace with actual task implementation
    end_time = time.time()
    return end_time - start_time

frameworks = ['LangChain', 'LlamaIndex', 'Haystack']
tasks = ['Text Classification', 'Question Answering', 'Document Retrieval']

results = {}
for framework in frameworks:
    results[framework] = [measure_performance(framework, task) for task in tasks]

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for i, framework in enumerate(frameworks):
    plt.bar([x + i*0.25 for x in range(len(tasks))], results[framework], width=0.25, label=framework)

plt.xlabel('Tasks')
plt.ylabel('Execution Time (seconds)')
plt.title('Performance Comparison')
plt.xticks([x + 0.25 for x in range(len(tasks))], tasks)
plt.legend()
plt.show()
```

Slide 8: Scalability and Integration

Exploring how each framework handles scalability and integrates with existing systems.

```python
import networkx as nx
import matplotlib.pyplot as plt

def create_integration_graph(framework, components):
    G = nx.Graph()
    G.add_node(framework, size=3000)
    for component in components:
        G.add_node(component, size=1000)
        G.add_edge(framework, component)
    return G

langchain_components = ['OpenAI', 'HuggingFace', 'Custom LLMs']
llama_components = ['Vector Stores', 'Document Loaders', 'Query Engines']
haystack_components = ['Elasticsearch', 'FAISS', 'NLP Models']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

for ax, framework, components in zip([ax1, ax2, ax3], 
                                     ['LangChain', 'LlamaIndex', 'Haystack'],
                                     [langchain_components, llama_components, haystack_components]):
    G = create_integration_graph(framework, components)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=[G.nodes[node]['size'] for node in G.nodes()], ax=ax)
    ax.set_title(f"{framework} Integration")

plt.tight_layout()
plt.show()
```

Slide 9: Real-Life Example: Content Summarization

Let's implement a content summarization system using LangChain to demonstrate a practical application.

```python
from langchain import OpenAI, PromptTemplate, LLMChain

llm = OpenAI(temperature=0.7)
template = """
Please summarize the following text in 3 sentences:

{text}

Summary:
"""

prompt = PromptTemplate(template=template, input_variables=["text"])
chain = LLMChain(llm=llm, prompt=prompt)

text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
"""

summary = chain.run(text)
print(summary)
```

Slide 10: Real-Life Example: Document Retrieval

Now, let's implement a document retrieval system using LlamaIndex to showcase another practical application.

```python
from llama_index import GPTSimpleVectorIndex, Document

# Sample documents
documents = [
    Document("The Earth is the third planet from the Sun and the only astronomical object known to harbor life."),
    Document("Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System."),
    Document("Jupiter is the fifth planet from the Sun and the largest in the Solar System.")
]

# Create an index
index = GPTSimpleVectorIndex.from_documents(documents)

# Perform a query
query = "Which planet is known to harbor life?"
response = index.query(query)

print(f"Query: {query}")
print(f"Response: {response}")
```

Slide 11: Customization and Extensibility in LangChain (Continued)

Let's complete our example of creating a custom agent with LangChain:

```python
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper

# Custom tools (as defined before)
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for answering questions about current events"
    )
]

# Custom prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: list[Tool]
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action}\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

# LLM
llm = OpenAI(temperature=0)

# Custom output parser
class CustomOutputParser:
    def parse(self, llm_output: str) -> Dict:
        # Parse the output and return a dictionary
        # This is a simplified version
        return {"action": "Search", "action_input": llm_output}

output_parser = CustomOutputParser()

# Create the agent
agent = LLMSingleActionAgent(
    llm_chain=LLMChain(llm=llm, prompt=prompt),
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=[tool.name for tool in tools]
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# Example usage
result = agent_executor.run("What's the latest news about AI?")
print(result)
```

This example demonstrates how LangChain allows for extensive customization, including defining custom tools, creating tailored prompt templates, and implementing custom output parsers. This flexibility enables developers to build highly specialized AI applications that can adapt to specific use cases and requirements.

Slide 12: Comparing Framework Ecosystems

Let's visualize the ecosystem and community support for each framework:

```python
import matplotlib.pyplot as plt

frameworks = ['LangChain', 'LlamaIndex', 'Haystack']
github_stars = [20000, 15000, 10000]  # Approximate values
contributors = [500, 300, 200]  # Approximate values
releases = [50, 30, 40]  # Approximate values

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.bar(frameworks, github_stars)
ax1.set_title('GitHub Stars')
ax1.set_ylabel('Number of Stars')

ax2.bar(frameworks, contributors)
ax2.set_title('Contributors')
ax2.set_ylabel('Number of Contributors')

ax3.bar(frameworks, releases)
ax3.set_title('Releases')
ax3.set_ylabel('Number of Releases')

plt.tight_layout()
plt.show()
```

This visualization provides a comparative view of the community engagement and development activity for each framework, helping developers assess the ecosystem support and potential longevity of each tool.

Slide 13: Choosing the Right Framework

When selecting an LLM framework for your AI application, consider these factors:

1. Use case complexity
2. Required customization
3. Integration with existing systems
4. Scalability needs
5. Community support and ecosystem
6. Learning curve and documentation

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edge("Use Case", "LangChain", weight=3)
G.add_edge("Use Case", "LlamaIndex", weight=2)
G.add_edge("Use Case", "Haystack", weight=2)
G.add_edge("Customization", "LangChain", weight=3)
G.add_edge("Customization", "LlamaIndex", weight=2)
G.add_edge("Customization", "Haystack", weight=2)
G.add_edge("Integration", "LangChain", weight=3)
G.add_edge("Integration", "LlamaIndex", weight=2)
G.add_edge("Integration", "Haystack", weight=3)
G.add_edge("Scalability", "LangChain", weight=2)
G.add_edge("Scalability", "LlamaIndex", weight=3)
G.add_edge("Scalability", "Haystack", weight=3)
G.add_edge("Community", "LangChain", weight=3)
G.add_edge("Community", "LlamaIndex", weight=2)
G.add_edge("Community", "Haystack", weight=2)
G.add_edge("Learning Curve", "LangChain", weight=2)
G.add_edge("Learning Curve", "LlamaIndex", weight=3)
G.add_edge("Learning Curve", "Haystack", weight=2)

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Framework Selection Factors")
plt.axis('off')
plt.tight_layout()
plt.show()
```

This graph visualizes the relationships between different factors and frameworks, helping you make an informed decision based on your specific requirements.

Slide 14: Additional Resources

For further exploration of LLM frameworks, consider these resources:

1. LangChain Documentation: [https://python.langchain.com/en/latest/](https://python.langchain.com/en/latest/)
2. LlamaIndex GitHub Repository: [https://github.com/jerryjliu/llama\_index](https://github.com/jerryjliu/llama_index)
3. Haystack Documentation: [https://haystack.deepset.ai/overview/intro](https://haystack.deepset.ai/overview/intro)

For academic papers on LLMs and their applications:

1. "Language Models are Few-Shot Learners" (GPT-3 paper): [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
2. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks": [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
3. "REALM: Retrieval-Augmented Language Model Pre-Training": [https://arxiv.org/abs/2002.08909](https://arxiv.org/abs/2002.08909)

These resources provide in-depth information on each framework and the underlying concepts of large language models and their applications in AI systems.

