## Core Components of Agent Decision-Making in LangChain

Slide 1: Core Components of Agent Decision-Making in LangChain

LangChain is a framework for developing applications powered by language models. In the context of agent decision-making, it provides several core components that work together to create intelligent agents capable of reasoning and taking actions. These components include the language model, memory, tools, and the agent itself.

```python
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from langchain.memory import ConversationBufferMemory

# Initialize the language model
llm = OpenAI(temperature=0)

# Define a tool
search_tool = Tool(
    name="Search",
    func=lambda x: f"Search results for: {x}",
    description="Useful for searching information"
)

# Create memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Define the agent
agent = LLMSingleActionAgent(
    llm_chain=LLMChain(llm=llm, prompt=StringPromptTemplate.from_template("")),
    output_parser=lambda x: x,
    stop=["\nObservation:"],
    allowed_tools=["Search"]
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[search_tool],
    memory=memory
)
```

Slide 2: Language Models in LangChain

The foundation of agent decision-making in LangChain is the language model. These models, such as GPT-3 or GPT-4, are responsible for understanding and generating human-like text. They form the core reasoning engine of the agent, interpreting input, generating responses, and making decisions based on the given context.

```python
from langchain.llms import OpenAI

# Initialize a language model with specific parameters
llm = OpenAI(
    model_name="text-davinci-002",
    temperature=0.7,
    max_tokens=100
)

# Use the language model to generate a response
prompt = "Explain the concept of artificial intelligence in simple terms."
response = llm(prompt)

print(response)
```

Slide 3: Memory Systems in LangChain

Memory systems in LangChain allow agents to retain and recall information from previous interactions. This is crucial for maintaining context and making informed decisions over time. LangChain offers various types of memory, including short-term and long-term memory implementations.

```python
from langchain.memory import ConversationBufferMemory

# Initialize a conversation buffer memory
memory = ConversationBufferMemory()

# Add a conversation turn to memory
memory.save_context({"input": "Hello, how are you?"}, {"output": "I'm doing well, thank you for asking!"})

# Retrieve the conversation history
history = memory.load_memory_variables({})

print(history)
```

Slide 4: Tools and Actions in LangChain

Tools in LangChain represent the actions an agent can take. These can include API calls, database queries, or any custom function. Tools allow agents to interact with external systems and perform specific tasks as part of their decision-making process.

```python
from langchain.agents import Tool

def search_function(query):
    # Simulated search function
    return f"Search results for: {query}"

# Define a search tool
search_tool = Tool(
    name="Search",
    func=search_function,
    description="Useful for searching information on the internet"
)

# Use the tool
result = search_tool.run("Latest AI advancements")
print(result)
```

Slide 5: Agent Types in LangChain

LangChain provides various agent types, each with different decision-making strategies. These include zero-shot agents, which can use tools without examples, and few-shot agents, which learn from a few examples. The choice of agent type depends on the specific use case and the level of guidance required.

```python
from langchain.agents import ZeroShotAgent, Tool
from langchain.llms import OpenAI

# Define a simple tool
simple_tool = Tool(
    name="Simple",
    func=lambda x: x,
    description="A simple echo tool"
)

# Create a zero-shot agent
agent = ZeroShotAgent.from_llm_and_tools(
    llm=OpenAI(temperature=0),
    tools=[simple_tool],
    prefix="Answer the following questions as best you can:",
    suffix="Question: {input}\nThought: Let's approach this step-by-step:\n{agent_scratchpad}"
)

# Use the agent
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[simple_tool],
    verbose=True
)

agent_executor.run("Echo the word 'Hello'")
```

Slide 6: Prompt Templates in LangChain

Prompt templates in LangChain are crucial for structuring the input to language models. They allow for dynamic generation of prompts based on variables, ensuring consistent and effective communication with the model.

```python
from langchain.prompts import PromptTemplate

# Define a prompt template
template = """
You are a helpful AI assistant. The user has asked: {question}
Please provide a clear and concise answer.
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

# Generate a prompt
question = "What is the capital of France?"
formatted_prompt = prompt.format(question=question)

print(formatted_prompt)
```

Slide 7: Chains in LangChain

Chains in LangChain allow for the composition of multiple components into a single, coherent workflow. They can combine language models, prompts, and other elements to create more complex decision-making processes.

```python
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Define a prompt template
template = "What is the capital of {country}?"
prompt = PromptTemplate(input_variables=["country"], template=template)

# Create an LLM chain
llm = OpenAI(temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
result = chain.run("France")
print(result)
```

Slide 8: Output Parsers in LangChain

Output parsers in LangChain are responsible for interpreting the raw output from language models into structured formats. This is essential for agents to make use of the generated information in a programmatic way.

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Define an output parser
output_parser = CommaSeparatedListOutputParser()

# Create a prompt template with formatting instructions
template = """
Generate a comma-separated list of 5 {subject}.

{format_instructions}
"""

prompt = PromptTemplate(
    input_variables=["subject"],
    template=template,
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

# Generate and parse the output
llm = OpenAI(temperature=0.7)
subject = "fruits"
output = llm(prompt.format(subject=subject))
parsed_output = output_parser.parse(output)

print(parsed_output)
```

Slide 9: Real-Life Example: Customer Support Agent

Let's create a simple customer support agent using LangChain components. This agent will handle basic inquiries and provide appropriate responses.

```python
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from langchain.memory import ConversationBufferMemory

# Define tools for the agent
def get_product_info(product):
    products = {
        "laptop": "Our latest laptop model with 16GB RAM and 512GB SSD",
        "smartphone": "A high-end smartphone with 5G capability and 128GB storage"
    }
    return products.get(product, "Product information not available")

def check_order_status(order_id):
    # Simulated order status checker
    return f"Order {order_id} is currently in transit"

product_tool = Tool(name="ProductInfo", func=get_product_info, description="Get information about a product")
order_tool = Tool(name="OrderStatus", func=check_order_status, description="Check the status of an order")

# Create the agent
llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")

agent = LLMSingleActionAgent(
    llm_chain=LLMChain(llm=llm, prompt=StringPromptTemplate.from_template(
        "You are a customer support agent. Respond to the following inquiry: {input}"
    )),
    output_parser=lambda x: x,
    stop=["\nObservation:"],
    allowed_tools=["ProductInfo", "OrderStatus"]
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[product_tool, order_tool],
    memory=memory,
    verbose=True
)

# Use the agent
response = agent_executor.run("Can you tell me about your laptop?")
print(response)

response = agent_executor.run("What's the status of order 12345?")
print(response)
```

Slide 10: Late Chunking in Retrieval Systems

Late chunking is a technique used in retrieval systems to improve the relevance and efficiency of search results. Unlike traditional methods that chunk documents before indexing, late chunking processes the entire document and only chunks the relevant portions at query time.

```python
import re

class LateChunkingRetriever:
    def __init__(self, documents):
        self.documents = documents

    def search(self, query, chunk_size=100):
        results = []
        for doc in self.documents:
            # Find all occurrences of the query in the document
            matches = re.finditer(re.escape(query), doc, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - chunk_size // 2)
                end = min(len(doc), match.end() + chunk_size // 2)
                chunk = doc[start:end]
                results.append(chunk)
        return results

# Example usage
documents = [
    "The quick brown fox jumps over the lazy dog. The fox is very quick and agile.",
    "Artificial intelligence is revolutionizing various industries. AI applications are becoming more common."
]

retriever = LateChunkingRetriever(documents)
results = retriever.search("fox")
print(results)
```

Slide 11: Benefits of Late Chunking

Late chunking offers several advantages over traditional chunking methods. It preserves context around the query terms, reduces index size, and allows for dynamic chunk sizing based on the query. This approach can lead to more relevant and coherent search results.

```python
import re

def traditional_chunking(documents, chunk_size=50):
    chunks = []
    for doc in documents:
        chunks.extend([doc[i:i+chunk_size] for i in range(0, len(doc), chunk_size)])
    return chunks

def late_chunking(documents, query, context_size=50):
    results = []
    for doc in documents:
        matches = re.finditer(re.escape(query), doc, re.IGNORECASE)
        for match in matches:
            start = max(0, match.start() - context_size)
            end = min(len(doc), match.end() + context_size)
            results.append(doc[start:end])
    return results

# Example usage
documents = [
    "The quick brown fox jumps over the lazy dog. The fox is very quick and agile.",
    "Artificial intelligence is revolutionizing various industries. AI applications are becoming more common."
]

print("Traditional Chunking:")
print(traditional_chunking(documents))

print("\nLate Chunking:")
print(late_chunking(documents, "fox"))
```

Slide 12: Implementing Late Chunking in a Retrieval System

Let's implement a basic retrieval system that uses late chunking to improve search results. This system will index documents and perform searches using late chunking techniques.

```python
import re
from collections import defaultdict

class LateChunkingRetriever:
    def __init__(self):
        self.documents = []
        self.index = defaultdict(list)

    def add_document(self, doc_id, content):
        self.documents.append((doc_id, content))
        words = re.findall(r'\w+', content.lower())
        for word in set(words):
            self.index[word].append(doc_id)

    def search(self, query, context_size=50):
        query_words = re.findall(r'\w+', query.lower())
        relevant_docs = set.intersection(*[set(self.index[word]) for word in query_words])
        
        results = []
        for doc_id in relevant_docs:
            doc_content = next(content for id, content in self.documents if id == doc_id)
            chunks = self._late_chunk(doc_content, query, context_size)
            results.extend((doc_id, chunk) for chunk in chunks)
        
        return results

    def _late_chunk(self, content, query, context_size):
        chunks = []
        matches = re.finditer(re.escape(query), content, re.IGNORECASE)
        for match in matches:
            start = max(0, match.start() - context_size)
            end = min(len(content), match.end() + context_size)
            chunks.append(content[start:end])
        return chunks

# Example usage
retriever = LateChunkingRetriever()
retriever.add_document(1, "The quick brown fox jumps over the lazy dog. The fox is very quick and agile.")
retriever.add_document(2, "Artificial intelligence is revolutionizing various industries. AI applications are becoming more common.")

results = retriever.search("fox")
for doc_id, chunk in results:
    print(f"Document {doc_id}: {chunk}")
```

Slide 13: Real-Life Example: Document Search Engine

Let's create a simple document search engine that uses late chunking to provide more relevant search results. This example demonstrates how late chunking can be applied in a practical scenario.

```python
import re
from collections import defaultdict

class DocumentSearchEngine:
    def __init__(self):
        self.documents = {}
        self.index = defaultdict(set)

    def add_document(self, doc_id, title, content):
        self.documents[doc_id] = {"title": title, "content": content}
        words = re.findall(r'\w+', content.lower())
        for word in set(words):
            self.index[word].add(doc_id)

    def search(self, query, context_size=50):
        query_words = re.findall(r'\w+', query.lower())
        relevant_docs = set.intersection(*[self.index[word] for word in query_words])
        
        results = []
        for doc_id in relevant_docs:
            doc = self.documents[doc_id]
            chunks = self._late_chunk(doc["content"], query, context_size)
            results.extend({"doc_id": doc_id, "title": doc["title"], "chunk": chunk} for chunk in chunks)
        
        return results

    def _late_chunk(self, content, query, context_size):
        chunks = []
        matches = re.finditer(re.escape(query), content, re.IGNORECASE)
        for match in matches:
            start = max(0, match.start() - context_size)
            end = min(len(content), match.end() + context_size)
            chunks.append(content[start:end])
        return chunks
```

Slide 14: Using the Document Search Engine

Now let's see how to use our Document Search Engine with late chunking in practice.

```python
# Initialize the search engine
search_engine = DocumentSearchEngine()

# Add some sample documents
search_engine.add_document(1, "The Fox and the Hound", "The quick brown fox jumps over the lazy dog. The fox is very quick and agile.")
search_engine.add_document(2, "AI Revolution", "Artificial intelligence is revolutionizing various industries. AI applications are becoming more common.")
search_engine.add_document(3, "Wildlife Behavior", "Foxes are known for their cunning behavior. They are adaptable and can thrive in various environments.")

# Perform a search
results = search_engine.search("fox")

# Display the results
for result in results:
    print(f"Document ID: {result['doc_id']}")
    print(f"Title: {result['title']}")
    print(f"Chunk: {result['chunk']}")
    print("---")
```

Slide 15: Additional Resources

For those interested in diving deeper into the topics of agent decision-making in LangChain and late chunking in retrieval systems, here are some valuable resources:

1.  LangChain Documentation: [https://python.langchain.com/en/latest/](https://python.langchain.com/en/latest/)
2.  "Attention Is All You Need" paper (foundational for many language models): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
3.  "REALM: Retrieval-Augmented Language Model Pre-Training" (discusses retrieval techniques): [https://arxiv.org/abs/2002.08909](https://arxiv.org/abs/2002.08909)
4.  "Dense Passage Retrieval for Open-Domain Question Answering" (relevant to retrieval systems): [https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)

These resources provide in-depth information on the concepts and techniques discussed in this presentation.

