## Building a Custom LangChain QnA Agent with Wikipedia Using Python

Slide 1: 

Introduction to LangChain

LangChain is a framework for building applications with large language models (LLMs). It provides abstractions for working with LLMs, data retrieval, and combining various components for advanced use cases.

```python
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

template = """
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0))

question = "What is the capital of France?"
print(llm_chain.run(question))
```

Slide 2: 

Setting up a Custom QnA Agent

To create a custom QnA agent with memory using Wikipedia as the information source, we need to initialize a few components from LangChain.

```python
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.utilities import WikipediaAPIWrapper

# Initialize the memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize the Wikipedia data retriever
wikipedia = WikipediaAPIWrapper()

# Initialize the language model
llm = OpenAI(temperature=0)
```

Slide 3: 

Creating the QnA Agent

We can now create the QnA agent by specifying the language model, the memory component, and the data retriever (Wikipedia API).

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for searching Wikipedia for information to answer queries"
    )
]

agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True)
```

Slide 4: 

Interacting with the QnA Agent

We can now ask questions to the QnA agent, and it will retrieve relevant information from Wikipedia and provide accurate answers while maintaining conversational context.

```python
query = "What is the capital of France?"
response = agent.run(query)
print(response)

query = "What is the population of Paris?"
response = agent.run(query)
print(response)
```

Slide 5: 

Handling Wikipedia Queries

The `WikipediaAPIWrapper` class from LangChain provides a convenient way to search and retrieve information from Wikipedia.

```python
from langchain.utilities import WikipediaAPIWrapper

wikipedia = WikipediaAPIWrapper()

# Search for a specific topic
result = wikipedia.run("Paris")
print(result)

# Search for a topic and get the first n characters
result = wikipedia.run("Paris", length=500)
print(result)
```

Slide 6: 

Memory Management

The `ConversationBufferMemory` class from LangChain allows the agent to maintain conversational context and refer to previous interactions.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

# Add a new interaction to memory
memory.save_context({"input": "What is the capital of France?"}, {"output": "The capital of France is Paris."})

# Retrieve the conversation history
print(memory.buffer)

# Clear the memory buffer
memory.clear()
```

Slide 7: 

Customizing the Language Model

LangChain supports various language models, including OpenAI, Anthropic, and others. You can customize the language model's behavior by adjusting parameters like temperature and max tokens.

```python
from langchain.llms import OpenAI

# Initialize the language model
llm = OpenAI(temperature=0.7, max_tokens=512)

# Use the language model for text generation
prompt = "Once upon a time, there was a"
output = llm(prompt)
print(output)
```

Slide 8: 

Combining Components

LangChain allows you to combine various components like language models, memory, and data retrievers to build complex applications.

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain

# Initialize the language model and memory
llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize the Wikipedia data retriever
wikipedia = WikipediaAPIWrapper()

# Create a conversational question-answering chain
qa_chain = load_qa_chain(llm, chain_type="stuff", memory=memory)
retriever = wikipedia.get_retriever()
chain = ConversationalRetrievalChain(retriever=retriever, question_answering_chain=qa_chain)

# Ask questions and receive answers
query = "What is the capital of France?"
result = chain({"question": query})
print(result["result"])
```

Slide 9: 

Handling Multiple Data Sources

LangChain supports integrating multiple data sources, such as databases, APIs, and file systems, to build more comprehensive applications.

```python
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader

# Load documents from a file
loader = TextLoader("path/to/file.txt")
documents = loader.load()

# Create a vector store index
index = VectorstoreIndexCreator().from_loaders([loader])

# Use the index for question answering
query = "What is the main topic of the document?"
result = index.query(query)
print(result)
```

Slide 10: 

Evaluation and Feedback

LangChain provides tools for evaluating the performance of your QnA agent and incorporating feedback to improve its accuracy.

```python
from langchain.evaluation import load_metric
from langchain.evaluation.qa import QAEvalChain

# Initialize the language model and memory
llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")

# Load the evaluation dataset
dataset = load_metric("squad")

# Create an evaluation chain
eval_chain = QAEvalChain.from_llm(llm, memory=memory)

# Evaluate the agent's performance
metrics = eval_chain.evaluate(dataset)
print(metrics)
```

Slide 11: 

Deployment and Scaling

LangChain provides utilities for deploying and scaling your QnA agent, whether it's on a local machine, a cloud platform, or a serverless environment.

```python
from langchain.callbacks import TrainingCallback
from langchain.agents import AgentExecutor

# Define a custom callback for monitoring
class MyCallback(TrainingCallback):
    def on_llm_start(self, **kwargs):
        print("Starting LLM call...")

    def on_llm_end(self, **kwargs):
        print("Finished LLM call.")

# Initialize the agent and the executor
agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True)
executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, callbacks=[MyCallback()])

# Run the agent
query = "What is the capital of France?"
result = executor.run(query)
print(result)
```

Slide 12: 

Advanced Topics

LangChain provides advanced features and integrations for more complex use cases, such as fine-tuning language models, building chatbots, and integrating with other frameworks and libraries.

```python
from langchain.agents import load_agent
from langchain.agents import AgentType

# Load a pre-trained agent
agent = load_agent(AgentType.CONVERSATIONAL_REACT_DESCRIPTION, llm=llm, memory=memory)

# Fine-tune the agent
agent.fine_tune(dataset)

# Interact with the fine-tuned agent
query = "What is the capital of Germany?"
result = agent.run(query)
print(result)
```

Slide 13: 

Resources and Community

LangChain has an active community and extensive documentation to help you get started and explore advanced topics.

```python
# Explore the LangChain documentation
import webbrowser
webbrowser.open("https://python.langchain.com/en/latest/index.html")

# Join the LangChain community
print("GitHub: https://github.com/hwchase17/langchain")
print("Discord: https://discord.gg/6AdqQrAkkN")
print("Twitter: https://twitter.com/LangChainAI")
```
