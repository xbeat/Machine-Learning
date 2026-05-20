## Basic LangChain Tutorial with Python
Slide 1: Introduction to LangChain

LangChain is a powerful framework for developing applications powered by language models. It provides a set of tools and components that simplify the process of building complex language model applications. In this tutorial, we'll explore the basics of LangChain using Python.

```python
from langchain import LLMChain, OpenAI, PromptTemplate

# Initialize the language model
llm = OpenAI(temperature=0.9)

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# Create the LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
print(chain.run("eco-friendly water bottles"))
```

Slide 2: Setting Up LangChain

To get started with LangChain, you need to install it and set up your environment. This slide covers the installation process and initial configuration.

```python
# Install LangChain
!pip install langchain

# Import necessary modules
from langchain import LLMChain, OpenAI, PromptTemplate

# Set up your OpenAI API key
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Test the setup
llm = OpenAI(temperature=0.7)
text = "What is the capital of France?"
print(llm(text))
```

Slide 3: Understanding Prompt Templates

Prompt templates are a key concept in LangChain. They allow you to create reusable prompts with variable inputs, making it easy to generate dynamic content.

```python
from langchain import PromptTemplate

# Create a simple prompt template
template = "Tell me a {adjective} joke about {topic}."
prompt = PromptTemplate(
    input_variables=["adjective", "topic"],
    template=template,
)

# Use the prompt template
result = prompt.format(adjective="funny", topic="programming")
print(result)
```

Slide 4: Working with Language Models

LangChain supports various language models. This slide demonstrates how to use different models and adjust their parameters.

```python
from langchain.llms import OpenAI, HuggingFaceHub

# Initialize OpenAI model
openai_llm = OpenAI(temperature=0.5, max_tokens=100)

# Initialize HuggingFace model
huggingface_llm = HuggingFaceHub(repo_id="gpt2", model_kwargs={"temperature": 0.7, "max_length": 100})

# Use the models
prompt = "Explain the concept of artificial intelligence in simple terms."
print("OpenAI response:", openai_llm(prompt))
print("HuggingFace response:", huggingface_llm(prompt))
```

Slide 5: Chains in LangChain

Chains are a fundamental concept in LangChain, allowing you to combine multiple components to create more complex workflows.

```python
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI

# Create two prompt templates
name_prompt = PromptTemplate(
    input_variables=["product"],
    template="Suggest a name for a company that makes {product}.",
)
slogan_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Create a catchy slogan for {company_name}.",
)

# Create two LLMChains
llm = OpenAI(temperature=0.7)
name_chain = LLMChain(llm=llm, prompt=name_prompt)
slogan_chain = LLMChain(llm=llm, prompt=slogan_prompt)

# Combine the chains
def generate_company_branding(product):
    company_name = name_chain.run(product)
    slogan = slogan_chain.run(company_name)
    return f"Company Name: {company_name}\nSlogan: {slogan}"

print(generate_company_branding("smart home devices"))
```

Slide 6: Memory in LangChain

LangChain provides memory components to maintain context across multiple interactions. This slide explores how to implement and use memory in your applications.

```python
from langchain import OpenAI, ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize the language model and memory
llm = OpenAI(temperature=0.7)
memory = ConversationBufferMemory()

# Create a conversation chain with memory
conversation = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose=True
)

# Have a conversation
print(conversation.predict(input="Hi, my name is Alice."))
print(conversation.predict(input="What's my name?"))
print(conversation.predict(input="What have we talked about so far?"))
```

Slide 7: Agents and Tools

Agents in LangChain can use tools to perform actions and gather information. This slide demonstrates how to create and use agents with custom tools.

```python
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

# Initialize the language model
llm = OpenAI(temperature=0)

# Load some tools
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# Create an agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use the agent
agent.run("What is the square root of the year Pythagoras was born?")
```

Slide 8: Document Loading and Text Splitting

LangChain provides utilities for loading and processing documents. This slide covers document loading and text splitting techniques.

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load a document
loader = TextLoader("path/to/your/document.txt")
document = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(document)

print(f"Number of document chunks: {len(texts)}")
print(f"First chunk: {texts[0].page_content[:100]}...")
```

Slide 9: Embeddings and Vector Stores

Embeddings are crucial for many language model applications. This slide explores how to generate embeddings and use vector stores in LangChain.

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Initialize the embedding model
embeddings = OpenAIEmbeddings()

# Create a vector store
texts = ["LangChain is awesome", "Vector databases are useful", "Large language models are powerful"]
vectorstore = Chroma.from_texts(texts, embeddings)

# Perform a similarity search
query = "What can language models do?"
results = vectorstore.similarity_search(query)

print(f"Most similar text to '{query}':")
print(results[0].page_content)
```

Slide 10: Question Answering with LangChain

LangChain simplifies the process of building question-answering systems. This slide demonstrates how to create a basic QA system using LangChain.

```python
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Load and preprocess the document
loader = TextLoader("path/to/your/document.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create a vector store
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)

# Initialize the QA chain
chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Answer a question
query = "What is the main topic of this document?"
docs = docsearch.similarity_search(query)
print(chain.run(input_documents=docs, question=query))
```

Slide 11: Real-Life Example: Automated News Summarizer

This slide presents a practical example of using LangChain to create an automated news summarizer.

```python
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter

# Load a news article
loader = WebBaseLoader("https://www.bbc.com/news/world-europe-66344811")
document = loader.load()

# Split the document
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(document)

# Create a summarization chain
llm = OpenAI(temperature=0.7)
template = "Summarize the following news article in 3 sentences:\n\n{text}"
prompt = PromptTemplate(template=template, input_variables=["text"])
chain = LLMChain(llm=llm, prompt=prompt)

# Generate summary
summary = chain.run(text=texts[0].page_content)
print("News Summary:")
print(summary)
```

Slide 12: Real-Life Example: Automated Recipe Generator

This slide showcases another practical application of LangChain: an automated recipe generator based on available ingredients.

```python
from langchain import OpenAI, PromptTemplate, LLMChain

# Initialize the language model
llm = OpenAI(temperature=0.7)

# Create a prompt template for recipe generation
template = """
Given the following ingredients, create a recipe:

Ingredients: {ingredients}

Please provide:
1. A creative name for the dish
2. A list of all ingredients with measurements
3. Step-by-step cooking instructions
4. Estimated cooking time
5. Serving size
"""

prompt = PromptTemplate(template=template, input_variables=["ingredients"])

# Create the recipe generation chain
recipe_chain = LLMChain(llm=llm, prompt=prompt)

# Generate a recipe
ingredients = "chicken breast, spinach, feta cheese, olive oil, garlic, lemon"
recipe = recipe_chain.run(ingredients)

print("Generated Recipe:")
print(recipe)
```

Slide 13: Challenges and Best Practices

When working with LangChain, it's important to be aware of common challenges and best practices. This slide discusses some key considerations for developing robust LangChain applications.

```python
from langchain import OpenAI, PromptTemplate, LLMChain
import time

def retry_with_exponential_backoff(func, max_retries=3, initial_delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = initial_delay * (2 ** attempt)
            print(f"Error occurred: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)

llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short paragraph about {topic}."
)
chain = LLMChain(llm=llm, prompt=prompt)

def generate_content():
    return chain.run("artificial intelligence")

result = retry_with_exponential_backoff(generate_content)
print("Generated content:")
print(result)
```

Slide 14: Additional Resources

For further exploration of LangChain and related topics, consider the following resources:

1. LangChain Documentation: [https://python.langchain.com/en/latest/](https://python.langchain.com/en/latest/)
2. "Language Models are Few-Shot Learners" (Brown et al., 2020): [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
3. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022): [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)
4. LangChain GitHub Repository: [https://github.com/hwchase17/langchain](https://github.com/hwchase17/langchain)

These resources provide in-depth information on LangChain, language models, and advanced prompting techniques.

