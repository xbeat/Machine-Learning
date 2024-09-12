## Hybrid Vector Search with LangChain in Python
Slide 1: Hybrid Vector Search with LangChain

Hybrid Vector Search combines traditional keyword-based search with modern vector-based similarity search. This approach leverages the strengths of both methods to provide more accurate and relevant search results. LangChain, a powerful framework for developing applications with large language models, offers tools to implement this hybrid search strategy efficiently.

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Initialize retrievers
bm25_retriever = BM25Retriever.from_documents(documents)
faiss_vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
vector_retriever = faiss_vectorstore.as_retriever()

# Create hybrid retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
```

Slide 2: Understanding Vector Search

Vector search is a technique that represents documents and queries as high-dimensional vectors. These vectors capture semantic meaning, allowing for similarity comparisons based on context rather than exact keyword matches. In LangChain, we can use various embedding models to convert text into these vector representations.

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

embeddings = OpenAIEmbeddings()

documents = [
    Document(page_content="The quick brown fox jumps over the lazy dog"),
    Document(page_content="A man's best friend is his dog"),
]

doc_vectors = embeddings.embed_documents([doc.page_content for doc in documents])
query_vector = embeddings.embed_query("What animal is known for loyalty?")
```

Slide 3: Keyword-Based Search with BM25

BM25 (Best Matching 25) is a popular ranking function used in information retrieval. It's based on the probabilistic retrieval framework and considers term frequency, inverse document frequency, and document length. LangChain provides a BM25Retriever that implements this algorithm for efficient keyword-based search.

```python
from langchain.retrievers import BM25Retriever
from langchain.schema import Document

documents = [
    Document(page_content="Python is a popular programming language"),
    Document(page_content="Java is widely used in enterprise applications"),
    Document(page_content="JavaScript is essential for web development"),
]

bm25_retriever = BM25Retriever.from_documents(documents)
results = bm25_retriever.get_relevant_documents("What language is used for web?")
```

Slide 4: Vector Stores in LangChain

Vector stores are specialized databases designed to store and efficiently search vector embeddings. LangChain supports various vector stores, including FAISS, Pinecone, and Chroma. These allow for fast similarity search operations on large collections of vectors.

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
documents = [
    "Artificial intelligence is revolutionizing industries",
    "Machine learning algorithms can process vast amounts of data",
    "Deep learning models have achieved human-level performance in many tasks",
]

vectorstore = FAISS.from_texts(documents, embeddings)
query = "What is the impact of AI on industries?"
similar_docs = vectorstore.similarity_search(query)
```

Slide 5: Implementing Hybrid Search

Hybrid search combines the strengths of keyword-based and vector-based search methods. LangChain's EnsembleRetriever allows us to create a hybrid search system by combining multiple retrievers with customizable weights.

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

documents = [
    "Climate change is a global challenge",
    "Renewable energy sources are crucial for sustainability",
    "Deforestation contributes to biodiversity loss",
]

bm25_retriever = BM25Retriever.from_texts(documents)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(documents, embeddings)
vector_retriever = vectorstore.as_retriever()

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)

results = hybrid_retriever.get_relevant_documents("How does deforestation affect the environment?")
```

Slide 6: Customizing Retriever Weights

The weights in the EnsembleRetriever determine the influence of each component retriever on the final results. Adjusting these weights allows us to fine-tune the balance between keyword matching and semantic similarity in our hybrid search system.

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

documents = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris",
    "The Statue of Liberty is a colossal neoclassical sculpture in New York Harbor",
    "The Great Wall of China is a series of fortifications and walls across China",
]

bm25_retriever = BM25Retriever.from_texts(documents)
embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma.from_texts(documents, embeddings)
vector_retriever = vectorstore.as_retriever()

# Emphasize semantic similarity
semantic_hybrid = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.2, 0.8]
)

# Emphasize keyword matching
keyword_hybrid = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.8, 0.2]
)

query = "What famous monument is in France?"
semantic_results = semantic_hybrid.get_relevant_documents(query)
keyword_results = keyword_hybrid.get_relevant_documents(query)
```

Slide 7: Preprocessing and Tokenization

Effective hybrid search often requires proper preprocessing and tokenization of both documents and queries. This step ensures consistent treatment of text data across both keyword-based and vector-based components of the search system.

```python
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and stem
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    processed_tokens = [ps.stem(token) for token in tokens if token not in stop_words]
    
    return " ".join(processed_tokens)

documents = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
]

processed_docs = [preprocess_text(doc) for doc in documents]
query = "How does a fox move?"
processed_query = preprocess_text(query)

print(processed_docs)
print(processed_query)
```

Slide 8: Handling Multi-Modal Data

Hybrid vector search can be extended to handle multi-modal data, such as text and images. This requires using specialized embedding models that can generate vectors for different types of data, allowing for cross-modal similarity search.

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Text embeddings
text_embeddings = OpenAIEmbeddings()
text_documents = ["A beautiful sunset over the ocean", "A bustling city street at night"]
text_vectorstore = FAISS.from_texts(text_documents, text_embeddings)

# Image embeddings
image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
image_model.eval()
preprocess = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def image_to_vector(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        return image_model(input_tensor).squeeze().numpy()

image_vectors = [image_to_vector("sunset.jpg"), image_to_vector("city_street.jpg")]
image_vectorstore = FAISS.from_embeddings(image_vectors, text_embeddings)

# Hybrid search
query = "night scene in a city"
text_results = text_vectorstore.similarity_search(query)
image_results = image_vectorstore.similarity_search(query)
```

Slide 9: Scalability and Performance Optimization

As the volume of data grows, optimizing the performance of hybrid vector search becomes crucial. Techniques such as indexing, caching, and distributed processing can significantly improve search efficiency and response times.

```python
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
import pinecone

# Initialize Pinecone
pinecone.init(api_key="your-api-key", environment="your-environment")

# Create a new index if it doesn't exist
if "hybrid-search" not in pinecone.list_indexes():
    pinecone.create_index("hybrid-search", dimension=1536, metric="cosine")

# Connect to the index
index = pinecone.Index("hybrid-search")

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create Pinecone vectorstore
vectorstore = Pinecone(index, embeddings.embed_query, "text")

# Batch upsert for efficiency
documents = [
    "Artificial intelligence and its applications",
    "The impact of climate change on ecosystems",
    "Advancements in quantum computing technology",
]

vectors = embeddings.embed_documents(documents)
metadata = [{"source": f"doc{i}"} for i in range(len(documents))]

vectorstore.add_documents(list(zip(documents, metadata, vectors)))

# Perform hybrid search
query = "How does AI affect climate research?"
results = vectorstore.similarity_search(query, k=5)
```

Slide 10: Relevance Feedback and Query Expansion

Incorporating relevance feedback and query expansion techniques can enhance the effectiveness of hybrid vector search. These methods use information from initial search results or user feedback to refine and improve subsequent queries.

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers import BM25Retriever
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Initialize components
llm = OpenAI()
embeddings = OpenAIEmbeddings()
documents = [
    "Machine learning algorithms can process vast amounts of data",
    "Deep learning models have achieved human-level performance in many tasks",
    "Natural language processing enables computers to understand human language",
]

# Create retrievers
bm25_retriever = BM25Retriever.from_texts(documents)
vectorstore = FAISS.from_texts(documents, embeddings)
vector_retriever = vectorstore.as_retriever()

# Query expansion prompt
expand_prompt = PromptTemplate(
    input_variables=["query"],
    template="Expand the following search query with related terms:\n\nQuery: {query}\n\nExpanded query:"
)
expand_chain = LLMChain(llm=llm, prompt=expand_prompt)

# Perform hybrid search with query expansion
original_query = "How does AI analyze data?"
expanded_query = expand_chain.run(original_query)

bm25_results = bm25_retriever.get_relevant_documents(expanded_query)
vector_results = vector_retriever.get_relevant_documents(expanded_query)

# Combine and rank results (simplified)
combined_results = list(set(bm25_results + vector_results))
```

Slide 11: Handling Multilingual Search

Extending hybrid vector search to support multiple languages requires careful consideration of language-specific preprocessing and embedding models. LangChain can be integrated with multilingual models to create a robust, language-agnostic search system.

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from transformers import MarianMTModel, MarianTokenizer

# Initialize multilingual embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Initialize translation model
model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_to_english(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Sample multilingual documents
documents = [
    "Artificial intelligence is revolutionizing industries",
    "La inteligencia artificial está revolucionando las industrias",
    "L'intelligence artificielle révolutionne les industries",
]

# Translate non-English documents
translated_docs = [doc if doc.startswith("Artificial") else translate_to_english(doc) for doc in documents]

# Create vector store
vectorstore = Chroma.from_texts(translated_docs, embeddings)

# Perform multilingual search
query = "How is AI changing businesses?"
results = vectorstore.similarity_search(query)

print(results)
```

Slide 12: Real-Life Example: E-commerce Product Search

Implementing hybrid vector search in an e-commerce platform can significantly improve product discovery. This example demonstrates how to combine text-based product descriptions with image-based similarity search for a more comprehensive shopping experience.

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Text-based search setup
text_embeddings = OpenAIEmbeddings()
product_descriptions = [
    "Blue denim jeans with a slim fit",
    "Red cotton t-shirt with a round neck",
    "Black leather boots with a rubber sole",
]
text_vectorstore = FAISS.from_texts(product_descriptions, text_embeddings)
text_retriever = text_vectorstore.as_retriever()

bm25_retriever = BM25Retriever.from_texts(product_descriptions)

# Image-based search setup
image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
image_model.eval()
preprocess = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def image_to_vector(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        return image_model(input_tensor).squeeze().numpy()

product_images = ["jeans.jpg", "tshirt.jpg", "boots.jpg"]
image_vectors = [image_to_vector(img) for img in product_images]
image_vectorstore = FAISS.from_embeddings(list(zip(product_descriptions, image_vectors)), text_embeddings)

# Hybrid search
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, text_retriever, image_vectorstore.as_retriever()],
    weights=[0.3, 0.3, 0.4]
)

query = "casual blue outfit"
results = hybrid_retriever.get_relevant_documents(query)
```

Slide 13: Real-Life Example: Content Recommendation System

A hybrid vector search can power a sophisticated content recommendation system for a streaming platform. This example shows how to combine user preferences, content metadata, and visual features to provide personalized recommendations.

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Sample content data
content_data = [
    {"title": "Stranger Things", "genre": "Sci-Fi", "description": "A group of kids face supernatural forces and government agencies."},
    {"title": "The Crown", "genre": "Drama", "description": "Chronicles the life of Queen Elizabeth II from the 1940s to modern times."},
    {"title": "Narcos", "genre": "Crime", "description": "A chronicled look at the criminal exploits of Colombian drug lord Pablo Escobar."},
]

# Create vector stores for different features
title_store = Chroma.from_texts([item['title'] for item in content_data], embeddings)
genre_store = Chroma.from_texts([item['genre'] for item in content_data], embeddings)
description_store = Chroma.from_texts([item['description'] for item in content_data], embeddings)

# Create retrievers
title_retriever = title_store.as_retriever()
genre_retriever = genre_store.as_retriever()
description_retriever = description_store.as_retriever()

# Create hybrid retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[title_retriever, genre_retriever, description_retriever],
    weights=[0.2, 0.3, 0.5]
)

# User preference (could be based on viewing history)
user_preference = "I like historical dramas with political intrigue"

# Get recommendations
recommendations = hybrid_retriever.get_relevant_documents(user_preference)
```

Slide 14: Evaluating Hybrid Vector Search Performance

To ensure the effectiveness of a hybrid vector search system, it's crucial to evaluate its performance using appropriate metrics. This slide demonstrates how to implement a simple evaluation framework using precision, recall, and F1 score.

```python
from sklearn.metrics import precision_score, recall_score, f1_score
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Sample data
documents = [
    "Python is a popular programming language",
    "Java is widely used in enterprise applications",
    "JavaScript is essential for web development",
    "C++ is known for its performance in system programming",
    "Ruby is praised for its simplicity and productivity"
]

# Ground truth relevance (1 for relevant, 0 for not relevant)
ground_truth = {
    "web programming": [0, 0, 1, 0, 0],
    "enterprise software": [0, 1, 0, 0, 0],
    "high-performance computing": [0, 0, 0, 1, 0]
}

# Set up retrievers
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(documents, embeddings)
bm25_retriever = BM25Retriever.from_texts(documents)
vector_retriever = vectorstore.as_retriever()

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

def evaluate_retriever(retriever, query, relevant_docs):
    results = retriever.get_relevant_documents(query)
    predictions = [1 if doc.page_content in results else 0 for doc in documents]
    
    precision = precision_score(relevant_docs, predictions)
    recall = recall_score(relevant_docs, predictions)
    f1 = f1_score(relevant_docs, predictions)
    
    return precision, recall, f1

# Evaluate the hybrid retriever
for query, relevance in ground_truth.items():
    precision, recall, f1 = evaluate_retriever(hybrid_retriever, query, relevance)
    print(f"Query: {query}")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
    print()
```

Slide 15: Additional Resources

For those interested in delving deeper into hybrid vector search and its applications, here are some valuable resources:

1. "Efficient and Effective Passage Search via Contextualized Late Interaction over BERT" by Xiong et al. (2020) ArXiv: [https://arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832)
2. "Dense Passage Retrieval for Open-Domain Question Answering" by Karpukhin et al. (2020) ArXiv: [https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)
3. "Hybrid Search in LangChain" - Official LangChain Documentation [https://python.langchain.com/en/latest/modules/indexes/retrievers/examples/hybrid\_search.html](https://python.langchain.com/en/latest/modules/indexes/retrievers/examples/hybrid_search.html)
4. "REALM: Retrieval-Augmented Language Model Pre-Training" by Guu et al. (2020) ArXiv: [https://arxiv.org/abs/2002.08909](https://arxiv.org/abs/2002.08909)

These resources provide in-depth information on various aspects of hybrid search, including theoretical foundations, implementation techniques, and practical applications in natural language processing and information retrieval.

