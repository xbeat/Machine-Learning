## Document Summarization with Python
Slide 1: Introduction to Document Summarization

Document summarization is a crucial task in natural language processing that involves condensing large volumes of text into concise, informative summaries. This process helps users quickly grasp the main ideas of a document without reading the entire content. In this presentation, we'll explore how to transform document summarization using sentence embeddings, clustering, and summarization techniques in Python.

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Remove stopwords and convert to lowercase
    stop_words = set(stopwords.words('english'))
    processed_sentences = [
        ' '.join([word.lower() for word in sentence.split() if word.lower() not in stop_words])
        for sentence in sentences
    ]
    
    return processed_sentences

# Example usage
text = "Document summarization is an important task in NLP. It helps users quickly understand the main points of a document."
processed_sentences = preprocess_text(text)
print(processed_sentences)
```

Slide 2: Understanding Sentence Embeddings

Sentence embeddings are dense vector representations of sentences that capture semantic meaning. These embeddings allow us to represent sentences in a way that machines can understand and process. Various techniques exist for creating sentence embeddings, including Word2Vec, GloVe, and more advanced models like BERT or Sentence-BERT.

```python
from sentence_transformers import SentenceTransformer

def create_sentence_embeddings(sentences):
    # Load a pre-trained Sentence-BERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Generate embeddings for the sentences
    embeddings = model.encode(sentences)
    
    return embeddings

# Example usage
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A sentence embedding captures the meaning of a sentence."
]
embeddings = create_sentence_embeddings(sentences)
print(f"Shape of embeddings: {embeddings.shape}")
print(f"Embedding of first sentence: {embeddings[0][:5]}...")  # Showing first 5 values
```

Slide 3: Clustering Sentences

Clustering is a technique used to group similar sentences together based on their embeddings. This helps identify main themes or topics within a document. We'll use the K-means algorithm for clustering, which is simple yet effective for this task.

```python
from sklearn.cluster import KMeans
import numpy as np

def cluster_sentences(embeddings, num_clusters):
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Find sentences closest to cluster centers
    cluster_centers = kmeans.cluster_centers_
    closest_sentences = []
    
    for center in cluster_centers:
        distances = np.linalg.norm(embeddings - center, axis=1)
        closest_sentence_idx = np.argmin(distances)
        closest_sentences.append(closest_sentence_idx)
    
    return cluster_labels, closest_sentences

# Example usage
num_clusters = 2
cluster_labels, closest_sentences = cluster_sentences(embeddings, num_clusters)
print(f"Cluster labels: {cluster_labels}")
print(f"Indices of sentences closest to cluster centers: {closest_sentences}")
```

Slide 4: Extractive Summarization

Extractive summarization involves selecting the most important sentences from the original text to form a summary. We'll use the clustering results to identify key sentences that represent the main ideas of the document.

```python
def extractive_summarization(sentences, cluster_labels, closest_sentences):
    summary = []
    for idx in closest_sentences:
        summary.append(sentences[idx])
    return summary

# Example usage
original_sentences = [
    "Extractive summarization selects important sentences.",
    "It uses clustering to identify key ideas.",
    "This method preserves the original wording.",
    "Summaries help quickly understand documents."
]
embeddings = create_sentence_embeddings(original_sentences)
cluster_labels, closest_sentences = cluster_sentences(embeddings, num_clusters=2)
summary = extractive_summarization(original_sentences, cluster_labels, closest_sentences)

print("Original sentences:")
print("\n".join(original_sentences))
print("\nExtracted summary:")
print("\n".join(summary))
```

Slide 5: Abstractive Summarization

Abstractive summarization generates new sentences that capture the essence of the original text. This approach often produces more fluent and concise summaries. We'll use a pre-trained T5 model for this task.

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

def abstractive_summarization(text, max_length=150):
    # Load pre-trained T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    # Prepare input
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate summary
    summary_ids = model.generate(inputs, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Example usage
text = "Abstractive summarization is a technique that generates new sentences to summarize a document. Unlike extractive summarization, which selects existing sentences, abstractive methods can produce more concise and fluent summaries. This approach often uses advanced language models to understand the content and generate summaries."
summary = abstractive_summarization(text)
print("Original text:")
print(text)
print("\nAbstractive summary:")
print(summary)
```

Slide 6: Evaluation Metrics

Evaluating the quality of summaries is crucial. Common metrics include ROUGE (Recall-Oriented Understudy for Gisting Evaluation) and BLEU (Bilingual Evaluation Understudy). These metrics compare generated summaries with human-written reference summaries.

```python
from rouge import Rouge

def evaluate_summary(reference, generated):
    rouge = Rouge()
    scores = rouge.get_scores(generated, reference)
    
    print("ROUGE Scores:")
    for metric, score in scores[0].items():
        print(f"{metric}: {score['f']:.4f}")

# Example usage
reference = "Abstractive summarization generates new sentences to summarize documents."
generated = "Abstractive summarization creates new sentences for document summaries."
evaluate_summary(reference, generated)
```

Slide 7: Handling Long Documents

When dealing with long documents, we need to consider memory constraints and processing time. One approach is to split the document into smaller chunks, summarize each chunk, and then combine the results.

```python
def chunk_and_summarize(text, chunk_size=1000, overlap=100):
    # Split text into overlapping chunks
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    # Summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        summary = abstractive_summarization(chunk, max_length=100)
        chunk_summaries.append(summary)
    
    # Combine chunk summaries
    final_summary = " ".join(chunk_summaries)
    return abstractive_summarization(final_summary, max_length=200)

# Example usage
long_text = "..." * 1000  # Long document placeholder
summary = chunk_and_summarize(long_text)
print("Summary of long document:")
print(summary)
```

Slide 8: Multilingual Summarization

Extending summarization to multiple languages involves using multilingual models and handling language-specific preprocessing. We'll use a multilingual T5 model for this task.

```python
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

def multilingual_summarization(text, language, max_length=150):
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    
    # Prepare input
    inputs = tokenizer.encode(f"summarize {language}: " + text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate summary
    summary_ids = model.generate(inputs, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Example usage
french_text = "Le résumé de documents est une tâche importante en traitement du langage naturel. Il permet aux utilisateurs de comprendre rapidement les points principaux d'un document sans avoir à lire l'intégralité du contenu."
summary = multilingual_summarization(french_text, "French")
print("Original French text:")
print(french_text)
print("\nSummary in French:")
print(summary)
```

Slide 9: Topic-Focused Summarization

Topic-focused summarization aims to generate summaries that emphasize specific topics or aspects of the document. This can be achieved by modifying the importance of sentences based on their relevance to the given topic.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def topic_focused_summarization(text, topic, num_sentences=3):
    sentences = sent_tokenize(text)
    
    # Calculate TF-IDF scores
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calculate relevance to the topic
    topic_vector = vectorizer.transform([topic])
    relevance_scores = tfidf_matrix.dot(topic_vector.T).toarray().flatten()
    
    # Select top sentences
    top_sentences = sorted(range(len(sentences)), key=lambda i: relevance_scores[i], reverse=True)[:num_sentences]
    summary = " ".join([sentences[i] for i in sorted(top_sentences)])
    
    return summary

# Example usage
text = "Artificial intelligence is a broad field that includes machine learning, natural language processing, and computer vision. Machine learning focuses on algorithms that can learn from data. Natural language processing deals with understanding and generating human language. Computer vision aims to enable machines to interpret and understand visual information from the world."
topic = "machine learning"
summary = topic_focused_summarization(text, topic)
print(f"Topic-focused summary on '{topic}':")
print(summary)
```

Slide 10: Real-Life Example: News Article Summarization

Let's apply our summarization techniques to a real-life scenario: summarizing news articles. This can help readers quickly grasp the main points of current events without reading entire articles.

```python
import requests
from bs4 import BeautifulSoup

def summarize_news_article(url):
    # Fetch article content
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract main content (this may need adjustment based on the website structure)
    article_body = soup.find('article')
    if article_body:
        paragraphs = article_body.find_all('p')
        content = ' '.join([p.text for p in paragraphs])
    else:
        content = soup.get_text()
    
    # Generate summary
    summary = abstractive_summarization(content, max_length=200)
    return summary

# Example usage
url = "https://www.bbc.com/news/science-environment-56837908"
summary = summarize_news_article(url)
print("News Article Summary:")
print(summary)
```

Slide 11: Real-Life Example: Academic Paper Summarization

Summarizing academic papers can help researchers quickly understand the key findings and methodologies of published works. This example demonstrates how to summarize the abstract of a research paper.

```python
import requests

def summarize_arxiv_paper(arxiv_id):
    # Fetch paper metadata
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)
    
    # Extract abstract
    start_tag = "<abstract>"
    end_tag = "</abstract>"
    start_index = response.text.find(start_tag) + len(start_tag)
    end_index = response.text.find(end_tag)
    abstract = response.text[start_index:end_index].strip()
    
    # Generate summary
    summary = abstractive_summarization(abstract, max_length=150)
    return summary

# Example usage
arxiv_id = "2103.00020"  # Sample arXiv ID
summary = summarize_arxiv_paper(arxiv_id)
print("Academic Paper Summary:")
print(summary)
```

Slide 12: Challenges and Future Directions

While we've made significant progress in document summarization, several challenges remain:

1. Maintaining factual accuracy in generated summaries
2. Handling domain-specific terminology and concepts
3. Improving coherence and readability of summaries
4. Addressing bias in summarization models

Slide 13: Challenges and Future Directions

Future research directions include:

1. Developing more efficient and accurate summarization models
2. Incorporating external knowledge for better context understanding
3. Enhancing multi-document summarization techniques
4. Improving evaluation metrics for summarization quality

```python
def visualize_future_directions():
    import matplotlib.pyplot as plt
    
    directions = ['Efficiency', 'Accuracy', 'Context', 'Multi-doc', 'Evaluation']
    importance = [0.8, 0.9, 0.7, 0.6, 0.8]
    
    plt.figure(figsize=(10, 6))
    plt.bar(directions, importance)
    plt.title('Importance of Future Research Directions in Summarization')
    plt.ylabel('Relative Importance')
    plt.ylim(0, 1)
    plt.show()

visualize_future_directions()
```

Slide 14: Conclusion

Document summarization is a powerful tool for managing information overload in the digital age. By leveraging sentence embeddings, clustering techniques, and advanced language models, we can create both extractive and abstractive summaries that capture the essence of documents. As natural language processing continues to evolve, we can expect even more sophisticated summarization methods that will further enhance our ability to quickly digest and understand large volumes of text.

```python
def summarize_presentation():
    topics = [
        "Introduction to Document Summarization",
        "Sentence Embeddings",
        "Clustering Sentences",
        "Extractive Summarization",
        "Abstractive Summarization",
        "Evaluation Metrics",
        "Handling Long Documents",
        "Multilingual Summarization",
        "Topic-Focused Summarization",
        "Real-Life Examples",
        "Challenges and Future Directions"
    ]
    
    print("Key Topics Covered in This Presentation:")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic}")

summarize_presentation()
```

Slide 15: Additional Resources

For those interested in diving deeper into document summarization and related topics, here are some valuable resources:

1. "A Survey of Deep Learning Techniques for Text Summarization" by Y. Dong (2018) ArXiv: [https://arxiv.org/abs/1707.02268](https://arxiv.org/abs/1707.02268)
2. "Neural Abstractive Text Summarization with Sequence-to-Sequence Models" by T. Shi et al. (2018) ArXiv: [https://arxiv.org/abs/1812.02303](https://arxiv.org/abs/1812.02303)
3. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by J. Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
4. "Text Summarization Techniques: A Brief Survey" by M. Allahyari et al. (2017) ArXiv: [https://arxiv](https://arxiv)
