## Transforming Document Summarization with Python
Slide 1: Introduction to Document Summarization

Document summarization is the process of distilling the most important information from a text while maintaining its core meaning. This technique is crucial in today's information-rich world, helping us quickly grasp the essence of lengthy documents. In this presentation, we'll explore how to leverage sentence embeddings, clustering, and summarization techniques using Python to transform raw text into concise, meaningful summaries.

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    processed_sentences = [
        ' '.join([word.lower() for word in sentence.split() if word.lower() not in stop_words])
        for sentence in sentences
    ]
    return processed_sentences

text = "Document summarization is a crucial technique in natural language processing. It helps in extracting key information from large texts. This process can be automated using various algorithms and methods."
processed_text = preprocess_text(text)
print(processed_text)
```

Slide 2: Understanding Sentence Embeddings

Sentence embeddings are dense vector representations of sentences that capture semantic meaning. They allow us to represent sentences in a way that machines can understand and process. Various techniques exist for creating sentence embeddings, including word averaging, Transformer-based models like BERT, and sentence-specific models like Universal Sentence Encoder.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

sentences = [
    "The cat sits on the mat.",
    "A feline rests on the floor covering."
]

embeddings = model.encode(sentences)
print(f"Shape of embeddings: {embeddings.shape}")
print(f"Embedding of first sentence:\n{embeddings[0][:10]}...")  # Showing first 10 values
```

Slide 3: Clustering Sentences

Clustering is a technique used to group similar sentences together based on their embeddings. This step helps identify the main topics or themes within a document. We'll use the K-means algorithm, a popular clustering method, to group our sentence embeddings.

```python
from sklearn.cluster import KMeans
import numpy as np

# Assuming we have embeddings from the previous slide
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(embeddings)

for i, sentence in enumerate(sentences):
    print(f"Sentence: {sentence}")
    print(f"Cluster: {cluster_assignments[i]}\n")

# Visualize cluster centers
centers = kmeans.cluster_centers_
print("Cluster centers:")
print(centers)
```

Slide 4: Extractive Summarization

Extractive summarization involves selecting the most representative sentences from the original text to form a summary. We'll use the clustered sentences and choose the sentences closest to each cluster center as our summary.

```python
from sklearn.metrics.pairwise import cosine_similarity

def get_summary(sentences, embeddings, cluster_assignments, centers):
    summary = []
    for cluster_id in range(len(centers)):
        cluster_sentences = [sent for i, sent in enumerate(sentences) if cluster_assignments[i] == cluster_id]
        cluster_embeddings = embeddings[cluster_assignments == cluster_id]
        
        if len(cluster_sentences) > 0:
            similarities = cosine_similarity(cluster_embeddings, [centers[cluster_id]])
            most_representative = cluster_sentences[similarities.argmax()]
            summary.append(most_representative)
    
    return summary

summary = get_summary(sentences, embeddings, cluster_assignments, centers)
print("Extractive Summary:")
for sent in summary:
    print(sent)
```

Slide 5: Abstractive Summarization

Unlike extractive summarization, abstractive summarization generates new sentences that capture the essence of the original text. We'll use a pre-trained T5 model for this task, which can produce human-like summaries.

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

text = "The quick brown fox jumps over the lazy dog. This sentence contains all letters of the English alphabet. It's often used as a typing exercise."

inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Abstractive Summary:")
print(summary)
```

Slide 6: Evaluating Summaries

Evaluating the quality of summaries is crucial for improving our summarization techniques. We'll use the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric, which compares the generated summary with a reference summary.

```python
from rouge import Rouge

reference_summary = "The sentence 'The quick brown fox jumps over the lazy dog' contains all English alphabet letters and is used for typing practice."
generated_summary = summary  # From the previous slide

rouge = Rouge()
scores = rouge.get_scores(generated_summary, reference_summary)

print("ROUGE Scores:")
for metric, score in scores[0].items():
    print(f"{metric}: {score['f']:.4f}")
```

Slide 7: Handling Long Documents

When dealing with long documents, we need to consider memory constraints and processing time. One approach is to chunk the document into smaller sections, summarize each section, and then combine the results.

```python
def chunk_text(text, max_chunk_size=1000):
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) >= max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

long_text = "..." * 1000  # Imagine this is a very long document
chunks = chunk_text(long_text)

summaries = []
for chunk in chunks:
    inputs = tokenizer("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summaries.append(summary)

final_summary = " ".join(summaries)
print("Final Summary of Long Document:")
print(final_summary[:500] + "...")  # Printing first 500 characters
```

Slide 8: Incorporating Domain Knowledge

For specialized domains, incorporating domain-specific knowledge can significantly improve summarization quality. We can achieve this by fine-tuning our models on domain-specific data or by using domain-specific embeddings.

```python
from gensim.models import KeyedVectors

# Load pre-trained domain-specific word embeddings (example using medical domain)
word_vectors = KeyedVectors.load_word2vec_format('medical_embeddings.bin', binary=True)

def get_sentence_embedding(sentence, word_vectors):
    words = sentence.split()
    word_embeddings = [word_vectors[word] for word in words if word in word_vectors]
    if not word_embeddings:
        return None
    return sum(word_embeddings) / len(word_embeddings)

medical_text = "Hypertension, also known as high blood pressure, is a chronic medical condition characterized by persistently elevated pressure within the blood vessels."
embedding = get_sentence_embedding(medical_text, word_vectors)

print("Domain-specific sentence embedding:")
print(embedding[:10])  # Printing first 10 values
```

Slide 9: Multi-lingual Summarization

In our globalized world, the ability to summarize documents in multiple languages is increasingly important. We'll use a multi-lingual model to demonstrate summarization across different languages.

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

text = {
    "en": "The United Nations is an international organization founded in 1945. It is currently made up of 193 Member States.",
    "fr": "Les Nations Unies sont une organisation internationale fondée en 1945. Elle est actuellement composée de 193 États membres.",
    "es": "Las Naciones Unidas es una organización internacional fundada en 1945. Actualmente está compuesta por 193 Estados Miembros."
}

for lang, content in text.items():
    tokenizer.src_lang = lang
    encoded = tokenizer(content, return_tensors="pt")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    summary = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(f"Summary ({lang} to English): {summary[0]}")
```

Slide 10: Real-life Example: News Article Summarization

Let's apply our summarization techniques to a real-life scenario: summarizing news articles. This can help readers quickly grasp the main points of a story without reading the entire article.

```python
import requests
from bs4 import BeautifulSoup

def fetch_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    return ' '.join([p.text for p in paragraphs])

url = "https://example.com/news-article"  # Replace with an actual news article URL
article_text = fetch_article(url)

inputs = tokenizer("summarize: " + article_text, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("News Article Summary:")
print(summary)
```

Slide 11: Real-life Example: Research Paper Summarization

Another practical application of document summarization is condensing research papers. This can help researchers quickly understand the key findings and methodologies of a study without reading the entire paper.

```python
import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

pdf_path = "research_paper.pdf"  # Replace with an actual PDF path
paper_text = extract_text_from_pdf(pdf_path)

# Chunk the paper text due to its length
chunks = chunk_text(paper_text, max_chunk_size=1000)

summaries = []
for chunk in chunks:
    inputs = tokenizer("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summaries.append(summary)

final_summary = " ".join(summaries)
print("Research Paper Summary:")
print(final_summary[:500] + "...")  # Printing first 500 characters
```

Slide 12: Challenges and Future Directions

While we've made significant progress in document summarization, several challenges remain. These include maintaining factual accuracy, handling domain-specific terminology, and generating truly abstractive summaries. Future research directions may focus on:

1. Improving factual consistency in generated summaries
2. Developing more efficient methods for handling extremely long documents
3. Enhancing multi-modal summarization (text + images/videos)
4. Creating more robust evaluation metrics that consider semantic similarity and factual accuracy

```python
import matplotlib.pyplot as plt
import numpy as np

challenges = ['Factual Accuracy', 'Domain Specificity', 'Abstractiveness', 'Long Documents', 'Multi-modal']
difficulty = [0.8, 0.7, 0.9, 0.6, 0.85]

plt.figure(figsize=(10, 6))
plt.bar(challenges, difficulty)
plt.title('Challenges in Document Summarization')
plt.ylabel('Difficulty Level')
plt.ylim(0, 1)
plt.show()
```

Slide 13: Ethical Considerations

As we advance in document summarization technology, it's crucial to consider the ethical implications. These include:

1. Bias in summarization models
2. Potential for misuse in spreading misinformation
3. Privacy concerns when summarizing sensitive documents

Researchers and practitioners must be mindful of these issues and work towards developing fair, transparent, and responsible summarization systems.

```python
import networkx as nx

G = nx.Graph()
G.add_edge("Summarization Model", "Bias")
G.add_edge("Summarization Model", "Misinformation")
G.add_edge("Summarization Model", "Privacy Concerns")
G.add_edge("Bias", "Fairness")
G.add_edge("Misinformation", "Fact-checking")
G.add_edge("Privacy Concerns", "Data Protection")

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')
plt.title("Ethical Considerations in Document Summarization")
plt.axis('off')
plt.show()
```

Slide 14: Additional Resources

For those interested in delving deeper into document summarization techniques, here are some valuable resources:

1. "Automatic Text Summarization with Machine Learning - An overview" by Khurana et al. (2019). ArXiv: [https://arxiv.org/abs/1906.04165](https://arxiv.org/abs/1906.04165)
2. "Neural Abstractive Text Summarization with Sequence-to-Sequence Models" by Shi et al. (2018). ArXiv: [https://arxiv.org/abs/1812.02303](https://arxiv.org/abs/1812.02303)
3. "A Survey on Neural Network-Based Summarization Methods" by Dong et al. (2021). ArXiv: [https://arxiv.org/abs/2110.00864](https://arxiv.org/abs/2110.00864)

These papers provide comprehensive overviews of various summarization techniques, including the latest advancements in neural network-based methods.

