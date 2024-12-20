## Unleashing Word2Vec for NLP in Python
Slide 1: Introduction to Word2Vec

Word2Vec is a powerful technique in Natural Language Processing (NLP) that transforms words into dense vector representations. These vectors capture semantic relationships between words, allowing machines to understand language contexts better. Word2Vec models are trained on large corpora of text and learn to predict words given their context or vice versa.

```python
import gensim.downloader as api

# Load pre-trained Word2Vec model
model = api.load('word2vec-google-news-300')

# Find similar words
similar_words = model.most_similar('python', topn=5)
print(similar_words)
```

Slide 2: Word2Vec Architectures

Word2Vec employs two main architectures: Continuous Bag of Words (CBOW) and Skip-gram. CBOW predicts a target word given its context, while Skip-gram predicts the context given a target word. Both architectures use neural networks to learn word representations.

```python
from gensim.models import Word2Vec
import numpy as np

# Sample sentences
sentences = [['I', 'love', 'python', 'programming'],
             ['Data', 'science', 'is', 'fascinating']]

# Train CBOW model
cbow_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

# Train Skip-gram model
sg_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

# Get vector representations
python_vector_cbow = cbow_model.wv['python']
python_vector_sg = sg_model.wv['python']

print("CBOW vector:", python_vector_cbow[:5])
print("Skip-gram vector:", python_vector_sg[:5])
```

Slide 3: Preparing Data for Word2Vec

Before training a Word2Vec model, we need to preprocess our text data. This involves tokenization, lowercasing, and removing punctuation. We'll use the NLTK library for these tasks.

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    
    return tokens

# Example usage
text = "Natural Language Processing is fascinating and powerful!"
processed_tokens = preprocess_text(text)
print(processed_tokens)
```

Slide 4: Training a Word2Vec Model

Now that we have preprocessed our data, we can train our own Word2Vec model using Gensim. We'll use a small corpus of sentences for demonstration purposes.

```python
from gensim.models import Word2Vec

# Sample corpus
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Natural language processing deals with the interaction between computers and humans using natural language",
    "Python is a popular programming language for data science and machine learning"
]

# Preprocess the corpus
processed_corpus = [preprocess_text(sentence) for sentence in corpus]

# Train the Word2Vec model
model = Word2Vec(sentences=processed_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Save the model
model.save("word2vec.model")

print("Model trained and saved successfully!")
```

Slide 5: Exploring Word Similarities

One of the most powerful features of Word2Vec is its ability to find similar words based on their vector representations. We can use the trained model to explore these similarities.

```python
# Load the saved model
loaded_model = Word2Vec.load("word2vec.model")

# Find similar words
similar_words = loaded_model.wv.most_similar("python", topn=5)
print("Words similar to 'python':")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")

# Compute similarity between two words
similarity = loaded_model.wv.similarity("machine", "learning")
print(f"Similarity between 'machine' and 'learning': {similarity:.4f}")
```

Slide 6: Word Analogies with Word2Vec

Word2Vec can capture complex relationships between words, allowing us to perform word analogies. For example, we can ask questions like "king is to queen as man is to what?"

```python
# Load a pre-trained model for better results
model = api.load('word2vec-google-news-300')

# Perform word analogy
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(f"king - man + woman = {result[0][0]}")

# More examples
print(model.most_similar(positive=['paris', 'germany'], negative=['france'], topn=1))
print(model.most_similar(positive=['bigger', 'cold'], negative=['big'], topn=1))
```

Slide 7: Visualizing Word Embeddings

Visualizing high-dimensional word embeddings can provide insights into the relationships between words. We'll use t-SNE to reduce the dimensionality of our word vectors and plot them in 2D space.

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_words(model, words):
    # Extract word vectors
    word_vectors = np.array([model.wv[word] for word in words])
    
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=0)
    words_tsne = tsne.fit_transform(word_vectors)
    
    # Plot the words
    plt.figure(figsize=(12, 8))
    for i, word in enumerate(words):
        plt.scatter(words_tsne[i, 0], words_tsne[i, 1])
        plt.annotate(word, (words_tsne[i, 0], words_tsne[i, 1]))
    plt.title("Word Embeddings Visualization")
    plt.show()

# Example usage
words_to_plot = ["king", "queen", "man", "woman", "prince", "princess", "boy", "girl"]
plot_words(model, words_to_plot)
```

Slide 8: Word2Vec for Text Classification

Word2Vec embeddings can be used as features for text classification tasks. We'll demonstrate how to use Word2Vec embeddings with a simple sentiment analysis model.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data (replace with your own dataset)
texts = ["I love this product", "This is terrible", "Great experience", "Awful service"]
labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# Function to get document vector (average of word vectors)
def get_doc_vector(text, model):
    words = preprocess_text(text)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

# Prepare features and labels
X = np.array([get_doc_vector(text, model) for text in texts])
y = np.array(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

Slide 9: Handling Out-of-Vocabulary Words

One limitation of Word2Vec is its inability to handle words not seen during training. We can address this by using subword information or creating a custom unknown token.

```python
def get_word_vector(word, model):
    if word in model.wv:
        return model.wv[word]
    else:
        # Option 1: Return a zero vector
        # return np.zeros(model.vector_size)
        
        # Option 2: Return the average of subword vectors
        subwords = [word[i:i+3] for i in range(len(word)-2)]
        subword_vectors = [model.wv[sw] for sw in subwords if sw in model.wv]
        return np.mean(subword_vectors, axis=0) if subword_vectors else np.zeros(model.vector_size)

# Example usage
unknown_word = "unknownword"
vector = get_word_vector(unknown_word, model)
print(f"Vector for '{unknown_word}': {vector[:5]}...")
```

Slide 10: Word2Vec for Named Entity Recognition

Word2Vec embeddings can enhance Named Entity Recognition (NER) systems by providing rich semantic information about words. Here's a simple example using spaCy with custom Word2Vec embeddings.

```python
import spacy
from spacy.tokens import Doc

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Custom component to add Word2Vec embeddings
class Word2VecEmbedder:
    def __init__(self, w2v_model):
        self.w2v_model = w2v_model
    
    def __call__(self, doc):
        for token in doc:
            if token.text in self.w2v_model.wv:
                token._.w2v_vector = self.w2v_model.wv[token.text]
        return doc

# Add custom attribute to tokens
Doc.set_extension("w2v_vector", default=None)

# Add Word2Vec embedder to pipeline
nlp.add_pipe("word2vec_embedder", last=True)

# Example usage
text = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp(text)

for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")
    if ent[0]._.w2v_vector is not None:
        print(f"Word2Vec vector: {ent[0]._.w2v_vector[:5]}...")
```

Slide 11: Word2Vec for Recommendation Systems

Word2Vec can be applied to recommendation systems by treating items as "words" and user interactions as "sentences". This approach can capture item similarities based on user behavior.

```python
from gensim.models import Word2Vec

# Sample user interaction data
user_interactions = [
    ["item1", "item2", "item3"],
    ["item2", "item4", "item5"],
    ["item1", "item3", "item5"],
    ["item4", "item6", "item7"]
]

# Train Word2Vec model on user interactions
model = Word2Vec(sentences=user_interactions, vector_size=100, window=5, min_count=1, workers=4)

# Function to get item recommendations
def get_recommendations(item, model, top_n=5):
    similar_items = model.wv.most_similar(item, topn=top_n)
    return [item for item, score in similar_items]

# Example usage
target_item = "item2"
recommendations = get_recommendations(target_item, model)
print(f"Recommendations for {target_item}: {recommendations}")
```

Slide 12: Fine-tuning Word2Vec for Domain-Specific Tasks

Pre-trained Word2Vec models can be fine-tuned for domain-specific tasks. This process involves continuing the training process on domain-specific data.

```python
from gensim.models import Word2Vec

# Load pre-trained model
pretrained_model = api.load('word2vec-google-news-300')

# Domain-specific corpus (example)
domain_corpus = [
    ["machine", "learning", "artificial", "intelligence"],
    ["neural", "networks", "deep", "learning"],
    ["natural", "language", "processing", "nlp"]
]

# Initialize new model with pre-trained weights
new_model = Word2Vec(
    vector_size=pretrained_model.vector_size,
    min_count=1
)
new_model.build_vocab(domain_corpus)

#  vectors from pre-trained model
new_model.wv.vectors[:] = pretrained_model.wv.vectors[pretrained_model.wv.key_to_index.values()]

# Continue training on domain-specific data
new_model.train(domain_corpus, total_examples=len(domain_corpus), epochs=10)

# Compare similarities
word = "intelligence"
print("Original model:")
print(pretrained_model.wv.most_similar(word, topn=5))
print("\nFine-tuned model:")
print(new_model.wv.most_similar(word, topn=5))
```

Slide 13: Word2Vec for Language Translation

Word2Vec can be used to build simple translation systems by aligning word embeddings across languages. This technique works best for closely related languages.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Simulated bilingual dictionary
en_words = ["cat", "dog", "house", "car"]
fr_words = ["chat", "chien", "maison", "voiture"]

# Load pre-trained English and French Word2Vec models (simulated here)
en_model = api.load('word2vec-google-news-300')
fr_model = api.load('word2vec-google-news-300')  # In reality, use a French model

# Create translation matrix
en_vecs = np.array([en_model.wv[word] for word in en_words])
fr_vecs = np.array([fr_model.wv[word] for word in fr_words])
translation_matrix = np.linalg.lstsq(en_vecs, fr_vecs, rcond=None)[0]

# Function to translate a word
def translate_word(word, src_model, tgt_model, translation_matrix):
    if word in src_model.wv:
        vec = src_model.wv[word] @ translation_matrix
        return tgt_model.wv.most_similar(positive=[vec], topn=1)[0][0]
    return "Unknown"

# Example translation
en_word = "book"
fr_translation = translate_word(en_word, en_model, fr_model, translation_matrix)
print(f"'{en_word}' in French: '{fr_translation}'")
```

Slide 14: Evaluating Word2Vec Models

Evaluating Word2Vec models is crucial to ensure their quality and suitability for downstream tasks. Common evaluation methods include analogy tasks, similarity tasks, and extrinsic evaluation on downstream NLP tasks.

```python
from gensim.models import KeyedVectors
import numpy as np

# Load pre-trained model
model = api.load('word2vec-google-news-300')

# Analogy task evaluation
def evaluate_analogy(model, a, b, c, expected):
    try:
        result = model.most_similar(positive=[b, c], negative=[a], topn=1)[0][0]
        return result == expected
    except KeyError:
        return False

analogies = [
    ('man', 'king', 'woman', 'queen'),
    ('paris', 'france', 'rome', 'italy'),
    ('big', 'bigger', 'small', 'smaller')
]

analogy_accuracy = sum(evaluate_analogy(model, *analogy) for analogy in analogies) / len(analogies)
print(f"Analogy task accuracy: {analogy_accuracy:.2f}")

# Similarity task evaluation
def evaluate_similarity(model, word1, word2, expected_similarity):
    try:
        similarity = model.similarity(word1, word2)
        return abs(similarity - expected_similarity) < 0.1
    except KeyError:
        return False

similarities = [
    ('cat', 'dog', 0.8),
    ('happy', 'sad', -0.5),
    ('king', 'queen', 0.7)
]

similarity_accuracy = sum(evaluate_similarity(model, *sim) for sim in similarities) / len(similarities)
print(f"Similarity task accuracy: {similarity_accuracy:.2f}")

# Extrinsic evaluation (pseudocode)
# def evaluate_on_downstream_task(model, task_data):
#     # Use word vectors from the model in a downstream task (e.g., text classification)
#     # Train and evaluate the downstream model
#     # Return performance metric (e.g., accuracy, F1-score)
#     pass

# downstream_performance = evaluate_on_downstream_task(model, task_data)
# print(f"Downstream task performance: {downstream_performance:.2f}")
```

Slide 15: Word2Vec Limitations and Alternatives

While Word2Vec is powerful, it has limitations. It struggles with polysemy, out-of-vocabulary words, and context-dependent meanings. Modern alternatives like BERT and GPT address some of these issues.

```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get contextual embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Example usage
text1 = "I love Python programming"
text2 = "Python is a type of snake"

embedding1 = get_bert_embedding(text1)
embedding2 = get_bert_embedding(text2)

print("BERT embedding dimensions:", embedding1.shape)
print("Cosine similarity:", np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
```

Slide 16: Additional Resources

For those interested in diving deeper into Word2Vec and its applications in NLP, here are some valuable resources:

1. Original Word2Vec paper: "Efficient Estimation of Word Representations in Vector Space" by Mikolov et al. (2013) ArXiv link: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
2. "Distributed Representations of Words and Phrases and their Compositionality" by Mikolov et al. (2013) ArXiv link: [https://arxiv.org/abs/1310.4546](https://arxiv.org/abs/1310.4546)
3. "word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method" by Goldberg and Levy (2014) ArXiv link: [https://arxiv.org/abs/1402.3722](https://arxiv.org/abs/1402.3722)
4. "GloVe: Global Vectors for Word Representation" by Pennington et al. (2014) ArXiv link: [https://arxiv.org/abs/1405.4053](https://arxiv.org/abs/1405.4053)

These papers provide in-depth explanations of Word2Vec and related techniques, offering valuable insights into the theoretical foundations and practical applications of word embeddings in NLP.

