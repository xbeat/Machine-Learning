## Comparing CSV Files Using Machine Learning in Python
Slide 1: Comparing CSV Files for Similarity Using Machine Learning in Python

Machine learning techniques can be applied to compare CSV files for similarity. This process involves data preprocessing, feature extraction, and utilizing various algorithms to measure the similarity between files. Let's explore this topic step by step.

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_csv(file_path):
    return pd.read_csv(file_path)

def preprocess_csv(df):
    return df.fillna('').astype(str).apply(' '.join, axis=1)

def compare_csv_files(file1, file2):
    df1 = load_csv(file1)
    df2 = load_csv(file2)
    
    text1 = preprocess_csv(df1)
    text2 = preprocess_csv(df2)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(text1), ' '.join(text2)])
    
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity

similarity_score = compare_csv_files('file1.csv', 'file2.csv')
print(f"Similarity score: {similarity_score}")
```

Slide 2: Data Preprocessing

Before comparing CSV files, it's crucial to preprocess the data. This step involves handling missing values, converting data types, and transforming the data into a format suitable for analysis.

```python
import pandas as pd

def preprocess_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df = df.fillna('')
    
    # Convert all columns to string type
    df = df.astype(str)
    
    # Combine all columns into a single text column
    df['text'] = df.apply(' '.join, axis=1)
    
    return df['text']

preprocessed_data = preprocess_csv('example.csv')
print(preprocessed_data.head())
```

Slide 3: Feature Extraction using TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) is a popular technique for converting text data into numerical features. It assigns weights to words based on their importance in the document and across the corpus.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(text_data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return tfidf_matrix, vectorizer

# Example usage
text_data = ["This is the first document.", "This document is the second document."]
tfidf_matrix, vectorizer = extract_features(text_data)

print("TF-IDF Matrix shape:", tfidf_matrix.shape)
print("Feature names:", vectorizer.get_feature_names_out())
```

Slide 4: Cosine Similarity

Cosine similarity is a measure of similarity between two non-zero vectors. In the context of comparing CSV files, we use it to determine how similar the TF-IDF vectors of two files are.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_cosine_similarity(matrix):
    similarity = cosine_similarity(matrix)
    return similarity

# Example usage
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
similarity_matrix = calculate_cosine_similarity(matrix)

print("Cosine Similarity Matrix:")
print(similarity_matrix)
```

Slide 5: Jaccard Similarity

Jaccard similarity is another method for comparing the similarity of two sets. It's particularly useful when dealing with binary or categorical data in CSV files.

```python
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Example usage
csv1_set = set(['apple', 'banana', 'cherry'])
csv2_set = set(['banana', 'cherry', 'date'])

similarity = jaccard_similarity(csv1_set, csv2_set)
print(f"Jaccard Similarity: {similarity}")
```

Slide 6: Comparing Numerical Data

When dealing with numerical data in CSV files, we can use statistical measures like correlation coefficients to compare similarity.

```python
import pandas as pd
import numpy as np

def compare_numerical_data(file1, file2, method='pearson'):
    df1 = pd.read_csv(file1, index_col=0)
    df2 = pd.read_csv(file2, index_col=0)
    
    # Ensure both dataframes have the same columns
    common_columns = df1.columns.intersection(df2.columns)
    df1 = df1[common_columns]
    df2 = df2[common_columns]
    
    correlation = df1.corrwith(df2, method=method)
    return correlation

# Example usage
correlation = compare_numerical_data('data1.csv', 'data2.csv')
print("Correlation between numerical columns:")
print(correlation)
```

Slide 7: Handling Large CSV Files

When dealing with large CSV files, it's important to use efficient methods to avoid memory issues. We can use Python's built-in CSV module and process the files in chunks.

```python
import csv
from collections import Counter

def process_large_csv(file_path, chunk_size=1000):
    word_counts = Counter()
    
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        chunk = []
        
        for i, row in enumerate(reader):
            chunk.append(' '.join(row))
            
            if i % chunk_size == 0 and i > 0:
                word_counts.update(' '.join(chunk).split())
                chunk = []
        
        # Process any remaining rows
        if chunk:
            word_counts.update(' '.join(chunk).split())
    
    return word_counts

# Example usage
word_counts = process_large_csv('large_file.csv')
print("Top 10 most common words:")
print(word_counts.most_common(10))
```

Slide 8: Real-Life Example: Comparing Product Catalogs

Imagine an e-commerce company wants to compare its product catalog with a competitor's. We can use our CSV comparison techniques to identify similarities and differences.

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compare_product_catalogs(catalog1, catalog2):
    df1 = pd.read_csv(catalog1)
    df2 = pd.read_csv(catalog2)
    
    # Combine product name and description
    df1['product_text'] = df1['name'] + ' ' + df1['description']
    df2['product_text'] = df2['name'] + ' ' + df2['description']
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(pd.concat([df1['product_text'], df2['product_text']]))
    
    similarity_matrix = cosine_similarity(tfidf_matrix[:len(df1)], tfidf_matrix[len(df1):])
    
    return similarity_matrix

# Example usage
similarity_matrix = compare_product_catalogs('our_catalog.csv', 'competitor_catalog.csv')
print("Shape of similarity matrix:", similarity_matrix.shape)
print("Average similarity score:", similarity_matrix.mean())
```

Slide 9: Real-Life Example: Comparing Scientific Papers

Researchers often need to compare large sets of scientific papers to identify similar works or potential plagiarism. We can apply our CSV comparison techniques to this task.

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compare_scientific_papers(papers_csv):
    df = pd.read_csv(papers_csv)
    
    # Combine title and abstract
    df['text'] = df['title'] + ' ' + df['abstract']
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    return similarity_matrix, df

# Example usage
similarity_matrix, papers_df = compare_scientific_papers('scientific_papers.csv')

# Find highly similar papers
threshold = 0.8
similar_papers = []

for i in range(len(papers_df)):
    for j in range(i+1, len(papers_df)):
        if similarity_matrix[i][j] > threshold:
            similar_papers.append((papers_df.iloc[i]['title'], papers_df.iloc[j]['title'], similarity_matrix[i][j]))

print("Highly similar papers:")
for paper1, paper2, similarity in similar_papers[:5]:
    print(f"{paper1} <-> {paper2}: {similarity:.2f}")
```

Slide 10: Dimensionality Reduction for Visualization

When comparing multiple CSV files, we can use dimensionality reduction techniques like t-SNE to visualize the similarities in a 2D space.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_similarities(similarity_matrix):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(similarity_matrix)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings[:, 0], embeddings[:, 1])
    plt.title("t-SNE Visualization of CSV File Similarities")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()

# Example usage
similarity_matrix = np.random.rand(100, 100)  # Replace with your actual similarity matrix
visualize_similarities(similarity_matrix)
```

Slide 11: Handling Categorical Data

When comparing CSV files with categorical data, we can use techniques like one-hot encoding and then apply similarity measures.

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import jaccard_score

def compare_categorical_data(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Combine dataframes to ensure consistent encoding
    combined_df = pd.concat([df1, df2]).reset_index(drop=True)
    
    # One-hot encode categorical columns
    encoder = OneHotEncoder(sparse=False)
    encoded_data = encoder.fit_transform(combined_df)
    
    # Split the encoded data back into two datasets
    encoded_df1 = encoded_data[:len(df1)]
    encoded_df2 = encoded_data[len(df1):]
    
    # Calculate Jaccard similarity for each row
    similarities = [jaccard_score(row1, row2) for row1, row2 in zip(encoded_df1, encoded_df2)]
    
    return np.mean(similarities)

# Example usage
avg_similarity = compare_categorical_data('categorical_data1.csv', 'categorical_data2.csv')
print(f"Average Jaccard similarity: {avg_similarity:.2f}")
```

Slide 12: Handling Time Series Data

When comparing CSV files containing time series data, we can use techniques like Dynamic Time Warping (DTW) to measure similarity.

```python
import numpy as np
from dtaidistance import dtw

def compare_time_series(file1, file2):
    ts1 = np.genfromtxt(file1, delimiter=',')
    ts2 = np.genfromtxt(file2, delimiter=',')
    
    distance = dtw.distance(ts1, ts2)
    max_length = max(len(ts1), len(ts2))
    similarity = 1 - (distance / max_length)
    
    return similarity

# Example usage
similarity = compare_time_series('timeseries1.csv', 'timeseries2.csv')
print(f"Time series similarity: {similarity:.2f}")
```

Slide 13: Ensemble Approach for Robust Comparison

To get a more robust comparison of CSV files, we can use an ensemble approach that combines multiple similarity measures.

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

def ensemble_comparison(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Text-based similarity
    text_data1 = df1.select_dtypes(include=[object]).apply(lambda x: ' '.join(x.astype(str)), axis=1)
    text_data2 = df2.select_dtypes(include=[object]).apply(lambda x: ' '.join(x.astype(str)), axis=1)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(pd.concat([text_data1, text_data2]))
    cosine_sim = cosine_similarity(tfidf_matrix[:len(df1)], tfidf_matrix[len(df1):])[0][0]
    
    # Numerical similarity
    num_data1 = df1.select_dtypes(include=[np.number])
    num_data2 = df2.select_dtypes(include=[np.number])
    common_cols = set(num_data1.columns) & set(num_data2.columns)
    
    if common_cols:
        corr, _ = pearsonr(num_data1[common_cols].mean(), num_data2[common_cols].mean())
    else:
        corr = 0
    
    # Combine similarities
    ensemble_similarity = (cosine_sim + (corr + 1) / 2) / 2
    
    return ensemble_similarity

# Example usage
similarity = ensemble_comparison('file1.csv', 'file2.csv')
print(f"Ensemble similarity: {similarity:.2f}")
```

Slide 14: Additional Resources

For further exploration of CSV file comparison and machine learning techniques, consider the following resources:

1. "A Survey of Text Similarity Approaches" by Gomaa and Fahmy (2013) ArXiv: [https://arxiv.org/abs/1303.0291](https://arxiv.org/abs/1303.0291)
2. "Efficient and Robust Automated Machine Learning" by Feurer et al. (2015) ArXiv: [https://arxiv.org/abs/1507.05909](https://arxiv.org/abs/1507.05909)
3. "Dynamic Time Warping: A Review" by Senin (2008) ArXiv: [https://arxiv.org/abs/1101.1058](https://arxiv.org/abs/1101.1058)

These papers provide in-depth discussions on various similarity measures and machine learning approaches that can be applied to CSV file comparison tasks.

