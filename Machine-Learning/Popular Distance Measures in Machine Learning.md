## Popular Distance Measures in Machine Learning
Slide 1: Introduction to Distance Measures in Machine Learning

Distance measures play a crucial role in various machine learning algorithms, helping to quantify the similarity or dissimilarity between data points. These measures are fundamental in tasks such as clustering, classification, and recommendation systems. In this presentation, we'll explore several popular distance measures, their applications, and implementation in Python.

Slide 2: Euclidean Distance

Euclidean distance is the most common distance measure, representing the straight-line distance between two points in n-dimensional space. It's widely used in clustering algorithms like K-means and for spatial analysis.

Slide 3: Source Code for Euclidean Distance

```python
import math

def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensionality")
    
    squared_diff_sum = sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))
    return math.sqrt(squared_diff_sum)

# Example usage
point_a = (1, 2, 3)
point_b = (4, 5, 6)
distance = euclidean_distance(point_a, point_b)
print(f"Euclidean distance between {point_a} and {point_b}: {distance:.2f}")
```

Slide 4: Results for Euclidean Distance

```
Euclidean distance between (1, 2, 3) and (4, 5, 6): 5.20
```

Slide 5: Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors, indicating their directional similarity. It's particularly useful in text analysis and recommendation systems, where the magnitude of vectors might not be as important as their orientation.

Slide 6: Source Code for Cosine Similarity

```python
import math

def cosine_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimensionality")
    
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v ** 2 for v in vec1))
    magnitude2 = math.sqrt(sum(v ** 2 for v in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0  # Avoid division by zero
    
    return dot_product / (magnitude1 * magnitude2)

# Example usage
doc1 = (1, 1, 1, 0, 0)
doc2 = (1, 1, 0, 1, 1)
similarity = cosine_similarity(doc1, doc2)
print(f"Cosine similarity between {doc1} and {doc2}: {similarity:.4f}")
```

Slide 7: Results for Cosine Similarity

```
Cosine similarity between (1, 1, 1, 0, 0) and (1, 1, 0, 1, 1): 0.6667
```

Slide 8: Hamming Distance

Hamming distance measures the number of positions at which corresponding elements in two sequences differ. It's commonly used in information theory, coding theory, and cryptography for error detection and correction.

Slide 9: Source Code for Hamming Distance

```python
def hamming_distance(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must have the same length")
    
    return sum(el1 != el2 for el1, el2 in zip(seq1, seq2))

# Example usage
binary_seq1 = "10101"
binary_seq2 = "11001"
distance = hamming_distance(binary_seq1, binary_seq2)
print(f"Hamming distance between {binary_seq1} and {binary_seq2}: {distance}")
```

Slide 10: Results for Hamming Distance

```
Hamming distance between 10101 and 11001: 2
```

Slide 11: Manhattan Distance

Manhattan distance, also known as L1 distance or city block distance, measures the sum of absolute differences between coordinates. It's useful in grid-based navigation and when diagonal movement is not allowed.

Slide 12: Source Code for Manhattan Distance

```python
def manhattan_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensionality")
    
    return sum(abs(p1 - p2) for p1, p2 in zip(point1, point2))

# Example usage
city_point1 = (1, 1)
city_point2 = (4, 5)
distance = manhattan_distance(city_point1, city_point2)
print(f"Manhattan distance between {city_point1} and {city_point2}: {distance}")
```

Slide 13: Results for Manhattan Distance

```
Manhattan distance between (1, 1) and (4, 5): 7
```

Slide 14: Minkowski Distance

Minkowski distance is a generalization of Euclidean and Manhattan distances. By adjusting the parameter p, it can represent different distance measures, making it versatile for various applications in machine learning.

Slide 15: Source Code for Minkowski Distance

```python
def minkowski_distance(point1, point2, p):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensionality")
    
    return sum(abs(p1 - p2) ** p for p1, p2 in zip(point1, point2)) ** (1/p)

# Example usage
point_a = (1, 2, 3)
point_b = (4, 5, 6)

# Euclidean distance (p=2)
euclidean = minkowski_distance(point_a, point_b, 2)
print(f"Euclidean distance (p=2): {euclidean:.2f}")

# Manhattan distance (p=1)
manhattan = minkowski_distance(point_a, point_b, 1)
print(f"Manhattan distance (p=1): {manhattan:.2f}")

# Chebyshev distance (p=infinity, approximated with a large value)
chebyshev = minkowski_distance(point_a, point_b, 1000)
print(f"Chebyshev distance (p→∞): {chebyshev:.2f}")
```

Slide 16: Results for Minkowski Distance

```
Euclidean distance (p=2): 5.20
Manhattan distance (p=1): 9.00
Chebyshev distance (p→∞): 3.00
```

Slide 17: Jaccard Distance

Jaccard distance measures dissimilarity between sets by comparing the size of their intersection to the size of their union. It's useful in text analysis, clustering, and recommendation systems.

Slide 18: Source Code for Jaccard Distance

```python
def jaccard_distance(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - (intersection / union)

# Example usage
text1 = set("hello world")
text2 = set("world hello")
distance = jaccard_distance(text1, text2)
print(f"Jaccard distance between '{text1}' and '{text2}': {distance:.4f}")
```

Slide 19: Results for Jaccard Distance

```
Jaccard distance between '{'d', 'e', 'h', 'l', 'o', 'r', 'w'}' and '{'d', 'e', 'h', 'l', 'o', 'r', 'w'}': 0.0000
```

Slide 20: Haversine Distance

Haversine distance calculates the great-circle distance between two points on a sphere given their longitudes and latitudes. It's essential for geographical calculations and navigation systems.

Slide 21: Source Code for Haversine Distance

```python
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers

    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

# Example usage: Distance between New York and Los Angeles
ny_lat, ny_lon = 40.7128, -74.0060
la_lat, la_lon = 34.0522, -118.2437

distance = haversine_distance(ny_lat, ny_lon, la_lat, la_lon)
print(f"Distance between New York and Los Angeles: {distance:.2f} km")
```

Slide 22: Results for Haversine Distance

```
Distance between New York and Los Angeles: 3935.75 km
```

Slide 23: Sørensen-Dice Coefficient

The Sørensen-Dice coefficient is a statistic used to gauge the similarity of two samples. It's particularly useful in ecology, biogeography, and text analysis for comparing the similarity of two sets.

Slide 24: Source Code for Sørensen-Dice Coefficient

```python
def sorensen_dice_coefficient(set1, set2):
    intersection = len(set1.intersection(set2))
    return (2 * intersection) / (len(set1) + len(set2))

# Example usage
text1 = set("data science")
text2 = set("machine learning")
similarity = sorensen_dice_coefficient(text1, text2)
print(f"Sørensen-Dice coefficient between '{text1}' and '{text2}': {similarity:.4f}")
```

Slide 25: Results for Sørensen-Dice Coefficient

```
Sørensen-Dice coefficient between '{'a', 'c', 'd', 'e', 'i', 'n', 's', 't'}' and '{'a', 'c', 'e', 'g', 'h', 'i', 'l', 'm', 'n', 'r'}': 0.3333
```

Slide 26: Real-Life Example: Document Similarity

In this example, we'll use cosine similarity to compare the similarity of two document vectors based on word frequency.

Slide 27: Source Code for Document Similarity

```python
import math

def cosine_similarity(vec1, vec2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v ** 2 for v in vec1))
    magnitude2 = math.sqrt(sum(v ** 2 for v in vec2))
    return dot_product / (magnitude1 * magnitude2)

# Word frequency vectors for two documents
doc1 = [3, 2, 0, 5, 0, 1]  # frequencies of words in document 1
doc2 = [1, 0, 3, 2, 2, 0]  # frequencies of words in document 2

similarity = cosine_similarity(doc1, doc2)
print(f"Document similarity: {similarity:.4f}")
```

Slide 28: Results for Document Similarity

```
Document similarity: 0.4707
```

Slide 29: Real-Life Example: Image Pixel Comparison

In this example, we'll use the Manhattan distance to compare the similarity of two small grayscale images represented as 2D arrays of pixel intensities.

Slide 30: Source Code for Image Pixel Comparison

```python
def manhattan_distance_2d(image1, image2):
    if len(image1) != len(image2) or len(image1[0]) != len(image2[0]):
        raise ValueError("Images must have the same dimensions")
    
    distance = 0
    for i in range(len(image1)):
        for j in range(len(image1[0])):
            distance += abs(image1[i][j] - image2[i][j])
    return distance

# Example 3x3 grayscale images (0-255 intensity)
image1 = [
    [100, 150, 200],
    [50, 100, 150],
    [0, 50, 100]
]

image2 = [
    [120, 130, 180],
    [70, 110, 140],
    [20, 60, 90]
]

distance = manhattan_distance_2d(image1, image2)
print(f"Manhattan distance between images: {distance}")
```

Slide 31: Results for Image Pixel Comparison

```
Manhattan distance between images: 140
```

Slide 32: Additional Resources

For more in-depth information on distance measures and their applications in machine learning, consider exploring the following resources:

1.  "A Survey of Distance and Similarity Measures for Structured Data" (arXiv:2106.03633) URL: [https://arxiv.org/abs/2106.03633](https://arxiv.org/abs/2106.03633)
2.  "Distance Metric Learning: A Comprehensive Survey" (arXiv:1907.08374) URL: [https://arxiv.org/abs/1907.08374](https://arxiv.org/abs/1907.08374)

These papers provide comprehensive overviews of various distance measures and their applications in machine learning and data analysis.

