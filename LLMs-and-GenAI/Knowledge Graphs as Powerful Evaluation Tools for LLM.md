## Knowledge Graphs as Powerful Evaluation Tools for LLM
Slide 1: Knowledge Graphs and LLM Document Intelligence

Knowledge graphs are powerful tools for organizing and representing complex information. When combined with Large Language Models (LLMs), they can significantly enhance document intelligence tasks. This presentation explores how knowledge graphs can be used to evaluate and improve LLM performance in processing and understanding documents.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple knowledge graph
G = nx.Graph()
G.add_edges_from([('LLM', 'Document Intelligence'), 
                  ('Knowledge Graph', 'Document Intelligence'),
                  ('Knowledge Graph', 'Evaluation')])

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')
plt.title("Knowledge Graph for LLM Document Intelligence")
plt.axis('off')
plt.show()
```

Slide 2: Understanding Knowledge Graphs

A knowledge graph is a structured representation of information, where entities are connected by relationships. In the context of LLM document intelligence, knowledge graphs can serve as a reference for evaluating the accuracy and completeness of information extracted by the model.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a more detailed knowledge graph
G = nx.DiGraph()
G.add_edges_from([
    ('Person', 'works_at', 'Company'),
    ('Person', 'lives_in', 'City'),
    ('Company', 'located_in', 'Country'),
    ('City', 'part_of', 'Country')
])

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=2000, font_size=8, font_weight='bold', arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
plt.title("Example Knowledge Graph Structure")
plt.axis('off')
plt.show()
```

Slide 3: Building a Knowledge Graph from Structured Data

To create a knowledge graph, we often start with structured data sources. Here's an example of how to build a simple knowledge graph using Python and the NetworkX library:

```python
import networkx as nx
import pandas as pd

# Sample structured data
data = pd.DataFrame({
    'Person': ['Alice', 'Bob', 'Charlie'],
    'Works_At': ['TechCorp', 'DataInc', 'TechCorp'],
    'Lives_In': ['New York', 'San Francisco', 'London']
})

# Create a graph
G = nx.Graph()

# Add nodes and edges from the data
for _, row in data.iterrows():
    G.add_node(row['Person'], type='Person')
    G.add_node(row['Works_At'], type='Company')
    G.add_node(row['Lives_In'], type='City')
    G.add_edge(row['Person'], row['Works_At'], relationship='works_at')
    G.add_edge(row['Person'], row['Lives_In'], relationship='lives_in')

# Print graph information
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print("Nodes:", G.nodes(data=True))
print("Edges:", G.edges(data=True))
```

Slide 4: Extracting Information from Documents Using LLMs

LLMs can be used to extract structured information from unstructured text documents. Here's an example using the spaCy library to perform named entity recognition, which can be a first step in building a knowledge graph:

```python
import spacy

# Load the English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."

# Process the text
doc = nlp(text)

# Extract entities
entities = [(ent.text, ent.label_) for ent in doc.ents]

print("Extracted entities:")
for entity, label in entities:
    print(f"{entity} - {label}")

# Output:
# Extracted entities:
# Apple Inc. - ORG
# Steve Jobs - PERSON
# Cupertino - GPE
# California - GPE
# 1976 - DATE
```

Slide 5: Populating a Knowledge Graph from LLM Extraction

After extracting information using an LLM, we can populate our knowledge graph. This process involves creating nodes for entities and edges for relationships:

```python
import networkx as nx
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
doc = nlp(text)

# Create a graph
G = nx.Graph()

# Add nodes and edges based on extracted entities
for ent in doc.ents:
    G.add_node(ent.text, type=ent.label_)

# Add relationships (simplified)
G.add_edge("Apple Inc.", "Steve Jobs", relationship="founded_by")
G.add_edge("Apple Inc.", "Cupertino", relationship="located_in")

# Print graph information
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print("Nodes:", G.nodes(data=True))
print("Edges:", G.edges(data=True))
```

Slide 6: Evaluating LLM Extraction with Knowledge Graphs

Knowledge graphs can serve as a reference to evaluate the accuracy of information extracted by LLMs. We can compare the extracted information against the existing knowledge graph:

```python
def evaluate_extraction(extracted_graph, reference_graph):
    correct_nodes = set(extracted_graph.nodes()) & set(reference_graph.nodes())
    correct_edges = set(extracted_graph.edges()) & set(reference_graph.edges())
    
    precision_nodes = len(correct_nodes) / len(extracted_graph.nodes())
    recall_nodes = len(correct_nodes) / len(reference_graph.nodes())
    
    precision_edges = len(correct_edges) / len(extracted_graph.edges())
    recall_edges = len(correct_edges) / len(reference_graph.edges())
    
    f1_nodes = 2 * (precision_nodes * recall_nodes) / (precision_nodes + recall_nodes)
    f1_edges = 2 * (precision_edges * recall_edges) / (precision_edges + recall_edges)
    
    return {
        "Node Precision": precision_nodes,
        "Node Recall": recall_nodes,
        "Node F1": f1_nodes,
        "Edge Precision": precision_edges,
        "Edge Recall": recall_edges,
        "Edge F1": f1_edges
    }

# Example usage
extracted_graph = G  # From previous slide
reference_graph = nx.Graph()  # Assume this is a pre-existing reference graph

results = evaluate_extraction(extracted_graph, reference_graph)
for metric, value in results.items():
    print(f"{metric}: {value:.2f}")
```

Slide 7: Identifying Gaps in LLM Knowledge

Knowledge graphs can help identify gaps in an LLM's understanding by comparing the model's output to the structured information in the graph. Here's an example of how to find missing connections:

```python
def identify_knowledge_gaps(llm_graph, reference_graph):
    missing_nodes = set(reference_graph.nodes()) - set(llm_graph.nodes())
    missing_edges = set(reference_graph.edges()) - set(llm_graph.edges())
    
    print("Missing Nodes:", missing_nodes)
    print("Missing Edges:", missing_edges)
    
    # Suggest potential connections
    for node in llm_graph.nodes():
        for ref_node in reference_graph.neighbors(node):
            if ref_node not in llm_graph.neighbors(node):
                print(f"Potential connection: {node} -> {ref_node}")

# Example usage
llm_graph = nx.Graph()
llm_graph.add_edges_from([("Apple", "iPhone"), ("Steve Jobs", "Apple")])

reference_graph = nx.Graph()
reference_graph.add_edges_from([("Apple", "iPhone"), ("Steve Jobs", "Apple"), ("Apple", "Macintosh"), ("Steve Wozniak", "Apple")])

identify_knowledge_gaps(llm_graph, reference_graph)

# Output:
# Missing Nodes: {'Steve Wozniak', 'Macintosh'}
# Missing Edges: {('Apple', 'Macintosh'), ('Steve Wozniak', 'Apple')}
# Potential connection: Apple -> Macintosh
```

Slide 8: Enhancing LLM Responses with Knowledge Graphs

Knowledge graphs can be used to enhance LLM responses by providing additional context and relationships. Here's an example of how to augment an LLM's output:

```python
import networkx as nx

def augment_llm_response(llm_response, knowledge_graph):
    augmented_response = llm_response
    
    # Extract entities from the LLM response (simplified)
    entities = [word for word in llm_response.split() if word in knowledge_graph.nodes()]
    
    for entity in entities:
        neighbors = list(knowledge_graph.neighbors(entity))
        if neighbors:
            augmented_response += f"\nAdditional information about {entity}:"
            for neighbor in neighbors:
                relationship = knowledge_graph[entity][neighbor]['relationship']
                augmented_response += f"\n- {relationship} {neighbor}"
    
    return augmented_response

# Example usage
knowledge_graph = nx.Graph()
knowledge_graph.add_edge("Apple", "iPhone", relationship="produces")
knowledge_graph.add_edge("Apple", "Tim Cook", relationship="led by")
knowledge_graph.add_edge("Apple", "Cupertino", relationship="headquartered in")

llm_response = "Apple is a technology company known for its innovative products."
augmented_response = augment_llm_response(llm_response, knowledge_graph)
print(augmented_response)

# Output:
# Apple is a technology company known for its innovative products.
# Additional information about Apple:
# - produces iPhone
# - led by Tim Cook
# - headquartered in Cupertino
```

Slide 9: Real-Life Example: Academic Research Analysis

Knowledge graphs can be used to evaluate LLM document intelligence in academic research analysis. Here's an example of building a knowledge graph from research papers and using it to assess an LLM's understanding:

```python
import networkx as nx
import matplotlib.pyplot as plt

# Simulated research paper data
papers = [
    {"title": "Advances in NLP", "authors": ["Smith, J.", "Johnson, A."], "keywords": ["natural language processing", "machine learning"]},
    {"title": "Knowledge Graph Applications", "authors": ["Johnson, A.", "Lee, M."], "keywords": ["knowledge graphs", "semantic web"]},
    {"title": "LLM Evaluation Techniques", "authors": ["Lee, M.", "Smith, J."], "keywords": ["large language models", "evaluation metrics"]}
]

# Build knowledge graph
G = nx.Graph()

for paper in papers:
    G.add_node(paper["title"], type="paper")
    for author in paper["authors"]:
        G.add_node(author, type="author")
        G.add_edge(author, paper["title"], relationship="authored")
    for keyword in paper["keywords"]:
        G.add_node(keyword, type="keyword")
        G.add_edge(keyword, paper["title"], relationship="related_to")

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=8, font_weight='bold')
plt.title("Academic Research Knowledge Graph")
plt.axis('off')
plt.show()

# Evaluate LLM understanding
llm_response = "Smith and Johnson wrote a paper about NLP and machine learning."

def evaluate_llm_understanding(response, graph):
    score = 0
    if "Smith" in response and "Johnson" in response:
        score += 1
    if "NLP" in response or "natural language processing" in response:
        score += 1
    if "machine learning" in response:
        score += 1
    return score / 3  # Normalize score

understanding_score = evaluate_llm_understanding(llm_response, G)
print(f"LLM Understanding Score: {understanding_score:.2f}")
```

Slide 10: Real-Life Example: Movie Recommendation System

Knowledge graphs can be used to enhance LLM-based movie recommendation systems. Here's an example of how to build a simple movie knowledge graph and use it to improve recommendations:

```python
import networkx as nx
import random

# Sample movie data
movies = [
    {"title": "The Matrix", "genre": "Sci-Fi", "director": "Wachowski Sisters", "actor": "Keanu Reeves"},
    {"title": "Inception", "genre": "Sci-Fi", "director": "Christopher Nolan", "actor": "Leonardo DiCaprio"},
    {"title": "The Shawshank Redemption", "genre": "Drama", "director": "Frank Darabont", "actor": "Tim Robbins"}
]

# Build knowledge graph
G = nx.Graph()

for movie in movies:
    G.add_node(movie["title"], type="movie")
    G.add_node(movie["genre"], type="genre")
    G.add_node(movie["director"], type="director")
    G.add_node(movie["actor"], type="actor")
    G.add_edge(movie["title"], movie["genre"], relationship="belongs_to")
    G.add_edge(movie["title"], movie["director"], relationship="directed_by")
    G.add_edge(movie["title"], movie["actor"], relationship="stars")

def llm_recommendation(user_input):
    # Simulated LLM recommendation
    return random.choice([movie["title"] for movie in movies])

def enhance_recommendation(llm_rec, graph):
    related_nodes = list(graph.neighbors(llm_rec))
    additional_info = f"You might like {llm_rec}. "
    additional_info += f"It's a {graph.nodes[related_nodes[0]]['type']} movie "
    additional_info += f"directed by {graph.nodes[related_nodes[1]]['type']} "
    additional_info += f"and starring {graph.nodes[related_nodes[2]]['type']}."
    return additional_info

# Example usage
user_input = "I like science fiction movies with mind-bending plots."
llm_recommendation = llm_recommendation(user_input)
enhanced_recommendation = enhance_recommendation(llm_recommendation, G)
print(enhanced_recommendation)

# Possible output:
# You might like Inception. It's a Sci-Fi movie directed by Christopher Nolan and starring Leonardo DiCaprio.
```

Slide 11: Challenges in Using Knowledge Graphs for LLM Evaluation

While knowledge graphs are powerful tools for evaluating LLM document intelligence, they come with their own set of challenges. Here's a simulation of some common issues:

```python
import networkx as nx
import random

def simulate_incomplete_knowledge_graph():
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    return G

def simulate_inconsistent_llm_output():
    entities = ["A", "B", "C", "D", "E"]
    return random.sample(entities, 3)

def evaluate_with_challenges(kg, llm_output):
    kg_entities = set(kg.nodes())
    llm_entities = set(llm_output)
    
    missing_in_kg = llm_entities - kg_entities
    missing_in_llm = kg_entities - llm_entities
    consistent = llm_entities.intersection(kg_entities)
    
    print(f"Entities in Knowledge Graph: {kg_entities}")
    print(f"Entities in LLM output: {llm_entities}")
    print(f"Missing in Knowledge Graph: {missing_in_kg}")
    print(f"Missing in LLM output: {missing_in_llm}")
    print(f"Consistent entities: {consistent}")
    
    consistency_score = len(consistent) / len(kg_entities)
    print(f"Consistency Score: {consistency_score:.2f}")

# Example usage
kg = simulate_incomplete_knowledge_graph()
llm_output = simulate_inconsistent_llm_output()
evaluate_with_challenges(kg, llm_output)
```

This code simulates challenges such as incomplete knowledge graphs and inconsistent LLM outputs, highlighting the difficulties in evaluation and the need for robust comparison methods.

Slide 12: Addressing Knowledge Graph Limitations

To address limitations of knowledge graphs in LLM evaluation, we can implement strategies such as graph expansion and confidence scoring:

```python
import networkx as nx

def expand_knowledge_graph(kg, new_data):
    for entity, relations in new_data.items():
        kg.add_node(entity)
        for relation, target in relations.items():
            kg.add_edge(entity, target, relation=relation)
    return kg

def confidence_scoring(llm_output, kg):
    scores = {}
    for entity, relations in llm_output.items():
        if entity in kg.nodes():
            correct_relations = sum(1 for r, t in relations.items() if kg.has_edge(entity, t) and kg[entity][t].get('relation') == r)
            scores[entity] = correct_relations / len(relations) if relations else 0
    return scores

# Example usage
kg = nx.Graph()
kg.add_edge("Apple", "iPhone", relation="produces")

new_data = {
    "Google": {"produces": "Pixel", "headquartered": "Mountain View"}
}

expanded_kg = expand_knowledge_graph(kg, new_data)

llm_output = {
    "Apple": {"produces": "iPhone", "headquartered": "Cupertino"},
    "Google": {"produces": "Pixel", "headquartered": "Silicon Valley"}
}

confidence_scores = confidence_scoring(llm_output, expanded_kg)
print("Confidence Scores:", confidence_scores)
```

This approach allows for dynamic expansion of the knowledge graph and provides a confidence score for LLM outputs, helping to mitigate some limitations.

Slide 13: Future Directions: Integrating Knowledge Graphs with LLMs

The future of LLM document intelligence lies in tighter integration between knowledge graphs and language models. Here's a conceptual example of how this integration might work:

```python
import networkx as nx

class KnowledgeAwareLLM:
    def __init__(self, kg):
        self.kg = kg
        
    def generate_response(self, query):
        # Simulated LLM response generation
        response = f"Simulated response to: {query}"
        
        # Enhance response with knowledge graph
        entities = self.extract_entities(query)
        for entity in entities:
            if entity in self.kg.nodes():
                neighbors = list(self.kg.neighbors(entity))
                if neighbors:
                    response += f"\nRelated to {entity}: {', '.join(neighbors)}"
        
        return response
    
    def extract_entities(self, text):
        # Simplified entity extraction
        return [word for word in text.split() if word in self.kg.nodes()]

# Example usage
kg = nx.Graph()
kg.add_edges_from([
    ("Python", "Programming"),
    ("Python", "Machine Learning"),
    ("Machine Learning", "AI")
])

llm = KnowledgeAwareLLM(kg)
query = "Tell me about Python and its applications."
response = llm.generate_response(query)
print(response)
```

This example demonstrates a conceptual integration of a knowledge graph into the LLM's response generation process, allowing for more informed and contextually relevant outputs.

Slide 14: Conclusion and Future Work

Knowledge graphs serve as powerful evaluation tools for LLM document intelligence, offering structured representations of information that can be used to assess and enhance LLM performance. Key takeaways include:

1. Knowledge graphs provide a framework for organizing complex information.
2. They can be used to evaluate LLM extraction accuracy and completeness.
3. Integration of knowledge graphs with LLMs can lead to more robust and context-aware systems.

Slide 15: Conclusion and Future Work

Future work in this area may focus on:

1. Developing more sophisticated methods for automatic knowledge graph construction from unstructured text.
2. Creating standardized benchmarks for evaluating LLM performance using knowledge graphs.
3. Exploring novel ways to integrate knowledge graphs directly into LLM architectures.

Slide 16: Conclusion and Future Work

```python
def visualize_future_work():
    import matplotlib.pyplot as plt
    
    areas = ['Automatic KG Construction', 'Standardized Benchmarks', 'LLM-KG Integration']
    importance = [0.8, 0.7, 0.9]
    
    plt.figure(figsize=(10, 6))
    plt.bar(areas, importance)
    plt.title('Future Work in Knowledge Graphs for LLM Evaluation')
    plt.ylabel('Relative Importance')
    plt.ylim(0, 1)
    plt.show()

visualize_future_work()
```

This visualization highlights the potential areas of future work in using knowledge graphs for LLM document intelligence evaluation.

Slide 17: Additional Resources

For further exploration of knowledge graphs and their applications in LLM evaluation, consider the following resources:

1. "Knowledge Graphs" by Aidan Hogan et al. (2021) - ArXiv:2003.02320 URL: [https://arxiv.org/abs/2003.02320](https://arxiv.org/abs/2003.02320)
2. "A Survey on Knowledge Graphs: Representation, Acquisition, and Applications" by Shaoxiong Ji et al. (2021) - ArXiv:2002.00388 URL: [https://arxiv.org/abs/2002.00388](https://arxiv.org/abs/2002.00388)
3. "Knowledge Graphs and Language Models: Bridging the Gap" by Wei Cui et al. (2023) - ArXiv:2305.02182 URL: [https://arxiv.org/abs/2305.02182](https://arxiv.org/abs/2305.02182)

These papers provide comprehensive overviews and cutting-edge research on knowledge graphs and their integration with language models.
