## Understanding Knowledge Graphs with Python
Slide 1: Introduction to Knowledge Graphs

Knowledge graphs are structured representations of information that capture relationships between entities. They form the backbone of many modern information systems, enabling efficient data retrieval and inference. In this presentation, we'll explore how to work with knowledge graphs using Python.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple knowledge graph
G = nx.Graph()
G.add_edge("Person", "Alice", relation="instance_of")
G.add_edge("Person", "Bob", relation="instance_of")
G.add_edge("Alice", "Bob", relation="knows")

# Visualize the graph
nx.draw(G, with_labels=True, node_color='lightblue', node_size=1500, font_size=10)
plt.title("Simple Knowledge Graph")
plt.show()
```

Slide 2: Components of a Knowledge Graph

A knowledge graph consists of nodes (entities) and edges (relationships). Nodes represent concepts or objects, while edges define how these entities are connected. This structure allows for complex queries and inferences.

```python
# Define a more complex knowledge graph
kg = nx.DiGraph()

# Add nodes (entities)
entities = ["Alice", "Bob", "Computer Science", "Python", "Data Structures"]
kg.add_nodes_from(entities)

# Add edges (relationships)
relationships = [
    ("Alice", "Computer Science", "studies"),
    ("Bob", "Computer Science", "teaches"),
    ("Computer Science", "Python", "includes"),
    ("Python", "Data Structures", "uses")
]
kg.add_edges_from(relationships)

# Visualize the graph
pos = nx.spring_layout(kg)
nx.draw(kg, pos, with_labels=True, node_color='lightgreen', node_size=2000, font_size=8)
nx.draw_networkx_edge_labels(kg, pos, edge_labels={(u, v): d['weight'] for u, v, d in kg.edges(data=True)})
plt.title("Components of a Knowledge Graph")
plt.show()
```

Slide 3: Creating a Knowledge Graph in Python

We can use the NetworkX library to create and manipulate knowledge graphs in Python. This example demonstrates how to create a simple graph representing relationships between people and their interests.

```python
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges
G.add_edge("Alice", "Programming", relation="interested_in")
G.add_edge("Bob", "Machine Learning", relation="expert_in")
G.add_edge("Charlie", "Data Science", relation="studies")
G.add_edge("Data Science", "Machine Learning", relation="includes")
G.add_edge("Programming", "Python", relation="uses")

# Print graph information
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print("Edges:", G.edges(data=True))
```

Slide 4: Querying a Knowledge Graph

One of the key advantages of knowledge graphs is the ability to perform complex queries. Let's explore how to extract information from our graph using various NetworkX functions.

```python
# Continuing from the previous graph G

# Find all nodes connected to "Alice"
alice_connections = list(G.neighbors("Alice"))
print("Alice's connections:", alice_connections)

# Find the path between "Charlie" and "Python"
path = nx.shortest_path(G, "Charlie", "Python")
print("Path from Charlie to Python:", path)

# Find all nodes within 2 steps of "Data Science"
nearby_nodes = nx.ego_graph(G, "Data Science", radius=2)
print("Nodes near Data Science:", list(nearby_nodes.nodes()))

# Find nodes with no incoming edges (root nodes)
root_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]
print("Root nodes:", root_nodes)
```

Slide 5: Adding Attributes to Nodes and Edges

Enhancing our knowledge graph with additional attributes provides richer information and enables more sophisticated queries.

```python
import networkx as nx

G = nx.Graph()

# Add nodes with attributes
G.add_node("Alice", age=30, occupation="Software Engineer")
G.add_node("Bob", age=35, occupation="Data Scientist")

# Add edges with attributes
G.add_edge("Alice", "Bob", relationship="colleague", years_known=5)

# Access node attributes
print("Alice's occupation:", G.nodes["Alice"]["occupation"])

# Access edge attributes
print("Relationship between Alice and Bob:", G.edges["Alice", "Bob"]["relationship"])

# Update attributes
G.nodes["Alice"]["skills"] = ["Python", "JavaScript", "SQL"]
print("Alice's skills:", G.nodes["Alice"]["skills"])
```

Slide 6: Inferencing in Knowledge Graphs

Inference allows us to derive new knowledge from existing information in the graph. Let's implement a simple inference rule to demonstrate this concept.

```python
def infer_friendship(G):
    new_friendships = []
    for person1 in G.nodes():
        for person2 in G.nodes():
            if person1 != person2:
                common_friends = set(G.neighbors(person1)) & set(G.neighbors(person2))
                if len(common_friends) >= 2 and not G.has_edge(person1, person2):
                    new_friendships.append((person1, person2))
    return new_friendships

# Create a sample graph
G = nx.Graph()
G.add_edges_from([
    ("Alice", "Bob"), ("Alice", "Charlie"), ("Bob", "David"),
    ("Charlie", "David"), ("Eve", "Alice"), ("Eve", "Bob")
])

# Apply inference
new_friends = infer_friendship(G)
print("Inferred friendships:", new_friends)

# Add inferred relationships to the graph
G.add_edges_from(new_friends, relationship="inferred_friend")

# Visualize the updated graph
nx.draw(G, with_labels=True, node_color='lightblue', node_size=1500, font_size=10)
plt.title("Graph with Inferred Relationships")
plt.show()
```

Slide 7: Reasoning with Knowledge Graphs

Knowledge graphs enable various forms of reasoning. Let's implement a simple reasoning algorithm to find indirect relationships between entities.

```python
def find_indirect_relations(G, start, end, max_depth=3):
    paths = nx.all_simple_paths(G, start, end, cutoff=max_depth)
    indirect_relations = []
    for path in paths:
        if len(path) > 2:
            relation = " -> ".join([G[path[i]][path[i+1]]['relation'] for i in range(len(path)-1)])
            indirect_relations.append(f"{start} to {end}: {relation}")
    return indirect_relations

# Create a sample knowledge graph
G = nx.DiGraph()
G.add_edge("Alice", "Bob", relation="knows")
G.add_edge("Bob", "Charlie", relation="works_with")
G.add_edge("Charlie", "DataScience", relation="studies")
G.add_edge("DataScience", "MachineLearning", relation="includes")

# Find indirect relations
indirect = find_indirect_relations(G, "Alice", "MachineLearning")
for relation in indirect:
    print(relation)
```

Slide 8: Embedding Knowledge Graphs

Graph embeddings are vector representations of nodes that capture the structure and relationships in the graph. These embeddings can be used for various machine learning tasks.

```python
import networkx as nx
from node2vec import Node2Vec
import numpy as np

# Create a sample knowledge graph
G = nx.karate_club_graph()

# Generate embeddings
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Get embeddings for specific nodes
node_embeddings = {node: model.wv[str(node)] for node in G.nodes()}

# Example: Find similar nodes
def find_similar_nodes(node, top_n=5):
    target_embedding = node_embeddings[node]
    similarities = []
    for other_node, embedding in node_embeddings.items():
        if other_node != node:
            similarity = np.dot(target_embedding, embedding) / (np.linalg.norm(target_embedding) * np.linalg.norm(embedding))
            similarities.append((other_node, similarity))
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

# Find nodes similar to node 0
similar_nodes = find_similar_nodes(0)
print("Nodes similar to node 0:", similar_nodes)
```

Slide 9: Knowledge Graph Completion

Knowledge graph completion involves predicting missing links or relationships in the graph. Here's a simple example using a rule-based approach.

```python
import networkx as nx

def complete_knowledge_graph(G):
    new_edges = []
    for node in G.nodes():
        neighbors = set(G.neighbors(node))
        for neighbor in neighbors:
            neighbor_neighbors = set(G.neighbors(neighbor))
            potential_connections = neighbor_neighbors - neighbors - {node}
            for potential in potential_connections:
                if G.degree(potential) > 1:  # Simple heuristic
                    new_edges.append((node, potential, {'relation': 'inferred'}))
    return new_edges

# Create a sample knowledge graph
G = nx.Graph()
G.add_edges_from([
    ('Alice', 'Bob'), ('Bob', 'Charlie'), ('Charlie', 'David'),
    ('David', 'Eve'), ('Eve', 'Frank'), ('Frank', 'George')
])

# Complete the graph
new_edges = complete_knowledge_graph(G)
G.add_edges_from(new_edges)

print("New inferred edges:")
for edge in new_edges:
    print(f"{edge[0]} - {edge[1]}")

# Visualize the completed graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist=new_edges, edge_color='r', style='dashed')
plt.title("Completed Knowledge Graph")
plt.show()
```

Slide 10: Querying Knowledge Graphs with SPARQL

SPARQL is a query language for RDF data, which is often used to represent knowledge graphs. Here's an example of how to use SPARQL with Python to query a knowledge graph.

```python
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, FOAF

# Create a sample RDF graph
g = Graph()
n = Namespace("http://example.org/")

# Add some triples
g.add((n.Alice, RDF.type, FOAF.Person))
g.add((n.Alice, FOAF.name, Literal("Alice")))
g.add((n.Alice, FOAF.knows, n.Bob))
g.add((n.Bob, RDF.type, FOAF.Person))
g.add((n.Bob, FOAF.name, Literal("Bob")))

# SPARQL query to find all people and their names
query = """
SELECT ?person ?name
WHERE {
    ?person rdf:type foaf:Person .
    ?person foaf:name ?name .
}
"""

# Execute the query
results = g.query(query)

print("People in the knowledge graph:")
for row in results:
    print(f"{row.person} - {row.name}")

# SPARQL query to find who Alice knows
query_knows = """
SELECT ?known_person
WHERE {
    <http://example.org/Alice> foaf:knows ?known_person .
}
"""

results_knows = g.query(query_knows)

print("\nPeople Alice knows:")
for row in results_knows:
    print(row.known_person)
```

Slide 11: Real-life Example: Scientific Collaboration Network

Let's create a knowledge graph representing a scientific collaboration network, demonstrating how knowledge graphs can be used to analyze research ecosystems.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph for the scientific collaboration network
G = nx.DiGraph()

# Add researchers and their fields
researchers = [
    ("Dr. Smith", "Quantum Physics"),
    ("Dr. Johnson", "Machine Learning"),
    ("Dr. Williams", "Bioinformatics"),
    ("Dr. Brown", "Climate Science"),
    ("Dr. Davis", "Neuroscience")
]

for researcher, field in researchers:
    G.add_node(researcher, role="researcher")
    G.add_node(field, role="field")
    G.add_edge(researcher, field, relation="specializes_in")

# Add collaborations and joint publications
collaborations = [
    ("Dr. Smith", "Dr. Johnson", "Quantum Machine Learning"),
    ("Dr. Johnson", "Dr. Williams", "ML in Genomics"),
    ("Dr. Williams", "Dr. Brown", "Climate Impact on Biodiversity"),
    ("Dr. Brown", "Dr. Davis", "Environmental Neurotoxicology")
]

for researcher1, researcher2, topic in collaborations:
    G.add_node(topic, role="publication")
    G.add_edge(researcher1, topic, relation="co-authored")
    G.add_edge(researcher2, topic, relation="co-authored")
    G.add_edge(researcher1, researcher2, relation="collaborates_with")

# Analyze the network
print("Network Statistics:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Find researchers with the most collaborations
collaborations = sorted(G.degree(), key=lambda x: x[1], reverse=True)
print("\nTop collaborators:")
for researcher, degree in collaborations[:3]:
    if G.nodes[researcher]['role'] == 'researcher':
        print(f"{researcher}: {degree} connections")

# Visualize the network
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=8, font_weight='bold')
nx.draw_networkx_labels(G, pos, {node: node for node in G.nodes() if G.nodes[node]['role'] == 'researcher'}, font_size=10)
nx.draw_networkx_labels(G, pos, {node: node for node in G.nodes() if G.nodes[node]['role'] == 'field'}, font_size=8, font_color='red')
nx.draw_networkx_labels(G, pos, {node: node for node in G.nodes() if G.nodes[node]['role'] == 'publication'}, font_size=6, font_color='green')
plt.title("Scientific Collaboration Network")
plt.axis('off')
plt.tight_layout()
plt.show()
```

Slide 12: Real-life Example: Recipe Recommendation System

In this example, we'll create a knowledge graph for a recipe recommendation system, showcasing how knowledge graphs can be used in everyday applications.

```python
import networkx as nx
import matplotlib.pyplot as plt
import random

# Create a directed graph for the recipe recommendation system
G = nx.DiGraph()

# Add ingredients
ingredients = ["Tomato", "Cheese", "Pasta", "Beef", "Chicken", "Rice", "Egg", "Flour", "Sugar", "Chocolate"]
for ingredient in ingredients:
    G.add_node(ingredient, type="ingredient")

# Add recipes
recipes = [
    ("Spaghetti Bolognese", ["Pasta", "Beef", "Tomato"]),
    ("Margherita Pizza", ["Flour", "Tomato", "Cheese"]),
    ("Chicken Fried Rice", ["Rice", "Chicken", "Egg"]),
    ("Chocolate Cake", ["Flour", "Egg", "Sugar", "Chocolate"]),
    ("Omelette", ["Egg", "Cheese"])
]

for recipe, ingredients in recipes:
    G.add_node(recipe, type="recipe")
    for ingredient in ingredients:
        G.add_edge(ingredient, recipe, relation="used_in")

# Add user preferences (simulated)
users = ["Alice", "Bob", "Charlie"]
for user in users:
    G.add_node(user, type="user")
    liked_recipes = random.sample(recipes, 2)
    for recipe, _ in liked_recipes:
        G.add_edge(user, recipe, relation="likes")

# Function to recommend recipes based on user's liked ingredients
def recommend_recipes(G, user):
    liked_recipes = [recipe for recipe in G.successors(user)]
    liked_ingredients = set()
    for recipe in liked_recipes:
        liked_ingredients.update(G.predecessors(recipe))
    
    recommendations = set()
    for ingredient in liked_ingredients:
        recommendations.update(G.successors(ingredient))
    
    return list(recommendations - set(liked_recipes))

# Example recommendation
user = "Alice"
recommended = recommend_recipes(G, user)
print(f"Recommended recipes for {user}: {recommended}")

# Visualize the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=8)
nx.draw_networkx_labels(G, pos)
plt.title("Recipe Recommendation Knowledge Graph")
plt.axis('off')
plt.show()
```

Slide 13: Temporal Knowledge Graphs

Temporal knowledge graphs incorporate time information, allowing us to represent and reason about dynamic relationships. Let's explore a simple implementation.

```python
import networkx as nx
from datetime import datetime, timedelta

class TemporalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_temporal_edge(self, source, target, relation, start_time, end_time=None):
        self.graph.add_edge(source, target, relation=relation, start_time=start_time, end_time=end_time)

    def query_at_time(self, time):
        result = nx.MultiDiGraph()
        for u, v, data in self.graph.edges(data=True):
            if data['start_time'] <= time and (data['end_time'] is None or time <= data['end_time']):
                result.add_edge(u, v, **data)
        return result

# Create a temporal knowledge graph
tkg = TemporalKnowledgeGraph()

# Add temporal edges
now = datetime.now()
tkg.add_temporal_edge("Alice", "CompanyA", "works_at", now - timedelta(days=365), now - timedelta(days=30))
tkg.add_temporal_edge("Alice", "CompanyB", "works_at", now - timedelta(days=29))
tkg.add_temporal_edge("Bob", "ProjectX", "manages", now - timedelta(days=180))

# Query the graph at different times
past_graph = tkg.query_at_time(now - timedelta(days=60))
current_graph = tkg.query_at_time(now)

print("Alice's employer 60 days ago:", list(past_graph.successors("Alice")))
print("Alice's current employer:", list(current_graph.successors("Alice")))
```

Slide 14: Knowledge Graph Visualization Techniques

Effective visualization is crucial for understanding complex knowledge graphs. Let's explore some advanced visualization techniques using Python libraries.

```python
import networkx as nx
import plotly.graph_objects as go

# Create a sample knowledge graph
G = nx.karate_club_graph()

# Create a layout for the graph
pos = nx.spring_layout(G, dim=3)

# Create edge traces
edge_x = []
edge_y = []
edge_z = []
for edge in G.edges():
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_z.extend([z0, z1, None])

edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

# Create node traces
node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]
node_z = [pos[node][2] for node in G.nodes()]

node_trace = go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=10,
        color=[],
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

# Color node points by the number of connections
node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append(f'Node {node}<br># of connections: {len(adjacencies[1])}')

node_trace.marker.color = node_adjacencies
node_trace.text = node_text

# Create the figure
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Interactive 3D Knowledge Graph Visualization',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0,l=0,r=0,t=0),
                    scene=dict(
                        xaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
                        yaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
                        zaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
                    ),
                    annotations=[
                        dict(
                            showarrow=False,
                            text="Data source: Zachary's Karate Club network",
                            xref="paper",
                            yref="paper",
                            x=0,
                            y=0.1,
                            xanchor="left",
                            yanchor="bottom",
                            font=dict(size=14)
                        )
                    ]
                )
            )

fig.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into knowledge graphs and their applications, here are some valuable resources:

1. "Knowledge Graphs" by Aidan Hogan et al. (2020) - A comprehensive survey of knowledge graph research and applications. ArXiv: [https://arxiv.org/abs/2003.02320](https://arxiv.org/abs/2003.02320)
2. "Neural Graph Collaborative Filtering" by Xiang Wang et al. (2019) - Explores the application of graph neural networks to recommendation systems. ArXiv: [https://arxiv.org/abs/1905.08108](https://arxiv.org/abs/1905.08108)
3. "GraphVite: A High-Performance CPU-GPU Hybrid System for Node Embedding" by Zhaocheng Zhu et al. (2019) - Discusses efficient techniques for embedding large-scale graphs. ArXiv: [https://arxiv.org/abs/1903.00757](https://arxiv.org/abs/1903.00757)
4. "A Survey on Knowledge Graphs: Representation, Acquisition and Applications" by Shaoxiong Ji et al. (2020) - Provides an overview of knowledge graph techniques and their practical applications. ArXiv: [https://arxiv.org/abs/2002.00388](https://arxiv.org/abs/2002.00388)

These resources offer in-depth insights into various aspects of knowledge graphs, from theoretical foundations to practical implementations and cutting-edge research.

