## Enhancing RAG with Knowledge Graphs in Python
Slide 1: Introduction to RAG and Knowledge Graphs

Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval. Knowledge graphs enhance RAG by providing structured, interconnected information. This combination allows for more accurate and context-aware responses.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple knowledge graph
G = nx.Graph()
G.add_edges_from([('RAG', 'LLM'), ('RAG', 'Knowledge Retrieval'),
                  ('Knowledge Graph', 'Structured Data'),
                  ('Knowledge Graph', 'Relationships')])

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10)
plt.title("RAG and Knowledge Graph Concepts")
plt.axis('off')
plt.show()
```

Slide 2: Understanding Knowledge Graphs

Knowledge graphs represent information as a network of entities and their relationships. They provide a structured way to store and query complex data, making them ideal for enhancing RAG applications.

```python
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import FOAF, XSD

# Create a simple knowledge graph
g = Graph()

# Define entities
person = URIRef("http://example.org/person")
city = URIRef("http://example.org/city")

# Add triples (subject, predicate, object)
g.add((person, RDF.type, FOAF.Person))
g.add((person, FOAF.name, Literal("John Doe", datatype=XSD.string)))
g.add((person, FOAF.age, Literal(30, datatype=XSD.integer)))
g.add((person, URIRef("http://example.org/livesIn"), city))
g.add((city, RDF.type, URIRef("http://example.org/Location")))
g.add((city, FOAF.name, Literal("New York", datatype=XSD.string)))

# Query the graph
for s, p, o in g:
    print(f"{s} - {p} - {o}")
```

Slide 3: Building a Knowledge Graph

To create a knowledge graph, we'll use the rdflib library. This example demonstrates how to construct a simple graph with entities and relationships.

```python
from rdflib import Graph, Literal, Namespace, RDF, URIRef

# Create a new graph
g = Graph()

# Create a custom namespace
ex = Namespace("http://example.org/")

# Add triples to the graph
g.add((ex.Alice, RDF.type, ex.Person))
g.add((ex.Alice, ex.knows, ex.Bob))
g.add((ex.Bob, RDF.type, ex.Person))
g.add((ex.Bob, ex.age, Literal(30)))

# Serialize the graph to Turtle format
print(g.serialize(format="turtle").decode("utf-8"))
```

Slide 4: Querying Knowledge Graphs

SPARQL is the standard query language for RDF graphs. We'll use rdflib to execute SPARQL queries on our knowledge graph.

```python
from rdflib import Graph, Namespace
from rdflib.plugins.sparql import prepareQuery

# Assume we have the graph 'g' from the previous slide

# Define a SPARQL query
query = prepareQuery("""
    SELECT ?person ?age
    WHERE {
        ?person a ex:Person .
        ?person ex:age ?age .
    }
""", initNs={"ex": Namespace("http://example.org/")})

# Execute the query
for row in g.query(query):
    print(f"Person: {row.person}, Age: {row.age}")
```

Slide 5: Integrating Knowledge Graphs with RAG

To enhance RAG with knowledge graphs, we'll create a custom retriever that queries our graph based on the input query.

```python
from rdflib import Graph, Namespace
from rdflib.plugins.sparql import prepareQuery

class KGRetriever:
    def __init__(self, graph):
        self.graph = graph
        self.ex = Namespace("http://example.org/")

    def retrieve(self, query):
        sparql_query = prepareQuery(f"""
            SELECT ?s ?p ?o
            WHERE {{
                ?s ?p ?o .
                FILTER(regex(str(?s), "{query}", "i") || regex(str(?o), "{query}", "i"))
            }}
        """)
        
        results = []
        for row in self.graph.query(sparql_query):
            results.append(f"{row.s} - {row.p} - {row.o}")
        return results

# Usage
kg_retriever = KGRetriever(g)  # Assuming 'g' is our knowledge graph
results = kg_retriever.retrieve("Alice")
for result in results:
    print(result)
```

Slide 6: Combining RAG with Knowledge Graph Retrieval

In this slide, we'll demonstrate how to combine the knowledge graph retriever with a language model to create an enhanced RAG system.

```python
from transformers import pipeline

class EnhancedRAG:
    def __init__(self, kg_retriever, model_name="gpt2"):
        self.kg_retriever = kg_retriever
        self.generator = pipeline("text-generation", model=model_name)

    def generate_response(self, query):
        # Retrieve relevant information from the knowledge graph
        kg_results = self.kg_retriever.retrieve(query)
        
        # Combine the query with KG results
        context = f"Query: {query}\nKnowledge Graph Info: {' '.join(kg_results)}"
        
        # Generate a response using the language model
        response = self.generator(context, max_length=100, num_return_sequences=1)
        
        return response[0]['generated_text']

# Usage
enhanced_rag = EnhancedRAG(kg_retriever)
response = enhanced_rag.generate_response("Tell me about Alice")
print(response)
```

Slide 7: Handling Complex Queries

For more complex queries, we can use the knowledge graph to break down the question into sub-queries and aggregate the results.

```python
import networkx as nx

class ComplexQueryHandler:
    def __init__(self, graph):
        self.graph = graph
        self.nx_graph = nx.Graph()
        self._build_nx_graph()

    def _build_nx_graph(self):
        for s, p, o in self.graph:
            self.nx_graph.add_edge(s, o, relation=p)

    def handle_query(self, query):
        # Example: Find the shortest path between two entities
        start_entity = URIRef("http://example.org/Alice")
        end_entity = URIRef("http://example.org/Charlie")
        
        try:
            path = nx.shortest_path(self.nx_graph, start_entity, end_entity)
            return self._format_path(path)
        except nx.NetworkXNoPath:
            return "No connection found between the entities."

    def _format_path(self, path):
        result = []
        for i in range(len(path) - 1):
            s, o = path[i], path[i+1]
            edge_data = self.nx_graph.get_edge_data(s, o)
            result.append(f"{s} - {edge_data['relation']} -> {o}")
        return " | ".join(result)

# Usage
complex_handler = ComplexQueryHandler(g)  # Assuming 'g' is our knowledge graph
result = complex_handler.handle_query("How is Alice connected to Charlie?")
print(result)
```

Slide 8: Updating the Knowledge Graph

As new information becomes available, we need to update our knowledge graph. This slide demonstrates how to add, modify, and remove information from the graph.

```python
from rdflib import Graph, Literal, Namespace, RDF, URIRef

class KnowledgeGraphManager:
    def __init__(self):
        self.graph = Graph()
        self.ex = Namespace("http://example.org/")

    def add_entity(self, entity, entity_type):
        self.graph.add((URIRef(self.ex[entity]), RDF.type, URIRef(self.ex[entity_type])))

    def add_relationship(self, subject, predicate, object):
        self.graph.add((URIRef(self.ex[subject]), URIRef(self.ex[predicate]), URIRef(self.ex[object])))

    def update_attribute(self, entity, attribute, value):
        self.graph.set((URIRef(self.ex[entity]), URIRef(self.ex[attribute]), Literal(value)))

    def remove_entity(self, entity):
        self.graph.remove((URIRef(self.ex[entity]), None, None))

    def print_graph(self):
        print(self.graph.serialize(format="turtle").decode("utf-8"))

# Usage
kg_manager = KnowledgeGraphManager()
kg_manager.add_entity("Alice", "Person")
kg_manager.add_relationship("Alice", "knows", "Bob")
kg_manager.update_attribute("Alice", "age", 30)
kg_manager.print_graph()

kg_manager.remove_entity("Alice")
kg_manager.print_graph()
```

Slide 9: Real-life Example: Movie Recommendation System

Let's create a simple movie recommendation system using a knowledge graph and RAG.

```python
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.plugins.sparql import prepareQuery

class MovieRecommendationSystem:
    def __init__(self):
        self.graph = Graph()
        self.ex = Namespace("http://example.org/")
        self._populate_graph()

    def _populate_graph(self):
        # Add some sample data
        self.graph.add((self.ex.Inception, RDF.type, self.ex.Movie))
        self.graph.add((self.ex.Inception, self.ex.genre, self.ex.SciFi))
        self.graph.add((self.ex.Inception, self.ex.director, self.ex.ChristopherNolan))
        
        self.graph.add((self.ex.Interstellar, RDF.type, self.ex.Movie))
        self.graph.add((self.ex.Interstellar, self.ex.genre, self.ex.SciFi))
        self.graph.add((self.ex.Interstellar, self.ex.director, self.ex.ChristopherNolan))

        self.graph.add((self.ex.Titanic, RDF.type, self.ex.Movie))
        self.graph.add((self.ex.Titanic, self.ex.genre, self.ex.Romance))
        self.graph.add((self.ex.Titanic, self.ex.director, self.ex.JamesCameron))

    def recommend(self, movie):
        query = prepareQuery("""
            SELECT ?recommended
            WHERE {
                ?movie ex:genre ?genre .
                ?movie ex:director ?director .
                ?recommended ex:genre ?genre .
                ?recommended ex:director ?director .
                FILTER (?movie != ?recommended)
            }
        """, initNs={"ex": self.ex})

        results = self.graph.query(query, initBindings={'movie': self.ex[movie]})
        return [str(row.recommended).split('/')[-1] for row in results]

# Usage
recommender = MovieRecommendationSystem()
recommendations = recommender.recommend("Inception")
print(f"Recommendations for Inception: {recommendations}")
```

Slide 10: Real-life Example: Academic Paper Citation Network

Let's create a knowledge graph representing academic paper citations and use it to find influential papers.

```python
import networkx as nx
from rdflib import Graph, Literal, Namespace, RDF, URIRef

class AcademicCitationNetwork:
    def __init__(self):
        self.graph = Graph()
        self.ex = Namespace("http://example.org/")
        self.nx_graph = nx.DiGraph()
        self._populate_graph()

    def _populate_graph(self):
        # Add some sample data
        self.graph.add((self.ex.Paper1, RDF.type, self.ex.Paper))
        self.graph.add((self.ex.Paper1, self.ex.cites, self.ex.Paper2))
        self.graph.add((self.ex.Paper1, self.ex.cites, self.ex.Paper3))
        
        self.graph.add((self.ex.Paper2, RDF.type, self.ex.Paper))
        self.graph.add((self.ex.Paper2, self.ex.cites, self.ex.Paper3))
        
        self.graph.add((self.ex.Paper3, RDF.type, self.ex.Paper))
        self.graph.add((self.ex.Paper3, self.ex.cites, self.ex.Paper4))
        
        self.graph.add((self.ex.Paper4, RDF.type, self.ex.Paper))

        # Build NetworkX graph
        for s, p, o in self.graph:
            if p == self.ex.cites:
                self.nx_graph.add_edge(str(s).split('/')[-1], str(o).split('/')[-1])

    def find_influential_papers(self, top_n=2):
        pagerank = nx.pagerank(self.nx_graph)
        sorted_papers = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        return sorted_papers[:top_n]

# Usage
citation_network = AcademicCitationNetwork()
influential_papers = citation_network.find_influential_papers()
print("Most influential papers:")
for paper, score in influential_papers:
    print(f"{paper}: {score:.4f}")
```

Slide 11: Scaling Knowledge Graphs for Large-scale RAG Applications

As RAG applications grow, efficient storage and querying of large knowledge graphs become crucial. This slide introduces techniques for scaling knowledge graphs.

```python
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.plugins.stores import sparqlstore

class ScalableKnowledgeGraph:
    def __init__(self, endpoint_url):
        self.store = sparqlstore.SPARQLUpdateStore()
        self.store.open((endpoint_url, endpoint_url))
        self.graph = Graph(store=self.store, identifier="http://example.org/graph")
        self.ex = Namespace("http://example.org/")

    def add_triple(self, subject, predicate, object):
        self.graph.add((URIRef(self.ex[subject]), URIRef(self.ex[predicate]), URIRef(self.ex[object])))

    def query(self, sparql_query):
        return self.graph.query(sparql_query)

# Usage (Note: This requires a running SPARQL endpoint)
# kg = ScalableKnowledgeGraph("http://localhost:3030/dataset/sparql")
# kg.add_triple("Alice", "knows", "Bob")

# query = """
#     SELECT ?s ?p ?o
#     WHERE {
#         ?s ?p ?o .
#     }
# """
# results = kg.query(query)
# for row in results:
#     print(row)
```

Slide 12: Enhancing RAG with Semantic Reasoning

Semantic reasoning allows us to infer new knowledge from existing facts in our knowledge graph, enhancing the capabilities of our RAG system.

```python
from rdflib import Graph, Namespace, RDF, RDFS, OWL

class SemanticReasoningRAG:
    def __init__(self):
        self.graph = Graph()
        self.ex = Namespace("http://example.org/")
        self._setup_ontology()

    def _setup_ontology(self):
        # Define classes and relationships
        self.graph.add((self.ex.Animal, RDF.type, OWL.Class))
        self.graph.add((self.ex.Mammal, RDFS.subClassOf, self.ex.Animal))
        self.graph.add((self.ex.Dog, RDFS.subClassOf, self.ex.Mammal))

    def add_fact(self, subject, predicate, object):
        self.graph.add((self.ex[subject], self.ex[predicate], self.ex[object]))

    def infer_knowledge(self):
        # Apply RDFS reasoning
        self.graph.transitive_store(rules='rdfs')

    def query_knowledge(self, query):
        return self.graph.query(query)

# Usage
reasoner = SemanticReasoningRAG()
reasoner.add_fact("Buddy", "type", "Dog")
reasoner.infer_knowledge()

query = """
    SELECT ?class
    WHERE {
        ex:Buddy rdf:type/rdfs:subClassOf* ?class .
    }
"""
results = reasoner.query_knowledge(query)
for row in results:
    print(f"Buddy is a {row.class}")
```

Slide 13: Visualizing Knowledge Graphs for Better Understanding

Visualizing knowledge graphs can help users better understand the relationships between entities and improve the interpretability of RAG responses.

```python
import networkx as nx
import matplotlib.pyplot as plt
from rdflib import Graph, Namespace

class KnowledgeGraphVisualizer:
    def __init__(self, rdf_graph):
        self.rdf_graph = rdf_graph
        self.nx_graph = nx.Graph()
        self._convert_to_networkx()

    def _convert_to_networkx(self):
        for s, p, o in self.rdf_graph:
            self.nx_graph.add_edge(s, o, label=p)

    def visualize(self):
        pos = nx.spring_layout(self.nx_graph)
        plt.figure(figsize=(12, 8))
        nx.draw(self.nx_graph, pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=8)
        edge_labels = nx.get_edge_attributes(self.nx_graph, 'label')
        nx.draw_networkx_edge_labels(self.nx_graph, pos, edge_labels=edge_labels, font_size=6)
        plt.title("Knowledge Graph Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Usage
g = Graph()
ex = Namespace("http://example.org/")
g.add((ex.Alice, ex.knows, ex.Bob))
g.add((ex.Bob, ex.worksAt, ex.Company))
g.add((ex.Alice, ex.livesIn, ex.City))

visualizer = KnowledgeGraphVisualizer(g)
visualizer.visualize()
```

Slide 14: Evaluating RAG Performance with Knowledge Graphs

To assess the effectiveness of our knowledge graph-enhanced RAG system, we need to develop evaluation metrics that consider both the accuracy of retrieved information and the relevance of generated responses.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RAGEvaluator:
    def __init__(self, kg_retriever, text_encoder):
        self.kg_retriever = kg_retriever
        self.text_encoder = text_encoder

    def evaluate_retrieval(self, query, ground_truth):
        retrieved = self.kg_retriever.retrieve(query)
        retrieved_encoding = self.text_encoder.encode(retrieved)
        truth_encoding = self.text_encoder.encode(ground_truth)
        
        similarity = cosine_similarity(retrieved_encoding, truth_encoding)[0][0]
        return similarity

    def evaluate_generation(self, generated, reference):
        gen_encoding = self.text_encoder.encode(generated)
        ref_encoding = self.text_encoder.encode(reference)
        
        similarity = cosine_similarity(gen_encoding, ref_encoding)[0][0]
        return similarity

    def combined_score(self, retrieval_score, generation_score, alpha=0.5):
        return alpha * retrieval_score + (1 - alpha) * generation_score

# Usage (pseudo-code)
# evaluator = RAGEvaluator(kg_retriever, text_encoder)
# retrieval_score = evaluator.evaluate_retrieval(query, ground_truth)
# generation_score = evaluator.evaluate_generation(generated_response, reference_response)
# final_score = evaluator.combined_score(retrieval_score, generation_score)
# print(f"Final RAG score: {final_score}")
```

Slide 15: Additional Resources

For more information on enhancing RAG applications with knowledge graphs, consider exploring the following resources:

1. "Knowledge Graphs" by Hogan et al. (2020) - A comprehensive survey on knowledge graphs and their applications. ArXiv: [https://arxiv.org/abs/2003.02320](https://arxiv.org/abs/2003.02320)
2. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" by Lewis et al. (2020) - The original RAG paper introducing the concept. ArXiv: [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
3. "Knowledge Graphs and Their Applications to Natural Language Processing" by Wen et al. (2023) - An overview of knowledge graph applications in NLP. ArXiv: [https://arxiv.org/abs/2303.02449](https://arxiv.org/abs/2303.02449)

These resources provide in-depth information on knowledge graphs and their integration with language models, helping you further enhance your RAG applications.

