## Leveraging Graph-Based Knowledge for Informed Retrieval-Augmented Generation

Slide 1: Understanding Graph RAG Architecture

Graph RAG extends traditional retrieval-augmented generation by incorporating graph-based knowledge representations. This architecture enables contextual understanding through entity relationships, allowing for more nuanced and comprehensive information retrieval compared to simple vector similarity approaches.

```python
from typing import List, Dict
import networkx as nx

class GraphRAG:
    def __init__(self):
        self.knowledge_graph = nx.Graph()
        self.node_embeddings = {}
    
    def add_entity(self, entity: str, properties: Dict):
        # Add node to knowledge graph with properties
        self.knowledge_graph.add_node(entity, **properties)
    
    def add_relationship(self, entity1: str, entity2: str, relationship_type: str):
        # Add edge between entities with relationship type
        self.knowledge_graph.add_edge(entity1, entity2, type=relationship_type)

# Example usage
graph_rag = GraphRAG()
graph_rag.add_entity("Company_A", {"industry": "tech", "size": "large"})
graph_rag.add_entity("Person_B", {"role": "CEO", "experience": 15})
graph_rag.add_relationship("Person_B", "Company_A", "LEADS")
```

Slide 2: Neo4j Integration Setup

Neo4j serves as the backbone for storing and querying graph-structured data in Graph RAG systems. This implementation demonstrates the essential setup for connecting Neo4j with Python and establishing the foundation for graph-based retrieval operations.

```python
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

class Neo4jConnector:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def close(self):
        self.driver.close()
        
    def create_entity(self, tx, entity_type: str, properties: dict):
        query = (
            f"CREATE (n:{entity_type} $props) "
            "RETURN n"
        )
        result = tx.run(query, props=properties)
        return result.single()[0]

# Example configuration
load_dotenv()
connector = Neo4jConnector(
    uri="neo4j://localhost:7687",
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)
```

Slide 3: Document Processing and Entity Extraction

Converting raw documents into graph-structured data requires sophisticated entity extraction and relationship identification. This implementation uses spaCy for named entity recognition and custom rules for relationship extraction.

```python
import spacy
from typing import Tuple, List

class DocumentProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
        
        return entities
    
    def extract_relationships(self, text: str) -> List[Tuple[str, str, str]]:
        doc = self.nlp(text)
        relationships = []
        
        for token in doc:
            if token.dep_ == "ROOT":
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child.text
                    elif child.dep_ in ["dobj", "pobj"]:
                        obj = child.text
                        relationships.append((subject, token.text, obj))
                        
        return relationships

# Example usage
processor = DocumentProcessor()
text = "Microsoft acquired GitHub in 2018."
entities = processor.extract_entities(text)
relationships = processor.extract_relationships(text)
```

Slide 4: Embedding Generation for Graph Nodes

Entity embeddings are crucial for semantic search within the knowledge graph. This implementation uses sentence transformers to generate high-quality embeddings for graph nodes while maintaining relationship context.

```python
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

class NodeEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def generate_node_embedding(self, 
                              node_text: str, 
                              context: List[str] = None) -> np.ndarray:
        # Combine node text with context if available
        text_to_embed = node_text
        if context:
            text_to_embed = f"{node_text} {' '.join(context)}"
            
        # Generate embedding
        embedding = self.model.encode(text_to_embed,
                                    normalize_embeddings=True)
        return embedding
    
    def batch_generate_embeddings(self, 
                                nodes: List[str],
                                contexts: List[List[str]] = None) -> np.ndarray:
        if contexts:
            texts = [f"{node} {' '.join(ctx)}" 
                    for node, ctx in zip(nodes, contexts)]
        else:
            texts = nodes
            
        embeddings = self.model.encode(texts,
                                     normalize_embeddings=True,
                                     batch_size=32)
        return embeddings

# Example usage
embedder = NodeEmbedder()
node = "Microsoft Corporation"
context = ["tech company", "software development", "cloud computing"]
embedding = embedder.generate_node_embedding(node, context)
```

Slide 5: Graph-based Query Expansion

Query expansion enriches the initial user query by incorporating related concepts from the knowledge graph. This implementation leverages graph traversal algorithms and semantic similarity to identify and weight relevant terms for query enhancement.

```python
class GraphQueryExpander:
    def __init__(self, neo4j_connector):
        self.connector = neo4j_connector
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def expand_query(self, query: str, max_terms: int = 5):
        # Generate query embedding
        query_embedding = self.model.encode(query)
        
        # Find related nodes in graph using Cypher
        cypher_query = """
        MATCH (n)-[r]-(m)
        WHERE n.text CONTAINS $query
        RETURN m.text AS related_text
        LIMIT $max_terms
        """
        
        with self.connector.driver.session() as session:
            result = session.run(cypher_query, 
                               query=query,
                               max_terms=max_terms)
            related_terms = [record["related_text"] for record in result]
            
        # Compute similarity scores
        related_embeddings = self.model.encode(related_terms)
        scores = torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding),
            torch.tensor(related_embeddings)
        )
        
        # Return weighted expanded query terms
        expanded_terms = [(term, score.item()) 
                         for term, score in zip(related_terms, scores)]
        return sorted(expanded_terms, key=lambda x: x[1], reverse=True)

# Example usage
expander = GraphQueryExpander(neo4j_connector)
expanded = expander.expand_query("machine learning")
# Output: [('deep learning', 0.89), ('neural networks', 0.85)]
```

Slide 6: Knowledge Graph Construction

The knowledge graph construction process involves creating nodes and edges from processed documents. This implementation demonstrates how to build a structured graph representation while maintaining entity relationships and attributes.

```python
class KnowledgeGraphBuilder:
    def __init__(self, neo4j_connector):
        self.connector = neo4j_connector
        self.processor = DocumentProcessor()
        
    def build_graph(self, document: str):
        # Extract entities and relationships
        entities = self.processor.extract_entities(document)
        relationships = self.processor.extract_relationships(document)
        
        # Create nodes for entities
        with self.connector.driver.session() as session:
            for entity, entity_type in entities:
                session.execute_write(self._create_entity_node,
                                   entity, entity_type)
            
            # Create relationships
            for subj, pred, obj in relationships:
                session.execute_write(self._create_relationship,
                                   subj, pred, obj)
                
    def _create_entity_node(self, tx, entity: str, entity_type: str):
        query = """
        MERGE (n:Entity {name: $entity, type: $entity_type})
        RETURN n
        """
        return tx.run(query, entity=entity, entity_type=entity_type)
        
    def _create_relationship(self, tx, subj: str, pred: str, obj: str):
        query = """
        MATCH (s:Entity {name: $subj})
        MATCH (o:Entity {name: $obj})
        MERGE (s)-[r:RELATES {type: $pred}]->(o)
        RETURN r
        """
        return tx.run(query, subj=subj, pred=pred, obj=obj)

# Example usage
builder = KnowledgeGraphBuilder(neo4j_connector)
document = "OpenAI develops GPT-4 for natural language processing."
builder.build_graph(document)
```

Slide 7: Semantic Graph Search

Semantic graph search combines traditional graph traversal with embedding-based similarity search. This implementation enables finding relevant information through both structural and semantic relationships in the knowledge graph.

```python
class SemanticGraphSearch:
    def __init__(self, neo4j_connector):
        self.connector = neo4j_connector
        self.embedder = NodeEmbedder()
        
    def search(self, query: str, max_results: int = 5):
        query_embedding = self.embedder.generate_node_embedding(query)
        
        # Convert embedding to string for Neo4j
        query_vector = ','.join(map(str, query_embedding))
        
        search_query = """
        CALL db.index.vector.queryNodes('node_embeddings', 
                                      $max_results, 
                                      $query_vector)
        YIELD node, score
        MATCH (node)-[r]-(related)
        RETURN node.name, related.name, r.type, score
        """
        
        with self.connector.driver.session() as session:
            results = session.run(search_query,
                                max_results=max_results,
                                query_vector=query_vector)
            
            return [(record["node.name"],
                    record["related.name"],
                    record["r.type"],
                    record["score"]) for record in results]

# Example usage
searcher = SemanticGraphSearch(neo4j_connector)
results = searcher.search("machine learning applications")
```

Slide 8: Graph Context Aggregation

Graph context aggregation combines information from multiple related nodes to provide comprehensive context for the LLM. This implementation demonstrates how to gather and structure contextual information from the knowledge graph.

```python
class GraphContextAggregator:
    def __init__(self, neo4j_connector):
        self.connector = neo4j_connector
        
    def aggregate_context(self, 
                         central_node: str, 
                         max_depth: int = 2,
                         max_nodes: int = 10):
        context_query = """
        MATCH path = (start:Entity {name: $node})-[*1..$depth]-(related)
        WITH related, min(length(path)) as distance
        ORDER BY distance
        LIMIT $max_nodes
        RETURN collect({
            name: related.name,
            type: related.type,
            distance: distance
        }) as context
        """
        
        with self.connector.driver.session() as session:
            result = session.run(context_query,
                               node=central_node,
                               depth=max_depth,
                               max_nodes=max_nodes)
            
            context_data = result.single()["context"]
            
            # Structure context by distance
            context_layers = {}
            for node in context_data:
                distance = node["distance"]
                if distance not in context_layers:
                    context_layers[distance] = []
                context_layers[distance].append({
                    "name": node["name"],
                    "type": node["type"]
                })
                
            return context_layers

# Example usage
aggregator = GraphContextAggregator(neo4j_connector)
context = aggregator.aggregate_context("GPT-4")
```

Slide 9: Query Result Ranking

Query result ranking determines the most relevant information from the graph based on both structural and semantic similarities. This implementation shows how to score and rank retrieved results for optimal relevance.

```python
class QueryResultRanker:
    def __init__(self):
        self.embedder = NodeEmbedder()
        
    def rank_results(self, 
                    query: str,
                    results: List[Dict],
                    alpha: float = 0.5):
        query_embedding = self.embedder.generate_node_embedding(query)
        
        ranked_results = []
        for result in results:
            # Calculate semantic similarity
            result_text = (f"{result['name']} "
                         f"{result.get('description', '')}")
            result_embedding = self.embedder.generate_node_embedding(
                result_text)
            
            semantic_score = float(torch.nn.functional.cosine_similarity(
                torch.tensor(query_embedding),
                torch.tensor(result_embedding),
                dim=0
            ))
            
            # Calculate structural score
            structural_score = 1.0 / (1.0 + result['distance'])
            
            # Combined score
            final_score = (alpha * semantic_score + 
                         (1 - alpha) * structural_score)
            
            ranked_results.append({
                **result,
                'score': final_score
            })
            
        return sorted(ranked_results,
                     key=lambda x: x['score'],
                     reverse=True)

# Example usage
ranker = QueryResultRanker()
ranked_results = ranker.rank_results(
    "machine learning frameworks",
    [{'name': 'TensorFlow', 'distance': 1},
     {'name': 'PyTorch', 'distance': 1}]
)
```

Slide 10: Graph-Augmented Response Generation

This implementation shows how to combine retrieved graph context with an LLM to generate comprehensive, contextually-aware responses. The system uses graph structure to enhance response accuracy and completeness.

```python
class GraphAugmentedGenerator:
    def __init__(self, neo4j_connector, openai_key: str):
        self.connector = neo4j_connector
        self.aggregator = GraphContextAggregator(neo4j_connector)
        self.openai_key = openai_key
        
    async def generate_response(self, 
                              query: str,
                              context_node: str):
        # Gather graph context
        context = self.aggregator.aggregate_context(context_node)
        
        # Format context for LLM
        formatted_context = self._format_context(context)
        
        # Create LLM prompt
        prompt = f"""Context: {formatted_context}
        Question: {query}
        Please provide a comprehensive answer using the context above."""
        
        # Generate response
        completion = await openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", 
                 "content": "You are a knowledgeable assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return completion.choices[0].message.content
        
    def _format_context(self, context_layers: Dict):
        formatted = []
        for distance, nodes in context_layers.items():
            layer_info = [f"{node['name']} ({node['type']})"
                         for node in nodes]
            formatted.append(f"Layer {distance}: {', '.join(layer_info)}")
        return "\n".join(formatted)

# Example usage
generator = GraphAugmentedGenerator(neo4j_connector, "your-openai-key")
response = await generator.generate_response(
    "What are the applications of transformers?",
    "transformer_architecture"
)
```

Slide 11: Performance Metrics and Evaluation

This implementation provides comprehensive metrics for evaluating Graph RAG performance, including relevance scores, response accuracy, and retrieval effectiveness.

```python
class GraphRAGEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def evaluate_retrieval(self,
                          queries: List[str],
                          expected: List[Set[str]],
                          retrieved: List[Set[str]]):
        results = {
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for query_exp, query_ret in zip(expected, retrieved):
            tp = len(query_exp.intersection(query_ret))
            fp = len(query_ret - query_exp)
            fn = len(query_exp - query_ret)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0)
            
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1'].append(f1)
            
        self.metrics['retrieval'] = {
            'avg_precision': np.mean(results['precision']),
            'avg_recall': np.mean(results['recall']),
            'avg_f1': np.mean(results['f1'])
        }
        
        return self.metrics['retrieval']

# Example usage
evaluator = GraphRAGEvaluator()
metrics = evaluator.evaluate_retrieval(
    queries=["What is machine learning?"],
    expected=[{"ML", "AI", "Neural Networks"}],
    retrieved=[{"ML", "AI", "Deep Learning"}]
)
```

Slide 12: Real-world Example: Research Paper Analysis

This implementation demonstrates Graph RAG application for analyzing research paper citations and relationships, showing practical usage in academic contexts.

```python
class ResearchGraphRAG:
    def __init__(self, neo4j_connector):
        self.connector = neo4j_connector
        self.processor = DocumentProcessor()
        
    def process_paper(self, title: str, abstract: str, citations: List[str]):
        # Create paper node
        with self.connector.driver.session() as session:
            session.execute_write(self._create_paper_node,
                                title, abstract)
            
            # Process citations
            for citation in citations:
                session.execute_write(self._add_citation,
                                   title, citation)
            
            # Extract and add research concepts
            concepts = self.processor.extract_entities(abstract)
            for concept, _ in concepts:
                session.execute_write(self._add_concept,
                                   title, concept)
    
    def _create_paper_node(self, tx, title: str, abstract: str):
        query = """
        MERGE (p:Paper {title: $title, abstract: $abstract})
        RETURN p
        """
        return tx.run(query, title=title, abstract=abstract)
    
    def _add_citation(self, tx, source: str, target: str):
        query = """
        MATCH (s:Paper {title: $source})
        MERGE (t:Paper {title: $target})
        MERGE (s)-[r:CITES]->(t)
        RETURN r
        """
        return tx.run(query, source=source, target=target)
    
    def _add_concept(self, tx, paper: str, concept: str):
        query = """
        MATCH (p:Paper {title: $paper})
        MERGE (c:Concept {name: $concept})
        MERGE (p)-[r:DISCUSSES]->(c)
        RETURN r
        """
        return tx.run(query, paper=paper, concept=concept)

# Example usage
research_rag = ResearchGraphRAG(neo4j_connector)
research_rag.process_paper(
    "Graph Neural Networks: A Review",
    "This paper reviews recent advances in GNN architectures...",
    ["Attention is All You Need", "Graph Attention Networks"]
)
```

Slide 13: Additional Resources

1.  [https://arxiv.org/abs/2307.01128](https://arxiv.org/abs/2307.01128) - "Graph-based Retrieval Augmented Generation: A Survey"
2.  [https://arxiv.org/abs/2308.07107](https://arxiv.org/abs/2308.07107) - "Knowledge Graphs Meet Multi-Modal Learning: A Comprehensive Survey"
3.  [https://arxiv.org/abs/2306.12672](https://arxiv.org/abs/2306.12672) - "Retrieval-Augmented Generation for Large Language Models: A Survey"
4.  [https://arxiv.org/abs/2305.11426](https://arxiv.org/abs/2305.11426) - "A Survey on Graph Neural Networks and Graph Transformers"
5.  [https://arxiv.org/abs/2309.05663](https://arxiv.org/abs/2309.05663) - "Graph Neural Networks for Natural Language Processing: A Survey"

