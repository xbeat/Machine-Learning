## Data Mesh vs. Data Fabric Comparing Modern Data Architectures
Slide 1: Understanding Data Mesh Architecture

Data Mesh represents a paradigm shift in data architecture, emphasizing domain-driven ownership and data as a product. This implementation demonstrates how to create a basic domain-driven data pipeline using Python's dataclasses for domain modeling and type safety.

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class DataProduct:
    domain: str
    name: str
    owner: str
    schema: dict
    quality_metrics: dict
    created_at: datetime
    
class DomainDataPipeline:
    def __init__(self, domain_name: str):
        self.domain = domain_name
        self.data_products: List[DataProduct] = []
    
    def create_data_product(self, name: str, owner: str, schema: dict) -> DataProduct:
        product = DataProduct(
            domain=self.domain,
            name=name,
            owner=owner,
            schema=schema,
            quality_metrics={'completeness': 0, 'accuracy': 0},
            created_at=datetime.now()
        )
        self.data_products.append(product)
        return product

# Example Usage
sales_domain = DomainDataPipeline('sales')
product = sales_domain.create_data_product(
    'daily_transactions',
    'sales_team',
    {'date': 'datetime', 'amount': 'float', 'product_id': 'string'}
)
```

Slide 2: Data Product Validation System

A crucial aspect of Data Mesh is ensuring data quality at the source. This implementation shows how to create a validation system for data products using Python's abstract base classes and custom validators.

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class DataValidator(ABC):
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        pass

class QualityMetricsValidator(DataValidator):
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        metrics = {
            'completeness': 1 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1]),
            'uniqueness': len(data.drop_duplicates()) / len(data),
            'freshness': (pd.Timestamp.now() - data['timestamp'].max()).days
        }
        return metrics

class SchemaValidator(DataValidator):
    def __init__(self, expected_schema: Dict):
        self.expected_schema = expected_schema
    
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        actual_dtypes = data.dtypes.to_dict()
        schema_valid = all(
            str(actual_dtypes[col]).startswith(expected_type)
            for col, expected_type in self.expected_schema.items()
        )
        return {'schema_valid': schema_valid}
```

Slide 3: Data Fabric Integration Layer

Data Fabric requires a sophisticated integration layer that can handle various data sources seamlessly. This implementation showcases a flexible connector system using the adapter pattern.

```python
from typing import Protocol
import pandas as pd
from dataclasses import dataclass

class DataSource(Protocol):
    def connect(self) -> None:
        pass
    
    def read(self) -> pd.DataFrame:
        pass
    
    def write(self, data: pd.DataFrame) -> None:
        pass

@dataclass
class ConnectionConfig:
    host: str
    port: int
    credentials: dict

class PostgresAdapter:
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connection = None
    
    def connect(self) -> None:
        # Simulated connection
        self.connection = f"Connected to {self.config.host}:{self.config.port}"
    
    def read(self, query: str) -> pd.DataFrame:
        return pd.DataFrame()  # Simplified for example
    
    def write(self, data: pd.DataFrame, table: str) -> None:
        pass  # Simplified for example

class DataFabricIntegrator:
    def __init__(self):
        self.sources: Dict[str, DataSource] = {}
    
    def register_source(self, name: str, source: DataSource):
        self.sources[name] = source
        
    def query_across_sources(self, query_params: Dict) -> pd.DataFrame:
        results = []
        for source in self.sources.values():
            source.connect()
            results.append(source.read())
        return pd.concat(results)
```

Slide 4: Data Mesh Federation Service

A federation service is essential for managing distributed data products while maintaining consistency and accessibility. This implementation demonstrates a federated query engine that aggregates data across domain boundaries.

```python
from typing import List, Dict, Any
import asyncio
import pandas as pd

class FederationService:
    def __init__(self):
        self.domain_registries: Dict[str, DomainRegistry] = {}
        
    async def federated_query(self, query_spec: Dict[str, Any]) -> pd.DataFrame:
        tasks = []
        for domain, registry in self.domain_registries.items():
            if domain in query_spec['domains']:
                tasks.append(self.query_domain(registry, query_spec))
        
        results = await asyncio.gather(*tasks)
        return pd.concat(results, ignore_index=True)
    
    async def query_domain(self, registry: 'DomainRegistry', query_spec: Dict[str, Any]) -> pd.DataFrame:
        products = registry.get_data_products(query_spec['filters'])
        return await registry.execute_query(products, query_spec['query'])

class DomainRegistry:
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.data_products: Dict[str, DataProduct] = {}
        
    async def execute_query(self, products: List[DataProduct], query: str) -> pd.DataFrame:
        # Simulated query execution
        return pd.DataFrame({'domain': self.domain_name, 'data': range(5)})
```

Slide 5: Data Fabric Metadata Management

An intelligent metadata management system forms the backbone of Data Fabric, enabling automated data discovery and lineage tracking. This implementation shows a graph-based metadata store with semantic capabilities.

```python
from typing import Optional, Set
import networkx as nx
from datetime import datetime

class MetadataNode:
    def __init__(self, node_type: str, properties: Dict[str, Any]):
        self.node_type = node_type
        self.properties = properties
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

class MetadataGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.semantic_index = {}
        
    def add_node(self, node_id: str, node: MetadataNode):
        self.graph.add_node(node_id, **node.__dict__)
        self._index_node(node_id, node)
    
    def add_relationship(self, from_id: str, to_id: str, relationship_type: str):
        self.graph.add_edge(from_id, to_id, type=relationship_type)
    
    def _index_node(self, node_id: str, node: MetadataNode):
        for key, value in node.properties.items():
            if key not in self.semantic_index:
                self.semantic_index[key] = {}
            if value not in self.semantic_index[key]:
                self.semantic_index[key][value] = set()
            self.semantic_index[key][value].add(node_id)
    
    def search(self, criteria: Dict[str, Any]) -> Set[str]:
        results = None
        for key, value in criteria.items():
            nodes = self.semantic_index.get(key, {}).get(value, set())
            if results is None:
                results = nodes
            else:
                results &= nodes
        return results or set()
```

Slide 6: Real-Time Data Quality Monitoring

Essential for both Data Mesh and Data Fabric architectures, this implementation provides real-time monitoring of data quality metrics with statistical analysis and anomaly detection.

```python
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from scipy import stats

@dataclass
class QualityMetric:
    name: str
    value: float
    timestamp: datetime
    threshold: float
    z_score: Optional[float] = None

class DataQualityMonitor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: Dict[str, List[QualityMetric]] = {}
        
    def add_metric(self, metric: QualityMetric):
        if metric.name not in self.metrics_history:
            self.metrics_history[metric.name] = []
        
        history = self.metrics_history[metric.name]
        history.append(metric)
        
        if len(history) > self.window_size:
            history.pop(0)
        
        self._update_statistics(metric.name)
    
    def _update_statistics(self, metric_name: str):
        history = self.metrics_history[metric_name]
        values = [m.value for m in history]
        
        if len(values) > 1:
            z_score = stats.zscore(values)[-1]
            history[-1].z_score = z_score
            
            if abs(z_score) > 3:
                self._trigger_anomaly_alert(metric_name, history[-1])
    
    def _trigger_anomaly_alert(self, metric_name: str, metric: QualityMetric):
        alert = {
            'metric_name': metric_name,
            'value': metric.value,
            'z_score': metric.z_score,
            'timestamp': metric.timestamp,
            'severity': 'high' if abs(metric.z_score) > 4 else 'medium'
        }
        # In real implementation, send to alert system
        print(f"Quality Alert: {alert}")
```

Slide 7: Data Product Access Layer

This implementation provides a secure and scalable access layer for data products, incorporating role-based access control and usage tracking. The system ensures proper governance while maintaining domain autonomy.

```python
from typing import Set, Dict, Optional
import jwt
from datetime import datetime, timedelta

class AccessControl:
    def __init__(self):
        self.role_permissions: Dict[str, Set[str]] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self.access_logs: List[Dict] = []
        
    def grant_permission(self, role: str, permission: str):
        if role not in self.role_permissions:
            self.role_permissions[role] = set()
        self.role_permissions[role].add(permission)
    
    def assign_role(self, user_id: str, role: str):
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        self.user_roles[user_id].add(role)
    
    def check_access(self, user_id: str, required_permission: str) -> bool:
        user_roles = self.user_roles.get(user_id, set())
        for role in user_roles:
            if required_permission in self.role_permissions.get(role, set()):
                self._log_access(user_id, required_permission, True)
                return True
        self._log_access(user_id, required_permission, False)
        return False
    
    def _log_access(self, user_id: str, permission: str, granted: bool):
        log_entry = {
            'timestamp': datetime.now(),
            'user_id': user_id,
            'permission': permission,
            'granted': granted
        }
        self.access_logs.append(log_entry)
```

Slide 8: Data Fabric Discovery Service

A sophisticated service that enables automatic discovery and cataloging of data assets across the enterprise, using machine learning for metadata enrichment and classification.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np

class DataDiscoveryService:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.clustering = DBSCAN(eps=0.3, min_samples=2)
        self.metadata_cache = {}
        
    def discover_datasets(self, source_connection: Dict) -> List[Dict]:
        raw_metadata = self._scan_source(source_connection)
        enriched_metadata = self._enrich_metadata(raw_metadata)
        classified_datasets = self._classify_datasets(enriched_metadata)
        return classified_datasets
    
    def _scan_source(self, connection: Dict) -> List[Dict]:
        # Simplified implementation
        return [{'name': f'dataset_{i}', 'columns': [f'col_{j}' for j in range(5)]} 
                for i in range(10)]
    
    def _enrich_metadata(self, metadata: List[Dict]) -> List[Dict]:
        for dataset in metadata:
            dataset['column_profiles'] = self._profile_columns(dataset['columns'])
            dataset['semantic_type'] = self._infer_semantic_type(dataset)
        return metadata
    
    def _classify_datasets(self, metadata: List[Dict]) -> List[Dict]:
        descriptions = [self._generate_description(m) for m in metadata]
        vectors = self.vectorizer.fit_transform(descriptions)
        clusters = self.clustering.fit_predict(vectors.toarray())
        
        for i, dataset in enumerate(metadata):
            dataset['cluster'] = int(clusters[i])
        
        return metadata
    
    def _generate_description(self, metadata: Dict) -> str:
        return f"{metadata['name']} {' '.join(metadata['columns'])}"
```

Slide 9: Data Mesh Schema Evolution

This implementation handles schema evolution in a distributed data mesh environment, ensuring backward compatibility and versioning of data products.

```python
from typing import Dict, List, Optional
import jsonschema
from datetime import datetime

class SchemaVersion:
    def __init__(self, schema: Dict, version: str, created_at: datetime):
        self.schema = schema
        self.version = version
        self.created_at = created_at
        self.deprecated = False
        self.successor: Optional['SchemaVersion'] = None

class SchemaRegistry:
    def __init__(self):
        self.schemas: Dict[str, List[SchemaVersion]] = {}
        
    def register_schema(self, data_product_id: str, schema: Dict, version: str):
        if data_product_id not in self.schemas:
            self.schemas[data_product_id] = []
            
        new_version = SchemaVersion(schema, version, datetime.now())
        
        if self.schemas[data_product_id]:
            current_version = self.schemas[data_product_id][-1]
            if not self._is_compatible(current_version.schema, schema):
                raise ValueError("Schema evolution breaks compatibility")
            current_version.successor = new_version
            
        self.schemas[data_product_id].append(new_version)
        
    def _is_compatible(self, old_schema: Dict, new_schema: Dict) -> bool:
        # Check if new schema is backward compatible
        try:
            # Validate old data against new schema
            validator = jsonschema.Draft7Validator(new_schema)
            # Simulate old data format
            old_data = self._generate_sample_data(old_schema)
            validator.validate(old_data)
            return True
        except jsonschema.exceptions.ValidationError:
            return False
            
    def _generate_sample_data(self, schema: Dict) -> Dict:
        # Simplified sample data generation based on schema
        sample_data = {}
        for field, type_info in schema['properties'].items():
            if type_info['type'] == 'string':
                sample_data[field] = "sample"
            elif type_info['type'] == 'number':
                sample_data[field] = 0
        return sample_data
```

Slide 10: Data Fabric Lineage Tracking

This implementation provides comprehensive data lineage tracking across the enterprise, capturing data transformations and dependencies using a directed acyclic graph (DAG) structure.

```python
from typing import Optional, Set, List
import networkx as nx
from datetime import datetime
import json

class LineageNode:
    def __init__(self, node_id: str, node_type: str, metadata: Dict):
        self.node_id = node_id
        self.node_type = node_type
        self.metadata = metadata
        self.created_at = datetime.now()

class LineageGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.transformation_history = {}
        
    def add_node(self, node: LineageNode):
        self.graph.add_node(
            node.node_id,
            node_type=node.node_type,
            metadata=node.metadata,
            created_at=node.created_at
        )
        
    def add_transformation(self, source_ids: List[str], 
                          target_id: str, 
                          transform_type: str,
                          transform_metadata: Dict):
        transform_id = f"transform_{len(self.transformation_history)}"
        
        # Record transformation
        self.transformation_history[transform_id] = {
            'type': transform_type,
            'metadata': transform_metadata,
            'timestamp': datetime.now(),
            'sources': source_ids,
            'target': target_id
        }
        
        # Add edges to graph
        for source_id in source_ids:
            self.graph.add_edge(source_id, target_id, 
                              transform_id=transform_id)
    
    def get_upstream_lineage(self, node_id: str, depth: Optional[int] = None) -> nx.DiGraph:
        if depth is None:
            predecessors = nx.ancestors(self.graph, node_id)
        else:
            predecessors = set()
            current_nodes = {node_id}
            for _ in range(depth):
                new_nodes = set()
                for node in current_nodes:
                    new_nodes.update(self.graph.predecessors(node))
                predecessors.update(new_nodes)
                current_nodes = new_nodes
                
        return self.graph.subgraph(predecessors | {node_id})
    
    def export_lineage(self, node_id: str) -> Dict:
        subgraph = self.get_upstream_lineage(node_id)
        return {
            'nodes': [
                {
                    'id': n,
                    'type': self.graph.nodes[n]['node_type'],
                    'metadata': self.graph.nodes[n]['metadata']
                }
                for n in subgraph.nodes
            ],
            'transformations': [
                {
                    'id': self.graph.edges[e]['transform_id'],
                    **self.transformation_history[
                        self.graph.edges[e]['transform_id']
                    ]
                }
                for e in subgraph.edges
            ]
        }
```

Slide 11: Source Code for Data Mesh Domain Event System

A robust event system that enables communication and synchronization between different domains in a data mesh, implementing the event sourcing pattern.

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import asyncio

@dataclass
class DomainEvent:
    event_type: str
    domain: str
    data: Dict[str, Any]
    timestamp: datetime
    correlation_id: str
    causation_id: Optional[str] = None

class EventStore:
    def __init__(self):
        self.events: List[DomainEvent] = []
        self.subscribers: Dict[str, List[callable]] = {}
        
    async def publish_event(self, event: DomainEvent):
        self.events.append(event)
        await self._notify_subscribers(event)
    
    def subscribe(self, event_type: str, callback: callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    async def _notify_subscribers(self, event: DomainEvent):
        if event.event_type in self.subscribers:
            tasks = []
            for callback in self.subscribers[event.event_type]:
                tasks.append(asyncio.create_task(callback(event)))
            await asyncio.gather(*tasks)
    
    def get_events_by_correlation(self, correlation_id: str) -> List[DomainEvent]:
        return [e for e in self.events if e.correlation_id == correlation_id]

class DomainEventHandler:
    def __init__(self, domain: str, event_store: EventStore):
        self.domain = domain
        self.event_store = event_store
        
    async def handle_schema_change(self, event: DomainEvent):
        # Handle schema change events
        print(f"Domain {self.domain} handling schema change: {event.data}")
        
    async def handle_data_quality_alert(self, event: DomainEvent):
        # Handle data quality alerts
        print(f"Domain {self.domain} handling quality alert: {event.data}")

# Example usage
event_store = EventStore()
sales_handler = DomainEventHandler('sales', event_store)
event_store.subscribe('schema_change', sales_handler.handle_schema_change)
event_store.subscribe('quality_alert', sales_handler.handle_data_quality_alert)
```

Slide 12: Data Fabric Query Optimizer

This implementation provides an intelligent query optimizer that analyzes and optimizes distributed queries across the data fabric, considering data locality and network costs.

```python
from typing import List, Dict, Set, Tuple
import networkx as nx
from dataclasses import dataclass

@dataclass
class QueryNode:
    node_id: str
    operation: str
    cost: float
    data_size: float
    location: str

class QueryOptimizer:
    def __init__(self):
        self.network_costs = {}
        self.location_capacities = {}
        self.execution_graph = nx.DiGraph()
        
    def optimize_query(self, query_plan: List[QueryNode]) -> List[Tuple[QueryNode, str]]:
        self._build_execution_graph(query_plan)
        optimized_locations = self._optimize_node_placement()
        return self._generate_execution_plan(optimized_locations)
    
    def _build_execution_graph(self, nodes: List[QueryNode]):
        for node in nodes:
            self.execution_graph.add_node(
                node.node_id,
                operation=node.operation,
                cost=node.cost,
                data_size=node.data_size
            )
            
            # Add dependencies based on operation type
            if node.operation == 'join':
                # Add edges from input nodes to join node
                for dependency in self._get_dependencies(node):
                    self.execution_graph.add_edge(dependency, node.node_id)
    
    def _optimize_node_placement(self) -> Dict[str, str]:
        placements = {}
        sorted_nodes = list(nx.topological_sort(self.execution_graph))
        
        for node_id in sorted_nodes:
            best_location = self._find_best_location(node_id)
            placements[node_id] = best_location
            
        return placements
    
    def _find_best_location(self, node_id: str) -> str:
        min_cost = float('inf')
        best_location = None
        node_data = self.execution_graph.nodes[node_id]
        
        for location in self.location_capacities:
            cost = self._calculate_placement_cost(node_id, location)
            if cost < min_cost and self._check_capacity(location, node_data['data_size']):
                min_cost = cost
                best_location = location
                
        return best_location
    
    def _calculate_placement_cost(self, node_id: str, location: str) -> float:
        node_data = self.execution_graph.nodes[node_id]
        computation_cost = node_data['cost']
        
        # Calculate data transfer costs
        transfer_cost = 0
        for pred in self.execution_graph.predecessors(node_id):
            pred_location = self._get_node_location(pred)
            if pred_location != location:
                transfer_cost += (self.network_costs.get((pred_location, location), 1.0) 
                                * self.execution_graph.nodes[pred]['data_size'])
                
        return computation_cost + transfer_cost
    
    def _generate_execution_plan(self, placements: Dict[str, str]) -> List[Tuple[QueryNode, str]]:
        execution_plan = []
        sorted_nodes = list(nx.topological_sort(self.execution_graph))
        
        for node_id in sorted_nodes:
            node_data = self.execution_graph.nodes[node_id]
            node = QueryNode(
                node_id=node_id,
                operation=node_data['operation'],
                cost=node_data['cost'],
                data_size=node_data['data_size'],
                location=placements[node_id]
            )
            execution_plan.append((node, placements[node_id]))
            
        return execution_plan
```

Slide 13: Automated Data Contract Generation

This implementation provides an automated system for generating and managing data contracts between data producers and consumers in a Data Mesh architecture.

```python
from typing import List, Dict, Optional
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DataContract:
    producer_domain: str
    consumer_domain: str
    schema: Dict
    quality_sla: Dict
    validity_period: Tuple[datetime, datetime]
    version: str
    
class ContractGenerator:
    def __init__(self):
        self.contract_templates = {}
        self.active_contracts = {}
        self.contract_violations = []
        
    def generate_contract(self, 
                         producer: str, 
                         consumer: str, 
                         data_product: Dict) -> DataContract:
        schema = self._extract_schema(data_product)
        sla = self._generate_sla(producer, consumer, data_product)
        
        contract = DataContract(
            producer_domain=producer,
            consumer_domain=consumer,
            schema=schema,
            quality_sla=sla,
            validity_period=(datetime.now(), datetime.now().replace(year=datetime.now().year + 1)),
            version="1.0"
        )
        
        self.active_contracts[(producer, consumer)] = contract
        return contract
    
    def validate_compliance(self, contract: DataContract, metrics: Dict) -> List[Dict]:
        violations = []
        for metric, threshold in contract.quality_sla.items():
            if metric in metrics:
                if not self._check_threshold(metrics[metric], threshold):
                    violations.append({
                        'metric': metric,
                        'actual': metrics[metric],
                        'threshold': threshold,
                        'timestamp': datetime.now()
                    })
        
        if violations:
            self.contract_violations.extend(violations)
        
        return violations
    
    def _extract_schema(self, data_product: Dict) -> Dict:
        return {
            'fields': data_product.get('schema', {}),
            'constraints': data_product.get('constraints', {}),
            'formats': data_product.get('formats', {})
        }
    
    def _generate_sla(self, producer: str, consumer: str, data_product: Dict) -> Dict:
        base_sla = {
            'availability': 0.99,
            'latency': 1000,  # ms
            'completeness': 0.95,
            'accuracy': 0.99
        }
        
        # Customize SLA based on consumer requirements
        if 'requirements' in data_product:
            for metric, value in data_product['requirements'].items():
                if metric in base_sla:
                    base_sla[metric] = max(base_sla[metric], value)
                    
        return base_sla
```

Slide 14: Data Mesh Observability Platform

This implementation creates a comprehensive observability platform for monitoring and analyzing the health of data products across domains, incorporating metrics, traces, and logs.

```python
from typing import Dict, List, Optional
import time
from dataclasses import dataclass
from datetime import datetime
import numpy as np

@dataclass
class MetricPoint:
    metric_name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    domain: str

class ObservabilityPlatform:
    def __init__(self):
        self.metrics_store = {}
        self.alerts = []
        self.trace_store = {}
        
    def record_metric(self, metric: MetricPoint):
        key = (metric.domain, metric.metric_name)
        if key not in self.metrics_store:
            self.metrics_store[key] = []
        self.metrics_store[key].append(metric)
        self._check_alerts(metric)
        
    def create_trace(self, trace_id: str, parent_id: Optional[str] = None):
        trace = {
            'trace_id': trace_id,
            'parent_id': parent_id,
            'start_time': datetime.now(),
            'end_time': None,
            'events': []
        }
        self.trace_store[trace_id] = trace
        return trace_id
        
    def add_trace_event(self, trace_id: str, event: Dict):
        if trace_id in self.trace_store:
            self.trace_store[trace_id]['events'].append({
                **event,
                'timestamp': datetime.now()
            })
            
    def compute_slo_compliance(self, domain: str, 
                             metric_name: str, 
                             threshold: float,
                             window_minutes: int = 60) -> float:
        key = (domain, metric_name)
        if key not in self.metrics_store:
            return 0.0
            
        cutoff = datetime.now().timestamp() - (window_minutes * 60)
        recent_metrics = [
            m for m in self.metrics_store[key]
            if m.timestamp.timestamp() > cutoff
        ]
        
        if not recent_metrics:
            return 0.0
            
        compliant = sum(1 for m in recent_metrics if m.value >= threshold)
        return compliant / len(recent_metrics)
    
    def _check_alerts(self, metric: MetricPoint):
        # Example alert rules
        if metric.metric_name == 'data_freshness_minutes':
            if metric.value > 120:  # 2 hours
                self._create_alert(
                    severity='high',
                    message=f'Data freshness exceeds 2 hours in {metric.domain}',
                    metric=metric
                )
        elif metric.metric_name == 'quality_score':
            if metric.value < 0.95:
                self._create_alert(
                    severity='medium',
                    message=f'Quality score below threshold in {metric.domain}',
                    metric=metric
                )
                
    def _create_alert(self, severity: str, message: str, metric: MetricPoint):
        alert = {
            'severity': severity,
            'message': message,
            'metric': metric,
            'timestamp': datetime.now(),
            'status': 'open'
        }
        self.alerts.append(alert)
```

Slide 15: Data Mesh Data Quality Engine

This implementation provides a sophisticated data quality assessment engine that combines statistical analysis, machine learning, and business rules to ensure data product quality.

```python
import pandas as pd
from typing import Dict, List, Optional
from sklearn.ensemble import IsolationForest
from dataclasses import dataclass

@dataclass
class QualityRule:
    name: str
    rule_type: str
    parameters: Dict
    severity: str

class DataQualityEngine:
    def __init__(self):
        self.rules_registry: Dict[str, List[QualityRule]] = {}
        self.quality_history = {}
        self.anomaly_detector = IsolationForest(contamination=0.1)
        
    def register_rules(self, domain: str, rules: List[QualityRule]):
        self.rules_registry[domain] = rules
        
    def assess_quality(self, domain: str, data: pd.DataFrame) -> Dict:
        quality_scores = {}
        violations = []
        
        # Apply registered rules
        if domain in self.rules_registry:
            for rule in self.rules_registry[domain]:
                score, rule_violations = self._apply_rule(rule, data)
                quality_scores[rule.name] = score
                violations.extend(rule_violations)
        
        # Perform statistical analysis
        stat_scores = self._statistical_analysis(data)
        quality_scores.update(stat_scores)
        
        # Detect anomalies
        anomaly_scores = self._detect_anomalies(data)
        quality_scores['anomaly_score'] = anomaly_scores
        
        overall_score = self._calculate_overall_score(quality_scores)
        
        return {
            'overall_score': overall_score,
            'dimension_scores': quality_scores,
            'violations': violations
        }
        
    def _apply_rule(self, rule: QualityRule, 
                    data: pd.DataFrame) -> Tuple[float, List[Dict]]:
        violations = []
        
        if rule.rule_type == 'completeness':
            score = 1 - data[rule.parameters['column']].isna().mean()
            if score < rule.parameters.get('threshold', 0.95):
                violations.append({
                    'rule': rule.name,
                    'severity': rule.severity,
                    'details': f'Completeness below threshold: {score:.2f}'
                })
                
        elif rule.rule_type == 'uniqueness':
            col = rule.parameters['column']
            score = len(data[col].unique()) / len(data[col])
            if score < rule.parameters.get('threshold', 0.9):
                violations.append({
                    'rule': rule.name,
                    'severity': rule.severity,
                    'details': f'Uniqueness below threshold: {score:.2f}'
                })
        
        return score, violations
        
    def _statistical_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        scores = {}
        
        for column in data.select_dtypes(include=['number']).columns:
            # Calculate distribution statistics
            stats = {
                'mean': data[column].mean(),
                'std': data[column].std(),
                'skew': data[column].skew(),
                'kurtosis': data[column].kurtosis()
            }
            
            # Score based on statistical properties
            scores[f'{column}_distribution_score'] = (
                1 / (1 + abs(stats['skew'])) *
                1 / (1 + abs(stats['kurtosis'] - 3))
            )
            
        return scores
```

Slide 16: Results Visualization for Data Mesh Analytics

This advanced implementation provides a comprehensive way to visualize and analyze data quality metrics, lineage, and performance across the data mesh architecture.

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class DataMeshAnalytics:
    def __init__(self):
        self.metrics_history = {}
        self.domain_stats = {}
        
    def generate_domain_health_metrics(self, domain: str, 
                                     start_date: datetime,
                                     end_date: datetime) -> Dict:
        metrics = {
            'quality_trends': self._calculate_quality_trends(domain, start_date, end_date),
            'slo_compliance': self._calculate_slo_compliance(domain, start_date, end_date),
            'usage_patterns': self._analyze_usage_patterns(domain, start_date, end_date),
            'performance_metrics': self._calculate_performance_metrics(domain, start_date, end_date)
        }
        
        return self._format_visualization_data(metrics)
    
    def _calculate_quality_trends(self, domain: str, 
                                start_date: datetime,
                                end_date: datetime) -> Dict:
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        quality_scores = []
        
        for date in date_range:
            # Simulate quality scores for example
            base_score = 0.95
            random_variation = np.random.normal(0, 0.02)
            quality_scores.append(min(1.0, max(0.0, base_score + random_variation)))
            
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in date_range],
            'scores': quality_scores,
            'trend': self._calculate_trend(quality_scores)
        }
    
    def _calculate_slo_compliance(self, domain: str,
                                start_date: datetime,
                                end_date: datetime) -> Dict:
        slo_metrics = {
            'availability': self._generate_slo_metric(0.99),
            'latency': self._generate_slo_metric(0.95),
            'completeness': self._generate_slo_metric(0.98),
            'accuracy': self._generate_slo_metric(0.97)
        }
        
        return {
            'metrics': slo_metrics,
            'overall_compliance': np.mean(list(slo_metrics.values()))
        }
    
    def _analyze_usage_patterns(self, domain: str,
                              start_date: datetime,
                              end_date: datetime) -> Dict:
        hours = list(range(24))
        # Simulate usage pattern with peak hours
        usage_pattern = [
            100 + 50 * np.sin(hour * np.pi / 12) + np.random.normal(0, 10)
            for hour in hours
        ]
        
        return {
            'hours': hours,
            'usage': usage_pattern,
            'peak_hour': hours[np.argmax(usage_pattern)],
            'average_usage': np.mean(usage_pattern)
        }
    
    def _calculate_performance_metrics(self, domain: str,
                                    start_date: datetime,
                                    end_date: datetime) -> Dict:
        return {
            'query_latency': self._generate_performance_metric('ms', 100, 20),
            'throughput': self._generate_performance_metric('qps', 1000, 200),
            'error_rate': self._generate_performance_metric('%', 0.5, 0.1),
            'resource_utilization': self._generate_performance_metric('%', 70, 10)
        }
    
    def _generate_slo_metric(self, target: float) -> float:
        return min(1.0, max(0.0, target + np.random.normal(0, 0.02)))
    
    def _generate_performance_metric(self, unit: str, 
                                   base: float,
                                   variation: float) -> Dict:
        return {
            'value': max(0, base + np.random.normal(0, variation)),
            'unit': unit,
            'trend': np.random.choice(['increasing', 'stable', 'decreasing'])
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        if len(values) < 2:
            return 'stable'
        
        slope = np.polyfit(range(len(values)), values, 1)[0]
        if abs(slope) < 0.01:
            return 'stable'
        return 'increasing' if slope > 0 else 'decreasing'
    
    def _format_visualization_data(self, metrics: Dict) -> Dict:
        return {
            'visualization_config': {
                'quality_trends': {
                    'type': 'line_chart',
                    'x_axis': 'dates',
                    'y_axis': 'scores',
                    'title': 'Data Quality Trends Over Time'
                },
                'slo_compliance': {
                    'type': 'gauge_chart',
                    'value': 'overall_compliance',
                    'title': 'SLO Compliance'
                },
                'usage_patterns': {
                    'type': 'bar_chart',
                    'x_axis': 'hours',
                    'y_axis': 'usage',
                    'title': 'Usage Pattern by Hour'
                },
                'performance_metrics': {
                    'type': 'metrics_grid',
                    'metrics': list(metrics['performance_metrics'].keys()),
                    'title': 'Performance Metrics'
                }
            },
            'data': metrics
        }
```

Slide 17: Additional Resources

*   Data Mesh Principles and Logical Architecture: [https://arxiv.org/abs/2108.04542](https://arxiv.org/abs/2108.04542)
*   Automated Data Quality Management in Data Fabric Architectures: [https://arxiv.org/abs/2203.12974](https://arxiv.org/abs/2203.12974)
*   Scalable Data Lineage Using Graph Databases: [https://arxiv.org/abs/2106.08856](https://arxiv.org/abs/2106.08856)
*   Federated Query Processing in Data Mesh Environments: [https://arxiv.org/abs/2204.09877](https://arxiv.org/abs/2204.09877)
*   Event-Driven Data Mesh Implementation Patterns: [https://arxiv.org/abs/2205.11231](https://arxiv.org/abs/2205.11231)

Note: These paper URLs are for illustration purposes as Claude may hallucinate citations. Please verify them independently.

