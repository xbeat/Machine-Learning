## The Evolution of Databases From File Systems to Modern Marvels
Slide 1: Database Connections in Python

Understanding how to establish secure and efficient database connections is fundamental in Python. This code demonstrates connecting to different database types using appropriate drivers and connection pooling for optimal resource management.

```python
import pymysql
import psycopg2
from sqlalchemy import create_engine
from contextlib import contextmanager

class DatabaseConnector:
    def __init__(self, db_type, host, user, password, database):
        self.db_type = db_type
        self.credentials = {
            'host': host,
            'user': user,
            'password': password,
            'database': database
        }
        
    @contextmanager
    def get_connection(self):
        if self.db_type == 'mysql':
            conn = pymysql.connect(**self.credentials)
        elif self.db_type == 'postgresql':
            conn = psycopg2.connect(**self.credentials)
        try:
            yield conn
        finally:
            conn.close()

# Example usage
db = DatabaseConnector('postgresql', 'localhost', 'user', 'pass', 'mydb')
with db.get_connection() as conn:
    with conn.cursor() as cursor:
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
```

Slide 2: Query Building and Parameter Binding

Safe query construction is crucial for preventing SQL injection. This implementation showcases proper parameter binding techniques and query building patterns using both standard Python DB-API and SQLAlchemy ORM.

```python
from typing import Dict, Any
import sqlalchemy as sa
from sqlalchemy.orm import Session

class QueryBuilder:
    def __init__(self, engine):
        self.engine = engine
        
    def safe_execute(self, query: str, params: Dict[str, Any] = None):
        with Session(self.engine) as session:
            result = session.execute(sa.text(query), params or {})
            return result.fetchall()
            
    def build_select(self, table: str, conditions: Dict[str, Any]):
        placeholders = ' AND '.join(f"{k} = :{k}" for k in conditions.keys())
        query = f"SELECT * FROM {table}"
        if conditions:
            query += f" WHERE {placeholders}"
        return self.safe_execute(query, conditions)

# Example usage
engine = create_engine('postgresql://user:pass@localhost/mydb')
qb = QueryBuilder(engine)
results = qb.build_select('users', {'status': 'active', 'role': 'admin'})
```

Slide 3: CRUD Operations with MongoDB

MongoDB's document-oriented structure requires a different approach to data operations. This implementation shows how to perform CRUD operations while handling MongoDB-specific features like document embedding and arrays.

```python
from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId

class MongoDBHandler:
    def __init__(self, connection_string: str, database: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database]
    
    def create_document(self, collection: str, document: dict):
        document['created_at'] = datetime.utcnow()
        return self.db[collection].insert_one(document)
    
    def find_documents(self, collection: str, query: dict, projection: dict = None):
        return list(self.db[collection].find(query, projection))
    
    def update_document(self, collection: str, query: dict, update: dict):
        return self.db[collection].update_one(
            query,
            {'$set': update, '$currentDate': {'last_modified': True}}
        )
    
    def delete_document(self, collection: str, query: dict):
        return self.db[collection].delete_one(query)

# Example usage
mongo = MongoDBHandler('mongodb://localhost:27017', 'myapp')
result = mongo.create_document('users', {
    'name': 'John Doe',
    'email': 'john@example.com',
    'preferences': {'theme': 'dark', 'notifications': True}
})
```

Slide 4: Redis Cache Integration

Redis serves as an efficient in-memory data structure store, commonly used for caching and real-time analytics. This implementation demonstrates key Redis operations and caching patterns with proper connection handling and data serialization.

```python
import redis
import json
from typing import Union, Any
from functools import wraps

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis = redis.Redis(host=host, port=port, db=db)
        
    def set_data(self, key: str, value: Any, expiry: int = 3600):
        serialized = json.dumps(value)
        return self.redis.setex(key, expiry, serialized)
    
    def get_data(self, key: str) -> Union[Any, None]:
        data = self.redis.get(key)
        return json.loads(data) if data else None
    
    def cache_decorator(self, expiry: int = 3600):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = f"{func.__name__}:{args}:{kwargs}"
                cached_result = self.get_data(cache_key)
                
                if cached_result is not None:
                    return cached_result
                    
                result = func(*args, **kwargs)
                self.set_data(cache_key, result, expiry)
                return result
            return wrapper
        return decorator

# Example usage
cache = RedisCache()

@cache.cache_decorator(expiry=300)
def expensive_computation(n: int):
    return sum(i * i for i in range(n))

result = expensive_computation(1000)  # First call computes
cached = expensive_computation(1000)  # Second call uses cache
```

Slide 5: Time Series Data Management

Time series data requires specialized handling for efficient storage and retrieval. This implementation shows how to manage time series data with proper indexing, aggregation, and downsampling capabilities.

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

class TimeSeriesManager:
    def __init__(self, engine):
        self.engine = engine
        
    def insert_timeseries(self, table: str, timestamp: datetime, 
                         values: Dict[str, float]):
        query = f"""
            INSERT INTO {table} (timestamp, {','.join(values.keys())})
            VALUES (:timestamp, {','.join(f':{k}' for k in values.keys())})
        """
        params = {'timestamp': timestamp, **values}
        with self.engine.begin() as conn:
            conn.execute(sa.text(query), params)
            
    def get_aggregated_data(self, table: str, start: datetime, 
                           end: datetime, interval: str = '1h'):
        query = f"""
            SELECT 
                time_bucket(:interval, timestamp) as bucket,
                avg(value) as avg_value,
                max(value) as max_value,
                min(value) as min_value
            FROM {table}
            WHERE timestamp BETWEEN :start AND :end
            GROUP BY bucket
            ORDER BY bucket
        """
        params = {'interval': interval, 'start': start, 'end': end}
        with self.engine.connect() as conn:
            result = conn.execute(sa.text(query), params)
            return pd.DataFrame(result.fetchall())

# Example usage
engine = create_engine('postgresql://user:pass@localhost/timeseries_db')
ts_manager = TimeSeriesManager(engine)

# Insert sample data
now = datetime.utcnow()
for i in range(24):
    timestamp = now - timedelta(hours=i)
    values = {'temperature': 20 + np.random.normal(0, 2),
              'humidity': 50 + np.random.normal(0, 5)}
    ts_manager.insert_timeseries('sensor_data', timestamp, values)

# Retrieve aggregated data
df = ts_manager.get_aggregated_data('sensor_data', 
                                   now - timedelta(days=1), 
                                   now)
```

Slide 6: Graph Database Operations with Neo4j

Graph databases excel at managing highly connected data. This implementation demonstrates creating, querying, and traversing graph structures using Neo4j's Python driver with proper transaction management.

```python
from neo4j import GraphDatabase
from typing import List, Dict, Any

class Neo4jGraph:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def create_node(self, label: str, properties: Dict[str, Any]):
        with self.driver.session() as session:
            return session.write_transaction(self._create_node_tx, 
                                          label, properties)
    
    @staticmethod
    def _create_node_tx(tx, label: str, properties: Dict[str, Any]):
        query = (
            f"CREATE (n:{label} $props) "
            "RETURN n"
        )
        result = tx.run(query, props=properties)
        return result.single()[0]
        
    def create_relationship(self, start_node: int, end_node: int, 
                          rel_type: str, properties: Dict[str, Any] = None):
        with self.driver.session() as session:
            return session.write_transaction(
                self._create_relationship_tx,
                start_node, end_node, rel_type, properties or {}
            )
    
    @staticmethod
    def _create_relationship_tx(tx, start_node: int, end_node: int, 
                              rel_type: str, properties: Dict[str, Any]):
        query = (
            "MATCH (a), (b) "
            "WHERE ID(a) = $start_id AND ID(b) = $end_id "
            f"CREATE (a)-[r:{rel_type} $props]->(b) "
            "RETURN r"
        )
        result = tx.run(query, start_id=start_node, end_id=end_node, 
                       props=properties)
        return result.single()[0]

# Example usage
graph_db = Neo4jGraph("bolt://localhost:7687", "neo4j", "password")

# Create nodes and relationships
person1 = graph_db.create_node("Person", {
    "name": "Alice",
    "age": 30
})
person2 = graph_db.create_node("Person", {
    "name": "Bob",
    "age": 35
})

relationship = graph_db.create_relationship(
    person1.id, person2.id, "KNOWS",
    {"since": "2020-01-01"}
)

graph_db.close()
```

Slide 7: Vector Database Implementation

Vector databases are essential for machine learning applications, particularly in similarity search and recommendation systems. This implementation showcases vector storage, indexing, and efficient nearest neighbor search.

```python
import numpy as np
from typing import List, Tuple
from scipy.spatial.distance import cosine
from collections import defaultdict

class VectorDatabase:
    def __init__(self, dimensions: int, index_size: int = 1000):
        self.dimensions = dimensions
        self.vectors = {}  # id -> vector mapping
        self.metadata = {}  # id -> metadata mapping
        self.index_size = index_size
        self.index = defaultdict(list)
        
    def insert_vector(self, id: str, vector: np.ndarray, 
                     metadata: dict = None):
        if vector.shape[0] != self.dimensions:
            raise ValueError(f"Vector must have {self.dimensions} dimensions")
            
        self.vectors[id] = vector
        if metadata:
            self.metadata[id] = metadata
        self._update_index(id, vector)
        
    def _update_index(self, id: str, vector: np.ndarray):
        # Simple LSH-like indexing
        projection = (vector > 0).astype(int)
        bucket = tuple(projection)
        self.index[bucket].append(id)
        
    def search_nearest(self, query_vector: np.ndarray, 
                      k: int = 5) -> List[Tuple[str, float]]:
        if query_vector.shape[0] != self.dimensions:
            raise ValueError(f"Query vector must have {self.dimensions} dimensions")
            
        # Get candidate vectors from index
        projection = (query_vector > 0).astype(int)
        candidates = self.index[tuple(projection)]
        
        # Calculate distances
        distances = []
        for vid in candidates:
            dist = cosine(query_vector, self.vectors[vid])
            distances.append((vid, dist))
            
        # Sort by distance and return top k
        return sorted(distances, key=lambda x: x[1])[:k]

# Example usage
vector_db = VectorDatabase(dimensions=128)

# Generate sample embeddings
embeddings = np.random.randn(10, 128)
for i, emb in enumerate(embeddings):
    vector_db.insert_vector(
        f"doc_{i}", 
        emb,
        {"title": f"Document {i}", "type": "text"}
    )

# Search for similar vectors
query = np.random.randn(128)
results = vector_db.search_nearest(query, k=3)
print("Nearest neighbors:", results)
```

Slide 8: Distributed Database Sharding

Implementing database sharding for horizontal scalability requires careful consideration of data distribution and query routing. This code demonstrates a basic sharding implementation with consistent hashing.

```python
import hashlib
from typing import List, Any, Dict
from collections import defaultdict
import pymongo

class DatabaseShard:
    def __init__(self, connection_string: str, shard_key: str):
        self.client = pymongo.MongoClient(connection_string)
        self.shard_key = shard_key
        
    def insert(self, collection: str, document: Dict[str, Any]):
        self.client[collection].insert_one(document)
        
    def find(self, collection: str, query: Dict[str, Any]):
        return self.client[collection].find(query)

class ShardManager:
    def __init__(self, shard_count: int):
        self.shard_count = shard_count
        self.shards = {}
        
    def add_shard(self, shard_id: int, connection_string: str, 
                  shard_key: str):
        if shard_id >= self.shard_count:
            raise ValueError(f"Shard ID must be < {self.shard_count}")
        self.shards[shard_id] = DatabaseShard(connection_string, 
                                            shard_key)
        
    def _get_shard_id(self, key: str) -> int:
        """Determine shard using consistent hashing"""
        hash_val = int(hashlib.md5(str(key).encode()).hexdigest(), 16)
        return hash_val % self.shard_count
        
    def insert_document(self, collection: str, document: Dict[str, Any]):
        if '_id' not in document:
            document['_id'] = str(ObjectId())
            
        shard_id = self._get_shard_id(document[self.shards[0].shard_key])
        self.shards[shard_id].insert(collection, document)
        
    def query_all_shards(self, collection: str, 
                        query: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        for shard in self.shards.values():
            results.extend(list(shard.find(collection, query)))
        return results

# Example usage
shard_manager = ShardManager(shard_count=3)

# Add shards
shard_manager.add_shard(0, "mongodb://localhost:27017/shard0", "user_id")
shard_manager.add_shard(1, "mongodb://localhost:27018/shard1", "user_id")
shard_manager.add_shard(2, "mongodb://localhost:27019/shard2", "user_id")

# Insert distributed data
user_data = {
    "user_id": "user123",
    "name": "John Doe",
    "email": "john@example.com"
}
shard_manager.insert_document("users", user_data)

# Query across all shards
results = shard_manager.query_all_shards("users", 
                                       {"name": "John Doe"})
```

Slide 9: Time Series Database Analytics

Time series databases require specialized analytics capabilities for handling temporal data patterns, seasonality, and trends. This implementation demonstrates advanced time series operations including rollups and window functions.

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class TimeSeriesMetrics:
    mean: float
    std: float
    min: float
    max: float
    percentiles: Dict[int, float]

class TimeSeriesAnalytics:
    def __init__(self):
        self.data: Dict[str, pd.DataFrame] = {}
        
    def add_series(self, name: str, timestamps: List[datetime], 
                  values: List[float]):
        self.data[name] = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        }).set_index('timestamp')
        
    def compute_rolling_stats(self, series_name: str, 
                            window: str = '1H') -> pd.DataFrame:
        if series_name not in self.data:
            raise KeyError(f"Series {series_name} not found")
            
        df = self.data[series_name]
        return pd.DataFrame({
            'rolling_mean': df['value'].rolling(window).mean(),
            'rolling_std': df['value'].rolling(window).std(),
            'rolling_min': df['value'].rolling(window).min(),
            'rolling_max': df['value'].rolling(window).max()
        })
        
    def detect_anomalies(self, series_name: str, 
                        z_score_threshold: float = 3.0) -> pd.Series:
        if series_name not in self.data:
            raise KeyError(f"Series {series_name} not found")
            
        df = self.data[series_name]
        z_scores = np.abs((df['value'] - df['value'].mean()) / 
                         df['value'].std())
        return z_scores > z_score_threshold
        
    def aggregate_by_period(self, series_name: str, 
                          period: str = '1D') -> TimeSeriesMetrics:
        if series_name not in self.data:
            raise KeyError(f"Series {series_name} not found")
            
        df = self.data[series_name]
        agg_data = df.resample(period).agg({
            'value': ['mean', 'std', 'min', 'max']
        })['value']
        
        percentiles = df.resample(period)['value'].quantile([
            0.25, 0.5, 0.75, 0.95, 0.99
        ]).to_dict()
        
        return TimeSeriesMetrics(
            mean=agg_data['mean'],
            std=agg_data['std'],
            min=agg_data['min'],
            max=agg_data['max'],
            percentiles=percentiles
        )

# Example usage
ts_analytics = TimeSeriesAnalytics()

# Generate sample time series data
now = datetime.now()
timestamps = [now - timedelta(minutes=i) for i in range(1000)]
values = np.random.normal(100, 15, 1000)
values[500:550] += 100  # Add anomaly

ts_analytics.add_series('sensor1', timestamps, values)

# Analyze the time series
rolling_stats = ts_analytics.compute_rolling_stats('sensor1', '1H')
anomalies = ts_analytics.detect_anomalies('sensor1', 3.0)
daily_metrics = ts_analytics.aggregate_by_period('sensor1', '1D')
```

Slide 10: Document Database Schema Validation

Implementing schema validation in document databases ensures data consistency while maintaining flexibility. This implementation shows how to create and enforce dynamic schemas.

```python
from typing import Any, Dict, List, Union, Optional
from enum import Enum
import jsonschema
from datetime import datetime

class FieldType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "number"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    ARRAY = "array"
    OBJECT = "object"

class SchemaValidator:
    def __init__(self):
        self.schemas: Dict[str, Dict] = {}
        
    def create_schema(self, collection_name: str, 
                     schema_definition: Dict[str, Dict[str, Any]]):
        """
        Create a JSON Schema for document validation
        """
        properties = {}
        required = []
        
        for field_name, field_def in schema_definition.items():
            field_type = field_def.get('type')
            is_required = field_def.get('required', False)
            
            if is_required:
                required.append(field_name)
                
            field_schema = {
                'type': field_type.value if isinstance(field_type, FieldType) 
                        else field_type
            }
            
            if 'enum' in field_def:
                field_schema['enum'] = field_def['enum']
                
            if field_type == FieldType.ARRAY:
                field_schema['items'] = field_def.get('items', {})
                
            if field_type == FieldType.OBJECT:
                field_schema['properties'] = field_def.get('properties', {})
                
            properties[field_name] = field_schema
            
        self.schemas[collection_name] = {
            'type': 'object',
            'properties': properties,
            'required': required,
            'additionalProperties': False
        }
        
    def validate_document(self, collection_name: str, 
                         document: Dict[str, Any]) -> List[str]:
        """
        Validate a document against its schema
        Returns list of validation errors
        """
        if collection_name not in self.schemas:
            raise ValueError(f"No schema defined for {collection_name}")
            
        try:
            jsonschema.validate(document, self.schemas[collection_name])
            return []
        except jsonschema.exceptions.ValidationError as e:
            return [e.message]

# Example usage
validator = SchemaValidator()

# Define schema for a user collection
user_schema = {
    'username': {
        'type': FieldType.STRING,
        'required': True
    },
    'email': {
        'type': FieldType.STRING,
        'required': True
    },
    'age': {
        'type': FieldType.INTEGER,
        'required': False
    },
    'roles': {
        'type': FieldType.ARRAY,
        'items': {'type': 'string'},
        'required': True
    }
}

validator.create_schema('users', user_schema)

# Validate documents
valid_doc = {
    'username': 'john_doe',
    'email': 'john@example.com',
    'roles': ['user', 'admin']
}

invalid_doc = {
    'username': 'jane_doe',
    'roles': 'admin'  # Should be an array
}

print(validator.validate_document('users', valid_doc))  # []
print(validator.validate_document('users', invalid_doc))  # ['roles' is not of type 'array']
```

Slide 11: Spatial Database Operations

Spatial databases require specialized indexing and query capabilities for geographic data. This implementation demonstrates spatial operations, including distance calculations and geofencing.

```python
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from rtree import index
from shapely.geometry import Point, Polygon

@dataclass
class SpatialPoint:
    id: str
    latitude: float
    longitude: float
    metadata: Dict

class SpatialDatabase:
    def __init__(self):
        self.idx = index.Index()
        self.points = {}
        self.geofences = {}
        
    def insert_point(self, point: SpatialPoint):
        self.points[point.id] = point
        # R-tree expects (left, bottom, right, top)
        self.idx.insert(
            hash(point.id), 
            (point.longitude, point.latitude, 
             point.longitude, point.latitude)
        )
        
    def create_geofence(self, fence_id: str, 
                       coordinates: List[Tuple[float, float]]):
        """Create a polygon geofence from coordinates"""
        self.geofences[fence_id] = Polygon(coordinates)
        
    def find_points_within_radius(self, lat: float, lon: float, 
                                radius_km: float) -> List[SpatialPoint]:
        """Find all points within radius kilometers of center"""
        center = Point(lon, lat)
        
        # Convert radius to rough bounding box
        # 1 degree ~ 111km at equator
        degree_radius = radius_km / 111.0
        bbox = (
            lon - degree_radius,
            lat - degree_radius,
            lon + degree_radius,
            lat + degree_radius
        )
        
        results = []
        for point_id in self.idx.intersection(bbox):
            point = self.points[str(hash(point_id))]
            point_geom = Point(point.longitude, point.latitude)
            
            if center.distance(point_geom) * 111.0 <= radius_km:
                results.append(point)
                
        return results
        
    def points_in_geofence(self, fence_id: str) -> List[SpatialPoint]:
        """Find all points within a specific geofence"""
        if fence_id not in self.geofences:
            raise ValueError(f"Geofence {fence_id} not found")
            
        fence = self.geofences[fence_id]
        bbox = fence.bounds
        
        results = []
        for point_id in self.idx.intersection(bbox):
            point = self.points[str(hash(point_id))]
            point_geom = Point(point.longitude, point.latitude)
            
            if fence.contains(point_geom):
                results.append(point)
                
        return results

# Example usage
spatial_db = SpatialDatabase()

# Insert some points
points = [
    SpatialPoint("store1", 40.7128, -74.0060, {"name": "NYC Store"}),
    SpatialPoint("store2", 34.0522, -118.2437, {"name": "LA Store"}),
    SpatialPoint("store3", 41.8781, -87.6298, {"name": "Chicago Store"})
]

for point in points:
    spatial_db.insert_point(point)

# Create a geofence (simple rectangle around NYC)
nyc_bounds = [
    (-74.1, 40.6),
    (-74.1, 40.8),
    (-73.9, 40.8),
    (-73.9, 40.6),
    (-74.1, 40.6)
]
spatial_db.create_geofence("nyc_area", nyc_bounds)

# Find points within 100km of Times Square
nearby = spatial_db.find_points_within_radius(40.7580, -73.9855, 100)

# Find points within NYC geofence
in_nyc = spatial_db.points_in_geofence("nyc_area")
```

Slide 12: Blockchain Database Implementation

This implementation demonstrates core blockchain concepts including block creation, chain validation, and consensus mechanisms using a simplified proof-of-work system.

```python
import hashlib
import time
from typing import List, Dict, Any
import json
from dataclasses import dataclass

@dataclass
class Block:
    index: int
    timestamp: float
    transactions: List[Dict[str, Any]]
    previous_hash: str
    nonce: int = 0
    
    @property
    def hash(self) -> str:
        """Calculate block hash"""
        block_string = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class BlockchainDatabase:
    def __init__(self, difficulty: int = 4):
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict[str, Any]] = []
        self.difficulty = difficulty
        self.create_genesis_block()
        
    def create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=[],
            previous_hash="0" * 64
        )
        self.mine_block(genesis_block)
        self.chain.append(genesis_block)
        
    def mine_block(self, block: Block) -> bool:
        """Mine a block (Proof of Work)"""
        target = "0" * self.difficulty
        
        while block.hash[:self.difficulty] != target:
            block.nonce += 1
            if block.nonce > 1000000:  # Prevent infinite loops
                return False
        return True
        
    def add_transaction(self, sender: str, recipient: str, 
                       data: Dict[str, Any]):
        """Add a new transaction to pending transactions"""
        self.pending_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'timestamp': time.time(),
            'data': data
        })
        
    def create_block(self) -> Block:
        """Create a new block with pending transactions"""
        previous_block = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            previous_hash=previous_block.hash
        )
        self.pending_transactions = []
        return new_block
        
    def add_block(self, block: Block) -> bool:
        """Add a mined block to the chain"""
        if not self.is_valid_block(block):
            return False
            
        self.chain.append(block)
        return True
        
    def is_valid_block(self, block: Block) -> bool:
        """Validate a block"""
        previous_block = self.chain[-1]
        
        if block.previous_hash != previous_block.hash:
            return False
            
        if block.index != len(self.chain):
            return False
            
        if block.hash[:self.difficulty] != "0" * self.difficulty:
            return False
            
        return True
        
    def is_chain_valid(self) -> bool:
        """Validate the entire chain"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            if current.previous_hash != previous.hash:
                return False
                
            if current.hash[:self.difficulty] != "0" * self.difficulty:
                return False
                
        return True

# Example usage
blockchain = BlockchainDatabase(difficulty=4)

# Add some transactions
blockchain.add_transaction(
    "user1", "user2", 
    {"document_hash": "abc123", "type": "transfer"}
)
blockchain.add_transaction(
    "user2", "user3", 
    {"document_hash": "def456", "type": "update"}
)

# Create and mine new block
new_block = blockchain.create_block()
if blockchain.mine_block(new_block):
    blockchain.add_block(new_block)

# Validate chain
print(f"Chain valid: {blockchain.is_chain_valid()}")
```

Slide 13: Event Sourcing Database Pattern

Event sourcing maintains a complete history of state changes as a sequence of immutable events. This implementation demonstrates event storage, replay, and state reconstruction capabilities.

```python
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

class EventType(Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"

@dataclass
class Event:
    event_id: str
    aggregate_id: str
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime
    version: int

class EventStore:
    def __init__(self):
        self.events: Dict[str, List[Event]] = {}
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.snapshot_frequency = 10
        
    def append_event(self, event: Event):
        """Add new event to the store"""
        if event.aggregate_id not in self.events:
            self.events[event.aggregate_id] = []
            
        self.events[event.aggregate_id].append(event)
        
        # Create snapshot if needed
        if len(self.events[event.aggregate_id]) % self.snapshot_frequency == 0:
            self._create_snapshot(event.aggregate_id)
            
    def get_events(self, aggregate_id: str, 
                  after_version: Optional[int] = None) -> List[Event]:
        """Get all events for an aggregate"""
        if aggregate_id not in self.events:
            return []
            
        events = self.events[aggregate_id]
        if after_version is not None:
            events = [e for e in events if e.version > after_version]
            
        return events
        
    def _create_snapshot(self, aggregate_id: str):
        """Create a snapshot of current state"""
        current_state = self.get_current_state(aggregate_id)
        self.snapshots[aggregate_id] = current_state
        
    def get_current_state(self, aggregate_id: str) -> Dict[str, Any]:
        """Reconstruct current state from events"""
        if aggregate_id not in self.events:
            return {}
            
        # Start from last snapshot if available
        if aggregate_id in self.snapshots:
            state = self.snapshots[aggregate_id].copy()
            events = self.get_events(aggregate_id, 
                                   len(self.snapshots[aggregate_id]))
        else:
            state = {}
            events = self.get_events(aggregate_id)
            
        # Apply all events
        for event in events:
            if event.event_type == EventType.CREATE:
                state.update(event.data)
            elif event.event_type == EventType.UPDATE:
                state.update(event.data)
            elif event.event_type == EventType.DELETE:
                for key in event.data:
                    state.pop(key, None)
                    
        return state

class EventSourcedEntity:
    def __init__(self, entity_id: str, event_store: EventStore):
        self.entity_id = entity_id
        self.event_store = event_store
        self.version = 0
        
    def apply_event(self, event_type: EventType, data: Dict[str, Any]):
        """Apply new event to entity"""
        event = Event(
            event_id=f"{self.entity_id}_{self.version + 1}",
            aggregate_id=self.entity_id,
            event_type=event_type,
            data=data,
            timestamp=datetime.now(),
            version=self.version + 1
        )
        self.event_store.append_event(event)
        self.version += 1
        
    @property
    def current_state(self) -> Dict[str, Any]:
        """Get current state of entity"""
        return self.event_store.get_current_state(self.entity_id)

# Example usage
event_store = EventStore()
user = EventSourcedEntity("user123", event_store)

# Create user
user.apply_event(EventType.CREATE, {
    "name": "John Doe",
    "email": "john@example.com"
})

# Update user
user.apply_event(EventType.UPDATE, {
    "email": "john.doe@example.com"
})

# Add new field
user.apply_event(EventType.UPDATE, {
    "age": 30
})

# Get current state
print(user.current_state)
```

Slide 14: Additional Resources

*   "A Comprehensive Survey of Database Architecture Optimization" - [https://arxiv.org/abs/2208.07745](https://arxiv.org/abs/2208.07745)
*   "Distributed Database Systems: Principles and Systems" - [https://arxiv.org/abs/2201.03192](https://arxiv.org/abs/2201.03192)
*   "Vector Databases: Architectures for Similarity Search" - [https://arxiv.org/abs/2310.14637](https://arxiv.org/abs/2310.14637)
*   "Time Series Database Systems: A Systematic Review" - [https://arxiv.org/abs/2301.05561](https://arxiv.org/abs/2301.05561)

Suggested search terms for further research:

*   Database optimization techniques
*   Distributed database architectures
*   Modern database management systems
*   NoSQL database design patterns
*   Time series database implementations
*   Vector database architectures
*   Blockchain database systems
*   Event sourcing patterns

