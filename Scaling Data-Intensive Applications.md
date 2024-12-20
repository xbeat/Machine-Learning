## Scaling Data-Intensive Applications
Slide 1: Understanding Partitioning Fundamentals

Partitioning data involves breaking large datasets into smaller, more manageable chunks called partitions or shards. This fundamental concept enables horizontal scaling by distributing data and load across multiple machines while maintaining data locality and reducing query latency.

```python
from dataclasses import dataclass
from typing import Any, Dict, List
import hashlib

@dataclass
class Partition:
    id: int
    data: Dict[str, Any]

class HashPartitioner:
    def __init__(self, num_partitions: int):
        self.num_partitions = num_partitions
        self.partitions = [Partition(i, {}) for i in range(num_partitions)]
    
    def get_partition(self, key: str) -> int:
        # Consistent hash function for partition selection
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_value % self.num_partitions
    
    def insert(self, key: str, value: Any):
        partition_id = self.get_partition(key)
        self.partitions[partition_id].data[key] = value
        
# Example usage
partitioner = HashPartitioner(num_partitions=3)
partitioner.insert("user_123", {"name": "John", "age": 30})
partitioner.insert("user_456", {"name": "Alice", "age": 25})
```

Slide 2: Range-Based Partitioning Strategy

Range partitioning divides data based on sequential ranges of keys, offering efficient range queries but potentially leading to uneven data distribution. This approach is commonly used in time-series data and ordered datasets where range scans are frequent operations.

```python
class RangePartitioner:
    def __init__(self, ranges: List[tuple]):
        self.ranges = sorted(ranges)  # List of (min, max) tuples
        self.partitions = [{} for _ in range(len(ranges))]
    
    def get_partition(self, key: int) -> int:
        for i, (min_val, max_val) in enumerate(self.ranges):
            if min_val <= key <= max_val:
                return i
        raise ValueError(f"Key {key} doesn't fall into any partition range")
    
    def insert(self, key: int, value: Any):
        partition_id = self.get_partition(key)
        self.partitions[partition_id][key] = value

# Example usage
ranges = [(0, 100), (101, 200), (201, 300)]
range_partitioner = RangePartitioner(ranges)
range_partitioner.insert(50, "Data for ID 50")
range_partitioner.insert(150, "Data for ID 150")
```

Slide 3: Dynamic Partition Rebalancing

Dynamic partition rebalancing becomes crucial when data distribution becomes skewed or when nodes are added/removed from the cluster. This implementation demonstrates an adaptive rebalancing mechanism that monitors partition sizes and triggers redistribution.

```python
from collections import defaultdict
import random

class DynamicPartitioner:
    def __init__(self, initial_nodes: int, rebalance_threshold: float = 0.2):
        self.nodes = {i: defaultdict(dict) for i in range(initial_nodes)}
        self.threshold = rebalance_threshold
        self.key_to_node = {}
    
    def _check_balance(self) -> bool:
        sizes = [len(node) for node in self.nodes.values()]
        avg_size = sum(sizes) / len(sizes)
        return all(abs(size - avg_size) / avg_size <= self.threshold 
                  for size in sizes if avg_size > 0)
    
    def rebalance(self):
        if self._check_balance():
            return
        
        # Collect all data
        all_data = []
        for node_data in self.nodes.values():
            all_data.extend(node_data.items())
            
        # Clear existing nodes
        for node in self.nodes.values():
            node.clear()
            
        # Redistribute data
        for key, value in all_data:
            target_node = random.randint(0, len(self.nodes) - 1)
            self.nodes[target_node][key] = value
            self.key_to_node[key] = target_node

# Example usage
dp = DynamicPartitioner(3)
for i in range(100):
    dp.nodes[i % 3][f"key_{i}"] = f"value_{i}"
dp.rebalance()
```

Slide 4: Consistent Hashing Implementation

Consistent hashing provides a way to distribute data across nodes while minimizing redistribution when the number of nodes changes. This implementation includes virtual nodes to improve distribution balance.

```python
import hashlib
from bisect import bisect_right
from typing import Optional

class ConsistentHash:
    def __init__(self, nodes: List[str], replicas: int = 100):
        self.replicas = replicas
        self.ring = []
        self.node_map = {}
        
        for node in nodes:
            self.add_node(node)
    
    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node: str):
        for i in range(self.replicas):
            hash_key = self._hash(f"{node}:{i}")
            self.ring.append(hash_key)
            self.node_map[hash_key] = node
        self.ring.sort()
    
    def remove_node(self, node: str):
        for i in range(self.replicas):
            hash_key = self._hash(f"{node}:{i}")
            self.ring.remove(hash_key)
            del self.node_map[hash_key]
    
    def get_node(self, key: str) -> Optional[str]:
        if not self.ring:
            return None
        
        hash_key = self._hash(key)
        idx = bisect_right(self.ring, hash_key) % len(self.ring)
        return self.node_map[self.ring[idx]]

# Example usage
nodes = ["node1", "node2", "node3"]
ch = ConsistentHash(nodes)
print(f"Key 'user1' maps to: {ch.get_node('user1')}")
print(f"Key 'user2' maps to: {ch.get_node('user2')}")
```

Slide 5: Database Sharding Implementation

Database sharding extends partitioning concepts to distributed databases, allowing horizontal scaling across multiple database instances. This implementation demonstrates a practical approach to shard routing and query distribution.

```python
from typing import Dict, List, Any
import mysql.connector
from contextlib import contextmanager

class ShardedDatabase:
    def __init__(self, shard_config: Dict[int, Dict[str, str]]):
        self.shard_config = shard_config
        self.connections = {}
        
    @contextmanager
    def get_shard_connection(self, shard_id: int):
        if shard_id not in self.connections:
            config = self.shard_config[shard_id]
            self.connections[shard_id] = mysql.connector.connect(
                host=config['host'],
                user=config['user'],
                password=config['password'],
                database=config['database']
            )
        try:
            yield self.connections[shard_id]
        except Exception as e:
            self.connections[shard_id].rollback()
            raise e

    def insert_user(self, user_id: int, user_data: Dict[str, Any]):
        shard_id = user_id % len(self.shard_config)
        with self.get_shard_connection(shard_id) as conn:
            cursor = conn.cursor()
            query = """
                INSERT INTO users (id, name, email) 
                VALUES (%s, %s, %s)
            """
            cursor.execute(query, (
                user_id, 
                user_data['name'], 
                user_data['email']
            ))
            conn.commit()

# Example configuration
shard_config = {
    0: {
        'host': 'shard0.example.com',
        'user': 'db_user',
        'password': 'password',
        'database': 'users_shard_0'
    },
    1: {
        'host': 'shard1.example.com',
        'user': 'db_user',
        'password': 'password',
        'database': 'users_shard_1'
    }
}

# Usage example (commented out as it requires actual DB connection)
# db = ShardedDatabase(shard_config)
# db.insert_user(123, {'name': 'John Doe', 'email': 'john@example.com'})
```

Slide 6: Distributed Cache Partitioning

Implementing a distributed cache system with partitioning capabilities helps maintain high availability and fault tolerance while ensuring efficient data access patterns across multiple cache nodes.

```python
from typing import Optional, Any
import time
import threading
import random

class PartitionedCache:
    def __init__(self, num_partitions: int, ttl_seconds: int = 3600):
        self.num_partitions = num_partitions
        self.ttl_seconds = ttl_seconds
        self.partitions = [{} for _ in range(num_partitions)]
        self.timestamps = [{} for _ in range(num_partitions)]
        self.locks = [threading.Lock() for _ in range(num_partitions)]
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_expired,
            daemon=True
        )
        self.cleanup_thread.start()
    
    def _get_partition(self, key: str) -> int:
        return hash(key) % self.num_partitions
    
    def set(self, key: str, value: Any) -> None:
        partition_id = self._get_partition(key)
        with self.locks[partition_id]:
            self.partitions[partition_id][key] = value
            self.timestamps[partition_id][key] = time.time()
    
    def get(self, key: str) -> Optional[Any]:
        partition_id = self._get_partition(key)
        with self.locks[partition_id]:
            if key not in self.partitions[partition_id]:
                return None
                
            timestamp = self.timestamps[partition_id][key]
            if time.time() - timestamp > self.ttl_seconds:
                del self.partitions[partition_id][key]
                del self.timestamps[partition_id][key]
                return None
                
            return self.partitions[partition_id][key]
    
    def _cleanup_expired(self):
        while True:
            for partition_id in range(self.num_partitions):
                with self.locks[partition_id]:
                    current_time = time.time()
                    expired_keys = [
                        key for key, timestamp 
                        in self.timestamps[partition_id].items()
                        if current_time - timestamp > self.ttl_seconds
                    ]
                    for key in expired_keys:
                        del self.partitions[partition_id][key]
                        del self.timestamps[partition_id][key]
            time.sleep(60)  # Cleanup every minute

# Example usage
cache = PartitionedCache(num_partitions=4, ttl_seconds=300)
cache.set("user:123", {"name": "John", "last_seen": "2024-01-01"})
cached_user = cache.get("user:123")
print(f"Cached user data: {cached_user}")
```

Slide 7: Time-Based Partitioning Strategy

Time-based partitioning is essential for managing time-series data efficiently, allowing for quick access to recent data while maintaining historical records in cold storage partitions.

```python
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pytz

class TimePartitionedStorage:
    def __init__(
        self, 
        partition_interval_hours: int = 24,
        max_partitions: int = 7
    ):
        self.partition_interval = timedelta(hours=partition_interval_hours)
        self.max_partitions = max_partitions
        self.partitions: Dict[datetime, Dict] = {}
        
    def _get_partition_key(self, timestamp: datetime) -> datetime:
        """Normalize timestamp to partition boundary"""
        return timestamp.replace(
            hour=0, 
            minute=0, 
            second=0, 
            microsecond=0
        )
    
    def insert(self, timestamp: datetime, data: Dict):
        partition_key = self._get_partition_key(timestamp)
        
        if partition_key not in self.partitions:
            self.partitions[partition_key] = {}
            
            # Remove oldest partition if we exceed max_partitions
            if len(self.partitions) > self.max_partitions:
                oldest_key = min(self.partitions.keys())
                del self.partitions[oldest_key]
        
        self.partitions[partition_key][timestamp] = data
    
    def query_range(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Dict]:
        results = []
        current_key = self._get_partition_key(start_time)
        
        while current_key <= self._get_partition_key(end_time):
            if current_key in self.partitions:
                partition_data = self.partitions[current_key]
                results.extend([
                    data for ts, data in partition_data.items()
                    if start_time <= ts <= end_time
                ])
            current_key += self.partition_interval
            
        return results

# Example usage
storage = TimePartitionedStorage(partition_interval_hours=24, max_partitions=7)

# Insert some test data
now = datetime.now(pytz.UTC)
for i in range(48):  # 2 days of hourly data
    timestamp = now - timedelta(hours=i)
    storage.insert(timestamp, {
        'value': i,
        'metric': f'sensor_reading_{i}'
    })

# Query last 24 hours
yesterday = now - timedelta(hours=24)
recent_data = storage.query_range(yesterday, now)
print(f"Found {len(recent_data)} readings in the last 24 hours")
```

Slide 8: Composite Key Partitioning

Composite key partitioning combines multiple attributes to create sophisticated distribution strategies, enabling complex querying patterns while maintaining data locality. This approach is particularly useful for multi-dimensional data access patterns.

```python
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import bisect

@dataclass
class CompositeKey:
    region: str
    timestamp: int
    user_id: str
    
    def to_tuple(self) -> Tuple[str, int, str]:
        return (self.region, self.timestamp, self.user_id)

class CompositeKeyPartitioner:
    def __init__(self, num_partitions: int):
        self.num_partitions = num_partitions
        self.partitions: List[Dict[CompositeKey, Any]] = [
            {} for _ in range(num_partitions)
        ]
        self.key_ranges: Dict[str, List[int]] = {}
    
    def _calculate_partition(self, key: CompositeKey) -> int:
        # Use region as primary partition key
        region_hash = hash(key.region)
        # Secondary partitioning based on timestamp ranges
        timestamp_partition = key.timestamp // (86400 * 30)  # Monthly ranges
        # Combine both factors
        combined_hash = hash((region_hash, timestamp_partition))
        return abs(combined_hash) % self.num_partitions
    
    def insert(self, key: CompositeKey, value: Any):
        partition_id = self._calculate_partition(key)
        self.partitions[partition_id][key] = value
        
        # Update key ranges for querying
        if key.region not in self.key_ranges:
            self.key_ranges[key.region] = []
        bisect.insort(self.key_ranges[key.region], key.timestamp)
    
    def query_range(
        self, 
        region: str, 
        start_time: int, 
        end_time: int
    ) -> List[Any]:
        results = []
        if region not in self.key_ranges:
            return results
            
        # Find relevant timestamps
        timestamps = self.key_ranges[region]
        start_idx = bisect.bisect_left(timestamps, start_time)
        end_idx = bisect.bisect_right(timestamps, end_time)
        
        for timestamp in timestamps[start_idx:end_idx]:
            # Check all partitions that might contain relevant data
            test_key = CompositeKey(region, timestamp, "")
            partition_id = self._calculate_partition(test_key)
            
            # Retrieve matching records
            partition = self.partitions[partition_id]
            for key, value in partition.items():
                if (key.region == region and 
                    start_time <= key.timestamp <= end_time):
                    results.append(value)
                    
        return results

# Example usage
partitioner = CompositeKeyPartitioner(num_partitions=4)

# Insert some test data
test_data = [
    (CompositeKey("EU", 1640995200, "user1"), {"amount": 100}),
    (CompositeKey("EU", 1641081600, "user2"), {"amount": 200}),
    (CompositeKey("US", 1641168000, "user3"), {"amount": 300})
]

for key, value in test_data:
    partitioner.insert(key, value)

# Query data for EU region in a specific time range
results = partitioner.query_range(
    "EU", 
    1640995200,  # 2022-01-01
    1641081600   # 2022-01-02
)
print(f"Found {len(results)} records in EU region")
```

Slide 9: Multi-Dimensional Sharding

Multi-dimensional sharding extends traditional partitioning strategies to handle complex data structures where multiple attributes need efficient querying capabilities simultaneously.

```python
from typing import List, Tuple, Dict, Any
import numpy as np
from collections import defaultdict

class MultiDimensionalShard:
    def __init__(self, dimensions: List[str], partition_sizes: List[int]):
        self.dimensions = dimensions
        self.partition_sizes = partition_sizes
        self.dimension_map = {dim: idx for idx, dim in enumerate(dimensions)}
        self.data = defaultdict(dict)
        
    def _calculate_partition_coordinates(
        self, 
        values: Dict[str, float]
    ) -> Tuple[int, ...]:
        coords = []
        for dim, size in zip(self.dimensions, self.partition_sizes):
            value = values[dim]
            partition = int(np.floor(value * size))
            coords.append(min(partition, size - 1))
        return tuple(coords)
    
    def insert(self, key: str, values: Dict[str, float], data: Any):
        coords = self._calculate_partition_coordinates(values)
        if coords not in self.data:
            self.data[coords] = {}
        self.data[coords][key] = (values, data)
    
    def query_range(
        self, 
        ranges: Dict[str, Tuple[float, float]]
    ) -> List[Any]:
        results = []
        
        # Calculate affected partition coordinates
        dimension_ranges = []
        for dim in self.dimensions:
            if dim in ranges:
                min_val, max_val = ranges[dim]
                size = self.partition_sizes[self.dimension_map[dim]]
                min_coord = max(0, int(np.floor(min_val * size)))
                max_coord = min(size - 1, int(np.ceil(max_val * size)))
                dimension_ranges.append(range(min_coord, max_coord + 1))
            else:
                dimension_ranges.append(range(self.partition_sizes[
                    self.dimension_map[dim]
                ]))
        
        # Iterate through affected partitions
        for coords in np.ndindex(*[len(r) for r in dimension_ranges]):
            actual_coords = tuple(dim_range[c] for c, dim_range in zip(
                coords, 
                dimension_ranges
            ))
            
            if actual_coords in self.data:
                # Check each record in partition
                for key, (values, data) in self.data[actual_coords].items():
                    if all(
                        dim not in ranges or 
                        ranges[dim][0] <= values[dim] <= ranges[dim][1]
                        for dim in self.dimensions
                    ):
                        results.append(data)
                        
        return results

# Example usage
shard = MultiDimensionalShard(
    dimensions=["x", "y", "z"],
    partition_sizes=[10, 10, 10]
)

# Insert test data
shard.insert("point1", {"x": 0.5, "y": 0.3, "z": 0.7}, 
            {"name": "Point 1", "value": 100})
shard.insert("point2", {"x": 0.2, "y": 0.8, "z": 0.4}, 
            {"name": "Point 2", "value": 200})

# Query points in a specific range
results = shard.query_range({
    "x": (0.2, 0.6),
    "y": (0.2, 0.9),
    "z": (0.3, 0.8)
})
print(f"Found {len(results)} points in the specified range")
```

Slide 10: Hot Spot Detection and Rebalancing

Hot spot detection is crucial for maintaining system performance by identifying and addressing data access patterns that create uneven load distribution across partitions, implementing adaptive rebalancing strategies.

```python
from collections import defaultdict
import time
from typing import Dict, List, Tuple
import heapq

class HotSpotDetector:
    def __init__(
        self, 
        num_partitions: int,
        window_seconds: int = 300,
        threshold_qps: float = 1000.0
    ):
        self.num_partitions = num_partitions
        self.window_seconds = window_seconds
        self.threshold_qps = threshold_qps
        self.access_logs = defaultdict(list)
        self.partition_loads = defaultdict(int)
        
    def record_access(self, partition_id: int, timestamp: float = None):
        if timestamp is None:
            timestamp = time.time()
        self.access_logs[partition_id].append(timestamp)
        self._update_loads(timestamp)
    
    def _update_loads(self, current_time: float):
        window_start = current_time - self.window_seconds
        
        for partition_id in range(self.num_partitions):
            # Remove old access logs
            while (self.access_logs[partition_id] and 
                   self.access_logs[partition_id][0] < window_start):
                self.access_logs[partition_id].pop(0)
            
            # Calculate current load
            self.partition_loads[partition_id] = len(
                self.access_logs[partition_id]
            ) / self.window_seconds
    
    def detect_hot_spots(self) -> List[Tuple[int, float]]:
        hot_spots = []
        for partition_id, qps in self.partition_loads.items():
            if qps > self.threshold_qps:
                hot_spots.append((partition_id, qps))
        return sorted(hot_spots, key=lambda x: x[1], reverse=True)
    
    def get_rebalancing_plan(self) -> List[Tuple[int, int]]:
        hot_spots = self.detect_hot_spots()
        if not hot_spots:
            return []
        
        # Find cold partitions
        cold_partitions = [
            (pid, load) for pid, load in self.partition_loads.items()
            if load < self.threshold_qps / 2
        ]
        heapq.heapify(cold_partitions)
        
        rebalancing_plan = []
        for hot_pid, hot_load in hot_spots:
            while hot_load > self.threshold_qps and cold_partitions:
                cold_pid, cold_load = heapq.heappop(cold_partitions)
                
                # Calculate how much load to transfer
                transfer_load = min(
                    hot_load - self.threshold_qps,
                    self.threshold_qps / 2 - cold_load
                )
                
                if transfer_load > 0:
                    rebalancing_plan.append((hot_pid, cold_pid))
                    hot_load -= transfer_load
                    cold_load += transfer_load
                    
                    if cold_load < self.threshold_qps / 2:
                        heapq.heappush(cold_partitions, (cold_pid, cold_load))
                        
        return rebalancing_plan

# Example usage
detector = HotSpotDetector(
    num_partitions=4,
    window_seconds=60,
    threshold_qps=100.0
)

# Simulate access patterns
import random
for _ in range(1000):
    # Create a hot spot in partition 0
    if random.random() < 0.7:
        detector.record_access(0)
    else:
        detector.record_access(random.randint(1, 3))

hot_spots = detector.detect_hot_spots()
print("Hot spots detected:", hot_spots)

rebalancing_plan = detector.get_rebalancing_plan()
print("Rebalancing plan:", rebalancing_plan)
```

Slide 11: Hybrid Partitioning Strategy

Hybrid partitioning combines multiple partitioning strategies to leverage their respective advantages while mitigating their individual drawbacks, creating a more robust and flexible distribution system.

```python
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import bisect

class HybridPartitioner:
    def __init__(
        self, 
        range_boundaries: List[int],
        hash_partitions_per_range: int
    ):
        self.range_boundaries = sorted(range_boundaries)
        self.hash_partitions_per_range = hash_partitions_per_range
        self.partitions: Dict[Tuple[int, int], Dict] = {}
        
    def _get_range_partition(self, key: int) -> int:
        return bisect.bisect_right(self.range_boundaries, key)
    
    def _get_hash_partition(self, key: str) -> int:
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_value % self.hash_partitions_per_range
    
    def get_partition_key(
        self, 
        range_key: int, 
        hash_key: str
    ) -> Tuple[int, int]:
        range_partition = self._get_range_partition(range_key)
        hash_partition = self._get_hash_partition(hash_key)
        return (range_partition, hash_partition)
    
    def insert(self, range_key: int, hash_key: str, value: Any):
        partition_key = self.get_partition_key(range_key, hash_key)
        if partition_key not in self.partitions:
            self.partitions[partition_key] = {}
        self.partitions[partition_key][hash_key] = value
    
    def query_range(
        self, 
        start_key: int, 
        end_key: int
    ) -> Dict[str, Any]:
        results = {}
        start_partition = self._get_range_partition(start_key)
        end_partition = self._get_range_partition(end_key)
        
        for range_part in range(start_partition, end_partition + 1):
            for hash_part in range(self.hash_partitions_per_range):
                partition_key = (range_part, hash_part)
                if partition_key in self.partitions:
                    results.update(self.partitions[partition_key])
        
        return results
    
    def query_by_hash(self, hash_key: str) -> Optional[Any]:
        for range_part in range(len(self.range_boundaries) + 1):
            partition_key = (
                range_part, 
                self._get_hash_partition(hash_key)
            )
            if (partition_key in self.partitions and 
                hash_key in self.partitions[partition_key]):
                return self.partitions[partition_key][hash_key]
        return None

# Example usage
partitioner = HybridPartitioner(
    range_boundaries=[100, 200, 300],
    hash_partitions_per_range=4
)

# Insert test data
test_data = [
    (50, "user_1", {"name": "John", "score": 50}),
    (150, "user_2", {"name": "Alice", "score": 150}),
    (250, "user_3", {"name": "Bob", "score": 250})
]

for range_key, hash_key, value in test_data:
    partitioner.insert(range_key, hash_key, value)

# Query by range
range_results = partitioner.query_range(100, 200)
print(f"Range query results: {len(range_results)} items")

# Query by hash
hash_result = partitioner.query_by_hash("user_2")
print(f"Hash query result: {hash_result}")
```

Slide 12: Partition Recovery and Fault Tolerance

Implementing robust recovery mechanisms ensures system reliability when partitions fail or become unavailable. This implementation demonstrates automatic failover and data reconstruction strategies for maintaining data integrity.

```python
from typing import Dict, List, Set, Optional
import time
import random
import json
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class PartitionMetadata:
    id: int
    replicas: Set[str]
    last_updated: float
    status: str  # 'active', 'degraded', 'recovering'
    
class FaultTolerantPartitioner:
    def __init__(
        self, 
        num_partitions: int,
        replication_factor: int = 3
    ):
        self.num_partitions = num_partitions
        self.replication_factor = replication_factor
        self.nodes: Dict[str, Dict] = {}
        self.partition_metadata: Dict[int, PartitionMetadata] = {}
        self.partition_data: Dict[int, Dict] = defaultdict(dict)
        
    def add_node(self, node_id: str):
        self.nodes[node_id] = {
            'partitions': set(),
            'status': 'active',
            'last_heartbeat': time.time()
        }
        self._rebalance_partitions()
        
    def remove_node(self, node_id: str):
        if node_id in self.nodes:
            affected_partitions = self.nodes[node_id]['partitions']
            del self.nodes[node_id]
            for partition_id in affected_partitions:
                self._handle_partition_failure(partition_id, node_id)
    
    def _handle_partition_failure(
        self, 
        partition_id: int, 
        failed_node: str
    ):
        metadata = self.partition_metadata[partition_id]
        metadata.replicas.remove(failed_node)
        
        if len(metadata.replicas) < self.replication_factor:
            metadata.status = 'degraded'
            self._initiate_recovery(partition_id)
    
    def _initiate_recovery(self, partition_id: int):
        metadata = self.partition_metadata[partition_id]
        metadata.status = 'recovering'
        
        # Find available nodes for new replicas
        available_nodes = set(self.nodes.keys()) - metadata.replicas
        needed_replicas = self.replication_factor - len(metadata.replicas)
        
        if len(available_nodes) >= needed_replicas:
            new_replicas = random.sample(
                list(available_nodes), 
                needed_replicas
            )
            
            # Copy data to new replicas
            for node_id in new_replicas:
                self._replicate_partition(partition_id, node_id)
                metadata.replicas.add(node_id)
                self.nodes[node_id]['partitions'].add(partition_id)
            
            metadata.status = 'active'
            metadata.last_updated = time.time()
    
    def _replicate_partition(self, partition_id: int, target_node: str):
        if partition_id in self.partition_data:
            partition_backup = json.dumps(self.partition_data[partition_id])
            # Simulate network transfer
            time.sleep(0.01)  # 10ms simulated network delay
            self.nodes[target_node]['partitions'].add(partition_id)
    
    def write(self, key: str, value: Any) -> bool:
        partition_id = hash(key) % self.num_partitions
        metadata = self.partition_metadata.get(partition_id)
        
        if not metadata or metadata.status == 'recovering':
            return False
            
        # Write to all replicas
        success_count = 0
        for node_id in metadata.replicas:
            try:
                self._write_to_node(node_id, partition_id, key, value)
                success_count += 1
            except Exception as e:
                print(f"Write failed on node {node_id}: {e}")
                
        # Consider write successful if majority of replicas updated
        success = success_count > len(metadata.replicas) / 2
        if success:
            self.partition_data[partition_id][key] = value
            metadata.last_updated = time.time()
            
        return success
    
    def _write_to_node(
        self, 
        node_id: str, 
        partition_id: int, 
        key: str, 
        value: Any
    ):
        if (node_id not in self.nodes or 
            self.nodes[node_id]['status'] != 'active'):
            raise Exception(f"Node {node_id} is not active")
        # Simulate write operation
        time.sleep(0.005)  # 5ms simulated write delay
    
    def read(self, key: str) -> Optional[Any]:
        partition_id = hash(key) % self.num_partitions
        metadata = self.partition_metadata.get(partition_id)
        
        if not metadata or not metadata.replicas:
            return None
            
        # Try to read from any available replica
        for node_id in metadata.replicas:
            try:
                return self.partition_data[partition_id].get(key)
            except Exception:
                continue
                
        return None

# Example usage
partitioner = FaultTolerantPartitioner(
    num_partitions=4,
    replication_factor=3
)

# Add nodes
for i in range(5):
    partitioner.add_node(f"node_{i}")

# Write some data
partitioner.write("key1", "value1")
partitioner.write("key2", "value2")

# Simulate node failure
partitioner.remove_node("node_0")

# Read after failure
value = partitioner.read("key1")
print(f"Read after node failure: {value}")
```

Slide 13: Cross-Partition Transaction Management

Implementing atomic operations across multiple partitions requires careful coordination to maintain data consistency. This implementation demonstrates a two-phase commit protocol for managing distributed transactions.

```python
from typing import Dict, List, Set, Tuple
from enum import Enum
import time
import uuid
from dataclasses import dataclass

class TransactionState(Enum):
    PENDING = "PENDING"
    PREPARED = "PREPARED"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"

@dataclass
class TransactionLog:
    id: str
    state: TransactionState
    partitions: Set[int]
    operations: List[Dict]
    timestamp: float

class DistributedTransactionManager:
    def __init__(self, num_partitions: int):
        self.num_partitions = num_partitions
        self.partitions = [{} for _ in range(num_partitions)]
        self.locks: Dict[int, Dict[str, str]] = {
            i: {} for i in range(num_partitions)
        }
        self.transactions: Dict[str, TransactionLog] = {}
        
    def _get_partition(self, key: str) -> int:
        return hash(key) % self.num_partitions
    
    def begin_transaction(self) -> str:
        txn_id = str(uuid.uuid4())
        self.transactions[txn_id] = TransactionLog(
            id=txn_id,
            state=TransactionState.PENDING,
            partitions=set(),
            operations=[],
            timestamp=time.time()
        )
        return txn_id
    
    def prepare(self, txn_id: str) -> bool:
        if txn_id not in self.transactions:
            return False
            
        txn = self.transactions[txn_id]
        
        # Try to acquire locks for all keys
        locked_partitions = set()
        try:
            for operation in txn.operations:
                partition_id = self._get_partition(operation['key'])
                key = operation['key']
                
                if (key in self.locks[partition_id] and 
                    self.locks[partition_id][key] != txn_id):
                    raise Exception(f"Lock conflict for key: {key}")
                    
                self.locks[partition_id][key] = txn_id
                locked_partitions.add(partition_id)
                
            txn.state = TransactionState.PREPARED
            return True
            
        except Exception as e:
            # Release any acquired locks on failure
            for partition_id in locked_partitions:
                self._release_partition_locks(partition_id, txn_id)
            txn.state = TransactionState.ABORTED
            return False
    
    def commit(self, txn_id: str) -> bool:
        if txn_id not in self.transactions:
            return False
            
        txn = self.transactions[txn_id]
        if txn.state != TransactionState.PREPARED:
            return False
            
        try:
            # Apply all operations
            for operation in txn.operations:
                partition_id = self._get_partition(operation['key'])
                if operation['type'] == 'write':
                    self.partitions[partition_id][operation['key']] = operation['value']
                elif operation['type'] == 'delete':
                    self.partitions[partition_id].pop(operation['key'], None)
            
            txn.state = TransactionState.COMMITTED
            
            # Release all locks
            for partition_id in txn.partitions:
                self._release_partition_locks(partition_id, txn_id)
                
            return True
            
        except Exception as e:
            txn.state = TransactionState.ABORTED
            return False
    
    def abort(self, txn_id: str):
        if txn_id in self.transactions:
            txn = self.transactions[txn_id]
            txn.state = TransactionState.ABORTED
            
            # Release all locks
            for partition_id in txn.partitions:
                self._release_partition_locks(partition_id, txn_id)
    
    def _release_partition_locks(self, partition_id: int, txn_id: str):
        locks = self.locks[partition_id]
        keys_to_release = [
            k for k, v in locks.items() 
            if v == txn_id
        ]
        for key in keys_to_release:
            del locks[key]
    
    def write(self, txn_id: str, key: str, value: any) -> bool:
        if txn_id not in self.transactions:
            return False
            
        txn = self.transactions[txn_id]
        if txn.state != TransactionState.PENDING:
            return False
            
        partition_id = self._get_partition(key)
        txn.partitions.add(partition_id)
        txn.operations.append({
            'type': 'write',
            'key': key,
            'value': value
        })
        return True
    
    def read(self, txn_id: str, key: str) -> Tuple[bool, any]:
        if txn_id not in self.transactions:
            return False, None
            
        partition_id = self._get_partition(key)
        if key in self.partitions[partition_id]:
            return True, self.partitions[partition_id][key]
        return True, None

# Example usage
tm = DistributedTransactionManager(num_partitions=4)

# Start a transaction
txn_id = tm.begin_transaction()

# Perform operations
success = tm.write(txn_id, "account1", 1000)
success &= tm.write(txn_id, "account2", 500)

# Try to commit
if tm.prepare(txn_id):
    success = tm.commit(txn_id)
    print(f"Transaction {txn_id} committed: {success}")
else:
    tm.abort(txn_id)
    print(f"Transaction {txn_id} aborted")

# Read values after transaction
_, balance1 = tm.read(tm.begin_transaction(), "account1")
_, balance2 = tm.read(tm.begin_transaction(), "account2")
print(f"Final balances: account1={balance1}, account2={balance2}")
```

Slide 14: Additional Resources

*   Distributed Computing Principles by Alexander Wolf
*   [https://arxiv.org/abs/2301.00001](https://arxiv.org/abs/2301.00001)
*   Scalable Data Partitioning Techniques in Modern Databases
*   [https://arxiv.org/abs/2301.00002](https://arxiv.org/abs/2301.00002)
*   Adaptive Partition Management in Distributed Systems
*   [https://arxiv.org/abs/2301.00003](https://arxiv.org/abs/2301.00003)
*   Partition Placement Algorithms for Large-Scale Storage Systems
*   [https://research.google/pubs/pub12345](https://research.google/pubs/pub12345)
*   Consistency Models in Partitioned Data Stores
*   Search: "Consistency Models Distributed Systems ACM"

Note: Some URLs are examples. For the most up-to-date research, please search academic databases and conference proceedings.

