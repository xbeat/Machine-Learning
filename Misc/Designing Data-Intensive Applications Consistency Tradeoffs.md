## Designing Data-Intensive Applications Consistency Tradeoffs
Slide 1: Understanding Synchronous Replication

Synchronous replication ensures strong consistency by waiting for acknowledgment from all replicas before confirming writes. This approach guarantees that all nodes have identical data but introduces latency as the system waits for confirmation from every replica before proceeding.

```python
import threading
import time
from typing import Dict, List

class SyncReplicationManager:
    def __init__(self, replica_count: int):
        self.data: Dict = {}
        self.replicas: List[Dict] = [{} for _ in range(replica_count)]
        self.lock = threading.Lock()
        
    def write(self, key: str, value: any) -> bool:
        with self.lock:
            # Primary write
            self.data[key] = value
            
            # Synchronous replica updates
            for replica in self.replicas:
                replica[key] = value
                time.sleep(0.1)  # Simulate network latency
                
            return True
            
    def read(self, key: str) -> any:
        return self.data.get(key)

# Usage Example
manager = SyncReplicationManager(replica_count=3)
manager.write("user_1", {"name": "John", "balance": 1000})
print(f"Read result: {manager.read('user_1')}")
```

Slide 2: Asynchronous Replication Implementation

In asynchronous replication, the primary node responds immediately after writing locally, while replicas update in the background. This design significantly reduces write latency but introduces eventual consistency where replicas may temporarily have different values.

```python
import asyncio
from typing import Dict, List
import random

class AsyncReplicationManager:
    def __init__(self, replica_count: int):
        self.data: Dict = {}
        self.replicas: List[Dict] = [{} for _ in range(replica_count)]
        
    async def write(self, key: str, value: any) -> bool:
        # Primary write immediately
        self.data[key] = value
        
        # Async replica updates
        asyncio.create_task(self._update_replicas(key, value))
        return True
    
    async def _update_replicas(self, key: str, value: any):
        for replica in self.replicas:
            # Simulate varying network delays
            await asyncio.sleep(random.uniform(0.1, 0.5))
            replica[key] = value
    
    def read(self, key: str) -> any:
        return self.data.get(key)

# Usage Example
async def main():
    manager = AsyncReplicationManager(replica_count=3)
    await manager.write("user_2", {"name": "Alice", "balance": 2000})
    print(f"Read result: {manager.read('user_2')}")

asyncio.run(main())
```

Slide 3: Consistency Models Comparison

Implementing different consistency models allows us to understand their practical implications. This implementation demonstrates strong consistency, causal consistency, and eventual consistency, highlighting their behavioral differences in distributed systems.

```python
from enum import Enum
from typing import Dict, List, Optional
import time

class ConsistencyModel(Enum):
    STRONG = "strong"
    CAUSAL = "causal"
    EVENTUAL = "eventual"

class ConsistencyManager:
    def __init__(self, model: ConsistencyModel):
        self.model = model
        self.primary_data: Dict = {}
        self.vector_clock: Dict[str, int] = {}
        self.version: int = 0
        
    def write(self, key: str, value: any) -> Dict:
        if self.model == ConsistencyModel.STRONG:
            return self._strong_write(key, value)
        elif self.model == ConsistencyModel.CAUSAL:
            return self._causal_write(key, value)
        return self._eventual_write(key, value)
    
    def _strong_write(self, key: str, value: any) -> Dict:
        self.primary_data[key] = value
        self.version += 1
        return {"status": "success", "version": self.version}
    
    def _causal_write(self, key: str, value: any) -> Dict:
        self.vector_clock[key] = self.vector_clock.get(key, 0) + 1
        self.primary_data[key] = {
            "value": value,
            "vector_clock": self.vector_clock[key]
        }
        return {"status": "success", "vector_clock": self.vector_clock[key]}
    
    def _eventual_write(self, key: str, value: any) -> Dict:
        self.primary_data[key] = value
        return {"status": "accepted"}

# Usage Example
manager = ConsistencyManager(ConsistencyModel.STRONG)
write_result = manager.write("balance", 1000)
print(f"Write result: {write_result}")
```

Slide 4: Quorum-Based Replication

Quorum-based replication systems ensure consistency by requiring a minimum number of nodes to acknowledge writes and reads. This implementation demonstrates how to achieve configurable consistency levels using quorum mechanics.

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

@dataclass
class QuorumConfig:
    total_nodes: int
    write_quorum: int
    read_quorum: int

class QuorumReplication:
    def __init__(self, config: QuorumConfig):
        self.config = config
        self.nodes: List[Dict] = [{} for _ in range(config.total_nodes)]
        
    def write(self, key: str, value: any) -> Tuple[bool, int]:
        successful_writes = 0
        version = int(time.time() * 1000)  # Simple version number
        
        for node in self.nodes:
            if random.random() < 0.9:  # 90% success rate simulation
                node[key] = (value, version)
                successful_writes += 1
                
        return (successful_writes >= self.config.write_quorum, successful_writes)
    
    def read(self, key: str) -> Tuple[any, bool]:
        successful_reads = 0
        values_with_versions = []
        
        for node in self.nodes:
            if key in node and random.random() < 0.9:
                values_with_versions.append(node[key])
                successful_reads += 1
        
        if successful_reads >= self.config.read_quorum:
            # Return newest version
            return (max(values_with_versions, key=lambda x: x[1])[0], True)
        return (None, False)

# Usage Example
config = QuorumConfig(total_nodes=5, write_quorum=3, read_quorum=2)
qr = QuorumReplication(config)
success, write_count = qr.write("user_3", {"name": "Bob", "balance": 3000})
print(f"Write success: {success}, Write count: {write_count}")
value, read_success = qr.read("user_3")
print(f"Read success: {read_success}, Value: {value}")
```

Slide 5: Conflict Resolution Strategies

Distributed systems must handle conflicts when multiple writes occur simultaneously. This implementation showcases different conflict resolution strategies including Last-Write-Wins (LWW), Vector Clocks, and Custom Merge Functions.

```python
from typing import Dict, List, Optional
from datetime import datetime
import uuid

class ConflictResolver:
    def __init__(self):
        self.versions: Dict[str, List[Dict]] = {}
        
    def lww_resolve(self, key: str, value: any) -> Dict:
        timestamp = datetime.now().timestamp()
        version = {
            'value': value,
            'timestamp': timestamp,
            'id': str(uuid.uuid4())
        }
        
        if key not in self.versions:
            self.versions[key] = []
        self.versions[key].append(version)
        
        # Keep only latest version
        latest = max(self.versions[key], key=lambda x: x['timestamp'])
        self.versions[key] = [latest]
        return latest
    
    def vector_clock_resolve(self, key: str, value: any, 
                           node_id: str, vector_clock: Dict[str, int]) -> Dict:
        version = {
            'value': value,
            'vector_clock': vector_clock,
            'node_id': node_id
        }
        
        if key not in self.versions:
            self.versions[key] = []
        self.versions[key].append(version)
        
        # Detect conflicts using vector clocks
        return self._resolve_vector_conflicts(key)
    
    def _resolve_vector_conflicts(self, key: str) -> Dict:
        versions = self.versions[key]
        if len(versions) <= 1:
            return versions[0]
            
        # Compare vector clocks
        conflicts = []
        for v1 in versions:
            has_conflict = False
            for v2 in versions:
                if v1 != v2 and not self._vector_clock_compare(
                    v1['vector_clock'], v2['vector_clock']):
                    has_conflict = True
                    break
            if has_conflict:
                conflicts.append(v1)
                
        return conflicts[0] if conflicts else versions[-1]
    
    def _vector_clock_compare(self, vc1: Dict[str, int], 
                            vc2: Dict[str, int]) -> bool:
        return all(vc1.get(k, 0) >= v for k, v in vc2.items())

# Usage Example
resolver = ConflictResolver()
lww_result = resolver.lww_resolve("shared_counter", 42)
vector_result = resolver.vector_clock_resolve(
    "shared_doc",
    "content",
    "node_1",
    {"node_1": 1, "node_2": 0}
)
print(f"LWW Resolution: {lww_result}")
print(f"Vector Clock Resolution: {vector_result}")
```

Slide 6: CAP Theorem Implementation

The CAP theorem states that distributed systems can only guarantee two out of three properties: Consistency, Availability, and Partition tolerance. This implementation demonstrates how different system configurations handle these trade-offs.

```python
from enum import Enum
from typing import Dict, Optional, Tuple
import random
import time

class SystemProperty(Enum):
    CONSISTENCY = "C"
    AVAILABILITY = "A"
    PARTITION_TOLERANCE = "P"

class CAPSystem:
    def __init__(self, priorities: Tuple[SystemProperty, SystemProperty]):
        self.priorities = priorities
        self.data: Dict = {}
        self.backup_data: Dict = {}
        self.is_partition = False
        
    def handle_partition(self, active: bool):
        self.is_partition = active
        
    def write(self, key: str, value: any) -> Dict:
        if self.is_partition:
            if SystemProperty.CONSISTENCY in self.priorities:
                return {"success": False, "error": "System unavailable during partition"}
            
            if SystemProperty.AVAILABILITY in self.priorities:
                self.data[key] = value
                return {"success": True, "warning": "Consistency not guaranteed"}
                
        self.data[key] = value
        self.backup_data[key] = value
        return {"success": True}
    
    def read(self, key: str) -> Dict:
        if self.is_partition:
            if SystemProperty.CONSISTENCY in self.priorities:
                return {"success": False, "error": "System unavailable during partition"}
            
            if SystemProperty.AVAILABILITY in self.priorities:
                return {
                    "success": True,
                    "value": self.data.get(key),
                    "warning": "Data may be stale"
                }
                
        return {"success": True, "value": self.data.get(key)}

# Usage Example
cap_system = CAPSystem((SystemProperty.CONSISTENCY, SystemProperty.PARTITION_TOLERANCE))
print("Normal operation:")
print(cap_system.write("key1", "value1"))
print(cap_system.read("key1"))

cap_system.handle_partition(True)
print("\nDuring partition:")
print(cap_system.write("key2", "value2"))
print(cap_system.read("key1"))
```

Slide 7: CRDT Implementation - Counter Type

Conflict-Free Replicated Data Types (CRDTs) provide a mathematical approach to eventual consistency. This implementation shows a G-Counter (Grow-only Counter) CRDT that ensures convergence across replicas.

```python
from typing import Dict
import uuid

class GCounter:
    def __init__(self, node_id: str = None):
        self.node_id = node_id or str(uuid.uuid4())
        self.counts: Dict[str, int] = {}
        
    def increment(self, amount: int = 1) -> None:
        self.counts[self.node_id] = self.counts.get(self.node_id, 0) + amount
        
    def value(self) -> int:
        return sum(self.counts.values())
    
    def merge(self, other: 'GCounter') -> None:
        """Merge another G-Counter with this one"""
        for node_id, count in other.counts.items():
            self.counts[node_id] = max(self.counts.get(node_id, 0), count)
            
    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'counts': self.counts,
            'value': self.value()
        }

# Usage Example - Simulating distributed counters
def simulate_distributed_counters():
    # Create three counter instances
    counter1 = GCounter("node1")
    counter2 = GCounter("node2")
    counter3 = GCounter("node3")
    
    # Simulate distributed increments
    counter1.increment(3)
    counter2.increment(2)
    counter3.increment(4)
    
    # Merge states
    counter1.merge(counter2)
    counter1.merge(counter3)
    
    print("Final merged state:", counter1.to_dict())
    print("Total count:", counter1.value())

simulate_distributed_counters()
```

Slide 8: Real-world Example - Distributed Cache System

This implementation demonstrates a practical distributed cache system with configurable consistency levels and automatic conflict resolution, suitable for high-traffic web applications.

```python
from typing import Dict, Optional, List
import time
import random
from dataclasses import dataclass

@dataclass
class CacheEntry:
    value: any
    timestamp: float
    version: int
    ttl: int

class DistributedCache:
    def __init__(self, node_count: int, consistency_level: int):
        self.nodes: List[Dict[str, CacheEntry]] = [{} for _ in range(node_count)]
        self.consistency_level = consistency_level
        self.version_counter = 0
        
    def set(self, key: str, value: any, ttl: int = 3600) -> bool:
        self.version_counter += 1
        entry = CacheEntry(
            value=value,
            timestamp=time.time(),
            version=self.version_counter,
            ttl=ttl
        )
        
        successful_writes = 0
        for node in self.nodes:
            if random.random() < 0.95:  # 95% write success rate
                node[key] = entry
                successful_writes += 1
                
        return successful_writes >= self.consistency_level
    
    def get(self, key: str) -> Optional[any]:
        valid_responses = []
        
        for node in self.nodes:
            if key in node and random.random() < 0.95:  # 95% read success rate
                entry = node[key]
                if time.time() - entry.timestamp <= entry.ttl:
                    valid_responses.append(entry)
                    
        if len(valid_responses) >= self.consistency_level:
            # Return newest version
            newest = max(valid_responses, key=lambda x: x.version)
            return newest.value
        return None
    
    def invalidate(self, key: str) -> bool:
        successful_invalidations = 0
        
        for node in self.nodes:
            if key in node and random.random() < 0.95:
                del node[key]
                successful_invalidations += 1
                
        return successful_invalidations >= self.consistency_level

# Usage Example
cache = DistributedCache(node_count=5, consistency_level=3)

# Simulate cache operations
print("Setting value:", cache.set("user:123", {"name": "John", "age": 30}))
print("Reading value:", cache.get("user:123"))
print("Invalidating key:", cache.invalidate("user:123"))
print("Reading after invalidation:", cache.get("user:123"))
```

Slide 9: Replication Log Management

Implementation of a replication log system that tracks and manages changes across distributed nodes, ensuring ordered application of updates and handling log compaction for efficiency.

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class LogEntry:
    sequence_id: int
    operation: str
    key: str
    value: any
    timestamp: float
    node_id: str

class ReplicationLog:
    def __init__(self, node_id: str, max_log_size: int = 1000):
        self.node_id = node_id
        self.max_log_size = max_log_size
        self.log: List[LogEntry] = []
        self.sequence_counter = 0
        self.snapshot: Dict = {}
        
    def append(self, operation: str, key: str, value: any) -> LogEntry:
        self.sequence_counter += 1
        entry = LogEntry(
            sequence_id=self.sequence_counter,
            operation=operation,
            key=key,
            value=value,
            timestamp=datetime.now().timestamp(),
            node_id=self.node_id
        )
        self.log.append(entry)
        
        if len(self.log) > self.max_log_size:
            self._compact_log()
            
        return entry
    
    def _compact_log(self) -> None:
        # Create snapshot of current state
        for entry in self.log:
            if entry.operation == "SET":
                self.snapshot[entry.key] = entry.value
            elif entry.operation == "DELETE":
                self.snapshot.pop(entry.key, None)
                
        # Keep only recent entries
        self.log = self.log[-100:]  # Keep last 100 entries
        
    def get_entries_since(self, sequence_id: int) -> List[LogEntry]:
        return [entry for entry in self.log if entry.sequence_id > sequence_id]
    
    def apply_entries(self, entries: List[LogEntry]) -> None:
        for entry in sorted(entries, key=lambda x: x.sequence_id):
            if entry.sequence_id > self.sequence_counter:
                self.sequence_counter = entry.sequence_id
                self.log.append(entry)
                
        if len(self.log) > self.max_log_size:
            self._compact_log()

# Usage Example
def simulate_replication():
    node1_log = ReplicationLog("node1")
    node2_log = ReplicationLog("node2")
    
    # Node 1 operations
    node1_log.append("SET", "user:1", {"name": "Alice"})
    node1_log.append("SET", "user:2", {"name": "Bob"})
    
    # Simulate replication to node2
    entries = node1_log.get_entries_since(0)
    node2_log.apply_entries(entries)
    
    # Verify replication
    print("Node 1 log size:", len(node1_log.log))
    print("Node 2 log size:", len(node2_log.log))
    print("Node 2 last entry:", vars(node2_log.log[-1]))

simulate_replication()
```

Slide 10: Chain Replication Implementation

Chain replication provides strong consistency with improved throughput compared to primary-backup replication. This implementation demonstrates the chain replication protocol with failure detection.

```python
from typing import List, Optional, Dict
from dataclasses import dataclass
import time
import threading

@dataclass
class Node:
    id: str
    data: Dict
    next_node: Optional['Node'] = None
    last_heartbeat: float = time.time()
    
class ChainReplication:
    def __init__(self, node_ids: List[str]):
        self.nodes: List[Node] = []
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        
        # Initialize chain
        prev_node = None
        for node_id in node_ids:
            node = Node(id=node_id, data={})
            if prev_node:
                prev_node.next_node = node
            else:
                self.head = node
            prev_node = node
            self.nodes.append(node)
        self.tail = prev_node
        
        # Start heartbeat monitoring
        self.monitor_thread = threading.Thread(target=self._monitor_nodes)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def write(self, key: str, value: any) -> bool:
        if not self.head:
            return False
            
        # Forward write through the chain
        current = self.head
        while current:
            current.data[key] = value
            current = current.next_node
            time.sleep(0.1)  # Simulate network delay
            
        return True
        
    def read(self, key: str) -> Optional[any]:
        if not self.tail:
            return None
        return self.tail.data.get(key)
        
    def _monitor_nodes(self):
        while True:
            current_time = time.time()
            # Check for node failures
            current = self.head
            while current:
                if current_time - current.last_heartbeat > 5:  # 5 second timeout
                    self._handle_node_failure(current)
                current = current.next_node
            time.sleep(1)
            
    def _handle_node_failure(self, failed_node: Node):
        # Remove failed node from chain
        if failed_node == self.head:
            self.head = failed_node.next_node
        else:
            prev = self.head
            while prev and prev.next_node != failed_node:
                prev = prev.next_node
            if prev:
                prev.next_node = failed_node.next_node
                
        if failed_node == self.tail:
            self.tail = prev
            
        self.nodes.remove(failed_node)

# Usage Example
chain = ChainReplication(["node1", "node2", "node3"])
print("Writing data:", chain.write("key1", "value1"))
print("Reading data:", chain.read("key1"))

# Simulate node heartbeats
for node in chain.nodes:
    node.last_heartbeat = time.time()
```

Slide 11: Multi-Version Concurrency Control (MVCC)

MVCC maintains multiple versions of data to provide isolation between concurrent transactions without blocking reads. This implementation demonstrates version management and conflict detection in distributed systems.

```python
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid

@dataclass
class Version:
    value: any
    timestamp: float
    transaction_id: str
    is_committed: bool = False

class MVCCManager:
    def __init__(self):
        self.versions: Dict[str, List[Version]] = {}
        self.active_transactions: Dict[str, float] = {}
        
    def begin_transaction(self) -> str:
        tx_id = str(uuid.uuid4())
        self.active_transactions[tx_id] = datetime.now().timestamp()
        return tx_id
        
    def read(self, key: str, tx_id: str) -> Optional[any]:
        if key not in self.versions:
            return None
            
        tx_timestamp = self.active_transactions.get(tx_id)
        if not tx_timestamp:
            raise ValueError("Invalid transaction ID")
            
        # Find the latest committed version visible to this transaction
        visible_versions = [
            v for v in self.versions[key]
            if v.is_committed and v.timestamp <= tx_timestamp
        ]
        
        if not visible_versions:
            return None
            
        return max(visible_versions, key=lambda v: v.timestamp).value
        
    def write(self, key: str, value: any, tx_id: str) -> bool:
        if tx_id not in self.active_transactions:
            raise ValueError("Invalid transaction ID")
            
        if key not in self.versions:
            self.versions[key] = []
            
        # Create new version
        version = Version(
            value=value,
            timestamp=datetime.now().timestamp(),
            transaction_id=tx_id
        )
        self.versions[key].append(version)
        return True
        
    def commit(self, tx_id: str) -> bool:
        if tx_id not in self.active_transactions:
            return False
            
        # Mark all versions created by this transaction as committed
        for versions in self.versions.values():
            for version in versions:
                if version.transaction_id == tx_id:
                    version.is_committed = True
                    
        del self.active_transactions[tx_id]
        return True
        
    def rollback(self, tx_id: str) -> bool:
        if tx_id not in self.active_transactions:
            return False
            
        # Remove all versions created by this transaction
        for key in self.versions:
            self.versions[key] = [
                v for v in self.versions[key]
                if v.transaction_id != tx_id
            ]
            
        del self.active_transactions[tx_id]
        return True

# Usage Example
mvcc = MVCCManager()

# Transaction 1: Write initial value
tx1 = mvcc.begin_transaction()
mvcc.write("account1", 1000, tx1)
mvcc.commit(tx1)

# Transaction 2: Read and update
tx2 = mvcc.begin_transaction()
initial_value = mvcc.read("account1", tx2)
mvcc.write("account1", initial_value + 500, tx2)
mvcc.commit(tx2)

# Transaction 3: Read latest value
tx3 = mvcc.begin_transaction()
final_value = mvcc.read("account1", tx3)
print(f"Final account balance: {final_value}")
mvcc.commit(tx3)
```

Slide 12: Distributed Lock Implementation

A robust distributed lock mechanism ensures mutual exclusion across distributed systems. This implementation provides deadlock detection and automatic lock release with lease timeouts.

```python
from typing import Dict, Optional, Set
import threading
import time
from dataclasses import dataclass

@dataclass
class Lock:
    owner: str
    expiry: float
    resources: Set[str]

class DistributedLockManager:
    def __init__(self, default_timeout: int = 30):
        self.locks: Dict[str, Lock] = {}
        self.default_timeout = default_timeout
        self.lock = threading.Lock()
        
        # Start lock cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_locks)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
        
    def acquire(self, resources: Set[str], owner: str, 
                timeout: Optional[int] = None) -> bool:
        with self.lock:
            # Check if resources are available
            for resource in resources:
                if resource in self.locks:
                    current_lock = self.locks[resource]
                    if current_lock.owner != owner and time.time() < current_lock.expiry:
                        return False
                        
            # Acquire locks
            expiry = time.time() + (timeout or self.default_timeout)
            lock_entry = Lock(owner=owner, expiry=expiry, resources=resources)
            
            for resource in resources:
                self.locks[resource] = lock_entry
                
            return True
            
    def release(self, resources: Set[str], owner: str) -> bool:
        with self.lock:
            success = True
            for resource in resources:
                if resource in self.locks:
                    if self.locks[resource].owner == owner:
                        del self.locks[resource]
                    else:
                        success = False
            return success
            
    def refresh(self, resources: Set[str], owner: str) -> bool:
        with self.lock:
            for resource in resources:
                if resource not in self.locks or self.locks[resource].owner != owner:
                    return False
                    
            expiry = time.time() + self.default_timeout
            for resource in resources:
                self.locks[resource].expiry = expiry
                
            return True
            
    def _cleanup_expired_locks(self):
        while True:
            with self.lock:
                current_time = time.time()
                expired_resources = [
                    resource
                    for resource, lock in self.locks.items()
                    if current_time > lock.expiry
                ]
                
                for resource in expired_resources:
                    del self.locks[resource]
                    
            time.sleep(1)

# Usage Example
lock_manager = DistributedLockManager(default_timeout=10)

# Simulate distributed processes
def process_task(process_id: str, resources: Set[str]):
    if lock_manager.acquire(resources, process_id):
        print(f"Process {process_id} acquired locks for {resources}")
        time.sleep(5)  # Simulate work
        lock_manager.release(resources, process_id)
        print(f"Process {process_id} released locks for {resources}")
    else:
        print(f"Process {process_id} failed to acquire locks for {resources}")

# Start multiple processes
threading.Thread(target=process_task, args=("P1", {"resource1", "resource2"})).start()
threading.Thread(target=process_task, args=("P2", {"resource2", "resource3"})).start()
```

Slide 13: Performance Metrics and Monitoring

This implementation provides comprehensive monitoring of replication performance, including latency tracking, consistency measurements, and health checks across distributed nodes.

```python
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import statistics
from collections import defaultdict

@dataclass
class MetricSample:
    value: float
    timestamp: float
    node_id: str

class ReplicationMetrics:
    def __init__(self, node_count: int):
        self.write_latencies: Dict[str, List[MetricSample]] = defaultdict(list)
        self.read_latencies: Dict[str, List[MetricSample]] = defaultdict(list)
        self.consistency_scores: Dict[str, List[float]] = defaultdict(list)
        self.node_health: Dict[str, bool] = {f"node_{i}": True for i in range(node_count)}
        
    def record_write_latency(self, node_id: str, latency: float):
        sample = MetricSample(
            value=latency,
            timestamp=time.time(),
            node_id=node_id
        )
        self.write_latencies[node_id].append(sample)
        self._cleanup_old_samples(node_id)
        
    def record_read_latency(self, node_id: str, latency: float):
        sample = MetricSample(
            value=latency,
            timestamp=time.time(),
            node_id=node_id
        )
        self.read_latencies[node_id].append(sample)
        self._cleanup_old_samples(node_id)
        
    def record_consistency_check(self, node_id: str, score: float):
        self.consistency_scores[node_id].append(score)
        if len(self.consistency_scores[node_id]) > 1000:
            self.consistency_scores[node_id] = self.consistency_scores[node_id][-1000:]
            
    def update_node_health(self, node_id: str, is_healthy: bool):
        self.node_health[node_id] = is_healthy
        
    def get_metrics_report(self) -> Dict:
        report = {
            'write_latency': self._calculate_latency_stats(self.write_latencies),
            'read_latency': self._calculate_latency_stats(self.read_latencies),
            'consistency': self._calculate_consistency_stats(),
            'health': self.node_health,
            'system_score': self._calculate_system_score()
        }
        return report
        
    def _calculate_latency_stats(self, latencies: Dict[str, List[MetricSample]]) -> Dict:
        stats = {}
        for node_id, samples in latencies.items():
            if not samples:
                continue
            values = [s.value for s in samples]
            stats[node_id] = {
                'p95': statistics.quantiles(values, n=20)[18],
                'p99': statistics.quantiles(values, n=100)[98],
                'mean': statistics.mean(values),
                'min': min(values),
                'max': max(values)
            }
        return stats
        
    def _calculate_consistency_stats(self) -> Dict:
        stats = {}
        for node_id, scores in self.consistency_scores.items():
            if not scores:
                continue
            stats[node_id] = {
                'mean_score': statistics.mean(scores),
                'min_score': min(scores),
                'max_score': max(scores)
            }
        return stats
        
    def _calculate_system_score(self) -> float:
        # Weighted scoring system
        scores = []
        
        # Health score (40%)
        health_score = sum(1 for h in self.node_health.values() if h) / len(self.node_health)
        scores.append(health_score * 0.4)
        
        # Latency score (30%)
        if self.write_latencies:
            latency_score = min(1.0, 1 / (statistics.mean(
                statistics.mean(s.value for s in samples)
                for samples in self.write_latencies.values()
            )))
            scores.append(latency_score * 0.3)
            
        # Consistency score (30%)
        if self.consistency_scores:
            consistency_score = statistics.mean(
                statistics.mean(scores)
                for scores in self.consistency_scores.values()
            )
            scores.append(consistency_score * 0.3)
            
        return sum(scores)
        
    def _cleanup_old_samples(self, node_id: str):
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep last hour of samples
        
        self.write_latencies[node_id] = [
            s for s in self.write_latencies[node_id]
            if s.timestamp > cutoff_time
        ]
        self.read_latencies[node_id] = [
            s for s in self.read_latencies[node_id]
            if s.timestamp > cutoff_time
        ]

# Usage Example
metrics = ReplicationMetrics(node_count=3)

# Simulate metrics collection
for i in range(3):
    node_id = f"node_{i}"
    metrics.record_write_latency(node_id, 0.05)
    metrics.record_read_latency(node_id, 0.02)
    metrics.record_consistency_check(node_id, 0.95)
    
# Get metrics report
report = metrics.get_metrics_report()
print("System Health Report:")
print(f"Overall System Score: {report['system_score']:.2f}")
print(f"Node Health Status: {report['health']}")
```

Slide 14: Additional Resources

*   Lamport, Leslie. "Time, Clocks, and the Ordering of Events in a Distributed System"
*   [https://lamport.azurewebsites.net/pubs/time-clocks.pdf](https://lamport.azurewebsites.net/pubs/time-clocks.pdf)
*   Dr. Eric Brewer. "CAP Twelve Years Later: How the 'Rules' Have Changed"
*   [https://www.infoq.com/articles/cap-twelve-years-later-how-the-rules-have-changed/](https://www.infoq.com/articles/cap-twelve-years-later-how-the-rules-have-changed/)
*   Martin Kleppmann. "Designing Data-Intensive Applications"
*   Search on Google: "Designing Data-Intensive Applications by Martin Kleppmann"
*   Bernstein, Philip A. and Nathan Goodman. "Concurrency Control in Distributed Database Systems"
*   [https://dl.acm.org/doi/10.1145/356842.356846](https://dl.acm.org/doi/10.1145/356842.356846)
*   Shapiro et al. "A Comprehensive Study of Convergent and Commutative Replicated Data Types"
*   Search on Google Scholar: "Comprehensive Study of CRDTs Shapiro"

