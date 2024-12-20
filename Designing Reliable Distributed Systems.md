## Designing Reliable Distributed Systems
Slide 1: Understanding Fault Tolerance in Distributed Systems

Fault tolerance is a critical property of distributed systems that ensures continuous operation even when components fail. It involves designing systems that can detect, respond to, and recover from various types of failures while maintaining data consistency and availability.

```python
from typing import List, Dict
import random
import time

class Node:
    def __init__(self, node_id: int):
        self.id = node_id
        self.data = {}
        self.is_alive = True
        self.replicas: List[Node] = []
    
    def replicate_to(self, nodes: List['Node']):
        self.replicas = nodes
        for node in self.replicas:
            node.data.update(self.data)
    
    def write(self, key: str, value: str):
        self.data[key] = value
        for replica in self.replicas:
            if replica.is_alive:
                replica.data[key] = value
```

Slide 2: Implementing Heartbeat Mechanism

A heartbeat mechanism enables nodes to detect failures in a distributed system by periodically sending signals to verify the status of other nodes. This implementation demonstrates a basic heartbeat protocol with timeout detection.

```python
import threading
from datetime import datetime, timedelta

class HeartbeatMonitor:
    def __init__(self, timeout_seconds: int = 5):
        self.last_heartbeat = {}
        self.timeout = timeout_seconds
        self.monitor_thread = threading.Thread(target=self._monitor_nodes)
        self.active = True
    
    def send_heartbeat(self, node_id: int):
        self.last_heartbeat[node_id] = datetime.now()
    
    def _monitor_nodes(self):
        while self.active:
            current_time = datetime.now()
            for node_id, last_beat in self.last_heartbeat.items():
                if current_time - last_beat > timedelta(seconds=self.timeout):
                    print(f"Node {node_id} failed - No heartbeat detected")
            time.sleep(1)
```

Slide 3: Consensus Algorithm Implementation

The following implementation demonstrates a simplified version of the Raft consensus algorithm, focusing on leader election and log replication aspects that ensure consistency across distributed nodes.

```python
class RaftNode:
    def __init__(self, node_id: int):
        self.id = node_id
        self.current_term = 0
        self.voted_for = None
        self.log = []
        self.commit_index = 0
        self.state = 'follower'
        self.votes_received = set()
        
    def start_election(self):
        self.state = 'candidate'
        self.current_term += 1
        self.voted_for = self.id
        self.votes_received = {self.id}
        return self.current_term
    
    def vote_request(self, candidate_id: int, term: int) -> bool:
        if term > self.current_term and self.voted_for is None:
            self.voted_for = candidate_id
            self.current_term = term
            return True
        return False
```

Slide 4: Replication Strategies

Replication ensures data availability and fault tolerance by maintaining multiple copies across different nodes. This implementation showcases both synchronous and asynchronous replication strategies with consistency checks.

```python
from enum import Enum
from typing import Optional

class ReplicationType(Enum):
    SYNC = "synchronous"
    ASYNC = "asynchronous"

class ReplicationManager:
    def __init__(self, primary_node: Node, replication_type: ReplicationType):
        self.primary = primary_node
        self.replicas: List[Node] = []
        self.replication_type = replication_type
        
    def add_replica(self, node: Node):
        self.replicas.append(node)
        
    def write_data(self, key: str, value: str) -> bool:
        success = self.primary.write(key, value)
        
        if self.replication_type == ReplicationType.SYNC:
            return self._sync_replicate(key, value)
        else:
            threading.Thread(target=self._async_replicate, 
                           args=(key, value)).start()
            return success
```

Slide 5: Failure Detection and Recovery

Failure detection mechanisms must be robust and accurate to prevent false positives while ensuring quick detection of actual failures. This implementation shows a comprehensive failure detection system with automatic recovery procedures.

```python
class FailureDetector:
    def __init__(self, cluster_nodes: List[Node], detection_interval: float = 1.0):
        self.nodes = cluster_nodes
        self.interval = detection_interval
        self.suspected_nodes = set()
        self.failure_timestamps = {}
        
    def detect_failures(self):
        for node in self.nodes:
            try:
                response = self._ping_node(node)
                if not response and node.id not in self.suspected_nodes:
                    self.suspected_nodes.add(node.id)
                    self.failure_timestamps[node.id] = time.time()
                elif response and node.id in self.suspected_nodes:
                    self._recover_node(node)
            except Exception as e:
                print(f"Error detecting failure for node {node.id}: {str(e)}")
                
    def _ping_node(self, node: Node) -> bool:
        return node.is_alive
        
    def _recover_node(self, node: Node):
        self.suspected_nodes.remove(node.id)
        node.is_alive = True
        self._synchronize_data(node)
```

Slide 6: Partition Tolerance Implementation

Partition tolerance ensures system functionality when network splits occur. This implementation demonstrates handling network partitions while maintaining consistency guarantees within each partition.

```python
class NetworkPartitionHandler:
    def __init__(self):
        self.partitions = {}
        self.partition_leaders = {}
        
    def detect_partition(self, nodes: List[Node]) -> Dict[str, List[Node]]:
        connected_groups = {}
        for node in nodes:
            partition_id = self._find_partition(node)
            if partition_id not in connected_groups:
                connected_groups[partition_id] = []
            connected_groups[partition_id].append(node)
            
        return connected_groups
    
    def handle_partition(self, partition_groups: Dict[str, List[Node]]):
        for partition_id, nodes in partition_groups.items():
            leader = self._elect_partition_leader(nodes)
            self.partition_leaders[partition_id] = leader
            for node in nodes:
                node.current_leader = leader
                
    def _find_partition(self, node: Node) -> str:
        # Simulated network connectivity check
        connected_nodes = [n for n in node.replicas if n.is_alive]
        return f"partition_{hash(tuple(sorted([n.id for n in connected_nodes])))}"
```

Slide 7: Quorum-based Consistency

Quorum-based systems ensure consistency by requiring a minimum number of nodes to agree on operations. This implementation shows a practical approach to maintaining quorum consensus.

```python
class QuorumSystem:
    def __init__(self, total_nodes: int):
        self.total_nodes = total_nodes
        self.read_quorum = total_nodes // 2 + 1
        self.write_quorum = total_nodes // 2 + 1
        
    def perform_write(self, nodes: List[Node], key: str, value: str) -> bool:
        successful_writes = 0
        for node in nodes:
            if node.is_alive and self._write_to_node(node, key, value):
                successful_writes += 1
                
        return successful_writes >= self.write_quorum
    
    def perform_read(self, nodes: List[Node], key: str) -> Optional[str]:
        values = []
        timestamps = []
        
        for node in nodes:
            if node.is_alive:
                value, timestamp = self._read_from_node(node, key)
                if value is not None:
                    values.append(value)
                    timestamps.append(timestamp)
                    
        if len(values) >= self.read_quorum:
            # Return the most recent value based on timestamp
            return values[timestamps.index(max(timestamps))]
        return None
```

Slide 8: Source Code for Quorum-based Consistency

```python
    def _write_to_node(self, node: Node, key: str, value: str) -> bool:
        try:
            node.write(key, value)
            return True
        except Exception:
            return False
            
    def _read_from_node(self, node: Node, key: str) -> Tuple[Optional[str], int]:
        try:
            value = node.data.get(key)
            timestamp = node.timestamps.get(key, 0)
            return value, timestamp
        except Exception:
            return None, 0
            
    def check_quorum_availability(self, nodes: List[Node]) -> bool:
        available_nodes = sum(1 for node in nodes if node.is_alive)
        return (available_nodes >= self.read_quorum and 
                available_nodes >= self.write_quorum)
```

Slide 9: Anti-Entropy Protocol Implementation

Anti-entropy protocols help maintain consistency across replicas by periodically comparing and reconciling differences. This implementation demonstrates a practical approach to detecting and resolving inconsistencies between nodes.

```python
class AntiEntropyProtocol:
    def __init__(self, sync_interval: int = 60):
        self.sync_interval = sync_interval
        self.merkle_trees = {}
        
    def generate_merkle_tree(self, node: Node) -> Dict:
        tree = {}
        for key, value in node.data.items():
            hash_key = self._hash_function(f"{key}:{value}")
            tree[key] = hash_key
        return tree
    
    def sync_nodes(self, node1: Node, node2: Node):
        tree1 = self.generate_merkle_tree(node1)
        tree2 = self.generate_merkle_tree(node2)
        
        differences = set(tree1.items()) ^ set(tree2.items())
        for key, _ in differences:
            if key in node1.data:
                node2.data[key] = node1.data[key]
            else:
                node1.data[key] = node2.data[key]
                
    def _hash_function(self, value: str) -> str:
        import hashlib
        return hashlib.sha256(value.encode()).hexdigest()
```

Slide 10: Vector Clock Implementation

Vector clocks provide a mechanism for tracking causality and ordering events in distributed systems. This implementation shows how to maintain and compare vector timestamps across distributed nodes.

```python
from typing import Dict, Optional

class VectorClock:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.clock: Dict[str, int] = {node_id: 0}
        
    def increment(self):
        self.clock[self.node_id] = self.clock.get(self.node_id, 0) + 1
        
    def update(self, other_clock: Dict[str, int]):
        for node_id, timestamp in other_clock.items():
            self.clock[node_id] = max(
                self.clock.get(node_id, 0),
                timestamp
            )
    
    def compare(self, other_clock: Dict[str, int]) -> Optional[str]:
        # Returns: 'before', 'after', 'concurrent', or None
        less_than = False
        greater_than = False
        
        for node_id in set(self.clock.keys()) | set(other_clock.keys()):
            a = self.clock.get(node_id, 0)
            b = other_clock.get(node_id, 0)
            
            if a < b:
                less_than = True
            if a > b:
                greater_than = True
                
        if less_than and not greater_than:
            return 'before'
        if greater_than and not less_than:
            return 'after'
        if less_than and greater_than:
            return 'concurrent'
        return None
```

Slide 11: Gossip Protocol Implementation

Gossip protocols enable efficient information dissemination in distributed systems. This implementation shows how nodes can spread updates through the network while maintaining eventual consistency.

```python
import random
from typing import Set, Dict, Any

class GossipNode:
    def __init__(self, node_id: str):
        self.id = node_id
        self.data: Dict[str, Any] = {}
        self.version: Dict[str, int] = {}
        self.peers: Set[str] = set()
        self.updates_to_spread: Dict[str, tuple] = {}
        
    def add_update(self, key: str, value: Any):
        self.data[key] = value
        self.version[key] = self.version.get(key, 0) + 1
        self.updates_to_spread[key] = (value, self.version[key])
        
    def gossip(self):
        if not self.peers:
            return
            
        target_peer = random.choice(list(self.peers))
        updates = self.updates_to_spread.copy()
        
        # Simulate sending updates to peer
        for key, (value, version) in updates.items():
            if (key not in self.version or 
                version > self.version[key]):
                self.data[key] = value
                self.version[key] = version
                
        # Clear processed updates
        self.updates_to_spread.clear()
```

Slide 12: CRDTs (Conflict-free Replicated Data Types) Implementation

CRDTs provide a mathematical approach to ensuring eventual consistency without coordination. This implementation demonstrates a Last-Write-Wins Register and a Grow-Only Counter CRDT.

```python
from dataclasses import dataclass
from typing import Optional, Set
import time

@dataclass
class Timestamp:
    clock: int
    node_id: str

    def __lt__(self, other):
        return (self.clock < other.clock or 
                (self.clock == other.clock and self.node_id < other.node_id))

class LWWRegister:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.value: Optional[str] = None
        self.timestamp = Timestamp(0, node_id)
    
    def write(self, value: str) -> None:
        new_timestamp = Timestamp(int(time.time() * 1000), self.node_id)
        if self.timestamp < new_timestamp:
            self.value = value
            self.timestamp = new_timestamp
    
    def merge(self, other: 'LWWRegister') -> None:
        if self.timestamp < other.timestamp:
            self.value = other.value
            self.timestamp = other.timestamp
```

Slide 13: Byzantine Fault Tolerance Implementation

Byzantine Fault Tolerance (BFT) ensures system reliability even when nodes exhibit arbitrary or malicious behavior. This implementation shows a simplified practical BFT consensus mechanism.

```python
from enum import Enum
from typing import List, Set, Dict

class MessageType(Enum):
    PREPARE = "PREPARE"
    COMMIT = "COMMIT"
    REPLY = "REPLY"

class BFTNode:
    def __init__(self, node_id: int, total_nodes: int):
        self.id = node_id
        self.total_nodes = total_nodes
        self.sequence_number = 0
        self.prepared_messages: Dict[int, Set[int]] = {}
        self.committed_messages: Dict[int, Set[int]] = {}
        
    def handle_request(self, client_request: str) -> bool:
        self.sequence_number += 1
        
        # Phase 1: Prepare
        prepare_count = self._broadcast_prepare(client_request)
        if prepare_count < self._min_quorum():
            return False
            
        # Phase 2: Commit
        commit_count = self._broadcast_commit(client_request)
        return commit_count >= self._min_quorum()
    
    def _min_quorum(self) -> int:
        # Minimum nodes needed to achieve consensus (2f + 1)
        f = (self.total_nodes - 1) // 3
        return 2 * f + 1
    
    def _broadcast_prepare(self, request: str) -> int:
        # Simulate prepare phase broadcasting
        self.prepared_messages[self.sequence_number] = {self.id}
        return len(self.prepared_messages[self.sequence_number])
```

Slide 14: Results and Performance Metrics

This slide presents the performance metrics and reliability measurements for the implemented fault tolerance mechanisms using synthetic workload testing.

```python
import time
from statistics import mean

def measure_system_performance(nodes: List[Node], iterations: int = 1000):
    results = {
        'write_latency': [],
        'read_latency': [],
        'consensus_time': [],
        'recovery_time': []
    }
    
    # Measure write latency
    start = time.time()
    for i in range(iterations):
        write_start = time.time()
        nodes[0].write(f"key_{i}", f"value_{i}")
        results['write_latency'].append(time.time() - write_start)
    
    print(f"Average Write Latency: {mean(results['write_latency'])*1000:.2f}ms")
    print(f"Average Consensus Time: {mean(results['consensus_time'])*1000:.2f}ms")
    print(f"System Availability: {(iterations-len(results['recovery_time']))/iterations*100:.2f}%")
```

Slide 15: Additional Resources

*   "Practical Byzantine Fault Tolerance" - [https://arxiv.org/abs/1801.10347](https://arxiv.org/abs/1801.10347)
*   "CRDTs: Making δ-CRDTs Delta-Based" - [https://arxiv.org/abs/1603.01529](https://arxiv.org/abs/1603.01529)
*   "The Part-Time Parliament: Paxos Made Simple" - [https://www.microsoft.com/en-us/research/publication/part-time-parliament/](https://www.microsoft.com/en-us/research/publication/part-time-parliament/)
*   "Gossip Protocols for Large-Scale Distributed Systems" - [https://dl.acm.org/doi/10.1145/1317379.1317382](https://dl.acm.org/doi/10.1145/1317379.1317382)
*   Search Keywords for Further Research:
    *   "Vector Clocks in Distributed Systems"
    *   "Anti-Entropy Protocols Implementation"
    *   "Quorum-based Systems Design"

