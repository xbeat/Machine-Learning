## Maintaining Consistency in Distributed Systems
Slide 1: Understanding Distributed Consensus

Distributed consensus ensures all nodes in a system agree on shared state despite failures. Consensus protocols like Raft and Paxos provide formal guarantees for agreement, validity, and termination through leader election and log replication mechanisms.

```python
import time
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional

class NodeState(Enum):
    FOLLOWER = 1
    CANDIDATE = 2
    LEADER = 3

@dataclass
class ConsensusNode:
    node_id: int
    state: NodeState
    current_term: int
    voted_for: Optional[int]
    log: List[str]
    
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for = None
        self.log = []
        
    def start_election(self):
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        return self.request_vote()
    
    def request_vote(self) -> dict:
        return {
            'term': self.current_term,
            'candidate_id': self.node_id,
            'last_log_index': len(self.log) - 1,
            'last_log_term': self.current_term - 1
        }
```

Slide 2: Implementing Raft Leader Election

The Raft consensus algorithm uses a leader-based approach where a single node coordinates all changes to the system. The leader election process ensures exactly one leader exists per term through timeout-based voting.

```python
import random
from threading import Timer

class RaftNode(ConsensusNode):
    def __init__(self, node_id: int, nodes: List[int]):
        super().__init__(node_id)
        self.nodes = nodes
        self.votes_received = set()
        self.election_timer = None
        self.reset_election_timeout()
    
    def reset_election_timeout(self):
        if self.election_timer:
            self.election_timer.cancel()
        timeout = random.uniform(150, 300)  # milliseconds
        self.election_timer = Timer(timeout/1000, self.start_election)
        self.election_timer.start()
    
    def receive_vote_request(self, request: dict) -> dict:
        if request['term'] > self.current_term:
            self.current_term = request['term']
            self.state = NodeState.FOLLOWER
            self.voted_for = None
        
        if (self.voted_for is None and 
            request['term'] >= self.current_term):
            self.voted_for = request['candidate_id']
            return {'term': self.current_term, 'vote_granted': True}
        
        return {'term': self.current_term, 'vote_granted': False}
```

Slide 3: Log Replication in Raft

Log replication ensures all nodes maintain consistent state by replicating leader commands in the same order. The leader appends entries to its log and replicates them to followers through AppendEntries RPCs.

```python
@dataclass
class LogEntry:
    term: int
    command: str
    index: int

class RaftLogReplication:
    def __init__(self):
        self.log: List[LogEntry] = []
        self.commit_index = -1
        self.last_applied = -1
        
    def append_entries(self, entries: List[LogEntry], 
                      leader_commit: int) -> bool:
        for entry in entries:
            if entry.index < len(self.log):
                if self.log[entry.index].term != entry.term:
                    self.log = self.log[:entry.index]
            if entry.index >= len(self.log):
                self.log.append(entry)
        
        if leader_commit > self.commit_index:
            self.commit_index = min(leader_commit, len(self.log) - 1)
            
        return True
```

Slide 4: Understanding Paxos Consensus

Paxos achieves consensus through a multi-phase protocol involving proposers, acceptors, and learners. The algorithm guarantees safety by ensuring only a single value can be chosen and progress through majority acceptance.

```python
from typing import Optional, Set

class PaxosAcceptor:
    def __init__(self, id: int):
        self.id = id
        self.promised_id = None
        self.accepted_id = None
        self.accepted_value = None
        
    def prepare(self, proposal_id: int) -> tuple[bool, Optional[int], 
                                                Optional[str]]:
        if (self.promised_id is None or 
            proposal_id > self.promised_id):
            self.promised_id = proposal_id
            return (True, self.accepted_id, self.accepted_value)
        return (False, None, None)
    
    def accept(self, proposal_id: int, value: str) -> bool:
        if (self.promised_id is None or 
            proposal_id >= self.promised_id):
            self.promised_id = proposal_id
            self.accepted_id = proposal_id
            self.accepted_value = value
            return True
        return False
```

Slide 5: Implementing Multi-Paxos for State Machine Replication

Multi-Paxos optimizes the basic Paxos protocol for sequence of values by having a stable leader that can skip the prepare phase. This implementation shows the leader election and proposal mechanisms.

```python
class MultiPaxos:
    def __init__(self, node_id: int, nodes: Set[int]):
        self.node_id = node_id
        self.nodes = nodes
        self.leader = None
        self.proposals = {}
        self.instance_id = 0
        self.accepted_values = {}
        
    def propose_value(self, value: str) -> bool:
        if self.leader != self.node_id:
            return False
            
        instance = self.instance_id
        self.instance_id += 1
        
        # Phase 1: Prepare
        proposal_id = self.generate_proposal_id()
        prepared_count = 0
        
        for node in self.nodes:
            if self.send_prepare(node, proposal_id, instance):
                prepared_count += 1
                
        # Phase 2: Accept
        if prepared_count > len(self.nodes) // 2:
            accepted_count = 0
            for node in self.nodes:
                if self.send_accept(node, proposal_id, 
                                  instance, value):
                    accepted_count += 1
                    
            if accepted_count > len(self.nodes) // 2:
                self.accepted_values[instance] = value
                return True
                
        return False
        
    def generate_proposal_id(self) -> int:
        return int(time.time() * 1000) << 32 | self.node_id
```

Slide 6: Byzantine Fault Tolerance in Distributed Systems

Byzantine fault tolerance addresses scenarios where nodes may behave maliciously or fail in arbitrary ways. This implementation demonstrates a practical BFT consensus mechanism for handling Byzantine failures.

```python
from collections import defaultdict
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

class BFTNode:
    def __init__(self, node_id: int, total_nodes: int, 
                 private_key: rsa.RSAPrivateKey):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.private_key = private_key
        self.view_number = 0
        self.sequence_number = 0
        self.prepared_messages = defaultdict(set)
        self.committed_messages = defaultdict(set)
        
    def create_prepare_message(self, client_request: str) -> dict:
        digest = self.compute_digest(client_request)
        signature = self.sign_message(digest)
        
        return {
            'type': 'PREPARE',
            'view': self.view_number,
            'sequence': self.sequence_number,
            'digest': digest,
            'node_id': self.node_id,
            'signature': signature
        }
        
    def compute_digest(self, message: str) -> bytes:
        digest = hashes.Hash(hashes.SHA256())
        digest.update(message.encode())
        return digest.finalize()
        
    def sign_message(self, digest: bytes) -> bytes:
        return self.private_key.sign(
            digest,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
```

Slide 7: Implementing Vector Clocks for Causality Tracking

Vector clocks enable tracking causality relationships between events in distributed systems. This implementation shows how to maintain and compare vector timestamps across distributed processes.

```python
from typing import Dict, List

class VectorClock:
    def __init__(self, process_id: str, processes: List[str]):
        self.process_id = process_id
        self.clock: Dict[str, int] = {p: 0 for p in processes}
        
    def increment(self):
        self.clock[self.process_id] += 1
        
    def update(self, other_clock: Dict[str, int]):
        for process_id, timestamp in other_clock.items():
            self.clock[process_id] = max(
                self.clock[process_id], 
                timestamp
            )
            
    def is_concurrent_with(self, other_clock: Dict[str, int]) -> bool:
        return (not self.happens_before(other_clock) and 
                not self.happens_after(other_clock))
    
    def happens_before(self, other_clock: Dict[str, int]) -> bool:
        return (any(self.clock[k] < v for k, v in other_clock.items()) and
                all(self.clock[k] <= v for k, v in other_clock.items()))
                
    def happens_after(self, other_clock: Dict[str, int]) -> bool:
        return (any(self.clock[k] > v for k, v in other_clock.items()) and
                all(self.clock[k] >= v for k, v in other_clock.items()))
```

Slide 8: Implementing a Distributed Lock Manager

A distributed lock manager provides synchronized access to shared resources across distributed processes. This implementation shows lock acquisition and release with deadlock prevention.

```python
import threading
from datetime import datetime, timedelta

class DistributedLock:
    def __init__(self, resource_id: str, timeout_ms: int = 5000):
        self.resource_id = resource_id
        self.owner = None
        self.expiry = None
        self.timeout = timedelta(milliseconds=timeout_ms)
        self.lock = threading.Lock()
        
    def acquire(self, requester_id: str) -> bool:
        with self.lock:
            current_time = datetime.now()
            
            # Check if lock is free or expired
            if (self.owner is None or 
                (self.expiry is not None and 
                 current_time > self.expiry)):
                self.owner = requester_id
                self.expiry = current_time + self.timeout
                return True
                
            # Handle deadlock prevention
            if (self.owner == requester_id and 
                current_time <= self.expiry):
                self.expiry = current_time + self.timeout
                return True
                
            return False
            
    def release(self, requester_id: str) -> bool:
        with self.lock:
            if self.owner == requester_id:
                self.owner = None
                self.expiry = None
                return True
            return False
```

Slide 9: Implementing a Conflict-free Replicated Data Type (CRDT)

CRDTs provide eventual consistency without coordination by using mathematically sound merge operations. This implementation shows a Grow-Only Set (G-Set) CRDT.

```python
from dataclasses import dataclass
from typing import Set, TypeVar, Generic

T = TypeVar('T')

@dataclass
class GSet(Generic[T]):
    elements: Set[T]
    
    def __init__(self):
        self.elements = set()
        
    def add(self, element: T):
        self.elements.add(element)
        
    def contains(self, element: T) -> bool:
        return element in self.elements
        
    def merge(self, other: 'GSet[T]') -> 'GSet[T]':
        merged = GSet[T]()
        merged.elements = self.elements.union(other.elements)
        return merged
        
    @property
    def value(self) -> Set[T]:
        return self.elements.copy()

# Example usage
def demonstrate_gset():
    replica1 = GSet[str]()
    replica2 = GSet[str]()
    
    # Concurrent operations
    replica1.add("A")
    replica2.add("B")
    
    # Merge replicas
    merged = replica1.merge(replica2)
    print(f"Merged set: {merged.value}")  # {'A', 'B'}
```

Slide 10: Real-time Clock Synchronization Protocol

Network Time Protocol (NTP) implementation for maintaining synchronized clocks across distributed nodes. This implementation handles clock skew and network delays through statistical filtering of time samples.

```python
import time
import statistics
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class TimeOffset:
    offset: float
    delay: float
    timestamp: float

class ClockSynchronization:
    def __init__(self, sync_interval: float = 10.0):
        self.offsets: List[TimeOffset] = []
        self.sync_interval = sync_interval
        self.last_sync = 0.0
        
    def request_time(self, server_address: str) -> Tuple[float, float]:
        t1 = time.time()
        server_time = self.get_server_time(server_address)
        t4 = time.time()
        
        t2, t3 = server_time
        
        delay = (t4 - t1) - (t3 - t2)
        offset = ((t2 - t1) + (t3 - t4)) / 2
        
        return offset, delay
        
    def update_time(self, server_address: str):
        offset, delay = self.request_time(server_address)
        self.offsets.append(TimeOffset(offset, delay, time.time()))
        
        # Keep only recent samples
        self.offsets = [o for o in self.offsets 
                       if time.time() - o.timestamp < 3600]
        
        # Calculate filtered offset
        filtered_offset = statistics.median(
            [o.offset for o in self.offsets]
        )
        return filtered_offset
        
    def get_server_time(self, server_address: str) -> Tuple[float, float]:
        # Simulated server response
        now = time.time()
        return (now - 0.001, now + 0.001)
```

Slide 11: Quorum-based Replicated Storage

Implementation of a quorum-based storage system that ensures consistency through read and write quorums while maintaining availability under partial failures.

```python
from typing import Dict, Set, Optional, Tuple
import random

class QuorumStorage:
    def __init__(self, nodes: Set[str], read_quorum: int, 
                 write_quorum: int):
        self.nodes = nodes
        self.read_quorum = read_quorum
        self.write_quorum = write_quorum
        self.data: Dict[str, Dict[str, Tuple[int, str]]] = {
            node: {} for node in nodes
        }
        
    def write(self, key: str, value: str) -> bool:
        version = int(time.time() * 1000)
        available_nodes = set(random.sample(
            list(self.nodes), 
            len(self.nodes) - 1
        ))
        
        successful_writes = 0
        for node in available_nodes:
            try:
                self.data[node][key] = (version, value)
                successful_writes += 1
            except Exception:
                continue
                
        return successful_writes >= self.write_quorum
        
    def read(self, key: str) -> Optional[str]:
        available_nodes = set(random.sample(
            list(self.nodes), 
            len(self.nodes) - 1
        ))
        
        versions = []
        successful_reads = 0
        
        for node in available_nodes:
            try:
                if key in self.data[node]:
                    version, value = self.data[node][key]
                    versions.append((version, value))
                    successful_reads += 1
            except Exception:
                continue
                
        if successful_reads >= self.read_quorum:
            return max(versions, key=lambda x: x[0])[1]
        return None
```

Slide 12: Implementing a Gossip Protocol

Gossip protocol implementation for efficient information dissemination in distributed systems, featuring configurable fanout and anti-entropy mechanisms.

```python
import random
from typing import Set, Dict, Any

class GossipNode:
    def __init__(self, node_id: str, peers: Set[str], 
                 fanout: int = 3):
        self.node_id = node_id
        self.peers = peers
        self.fanout = min(fanout, len(peers))
        self.state: Dict[str, Any] = {}
        self.version_vector: Dict[str, int] = {}
        
    def update_state(self, key: str, value: Any):
        self.state[key] = value
        self.version_vector[self.node_id] = \
            self.version_vector.get(self.node_id, 0) + 1
        
    def select_peers(self) -> Set[str]:
        return set(random.sample(list(self.peers), self.fanout))
        
    def generate_digest(self) -> Dict[str, int]:
        return self.version_vector.copy()
        
    def get_updates(self, digest: Dict[str, int]) -> Dict[str, Any]:
        updates = {}
        for key, value in self.state.items():
            node = key.split(':')[0]
            if (node not in digest or 
                digest[node] < self.version_vector[node]):
                updates[key] = value
        return updates
        
    def merge_updates(self, updates: Dict[str, Any], 
                     digest: Dict[str, int]):
        self.state.update(updates)
        for node, version in digest.items():
            self.version_vector[node] = max(
                self.version_vector.get(node, 0),
                version
            )
```

Slide 13: Additional Resources

*   [https://arxiv.org/abs/1902.01367](https://arxiv.org/abs/1902.01367) - "The Part-Time Parliament: Paxos Made Simple"
*   [https://arxiv.org/abs/2008.13655](https://arxiv.org/abs/2008.13655) - "Raft Refloated: Do We Have Consensus?"
*   [https://arxiv.org/abs/1802.07000](https://arxiv.org/abs/1802.07000) - "Byzantine Fault Tolerance in the Age of Blockchains"
*   [https://dl.acm.org/doi/10.1145/571825.571827](https://dl.acm.org/doi/10.1145/571825.571827) - "Vector Timestamps: Foundations and Applications"
*   Search terms for further reading:
    *   "Practical Byzantine Fault Tolerance (PBFT)"
    *   "CRDTs for Distributed Systems"
    *   "Distributed Consensus Algorithms"

