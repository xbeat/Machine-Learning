## Concurrency Control in Databases
Slide 1: Understanding Database Transactions and ACID Properties

A database transaction represents a unit of work that must be executed atomically, ensuring data consistency. ACID properties (Atomicity, Consistency, Isolation, Durability) form the foundation of reliable transaction processing in concurrent database operations.

```python
import threading
import time
from typing import Dict, List

class TransactionManager:
    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()
        self.transaction_log = []
    
    def begin_transaction(self) -> str:
        transaction_id = f"txn_{time.time()}"
        self.transaction_log.append(f"Started transaction {transaction_id}")
        return transaction_id
    
    def commit(self, transaction_id: str):
        with self.lock:
            self.transaction_log.append(f"Committed {transaction_id}")
    
    def rollback(self, transaction_id: str):
        with self.lock:
            self.transaction_log.append(f"Rolled back {transaction_id}")

# Example usage
tm = TransactionManager()
txn = tm.begin_transaction()
try:
    # Perform operations
    tm.commit(txn)
except Exception:
    tm.rollback(txn)
```

Slide 2: Implementing Two-Phase Locking (2PL)

Two-Phase Locking is a concurrency control protocol that ensures serializability by dividing lock operations into two phases: expanding (acquiring locks) and shrinking (releasing locks). This prevents potential conflicts between concurrent transactions.

```python
from enum import Enum
from typing import Set, Dict
from threading import Lock

class LockMode(Enum):
    SHARED = 1
    EXCLUSIVE = 2

class LockManager:
    def __init__(self):
        self.lock_table: Dict[str, Set[str]] = {}  # resource -> transaction_ids
        self.lock_mode: Dict[str, LockMode] = {}   # resource -> lock_mode
        self._lock = Lock()
    
    def acquire_lock(self, transaction_id: str, resource: str, mode: LockMode) -> bool:
        with self._lock:
            if resource not in self.lock_table:
                self.lock_table[resource] = {transaction_id}
                self.lock_mode[resource] = mode
                return True
            
            if mode == LockMode.SHARED and self.lock_mode[resource] == LockMode.SHARED:
                self.lock_table[resource].add(transaction_id)
                return True
                
            return False
    
    def release_lock(self, transaction_id: str, resource: str):
        with self._lock:
            if resource in self.lock_table:
                self.lock_table[resource].discard(transaction_id)
                if not self.lock_table[resource]:
                    del self.lock_table[resource]
                    del self.lock_mode[resource]

# Example usage
lock_manager = LockManager()
tx1_acquired = lock_manager.acquire_lock("T1", "account_123", LockMode.EXCLUSIVE)
tx2_acquired = lock_manager.acquire_lock("T2", "account_123", LockMode.SHARED)
```

Slide 3: Implementing Multi-Version Concurrency Control (MVCC)

MVCC maintains multiple versions of data items to enhance concurrency by allowing readers to access consistent snapshots without blocking writers. Each transaction sees a consistent snapshot of the database as it existed at the start of the transaction.

```python
from dataclasses import dataclass
from typing import Dict, List
import time

@dataclass
class Version:
    value: any
    timestamp: float
    transaction_id: str
    is_deleted: bool = False

class MVCCDatabase:
    def __init__(self):
        self.versions: Dict[str, List[Version]] = {}
        self.active_transactions: Dict[str, float] = {}
    
    def start_transaction(self) -> str:
        txn_id = f"tx_{time.time()}"
        self.active_transactions[txn_id] = time.time()
        return txn_id
    
    def write(self, key: str, value: any, transaction_id: str):
        if key not in self.versions:
            self.versions[key] = []
            
        version = Version(
            value=value,
            timestamp=self.active_transactions[transaction_id],
            transaction_id=transaction_id
        )
        self.versions[key].append(version)
    
    def read(self, key: str, transaction_id: str) -> any:
        if key not in self.versions:
            return None
            
        transaction_start_time = self.active_transactions[transaction_id]
        valid_versions = [v for v in self.versions[key]
                         if v.timestamp <= transaction_start_time
                         and not v.is_deleted]
        
        if not valid_versions:
            return None
            
        return max(valid_versions, key=lambda v: v.timestamp).value

# Example usage
db = MVCCDatabase()
tx1 = db.start_transaction()
db.write("user_1", {"name": "Alice"}, tx1)
tx2 = db.start_transaction()
print(db.read("user_1", tx2))  # Returns {"name": "Alice"}
```

Slide 4: Implementing Optimistic Concurrency Control

Optimistic concurrency control assumes conflicts between transactions are rare and validates transactions only at commit time. This approach eliminates the overhead of locking but may require transaction rollbacks if conflicts are detected.

```python
from collections import defaultdict
import copy

class OptimisticConcurrencyControl:
    def __init__(self):
        self.data = {}
        self.version_numbers = defaultdict(int)
        self.read_sets = {}
        self.write_sets = {}
    
    def begin_transaction(self, transaction_id: str):
        self.read_sets[transaction_id] = {}
        self.write_sets[transaction_id] = {}
    
    def read(self, transaction_id: str, key: str) -> any:
        current_version = self.version_numbers[key]
        value = self.data.get(key)
        self.read_sets[transaction_id][key] = current_version
        return copy.deepcopy(value)
    
    def write(self, transaction_id: str, key: str, value: any):
        self.write_sets[transaction_id][key] = value
    
    def validate(self, transaction_id: str) -> bool:
        for key, read_version in self.read_sets[transaction_id].items():
            if self.version_numbers[key] > read_version:
                return False
        return True
    
    def commit(self, transaction_id: str) -> bool:
        if not self.validate(transaction_id):
            return False
            
        for key, value in self.write_sets[transaction_id].items():
            self.data[key] = value
            self.version_numbers[key] += 1
            
        del self.read_sets[transaction_id]
        del self.write_sets[transaction_id]
        return True

# Example usage
occ = OptimisticConcurrencyControl()
occ.begin_transaction("T1")
occ.write("T1", "balance", 100)
success = occ.commit("T1")
print(f"Transaction committed: {success}")
```

Slide 5: Implementing Serializable Snapshot Isolation (SSI)

Serializable Snapshot Isolation provides stronger guarantees than traditional snapshot isolation by detecting write-skew anomalies. It tracks read and write dependencies between transactions to identify potential serialization conflicts.

```python
from dataclasses import dataclass
from typing import Dict, Set
from enum import Enum
import time

class TransactionStatus(Enum):
    ACTIVE = 1
    COMMITTED = 2
    ABORTED = 3

@dataclass
class Transaction:
    id: str
    start_time: float
    status: TransactionStatus
    read_set: Set[str]
    write_set: Set[str]

class SerializableSnapshotIsolation:
    def __init__(self):
        self.transactions: Dict[str, Transaction] = {}
        self.committed_values: Dict[str, Dict[float, any]] = {}
        
    def start_transaction(self) -> str:
        txn_id = f"tx_{time.time()}"
        self.transactions[txn_id] = Transaction(
            id=txn_id,
            start_time=time.time(),
            status=TransactionStatus.ACTIVE,
            read_set=set(),
            write_set=set()
        )
        return txn_id
    
    def read(self, txn_id: str, key: str) -> any:
        transaction = self.transactions[txn_id]
        transaction.read_set.add(key)
        
        if key not in self.committed_values:
            return None
            
        valid_versions = {ts: val for ts, val in self.committed_values[key].items()
                         if ts < transaction.start_time}
        
        if not valid_versions:
            return None
            
        return valid_versions[max(valid_versions.keys())]
    
    def write(self, txn_id: str, key: str, value: any):
        transaction = self.transactions[txn_id]
        transaction.write_set.add(key)
        
        if key not in self.committed_values:
            self.committed_values[key] = {}
            
        self.committed_values[key][transaction.start_time] = value
    
    def check_conflicts(self, txn_id: str) -> bool:
        transaction = self.transactions[txn_id]
        
        for other_txn in self.transactions.values():
            if (other_txn.id != txn_id and 
                other_txn.status == TransactionStatus.ACTIVE):
                if (transaction.write_set & other_txn.read_set or
                    transaction.read_set & other_txn.write_set):
                    return False
        return True

# Example usage
ssi = SerializableSnapshotIsolation()
tx1 = ssi.start_transaction()
ssi.write(tx1, "account_1", 1000)
tx2 = ssi.start_transaction()
balance = ssi.read(tx2, "account_1")
print(f"Transaction {tx2} read balance: {balance}")
```

Slide 6: Implementing Deadlock Detection

Deadlock detection is crucial in database systems to identify and resolve circular wait conditions between transactions. This implementation uses a wait-for graph to detect potential deadlocks.

```python
from collections import defaultdict
from typing import Dict, Set, List
import threading

class DeadlockDetector:
    def __init__(self):
        self.wait_for_graph: Dict[str, Set[str]] = defaultdict(set)
        self.locks: Dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
    
    def add_wait(self, waiting_txn: str, holding_txn: str):
        with self._lock:
            self.wait_for_graph[waiting_txn].add(holding_txn)
    
    def remove_wait(self, waiting_txn: str, holding_txn: str):
        with self._lock:
            if waiting_txn in self.wait_for_graph:
                self.wait_for_graph[waiting_txn].discard(holding_txn)
                if not self.wait_for_graph[waiting_txn]:
                    del self.wait_for_graph[waiting_txn]
    
    def detect_cycle(self) -> List[str]:
        def dfs(node: str, visited: Set[str], path: Set[str]) -> List[str]:
            if node in path:
                cycle_start_idx = list(path).index(node)
                return list(path)[cycle_start_idx:]
            
            if node in visited:
                return []
                
            visited.add(node)
            path.add(node)
            
            for neighbor in self.wait_for_graph.get(node, []):
                cycle = dfs(neighbor, visited, path)
                if cycle:
                    return cycle
            
            path.remove(node)
            return []
        
        with self._lock:
            visited = set()
            for node in self.wait_for_graph:
                if node not in visited:
                    cycle = dfs(node, visited, set())
                    if cycle:
                        return cycle
            return []
    
    def resolve_deadlock(self) -> str:
        cycle = self.detect_cycle()
        if cycle:
            # Choose youngest transaction to abort
            victim = max(cycle)
            self.abort_transaction(victim)
            return victim
        return ""
    
    def abort_transaction(self, transaction_id: str):
        with self._lock:
            # Remove all edges involving this transaction
            self.wait_for_graph.pop(transaction_id, None)
            for txn in self.wait_for_graph:
                self.wait_for_graph[txn].discard(transaction_id)

# Example usage
detector = DeadlockDetector()
detector.add_wait("T1", "T2")
detector.add_wait("T2", "T3")
detector.add_wait("T3", "T1")
cycle = detector.detect_cycle()
print(f"Detected deadlock cycle: {cycle}")
victim = detector.resolve_deadlock()
print(f"Chose transaction {victim} as victim")
```

Slide 7: Implementing Row-Level Locking

Row-level locking provides fine-grained concurrency control by allowing multiple transactions to access different rows of the same table simultaneously, improving overall system throughput.

```python
from enum import Enum
from typing import Dict, Set
import threading
from dataclasses import dataclass

class LockType(Enum):
    SHARED = 1
    EXCLUSIVE = 2

@dataclass
class RowLock:
    type: LockType
    holders: Set[str]

class RowLevelLockManager:
    def __init__(self):
        self.locks: Dict[str, Dict[int, RowLock]] = {}  # table -> {row_id -> lock}
        self._lock = threading.Lock()
    
    def acquire_lock(self, txn_id: str, table: str, row_id: int, 
                    lock_type: LockType) -> bool:
        with self._lock:
            if table not in self.locks:
                self.locks[table] = {}
            
            if row_id not in self.locks[table]:
                self.locks[table][row_id] = RowLock(lock_type, {txn_id})
                return True
            
            current_lock = self.locks[table][row_id]
            
            # Check if compatible
            if (current_lock.type == LockType.SHARED and 
                lock_type == LockType.SHARED):
                current_lock.holders.add(txn_id)
                return True
            
            if len(current_lock.holders) == 1 and txn_id in current_lock.holders:
                if (current_lock.type == LockType.SHARED and 
                    lock_type == LockType.EXCLUSIVE):
                    current_lock.type = LockType.EXCLUSIVE
                    return True
            
            return False
    
    def release_lock(self, txn_id: str, table: str, row_id: int):
        with self._lock:
            if (table in self.locks and 
                row_id in self.locks[table]):
                lock = self.locks[table][row_id]
                lock.holders.discard(txn_id)
                
                if not lock.holders:
                    del self.locks[table][row_id]
                    if not self.locks[table]:
                        del self.locks[table]

# Example usage
lock_manager = RowLevelLockManager()
success1 = lock_manager.acquire_lock("T1", "users", 1, LockType.SHARED)
success2 = lock_manager.acquire_lock("T2", "users", 1, LockType.SHARED)
success3 = lock_manager.acquire_lock("T3", "users", 1, LockType.EXCLUSIVE)
print(f"Shared lock T1: {success1}")
print(f"Shared lock T2: {success2}")
print(f"Exclusive lock T3: {success3}")
```

Slide 8: Implementing Timestamp-Based Concurrency Control

Timestamp-based concurrency control assigns unique timestamps to transactions and uses them to determine the serialization order. This implementation manages read and write timestamps for each data item to ensure consistency.

```python
from dataclasses import dataclass
from typing import Dict, Optional
import time

@dataclass
class DataItem:
    value: any
    write_timestamp: float
    read_timestamp: float

class TimestampBasedCC:
    def __init__(self):
        self.data: Dict[str, DataItem] = {}
        self.transaction_timestamps: Dict[str, float] = {}
    
    def start_transaction(self) -> str:
        txn_id = f"tx_{time.time()}"
        self.transaction_timestamps[txn_id] = time.time()
        return txn_id
    
    def read(self, txn_id: str, key: str) -> Optional[any]:
        if key not in self.data:
            return None
            
        txn_ts = self.transaction_timestamps[txn_id]
        item = self.data[key]
        
        if txn_ts < item.write_timestamp:
            raise ValueError("Transaction too old to read this item")
        
        item.read_timestamp = max(item.read_timestamp, txn_ts)
        return item.value
    
    def write(self, txn_id: str, key: str, value: any):
        txn_ts = self.transaction_timestamps[txn_id]
        
        if key in self.data:
            item = self.data[key]
            if txn_ts < item.read_timestamp:
                raise ValueError("Transaction too old to write this item")
            if txn_ts < item.write_timestamp:
                raise ValueError("Transaction too old to write this item")
        
        self.data[key] = DataItem(
            value=value,
            write_timestamp=txn_ts,
            read_timestamp=txn_ts
        )
    
    def commit(self, txn_id: str):
        del self.transaction_timestamps[txn_id]

# Example usage
tcc = TimestampBasedCC()
tx1 = tcc.start_transaction()
try:
    tcc.write(tx1, "stock_item", 100)
    value = tcc.read(tx1, "stock_item")
    print(f"Read value: {value}")
    tcc.commit(tx1)
except ValueError as e:
    print(f"Transaction aborted: {e}")
```

Slide 9: Implementing Multiversion Timestamp Ordering (MVTO)

Multiversion Timestamp Ordering combines the benefits of MVCC and timestamp-based concurrency control, maintaining multiple versions of data items and using timestamps to determine visibility and conflict resolution.

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import time

@dataclass
class Version:
    value: any
    timestamp: float
    read_timestamp: float

class MVTODatabase:
    def __init__(self):
        self.versions: Dict[str, List[Version]] = {}
        self.active_transactions: Dict[str, float] = {}
    
    def start_transaction(self) -> str:
        txn_id = f"tx_{time.time()}"
        self.active_transactions[txn_id] = time.time()
        return txn_id
    
    def find_readable_version(self, key: str, timestamp: float) -> Optional[Version]:
        if key not in self.versions:
            return None
            
        valid_versions = [v for v in self.versions[key]
                         if v.timestamp <= timestamp]
        
        if not valid_versions:
            return None
            
        return max(valid_versions, key=lambda v: v.timestamp)
    
    def read(self, txn_id: str, key: str) -> Optional[any]:
        timestamp = self.active_transactions[txn_id]
        version = self.find_readable_version(key, timestamp)
        
        if version:
            version.read_timestamp = max(version.read_timestamp, timestamp)
            return version.value
        return None
    
    def write(self, txn_id: str, key: str, value: any):
        timestamp = self.active_transactions[txn_id]
        
        if key in self.versions:
            # Check for write-write conflicts
            conflict_versions = [v for v in self.versions[key]
                               if v.timestamp > timestamp]
            if conflict_versions:
                raise ValueError("Write-write conflict detected")
        
        new_version = Version(
            value=value,
            timestamp=timestamp,
            read_timestamp=timestamp
        )
        
        if key not in self.versions:
            self.versions[key] = []
        self.versions[key].append(new_version)
        self.versions[key].sort(key=lambda v: v.timestamp)
    
    def commit(self, txn_id: str):
        del self.active_transactions[txn_id]

# Example usage
mvto = MVTODatabase()
tx1 = mvto.start_transaction()
try:
    mvto.write(tx1, "product", {"name": "Widget", "price": 10})
    tx2 = mvto.start_transaction()
    value = mvto.read(tx2, "product")
    print(f"Read value in T2: {value}")
    mvto.commit(tx1)
    mvto.commit(tx2)
except ValueError as e:
    print(f"Transaction conflict: {e}")
```

Slide 10: Implementation of Predicate Locking

Predicate locking prevents phantom reads by locking not just existing records but also the space of possible records that could match a predicate. This implementation demonstrates a simplified version of predicate locking.

```python
from dataclasses import dataclass
from typing import Dict, Set, List, Callable
import threading

@dataclass
class Predicate:
    field: str
    operator: str
    value: any

    def matches(self, record: Dict) -> bool:
        if self.field not in record:
            return False
        
        if self.operator == "=":
            return record[self.field] == self.value
        elif self.operator == ">":
            return record[self.field] > self.value
        elif self.operator == "<":
            return record[self.field] < self.value
        return False

class PredicateLockManager:
    def __init__(self):
        self.predicate_locks: Dict[str, List[Predicate]] = {}
        self.lock_holders: Dict[str, Set[str]] = {}
        self._lock = threading.Lock()
    
    def conflicts_with_existing(self, predicate: Predicate) -> bool:
        for existing_predicates in self.predicate_locks.values():
            for existing_pred in existing_predicates:
                if (existing_pred.field == predicate.field and
                    existing_pred.operator in ["=", ">", "<"]):
                    return True
        return False
    
    def acquire_lock(self, txn_id: str, predicate: Predicate) -> bool:
        with self._lock:
            if self.conflicts_with_existing(predicate):
                return False
            
            if txn_id not in self.predicate_locks:
                self.predicate_locks[txn_id] = []
            
            self.predicate_locks[txn_id].append(predicate)
            return True
    
    def release_locks(self, txn_id: str):
        with self._lock:
            if txn_id in self.predicate_locks:
                del self.predicate_locks[txn_id]

# Example usage
lock_manager = PredicateLockManager()
pred1 = Predicate("age", ">", 25)
pred2 = Predicate("age", "=", 30)

success1 = lock_manager.acquire_lock("T1", pred1)
success2 = lock_manager.acquire_lock("T2", pred2)

print(f"T1 lock acquisition: {success1}")
print(f"T2 lock acquisition: {success2}")

# Test predicate matching
record = {"age": 35, "name": "John"}
print(f"Record matches predicate 1: {pred1.matches(record)}")
print(f"Record matches predicate 2: {pred2.matches(record)}")
```

Slide 11: Implementing Read-Write Lock Manager

Read-Write Lock Manager provides granular control over concurrent access to shared resources by distinguishing between read (shared) and write (exclusive) operations, optimizing for scenarios where reads are more frequent than writes.

```python
from enum import Enum
from typing import Dict, Set
import threading
import time

class LockMode(Enum):
    READ = 1
    WRITE = 2

class ReadWriteLockManager:
    def __init__(self):
        self.locks: Dict[str, Dict[str, LockMode]] = {}  # resource -> {txn_id: mode}
        self.waiting: Dict[str, Set[str]] = {}  # resource -> {waiting_txn_ids}
        self._lock = threading.Lock()
    
    def can_acquire_lock(self, resource: str, txn_id: str, mode: LockMode) -> bool:
        if resource not in self.locks:
            return True
            
        current_locks = self.locks[resource]
        
        if txn_id in current_locks:
            if current_locks[txn_id] == mode:
                return True
            if mode == LockMode.WRITE:
                return len(current_locks) == 1
            
        if mode == LockMode.READ:
            return all(m == LockMode.READ for m in current_locks.values())
        else:  # WRITE mode
            return len(current_locks) == 0
    
    def acquire_lock(self, txn_id: str, resource: str, mode: LockMode, 
                    timeout: float = 5.0) -> bool:
        start_time = time.time()
        
        while True:
            with self._lock:
                if self.can_acquire_lock(resource, txn_id, mode):
                    if resource not in self.locks:
                        self.locks[resource] = {}
                    self.locks[resource][txn_id] = mode
                    return True
                
                if resource not in self.waiting:
                    self.waiting[resource] = set()
                self.waiting[resource].add(txn_id)
            
            if time.time() - start_time > timeout:
                with self._lock:
                    if resource in self.waiting:
                        self.waiting[resource].discard(txn_id)
                return False
            
            time.sleep(0.1)
    
    def release_lock(self, txn_id: str, resource: str):
        with self._lock:
            if resource in self.locks and txn_id in self.locks[resource]:
                del self.locks[resource][txn_id]
                if not self.locks[resource]:
                    del self.locks[resource]
            
            if resource in self.waiting:
                self.waiting[resource].discard(txn_id)
                if not self.waiting[resource]:
                    del self.waiting[resource]

# Example usage
lock_manager = ReadWriteLockManager()

def reader_transaction(txn_id: str, resource: str):
    success = lock_manager.acquire_lock(txn_id, resource, LockMode.READ)
    if success:
        print(f"Transaction {txn_id} acquired READ lock")
        time.sleep(1)  # Simulate reading
        lock_manager.release_lock(txn_id, resource)
        print(f"Transaction {txn_id} released READ lock")
    else:
        print(f"Transaction {txn_id} failed to acquire READ lock")

def writer_transaction(txn_id: str, resource: str):
    success = lock_manager.acquire_lock(txn_id, resource, LockMode.WRITE)
    if success:
        print(f"Transaction {txn_id} acquired WRITE lock")
        time.sleep(1)  # Simulate writing
        lock_manager.release_lock(txn_id, resource)
        print(f"Transaction {txn_id} released WRITE lock")
    else:
        print(f"Transaction {txn_id} failed to acquire WRITE lock")

# Create threads for concurrent access
threads = []
for i in range(3):
    threads.append(threading.Thread(target=reader_transaction, 
                                 args=(f"R{i}", "resource1")))
threads.append(threading.Thread(target=writer_transaction, 
                              args=("W1", "resource1")))

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
```

Slide 12: Implementing Intent Lock Protocol

Intent locks allow for efficient hierarchical locking in database systems by indicating planned operations at lower levels in the hierarchy. This implementation demonstrates a simplified version of intention locks.

```python
from enum import Enum
from typing import Dict, Set, Optional
import threading

class LockMode(Enum):
    IS = 1  # Intent Shared
    IX = 2  # Intent Exclusive
    S = 3   # Shared
    X = 4   # Exclusive
    SIX = 5 # Shared + Intent Exclusive

class IntentLockManager:
    def __init__(self):
        self.lock_table: Dict[str, Dict[str, LockMode]] = {}
        self._lock = threading.Lock()
        
        # Lock compatibility matrix
        self.compatible = {
            LockMode.IS: {LockMode.IS, LockMode.IX, LockMode.S, LockMode.SIX},
            LockMode.IX: {LockMode.IS, LockMode.IX},
            LockMode.S: {LockMode.IS, LockMode.S},
            LockMode.X: set(),
            LockMode.SIX: {LockMode.IS}
        }
    
    def get_parent_path(self, resource: str) -> list:
        parts = resource.split('/')
        paths = []
        for i in range(1, len(parts)):
            paths.append('/'.join(parts[:i]))
        return paths
    
    def is_compatible(self, resource: str, mode: LockMode, 
                     txn_id: str) -> bool:
        if resource not in self.lock_table:
            return True
            
        current_locks = self.lock_table[resource]
        for holder, held_mode in current_locks.items():
            if holder != txn_id and mode not in self.compatible[held_mode]:
                return False
        return True
    
    def acquire_lock(self, txn_id: str, resource: str, 
                    mode: LockMode) -> bool:
        with self._lock:
            # Check parent path compatibility
            parent_paths = self.get_parent_path(resource)
            for path in parent_paths:
                if not self.is_compatible(path, mode, txn_id):
                    return False
            
            # Acquire intention locks on parent paths
            for path in parent_paths:
                intent_mode = LockMode.IX if mode in {LockMode.X, LockMode.IX} else LockMode.IS
                if path not in self.lock_table:
                    self.lock_table[path] = {}
                self.lock_table[path][txn_id] = intent_mode
            
            # Acquire actual lock
            if resource not in self.lock_table:
                self.lock_table[resource] = {}
            self.lock_table[resource][txn_id] = mode
            return True
    
    def release_lock(self, txn_id: str, resource: str):
        with self._lock:
            # Release lock on resource
            if resource in self.lock_table:
                if txn_id in self.lock_table[resource]:
                    del self.lock_table[resource][txn_id]
                if not self.lock_table[resource]:
                    del self.lock_table[resource]
            
            # Release intention locks on parent paths
            for path in self.get_parent_path(resource):
                if path in self.lock_table and txn_id in self.lock_table[path]:
                    del self.lock_table[path][txn_id]
                    if not self.lock_table[path]:
                        del self.lock_table[path]

# Example usage
lock_manager = IntentLockManager()

# Acquire locks on hierarchical resources
success1 = lock_manager.acquire_lock("T1", "/db/table1", LockMode.IX)
success2 = lock_manager.acquire_lock("T1", "/db/table1/row1", LockMode.X)
success3 = lock_manager.acquire_lock("T2", "/db/table1", LockMode.S)

print(f"T1 IX lock on table: {success1}")
print(f"T1 X lock on row: {success2}")
print(f"T2 S lock on table: {success3}")
```

Slide 13: Implementing Conflict Serializability Checker

A crucial component in database systems that verifies whether a schedule of transactions is conflict serializable by constructing and analyzing a precedence graph to detect cycles that would indicate non-serializability.

```python
from typing import Dict, Set, List, Tuple
from collections import defaultdict
from dataclasses import dataclass
import copy

@dataclass
class Operation:
    transaction: str
    type: str  # 'R' for read, 'W' for write
    item: str
    
class SerializabilityChecker:
    def __init__(self):
        self.operations: List[Operation] = []
        self.transactions: Set[str] = set()
        self.precedence_graph: Dict[str, Set[str]] = defaultdict(set)
    
    def add_operation(self, transaction: str, op_type: str, item: str):
        self.operations.append(Operation(transaction, op_type, item))
        self.transactions.add(transaction)
    
    def build_precedence_graph(self):
        self.precedence_graph.clear()
        n = len(self.operations)
        
        for i in range(n):
            op1 = self.operations[i]
            for j in range(i + 1, n):
                op2 = self.operations[j]
                
                # Skip if same transaction
                if op1.transaction == op2.transaction:
                    continue
                
                # Check for conflicts
                if op1.item == op2.item and (
                    op1.type == 'W' or op2.type == 'W'
                ):
                    self.precedence_graph[op1.transaction].add(op2.transaction)
    
    def detect_cycle(self) -> List[str]:
        def dfs(node: str, visited: Set[str], path: Set[str]) -> List[str]:
            if node in path:
                cycle_start = node
                cycle = []
                for n in list(path) + [node]:
                    cycle.append(n)
                    if n == cycle_start and len(cycle) > 1:
                        break
                return cycle
            
            if node in visited:
                return []
                
            visited.add(node)
            path.add(node)
            
            for neighbor in self.precedence_graph[node]:
                cycle = dfs(neighbor, visited, path)
                if cycle:
                    return cycle
            
            path.remove(node)
            return []
        
        visited = set()
        for node in self.transactions:
            if node not in visited:
                cycle = dfs(node, visited, set())
                if cycle:
                    return cycle
        return []
    
    def is_conflict_serializable(self) -> Tuple[bool, List[str]]:
        self.build_precedence_graph()
        cycle = self.detect_cycle()
        return (len(cycle) == 0, cycle)
    
    def print_schedule(self):
        for op in self.operations:
            print(f"T{op.transaction}: {op.type}({op.item})")

# Example usage
checker = SerializabilityChecker()

# Schedule 1: T1: R(A) T2: R(A) T2: W(A) T1: W(A)
checker.add_operation("1", "R", "A")
checker.add_operation("2", "R", "A")
checker.add_operation("2", "W", "A")
checker.add_operation("1", "W", "A")

print("Schedule:")
checker.print_schedule()

serializable, cycle = checker.is_conflict_serializable()
if serializable:
    print("Schedule is conflict serializable")
else:
    print(f"Schedule is NOT conflict serializable. Cycle found: {' -> '.join(cycle)}")

# Test another schedule
checker = SerializabilityChecker()
# Schedule 2: T1: R(A) T2: W(B) T1: W(A) T2: R(B)
checker.add_operation("1", "R", "A")
checker.add_operation("2", "W", "B")
checker.add_operation("1", "W", "A")
checker.add_operation("2", "R", "B")

print("\nSecond Schedule:")
checker.print_schedule()

serializable, cycle = checker.is_conflict_serializable()
if serializable:
    print("Schedule is conflict serializable")
else:
    print(f"Schedule is NOT conflict serializable. Cycle found: {' -> '.join(cycle)}")
```

Slide 14: Additional Resources

*   ArXiv papers and additional resources for learning about database concurrency control:
    *   "A Survey of Distributed Database Management Systems" - [https://arxiv.org/abs/1912.08595](https://arxiv.org/abs/1912.08595)
    *   "Optimistic Concurrency Control by Validation" - [https://arxiv.org/abs/2007.06879](https://arxiv.org/abs/2007.06879)
    *   "Multi-Version Concurrency Control: Theory and Practice" - [https://arxiv.org/abs/1908.09203](https://arxiv.org/abs/1908.09203)
    *   Google Scholar search suggestion: "modern database concurrency control mechanisms"
    *   ACM Digital Library: Search for "transaction processing systems"
    *   IEEE Xplore: Browse "distributed database transactions"
    *   Database systems books:
        *   "Transaction Processing: Concepts and Techniques" by Jim Gray
        *   "Principles of Transaction Processing" by Philip A. Bernstein
        *   "Database Management Systems" by Raghu Ramakrishnan

Note: URLs are generic suggestions for finding relevant academic papers. Please verify current links and citations when accessing these resources.

