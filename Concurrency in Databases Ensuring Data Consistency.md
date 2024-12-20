## Concurrency in Databases Ensuring Data Consistency
Slide 1: Understanding Database Transactions

A transaction represents a unit of work that maintains ACID properties - Atomicity, Consistency, Isolation, and Durability. In concurrent systems, multiple transactions executing simultaneously must be managed to prevent data inconsistencies and maintain data integrity.

```python
import threading
import time
from datetime import datetime

class BankAccount:
    def __init__(self, balance):
        self.balance = balance
        self._lock = threading.Lock()
    
    def transfer(self, amount, target_account):
        with self._lock:  # Ensure thread safety
            if self.balance >= amount:
                self.balance -= amount
                time.sleep(0.1)  # Simulate some processing time
                target_account.deposit(amount)
                return True
            return False
    
    def deposit(self, amount):
        with self._lock:
            self.balance += amount
```

Slide 2: Implementing Basic Lock Mechanisms

Database systems use various locking mechanisms to control concurrent access. Two-phase locking (2PL) is a common protocol where transactions must acquire all locks before releasing any, ensuring serializability of concurrent transactions.

```python
class LockManager:
    def __init__(self):
        self.locks = {}
        self._lock = threading.Lock()
    
    def acquire_lock(self, resource_id, transaction_id, lock_type='shared'):
        with self._lock:
            if resource_id not in self.locks:
                self.locks[resource_id] = {'owner': transaction_id, 'type': lock_type}
                return True
            return False
    
    def release_lock(self, resource_id, transaction_id):
        with self._lock:
            if resource_id in self.locks and self.locks[resource_id]['owner'] == transaction_id:
                del self.locks[resource_id]
                return True
            return False
```

Slide 3: Isolation Levels Implementation

Isolation levels determine how transactions interact with each other. We implement different isolation levels including Read Uncommitted, Read Committed, Repeatable Read, and Serializable to demonstrate their characteristics and trade-offs.

```python
from enum import Enum

class IsolationLevel(Enum):
    READ_UNCOMMITTED = 1
    READ_COMMITTED = 2
    REPEATABLE_READ = 3
    SERIALIZABLE = 4

class TransactionManager:
    def __init__(self, isolation_level=IsolationLevel.READ_COMMITTED):
        self.isolation_level = isolation_level
        self.active_transactions = {}
        self.version_history = {}
        self._lock = threading.Lock()
    
    def begin_transaction(self):
        transaction_id = str(datetime.now().timestamp())
        self.active_transactions[transaction_id] = {'snapshot': {}, 'locks': set()}
        return transaction_id
```

Slide 4: Implementing MVCC (Multiversion Concurrency Control)

MVCC maintains multiple versions of data items to allow concurrent transactions to see different versions of the same data. This approach improves read performance by avoiding blocking reads while maintaining consistency.

```python
class MVCCDatabase:
    def __init__(self):
        self.data = {}
        self.versions = {}
        self.timestamp = 0
        self._lock = threading.Lock()
    
    def write(self, key, value, transaction_id):
        with self._lock:
            self.timestamp += 1
            if key not in self.versions:
                self.versions[key] = []
            self.versions[key].append({
                'value': value,
                'timestamp': self.timestamp,
                'transaction_id': transaction_id
            })
            self.data[key] = value
```

Slide 5: Deadlock Detection and Prevention

A critical aspect of concurrent transaction management is handling deadlocks. This implementation shows a deadlock detection algorithm using wait-for graphs and implements timeout-based deadlock prevention.

```python
import networkx as nx

class DeadlockDetector:
    def __init__(self):
        self.wait_for_graph = nx.DiGraph()
        self._lock = threading.Lock()
    
    def add_wait(self, waiting_transaction, holding_transaction):
        with self._lock:
            self.wait_for_graph.add_edge(waiting_transaction, holding_transaction)
            return self.check_deadlock()
    
    def check_deadlock(self):
        try:
            cycle = nx.find_cycle(self.wait_for_graph)
            return cycle
        except nx.NetworkXNoCycle:
            return None
```

Slide 6: Optimistic Concurrency Control

Optimistic concurrency control assumes conflicts between transactions are rare. Instead of locking, transactions proceed without restrictions but must validate their operations before committing to ensure no conflicts occurred during execution.

```python
class OptimisticConcurrencyControl:
    def __init__(self):
        self.data = {}
        self.versions = {}
        self.committed_txns = set()
        self._lock = threading.Lock()
    
    def read(self, key, txn_id):
        with self._lock:
            if key in self.data:
                return self.data[key], self.versions[key]
            return None, 0
    
    def validate(self, txn_id, read_set):
        with self._lock:
            for key, version in read_set.items():
                if key in self.versions and self.versions[key] > version:
                    return False  # Validation failed
            return True
```

Slide 7: Timestamp-Based Concurrency Control

This approach assigns timestamps to transactions and uses them to determine the ordering of operations. Each data item maintains read and write timestamps to ensure serializability based on transaction order.

```python
from collections import defaultdict
from time import time

class TimestampManager:
    def __init__(self):
        self.data = {}
        self.read_timestamps = defaultdict(int)
        self.write_timestamps = defaultdict(int)
        self._lock = threading.Lock()
    
    def read(self, key, txn_timestamp):
        with self._lock:
            if key not in self.data:
                return None
            if txn_timestamp < self.write_timestamps[key]:
                raise Exception("Transaction too old")
            self.read_timestamps[key] = max(self.read_timestamps[key], txn_timestamp)
            return self.data[key]
```

Slide 8: Write-Ahead Logging Implementation

Write-ahead logging ensures durability and atomicity by recording all database modifications in a log before they are applied to the actual database, enabling crash recovery and transaction rollback.

```python
import pickle
from pathlib import Path

class WALManager:
    def __init__(self, log_path="wal.log"):
        self.log_path = Path(log_path)
        self.current_lsn = 0  # Log Sequence Number
        self._lock = threading.Lock()
    
    def log_operation(self, txn_id, operation, data):
        with self._lock:
            log_entry = {
                'lsn': self.current_lsn,
                'txn_id': txn_id,
                'operation': operation,
                'data': data,
                'timestamp': time()
            }
            with open(self.log_path, 'ab') as f:
                pickle.dump(log_entry, f)
            self.current_lsn += 1
            return self.current_lsn - 1
```

Slide 9: Implementing ACID Properties

This implementation demonstrates how to maintain ACID properties in a custom transaction manager, showing the interplay between different mechanisms to ensure data consistency and reliability.

```python
class TransactionManager:
    def __init__(self):
        self.lock_manager = LockManager()
        self.wal_manager = WALManager()
        self.active_txns = {}
        self.committed_txns = set()
        self._lock = threading.Lock()
    
    def begin_transaction(self):
        txn_id = str(time())
        self.active_txns[txn_id] = {
            'read_set': set(),
            'write_set': set(),
            'locks': set()
        }
        self.wal_manager.log_operation(txn_id, 'BEGIN', None)
        return txn_id
    
    def commit(self, txn_id):
        if txn_id not in self.active_txns:
            raise Exception("Invalid transaction")
        
        self.wal_manager.log_operation(txn_id, 'COMMIT', None)
        self._release_all_locks(txn_id)
        self.committed_txns.add(txn_id)
        del self.active_txns[txn_id]
```

Slide 10: Practical Implementation of Row-Level Locking

Row-level locking provides finer granularity control over concurrent access to database records. This implementation shows how to manage locks at the row level while maintaining transaction isolation.

```python
class Row:
    def __init__(self, data):
        self.data = data
        self.shared_locks = set()
        self.exclusive_lock = None
        self._lock = threading.Lock()

class RowLevelLockingDB:
    def __init__(self):
        self.rows = {}
        self._lock = threading.Lock()
    
    def acquire_lock(self, row_id, txn_id, lock_type='shared'):
        if row_id not in self.rows:
            return False
        
        row = self.rows[row_id]
        with row._lock:
            if lock_type == 'shared':
                if row.exclusive_lock is None:
                    row.shared_locks.add(txn_id)
                    return True
            elif lock_type == 'exclusive':
                if not row.shared_locks and row.exclusive_lock is None:
                    row.exclusive_lock = txn_id
                    return True
            return False
```

Slide 11: Snapshot Isolation Implementation

Snapshot isolation provides a consistent view of the database at the start of a transaction. This implementation demonstrates how to maintain multiple versions of data and handle transaction visibility.

```python
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Set

@dataclass
class Version:
    value: any
    txn_id: str
    timestamp: float

class SnapshotIsolationDB:
    def __init__(self):
        self.versions: Dict[str, List[Version]] = defaultdict(list)
        self.committed_txns: Dict[str, float] = {}
        self.active_txns: Set[str] = set()
        self._lock = threading.Lock()
    
    def start_transaction(self) -> str:
        txn_id = str(time())
        with self._lock:
            self.active_txns.add(txn_id)
        return txn_id
    
    def read(self, key: str, txn_id: str) -> any:
        with self._lock:
            if key not in self.versions:
                return None
            
            txn_start_time = float(txn_id)
            visible_versions = [v for v in self.versions[key]
                              if v.timestamp < txn_start_time]
            
            if not visible_versions:
                return None
            
            return max(visible_versions, key=lambda v: v.timestamp).value
```

Slide 12: Real-world Example: Bank Transfer System

This implementation demonstrates a complete banking system with concurrent transfers, maintaining consistency using transaction isolation and deadlock prevention.

```python
class BankingSystem:
    def __init__(self):
        self.accounts = {}
        self.lock_manager = LockManager()
        self.transaction_manager = TransactionManager()
        self._lock = threading.Lock()
    
    def transfer(self, from_acc: str, to_acc: str, amount: float) -> bool:
        txn_id = self.transaction_manager.begin_transaction()
        
        try:
            # Acquire locks in a consistent order to prevent deadlocks
            accounts = sorted([from_acc, to_acc])
            for acc in accounts:
                if not self.lock_manager.acquire_lock(acc, txn_id, 'exclusive'):
                    raise Exception("Could not acquire locks")
            
            if self.accounts[from_acc].balance < amount:
                raise Exception("Insufficient funds")
            
            # Perform transfer
            self.accounts[from_acc].balance -= amount
            self.accounts[to_acc].balance += amount
            
            self.transaction_manager.commit(txn_id)
            return True
            
        except Exception as e:
            self.transaction_manager.rollback(txn_id)
            return False
        
        finally:
            self.lock_manager.release_all_locks(txn_id)
```

Slide 13: Real-world Example: Inventory Management System

This implementation shows a concurrent inventory system handling multiple orders while maintaining stock consistency and preventing overselling through optimistic concurrency control.

```python
from datetime import datetime
from typing import Dict, Optional, Tuple

class InventorySystem:
    def __init__(self):
        self.inventory = {}
        self.version_numbers = {}
        self.transactions = {}
        self._lock = threading.Lock()
    
    def create_order(self, items: Dict[str, int]) -> Optional[str]:
        txn_id = str(datetime.now().timestamp())
        read_versions = {}
        
        # First phase: Read and validate
        for item_id, quantity in items.items():
            stock, version = self.read_item(item_id)
            if stock is None or stock < quantity:
                return None
            read_versions[item_id] = version
        
        # Second phase: Validate and write
        with self._lock:
            # Verify versions haven't changed
            for item_id, version in read_versions.items():
                if self.version_numbers[item_id] != version:
                    return None
            
            # Update inventory
            for item_id, quantity in items.items():
                self.inventory[item_id] -= quantity
                self.version_numbers[item_id] += 1
            
            return txn_id
    
    def read_item(self, item_id: str) -> Tuple[Optional[int], int]:
        with self._lock:
            return (self.inventory.get(item_id), 
                    self.version_numbers.get(item_id, 0))
```

Slide 14: Results Analysis: Performance Metrics

This slide demonstrates the performance analysis of different concurrency control mechanisms implemented in previous slides, comparing throughput and latency under various loads.

```python
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

def benchmark_concurrent_operations(
    system: object,
    num_threads: int,
    operations_per_thread: int
) -> Dict[str, float]:
    results = []
    
    def run_operations():
        start_time = time.time()
        successful_ops = 0
        
        for _ in range(operations_per_thread):
            if system.create_order({'item1': 1, 'item2': 1}):
                successful_ops += 1
                
        end_time = time.time()
        return {
            'duration': end_time - start_time,
            'successful_ops': successful_ops
        }
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(run_operations) 
                  for _ in range(num_threads)]
        results = [f.result() for f in futures]
    
    total_time = max(r['duration'] for r in results)
    total_successful_ops = sum(r['successful_ops'] for r in results)
    
    return {
        'throughput': total_successful_ops / total_time,
        'average_latency': statistics.mean(r['duration'] for r in results),
        'success_rate': total_successful_ops / (num_threads * operations_per_thread)
    }
```

Slide 15: Additional Resources

*   Database Concurrency Control Methods:
    *   [https://arxiv.org/abs/1908.06206](https://arxiv.org/abs/1908.06206)
*   Modern Timestamp-based Concurrency Control:
    *   [https://arxiv.org/abs/2003.01465](https://arxiv.org/abs/2003.01465)
*   Practical Snapshot Isolation:
    *   [https://arxiv.org/abs/2012.07456](https://arxiv.org/abs/2012.07456)
*   General Database Design and Implementation:
    *   [https://dl.acm.org/doi/10.1145/3318464.3386131](https://dl.acm.org/doi/10.1145/3318464.3386131)
*   Further reading on concurrent data structures:
    *   Search "Concurrent Data Structures: Theory to Practice" on Google Scholar
*   Advanced topics in transaction processing:
    *   Visit [https://www.vldb.org/proceedings/](https://www.vldb.org/proceedings/) for latest research papers

