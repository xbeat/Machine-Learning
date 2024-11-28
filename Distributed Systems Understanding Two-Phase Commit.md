## Distributed Systems Understanding Two-Phase Commit
Slide 1: Understanding Two-Phase Commit Protocol

The Two-Phase Commit (2PC) protocol is a distributed algorithm that ensures atomic transaction commitment across multiple nodes in a distributed system. It coordinates all participating processes that take part in a distributed atomic transaction to either commit or abort the transaction.

```python
class Coordinator:
    def __init__(self):
        self.participants = []
        self.state = "INIT"
    
    def add_participant(self, participant):
        self.participants.append(participant)
    
    def execute_2pc(self, transaction):
        # Phase 1: Prepare
        prepare_responses = []
        for participant in self.participants:
            response = participant.prepare(transaction)
            prepare_responses.append(response)
        
        # Decision
        if all(prepare_responses):
            self.state = "COMMIT"
            for participant in self.participants:
                participant.commit()
            return True
        else:
            self.state = "ABORT"
            for participant in self.participants:
                participant.abort()
            return False
```

Slide 2: Participant Implementation in 2PC

The participant node in 2PC must maintain its own transaction state and respond to coordinator requests. Each participant implements prepare, commit, and abort operations while handling potential failures and recovery scenarios.

```python
class Participant:
    def __init__(self, name):
        self.name = name
        self.state = "READY"
        self.transaction_log = []
        
    def prepare(self, transaction):
        try:
            # Validate and prepare transaction
            self.validate_transaction(transaction)
            self.state = "PREPARED"
            self.transaction_log.append(("PREPARE", transaction))
            return True
        except Exception:
            return False
            
    def commit(self):
        self.state = "COMMITTED"
        self.transaction_log.append(("COMMIT", None))
        
    def abort(self):
        self.state = "ABORTED"
        self.transaction_log.append(("ABORT", None))
        
    def validate_transaction(self, transaction):
        # Implement validation logic
        pass
```

Slide 3: Transaction Manager Implementation

The Transaction Manager acts as an intermediary between the application and the 2PC protocol, managing transaction boundaries and coordinating with the coordinator to ensure atomic commitment across all participants.

```python
class TransactionManager:
    def __init__(self):
        self.coordinator = Coordinator()
        self.active_transactions = {}
        self.transaction_counter = 0
        
    def begin_transaction(self):
        txn_id = self.transaction_counter
        self.transaction_counter += 1
        self.active_transactions[txn_id] = {
            'operations': [],
            'state': 'ACTIVE'
        }
        return txn_id
        
    def commit_transaction(self, txn_id):
        if txn_id not in self.active_transactions:
            raise ValueError("Invalid transaction ID")
            
        transaction = self.active_transactions[txn_id]
        success = self.coordinator.execute_2pc(transaction)
        
        if success:
            transaction['state'] = 'COMMITTED'
        else:
            transaction['state'] = 'ABORTED'
        
        return success
```

Slide 4: Handling Network Failures

In distributed systems, network failures are common and must be handled gracefully. This implementation shows how to manage timeouts and retries during the 2PC protocol execution.

```python
import time
from threading import Timer

class FaultTolerantCoordinator(Coordinator):
    def __init__(self, timeout_seconds=5):
        super().__init__()
        self.timeout_seconds = timeout_seconds
        self.prepare_responses = {}
        
    def execute_2pc_with_timeout(self, transaction):
        timer = Timer(self.timeout_seconds, self.handle_timeout)
        timer.start()
        
        try:
            return self.execute_2pc(transaction)
        finally:
            timer.cancel()
            
    def handle_timeout(self):
        self.state = "ABORT"
        for participant in self.participants:
            try:
                participant.abort()
            except Exception:
                continue
```

Slide 5: Recovery Mechanism

The recovery mechanism ensures system consistency after failures by implementing a Write-Ahead Log (WAL) and recovery protocol that handles coordinator or participant crashes during the 2PC process.

```python
class RecoverableParticipant(Participant):
    def __init__(self, name, log_file):
        super().__init__(name)
        self.log_file = log_file
        self.recovery_state = self.recover_state()
        
    def write_log(self, entry):
        with open(self.log_file, 'a') as f:
            f.write(f"{time.time()},{entry}\n")
            
    def recover_state(self):
        try:
            with open(self.log_file, 'r') as f:
                logs = f.readlines()
                
            for log in logs:
                timestamp, entry = log.strip().split(',')
                self.process_recovery_entry(entry)
                
        except FileNotFoundError:
            return "INIT"
```

Slide 6: Transaction State Machine

The state machine implementation for 2PC transactions manages the lifecycle of distributed transactions, tracking their progress through different states and ensuring consistency in state transitions.

```python
from enum import Enum

class TransactionState(Enum):
    INIT = 0
    PREPARING = 1
    PREPARED = 2
    COMMITTING = 3
    COMMITTED = 4
    ABORTING = 5
    ABORTED = 6

class TransactionStateMachine:
    def __init__(self):
        self.state = TransactionState.INIT
        self.state_history = []
        
    def transition_to(self, new_state):
        valid_transitions = {
            TransactionState.INIT: [TransactionState.PREPARING],
            TransactionState.PREPARING: [TransactionState.PREPARED, TransactionState.ABORTING],
            TransactionState.PREPARED: [TransactionState.COMMITTING, TransactionState.ABORTING],
            TransactionState.COMMITTING: [TransactionState.COMMITTED],
            TransactionState.ABORTING: [TransactionState.ABORTED]
        }
        
        if new_state in valid_transitions.get(self.state, []):
            self.state_history.append((self.state, time.time()))
            self.state = new_state
            return True
        return False
```

Slide 7: Implementing Distributed Lock Manager

The Distributed Lock Manager (DLM) coordinates resource access across participants in the 2PC protocol, preventing deadlocks and ensuring transaction isolation.

```python
class DistributedLockManager:
    def __init__(self):
        self.locks = {}
        self.waiting_transactions = {}
        self.lock_timeout = 30  # seconds
        
    def acquire_lock(self, resource_id, transaction_id):
        if resource_id not in self.locks:
            self.locks[resource_id] = {
                'owner': transaction_id,
                'timestamp': time.time()
            }
            return True
            
        current_owner = self.locks[resource_id]['owner']
        if current_owner == transaction_id:
            return True
            
        if self.is_lock_expired(resource_id):
            self.release_lock(resource_id, current_owner)
            return self.acquire_lock(resource_id, transaction_id)
            
        self.add_to_waiting_list(resource_id, transaction_id)
        return False
        
    def is_lock_expired(self, resource_id):
        lock_time = self.locks[resource_id]['timestamp']
        return (time.time() - lock_time) > self.lock_timeout
```

Slide 8: Implementing Deadlock Detection

The deadlock detection system monitors resource allocation and waiting graphs to identify potential deadlocks in distributed transactions using cycle detection algorithms.

```python
from collections import defaultdict

class DeadlockDetector:
    def __init__(self):
        self.wait_for_graph = defaultdict(set)
        
    def add_edge(self, transaction_id, waiting_for_id):
        self.wait_for_graph[transaction_id].add(waiting_for_id)
        
    def remove_edge(self, transaction_id, waiting_for_id):
        if transaction_id in self.wait_for_graph:
            self.wait_for_graph[transaction_id].remove(waiting_for_id)
            
    def detect_cycles(self):
        visited = set()
        path = []
        
        def dfs(node):
            if node in path:
                cycle_start = path.index(node)
                return path[cycle_start:]
            if node in visited:
                return None
                
            visited.add(node)
            path.append(node)
            
            for neighbor in self.wait_for_graph[node]:
                cycle = dfs(neighbor)
                if cycle:
                    return cycle
                    
            path.pop()
            return None
            
        for node in self.wait_for_graph:
            cycle = dfs(node)
            if cycle:
                return cycle
        return None
```

Slide 9: Real-world Implementation: Distributed Banking System

A practical implementation of 2PC in a distributed banking system, showcasing account transfers across multiple banks while maintaining ACID properties.

```python
class BankAccount:
    def __init__(self, account_id, balance):
        self.account_id = account_id
        self.balance = balance
        self.pending_operations = []
        
class DistributedBankSystem:
    def __init__(self):
        self.accounts = {}
        self.transaction_manager = TransactionManager()
        self.lock_manager = DistributedLockManager()
        
    def transfer(self, from_account, to_account, amount):
        txn_id = self.transaction_manager.begin_transaction()
        
        try:
            # Acquire locks
            if not (self.lock_manager.acquire_lock(from_account, txn_id) and 
                   self.lock_manager.acquire_lock(to_account, txn_id)):
                raise Exception("Could not acquire locks")
                
            # Prepare phase
            if not self.prepare_transfer(txn_id, from_account, to_account, amount):
                raise Exception("Prepare phase failed")
                
            # Commit phase
            success = self.transaction_manager.commit_transaction(txn_id)
            
            if success:
                self.apply_transfer(from_account, to_account, amount)
                
            return success
            
        finally:
            self.lock_manager.release_all_locks(txn_id)
```

Slide 10: Performance Monitoring of 2PC

The performance monitoring system tracks key metrics of the Two-Phase Commit protocol, including latency, throughput, and failure rates across distributed nodes.

```python
import statistics
from collections import deque
from time import perf_counter

class TwoPCMetricsCollector:
    def __init__(self, window_size=1000):
        self.prepare_latencies = deque(maxlen=window_size)
        self.commit_latencies = deque(maxlen=window_size)
        self.transaction_outcomes = deque(maxlen=window_size)
        
    def record_prepare_phase(self, duration_ms):
        self.prepare_latencies.append(duration_ms)
        
    def record_commit_phase(self, duration_ms):
        self.commit_latencies.append(duration_ms)
        
    def record_transaction_outcome(self, success):
        self.transaction_outcomes.append(success)
        
    def get_metrics(self):
        metrics = {
            'prepare_phase_avg_ms': statistics.mean(self.prepare_latencies),
            'commit_phase_avg_ms': statistics.mean(self.commit_latencies),
            'success_rate': sum(self.transaction_outcomes) / len(self.transaction_outcomes),
            'total_transactions': len(self.transaction_outcomes)
        }
        return metrics
```

Slide 11: Source Code for Performance Monitoring Results Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

class MetricsVisualizer:
    def __init__(self, metrics_collector):
        self.collector = metrics_collector
        
    def plot_latency_distribution(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Prepare phase latency histogram
        ax1.hist(self.collector.prepare_latencies, bins=30)
        ax1.set_title('Prepare Phase Latency Distribution')
        ax1.set_xlabel('Latency (ms)')
        ax1.set_ylabel('Frequency')
        
        # Commit phase latency histogram
        ax2.hist(self.collector.commit_latencies, bins=30)
        ax2.set_title('Commit Phase Latency Distribution')
        ax2.set_xlabel('Latency (ms)')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
```

Slide 12: Implementing Saga Pattern as 2PC Alternative

The Saga pattern provides a different approach to distributed transactions, implementing compensating transactions for failure recovery without blocking resources.

```python
class SagaTransaction:
    def __init__(self):
        self.steps = []
        self.compensating_actions = []
        
    def add_step(self, action, compensation):
        self.steps.append(action)
        self.compensating_actions.append(compensation)
        
    def execute(self):
        results = []
        for i, step in enumerate(self.steps):
            try:
                result = step()
                results.append(result)
            except Exception as e:
                # Failure occurred, execute compensating actions
                self.rollback(i)
                raise Exception(f"Saga failed at step {i}: {str(e)}")
        return results
        
    def rollback(self, failed_step_index):
        # Execute compensating actions in reverse order
        for i in range(failed_step_index - 1, -1, -1):
            try:
                self.compensating_actions[i]()
            except Exception as e:
                # Log compensation failure
                print(f"Compensation failed at step {i}: {str(e)}")
```

Slide 13: Additional Resources

*   "The Theory of Two-Phase Locking" - [https://dl.acm.org/doi/10.1145/319838.319848](https://dl.acm.org/doi/10.1145/319838.319848)
*   "Making 2PC Work in Partitioned Networks" - [https://www.computer.org/csdl/magazine/dt/2020/01/09070117/1h5XKvOayLG](https://www.computer.org/csdl/magazine/dt/2020/01/09070117/1h5XKvOayLG)
*   "Performance Analysis of Two-Phase Commit" - Search on Google Scholar for recent papers
*   "Consensus Protocols: Two-Phase Commit versus Paxos" - [https://www.researchgate.net/publication/Reference\_Guide](https://www.researchgate.net/publication/Reference_Guide)
*   "Practical Implementation of Distributed Transactions" - Available in ACM Digital Library

\[End of presentation\]

