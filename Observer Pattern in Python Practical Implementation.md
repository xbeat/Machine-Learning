## Observer Pattern in Python Practical Implementation
Slide 1: Observer Pattern Fundamentals

The Observer pattern enables a subscription mechanism where multiple objects (observers) monitor changes in another object (subject). This fundamental design pattern is crucial for implementing event handling systems and maintaining loose coupling between components.

```python
from abc import ABC, abstractmethod
from typing import List

class Subject:
    def __init__(self):
        self._observers: List[Observer] = []
        self._state = None

    def attach(self, observer) -> None:
        self._observers.append(observer)

    def detach(self, observer) -> None:
        self._observers.remove(observer)

    def notify(self) -> None:
        for observer in self._observers:
            observer.update(self._state)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self.notify()

class Observer(ABC):
    @abstractmethod
    def update(self, state) -> None:
        pass
```

Slide 2: Simple Stock Market Implementation

A practical implementation of the Observer pattern simulating a stock market system where multiple traders (observers) monitor price changes of a specific stock (subject), demonstrating real-time updates and automated trading responses.

```python
class Stock(Subject):
    def __init__(self, symbol: str, price: float):
        super().__init__()
        self.symbol = symbol
        self._price = price
    
    @property
    def price(self) -> float:
        return self._price
    
    @price.setter
    def price(self, new_price: float) -> None:
        self._price = new_price
        self.state = (self.symbol, new_price)

class Trader(Observer):
    def __init__(self, name: str):
        self.name = name
        self.stocks_watching = {}

    def update(self, state) -> None:
        symbol, price = state
        print(f"Trader {self.name} notified: {symbol} price changed to ${price}")
        self._execute_trading_strategy(symbol, price)

    def _execute_trading_strategy(self, symbol: str, price: float) -> None:
        # Simple trading strategy based on price thresholds
        if price < 50.0:
            print(f"{self.name} - BUY signal for {symbol} at ${price}")
        elif price > 100.0:
            print(f"{self.name} - SELL signal for {symbol} at ${price}")
```

Slide 3: Event Management System

This implementation showcases an event management system where multiple event handlers observe and respond to different types of events, demonstrating the flexibility of the Observer pattern in handling complex event-driven architectures.

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Set

class EventType(Enum):
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    DATA_CHANGED = "data_changed"

@dataclass
class Event:
    type: EventType
    data: dict

class EventManager(Subject):
    def __init__(self):
        super().__init__()
        self._event_observers: Dict[EventType, Set[Observer]] = {
            event_type: set() for event_type in EventType
        }

    def attach(self, event_type: EventType, observer: Observer) -> None:
        self._event_observers[event_type].add(observer)

    def detach(self, event_type: EventType, observer: Observer) -> None:
        self._event_observers[event_type].discard(observer)

    def notify(self, event: Event) -> None:
        for observer in self._event_observers[event.type]:
            observer.update(event)

    def dispatch_event(self, event: Event) -> None:
        self.state = event
        self.notify(event)
```

Slide 4: Event Handlers Implementation

Implementing specific event handlers that respond to different types of events in the system, showcasing how the Observer pattern can be used to create a modular and extensible event handling system.

```python
class SecurityMonitor(Observer):
    def update(self, event: Event) -> None:
        if event.type == EventType.USER_LOGIN:
            self._log_login_attempt(event.data)
        elif event.type == EventType.USER_LOGOUT:
            self._log_logout(event.data)

    def _log_login_attempt(self, data: dict) -> None:
        print(f"Security Log: Login attempt by {data['username']} from {data['ip']}")
        if data.get('failed', False):
            print(f"WARNING: Failed login attempt for {data['username']}")

    def _log_logout(self, data: dict) -> None:
        print(f"Security Log: User {data['username']} logged out")

class DataAnalytics(Observer):
    def update(self, event: Event) -> None:
        if event.type == EventType.DATA_CHANGED:
            self._analyze_data_changes(event.data)

    def _analyze_data_changes(self, data: dict) -> None:
        print(f"Analytics: Processing data change in {data['entity']}")
        print(f"Change type: {data['change_type']}")
        print(f"Timestamp: {data['timestamp']}")
```

Slide 5: Real-time Weather Monitoring System

A comprehensive implementation of a weather monitoring system that demonstrates the Observer pattern in handling real-time data updates from multiple weather stations and notifying various subscribers about weather changes.

```python
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class WeatherData:
    temperature: float
    humidity: float
    pressure: float
    timestamp: datetime
    station_id: str

class WeatherStation(Subject):
    def __init__(self, station_id: str):
        super().__init__()
        self.station_id = station_id
        self._current_data: Optional[WeatherData] = None

    def set_measurements(self, temp: float, humidity: float, 
                        pressure: float) -> None:
        self._current_data = WeatherData(
            temperature=temp,
            humidity=humidity,
            pressure=pressure,
            timestamp=datetime.now(),
            station_id=self.station_id
        )
        self.state = self._current_data
        self.notify()

class WeatherDisplay(Observer):
    def update(self, weather_data: WeatherData) -> None:
        print(f"\nWeather Update from Station {weather_data.station_id}")
        print(f"Temperature: {weather_data.temperature}°C")
        print(f"Humidity: {weather_data.humidity}%")
        print(f"Pressure: {weather_data.pressure} hPa")
        print(f"Timestamp: {weather_data.timestamp}")
```

Slide 6: Advanced Weather Analysis Integration

Extending the weather monitoring system with sophisticated analysis capabilities, this implementation adds statistical processing and alert generation based on weather patterns, demonstrating advanced observer interactions.

```python
from statistics import mean, stdev
from collections import deque
from typing import Dict, Deque

class WeatherAnalyzer(Observer):
    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        self.readings: Dict[str, Deque[WeatherData]] = {}
        self.alert_thresholds = {
            'temp_change': 5.0,  # °C
            'pressure_change': 10.0,  # hPa
            'humidity_change': 20.0  # %
        }

    def update(self, weather_data: WeatherData) -> None:
        if weather_data.station_id not in self.readings:
            self.readings[weather_data.station_id] = deque(maxlen=self.window_size)
        
        self.readings[weather_data.station_id].append(weather_data)
        self._analyze_trends(weather_data.station_id)

    def _analyze_trends(self, station_id: str) -> None:
        if len(self.readings[station_id]) < 2:
            return

        recent_data = list(self.readings[station_id])
        temp_trend = [d.temperature for d in recent_data]
        pressure_trend = [d.pressure for d in recent_data]
        
        temp_change = temp_trend[-1] - temp_trend[-2]
        pressure_change = pressure_trend[-1] - pressure_trend[-2]

        if abs(temp_change) > self.alert_thresholds['temp_change']:
            print(f"ALERT: Significant temperature change detected at station {station_id}")
            print(f"Change: {temp_change:+.1f}°C")

        if abs(pressure_change) > self.alert_thresholds['pressure_change']:
            print(f"ALERT: Significant pressure change detected at station {station_id}")
            print(f"Change: {pressure_change:+.1f} hPa")
```

Slide 7: File System Monitor Implementation

A practical implementation of the Observer pattern for monitoring file system changes, demonstrating real-time notifications for file creation, modification, and deletion events.

```python
from pathlib import Path
from typing import Set, Dict
from datetime import datetime
import hashlib

class FileSystemEvent:
    def __init__(self, event_type: str, file_path: Path, details: dict = None):
        self.event_type = event_type
        self.file_path = file_path
        self.timestamp = datetime.now()
        self.details = details or {}

class FileSystemMonitor(Subject):
    def __init__(self, directory: Path):
        super().__init__()
        self.directory = directory
        self.file_hashes: Dict[Path, str] = {}
        self.watched_files: Set[Path] = set()
        self._scan_directory()

    def _scan_directory(self) -> None:
        current_files = set(self.directory.glob('**/*'))
        
        # Check for new files
        new_files = current_files - self.watched_files
        for file_path in new_files:
            if file_path.is_file():
                self.watched_files.add(file_path)
                self._update_file_hash(file_path)
                self.state = FileSystemEvent('created', file_path)
                self.notify()

        # Check for deleted files
        deleted_files = self.watched_files - current_files
        for file_path in deleted_files:
            self.watched_files.remove(file_path)
            self.file_hashes.pop(file_path, None)
            self.state = FileSystemEvent('deleted', file_path)
            self.notify()

    def _update_file_hash(self, file_path: Path) -> None:
        if file_path.is_file():
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                if file_path in self.file_hashes:
                    if self.file_hashes[file_path] != file_hash:
                        self.state = FileSystemEvent('modified', file_path)
                        self.notify()
                self.file_hashes[file_path] = file_hash
```

Slide 8: File System Event Handlers

Implementation of specific handlers for different types of file system events, showcasing how the Observer pattern can be used to create modular and extensible file monitoring systems.

```python
class FileLogger(Observer):
    def __init__(self, log_file: Path):
        self.log_file = log_file

    def update(self, event: FileSystemEvent) -> None:
        log_entry = (
            f"[{event.timestamp.isoformat()}] "
            f"{event.event_type.upper()}: {event.file_path}"
        )
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')

class BackupManager(Observer):
    def __init__(self, backup_dir: Path):
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(exist_ok=True)

    def update(self, event: FileSystemEvent) -> None:
        if event.event_type in ['created', 'modified']:
            self._create_backup(event.file_path)

    def _create_backup(self, file_path: Path) -> None:
        backup_path = self.backup_dir / f"{file_path.name}_{datetime.now().timestamp()}"
        with open(file_path, 'rb') as src, open(backup_path, 'wb') as dst:
            dst.write(src.read())
        print(f"Created backup: {backup_path}")

class SecurityAuditor(Observer):
    def __init__(self, sensitive_extensions: Set[str] = None):
        self.sensitive_extensions = sensitive_extensions or {'.pdf', '.doc', '.docx'}

    def update(self, event: FileSystemEvent) -> None:
        if event.file_path.suffix in self.sensitive_extensions:
            print(f"Security Alert: Sensitive file {event.event_type}: {event.file_path}")
```

Slide 9: Database Change Observer System

A sophisticated implementation of the Observer pattern for monitoring database changes, supporting real-time notifications for insert, update, and delete operations while maintaining transaction consistency.

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

@dataclass
class DatabaseEvent:
    operation: str  # 'INSERT', 'UPDATE', 'DELETE'
    table: str
    record_id: Any
    data: Dict[str, Any]
    timestamp: datetime
    transaction_id: str

class DatabaseMonitor(Subject):
    def __init__(self):
        super().__init__()
        self._transaction_log: List[DatabaseEvent] = []
        self._active_transaction: Optional[str] = None
    
    def begin_transaction(self) -> str:
        self._active_transaction = f"txn_{datetime.now().timestamp()}"
        return self._active_transaction
    
    def commit_transaction(self) -> None:
        if self._active_transaction:
            for event in self._transaction_log:
                self.state = event
                self.notify()
            self._transaction_log.clear()
            self._active_transaction = None
    
    def record_change(self, operation: str, table: str, 
                     record_id: Any, data: Dict[str, Any]) -> None:
        event = DatabaseEvent(
            operation=operation,
            table=table,
            record_id=record_id,
            data=data,
            timestamp=datetime.now(),
            transaction_id=self._active_transaction
        )
        if self._active_transaction:
            self._transaction_log.append(event)
        else:
            self.state = event
            self.notify()
```

Slide 10: Database Change Handlers

Implementation of specialized handlers for database changes, demonstrating how the Observer pattern can be used to implement audit logging, caching, and real-time analytics.

```python
class DatabaseAuditor(Observer):
    def __init__(self, audit_file: str):
        self.audit_file = audit_file
        
    def update(self, event: DatabaseEvent) -> None:
        audit_entry = {
            'timestamp': event.timestamp.isoformat(),
            'operation': event.operation,
            'table': event.table,
            'record_id': str(event.record_id),
            'transaction_id': event.transaction_id,
            'data': event.data
        }
        with open(self.audit_file, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')

class CacheInvalidator(Observer):
    def __init__(self):
        self.cache: Dict[str, Dict[Any, Any]] = {}
        
    def update(self, event: DatabaseEvent) -> None:
        cache_key = f"{event.table}_{event.record_id}"
        if event.operation in ['UPDATE', 'DELETE']:
            self.cache.pop(cache_key, None)
            print(f"Cache invalidated for: {cache_key}")
        elif event.operation == 'INSERT':
            self.cache[cache_key] = event.data
            print(f"Cache updated for: {cache_key}")

class RealTimeAnalytics(Observer):
    def __init__(self):
        self.operation_counts: Dict[str, int] = {
            'INSERT': 0, 'UPDATE': 0, 'DELETE': 0
        }
        self.table_stats: Dict[str, Dict[str, int]] = {}
        
    def update(self, event: DatabaseEvent) -> None:
        # Update operation counts
        self.operation_counts[event.operation] += 1
        
        # Update table-specific stats
        if event.table not in self.table_stats:
            self.table_stats[event.table] = {
                'INSERT': 0, 'UPDATE': 0, 'DELETE': 0
            }
        self.table_stats[event.table][event.operation] += 1
        
        # Print real-time analytics
        print(f"\nReal-time Database Analytics:")
        print(f"Total operations: {sum(self.operation_counts.values())}")
        print(f"Operations breakdown: {self.operation_counts}")
        print(f"Table stats: {self.table_stats}")
```

Slide 11: Message Queue Observer System

An implementation of the Observer pattern for a message queue system, enabling asynchronous communication between components while maintaining message ordering and delivery guarantees.

```python
from queue import PriorityQueue
from dataclasses import dataclass
from typing import Any, Callable
from threading import Lock
import time

@dataclass(order=True)
class Message:
    priority: int
    timestamp: float
    topic: str
    payload: Any
    
class MessageBroker(Subject):
    def __init__(self):
        super().__init__()
        self._queues: Dict[str, PriorityQueue] = {}
        self._subscribers: Dict[str, List[Observer]] = {}
        self._lock = Lock()
    
    def create_topic(self, topic: str) -> None:
        with self._lock:
            if topic not in self._queues:
                self._queues[topic] = PriorityQueue()
                self._subscribers[topic] = []
    
    def publish(self, topic: str, payload: Any, priority: int = 1) -> None:
        if topic not in self._queues:
            self.create_topic(topic)
            
        message = Message(
            priority=priority,
            timestamp=time.time(),
            topic=topic,
            payload=payload
        )
        
        self._queues[topic].put(message)
        self.state = message
        self.notify()
    
    def subscribe(self, topic: str, observer: Observer) -> None:
        with self._lock:
            if topic not in self._subscribers:
                self.create_topic(topic)
            self._subscribers[topic].append(observer)
```

Slide 12: Message Queue Handlers

Specialized handlers for processing messages from different topics, demonstrating the Observer pattern's utility in building robust message processing systems.

```python
class MessageProcessor(Observer):
    def __init__(self, processor_id: str, 
                 processing_func: Callable[[Any], None]):
        self.processor_id = processor_id
        self.processing_func = processing_func
        self.processed_count = 0
        
    def update(self, message: Message) -> None:
        print(f"Processor {self.processor_id} received message on topic: {message.topic}")
        try:
            self.processing_func(message.payload)
            self.processed_count += 1
            print(f"Successfully processed message {self.processed_count}")
        except Exception as e:
            print(f"Error processing message: {str(e)}")

class MessageLogger(Observer):
    def __init__(self, log_file: str):
        self.log_file = log_file
        
    def update(self, message: Message) -> None:
        log_entry = (
            f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(message.timestamp))}] "
            f"Topic: {message.topic}, Priority: {message.priority}, "
            f"Payload: {message.payload}"
        )
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')

class MessageAnalytics(Observer):
    def __init__(self):
        self.topic_stats = {}
        self.priority_stats = {}
        
    def update(self, message: Message) -> None:
        # Update topic statistics
        self.topic_stats[message.topic] = self.topic_stats.get(message.topic, 0) + 1
        
        # Update priority statistics
        self.priority_stats[message.priority] = (
            self.priority_stats.get(message.priority, 0) + 1
        )
        
        # Calculate and print analytics
        print("\nMessage Queue Analytics:")
        print(f"Messages per topic: {self.topic_stats}")
        print(f"Messages per priority: {self.priority_stats}")
```

Slide 13: Results and Performance Metrics

Comprehensive demonstration of the Observer pattern implementations with performance metrics and system behavior analysis.

```python
def run_performance_test():
    # Message Broker Performance Test
    broker = MessageBroker()
    logger = MessageLogger("message_queue.log")
    analytics = MessageAnalytics()
    
    # Add observers
    broker.subscribe("orders", logger)
    broker.subscribe("orders", analytics)
    
    # Performance metrics
    start_time = time.time()
    message_count = 1000
    
    # Generate test messages
    for i in range(message_count):
        broker.publish(
            topic="orders",
            payload={"order_id": i, "amount": i * 10.5},
            priority=i % 3 + 1
        )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nPerformance Metrics:")
    print(f"Total messages processed: {message_count}")
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Average time per message: {(processing_time/message_count)*1000:.2f} ms")
    print(f"Messages per second: {message_count/processing_time:.2f}")
    
    # Memory usage analysis
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    run_performance_test()
```

Slide 14: Additional Resources

*   Design Patterns: Elements of Reusable Object-Oriented Software
    *   Search: "Gang of Four Design Patterns Book"
*   Python Design Patterns Guide
    *   [https://python-patterns.guide/](https://python-patterns.guide/)
*   Advanced Python Design Patterns
    *   [https://refactoring.guru/design-patterns/python](https://refactoring.guru/design-patterns/python)
*   Real-world Observer Pattern Applications
    *   Search: "Observer Pattern in Large Scale Systems"
*   Event-Driven Architecture with Python
    *   Search: "Python Event-Driven Programming Best Practices"

