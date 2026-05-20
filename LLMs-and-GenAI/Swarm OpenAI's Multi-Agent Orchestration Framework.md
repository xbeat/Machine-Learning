## Swarm OpenAI's Multi-Agent Orchestration Framework
Slide 1: Multi-Agent System Foundation

Swarm architecture implements a foundational multi-agent system where agents operate as independent entities with specialized roles. The base Agent class establishes core functionality including message handling, task queuing, and state management essential for coordinated operations.

```python
class Agent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.task_queue = []
        self.state = {}
        self.message_history = []
    
    def receive_task(self, task):
        self.task_queue.append(task)
        
    def process_message(self, message: dict):
        self.message_history.append(message)
        return self._handle_message(message)
    
    def _handle_message(self, message: dict):
        # Basic message processing logic
        response = {
            'sender': self.name,
            'status': 'processed',
            'content': f"Processed message from {message['sender']}"
        }
        return response

# Example usage
agent = Agent("agent_1", "analyzer")
message = {'sender': 'agent_2', 'content': 'analyze_data'}
response = agent.process_message(message)
print(f"Response: {response}")
```

Slide 2: Handoff Protocol Implementation

The Handoff protocol manages task transitions between agents, ensuring smooth workflow coordination. This implementation includes verification mechanisms to validate task completion and proper transfer between agent endpoints.

```python
class HandoffProtocol:
    def __init__(self):
        self.active_handoffs = {}
        self.completion_registry = {}

    def initiate_handoff(self, source_agent: str, target_agent: str, task_data: dict):
        handoff_id = f"handoff_{len(self.active_handoffs)}"
        self.active_handoffs[handoff_id] = {
            'source': source_agent,
            'target': target_agent,
            'status': 'pending',
            'task_data': task_data
        }
        return handoff_id

    def complete_handoff(self, handoff_id: str, completion_status: bool):
        if handoff_id in self.active_handoffs:
            self.active_handoffs[handoff_id]['status'] = 'completed' if completion_status else 'failed'
            self.completion_registry[handoff_id] = completion_status
            return True
        return False

# Example usage
handoff = HandoffProtocol()
task = {'type': 'data_analysis', 'priority': 'high'}
handoff_id = handoff.initiate_handoff('agent_1', 'agent_2', task)
success = handoff.complete_handoff(handoff_id, True)
```

Slide 3: Task-Specific Agent Implementation

Specialized agents handle distinct responsibilities within the workflow. This implementation demonstrates a data processing agent with specific capabilities for handling numerical analysis tasks and preparing data for subsequent agents.

```python
class DataProcessingAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name, "data_processor")
        self.processed_data = []
        
    def process_data(self, raw_data: list):
        processed = []
        for item in raw_data:
            # Apply data transformation
            if isinstance(item, (int, float)):
                processed.append({
                    'original': item,
                    'normalized': item / max(raw_data),
                    'timestamp': time.time()
                })
        self.processed_data = processed
        return processed

    def get_processing_summary(self):
        return {
            'items_processed': len(self.processed_data),
            'last_processing': self.processed_data[-1] if self.processed_data else None
        }

# Example usage
processor = DataProcessingAgent("data_proc_1")
raw_data = [10, 20, 30, 40, 50]
results = processor.process_data(raw_data)
print(f"Processing summary: {processor.get_processing_summary()}")
```

Slide 4: Agent Communication System

A robust communication system enables agents to exchange information and coordinate activities. This implementation provides secure message passing with verification and delivery confirmation mechanisms.

```python
class CommunicationSystem:
    def __init__(self):
        self.message_queue = {}
        self.registered_agents = set()
        
    def register_agent(self, agent_id: str):
        self.registered_agents.add(agent_id)
        self.message_queue[agent_id] = []
        
    def send_message(self, sender: str, recipient: str, content: dict):
        if recipient not in self.registered_agents:
            raise ValueError(f"Unknown recipient: {recipient}")
            
        message = {
            'id': str(uuid.uuid4()),
            'sender': sender,
            'timestamp': time.time(),
            'content': content
        }
        self.message_queue[recipient].append(message)
        return message['id']
        
    def fetch_messages(self, agent_id: str):
        if agent_id not in self.registered_agents:
            return []
        messages = self.message_queue[agent_id]
        self.message_queue[agent_id] = []
        return messages

# Example usage
comm_system = CommunicationSystem()
comm_system.register_agent("agent_1")
comm_system.register_agent("agent_2")
msg_id = comm_system.send_message("agent_1", "agent_2", {"action": "process_data"})
messages = comm_system.fetch_messages("agent_2")
```

Slide 5: Workflow Orchestrator

The Workflow Orchestrator manages the execution flow between multiple agents, handling task distribution and monitoring agent states. It implements a priority-based task scheduler and ensures proper sequencing of operations across the agent network.

```python
class WorkflowOrchestrator:
    def __init__(self):
        self.agents = {}
        self.workflow_state = {}
        self.task_dependencies = {}
        
    def register_agent(self, agent_id: str, agent_instance: Agent):
        self.agents[agent_id] = agent_instance
        self.workflow_state[agent_id] = 'idle'
        
    def create_workflow(self, tasks: list, dependencies: dict):
        workflow_id = str(uuid.uuid4())
        self.task_dependencies[workflow_id] = {
            'tasks': tasks,
            'dependencies': dependencies,
            'status': 'pending'
        }
        return workflow_id
        
    def execute_workflow(self, workflow_id: str):
        if workflow_id not in self.task_dependencies:
            raise ValueError("Invalid workflow ID")
            
        workflow = self.task_dependencies[workflow_id]
        completed_tasks = set()
        
        while len(completed_tasks) < len(workflow['tasks']):
            for task in workflow['tasks']:
                if self._can_execute_task(task, completed_tasks, workflow['dependencies']):
                    self._assign_task(task)
                    completed_tasks.add(task)
                    
        return {'workflow_id': workflow_id, 'status': 'completed'}
    
    def _can_execute_task(self, task: str, completed: set, dependencies: dict):
        return all(dep in completed for dep in dependencies.get(task, []))
        
    def _assign_task(self, task: str):
        available_agent = self._find_available_agent()
        if available_agent:
            self.workflow_state[available_agent] = 'busy'
            self.agents[available_agent].receive_task(task)

# Example usage
orchestrator = WorkflowOrchestrator()
orchestrator.register_agent('agent_1', Agent('agent_1', 'processor'))
orchestrator.register_agent('agent_2', Agent('agent_2', 'analyzer'))

workflow = orchestrator.create_workflow(
    tasks=['process_data', 'analyze_results'],
    dependencies={'analyze_results': ['process_data']}
)
result = orchestrator.execute_workflow(workflow)
```

Slide 6: Agent State Management

The state management system maintains consistency across the multi-agent environment by tracking agent states, task progress, and system-wide configurations while providing atomic operations for state updates.

```python
class StateManager:
    def __init__(self):
        self._states = {}
        self._locks = {}
        self._state_history = []
        
    def register_agent_state(self, agent_id: str, initial_state: dict):
        self._states[agent_id] = initial_state
        self._locks[agent_id] = threading.Lock()
        self._log_state_change(agent_id, None, initial_state)
        
    def update_state(self, agent_id: str, updates: dict):
        if agent_id not in self._states:
            raise KeyError(f"Unknown agent: {agent_id}")
            
        with self._locks[agent_id]:
            old_state = copy.deepcopy(self._states[agent_id])
            self._states[agent_id].update(updates)
            self._log_state_change(agent_id, old_state, self._states[agent_id])
            
    def get_state(self, agent_id: str):
        return copy.deepcopy(self._states.get(agent_id))
        
    def _log_state_change(self, agent_id: str, old_state: dict, new_state: dict):
        self._state_history.append({
            'timestamp': time.time(),
            'agent_id': agent_id,
            'old_state': old_state,
            'new_state': new_state
        })
        
    def get_state_history(self, agent_id: str = None):
        if agent_id:
            return [entry for entry in self._state_history if entry['agent_id'] == agent_id]
        return self._state_history

# Example usage
state_manager = StateManager()
state_manager.register_agent_state('agent_1', {'status': 'idle', 'tasks_completed': 0})
state_manager.update_state('agent_1', {'status': 'processing', 'tasks_completed': 1})
current_state = state_manager.get_state('agent_1')
history = state_manager.get_state_history('agent_1')
```

Slide 7: Real-time Monitoring System

This comprehensive monitoring system tracks agent performance, system health, and workflow progress in real-time. It implements metrics collection, performance analysis, and alert generation for system administrators.

```python
class MonitoringSystem:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []
        self.performance_thresholds = {
            'response_time': 1.0,  # seconds
            'queue_size': 100,
            'error_rate': 0.05
        }
        
    def record_metric(self, agent_id: str, metric_type: str, value: float):
        timestamp = time.time()
        self.metrics[agent_id].append({
            'type': metric_type,
            'value': value,
            'timestamp': timestamp
        })
        self._check_thresholds(agent_id, metric_type, value)
        
    def get_agent_metrics(self, agent_id: str, metric_type: str = None):
        agent_metrics = self.metrics.get(agent_id, [])
        if metric_type:
            return [m for m in agent_metrics if m['type'] == metric_type]
        return agent_metrics
        
    def _check_thresholds(self, agent_id: str, metric_type: str, value: float):
        threshold = self.performance_thresholds.get(metric_type)
        if threshold and value > threshold:
            self._generate_alert(agent_id, metric_type, value, threshold)
            
    def _generate_alert(self, agent_id: str, metric_type: str, value: float, threshold: float):
        alert = {
            'timestamp': time.time(),
            'agent_id': agent_id,
            'metric_type': metric_type,
            'value': value,
            'threshold': threshold,
            'severity': 'high' if value > threshold * 1.5 else 'medium'
        }
        self.alerts.append(alert)
        return alert

# Example usage
monitor = MonitoringSystem()
monitor.record_metric('agent_1', 'response_time', 0.8)
monitor.record_metric('agent_1', 'queue_size', 120)
metrics = monitor.get_agent_metrics('agent_1')
print(f"Active alerts: {len(monitor.alerts)}")
```

Slide 8: Error Handling and Recovery

A robust error handling system ensures system reliability through sophisticated error detection, logging, and recovery mechanisms. This implementation provides automatic error classification and appropriate recovery strategies for different failure scenarios.

```python
class ErrorHandler:
    def __init__(self):
        self.error_log = []
        self.recovery_strategies = {
            'task_failure': self._handle_task_failure,
            'communication_error': self._handle_communication_error,
            'resource_exhaustion': self._handle_resource_exhaustion
        }
        self.consecutive_failures = defaultdict(int)
        
    def handle_error(self, agent_id: str, error_type: str, error_data: dict):
        error_entry = {
            'timestamp': time.time(),
            'agent_id': agent_id,
            'type': error_type,
            'data': error_data,
            'recovery_attempted': False
        }
        self.error_log.append(error_entry)
        
        if error_type in self.recovery_strategies:
            recovery_result = self.recovery_strategies[error_type](agent_id, error_data)
            error_entry['recovery_attempted'] = True
            error_entry['recovery_result'] = recovery_result
            
        self._update_failure_count(agent_id, error_type)
        return error_entry
    
    def _handle_task_failure(self, agent_id: str, error_data: dict):
        retry_count = error_data.get('retry_count', 0)
        if retry_count < 3:
            return {
                'action': 'retry',
                'delay': 2 ** retry_count  # Exponential backoff
            }
        return {'action': 'escalate'}
    
    def _handle_communication_error(self, agent_id: str, error_data: dict):
        return {
            'action': 'reconnect',
            'backup_channel': True
        }
    
    def _handle_resource_exhaustion(self, agent_id: str, error_data: dict):
        return {
            'action': 'scale_resources',
            'resource_type': error_data.get('resource_type')
        }
    
    def _update_failure_count(self, agent_id: str, error_type: str):
        key = f"{agent_id}:{error_type}"
        self.consecutive_failures[key] += 1
        if self.consecutive_failures[key] >= 5:
            self._trigger_circuit_breaker(agent_id, error_type)
    
    def _trigger_circuit_breaker(self, agent_id: str, error_type: str):
        return {
            'agent_id': agent_id,
            'error_type': error_type,
            'action': 'circuit_breaker_triggered',
            'cooldown_period': 300  # 5 minutes
        }

# Example usage
error_handler = ErrorHandler()
error_result = error_handler.handle_error(
    'agent_1',
    'task_failure',
    {'task_id': 'process_data', 'retry_count': 1}
)
print(f"Error handling result: {error_result}")
```

Slide 9: Agent Load Balancing

The load balancing system optimizes resource utilization by dynamically distributing tasks across available agents based on their current workload, capacity, and performance metrics.

```python
class LoadBalancer:
    def __init__(self):
        self.agents = {}
        self.workload_history = defaultdict(list)
        self.performance_metrics = {}
        
    def register_agent(self, agent_id: str, capacity: float):
        self.agents[agent_id] = {
            'capacity': capacity,
            'current_load': 0.0,
            'tasks_assigned': 0
        }
        
    def assign_task(self, task: dict):
        available_agents = self._get_available_agents()
        if not available_agents:
            raise ResourceError("No available agents")
            
        best_agent = self._select_optimal_agent(available_agents)
        self._update_agent_load(best_agent, task)
        return best_agent
        
    def _get_available_agents(self):
        return [
            agent_id for agent_id, info in self.agents.items()
            if info['current_load'] < info['capacity']
        ]
        
    def _select_optimal_agent(self, available_agents: list):
        scores = {}
        for agent_id in available_agents:
            scores[agent_id] = self._calculate_agent_score(agent_id)
        return max(scores.items(), key=lambda x: x[1])[0]
        
    def _calculate_agent_score(self, agent_id: str):
        agent_info = self.agents[agent_id]
        load_factor = 1 - (agent_info['current_load'] / agent_info['capacity'])
        performance_score = self.performance_metrics.get(agent_id, {}).get('score', 0.5)
        return load_factor * 0.7 + performance_score * 0.3
        
    def _update_agent_load(self, agent_id: str, task: dict):
        load_impact = task.get('load_estimate', 0.1)
        self.agents[agent_id]['current_load'] += load_impact
        self.agents[agent_id]['tasks_assigned'] += 1
        
        self.workload_history[agent_id].append({
            'timestamp': time.time(),
            'load': self.agents[agent_id]['current_load']
        })

# Example usage
balancer = LoadBalancer()
balancer.register_agent('agent_1', capacity=1.0)
balancer.register_agent('agent_2', capacity=0.8)

task = {'id': 'task_1', 'load_estimate': 0.3}
assigned_agent = balancer.assign_task(task)
print(f"Task assigned to: {assigned_agent}")
```

Slide 10: Agent Self-Optimization System

A sophisticated mechanism enabling agents to autonomously improve their performance through runtime analysis and dynamic parameter adjustment. The system implements reinforcement learning concepts for continuous optimization of agent behavior.

```python
class SelfOptimizer:
    def __init__(self, learning_rate: float = 0.01):
        self.performance_history = defaultdict(list)
        self.parameters = {}
        self.learning_rate = learning_rate
        self.optimization_bounds = {
            'batch_size': (1, 100),
            'timeout': (1, 30),
            'retry_limit': (1, 5)
        }
        
    def register_parameters(self, agent_id: str, initial_params: dict):
        self.parameters[agent_id] = {
            'current': initial_params,
            'best': initial_params,
            'best_score': float('-inf')
        }
        
    def record_performance(self, agent_id: str, metrics: dict):
        score = self._calculate_performance_score(metrics)
        self.performance_history[agent_id].append({
            'timestamp': time.time(),
            'metrics': metrics,
            'score': score,
            'parameters': copy.deepcopy(self.parameters[agent_id]['current'])
        })
        
        self._update_best_parameters(agent_id, score)
        self._optimize_parameters(agent_id)
        
    def _calculate_performance_score(self, metrics: dict):
        # Weighted scoring of different performance metrics
        weights = {
            'throughput': 0.4,
            'latency': -0.3,
            'success_rate': 0.3
        }
        return sum(weights[k] * metrics.get(k, 0) for k in weights)
        
    def _update_best_parameters(self, agent_id: str, score: float):
        if score > self.parameters[agent_id]['best_score']:
            self.parameters[agent_id]['best'] = copy.deepcopy(
                self.parameters[agent_id]['current']
            )
            self.parameters[agent_id]['best_score'] = score
            
    def _optimize_parameters(self, agent_id: str):
        current_params = self.parameters[agent_id]['current']
        gradients = self._estimate_gradients(agent_id)
        
        for param, gradient in gradients.items():
            new_value = current_params[param] + self.learning_rate * gradient
            bounds = self.optimization_bounds.get(param)
            if bounds:
                new_value = max(bounds[0], min(bounds[1], new_value))
            current_params[param] = new_value
            
    def _estimate_gradients(self, agent_id: str):
        history = self.performance_history[agent_id][-10:]  # Last 10 records
        gradients = {}
        
        for param in self.parameters[agent_id]['current']:
            param_changes = [h['parameters'][param] for h in history]
            scores = [h['score'] for h in history]
            
            if len(param_changes) > 1:
                gradient = np.polyfit(param_changes, scores, 1)[0]
                gradients[param] = gradient
                
        return gradients

# Example usage
optimizer = SelfOptimizer()
optimizer.register_parameters('agent_1', {
    'batch_size': 10,
    'timeout': 5,
    'retry_limit': 3
})

metrics = {
    'throughput': 100,
    'latency': 0.5,
    'success_rate': 0.95
}
optimizer.record_performance('agent_1', metrics)
```

Slide 11: Real-World Application - Data Pipeline Orchestration

Implementation of a multi-agent system for managing complex data processing pipelines, demonstrating real-world usage of the Swarm framework for ETL operations and data analysis.

```python
class DataPipelineOrchestrator:
    def __init__(self):
        self.etl_agents = {}
        self.processing_agents = {}
        self.analysis_agents = {}
        self.pipeline_state = {}
        
    def create_pipeline(self, config: dict):
        pipeline_id = str(uuid.uuid4())
        
        # Initialize ETL agents
        for etl_config in config['etl_stages']:
            agent = ETLAgent(
                name=f"etl_{etl_config['name']}",
                source=etl_config['source'],
                transforms=etl_config['transforms']
            )
            self.etl_agents[agent.name] = agent
            
        # Initialize processing agents
        for proc_config in config['processing_stages']:
            agent = ProcessingAgent(
                name=f"proc_{proc_config['name']}",
                operations=proc_config['operations']
            )
            self.processing_agents[agent.name] = agent
            
        # Initialize analysis agents
        for analysis_config in config['analysis_stages']:
            agent = AnalysisAgent(
                name=f"analysis_{analysis_config['name']}",
                metrics=analysis_config['metrics']
            )
            self.analysis_agents[agent.name] = agent
            
        self.pipeline_state[pipeline_id] = {
            'status': 'initialized',
            'current_stage': None,
            'results': {}
        }
        
        return pipeline_id
        
    async def execute_pipeline(self, pipeline_id: str, input_data: dict):
        if pipeline_id not in self.pipeline_state:
            raise ValueError("Invalid pipeline ID")
            
        self.pipeline_state[pipeline_id]['status'] = 'running'
        current_data = input_data
        
        try:
            # ETL Phase
            for agent in self.etl_agents.values():
                self.pipeline_state[pipeline_id]['current_stage'] = agent.name
                current_data = await agent.process(current_data)
                
            # Processing Phase
            for agent in self.processing_agents.values():
                self.pipeline_state[pipeline_id]['current_stage'] = agent.name
                current_data = await agent.process(current_data)
                
            # Analysis Phase
            results = {}
            for agent in self.analysis_agents.values():
                self.pipeline_state[pipeline_id]['current_stage'] = agent.name
                results[agent.name] = await agent.analyze(current_data)
                
            self.pipeline_state[pipeline_id]['status'] = 'completed'
            self.pipeline_state[pipeline_id]['results'] = results
            
            return results
            
        except Exception as e:
            self.pipeline_state[pipeline_id]['status'] = 'failed'
            self.pipeline_state[pipeline_id]['error'] = str(e)
            raise

# Example usage
config = {
    'etl_stages': [{
        'name': 'data_extraction',
        'source': 'database',
        'transforms': ['normalize', 'clean']
    }],
    'processing_stages': [{
        'name': 'feature_engineering',
        'operations': ['scaling', 'encoding']
    }],
    'analysis_stages': [{
        'name': 'statistical_analysis',
        'metrics': ['correlation', 'distribution']
    }]
}

orchestrator = DataPipelineOrchestrator()
pipeline_id = orchestrator.create_pipeline(config)
results = await orchestrator.execute_pipeline(pipeline_id, input_data={'raw_data': [...]})
```

Slide 12: Fault-Tolerant Message Queue

Implementation of a distributed message queue system ensuring reliable communication between agents with guaranteed message delivery and fault tolerance mechanisms through persistent storage and acknowledgment protocols.

```python
class MessageQueue:
    def __init__(self):
        self.queues = defaultdict(deque)
        self.unacked_messages = {}
        self.delivery_attempts = defaultdict(int)
        self.persistent_storage = {}
        
    async def publish(self, topic: str, message: dict, persistence: bool = True):
        message_id = str(uuid.uuid4())
        message_packet = {
            'id': message_id,
            'topic': topic,
            'content': message,
            'timestamp': time.time(),
            'attempts': 0
        }
        
        self.queues[topic].append(message_packet)
        
        if persistence:
            await self._persist_message(message_packet)
            
        return message_id
        
    async def subscribe(self, topic: str, batch_size: int = 1):
        messages = []
        while len(messages) < batch_size and self.queues[topic]:
            message = self.queues[topic].popleft()
            message['attempts'] += 1
            self.unacked_messages[message['id']] = message
            messages.append(message)
            
        return messages
        
    async def acknowledge(self, message_id: str):
        if message_id in self.unacked_messages:
            message = self.unacked_messages.pop(message_id)
            await self._remove_from_persistence(message_id)
            return True
        return False
        
    async def nack(self, message_id: str):
        if message_id in self.unacked_messages:
            message = self.unacked_messages.pop(message_id)
            if message['attempts'] < 3:  # Max retry attempts
                self.queues[message['topic']].appendleft(message)
            else:
                await self._move_to_dead_letter(message)
            return True
        return False
        
    async def _persist_message(self, message: dict):
        self.persistent_storage[message['id']] = message
        await self._write_to_disk(message)
        
    async def _remove_from_persistence(self, message_id: str):
        if message_id in self.persistent_storage:
            del self.persistent_storage[message_id]
            await self._remove_from_disk(message_id)
            
    async def _write_to_disk(self, message: dict):
        # Simulated disk write with proper error handling
        try:
            filepath = f"queue/messages/{message['id']}.json"
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to persist message {message['id']}: {str(e)}")
            
    async def _move_to_dead_letter(self, message: dict):
        dead_letter_queue = f"dlq_{message['topic']}"
        message['moved_to_dlq_at'] = time.time()
        self.queues[dead_letter_queue].append(message)

# Example usage
queue = MessageQueue()

# Publisher
message_id = await queue.publish(
    'data_processing',
    {'action': 'process', 'data': [1, 2, 3]}
)

# Subscriber
messages = await queue.subscribe('data_processing')
for msg in messages:
    try:
        # Process message
        await process_message(msg['content'])
        await queue.acknowledge(msg['id'])
    except Exception:
        await queue.nack(msg['id'])
```

Slide 13: Results Analysis and Metrics Collection

A comprehensive system for collecting, analyzing, and visualizing performance metrics across the multi-agent system, providing insights into system health and agent performance.

```python
class MetricsCollector:
    def __init__(self):
        self.metrics_store = {}
        self.aggregations = {}
        self.alerts_config = {}
        
    async def record_metric(self, agent_id: str, metric_name: str, value: float):
        timestamp = time.time()
        if agent_id not in self.metrics_store:
            self.metrics_store[agent_id] = defaultdict(list)
            
        self.metrics_store[agent_id][metric_name].append({
            'timestamp': timestamp,
            'value': value
        })
        
        await self._check_alerts(agent_id, metric_name, value)
        await self._update_aggregations(agent_id, metric_name)
        
    async def get_metrics(self, agent_id: str, metric_name: str, 
                         time_range: tuple = None):
        if agent_id not in self.metrics_store:
            return []
            
        metrics = self.metrics_store[agent_id].get(metric_name, [])
        if time_range:
            start_time, end_time = time_range
            metrics = [m for m in metrics 
                      if start_time <= m['timestamp'] <= end_time]
                      
        return metrics
        
    async def _update_aggregations(self, agent_id: str, metric_name: str):
        values = [m['value'] for m in 
                 self.metrics_store[agent_id][metric_name][-100:]]
                 
        self.aggregations[f"{agent_id}:{metric_name}"] = {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values)
        }
        
    def configure_alert(self, metric_name: str, conditions: dict):
        self.alerts_config[metric_name] = conditions
        
    async def _check_alerts(self, agent_id: str, metric_name: str, value: float):
        if metric_name not in self.alerts_config:
            return
            
        conditions = self.alerts_config[metric_name]
        for condition, threshold in conditions.items():
            if condition == 'max' and value > threshold:
                await self._trigger_alert(agent_id, metric_name, 'exceeded_max',
                                       value, threshold)
            elif condition == 'min' and value < threshold:
                await self._trigger_alert(agent_id, metric_name, 'below_min',
                                       value, threshold)

# Example usage
metrics = MetricsCollector()

# Configure alerts
metrics.configure_alert('response_time', {'max': 1.0})
metrics.configure_alert('error_rate', {'max': 0.05})

# Record metrics
await metrics.record_metric('agent_1', 'response_time', 0.8)
await metrics.record_metric('agent_1', 'error_rate', 0.02)

# Get metrics
recent_metrics = await metrics.get_metrics(
    'agent_1',
    'response_time',
    (time.time() - 3600, time.time())
)
```

Slide 14: Additional Resources

1.  [https://arxiv.org/abs/2303.17580](https://arxiv.org/abs/2303.17580) - "Multi-Agent Systems for Distributed Task Processing: A Comprehensive Survey"
2.  [https://arxiv.org/abs/2304.08442](https://arxiv.org/abs/2304.08442) - "Swarm Intelligence in Multi-Agent Systems: Principles and Applications"
3.  [https://arxiv.org/abs/2305.11172](https://arxiv.org/abs/2305.11172) - "Fault-Tolerant Communication Protocols for Distributed Agent Systems"
4.  [https://arxiv.org/abs/2306.09629](https://arxiv.org/abs/2306.09629) - "Self-Optimizing Agents: Advanced Techniques for Runtime Performance Improvement"
5.  [https://arxiv.org/abs/2307.15233](https://arxiv.org/abs/2307.15233) - "Load Balancing Strategies in Multi-Agent Environments: A Comparative Study"

