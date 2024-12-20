## Docker Cheat Sheet for Python Developers

Slide 1: Basic Docker Commands in Python

Docker SDK for Python enables programmatic control of containers through intuitive APIs. This approach provides automation capabilities for container lifecycle management, allowing systematic deployment and monitoring of containerized applications through Python scripts.

```python
import docker

# Initialize Docker client
client = docker.from_client()

# List all containers
containers = client.containers.list(all=True)
for container in containers:
    print(f"Container ID: {container.id}")
    print(f"Image: {container.image.tags}")
    print(f"Status: {container.status}")

# Pull an image
image = client.images.pull('python:3.9-slim')
print(f"Pulled image: {image.tags}")
```

Slide 2: Container Creation and Management

The Docker Python SDK provides comprehensive container management capabilities. This includes creating containers with specific configurations, starting and stopping them programmatically, and handling container lifecycle events through event listeners.

```python
import docker
from docker.types import Mount

client = docker.from_client()

# Create and run a container
container = client.containers.run(
    'python:3.9-slim',
    name='python_container',
    command='python -c "print(\'Hello from container\')"',
    detach=True,
    mounts=[Mount(
        target='/app',
        source='/local/path',
        type='bind'
    )]
)

# Wait for container to finish and get logs
result = container.wait()
logs = container.logs().decode('utf-8')
print(f"Exit Code: {result['StatusCode']}")
print(f"Output: {logs}")
```

Slide 3: Docker Network Management

Understanding Docker networking is crucial for container orchestration. This implementation demonstrates creating custom networks, connecting containers, and managing network configurations through Python, enabling sophisticated multi-container applications.

```python
import docker

client = docker.from_client()

# Create a custom network
network = client.networks.create(
    name='my_network',
    driver='bridge',
    ipam=docker.types.IPAMConfig(
        pool_configs=[docker.types.IPAMPool(
            subnet='172.20.0.0/16'
        )]
    )
)

# Connect containers to network
container1 = client.containers.run(
    'nginx',
    name='web',
    network='my_network',
    detach=True
)

container2 = client.containers.run(
    'redis',
    name='cache',
    network='my_network',
    detach=True
)
```

Slide 4: Volume Management and Data Persistence

Docker volumes provide persistent storage for containers. This implementation shows how to create, manage, and utilize volumes programmatically, ensuring data persistence across container lifecycles while maintaining isolation.

```python
import docker

client = docker.from_client()

# Create a volume
volume = client.volumes.create(
    name='data_volume',
    driver='local',
    driver_opts={
        'type': 'none',
        'device': '/path/on/host',
        'o': 'bind'
    }
)

# Use volume in container
container = client.containers.run(
    'postgres:13',
    name='db',
    volumes={
        'data_volume': {
            'bind': '/var/lib/postgresql/data',
            'mode': 'rw'
        }
    },
    environment={
        'POSTGRES_PASSWORD': 'secret'
    },
    detach=True
)
```

Slide 5: Container Health Monitoring

Implementing robust health monitoring ensures container reliability. This code demonstrates setting up health checks, monitoring container metrics, and implementing automated responses to container state changes.

```python
import docker
import time
from datetime import datetime

client = docker.from_client()

def monitor_container_health(container_name):
    container = client.containers.get(container_name)
    stats = container.stats(stream=True)
    
    for stat in stats:
        cpu_stats = stat['cpu_stats']
        memory_stats = stat['memory_stats']
        
        cpu_usage = cpu_stats['cpu_usage']['total_usage']
        memory_usage = memory_stats['usage']
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] CPU Usage: {cpu_usage}")
        print(f"[{timestamp}] Memory Usage: {memory_usage} bytes")
        
        # Implement automatic restart if memory usage exceeds threshold
        if memory_usage > 1000000000:  # 1GB
            print(f"Memory threshold exceeded. Restarting {container_name}")
            container.restart()
            
        time.sleep(5)
```

Slide 6: Docker Image Building Automation

Automating Docker image building processes enables consistent deployment workflows. This implementation showcases creating custom images programmatically, including handling build contexts and managing build arguments.

```python
import docker
import io
import tarfile

client = docker.from_client()

# Create build context
dockerfile_content = '''
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
'''

# Create tar archive for build context
context = io.BytesIO()
tar = tarfile.open(fileobj=context, mode='w:gz')

# Add Dockerfile
dockerfile_bytes = dockerfile_content.encode('utf-8')
dockerfile_info = tarfile.TarInfo('Dockerfile')
dockerfile_info.size = len(dockerfile_bytes)
tar.addfile(dockerfile_info, io.BytesIO(dockerfile_bytes))

tar.close()
context.seek(0)

# Build image
image, logs = client.images.build(
    fileobj=context,
    tag='custom-app:latest',
    custom_context=True,
    encoding='gzip'
)

# Print build logs
for log in logs:
    if 'stream' in log:
        print(log['stream'].strip())
```

Slide 7: Multi-Container Application Deployment

Orchestrating multiple containers requires careful coordination of networking, dependencies, and configuration. This implementation showcases a Flask application with Redis caching and PostgreSQL database, demonstrating real-world microservices architecture deployment.

```python
import docker
from datetime import datetime

def deploy_microservices():
    client = docker.from_client()
    
    # Network configuration
    network = client.networks.create('microservices_net', driver='bridge')
    
    # Database service
    db = client.containers.run(
        'postgres:13',
        name='db',
        environment={'POSTGRES_PASSWORD': 'secret'},
        network='microservices_net',
        detach=True
    )
    
    # Cache service
    cache = client.containers.run(
        'redis:alpine',
        name='cache',
        network='microservices_net',
        detach=True
    )
    
    # Web application
    webapp = client.containers.run(
        'python:3.9-slim',
        name='webapp',
        command='python -m http.server 8080',
        ports={'8080/tcp': 8080},
        network='microservices_net',
        detach=True
    )
    
    return {
        'database': db.short_id,
        'cache': cache.short_id,
        'webapp': webapp.short_id,
        'network': network.short_id
    }

# Deploy and get container IDs
deployment = deploy_microservices()
for service, container_id in deployment.items():
    print(f"{service}: {container_id}")
```

Slide 8: Container Resource Management

Managing container resources ensures optimal performance and prevents resource exhaustion. This implementation demonstrates setting CPU, memory limits, and monitoring resource utilization using the Docker Python SDK.

```python
import docker

client = docker.from_client()

# Resource-constrained container
container = client.containers.run(
    'python:3.9-slim',
    name='resource_managed_app',
    command='python -c "while True: pass"',
    detach=True,
    mem_limit='512m',
    memswap_limit='512m',
    cpu_period=100000,
    cpu_quota=50000,  # 50% CPU limit
    cpuset_cpus='0,1'  # Use only CPUs 0 and 1
)

# Get resource usage statistics
stats = container.stats(stream=False)
cpu_usage = stats['cpu_stats']['cpu_usage']['total_usage']
mem_usage = stats['memory_stats']['usage']

print(f"CPU Usage: {cpu_usage}")
print(f"Memory Usage: {mem_usage} bytes")

# Cleanup
container.stop()
container.remove()
```

Slide 9: Docker Registry Integration

Interacting with Docker registries enables automated image distribution and deployment. This implementation shows pushing, pulling, and managing images across different registries using Python automation.

```python
import docker
import base64
import json

client = docker.from_client()

def registry_operations(image_name, registry_url):
    # Authenticate with registry
    auth_config = {
        'username': 'user',
        'password': 'secret',
        'serveraddress': registry_url
    }
    
    # Pull image
    image = client.images.pull(f'{registry_url}/{image_name}')
    
    # Tag image for new registry
    image.tag(f'{registry_url}/modified-{image_name}')
    
    # Push to registry
    push_result = client.images.push(
        f'{registry_url}/modified-{image_name}',
        auth_config=auth_config
    )
    
    # List registry images
    registry_images = client.images.search(image_name)
    return registry_images

# Example usage
images = registry_operations('python:3.9-slim', 'registry.example.com')
for img in images:
    print(f"Name: {img['name']}, Stars: {img['star_count']}")
```

Slide 10: Container Logging and Monitoring

Container monitoring requires systematic collection and analysis of performance metrics and logs. This implementation demonstrates setting up logging infrastructure and monitoring mechanisms for containerized applications using Docker's Python SDK.

```python
import docker
import json
from datetime import datetime

def setup_container_monitoring(container_name):
    client = docker.from_client()
    container = client.containers.get(container_name)
    
    # Configure logging
    log_config = {
        'type': 'json-file',
        'config': {'max-size': '10m'}
    }
    
    # Collect metrics
    stats = container.stats(stream=False)
    metrics = {
        'container_id': container.id[:12],
        'memory_usage': stats['memory_stats'].get('usage', 0),
        'cpu_percent': calculate_cpu_percent(stats),
        'network_rx': stats['networks']['eth0']['rx_bytes'],
        'network_tx': stats['networks']['eth0']['tx_bytes']
    }
    
    # Get container logs
    logs = container.logs(
        stdout=True,
        stderr=True,
        timestamps=True,
        tail=50
    ).decode('utf-8')
    
    return metrics, logs

def calculate_cpu_percent(stats):
    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                stats['precpu_stats']['cpu_usage']['total_usage']
    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                   stats['precpu_stats']['system_cpu_usage']
    return (cpu_delta / system_delta) * 100.0

# Usage example
metrics, logs = setup_container_monitoring('webapp')
print(json.dumps(metrics, indent=2))
```

Slide 11: Docker Compose with Python

Docker Compose automation through Python enables sophisticated multi-container application management. This implementation shows programmatic creation and control of complex container environments defined in compose files.

```python
import docker
from yaml import safe_load

def deploy_compose_stack(compose_file):
    client = docker.from_client()
    
    # Read compose file
    with open(compose_file, 'r') as f:
        compose_config = safe_load(f)
    
    # Process services
    services = {}
    for service_name, config in compose_config['services'].items():
        container = client.containers.run(
            image=config['image'],
            name=f"{service_name}",
            environment=config.get('environment', {}),
            ports=config.get('ports', {}),
            volumes=config.get('volumes', []),
            network=config.get('network_mode', 'bridge'),
            detach=True
        )
        services[service_name] = container
    
    return services

# Example compose deployment
compose_services = deploy_compose_stack('docker-compose.yml')
for name, container in compose_services.items():
    print(f"Service: {name}, Status: {container.status}")
```

Slide 12: Container Security Implementation

Implementing container security involves configuration of various security mechanisms. This implementation demonstrates setting up security policies, resource constraints, and access controls for Docker containers.

```python
import docker
from docker.types import Ulimit, RestartPolicy

def create_secure_container():
    client = docker.from_client()
    
    security_opts = [
        "no-new-privileges",
        "seccomp=default"
    ]
    
    container = client.containers.run(
        'python:3.9-slim',
        name='secure_container',
        command='python app.py',
        user='nobody',
        read_only=True,
        security_opt=security_opts,
        cap_drop=['ALL'],
        cap_add=['NET_BIND_SERVICE'],
        ulimits=[
            Ulimit(name='nofile', soft=1024, hard=2048)
        ],
        restart_policy=RestartPolicy(
            name='on-failure',
            max_retry_count=3
        ),
        detach=True,
        environment={
            'PYTHONUNBUFFERED': '1'
        }
    )
    
    return container.id

# Deploy secure container
container_id = create_secure_container()
print(f"Secure container deployed: {container_id}")
```

Slide 13: Resource Usage Analytics

Resource analytics provide insights into container performance patterns. This implementation creates a monitoring system that collects and analyzes container resource utilization data.

```python
import docker
import time
import json
from collections import deque

def analyze_container_resources(container_name, sample_count=10):
    client = docker.from_client()
    container = client.containers.get(container_name)
    metrics_history = deque(maxlen=sample_count)
    
    for _ in range(sample_count):
        stats = container.stats(stream=False)
        
        metrics = {
            'timestamp': time.time(),
            'memory': {
                'usage': stats['memory_stats']['usage'],
                'limit': stats['memory_stats']['limit'],
                'percent': (stats['memory_stats']['usage'] / 
                          stats['memory_stats']['limit']) * 100
            },
            'cpu': {
                'total_usage': stats['cpu_stats']['cpu_usage']['total_usage'],
                'system_usage': stats['cpu_stats']['system_cpu_usage']
            }
        }
        
        metrics_history.append(metrics)
        time.sleep(1)
    
    # Calculate aggregates
    analysis = {
        'avg_memory_percent': sum(m['memory']['percent'] 
                                for m in metrics_history) / sample_count,
        'max_memory_usage': max(m['memory']['usage'] 
                              for m in metrics_history),
        'samples_collected': len(metrics_history)
    }
    
    return analysis

# Analyze container resources
analysis = analyze_container_resources('webapp')
print(json.dumps(analysis, indent=2))
```

Slide 14: Additional Resources

\[1\] [https://arxiv.org/abs/2006.14800](https://arxiv.org/abs/2006.14800) - "Container Performance Analysis for Distributed Deep Learning Workloads" \[2\] [https://arxiv.org/abs/2103.05860](https://arxiv.org/abs/2103.05860) - "Automated Container Deployment and Resource Management in Cloud Computing" \[3\] [https://arxiv.org/abs/2105.07147](https://arxiv.org/abs/2105.07147) - "Security Analysis of Container Images in Distributed Systems" \[4\] [https://arxiv.org/abs/1909.13739](https://arxiv.org/abs/1909.13739) - "Performance Optimization Techniques for Containerized Applications" \[5\] [https://arxiv.org/abs/2004.12226](https://arxiv.org/abs/2004.12226) - "Resource Management and Scheduling in Container-based Cloud Platforms"

