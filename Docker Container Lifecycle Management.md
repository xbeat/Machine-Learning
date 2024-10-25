## Docker Container Lifecycle Management

Slide 1: Docker Container Lifecycle Management

Docker containers follow a well-defined lifecycle from creation to termination. Understanding container states and transitions is crucial for effective container orchestration and management in production environments. The lifecycle API enables programmatic control over containers.

```python
import docker

# Initialize Docker client
client = docker.from_docker.client()

# Create and manage container lifecycle
container = client.containers.run('python:3.9', 
                                command='python -c "print(\'Hello Docker\')"',
                                name='demo_container',
                                detach=True)

# Check container status
print(f"Container Status: {container.status}")

# Pause container
container.pause()
print(f"Paused Status: {container.status}")

# Resume container
container.unpause()

# Stop container
container.stop()

# Remove container
container.remove()
```

Slide 2: Building Custom Docker Images

Understanding how to programmatically build Docker images enables automated deployment pipelines and consistent environment management. This approach demonstrates creating images with specific requirements and configurations.

```python
import docker
from io import BytesIO

# Dockerfile content as a string
dockerfile = '''
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
'''

# Create build context
build_context = BytesIO(dockerfile.encode('utf-8'))

# Initialize Docker client
client = docker.from_docker.client()

# Build image
image, build_logs = client.images.build(
    fileobj=build_context,
    tag='my-python-app:1.0',
    rm=True
)

# Print build logs
for log in build_logs:
    if 'stream' in log:
        print(log['stream'].strip())
```

Slide 3: Docker Network Management

Docker networks enable container communication and isolation. Understanding network management is essential for microservices architecture and distributed systems. This implementation shows how to create and manage custom networks.

```python
import docker

# Initialize Docker client
client = docker.from_docker.client()

# Create custom network
network = client.networks.create(
    name='my_network',
    driver='bridge',
    ipam=docker.types.IPAMConfig(
        pool_configs=[docker.types.IPAMPool(
            subnet='172.20.0.0/16',
            gateway='172.20.0.1'
        )]
    )
)

# Connect containers to network
container1 = client.containers.run('nginx', detach=True, name='web')
container2 = client.containers.run('redis', detach=True, name='cache')

network.connect(container1)
network.connect(container2)

# List connected containers
connected_containers = network.containers
for container in connected_containers:
    print(f"Container {container.name} connected to network")

# Cleanup
network.disconnect(container1)
network.disconnect(container2)
network.remove()
```

Slide 4: Docker Volume Management

Docker volumes provide persistent storage and data sharing between containers. This implementation demonstrates volume creation, mounting, and management for stateful applications.

```python
import docker

# Initialize Docker client
client = docker.from_docker.client()

# Create volume
volume = client.volumes.create(
    name='data_volume',
    driver='local',
    driver_opts={
        'type': 'none',
        'o': 'bind',
        'device': '/path/on/host'
    }
)

# Run container with volume
container = client.containers.run(
    'postgres:13',
    detach=True,
    volumes={
        'data_volume': {
            'bind': '/var/lib/postgresql/data',
            'mode': 'rw'
        }
    },
    environment={
        'POSTGRES_PASSWORD': 'mysecretpassword'
    }
)

# List volume details
print(f"Volume Name: {volume.name}")
print(f"Volume Driver: {volume.attrs['Driver']}")
print(f"Mount Point: {volume.attrs['Mountpoint']}")

# Cleanup
container.stop()
container.remove()
volume.remove()
```

Slide 5: Docker Resource Management

Effective resource management ensures optimal container performance and system stability. This implementation shows how to monitor and control container resource usage programmatically.

```python
import docker
import json

# Initialize Docker client
client = docker.from_docker.client()

# Create container with resource constraints
container = client.containers.run(
    'python:3.9',
    command='python -c "while True: pass"',
    detach=True,
    mem_limit='512m',
    memswap_limit='512m',
    cpu_period=100000,
    cpu_quota=50000,  # 50% CPU limit
    name='resource_limited_container'
)

# Get container stats
stats = container.stats(stream=False)
print(json.dumps(stats, indent=2))

# Update container resources
container.update(
    mem_limit='1g',
    cpu_quota=75000  # 75% CPU limit
)

# Monitor resource usage
def print_usage(container):
    stats = container.stats(stream=False)
    cpu_stats = stats['cpu_stats']
    mem_stats = stats['memory_stats']
    
    cpu_usage = cpu_stats['cpu_usage']['total_usage']
    mem_usage = mem_stats['usage'] / (1024 * 1024)  # Convert to MB
    
    print(f"CPU Usage: {cpu_usage}")
    print(f"Memory Usage: {mem_usage:.2f} MB")

print_usage(container)

# Cleanup
container.stop()
container.remove()
```

Slide 6: Docker Health Checks

Implementing robust health checks ensures container reliability and enables automated recovery from failures. This example demonstrates custom health check implementation and monitoring.

```python
import docker
import time

# Initialize Docker client
client = docker.from_docker.client()

# Dockerfile content with healthcheck
dockerfile = '''
FROM python:3.9-slim
WORKDIR /app
COPY app.py .
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
CMD ["python", "app.py"]
'''

# Create container with health check
container = client.containers.run(
    'python:3.9',
    detach=True,
    healthcheck={
        "test": ["CMD", "curl", "-f", "http://localhost:8080/health"],
        "interval": 30000000000,  # 30 seconds
        "timeout": 3000000000,    # 3 seconds
        "retries": 3
    },
    name='healthcheck_container'
)

# Monitor health status
def monitor_health(container):
    container.reload()
    health_status = container.attrs['State']['Health']['Status']
    print(f"Container Health Status: {health_status}")
    
    if health_status != 'healthy':
        print("Implementing recovery actions...")
        container.restart()

# Check health periodically
for _ in range(3):
    monitor_health(container)
    time.sleep(10)

# Cleanup
container.stop()
container.remove()
```

Slide 7: Docker Multi-Container Applications

Managing multiple interconnected containers is essential for microservices architecture. This implementation demonstrates container orchestration and communication patterns.

```python
import docker
import yaml

# Initialize Docker client
client = docker.from_docker.client()

# Docker Compose-like configuration
compose_config = {
    'version': '3',
    'services': {
        'web': {
            'image': 'nginx:latest',
            'ports': ['80:80'],
            'depends_on': ['app']
        },
        'app': {
            'image': 'python:3.9',
            'volumes': ['./app:/app'],
            'depends_on': ['db']
        },
        'db': {
            'image': 'postgres:13',
            'environment': {
                'POSTGRES_PASSWORD': 'secret'
            }
        }
    }
}

# Create network for communication
network = client.networks.create('app_network', driver='bridge')

# Deploy containers
containers = {}
for service, config in compose_config['services'].items():
    containers[service] = client.containers.run(
        config['image'],
        detach=True,
        name=service,
        network='app_network'
    )

# Verify container communication
def test_connection(container, target):
    exit_code, output = container.exec_run(
        f"ping -c 1 {target}"
    )
    return exit_code == 0

# Test inter-container connectivity
print(test_connection(containers['web'], 'app'))
print(test_connection(containers['app'], 'db'))

# Cleanup
for container in containers.values():
    container.stop()
    container.remove()
network.remove()
```

Slide 8: Docker Security Implementation

Implementing container security measures is crucial for protecting applications and data. This code demonstrates security best practices and configuration.

```python
import docker
from docker.types import Mount

# Initialize Docker client
client = docker.from_docker.client()

# Security configuration
security_opts = {
    'seccomp': '/path/to/seccomp/profile.json',
    'capabilities': {
        'drop': ['ALL'],
        'add': ['NET_BIND_SERVICE']
    }
}

# Create container with security options
container = client.containers.run(
    'nginx:latest',
    detach=True,
    security_opt=['seccomp=/path/to/seccomp/profile.json'],
    cap_drop=['ALL'],
    cap_add=['NET_BIND_SERVICE'],
    read_only=True,
    mount_points=[
        Mount(
            target='/etc/nginx/conf.d',
            source='nginx_config',
            type='volume',
            read_only=True
        )
    ],
    environment={
        'NGINX_ENTRYPOINT_QUIET_LOGS': '1'
    }
)

# Verify security settings
container.reload()
config = container.attrs['HostConfig']
print(f"Security Options: {config['SecurityOpt']}")
print(f"Capabilities: {config['CapDrop']}, {config['CapAdd']}")
print(f"Read-only: {config['ReadonlyRootfs']}")

# Cleanup
container.stop()
container.remove()
```

Slide 9: Docker Monitoring and Logging

Implementing comprehensive monitoring and logging is essential for maintaining container-based applications. This implementation shows how to collect and analyze container metrics and logs.

```python
import docker
import json
from datetime import datetime

# Initialize Docker client
client = docker.from_docker.client()

# Create container with logging configuration
container = client.containers.run(
    'nginx',
    detach=True,
    name='monitored_container',
    log_config={
        'type': 'json-file',
        'config': {
            'max-size': '10m',
            'max-file': '3'
        }
    }
)

# Implement monitoring function
def monitor_container(container):
    # Get container stats
    stats = container.stats(stream=False)
    
    # Process CPU metrics
    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                stats['precpu_stats']['cpu_usage']['total_usage']
    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                   stats['precpu_stats']['system_cpu_usage']
    cpu_usage = (cpu_delta / system_delta) * 100
    
    # Process memory metrics
    memory_usage = stats['memory_stats']['usage'] / (1024 * 1024)  # MB
    memory_limit = stats['memory_stats']['limit'] / (1024 * 1024)  # MB
    
    return {
        'timestamp': datetime.now().isoformat(),
        'cpu_usage': round(cpu_usage, 2),
        'memory_usage_mb': round(memory_usage, 2),
        'memory_limit_mb': round(memory_limit, 2)
    }

# Collect and display metrics
metrics = monitor_container(container)
print(json.dumps(metrics, indent=2))

# Get container logs
logs = container.logs(
    timestamps=True,
    tail=100
).decode('utf-8')

print("Recent Logs:")
print(logs)

# Cleanup
container.stop()
container.remove()
```

Slide 10: Docker CI/CD Integration

Automating Docker operations within CI/CD pipelines ensures consistent deployment and testing. This implementation demonstrates integration with common CI/CD practices.

```python
import docker
import os
import subprocess

class DockerCICD:
    def __init__(self):
        self.client = docker.from_docker.client()
        self.image_name = 'myapp'
        self.version = os.getenv('CI_COMMIT_SHA', 'latest')
    
    def build_image(self):
        print(f"Building image: {self.image_name}:{self.version}")
        image, logs = self.client.images.build(
            path='.',
            tag=f"{self.image_name}:{self.version}",
            buildargs={
                'VERSION': self.version
            }
        )
        return image
    
    def run_tests(self):
        print("Running tests in container...")
        container = self.client.containers.run(
            f"{self.image_name}:{self.version}",
            command=['python', '-m', 'pytest'],
            detach=True
        )
        
        for log in container.logs(stream=True):
            print(log.decode())
            
        container.wait()
        exit_code = container.attrs['State']['ExitCode']
        container.remove()
        return exit_code == 0
    
    def push_image(self):
        print("Pushing image to registry...")
        registry_url = os.getenv('REGISTRY_URL')
        subprocess.run([
            'docker', 'tag',
            f"{self.image_name}:{self.version}",
            f"{registry_url}/{self.image_name}:{self.version}"
        ])
        subprocess.run([
            'docker', 'push',
            f"{registry_url}/{self.image_name}:{self.version}"
        ])

# Usage in CI/CD pipeline
cicd = DockerCICD()
image = cicd.build_image()

if cicd.run_tests():
    cicd.push_image()
    print("Pipeline completed successfully")
else:
    print("Tests failed, aborting pipeline")
```

Slide 11: Advanced Docker Network Patterns

Docker network patterns enable sophisticated microservices communication through overlay networks, load balancing, and service discovery. This implementation demonstrates creating custom network topologies with traffic isolation and service mesh capabilities.

```python
import docker
from docker.types import IPAMConfig, IPAMPool

# Initialize Docker client
client = docker.from_docker.client()

def create_mesh_network():
    # Create overlay network with encryption
    network = client.networks.create(
        'service_mesh',
        driver='overlay',
        ipam=IPAMConfig(
            pool_configs=[IPAMPool(subnet='10.0.0.0/24')]
        ),
        options={'encrypt': 'true'}
    )
    
    # Deploy services in mesh
    services = {
        'api': client.containers.run(
            'python:3.9',
            network='service_mesh',
            name='api_service',
            detach=True
        ),
        'cache': client.containers.run(
            'redis:latest',
            network='service_mesh',
            name='cache_service',
            detach=True
        )
    }
    
    # Configure service discovery
    service_map = {}
    for name, container in services.items():
        container.reload()
        ip = container.attrs['NetworkSettings']['Networks']['service_mesh']['IPAddress']
        service_map[name] = ip
        
    return network, service_map

# Execute network setup
network, services = create_mesh_network()
print(f"Service Mesh Network: {network.name}")
print(f"Service Discovery Map: {services}")

# Cleanup resources
for container in client.containers.list():
    container.remove(force=True)
network.remove()
```

Slide 12: Docker Image Layer Optimization

Understanding and optimizing Docker image layers is crucial for maintaining efficient container deployments. This implementation shows techniques for analyzing and optimizing image layer structure.

```python
import docker
import json

class ImageOptimizer:
    def __init__(self):
        self.client = docker.from_docker.client()
    
    def analyze_layers(self, image_name):
        image = self.client.images.get(image_name)
        layers = image.history()
        
        size_map = {}
        total_size = 0
        
        for layer in layers:
            command = layer.get('CreatedBy', 'base')
            size = layer.get('Size', 0)
            size_map[command] = size
            total_size += size
            
        return size_map, total_size
    
    def optimize_dockerfile(self, dockerfile_content):
        # Basic optimization rules
        optimized = []
        current_run = []
        
        for line in dockerfile_content.split('\n'):
            if line.startswith('RUN'):
                current_run.append(line[4:])
            else:
                if current_run:
                    optimized.append('RUN ' + ' && '.join(current_run))
                    current_run = []
                optimized.append(line)
        
        return '\n'.join(optimized)

# Usage example
optimizer = ImageOptimizer()

# Analyze existing image
image_name = 'python:3.9-slim'
layers, total_size = optimizer.analyze_layers(image_name)

print(f"Image Size Analysis for {image_name}:")
print(f"Total Size: {total_size / 1024 / 1024:.2f} MB")
print("Layer Breakdown:")
for cmd, size in layers.items():
    print(f"- {size / 1024 / 1024:.2f} MB: {cmd[:100]}")

# Optimize Dockerfile
sample_dockerfile = """
FROM python:3.9-slim
RUN apt-get update
RUN apt-get install -y gcc
RUN pip install numpy
RUN pip install pandas
COPY . /app
CMD ["python", "app.py"]
"""

optimized = optimizer.optimize_dockerfile(sample_dockerfile)
print("\nOptimized Dockerfile:")
print(optimized)
```

Slide 13: Docker Development Workflow Automation

Automating development workflows with Docker ensures consistency between development and production environments while streamlining the development process through automated builds and testing.

```python
import docker
import subprocess
from pathlib import Path

class DockerDevelopment:
    def __init__(self, project_name):
        self.client = docker.from_docker.client()
        self.project_name = project_name
        self.dev_image = f"{project_name}-dev"
    
    def create_dev_environment(self):
        # Build development image
        dockerfile = f"""
            FROM python:3.9
            WORKDIR /code
            COPY requirements.txt .
            RUN pip install -r requirements.txt
            RUN pip install pytest pytest-cov
            VOLUME /code
            CMD ["python", "main.py"]
        """
        
        Path('Dockerfile.dev').write_text(dockerfile)
        self.client.images.build(
            path='.',
            dockerfile='Dockerfile.dev',
            tag=self.dev_image
        )
    
    def run_tests(self, watch=False):
        cmd = ['pytest', '--cov=src']
        if watch:
            cmd.append('-f')
            
        container = self.client.containers.run(
            self.dev_image,
            command=cmd,
            volumes={
                str(Path.cwd()): {'bind': '/code', 'mode': 'rw'}
            },
            detach=True
        )
        
        for log in container.logs(stream=True):
            print(log.decode(), end='')
        
        container.wait()
        container.remove()
    
    def start_dev_server(self):
        container = self.client.containers.run(
            self.dev_image,
            command=['python', 'main.py'],
            volumes={
                str(Path.cwd()): {'bind': '/code', 'mode': 'rw'}
            },
            ports={'8000/tcp': 8000},
            detach=True
        )
        return container

# Usage example
dev = DockerDevelopment('myapp')
dev.create_dev_environment()
dev.run_tests(watch=True)
server = dev.start_dev_server()

print("Development environment ready!")
print("Access the application at http://localhost:8000")
```

Slide 14: Additional Resources

[http://arxiv.org/abs/2006.14800](http://arxiv.org/abs/2006.14800) - "A Systematic Literature Review of Container Security" [http://arxiv.org/abs/2104.01937](http://arxiv.org/abs/2104.01937) - "Container Orchestration: A Survey" [http://arxiv.org/abs/2003.05656](http://arxiv.org/abs/2003.05656) - "Performance Analysis of Container Networking in Cloud Infrastructure" [http://arxiv.org/abs/1904.05525](http://arxiv.org/abs/1904.05525) - "Microservices: A Systematic Mapping Study" [http://arxiv.org/abs/2006.05228](http://arxiv.org/abs/2006.05228) - "Container Security: Issues, Challenges, and the Road Ahead"

