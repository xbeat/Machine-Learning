## Docker Components Explained
Slide 1: Docker Image Handling with Python

The Docker SDK for Python provides a high-level API to interact with Docker images programmatically. This implementation demonstrates connecting to Docker daemon, listing available images, and performing basic image operations using the docker-py library.

```python
import docker
from typing import List, Dict

class DockerImageManager:
    def __init__(self):
        # Initialize Docker client
        self.client = docker.from_client()
    
    def list_images(self) -> List[Dict]:
        # List all images with their details
        images = self.client.images.list()
        return [{'id': img.id, 'tags': img.tags} for img in images]
    
    def pull_image(self, image_name: str) -> None:
        # Pull an image from Docker Hub
        try:
            self.client.images.pull(image_name)
            print(f"Successfully pulled {image_name}")
        except docker.errors.ImageNotFound:
            print(f"Image {image_name} not found")

# Example usage
manager = DockerImageManager()
print("Available images:")
print(manager.list_images())
manager.pull_image("python:3.9-slim")
```

Slide 2: Container Lifecycle Management

This implementation showcases container lifecycle management including creation, starting, stopping, and removal. The code demonstrates proper error handling and resource cleanup using Python context managers.

```python
import docker
from contextlib import contextmanager

class ContainerLifecycle:
    def __init__(self):
        self.client = docker.from_client()
    
    @contextmanager
    def create_container(self, image_name: str, command: str):
        container = self.client.containers.create(
            image=image_name,
            command=command,
            detach=True
        )
        try:
            yield container
            container.start()
            print(f"Container {container.id[:12]} started")
        finally:
            container.stop()
            container.remove()
            print(f"Container {container.id[:12]} cleaned up")
    
    def run_task(self, image_name: str, command: str):
        with self.create_container(image_name, command) as container:
            logs = container.logs(stream=True)
            for log in logs:
                print(log.decode().strip())

# Example usage
lifecycle = ContainerLifecycle()
lifecycle.run_task("python:3.9-slim", "python -c 'print(\"Hello from container!\")'")
```

Slide 3: Docker Volume Management

Docker volumes are crucial for persistent data storage. This implementation provides a comprehensive interface for volume management, including creation, mounting, and cleanup operations using the Python Docker SDK.

```python
import docker
from typing import Optional

class VolumeManager:
    def __init__(self):
        self.client = docker.from_client()
    
    def create_volume(self, name: str) -> docker.models.volumes.Volume:
        return self.client.volumes.create(name)
    
    def mount_volume(self, volume_name: str, container_name: str, 
                    mount_point: str, image: str = "alpine"):
        volume = self.create_volume(volume_name)
        container = self.client.containers.run(
            image=image,
            name=container_name,
            volumes={volume_name: {'bind': mount_point, 'mode': 'rw'}},
            detach=True
        )
        return container, volume

    def cleanup(self, container_name: str, volume_name: str):
        try:
            container = self.client.containers.get(container_name)
            container.stop()
            container.remove()
        except docker.errors.NotFound:
            pass
        
        try:
            volume = self.client.volumes.get(volume_name)
            volume.remove()
        except docker.errors.NotFound:
            pass

# Example usage
vm = VolumeManager()
container, volume = vm.mount_volume("data_volume", "test_container", "/data")
print(f"Created volume {volume.name} mounted at {container.name}")
vm.cleanup("test_container", "data_volume")
```

Slide 4: Docker Network Configuration

Creating and managing Docker networks is essential for container communication. This implementation demonstrates network creation, connection management, and network inspection functionality using Python.

```python
import docker
from typing import Dict, List

class NetworkManager:
    def __init__(self):
        self.client = docker.from_client()
    
    def create_network(self, name: str, driver: str = "bridge", 
                      internal: bool = False) -> docker.models.networks.Network:
        return self.client.networks.create(
            name=name,
            driver=driver,
            internal=internal,
            check_duplicate=True
        )
    
    def connect_container(self, network_name: str, container_name: str,
                         aliases: Optional[List[str]] = None):
        network = self.client.networks.get(network_name)
        container = self.client.containers.get(container_name)
        network.connect(container, aliases=aliases)
        
    def inspect_network(self, network_name: str) -> Dict:
        network = self.client.networks.get(network_name)
        return network.attrs

# Example usage
nm = NetworkManager()
network = nm.create_network("app_network")
print(f"Created network: {network.name}")
inspection = nm.inspect_network("app_network")
print(f"Network details: {inspection['Driver']}")
```

Slide 5: Dockerfile Generation with Python

This implementation creates a Python class that generates Dockerfile content programmatically. It provides a flexible interface to define build instructions and handles multi-stage builds with proper formatting and validation.

```python
class DockerfileGenerator:
    def __init__(self):
        self.instructions = []
    
    def add_instruction(self, instruction: str, *args: str):
        formatted_args = " ".join(str(arg) for arg in args)
        self.instructions.append(f"{instruction} {formatted_args}")
    
    def from_base(self, image: str, alias: str = None):
        if alias:
            self.add_instruction("FROM", f"{image} AS {alias}")
        else:
            self.add_instruction("FROM", image)
    
    def add_environment(self, env_vars: dict):
        for key, value in env_vars.items():
            self.add_instruction("ENV", f"{key}={value}")
    
    def generate(self) -> str:
        return "\n".join(self.instructions)

# Example usage
generator = DockerfileGenerator()
generator.from_base("python:3.9-slim")
generator.add_instruction("WORKDIR", "/app")
generator.add_environment({"PYTHONUNBUFFERED": "1"})
generator.add_instruction("COPY", "requirements.txt", "./")
generator.add_instruction("RUN", "pip install -r requirements.txt")

dockerfile_content = generator.generate()
print("Generated Dockerfile:")
print(dockerfile_content)
```

Slide 6: Docker Build Process Automation

This implementation provides a comprehensive build process automation system that handles Docker image building with proper error handling, build arguments, and cache management.

```python
import docker
import os
from typing import Dict, Optional

class DockerBuildAutomation:
    def __init__(self, client: Optional[docker.DockerClient] = None):
        self.client = client or docker.from_client()
    
    def build_image(self, 
                   path: str, 
                   tag: str, 
                   buildargs: Dict[str, str] = None,
                   nocache: bool = False) -> tuple:
        try:
            image, logs = self.client.images.build(
                path=path,
                tag=tag,
                buildargs=buildargs,
                nocache=nocache,
                rm=True
            )
            build_logs = []
            for log in logs:
                if 'stream' in log:
                    build_logs.append(log['stream'].strip())
            return image, build_logs
        except docker.errors.BuildError as e:
            print(f"Build failed: {str(e)}")
            raise
    
    def push_image(self, repository: str, tag: str):
        try:
            for line in self.client.images.push(
                repository=repository,
                tag=tag,
                stream=True,
                decode=True
            ):
                if 'status' in line:
                    print(f"Push status: {line['status']}")
        except docker.errors.APIError as e:
            print(f"Push failed: {str(e)}")
            raise

# Example usage
builder = DockerBuildAutomation()
image, logs = builder.build_image(
    path="./",
    tag="myapp:latest",
    buildargs={"VERSION": "1.0.0"}
)
print("\n".join(logs))
```

Slide 7: Container Resource Monitoring

A sophisticated monitoring system that tracks container resource usage including CPU, memory, and network statistics, implementing real-time metrics collection and analysis.

```python
import docker
import time
from datetime import datetime
from typing import Dict, Generator

class ContainerMonitor:
    def __init__(self):
        self.client = docker.from_client()
    
    def get_container_stats(self, 
                          container_id: str) -> Generator[Dict, None, None]:
        container = self.client.containers.get(container_id)
        stats = container.stats(stream=True, decode=True)
        
        for stat in stats:
            cpu_delta = stat['cpu_stats']['cpu_usage']['total_usage'] - \
                       stat['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stat['cpu_stats']['system_cpu_usage'] - \
                          stat['precpu_stats']['system_cpu_usage']
            
            cpu_percent = 0.0
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * 100
            
            memory_usage = stat['memory_stats']['usage']
            memory_limit = stat['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100
            
            yield {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': round(cpu_percent, 2),
                'memory_percent': round(memory_percent, 2),
                'memory_usage_mb': round(memory_usage / (1024 * 1024), 2)
            }

# Example usage
monitor = ContainerMonitor()
container = monitor.client.containers.run(
    "nginx:latest",
    name="monitored_container",
    detach=True
)

try:
    for stats in monitor.get_container_stats(container.id):
        print(f"Stats: {stats}")
        time.sleep(1)
except KeyboardInterrupt:
    container.stop()
    container.remove()
```

Slide 8: Docker Compose Management in Python

This implementation provides a programmatic interface to Docker Compose operations, allowing dynamic service configuration and orchestration of multi-container applications through Python code.

```python
import yaml
from typing import Dict, List
import docker

class ComposeManager:
    def __init__(self):
        self.client = docker.from_client()
        self.compose_config = {
            'version': '3.8',
            'services': {}
        }
    
    def add_service(self, 
                   name: str, 
                   image: str,
                   ports: List[str] = None,
                   environment: Dict[str, str] = None,
                   volumes: List[str] = None):
        service = {'image': image}
        
        if ports:
            service['ports'] = ports
        if environment:
            service['environment'] = environment
        if volumes:
            service['volumes'] = volumes
            
        self.compose_config['services'][name] = service
    
    def generate_compose_file(self, filename: str = 'docker-compose.yml'):
        with open(filename, 'w') as f:
            yaml.dump(self.compose_config, f, default_flow_style=False)
    
    def deploy_stack(self, stack_name: str):
        self.client.swarm.init()
        with open('docker-compose.yml') as f:
            config = yaml.safe_load(f)
            self.client.stacks.deploy(stack_name, config)

# Example usage
compose = ComposeManager()

# Add web service
compose.add_service(
    name='web',
    image='nginx:latest',
    ports=['80:80'],
    environment={'NGINX_HOST': 'foobar.com'},
    volumes=['./nginx.conf:/etc/nginx/nginx.conf:ro']
)

# Add database service
compose.add_service(
    name='db',
    image='postgres:13',
    environment={
        'POSTGRES_USER': 'user',
        'POSTGRES_PASSWORD': 'password'
    },
    volumes=['pgdata:/var/lib/postgresql/data']
)

compose.generate_compose_file()
```

Slide 9: Container Health Monitoring System

This implementation creates a sophisticated health monitoring system for Docker containers, including custom health checks, metric collection, and alerting mechanisms.

```python
import docker
import time
from datetime import datetime
import json
from typing import Callable, Dict, Optional

class HealthMonitor:
    def __init__(self):
        self.client = docker.from_client()
        self.health_checks = {}
        self.alert_callbacks = []
    
    def add_health_check(self, 
                        container_name: str, 
                        check_function: Callable[[str], bool],
                        interval: int = 30):
        self.health_checks[container_name] = {
            'function': check_function,
            'interval': interval,
            'last_check': None,
            'status': None
        }
    
    def add_alert_callback(self, callback: Callable[[str, str], None]):
        self.alert_callbacks.append(callback)
    
    def check_container_health(self, container_name: str) -> Dict:
        try:
            container = self.client.containers.get(container_name)
            check = self.health_checks[container_name]
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'container': container_name,
                'state': container.status,
                'health_check': check['function'](container_name)
            }
            
            if not status['health_check'] and self.alert_callbacks:
                for callback in self.alert_callbacks:
                    callback(container_name, json.dumps(status))
            
            check['last_check'] = status['timestamp']
            check['status'] = status['health_check']
            
            return status
            
        except docker.errors.NotFound:
            return {'error': f'Container {container_name} not found'}
    
    def monitor(self):
        while True:
            for container_name in self.health_checks:
                check = self.health_checks[container_name]
                if (not check['last_check'] or 
                    (datetime.now() - datetime.fromisoformat(check['last_check'])).seconds 
                    >= check['interval']):
                    status = self.check_container_health(container_name)
                    print(f"Health check result: {status}")
            time.sleep(1)

# Example usage
def custom_health_check(container_name: str) -> bool:
    # Custom health check implementation
    return True

def alert_handler(container_name: str, status: str):
    print(f"Alert for {container_name}: {status}")

monitor = HealthMonitor()
monitor.add_health_check("web_app", custom_health_check, interval=10)
monitor.add_alert_callback(alert_handler)

# Start monitoring in a separate thread
import threading
threading.Thread(target=monitor.monitor, daemon=True).start()
```

Slide 10: Docker Security Analysis

This implementation provides a comprehensive security analysis tool for Docker containers, including vulnerability scanning, security policy enforcement, and compliance checking capabilities.

```python
import docker
from typing import Dict, List, Optional
import json
import subprocess

class SecurityAnalyzer:
    def __init__(self):
        self.client = docker.from_client()
        self.security_policies = {}
    
    def scan_image(self, image_name: str) -> Dict:
        try:
            # Simulate vulnerability scanning
            scan_results = {
                'image': image_name,
                'vulnerabilities': self._perform_vulnerability_scan(image_name),
                'security_score': self._calculate_security_score(image_name)
            }
            return scan_results
        except Exception as e:
            return {'error': str(e)}
    
    def _perform_vulnerability_scan(self, image_name: str) -> List[Dict]:
        image = self.client.images.get(image_name)
        vulnerabilities = []
        
        # Analyze image layers
        for layer in image.history():
            if 'Created' in layer:
                cmd = layer.get('CreatedBy', '')
                if self._check_layer_vulnerability(cmd):
                    vulnerabilities.append({
                        'layer_id': layer.get('Id', 'unknown'),
                        'command': cmd,
                        'severity': self._assess_severity(cmd)
                    })
        
        return vulnerabilities
    
    def _calculate_security_score(self, image_name: str) -> float:
        image = self.client.images.get(image_name)
        score = 100.0
        
        # Basic security checks
        config = image.attrs.get('Config', {})
        if config.get('User') == 'root':
            score -= 20
        if 'Healthcheck' not in config:
            score -= 10
        
        return max(0.0, score)
    
    def enforce_security_policy(self, container_id: str) -> Dict:
        container = self.client.containers.get(container_id)
        policy_results = {
            'container_id': container_id,
            'policy_violations': [],
            'compliant': True
        }
        
        # Check security configurations
        config = container.attrs['HostConfig']
        
        if config.get('Privileged'):
            policy_results['policy_violations'].append(
                'Container running in privileged mode'
            )
            policy_results['compliant'] = False
        
        if not config.get('ReadonlyRootfs'):
            policy_results['policy_violations'].append(
                'Root filesystem is not read-only'
            )
        
        return policy_results
    
    def _check_layer_vulnerability(self, command: str) -> bool:
        # Simplified vulnerability check
        risky_commands = ['curl', 'wget', 'apt-get', 'npm install']
        return any(cmd in command.lower() for cmd in risky_commands)
    
    def _assess_severity(self, command: str) -> str:
        if 'sudo' in command.lower():
            return 'HIGH'
        if 'apt-get' in command.lower():
            return 'MEDIUM'
        return 'LOW'

# Example usage
analyzer = SecurityAnalyzer()

# Scan an image
results = analyzer.scan_image("nginx:latest")
print(f"Security scan results: {json.dumps(results, indent=2)}")

# Enforce security policy on a running container
container = analyzer.client.containers.run(
    "nginx:latest",
    detach=True,
    name="secure_test"
)
policy_check = analyzer.enforce_security_policy(container.id)
print(f"Policy check results: {json.dumps(policy_check, indent=2)}")

# Cleanup
container.stop()
container.remove()
```

Slide 11: Docker Log Analysis and Metrics

An advanced implementation for collecting, analyzing, and processing Docker container logs with pattern matching and metric extraction capabilities.

```python
import docker
from datetime import datetime, timedelta
import re
from typing import Dict, List, Generator
from collections import defaultdict

class LogAnalyzer:
    def __init__(self):
        self.client = docker.from_client()
        self.patterns = {
            'error': r'(?i)(error|exception|failed|failure)',
            'warning': r'(?i)(warning|warn)',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
    
    def collect_logs(self, 
                    container_id: str, 
                    since: datetime = None,
                    until: datetime = None) -> Generator[str, None, None]:
        container = self.client.containers.get(container_id)
        
        if since:
            since = int(since.timestamp())
        if until:
            until = int(until.timestamp())
            
        logs = container.logs(
            timestamps=True,
            since=since,
            until=until,
            stream=True
        )
        
        for log in logs:
            yield log.decode('utf-8').strip()
    
    def analyze_logs(self, container_id: str) -> Dict:
        metrics = {
            'error_count': 0,
            'warning_count': 0,
            'unique_ips': set(),
            'timestamps': [],
            'patterns': defaultdict(list)
        }
        
        for log_entry in self.collect_logs(container_id):
            metrics['timestamps'].append(
                self._extract_timestamp(log_entry)
            )
            
            # Pattern matching
            for pattern_name, pattern in self.patterns.items():
                matches = re.findall(pattern, log_entry)
                if matches:
                    metrics['patterns'][pattern_name].extend(matches)
                    
                    if pattern_name == 'error':
                        metrics['error_count'] += len(matches)
                    elif pattern_name == 'warning':
                        metrics['warning_count'] += len(matches)
                    elif pattern_name == 'ip_address':
                        metrics['unique_ips'].update(matches)
        
        # Convert set to list for JSON serialization
        metrics['unique_ips'] = list(metrics['unique_ips'])
        
        return metrics
    
    def _extract_timestamp(self, log_entry: str) -> datetime:
        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', log_entry)
        if timestamp_match:
            return datetime.fromisoformat(timestamp_match.group(1))
        return datetime.now()

# Example usage
analyzer = LogAnalyzer()

# Create a test container with some logs
container = analyzer.client.containers.run(
    "nginx:latest",
    detach=True,
    name="log_test"
)

# Wait for some logs to generate
import time
time.sleep(5)

# Analyze logs
log_metrics = analyzer.analyze_logs(container.id)
print(f"Log analysis results: {json.dumps(log_metrics, default=str, indent=2)}")

# Cleanup
container.stop()
container.remove()
```

Slide 12: Docker Configuration Management

This implementation provides a sophisticated configuration management system for Docker environments, handling environment variables, secrets, and configuration files with proper encryption and validation.

```python
import docker
import base64
import json
from typing import Dict, Optional
from cryptography.fernet import Fernet
import os

class ConfigurationManager:
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.client = docker.from_client()
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
    def create_secret(self, name: str, data: str) -> Dict:
        try:
            # Encrypt sensitive data
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            encoded_data = base64.b64encode(encrypted_data).decode()
            
            secret = self.client.secrets.create(
                name=name,
                data=encoded_data
            )
            
            return {
                'id': secret.id,
                'name': secret.name,
                'created_at': secret.attrs['CreatedAt']
            }
        except Exception as e:
            return {'error': f'Failed to create secret: {str(e)}'}
    
    def apply_configuration(self, 
                          container_name: str, 
                          config: Dict[str, str],
                          secrets: Dict[str, str] = None) -> Dict:
        try:
            container = self.client.containers.get(container_name)
            
            # Update environment variables
            current_config = container.attrs['Config']
            env_list = current_config.get('Env', [])
            
            # Convert existing env to dict
            env_dict = dict(item.split('=', 1) for item in env_list if '=' in item)
            
            # Update with new config
            env_dict.update(config)
            
            # Convert back to list format
            new_env = [f"{k}={v}" for k, v in env_dict.items()]
            
            # Apply secrets if provided
            if secrets:
                for secret_name, secret_value in secrets.items():
                    secret_config = self.create_secret(
                        f"{container_name}_{secret_name}",
                        secret_value
                    )
                    if 'error' not in secret_config:
                        new_env.append(f"{secret_name}={secret_config['id']}")
            
            # Update container
            container.update(
                env=new_env
            )
            
            return {
                'status': 'success',
                'container': container_name,
                'updated_config': config,
                'secrets_applied': len(secrets) if secrets else 0
            }
            
        except Exception as e:
            return {'error': f'Failed to apply configuration: {str(e)}'}
    
    def generate_config_file(self, 
                           template_path: str,
                           output_path: str,
                           variables: Dict[str, str]) -> Dict:
        try:
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            # Replace variables in template
            for key, value in variables.items():
                template_content = template_content.replace(
                    f"${{{key}}}", value
                )
            
            # Write configured file
            with open(output_path, 'w') as f:
                f.write(template_content)
            
            return {
                'status': 'success',
                'template': template_path,
                'output': output_path,
                'variables_replaced': len(variables)
            }
        except Exception as e:
            return {'error': f'Failed to generate config file: {str(e)}'}

# Example usage
config_manager = ConfigurationManager()

# Create a test container
container = config_manager.client.containers.run(
    "nginx:latest",
    detach=True,
    name="config_test"
)

# Apply configuration
config_result = config_manager.apply_configuration(
    "config_test",
    {
        'APP_ENV': 'production',
        'DEBUG': 'false'
    },
    secrets={
        'API_KEY': 'very-secret-key'
    }
)
print(f"Configuration result: {json.dumps(config_result, indent=2)}")

# Generate config file from template
with open('nginx.conf.template', 'w') as f:
    f.write("""
server {
    listen ${PORT};
    server_name ${SERVER_NAME};
    root ${ROOT_PATH};
}
    """)

config_file_result = config_manager.generate_config_file(
    'nginx.conf.template',
    'nginx.conf',
    {
        'PORT': '80',
        'SERVER_NAME': 'example.com',
        'ROOT_PATH': '/var/www/html'
    }
)
print(f"Config file generation result: {json.dumps(config_file_result, indent=2)}")

# Cleanup
container.stop()
container.remove()
os.remove('nginx.conf.template')
os.remove('nginx.conf')
```

Slide 13: Docker Resource Optimization

Advanced implementation for monitoring and optimizing Docker resource usage, including CPU, memory, and network optimization strategies with automated scaling capabilities.

```python
import docker
from typing import Dict, List, Optional
import psutil
import time
from dataclasses import dataclass

@dataclass
class ResourceThresholds:
    cpu_percent: float = 80.0
    memory_percent: float = 85.0
    network_usage_mbps: float = 100.0

class ResourceOptimizer:
    def __init__(self, thresholds: Optional[ResourceThresholds] = None):
        self.client = docker.from_client()
        self.thresholds = thresholds or ResourceThresholds()
    
    def analyze_resources(self, container_id: str) -> Dict:
        try:
            container = self.client.containers.get(container_id)
            stats = next(container.stats(stream=False))
            
            # Calculate CPU usage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            cpu_percent = 0.0
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * 100.0 * psutil.cpu_count()
            
            # Calculate memory usage
            memory_stats = stats['memory_stats']
            memory_usage = memory_stats['usage']
            memory_limit = memory_stats['limit']
            memory_percent = (memory_usage / memory_limit) * 100.0
            
            # Calculate network usage
            network_stats = stats['networks']
            network_usage = sum(
                interface['rx_bytes'] + interface['tx_bytes']
                for interface in network_stats.values()
            )
            
            return {
                'container_id': container_id,
                'cpu_percent': round(cpu_percent, 2),
                'memory_percent': round(memory_percent, 2),
                'network_usage_mb': round(network_usage / (1024 * 1024), 2),
                'needs_optimization': self._check_optimization_needed({
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'network_usage_mbps': network_usage / (1024 * 1024 * 8)
                })
            }
        except Exception as e:
            return {'error': f'Failed to analyze resources: {str(e)}'}
    
    def optimize_container(self, container_id: str) -> Dict:
        try:
            analysis = self.analyze_resources(container_id)
            if not analysis.get('needs_optimization'):
                return {'status': 'optimization not needed'}
            
            container = self.client.containers.get(container_id)
            current_config = container.attrs['HostConfig']
            
            # Apply optimization strategies
            new_config = self._calculate_optimized_config(
                current_config,
                analysis
            )
            
            # Update container with new configuration
            container.update(**new_config)
            
            return {
                'status': 'success',
                'container_id': container_id,
                'optimizations_applied': new_config
            }
        except Exception as e:
            return {'error': f'Failed to optimize container: {str(e)}'}
    
    def _check_optimization_needed(self, metrics: Dict[str, float]) -> bool:
        return any([
            metrics['cpu_percent'] > self.thresholds.cpu_percent,
            metrics['memory_percent'] > self.thresholds.memory_percent,
            metrics['network_usage_mbps'] > self.thresholds.network_usage_mbps
        ])
    
    def _calculate_optimized_config(self, 
                                  current_config: Dict,
                                  analysis: Dict) -> Dict:
        new_config = {}
        
        # CPU optimization
        if analysis['cpu_percent'] > self.thresholds.cpu_percent:
            new_config['cpu_shares'] = current_config.get('CpuShares', 1024) * 1.5
        
        # Memory optimization
        if analysis['memory_percent'] > self.thresholds.memory_percent:
            current_memory = current_config.get('Memory', 0)
            if current_memory > 0:
                new_config['memory'] = int(current_memory * 1.2)
        
        return new_config

# Example usage
optimizer = ResourceOptimizer()

# Create a test container with some load
container = optimizer.client.containers.run(
    "nginx:latest",
    detach=True,
    name="resource_test"
)

# Analyze resources
time.sleep(5)  # Wait for container to stabilize
analysis = optimizer.analyze_resources(container.id)
print(f"Resource analysis: {json.dumps(analysis, indent=2)}")

# Optimize if needed
if analysis.get('needs_optimization'):
    optimization_result = optimizer.optimize_container(container.id)
    print(f"Optimization result: {json.dumps(optimization_result, indent=2)}")

# Cleanup
container.stop()
container.remove()
```

Slide 14: Additional Resources

*   "Container Security: A Comprehensive Analysis" - [https://arxiv.org/abs/2104.12762](https://arxiv.org/abs/2104.12762)
*   "Automated Docker Container Management: A Survey" - [https://arxiv.org/abs/2203.09675](https://arxiv.org/abs/2203.09675)
*   "Resource Optimization in Container-Based Cloud Environments" - [https://arxiv.org/abs/2105.14852](https://arxiv.org/abs/2105.14852)
*   "Docker Configuration Management: Best Practices and Security Implications" - [https://arxiv.org/abs/2201.08753](https://arxiv.org/abs/2201.08753)
*   "Performance Analysis of Container Orchestration Systems" - [https://arxiv.org/abs/2202.13275](https://arxiv.org/abs/2202.13275)

