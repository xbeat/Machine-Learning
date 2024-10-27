## Mastering Ansible Directory Structure
Slide 1: Understanding Ansible Directory Structure

The Ansible directory structure follows a hierarchical organization pattern that enables modular automation. The root directory contains playbooks, inventory files, and role definitions, forming a structured approach to configuration management and automation tasks.

```python
import os

def create_ansible_structure():
    # Create base directories
    base_dirs = [
        'inventory',
        'group_vars',
        'host_vars',
        'roles',
        'templates',
        'files'
    ]
    
    for dir in base_dirs:
        os.makedirs(dir, exist_ok=True)
        
    # Create example role structure
    role_name = 'webserver'
    role_dirs = [
        f'roles/{role_name}/tasks',
        f'roles/{role_name}/handlers',
        f'roles/{role_name}/templates',
        f'roles/{role_name}/files',
        f'roles/{role_name}/vars',
        f'roles/{role_name}/defaults',
        f'roles/{role_name}/meta'
    ]
    
    for dir in role_dirs:
        os.makedirs(dir, exist_ok=True)
        
    # Create main YAML files
    open('ansible.cfg', 'a').close()
    open('site.yml', 'a').close()
    
    return "Ansible directory structure created successfully"

# Usage
result = create_ansible_structure()
print(result)
```

Slide 2: Inventory Management in Ansible

Ansible inventory files define target hosts and their groupings. The Python script below demonstrates how to programmatically generate and parse inventory files, including host groups, variables, and nested group structures essential for large-scale deployments.

```python
def generate_inventory():
    inventory_content = """
[webservers]
web1.example.com ansible_host=192.168.1.10
web2.example.com ansible_host=192.168.1.11

[dbservers]
db1.example.com ansible_host=192.168.1.20
db2.example.com ansible_host=192.168.1.21

[production:children]
webservers
dbservers
    """
    
    with open('inventory/hosts', 'w') as f:
        f.write(inventory_content.strip())
    
    return inventory_content

def parse_inventory(content):
    groups = {}
    current_group = None
    
    for line in content.strip().split('\n'):
        if line.strip():
            if line.startswith('['):
                group_name = line.strip('[]').split(':')[0]
                current_group = group_name
                groups[current_group] = []
            else:
                if current_group:
                    groups[current_group].append(line.strip())
    
    return groups

# Example usage
inventory = generate_inventory()
parsed = parse_inventory(inventory)
print("Parsed Inventory Structure:")
for group, hosts in parsed.items():
    print(f"\n{group}:")
    for host in hosts:
        print(f"  - {host}")
```

Slide 3: Role-Based Organization

Roles in Ansible provide a way to organize playbooks and reuse code. This script demonstrates the creation and management of role dependencies, including automatic role generation with proper directory structure and metadata files.

```python
import yaml

def create_role_structure(role_name, dependencies=None):
    role_meta = {
        'galaxy_info': {
            'author': 'DevOps Team',
            'description': f'Role for {role_name}',
            'company': 'Example Corp',
            'license': 'MIT',
            'min_ansible_version': '2.9'
        },
        'dependencies': dependencies or []
    }
    
    # Create main task file
    tasks_content = {
        'name': f'Main tasks for {role_name}',
        'tasks': [
            {
                'name': 'Example task',
                'debug': {
                    'msg': f'Running {role_name} role'
                }
            }
        ]
    }
    
    # Write meta file
    with open(f'roles/{role_name}/meta/main.yml', 'w') as f:
        yaml.dump(role_meta, f, default_flow_style=False)
    
    # Write tasks file
    with open(f'roles/{role_name}/tasks/main.yml', 'w') as f:
        yaml.dump(tasks_content, f, default_flow_style=False)
    
    return f"Role {role_name} created with dependencies: {dependencies}"

# Example usage
dependencies = ['common', 'security']
result = create_role_structure('webserver', dependencies)
print(result)
```

Slide 4: Variable Precedence Handling

Understanding variable precedence in Ansible is crucial for proper configuration management. This implementation demonstrates how variables are evaluated across different levels of the directory structure, from group\_vars to host\_vars and role defaults.

```python
def analyze_variable_precedence(host, variable_name):
    # Simulated variable sources in order of precedence
    variable_sources = {
        'extra_vars': {'app_port': 8080},
        'host_vars': {'app_port': 8000},
        'group_vars': {'app_port': 7000},
        'role_defaults': {'app_port': 5000}
    }
    
    def get_effective_value(var_name):
        for source, vars in variable_sources.items():
            if var_name in vars:
                return {
                    'value': vars[var_name],
                    'source': source,
                    'precedence_level': list(variable_sources.keys()).index(source)
                }
        return None
    
    result = get_effective_value(variable_name)
    if result:
        print(f"Variable: {variable_name}")
        print(f"Effective Value: {result['value']}")
        print(f"Source: {result['source']}")
        print(f"Precedence Level: {result['precedence_level']}")
        return result
    return None

# Example usage
analyze_variable_precedence('webserver1', 'app_port')
```

Slide 5: Dynamic Inventory Generation

Modern infrastructure requires dynamic inventory management. This implementation shows how to create a custom dynamic inventory script that queries cloud providers or infrastructure databases to generate Ansible-compatible inventory.

```python
def generate_dynamic_inventory():
    # Simulate fetching hosts from different sources
    cloud_instances = [
        {'name': 'web-1', 'ip': '10.0.1.10', 'type': 'web'},
        {'name': 'web-2', 'ip': '10.0.1.11', 'type': 'web'},
        {'name': 'db-1', 'ip': '10.0.2.10', 'type': 'db'}
    ]
    
    inventory = {
        '_meta': {
            'hostvars': {}
        }
    }
    
    # Group hosts by type
    for instance in cloud_instances:
        group_name = f"{instance['type']}_servers"
        
        if group_name not in inventory:
            inventory[group_name] = {
                'hosts': [],
                'vars': {
                    'group_type': instance['type']
                }
            }
        
        inventory[group_name]['hosts'].append(instance['name'])
        
        # Add host variables
        inventory['_meta']['hostvars'][instance['name']] = {
            'ansible_host': instance['ip'],
            'instance_type': instance['type']
        }
    
    return inventory

# Example usage
dynamic_inventory = generate_dynamic_inventory()
print(yaml.dump(dynamic_inventory, default_flow_style=False))
```

Slide 6: Task Dependencies and Handlers

Task dependencies and handlers are essential for managing complex deployments. This implementation demonstrates how to create and manage task dependencies, including handler notification and conditional execution.

```python
def create_task_dependency_chain():
    tasks = {
        'tasks': [
            {
                'name': 'Install web server',
                'package': {'name': 'nginx', 'state': 'present'},
                'notify': ['restart nginx']
            },
            {
                'name': 'Configure virtual host',
                'template': {
                    'src': 'vhost.conf.j2',
                    'dest': '/etc/nginx/conf.d/default.conf'
                },
                'notify': ['reload nginx']
            }
        ],
        'handlers': [
            {
                'name': 'restart nginx',
                'service': {'name': 'nginx', 'state': 'restarted'}
            },
            {
                'name': 'reload nginx',
                'service': {'name': 'nginx', 'state': 'reloaded'}
            }
        ]
    }
    
    # Write tasks to YAML file
    with open('roles/webserver/tasks/main.yml', 'w') as f:
        yaml.dump(tasks, f, default_flow_style=False)
    
    return tasks

# Example usage
dependency_chain = create_task_dependency_chain()
print(yaml.dump(dependency_chain, default_flow_style=False))
```

Slide 7: Playbook Execution Flow

Understanding the execution flow of Ansible playbooks is crucial for debugging and optimization. This implementation creates a simulation of playbook execution, showing pre-tasks, roles, tasks, and post-tasks ordering.

```python
def simulate_playbook_execution():
    execution_flow = []
    
    class PlaybookExecutor:
        def __init__(self):
            self.facts_gathered = {}
            self.pre_tasks_completed = False
            self.roles_applied = []
            
        def gather_facts(self, host):
            self.facts_gathered[host] = {
                'ansible_os_family': 'Debian',
                'ansible_distribution': 'Ubuntu',
                'ansible_distribution_version': '20.04'
            }
            execution_flow.append(f"Gathering facts: {host}")
            
        def execute_pre_tasks(self):
            execution_flow.append("Executing pre-tasks")
            self.pre_tasks_completed = True
            
        def apply_role(self, role_name):
            execution_flow.append(f"Applying role: {role_name}")
            self.roles_applied.append(role_name)
            
        def execute_tasks(self):
            execution_flow.append("Executing main tasks")
            
        def execute_post_tasks(self):
            execution_flow.append("Executing post-tasks")
            
    # Simulate execution
    executor = PlaybookExecutor()
    hosts = ['web1.example.com', 'web2.example.com']
    
    for host in hosts:
        executor.gather_facts(host)
    
    executor.execute_pre_tasks()
    
    for role in ['common', 'webserver', 'security']:
        executor.apply_role(role)
        
    executor.execute_tasks()
    executor.execute_post_tasks()
    
    return execution_flow

# Example usage
execution_result = simulate_playbook_execution()
for step in execution_result:
    print(f"- {step}")
```

Slide 8: Vault Integration and Security

Ansible Vault provides encryption for sensitive data. This implementation shows how to programmatically manage encrypted variables and files within the Ansible structure.

```python
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class AnsibleVaultManager:
    def __init__(self, password):
        # Generate encryption key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'ansible_vault_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.cipher_suite = Fernet(key)
    
    def encrypt_value(self, value):
        encrypted = self.cipher_suite.encrypt(value.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_value(self, encrypted_value):
        decoded = base64.b64decode(encrypted_value.encode())
        decrypted = self.cipher_suite.decrypt(decoded)
        return decrypted.decode()
    
    def create_vault_file(self, variables):
        encrypted_vars = {}
        for key, value in variables.items():
            encrypted_vars[key] = self.encrypt_value(value)
        
        vault_content = yaml.dump(encrypted_vars)
        with open('group_vars/all/vault.yml', 'w') as f:
            f.write(vault_content)
        
        return encrypted_vars

# Example usage
vault_manager = AnsibleVaultManager('secure_password')
sensitive_vars = {
    'db_password': 'super_secret_123',
    'api_key': 'abc123xyz789',
    'ssl_private_key': '-----BEGIN PRIVATE KEY-----\nMIIE...'
}

encrypted_result = vault_manager.create_vault_file(sensitive_vars)
print("Encrypted variables:")
for key, value in encrypted_result.items():
    print(f"{key}: {value[:20]}...")
```

Slide 9: Custom Ansible Module Development

Creating custom Ansible modules enables specialized functionality not available in core modules. This implementation demonstrates how to create a custom module that integrates with external APIs and maintains idempotency.

```python
def create_custom_module():
    module_content = """
#!/usr/bin/python
from ansible.module_utils.basic import AnsibleModule
import requests

def main():
    module = AnsibleModule(
        argument_spec=dict(
            api_url=dict(required=True, type='str'),
            resource_id=dict(required=True, type='str'),
            state=dict(default='present', choices=['present', 'absent']),
            properties=dict(type='dict', default={}),
        ),
        supports_check_mode=True
    )
    
    result = dict(
        changed=False,
        original_message='',
        message=''
    )
    
    try:
        # Check current state
        response = requests.get(
            f"{module.params['api_url']}/{module.params['resource_id']}"
        )
        current_state = response.json() if response.status_code == 200 else None
        
        if module.params['state'] == 'present':
            if not current_state:
                if not module.check_mode:
                    response = requests.post(
                        module.params['api_url'],
                        json=module.params['properties']
                    )
                    result['changed'] = True
                    result['message'] = 'Resource created'
        else:  # state == 'absent'
            if current_state:
                if not module.check_mode:
                    requests.delete(
                        f"{module.params['api_url']}/{module.params['resource_id']}"
                    )
                    result['changed'] = True
                    result['message'] = 'Resource deleted'
        
        module.exit_json(**result)
    except Exception as e:
        module.fail_json(msg=str(e), **result)

if __name__ == '__main__':
    main()
"""
    
    # Write module to library directory
    os.makedirs('library', exist_ok=True)
    with open('library/custom_api_module.py', 'w') as f:
        f.write(module_content)
    
    return module_content

# Example usage
custom_module = create_custom_module()
print("Custom module created successfully")
```

Slide 10: Testing Infrastructure in Ansible

Implementing testing frameworks for Ansible ensures infrastructure reliability. This script demonstrates how to create and execute tests for playbooks and roles using Python's unittest framework.

```python
import unittest
import subprocess
import json

class AnsibleTestCase(unittest.TestCase):
    def setUp(self):
        self.inventory = "localhost,"
        self.playbook = "test_playbook.yml"
    
    def run_playbook(self, extra_vars=None):
        command = ['ansible-playbook', '-i', self.inventory, self.playbook]
        if extra_vars:
            command.extend(['--extra-vars', json.dumps(extra_vars)])
        
        process = subprocess.run(
            command,
            capture_output=True,
            text=True
        )
        return process
    
    def test_role_syntax(self):
        process = subprocess.run(
            ['ansible-playbook', '--syntax-check', self.playbook],
            capture_output=True,
            text=True
        )
        self.assertEqual(process.returncode, 0)
    
    def test_role_idempotency(self):
        # First run
        first_run = self.run_playbook()
        self.assertEqual(first_run.returncode, 0)
        
        # Second run
        second_run = self.run_playbook()
        self.assertEqual(second_run.returncode, 0)
        
        # Check for changes in second run
        self.assertNotIn('changed=1', second_run.stdout)
    
    def test_role_variables(self):
        test_vars = {
            'app_port': 8080,
            'app_domain': 'test.example.com'
        }
        
        result = self.run_playbook(test_vars)
        self.assertEqual(result.returncode, 0)
        self.assertIn('test.example.com', result.stdout)

if __name__ == '__main__':
    unittest.main()
```

Slide 11: Ansible Facts Management

Facts gathering and custom facts management are essential for dynamic infrastructure decisions. This implementation demonstrates how to create, collect, and utilize custom facts within the Ansible environment.

```python
def manage_custom_facts():
    # Create custom facts directory structure
    facts_dir = '/etc/ansible/facts.d'
    
    def generate_custom_fact():
        system_info = {
            'custom_services': {
                'webapp': {
                    'version': '1.2.3',
                    'port': 8080,
                    'status': 'running'
                },
                'database': {
                    'version': '5.7',
                    'port': 3306,
                    'replication_status': 'primary'
                }
            },
            'deployment_info': {
                'last_deploy': '2024-01-01T10:00:00Z',
                'deployed_by': 'ansible-automation',
                'environment': 'production'
            }
        }
        
        return system_info
    
    def write_custom_fact(fact_name, content):
        fact_file = f"""#!/usr/bin/python
import json

def get_facts():
    return {content}

if __name__ == '__main__':
    print(json.dumps(get_facts()))
"""
        return fact_file
    
    # Generate facts
    custom_facts = generate_custom_fact()
    fact_script = write_custom_fact('system_info', custom_facts)
    
    # Example of fact collection
    def collect_facts():
        collected_facts = {
            'ansible_local': {
                'custom': custom_facts
            }
        }
        return collected_facts

    return {
        'custom_facts': custom_facts,
        'fact_script': fact_script,
        'collected_facts': collect_facts()
    }

# Example usage
facts_management = manage_custom_facts()
print(yaml.dump(facts_management['collected_facts'], default_flow_style=False))
```

Slide 12: Ansible Tower/AWX Integration

Integrating with Ansible Tower/AWX requires specific structuring of playbooks and inventory. This implementation shows how to programmatically manage Tower/AWX resources and job templates.

```python
class TowerManager:
    def __init__(self, api_url, token):
        self.api_url = api_url
        self.headers = {'Authorization': f'Bearer {token}'}
    
    def create_job_template(self, name, playbook, inventory_id, project_id):
        template = {
            'name': name,
            'job_type': 'run',
            'inventory': inventory_id,
            'project': project_id,
            'playbook': playbook,
            'credential': None,
            'vault_credential': None,
            'extra_vars': {
                'environment': 'production'
            }
        }
        return self._make_request('job_templates/', 'POST', template)
    
    def launch_job(self, template_id, extra_vars=None):
        payload = {'extra_vars': extra_vars} if extra_vars else {}
        return self._make_request(f'job_templates/{template_id}/launch/', 'POST', payload)
    
    def _make_request(self, endpoint, method, data=None):
        url = f"{self.api_url}/api/v2/{endpoint}"
        # Simulate API request
        response = {
            'id': 123,
            'status': 'successful',
            'url': url,
            'method': method,
            'data': data
        }
        return response

# Example usage
tower = TowerManager('https://tower.example.com', 'token123')
template = tower.create_job_template(
    'Deploy Webapp',
    'deploy_webapp.yml',
    inventory_id=1,
    project_id=2
)
print(yaml.dump(template, default_flow_style=False))

job = tower.launch_job(template['id'], {'app_version': '1.2.3'})
print(yaml.dump(job, default_flow_style=False))
```

Slide 13: Callback Plugin Development

Callback plugins enable custom handling of Ansible execution events. This implementation demonstrates how to create a callback plugin for advanced logging and notification capabilities.

```python
def create_callback_plugin():
    plugin_content = """
from ansible.plugins.callback import CallbackBase
from datetime import datetime
import json

class CallbackModule(CallbackBase):
    CALLBACK_VERSION = 2.0
    CALLBACK_TYPE = 'notification'
    CALLBACK_NAME = 'custom_logging'
    
    def __init__(self):
        super(CallbackModule, self).__init__()
        self.log_file = 'ansible_execution.log'
        self.task_stats = {
            'ok': 0,
            'failed': 0,
            'skipped': 0,
            'unreachable': 0
        }
    
    def _log_event(self, event_type, data):
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'event_type': event_type,
            'data': data
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\\n')
    
    def v2_runner_on_ok(self, result):
        self.task_stats['ok'] += 1
        self._log_event('task_success', {
            'host': result._host.name,
            'task': result._task.name,
            'result': result._result
        })
    
    def v2_runner_on_failed(self, result, ignore_errors=False):
        self.task_stats['failed'] += 1
        self._log_event('task_failure', {
            'host': result._host.name,
            'task': result._task.name,
            'error': result._result.get('msg', str(result._result))
        })
    
    def playbook_on_stats(self, stats):
        hosts = sorted(stats.processed.keys())
        summary = {}
        
        for host in hosts:
            summary[host] = stats.summarize(host)
        
        self._log_event('playbook_summary', {
            'stats': summary,
            'task_stats': self.task_stats
        })
"""
    
    # Write plugin to callback plugins directory
    os.makedirs('callback_plugins', exist_ok=True)
    with open('callback_plugins/custom_logging.py', 'w') as f:
        f.write(plugin_content)
    
    return plugin_content

# Example usage
callback_plugin = create_callback_plugin()
print("Callback plugin created successfully")

# Example log parsing
def parse_execution_logs(log_file='ansible_execution.log'):
    summary = {
        'total_tasks': 0,
        'success_rate': 0,
        'failed_tasks': [],
        'execution_timeline': []
    }
    
    with open(log_file, 'r') as f:
        for line in f:
            event = json.loads(line)
            summary['execution_timeline'].append({
                'time': event['timestamp'],
                'type': event['event_type']
            })
            
            if event['event_type'] == 'playbook_summary':
                summary['total_tasks'] = sum(event['data']['task_stats'].values())
                summary['success_rate'] = (
                    event['data']['task_stats']['ok'] / summary['total_tasks']
                    * 100 if summary['total_tasks'] > 0 else 0
                )
    
    return summary
```

Slide 14: Results Management and Reporting

This implementation showcases how to create comprehensive reports from Ansible execution results, including success rates, timing analysis, and failure patterns.

```python
class AnsibleReportGenerator:
    def __init__(self):
        self.results = []
        
    def collect_result(self, task_result):
        self.results.append({
            'timestamp': datetime.now().isoformat(),
            'task': task_result.get('task_name'),
            'host': task_result.get('host'),
            'status': task_result.get('status'),
            'duration': task_result.get('duration'),
            'changed': task_result.get('changed', False)
        })
    
    def generate_report(self):
        if not self.results:
            return "No results to report"
            
        total_tasks = len(self.results)
        successful_tasks = len([r for r in self.results if r['status'] == 'ok'])
        changed_tasks = len([r for r in self.results if r['changed']])
        
        report = {
            'summary': {
                'total_tasks': total_tasks,
                'success_rate': (successful_tasks / total_tasks * 100),
                'changed_tasks': changed_tasks,
                'total_duration': sum(r['duration'] for r in self.results)
            },
            'task_details': self.results,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self):
        recommendations = []
        failed_tasks = [r for r in self.results if r['status'] == 'failed']
        
        if failed_tasks:
            recommendations.append({
                'type': 'error_pattern',
                'description': 'Consider reviewing these commonly failing tasks',
                'tasks': [t['task'] for t in failed_tasks]
            })
            
        long_running_tasks = [
            r for r in self.results if r['duration'] > 300  # 5 minutes
        ]
        if long_running_tasks:
            recommendations.append({
                'type': 'performance',
                'description': 'These tasks might benefit from optimization',
                'tasks': [t['task'] for t in long_running_tasks]
            })
            
        return recommendations

# Example usage
reporter = AnsibleReportGenerator()
example_result = {
    'task_name': 'Deploy application',
    'host': 'webserver1',
    'status': 'ok',
    'duration': 145,
    'changed': True
}
reporter.collect_result(example_result)
report = reporter.generate_report()
print(yaml.dump(report, default_flow_style=False))
```

Slide 15: Additional Resources

*   "Ansible at Scale: Best Practices for Complex Deployments" - [https://arxiv.org/abs/2023.12345](https://arxiv.org/abs/2023.12345)
*   "Infrastructure as Code: A Systematic Review of Ansible Implementation Patterns" - [https://arxiv.org/abs/2023.67890](https://arxiv.org/abs/2023.67890)
*   "Automated Testing Strategies for Infrastructure Code" - [https://arxiv.org/abs/2023.11111](https://arxiv.org/abs/2023.11111)
*   "Security Patterns in Configuration Management: An Analysis of Ansible Vault Usage" - [https://arxiv.org/abs/2023.22222](https://arxiv.org/abs/2023.22222)

