## Securing Python Mitigating Shell Injection Vulnerabilities
Slide 1: Understanding Shell Injection Vulnerability

Shell injection is a critical security vulnerability in Python applications that can occur when user input is directly used in shell commands. This vulnerability allows attackers to execute arbitrary commands on the system, potentially leading to severe security breaches.

Slide 2: Source Code for Understanding Shell Injection Vulnerability

```python
import subprocess

# Vulnerable function
def run_command(user_input):
    command = f"echo {user_input}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

# Example usage
user_input = input("Enter your name: ")
output = run_command(user_input)
print(f"Output: {output}")

# Potential attack
malicious_input = "; ls -la"
output = run_command(malicious_input)
print(f"Malicious output: {output}")
```

Slide 3: The Subprocess Module and Its Risks

The subprocess module in Python provides powerful tools for executing shell commands. However, when used improperly, it can lead to severe security vulnerabilities. Methods like Popen, run, and check\_output can execute commands based on arguments, which can be dangerous if user input is not properly sanitized.

Slide 4: Source Code for The Subprocess Module and Its Risks

```python
import subprocess

# Potentially risky usage of subprocess
user_input = input("Enter a file name to search: ")
command = f"find / -name {user_input}"
result = subprocess.run(command, shell=True, capture_output=True, text=True)
print(result.stdout)

# An attacker could input: *.txt -exec rm {} \;
# This would find and delete all .txt files
```

Slide 5: Command Injection Vulnerabilities

Command injection occurs when an attacker can manipulate the input in such a way that additional commands are executed by the system. This can happen when user input is directly incorporated into shell commands without proper validation or escaping.

Slide 6: Source Code for Command Injection Vulnerabilities

```python
import os

def get_user_files(username):
    # Vulnerable function
    command = f"ls -l /home/{username}"
    return os.popen(command).read()

# Normal usage
print(get_user_files("alice"))

# Potential attack
malicious_username = "alice; rm -rf /"
print(get_user_files(malicious_username))
# This could potentially delete all files on the system
```

Slide 7: Best Practices for Secure Command Execution

To mitigate shell injection vulnerabilities, it's crucial to follow best practices when executing commands. These include using internal language features instead of shell commands when possible, validating and sanitizing user inputs, using arrays of program arguments instead of single strings, and avoiding shell=True unless absolutely necessary.

Slide 8: Source Code for Best Practices for Secure Command Execution

```python
import subprocess
import shlex

def secure_command_execution(command, args):
    # Use shlex.quote to escape each argument
    escaped_args = [shlex.quote(arg) for arg in args]
    full_command = f"{command} {' '.join(escaped_args)}"
    
    # Use subprocess.run with shell=False
    result = subprocess.run(full_command, shell=False, capture_output=True, text=True)
    return result.stdout

# Example usage
command = "echo"
user_input = input("Enter your message: ")
output = secure_command_execution(command, [user_input])
print(f"Secure output: {output}")
```

Slide 9: Input Validation and Sanitization

One of the most effective ways to prevent shell injection is to implement strict input validation and sanitization. This involves checking user inputs against a whitelist of allowed characters or patterns and removing or escaping potentially dangerous characters.

Slide 10: Source Code for Input Validation and Sanitization

```python
import re

def sanitize_input(user_input):
    # Remove all characters except alphanumeric and spaces
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', '', user_input)
    return sanitized

def execute_command(command, user_input):
    sanitized_input = sanitize_input(user_input)
    full_command = f"{command} {sanitized_input}"
    # Execute the command securely
    # (implementation details omitted for brevity)
    return f"Executed: {full_command}"

# Example usage
user_input = input("Enter a search term: ")
result = execute_command("search", user_input)
print(result)
```

Slide 11: Using Internal Language Features

Whenever possible, it's better to use Python's built-in functions and modules instead of executing shell commands. This approach significantly reduces the risk of shell injection vulnerabilities.

Slide 12: Source Code for Using Internal Language Features

```python
import os

def list_files(directory):
    # Using os.listdir instead of 'ls' command
    return os.listdir(directory)

def search_files(directory, pattern):
    # Using os.walk and string methods instead of 'find' command
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern in file:
                matching_files.append(os.path.join(root, file))
    return matching_files

# Example usage
print("Files in current directory:", list_files('.'))
print("Python files in current directory:", search_files('.', '.py'))
```

Slide 13: Real-Life Example: Web Application File Viewer

Consider a web application that allows users to view files on the server. A vulnerable implementation might directly use user input in a shell command, while a secure version would use Python's built-in functions and implement proper input validation.

Slide 14: Source Code for Web Application File Viewer

```python
import os
from flask import Flask, request, abort

app = Flask(__name__)

@app.route('/view_file')
def view_file():
    filename = request.args.get('filename', '')
    
    # Vulnerable version (DO NOT USE)
    # return os.popen(f"cat {filename}").read()
    
    # Secure version
    if not filename or '..' in filename or filename.startswith('/'):
        abort(400)  # Bad Request
    
    safe_path = os.path.join('/safe/directory', filename)
    if not os.path.exists(safe_path):
        abort(404)  # Not Found
    
    with open(safe_path, 'r') as file:
        return file.read()

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 15: Real-Life Example: System Information Tool

A system information tool that collects various data about the server could be vulnerable to shell injection if implemented incorrectly. Here's a comparison of a vulnerable and a secure implementation.

Slide 16: Source Code for System Information Tool

```python
import platform
import psutil

def get_system_info():
    # Vulnerable version (DO NOT USE)
    # os.system('uname -a > sysinfo.txt && df -h >> sysinfo.txt && free -m >> sysinfo.txt')
    
    # Secure version
    info = {
        'system': platform.system(),
        'release': platform.release(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'disk_usage': psutil.disk_usage('/').percent
    }
    
    return info

# Example usage
system_info = get_system_info()
for key, value in system_info.items():
    print(f"{key}: {value}")
```

Slide 17: Additional Resources

For more information on Python security and best practices, consider the following resources:

1.  OWASP Python Security Project: [https://owasp.org/www-project-python-security/](https://owasp.org/www-project-python-security/)
2.  Python Security Documentation: [https://docs.python.org/3/library/security.html](https://docs.python.org/3/library/security.html)
3.  "Secure Programming HOWTO" by David A. Wheeler: [https://dwheeler.com/secure-programs/](https://dwheeler.com/secure-programs/)

Remember to always keep your Python installation and dependencies up to date, and regularly review your code for potential security vulnerabilities.

