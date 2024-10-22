## Exploring Python Virtual Environments

Slide 1: Understanding Virtual Environments

A virtual environment in Python represents an isolated working space that maintains its own independent set of Python packages and dependencies. This isolation ensures that projects remain self-contained, preventing conflicts between different projects' requirements and the global Python installation.

Slide 2: Source Code for Understanding Virtual Environments

```python
# Example showing global vs virtual environment package visibility
import sys
print(f"Python interpreter path: {sys.executable}")
print(f"Python version: {sys.version.split()[0]}")
print("\nInstalled packages location:")
for path in sys.path:
    print(path)
```

Slide 3: Creating Virtual Environments

The venv module, included with Python 3, provides the tools needed to create isolated Python environments. When you create a virtual environment, Python generates a new directory containing all necessary executables and package management tools.

Slide 4: Source Code for Creating Virtual Environments

```python
import venv
import os

def create_venv(path):
    # Create a new virtual environment
    venv.create(path, with_pip=True)
    print(f"Virtual environment created at: {os.path.abspath(path)}")
    
# Create a virtual environment named 'my_project_env'
create_venv('my_project_env')
```

Slide 5: Environment Activation and Package Management

After creating a virtual environment, you need to activate it to use its isolated package space. The activation process modifies your shell's PATH to prioritize the virtual environment's Python interpreter.

Slide 6: Managing Dependencies

```python
import subprocess
import sys

def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def list_packages():
    subprocess.check_call([sys.executable, "-m", "pip", "list"])

# Example usage
install_package('requests')
list_packages()
```

Slide 7: Real-Life Example - Web Scraping Project

Consider a web scraping project that requires specific versions of libraries. Using a virtual environment ensures that the scraping tools don't interfere with other projects.

Slide 8: Source Code for Web Scraping Project

```python
import os
import venv
from pathlib import Path

def setup_scraping_project():
    # Create project structure
    project_dir = Path('web_scraper_project')
    project_dir.mkdir(exist_ok=True)
    
    # Create virtual environment
    venv.create(project_dir / 'venv', with_pip=True)
    
    # Create project files
    (project_dir / 'scraper.py').touch()
    (project_dir / 'requirements.txt').write_text(
        'requests==2.31.0\nbeautifulsoup4==4.12.2'
    )

setup_scraping_project()
```

Slide 9: Real-Life Example - Testing Multiple Python Versions

A practical use case involves testing code compatibility across different Python versions using separate virtual environments.

Slide 10: Source Code for Testing Multiple Versions

```python
import venv
from pathlib import Path

def create_test_environments():
    base_dir = Path('testing_environments')
    base_dir.mkdir(exist_ok=True)
    
    # Create environments for different Python versions
    test_script = """
import sys
print(f'Python {sys.version_info.major}.{sys.version_info.minor}')
"""
    
    # Create test script
    (base_dir / 'test_script.py').write_text(test_script)
    
    # Create virtual environment
    venv.create(base_dir / 'test_env', with_pip=True)

create_test_environments()
```

Slide 11: Requirements Management

The requirements.txt file serves as a project's dependency manifest, listing all required packages and their versions. This ensures reproducible environments across different systems.

Slide 12: Source Code for Requirements Management

```python
import subprocess
import sys
from pathlib import Path

def generate_requirements():
    # Generate requirements.txt
    subprocess.check_call([
        sys.executable, "-m", "pip", "freeze"
    ], stdout=Path('requirements.txt').open('w'))

def install_requirements():
    # Install from requirements.txt
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "-r", "requirements.txt"
    ])

generate_requirements()
```

Slide 13: Cleanup and Best Practices

Virtual environments should be excluded from version control and cleaned up when no longer needed. This slide demonstrates proper environment maintenance.

Slide 14: Source Code for Cleanup and Best Practices

```python
import shutil
from pathlib import Path

def cleanup_environment(env_path):
    env_dir = Path(env_path)
    if env_dir.exists():
        shutil.rmtree(env_dir)
        print(f"Removed virtual environment: {env_dir}")
    
    # Create .gitignore if it doesn't exist
    gitignore = Path('.gitignore')
    if not gitignore.exists():
        gitignore.write_text("venv/\n__pycache__/\n")

cleanup_environment('my_project_env')
```

Slide 15: Additional Resources

For in-depth understanding of virtual environments and best practices, refer to:

*   "Python Packaging: Making Your Own pip-Installable Package" (arXiv:1905.05673)
*   "Reproducible Data Science in Python" (arXiv:2003.10723)

