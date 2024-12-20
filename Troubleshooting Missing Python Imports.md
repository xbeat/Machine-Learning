## Troubleshooting Missing Python Imports
Slide 1: Understanding ImportError in Python

ImportError is a common exception in Python that occurs when a module or attribute cannot be imported. This error can be intimidating for beginners, but understanding its causes and solutions is crucial for effective Python programming.

Slide 2: Source Code for Understanding ImportError in Python

```python
# Attempting to import a non-existent module
try:
    import non_existent_module
except ImportError as e:
    print(f"ImportError occurred: {e}")

# Attempting to import a non-existent attribute from a module
try:
    from math import non_existent_function
except ImportError as e:
    print(f"ImportError occurred: {e}")
```

Slide 3: Common Causes of ImportError

ImportError can occur due to various reasons, including:

1.  The module is not installed in your Python environment.
2.  The module name is misspelled.
3.  The module is not in the Python path.
4.  The module is installed but for a different Python version.
5.  There are circular imports in your code.

Slide 4: Source Code for Common Causes of ImportError

```python
# Example of a misspelled module name
try:
    import pandas as pd  # Correct spelling
except ImportError:
    print("pandas module not found")

try:
    import panadas as pd  # Misspelled
except ImportError:
    print("panadas module not found (misspelled)")

# Example of a module not in the Python path
import sys
print(f"Python path: {sys.path}")
```

Slide 5: Checking Installed Packages

To verify if a package is installed and its version, you can use pip, Python's package installer. This helps in diagnosing ImportError related to missing or outdated packages.

Slide 6: Source Code for Checking Installed Packages

```python
import subprocess

def check_package(package_name):
    try:
        result = subprocess.run(["pip", "show", package_name], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{package_name} is installed:")
            print(result.stdout)
        else:
            print(f"{package_name} is not installed.")
    except Exception as e:
        print(f"An error occurred: {e}")

check_package("numpy")
check_package("non_existent_package")
```

Slide 7: Installing and Upgrading Packages

If a package is missing or outdated, you can install or upgrade it using pip. This is often the solution to ImportError caused by missing modules.

Slide 8: Source Code for Installing and Upgrading Packages

```python
import subprocess

def install_or_upgrade_package(package_name):
    try:
        subprocess.run(["pip", "install", "--upgrade", package_name], check=True)
        print(f"Successfully installed/upgraded {package_name}")
    except subprocess.CalledProcessError:
        print(f"Failed to install/upgrade {package_name}")

install_or_upgrade_package("requests")
```

Slide 9: Handling ImportError Gracefully

When developing Python applications, it's good practice to handle ImportError gracefully. This allows your program to provide meaningful feedback or use alternative modules when imports fail.

Slide 10: Source Code for Handling ImportError Gracefully

```python
try:
    import numpy as np
    print("NumPy is available. Using NumPy for calculations.")
except ImportError:
    print("NumPy is not available. Using built-in functions for calculations.")
    
    def array_sum(arr):
        return sum(arr)
else:
    def array_sum(arr):
        return np.sum(arr)

# Example usage
numbers = [1, 2, 3, 4, 5]
result = array_sum(numbers)
print(f"Sum of the array: {result}")
```

Slide 11: Real-Life Example: Web Scraping

Let's consider a real-life example of handling ImportError in a web scraping script. We'll try to use the 'requests' library, but fall back to 'urllib' if 'requests' is not available.

Slide 12: Source Code for Web Scraping Example

```python
def fetch_webpage(url):
    try:
        import requests
        print("Using requests library")
        response = requests.get(url)
        return response.text
    except ImportError:
        print("Requests not available. Using urllib")
        from urllib.request import urlopen
        with urlopen(url) as response:
            return response.read().decode('utf-8')

# Example usage
url = "https://example.com"
content = fetch_webpage(url)
print(f"Fetched {len(content)} characters from {url}")
```

Slide 13: Real-Life Example: Data Analysis

Another common scenario is handling ImportError in data analysis tasks. Let's create a function that calculates the mean of a dataset, using NumPy if available, or falling back to a custom implementation.

Slide 14: Source Code for Data Analysis Example

```python
def calculate_mean(data):
    try:
        import numpy as np
        print("Using NumPy for mean calculation")
        return np.mean(data)
    except ImportError:
        print("NumPy not available. Using custom mean calculation")
        return sum(data) / len(data)

# Example usage
dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mean_value = calculate_mean(dataset)
print(f"Mean of the dataset: {mean_value}")
```

Slide 15: Additional Resources

For more information on handling ImportError and Python package management:

1.  Python Official Documentation on Modules: [https://docs.python.org/3/tutorial/modules.html](https://docs.python.org/3/tutorial/modules.html)
2.  pip documentation: [https://pip.pypa.io/en/stable/](https://pip.pypa.io/en/stable/)
3.  "Mastering ImportError Handling in Python" (arXiv:2103.12345)

Note: The arXiv reference is fictional and used for illustrative purposes only. Always verify the authenticity of sources.

