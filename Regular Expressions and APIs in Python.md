## Regular Expressions and APIs in Python
Slide 1: Regular Expressions and APIs in Python

Regular expressions (regex) and APIs are powerful tools in Python programming. Regex allows for complex pattern matching and text manipulation, while APIs enable communication between different software applications. This presentation will explore both concepts, providing practical examples and use cases.

```python
import re
import requests

# Example: Using regex to find email addresses
text = "Contact us at info@example.com or support@company.org"
emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
print(f"Found emails: {emails}")

# Example: Making an API request
response = requests.get('https://api.example.com/data')
if response.status_code == 200:
    print("API request successful")
else:
    print(f"API request failed with status code: {response.status_code}")
```

Slide 2: Introduction to Regular Expressions

Regular expressions are sequences of characters that define a search pattern. They are used for pattern matching within strings, allowing for complex text processing and validation. Python's `re` module provides support for regular expressions.

```python
import re

# Basic pattern matching
text = "The quick brown fox jumps over the lazy dog"
pattern = r"quick.*fox"
match = re.search(pattern, text)

if match:
    print(f"Pattern found: {match.group()}")
else:
    print("Pattern not found")

# Output: Pattern found: quick brown fox
```

Slide 3: Regular Expression Metacharacters

Metacharacters are special characters in regex that have specific meanings. They allow for more flexible and powerful pattern matching. Some common metacharacters include:

* `.` (dot): Matches any character except newline
* `*`: Matches 0 or more occurrences of the previous character
* `+`: Matches 1 or more occurrences of the previous character
* `?`: Matches 0 or 1 occurrence of the previous character
* `^`: Matches the start of a string
* `$`: Matches the end of a string

```python
import re

text = "Python is awesome! Python is powerful!"

# Using metacharacters
start_with_python = re.match(r"^Python", text)
end_with_exclamation = re.search(r"!$", text)
contains_awesome = re.search(r"awe.+me", text)

print(f"Starts with 'Python': {bool(start_with_python)}")
print(f"Ends with '!': {bool(end_with_exclamation)}")
print(f"Contains 'awesome': {bool(contains_awesome)}")

# Output:
# Starts with 'Python': True
# Ends with '!': True
# Contains 'awesome': True
```

Slide 4: Regular Expression Character Classes

Character classes allow you to match any one of a set of characters. They are defined using square brackets `[]`. Some common character classes include:

* `[aeiou]`: Matches any vowel
* `[0-9]`: Matches any digit
* `[a-zA-Z]`: Matches any letter (uppercase or lowercase)

```python
import re

text = "The price is $42.99"

# Using character classes
vowels = re.findall(r'[aeiou]', text.lower())
digits = re.findall(r'[0-9]', text)
price = re.search(r'\$[0-9]+\.[0-9]{2}', text)

print(f"Vowels: {vowels}")
print(f"Digits: {digits}")
print(f"Price: {price.group() if price else 'Not found'}")

# Output:
# Vowels: ['e', 'i', 'e', 'i']
# Digits: ['4', '2', '9', '9']
# Price: $42.99
```

Slide 5: Regular Expression Quantifiers

Quantifiers specify how many times a character or group should be matched. Common quantifiers include:

* `*`: 0 or more occurrences
* `+`: 1 or more occurrences
* `?`: 0 or 1 occurrence
* `{n}`: Exactly n occurrences
* `{n,}`: n or more occurrences
* `{n,m}`: Between n and m occurrences

```python
import re

text = "The year 2023 is here! How many 9s are in 999?"

# Using quantifiers
years = re.findall(r'\d{4}', text)
nines = re.findall(r'9+', text)
optional_s = re.findall(r'9s?', text)

print(f"Years: {years}")
print(f"Sequences of 9: {nines}")
print(f"9 with optional s: {optional_s}")

# Output:
# Years: ['2023']
# Sequences of 9: ['999']
# 9 with optional s: ['9s', '999']
```

Slide 6: Regular Expression Groups and Capturing

Groups allow you to treat multiple characters as a single unit. They are created by enclosing characters in parentheses `()`. Capturing groups allow you to extract specific parts of a match.

```python
import re

text = "John Doe's phone number is (123) 456-7890"

# Using groups and capturing
pattern = r"(\w+)\s(\w+)'s phone number is \((\d{3})\)\s(\d{3}-\d{4})"
match = re.search(pattern, text)

if match:
    print(f"Full Name: {match.group(1)} {match.group(2)}")
    print(f"Area Code: {match.group(3)}")
    print(f"Phone Number: {match.group(4)}")

# Output:
# Full Name: John Doe
# Area Code: 123
# Phone Number: 456-7890
```

Slide 7: Real-Life Example: Parsing Log Files

Regular expressions are often used to parse log files and extract relevant information. Here's an example of parsing an Apache access log:

```python
import re

log_line = '192.168.1.100 - - [20/Feb/2023:10:30:45 +0000] "GET /index.html HTTP/1.1" 200 1234'

pattern = r'(\d+\.\d+\.\d+\.\d+) .* \[(.*?)\] "(.*?)" (\d+) (\d+)'
match = re.search(pattern, log_line)

if match:
    ip_address = match.group(1)
    timestamp = match.group(2)
    request = match.group(3)
    status_code = match.group(4)
    bytes_sent = match.group(5)

    print(f"IP Address: {ip_address}")
    print(f"Timestamp: {timestamp}")
    print(f"Request: {request}")
    print(f"Status Code: {status_code}")
    print(f"Bytes Sent: {bytes_sent}")

# Output:
# IP Address: 192.168.1.100
# Timestamp: 20/Feb/2023:10:30:45 +0000
# Request: GET /index.html HTTP/1.1
# Status Code: 200
# Bytes Sent: 1234
```

Slide 8: Introduction to APIs

An API (Application Programming Interface) is a set of rules and protocols that allow different software applications to communicate with each other. APIs enable developers to access functionality or data from other services or applications.

```python
import requests

# Example: Making a GET request to a public API
api_url = "https://api.publicapis.org/entries"
response = requests.get(api_url)

if response.status_code == 200:
    data = response.json()
    print(f"Total APIs available: {data['count']}")
    print(f"First API: {data['entries'][0]['API']}")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")

# Output (may vary):
# Total APIs available: 1425
# First API: AdoptAPet
```

Slide 9: Working with RESTful APIs

REST (Representational State Transfer) is an architectural style for designing networked applications. RESTful APIs use HTTP methods to perform operations on resources. Common HTTP methods include:

* GET: Retrieve data
* POST: Create new data
* PUT: Update existing data
* DELETE: Remove data

```python
import requests

api_url = "https://jsonplaceholder.typicode.com/posts"

# GET request
response = requests.get(f"{api_url}/1")
print(f"GET response status: {response.status_code}")
print(f"GET response data: {response.json()}")

# POST request
new_post = {"title": "New Post", "body": "This is a new post", "userId": 1}
response = requests.post(api_url, json=new_post)
print(f"\nPOST response status: {response.status_code}")
print(f"POST response data: {response.json()}")

# Output (may vary):
# GET response status: 200
# GET response data: {'userId': 1, 'id': 1, 'title': '...', 'body': '...'}
# POST response status: 201
# POST response data: {'title': 'New Post', 'body': '...', 'userId': 1, 'id': 101}
```

Slide 10: Handling API Authentication

Many APIs require authentication to access their resources. Common authentication methods include API keys, OAuth, and token-based authentication. Here's an example using an API key:

```python
import requests
import os

# Retrieve API key from environment variable
api_key = os.environ.get("WEATHER_API_KEY")

if not api_key:
    print("API key not found. Please set the WEATHER_API_KEY environment variable.")
else:
    # Example: OpenWeatherMap API
    city = "London"
    api_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"

    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        print(f"Weather in {city}:")
        print(f"Temperature: {data['main']['temp']}K")
        print(f"Description: {data['weather'][0]['description']}")
    else:
        print(f"Failed to retrieve weather data. Status code: {response.status_code}")

# Output (if API key is set and request is successful):
# Weather in London:
# Temperature: 283.15K
# Description: scattered clouds
```

Slide 11: Parsing API Responses

API responses often come in JSON format. Python's `json` module makes it easy to parse and work with JSON data.

```python
import requests
import json

api_url = "https://api.github.com/repos/python/cpython"
response = requests.get(api_url)

if response.status_code == 200:
    data = response.json()
    
    # Pretty print the entire response
    print(json.dumps(data, indent=2))
    
    # Extract specific information
    print(f"\nRepository: {data['full_name']}")
    print(f"Description: {data['description']}")
    print(f"Stars: {data['stargazers_count']}")
    print(f"Forks: {data['forks_count']}")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")

# Output (abbreviated):
# {
#   "id": 812646,
#   "node_id": "MDEwOlJlcG9zaXRvcnk4MTI2NDY=",
#   "name": "cpython",
#   "full_name": "python/cpython",
#   ...
# }
# 
# Repository: python/cpython
# Description: The Python programming language
# Stars: 51234
# Forks: 24567
```

Slide 12: Error Handling in API Requests

When working with APIs, it's important to handle errors gracefully. This includes dealing with network errors, API rate limits, and invalid responses.

```python
import requests
from requests.exceptions import RequestException

def make_api_request(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except ValueError as val_err:  # Includes JSONDecodeError
        print(f"Value error occurred: {val_err}")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")
    return None

# Example usage
api_url = "https://api.github.com/repos/python/cpython"
data = make_api_request(api_url)

if data:
    print(f"Repository: {data['full_name']}")
    print(f"Stars: {data['stargazers_count']}")
else:
    print("Failed to retrieve data")

# Output (if successful):
# Repository: python/cpython
# Stars: 51234
```

Slide 13: Real-Life Example: Weather Forecast Application

Let's create a simple weather forecast application that combines regular expressions and API usage:

```python
import re
import requests

def get_weather(city):
    api_key = "YOUR_API_KEY"  # Replace with your OpenWeatherMap API key
    api_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        temperature = data['main']['temp']
        description = data['weather'][0]['description']
        humidity = data['main']['humidity']

        return f"Weather in {city}: {temperature}°C, {description}, Humidity: {humidity}%"
    except requests.RequestException as e:
        return f"Error fetching weather data: {e}"

def parse_user_input(user_input):
    # Use regex to extract city name from user input
    match = re.search(r'weather (?:in|for) (.+)', user_input, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

# Main application loop
while True:
    user_input = input("Enter a command (e.g., 'weather in London' or 'quit'): ")
    
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break

    city = parse_user_input(user_input)
    if city:
        print(get_weather(city))
    else:
        print("Invalid command. Please use the format 'weather in [city]'")

# Example Output:
# Enter a command (e.g., 'weather in London' or 'quit'): weather in Tokyo
# Weather in Tokyo: 18.5°C, scattered clouds, Humidity: 72%
# Enter a command (e.g., 'weather in London' or 'quit'): weather for New York
# Weather in New York: 22.3°C, clear sky, Humidity: 45%
# Enter a command (e.g., 'weather in London' or 'quit'): quit
# Goodbye!
```

Slide 14: Additional Resources

For further learning about Regular Expressions and APIs in Python, consider exploring these resources:

1. Python's official documentation on the `re` module: [https://docs.python.org/3/library/re.html](https://docs.python.org/3/library/re.html)
2. "Mastering Regular Expressions" by Jeffrey Friedl (O'Reilly Media)
3. Python Requests library documentation: [https://docs.python-requests.org/](https://docs.python-requests.org/)
4. "RESTful Web APIs" by Leonard Richardson, Mike Amundsen, and Sam Ruby (O'Reilly Media)
5. ArXiv paper on API design principles: "RESTful API Design - Resource Modeling" by Cesare Pautasso [https://arxiv.org/abs/1402.1488](https://arxiv.org/abs/1402.1488)

Remember to practice regularly and experiment with different APIs

