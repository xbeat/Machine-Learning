## Explaining APIs with Python
Slide 1: What is an API?

An API, or Application Programming Interface, is a set of rules and protocols that allow different software applications to communicate with each other. It defines the methods and data structures that developers can use to interact with a particular software component or service.

```python
# Simple example of an API function
def get_user_info(user_id):
    # In a real API, this would fetch data from a database or service
    user_data = {
        "id": user_id,
        "name": "John Doe",
        "email": "john.doe@example.com"
    }
    return user_data

# Using the API function
user = get_user_info(12345)
print(f"User name: {user['name']}")
print(f"User email: {user['email']}")
```

Slide 2: API Architecture

APIs typically follow a client-server architecture. The client sends requests to the server, and the server responds with the requested data or performs the requested action. This architecture allows for separation of concerns and enables different applications to interact seamlessly.

```python
import json

class APIServer:
    def handle_request(self, endpoint, method, data=None):
        if endpoint == "/users" and method == "GET":
            return json.dumps({"users": ["Alice", "Bob", "Charlie"]})
        elif endpoint == "/users" and method == "POST":
            return json.dumps({"message": "User created successfully"})
        else:
            return json.dumps({"error": "Not found"}), 404

server = APIServer()
response = server.handle_request("/users", "GET")
print(response)
```

Slide 3: Types of APIs

There are several types of APIs, including Web APIs (REST, GraphQL), Library APIs, and Operating System APIs. Web APIs are the most common for internet-based services, while Library APIs are used for software components within a programming language.

```python
# Example of a Library API (Python's built-in random module)
import random

# Generate a random number between 1 and 10
random_number = random.randint(1, 10)
print(f"Random number: {random_number}")

# Shuffle a list
my_list = [1, 2, 3, 4, 5]
random.shuffle(my_list)
print(f"Shuffled list: {my_list}")
```

Slide 4: RESTful APIs

REST (Representational State Transfer) is a popular architectural style for designing networked applications. RESTful APIs use HTTP methods to perform operations on resources, typically represented as URLs.

```python
import requests

# Simulating a RESTful API request
def get_user(user_id):
    url = f"https://api.example.com/users/{user_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "User not found"}

# Using the simulated API
user_data = get_user(123)
print(user_data)
```

Slide 5: API Authentication

Most APIs require authentication to ensure secure access. Common methods include API keys, OAuth tokens, and JSON Web Tokens (JWT). Authentication helps protect sensitive data and control access to API resources.

```python
import hmac
import hashlib

def generate_api_signature(api_key, secret_key, payload):
    message = f"{api_key}:{payload}"
    signature = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).hexdigest()
    return signature

# Example usage
api_key = "my_api_key"
secret_key = "my_secret_key"
payload = "user_id=123&action=get_profile"

signature = generate_api_signature(api_key, secret_key, payload)
print(f"API Signature: {signature}")
```

Slide 6: API Rate Limiting

To prevent abuse and ensure fair usage, APIs often implement rate limiting. This restricts the number of requests a client can make within a specified time frame.

```python
import time

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []

    def is_allowed(self):
        current_time = time.time()
        self.requests = [req for req in self.requests if current_time - req < self.time_window]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(current_time)
            return True
        return False

# Example usage
limiter = RateLimiter(max_requests=5, time_window=60)  # 5 requests per minute

for i in range(10):
    if limiter.is_allowed():
        print(f"Request {i+1} allowed")
    else:
        print(f"Request {i+1} blocked (rate limit exceeded)")
    time.sleep(10)  # Wait 10 seconds between requests
```

Slide 7: API Versioning

API versioning allows developers to make changes to an API without breaking existing client applications. It ensures backward compatibility and smooth transitions between different API versions.

```python
class APIv1:
    def get_user(self, user_id):
        return {"id": user_id, "name": "John Doe"}

class APIv2:
    def get_user(self, user_id):
        return {"id": user_id, "name": "John Doe", "email": "john.doe@example.com"}

def api_factory(version):
    if version == "v1":
        return APIv1()
    elif version == "v2":
        return APIv2()
    else:
        raise ValueError("Invalid API version")

# Using different API versions
api_v1 = api_factory("v1")
api_v2 = api_factory("v2")

print("API v1:", api_v1.get_user(123))
print("API v2:", api_v2.get_user(123))
```

Slide 8: API Documentation

Good API documentation is crucial for developers to understand how to use an API effectively. It typically includes endpoint descriptions, request/response formats, and usage examples.

```python
def generate_api_docs(api_functions):
    docs = []
    for func in api_functions:
        doc = f"Function: {func.__name__}\n"
        doc += f"Description: {func.__doc__}\n"
        doc += f"Parameters: {func.__annotations__}\n"
        docs.append(doc)
    return "\n".join(docs)

def get_user(user_id: int) -> dict:
    """Retrieve user information by user ID."""
    pass

def create_user(name: str, email: str) -> dict:
    """Create a new user with the given name and email."""
    pass

api_functions = [get_user, create_user]
api_documentation = generate_api_docs(api_functions)
print(api_documentation)
```

Slide 9: API Testing

Testing is essential to ensure API reliability and correctness. Unit tests, integration tests, and end-to-end tests help validate API functionality and catch potential issues.

```python
import unittest

class APITestCase(unittest.TestCase):
    def setUp(self):
        self.api = API()

    def test_get_user(self):
        user = self.api.get_user(123)
        self.assertEqual(user['id'], 123)
        self.assertIn('name', user)

    def test_create_user(self):
        new_user = self.api.create_user("Alice", "alice@example.com")
        self.assertIn('id', new_user)
        self.assertEqual(new_user['name'], "Alice")
        self.assertEqual(new_user['email'], "alice@example.com")

if __name__ == '__main__':
    unittest.main()
```

Slide 10: API Security

Ensuring API security is crucial to protect sensitive data and prevent unauthorized access. Common security measures include HTTPS encryption, input validation, and proper error handling.

```python
import re

def validate_input(data):
    # Example input validation for a user registration API
    if not re.match(r'^[a-zA-Z0-9_]{3,20}$', data['username']):
        raise ValueError("Invalid username format")
    
    if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', data['email']):
        raise ValueError("Invalid email format")
    
    if len(data['password']) < 8:
        raise ValueError("Password must be at least 8 characters long")

# Example usage
try:
    user_data = {
        'username': 'john_doe',
        'email': 'john.doe@example.com',
        'password': 'securepass123'
    }
    validate_input(user_data)
    print("Input validation passed")
except ValueError as e:
    print(f"Input validation failed: {str(e)}")
```

Slide 11: Real-life Example: Weather API

A weather API allows developers to retrieve weather information for different locations. This example demonstrates how to fetch and display weather data for a given city.

```python
import json
from urllib import request, parse

def get_weather(city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    api_key = "YOUR_API_KEY"  # Replace with a real API key
    
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }
    
    url = f"{base_url}?{parse.urlencode(params)}"
    
    with request.urlopen(url) as response:
        data = json.loads(response.read().decode())
        
    return {
        "city": data["name"],
        "temperature": data["main"]["temp"],
        "description": data["weather"][0]["description"]
    }

# Example usage
city = "London"
weather = get_weather(city)
print(f"Weather in {weather['city']}:")
print(f"Temperature: {weather['temperature']}°C")
print(f"Description: {weather['description']}")
```

Slide 12: Real-life Example: Social Media API

Social media APIs allow developers to integrate social media features into their applications. This example demonstrates how to post a tweet using a simulated Twitter API.

```python
import json
import random

class TwitterAPI:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret

    def post_tweet(self, message):
        # Simulate posting a tweet
        tweet_id = random.randint(1000000000, 9999999999)
        response = {
            "id": tweet_id,
            "text": message,
            "created_at": "2023-04-15T12:00:00Z"
        }
        return json.dumps(response)

# Example usage
api = TwitterAPI("your_api_key", "your_api_secret")
tweet = "Hello, world! This is my first tweet using the Twitter API."
result = api.post_tweet(tweet)
print("Tweet posted successfully:")
print(result)
```

Slide 13: Why APIs are Popular

APIs have become increasingly popular due to their ability to facilitate integration between different systems, promote code reuse, and enable the development of complex applications by combining existing services. They allow developers to leverage the functionality of other applications without reinventing the wheel.

```python
# Demonstration of how APIs enable integration and code reuse

def weather_api(city):
    # Simulated weather API
    return f"The temperature in {city} is 25°C"

def translation_api(text, target_language):
    # Simulated translation API
    return f"Translated '{text}' to {target_language}"

def travel_app(city, language):
    weather_info = weather_api(city)
    translated_weather = translation_api(weather_info, language)
    return translated_weather

# Using multiple APIs to create a new application
result = travel_app("Paris", "French")
print(result)
```

Slide 14: Additional Resources

For more information on APIs and their implementation, consider exploring the following resources:

1.  "RESTful Web APIs" by Leonard Richardson, Mike Amundsen, and Sam Ruby
2.  "Designing Web APIs" by Brenda Jin, Saurabh Sahni, and Amir Shevat
3.  Python Requests library documentation: [https://docs.python-requests.org/](https://docs.python-requests.org/)
4.  OpenAPI Specification: [https://swagger.io/specification/](https://swagger.io/specification/)
5.  ArXiv paper on API design principles: [https://arxiv.org/abs/2001.09563](https://arxiv.org/abs/2001.09563)

These resources provide in-depth knowledge on API design, implementation, and best practices.

