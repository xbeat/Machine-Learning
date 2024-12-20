## Secure API Development with Python:
Slide 1: API Security Basics

API security is crucial for protecting sensitive data and ensuring the integrity of web services. It involves implementing measures to authenticate users, encrypt data, and prevent unauthorized access.

```python
import requests
from requests.auth import HTTPBasicAuth

def secure_api_request(url, username, password):
    response = requests.get(url, auth=HTTPBasicAuth(username, password))
    return response.json()

# Example usage
api_url = "https://api.example.com/data"
result = secure_api_request(api_url, "user123", "password123")
print(result)
```

Slide 2: Authentication with API Keys

API keys are a common method for authenticating requests to an API. They provide a simple way to identify and authorize clients.

```python
import requests

def api_request_with_key(url, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    return response.json()

# Example usage
api_url = "https://api.example.com/data"
api_key = "your_api_key_here"
result = api_request_with_key(api_url, api_key)
print(result)
```

Slide 3: Input Validation

Proper input validation is essential to prevent injection attacks and ensure data integrity. Always validate and sanitize user input before processing it.

```python
import re

def validate_input(user_input):
    # Remove any non-alphanumeric characters
    sanitized_input = re.sub(r'[^a-zA-Z0-9]', '', user_input)
    
    # Check if the input meets certain criteria
    if len(sanitized_input) < 5 or len(sanitized_input) > 20:
        return False
    
    return True

# Example usage
user_input = "user@123"
if validate_input(user_input):
    print("Input is valid")
else:
    print("Input is invalid")
```

Slide 4: Rate Limiting

Rate limiting helps prevent abuse and ensures fair usage of your API. It restricts the number of requests a client can make within a given time frame.

```python
from flask import Flask, request
from flask_limiter import Limiter

app = Flask(__name__)
limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route("/api/data")
@limiter.limit("5 per minute")
def api_data():
    return "API response data"

if __name__ == "__main__":
    app.run()
```

Slide 5: HTTPS and SSL/TLS

Always use HTTPS to encrypt data in transit. This prevents eavesdropping and man-in-the-middle attacks.

```python
import ssl
import socket

def check_ssl_certificate(hostname, port=443):
    context = ssl.create_default_context()
    with socket.create_connection((hostname, port)) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as secure_sock:
            cert = secure_sock.getpeercert()
            return cert

# Example usage
hostname = "www.example.com"
cert_info = check_ssl_certificate(hostname)
print(f"SSL certificate for {hostname}:")
print(cert_info)
```

Slide 6: JSON Web Tokens (JWT)

JWTs provide a secure way to transmit information between parties as a JSON object. They are commonly used for authentication and information exchange.

```python
import jwt
from datetime import datetime, timedelta

def create_jwt(user_id, secret_key):
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    token = jwt.encode(payload, secret_key, algorithm="HS256")
    return token

def verify_jwt(token, secret_key):
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return "Token has expired"
    except jwt.InvalidTokenError:
        return "Invalid token"

# Example usage
secret_key = "your_secret_key"
user_id = 12345
token = create_jwt(user_id, secret_key)
print(f"Generated JWT: {token}")

verified_payload = verify_jwt(token, secret_key)
print(f"Verified payload: {verified_payload}")
```

Slide 7: Cross-Origin Resource Sharing (CORS)

CORS is a security mechanism that allows or restricts resource requests from one domain to another. It helps prevent unauthorized access to your API from malicious websites.

```python
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://trusted-domain.com"}})

@app.route("/api/data")
def api_data():
    return "API response data"

if __name__ == "__main__":
    app.run()
```

Slide 8: API Versioning

Versioning your API allows you to make changes without breaking existing client integrations. It provides a smooth transition for users when updating your API.

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/api/v1/users")
def api_v1_users():
    return jsonify({"version": "1.0", "users": ["Alice", "Bob"]})

@app.route("/api/v2/users")
def api_v2_users():
    return jsonify({"version": "2.0", "users": [{"name": "Alice", "id": 1}, {"name": "Bob", "id": 2}]})

if __name__ == "__main__":
    app.run()
```

Slide 9: Logging and Monitoring

Proper logging and monitoring are essential for detecting and responding to security incidents. They help you track API usage and identify potential threats.

```python
import logging
from flask import Flask, request

app = Flask(__name__)
logging.basicConfig(filename='api.log', level=logging.INFO)

@app.route("/api/data")
def api_data():
    logging.info(f"API request from {request.remote_addr}")
    return "API response data"

if __name__ == "__main__":
    app.run()
```

Slide 10: API Documentation

Well-documented APIs are crucial for developers to understand how to use your API securely. Tools like Swagger can help generate interactive API documentation.

```python
from flask import Flask
from flask_restx import Api, Resource

app = Flask(__name__)
api = Api(app, version='1.0', title='Sample API', description='A sample API')

@api.route('/hello')
class HelloWorld(Resource):
    @api.doc(responses={200: 'Success', 400: 'Validation Error'})
    def get(self):
        """Returns a greeting message"""
        return {'message': 'Hello, World!'}

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 11: Real-Life Example: Weather API

A weather API that provides temperature data for different cities. This example demonstrates authentication, input validation, and rate limiting.

```python
from flask import Flask, request, jsonify
from flask_limiter import Limiter
import re

app = Flask(__name__)
limiter = Limiter(app, key_func=lambda: request.headers.get('X-API-Key'))

# Simulated weather data
weather_data = {
    "new york": 20,
    "london": 15,
    "tokyo": 25
}

def validate_city(city):
    return re.match(r'^[a-zA-Z\s]+$', city) is not None

@app.route("/api/weather")
@limiter.limit("5 per minute")
def get_weather():
    api_key = request.headers.get('X-API-Key')
    if api_key != 'valid_api_key':
        return jsonify({"error": "Invalid API key"}), 401

    city = request.args.get('city', '').lower()
    if not validate_city(city):
        return jsonify({"error": "Invalid city name"}), 400

    temperature = weather_data.get(city)
    if temperature is None:
        return jsonify({"error": "City not found"}), 404

    return jsonify({"city": city, "temperature": temperature})

if __name__ == "__main__":
    app.run(debug=True)
```

Slide 12: Real-Life Example: URL Shortener API

A URL shortener API that demonstrates input validation, rate limiting, and logging.

```python
from flask import Flask, request, jsonify, redirect
from flask_limiter import Limiter
import logging
import shortuuid

app = Flask(__name__)
limiter = Limiter(app, key_func=lambda: request.remote_addr)
logging.basicConfig(filename='url_shortener.log', level=logging.INFO)

url_database = {}

def validate_url(url):
    return url.startswith('http://') or url.startswith('https://')

@app.route("/api/shorten", methods=['POST'])
@limiter.limit("10 per minute")
def shorten_url():
    long_url = request.json.get('url')
    if not long_url or not validate_url(long_url):
        return jsonify({"error": "Invalid URL"}), 400

    short_code = shortuuid.uuid()[:8]
    url_database[short_code] = long_url
    short_url = f"http://short.url/{short_code}"
    
    logging.info(f"URL shortened: {long_url} -> {short_url}")
    return jsonify({"short_url": short_url})

@app.route("/<short_code>")
def redirect_url(short_code):
    long_url = url_database.get(short_code)
    if long_url:
        logging.info(f"Redirect: {short_code} -> {long_url}")
        return redirect(long_url)
    return "URL not found", 404

if __name__ == "__main__":
    app.run(debug=True)
```

Slide 13: Continuous Security Testing

Regular security testing helps identify vulnerabilities in your API. Automated tools and manual penetration testing can be used to assess your API's security posture.

```python
import requests
import concurrent.futures

def test_endpoint(url, method='GET', data=None):
    try:
        if method == 'GET':
            response = requests.get(url)
        elif method == 'POST':
            response = requests.post(url, json=data)
        else:
            return f"Unsupported method: {method}"

        return f"{url}: {response.status_code}"
    except requests.RequestException as e:
        return f"{url}: Error - {str(e)}"

def run_security_tests(base_url, endpoints):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for endpoint in endpoints:
            url = f"{base_url}{endpoint['path']}"
            futures.append(executor.submit(test_endpoint, url, endpoint['method'], endpoint.get('data')))

        for future in concurrent.futures.as_completed(futures):
            print(future.result())

# Example usage
base_url = "https://api.example.com"
endpoints = [
    {"path": "/users", "method": "GET"},
    {"path": "/login", "method": "POST", "data": {"username": "test", "password": "test"}},
    {"path": "/data", "method": "GET"},
]

run_security_tests(base_url, endpoints)
```

Slide 14: Additional Resources

For more information on API security best practices and advanced topics, consider exploring the following resources:

1. OWASP API Security Top 10: [https://owasp.org/www-project-api-security/](https://owasp.org/www-project-api-security/)
2. "REST API Design Rulebook" by Mark Masse
3. "API Security in Action" by Neil Madden
4. Python requests library documentation: [https://docs.python-requests.org/](https://docs.python-requests.org/)
5. Flask-RESTX documentation: [https://flask-restx.readthedocs.io/](https://flask-restx.readthedocs.io/)

Remember to stay updated with the latest security trends and best practices to ensure the ongoing protection of your APIs.

