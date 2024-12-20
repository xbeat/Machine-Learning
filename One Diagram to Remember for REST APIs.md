## One Diagram to Remember for REST APIs

Slide 1: Understanding REST APIs

REST (Representational State Transfer) is an architectural style for designing networked applications. It utilizes simple HTTP protocols to facilitate communication between clients and servers. REST APIs have become essential in modern web development due to their simplicity, scalability, and stateless nature.

```python
import http.client

def make_rest_request(method, endpoint, headers=None, body=None):
    conn = http.client.HTTPSConnection("api.example.com")
    conn.request(method, endpoint, body, headers)
    response = conn.getresponse()
    return response.read().decode()

# Example GET request
result = make_rest_request("GET", "/users/1")
print(result)
```

Slide 2: REST API Principles

REST APIs adhere to several key principles that define their behavior and structure. These principles include statelessness, client-server architecture, and cacheability. Statelessness means that each request from a client to a server must contain all the information needed to understand and process the request. The client-server architecture separates concerns between the client and server, allowing for better scalability and maintainability.

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class SimpleRESTHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = {"message": "This is a stateless response"}
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_error(404)

httpd = HTTPServer(('localhost', 8000), SimpleRESTHandler)
print("Server running on http://localhost:8000")
httpd.serve_forever()
```

Slide 3: HTTP Methods in REST APIs

REST APIs primarily use four HTTP methods to perform CRUD (Create, Read, Update, Delete) operations on resources. These methods are GET (retrieve data), POST (submit data), PUT (update data), and DELETE (remove data). Each method serves a specific purpose and follows certain conventions in REST architecture.

```python
import requests

base_url = "https://api.example.com/users"

# GET request to retrieve user data
response = requests.get(f"{base_url}/1")
print("GET:", response.json())

# POST request to create a new user
new_user = {"name": "John Doe", "email": "john@example.com"}
response = requests.post(base_url, json=new_user)
print("POST:", response.json())

# PUT request to update user data
updated_user = {"name": "Jane Doe", "email": "jane@example.com"}
response = requests.put(f"{base_url}/1", json=updated_user)
print("PUT:", response.json())

# DELETE request to remove a user
response = requests.delete(f"{base_url}/1")
print("DELETE:", response.status_code)
```

Slide 4: API Versioning

API versioning is crucial for maintaining backward compatibility and ensuring smooth transitions between different iterations of an API. Common versioning methods include URL versioning and header versioning. Proper versioning allows developers to introduce new features or make changes without breaking existing client applications.

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/v1/users', methods=['GET'])
def get_users_v1():
    return jsonify({"version": "1.0", "users": ["Alice", "Bob"]})

@app.route('/api/v2/users', methods=['GET'])
def get_users_v2():
    return jsonify({"version": "2.0", "users": [{"name": "Alice", "id": 1}, {"name": "Bob", "id": 2}]})

@app.route('/api/users', methods=['GET'])
def get_users():
    version = request.headers.get('API-Version', '1.0')
    if version == '1.0':
        return get_users_v1()
    elif version == '2.0':
        return get_users_v2()
    else:
        return jsonify({"error": "Unsupported API version"}), 400

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 5: REST API Security

Security is a critical aspect of REST API design. Common security measures include authentication, authorization, and encryption. These measures help protect sensitive data and ensure that only authorized users can access certain resources or perform specific actions.

```python
from flask import Flask, request, jsonify
import jwt
import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

def token_required(f):
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/login', methods=['POST'])
def login():
    auth = request.authorization
    if auth and auth.username == 'admin' and auth.password == 'password':
        token = jwt.encode({'user': auth.username, 'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)},
                           app.config['SECRET_KEY'], algorithm="HS256")
        return jsonify({'token': token})
    return jsonify({'message': 'Could not verify!'}), 401

@app.route('/protected', methods=['GET'])
@token_required
def protected():
    return jsonify({'message': 'This is a protected route!'})

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 6: RESTful Resource Naming

Proper resource naming is crucial for creating intuitive and easy-to-use REST APIs. Resource names should be nouns, not verbs, and should represent the entities being manipulated. Use plural nouns for collections and singular nouns for specific resources. Hierarchical relationships can be represented using nested resources.

```python
from flask import Flask, jsonify

app = Flask(__name__)

# Good resource naming
@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify({"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]})

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    return jsonify({"id": user_id, "name": "Alice"})

@app.route('/api/users/<int:user_id>/posts', methods=['GET'])
def get_user_posts(user_id):
    return jsonify({"posts": [{"id": 1, "title": "First Post"}, {"id": 2, "title": "Second Post"}]})

# Bad resource naming (avoid verbs in URLs)
@app.route('/api/getUsers', methods=['GET'])
def get_users_bad():
    return jsonify({"error": "This is not a RESTful naming convention"})

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 7: HTTP Status Codes in REST APIs

HTTP status codes are an integral part of REST APIs, providing standardized responses to client requests. These codes help clients understand the outcome of their requests and handle different scenarios appropriately. Status codes are grouped into five classes: informational (1xx), successful (2xx), redirection (3xx), client error (4xx), and server error (5xx).

```python
from flask import Flask, jsonify, request

app = Flask(__name__)

users = {}

@app.route('/api/users', methods=['POST'])
def create_user():
    data = request.json
    if 'name' not in data:
        return jsonify({"error": "Name is required"}), 400
    user_id = len(users) + 1
    users[user_id] = data['name']
    return jsonify({"id": user_id, "name": data['name']}), 201

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    if user_id not in users:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"id": user_id, "name": users[user_id]}), 200

@app.route('/api/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    if user_id not in users:
        return jsonify({"error": "User not found"}), 404
    data = request.json
    if 'name' not in data:
        return jsonify({"error": "Name is required"}), 400
    users[user_id] = data['name']
    return jsonify({"id": user_id, "name": users[user_id]}), 200

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    if user_id not in users:
        return jsonify({"error": "User not found"}), 404
    del users[user_id]
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 8: Content Negotiation in REST APIs

Content negotiation is a mechanism in REST APIs that allows clients to request specific representations of resources. This feature enables APIs to serve different formats (e.g., JSON, XML) or versions of the same resource based on client preferences. Content negotiation is typically implemented using HTTP headers such as Accept and Content-Type.

```python
from flask import Flask, jsonify, request
import xml.etree.ElementTree as ET

app = Flask(__name__)

user = {"id": 1, "name": "Alice", "email": "alice@example.com"}

def user_to_xml(user):
    root = ET.Element("user")
    for key, value in user.items():
        ET.SubElement(root, key).text = str(value)
    return ET.tostring(root, encoding='unicode')

@app.route('/api/user', methods=['GET'])
def get_user():
    accept_header = request.headers.get('Accept', '')
    
    if 'application/json' in accept_header:
        return jsonify(user), 200, {'Content-Type': 'application/json'}
    elif 'application/xml' in accept_header:
        xml_data = user_to_xml(user)
        return xml_data, 200, {'Content-Type': 'application/xml'}
    else:
        return jsonify({"error": "Unsupported media type"}), 406

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 9: Pagination in REST APIs

Pagination is a crucial feature for REST APIs that handle large datasets. It allows clients to request a specific subset of data, improving performance and reducing bandwidth usage. Common pagination strategies include offset-based and cursor-based pagination. Implementing pagination helps in efficiently managing and presenting large collections of resources.

```python
from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data
items = [f"Item {i}" for i in range(1, 101)]

@app.route('/api/items', methods=['GET'])
def get_items():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    start = (page - 1) * per_page
    end = start + per_page
    
    paginated_items = items[start:end]
    
    response = {
        "items": paginated_items,
        "page": page,
        "per_page": per_page,
        "total_items": len(items),
        "total_pages": -(-len(items) // per_page)  # Ceiling division
    }
    
    if page < response["total_pages"]:
        response["next_page"] = f"/api/items?page={page+1}&per_page={per_page}"
    if page > 1:
        response["prev_page"] = f"/api/items?page={page-1}&per_page={per_page}"
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 10: HATEOAS in REST APIs

HATEOAS (Hypermedia as the Engine of Application State) is a constraint of the REST application architecture. It keeps the REST style architecture unique from most other network application architectures. The principle is that a client interacts with a network application whose application servers provide information dynamically through hypermedia. A REST client needs no prior knowledge about how to interact with an application or server beyond a generic understanding of hypermedia.

```python
from flask import Flask, jsonify, request, url_for

app = Flask(__name__)

users = {
    1: {"id": 1, "name": "Alice", "email": "alice@example.com"},
    2: {"id": 2, "name": "Bob", "email": "bob@example.com"}
}

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = users.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    response = {
        "user": user,
        "_links": {
            "self": {"href": url_for('get_user', user_id=user_id, _external=True)},
            "collection": {"href": url_for('get_users', _external=True)},
            "update": {"href": url_for('update_user', user_id=user_id, _external=True), "method": "PUT"},
            "delete": {"href": url_for('delete_user', user_id=user_id, _external=True), "method": "DELETE"}
        }
    }
    return jsonify(response)

@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify({"users": list(users.values())})

@app.route('/api/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    return jsonify({"message": "User updated"})

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    return jsonify({"message": "User deleted"})

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 11: Real-life Example: Weather API

A weather API is a common real-world application of REST principles. This example demonstrates how to create a simple weather API that provides current weather conditions for a given city. The API uses mock data for simplicity, but in a real-world scenario, it would integrate with a weather data provider.

```python
from flask import Flask, jsonify, request
import random

app = Flask(__name__)

weather_conditions = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy"]

def get_mock_weather(city):
    return {
        "city": city,
        "temperature": round(random.uniform(0, 35), 1),
        "condition": random.choice(weather_conditions),
        "humidity": random.randint(30, 90),
        "wind_speed": round(random.uniform(0, 20), 1)
    }

@app.route('/api/weather', methods=['GET'])
def get_weather():
    city = request.args.get('city', 'London')
    weather_data = get_mock_weather(city)
    
    response = {
        "data": weather_data,
        "_links": {
            "self": {"href": f"/api/weather?city={city}"},
            "forecast": {"href": f"/api/weather/forecast?city={city}"}
        }
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 12: Real-life Example: Library Management API

A library management system is another practical application of REST APIs. This example showcases a simple API for managing books in a library, including endpoints for retrieving, adding, updating, and deleting books.

```python
from flask import Flask, jsonify, request

app = Flask(__name__)

books = {
    1: {"id": 1, "title": "To Kill a Mockingbird", "author": "Harper Lee", "available": True},
    2: {"id": 2, "title": "1984", "author": "George Orwell", "available": False}
}

@app.route('/api/books', methods=['GET'])
def get_books():
    return jsonify({"books": list(books.values())})

@app.route('/api/books/<int:book_id>', methods=['GET'])
def get_book(book_id):
    book = books.get(book_id)
    if not book:
        return jsonify({"error": "Book not found"}), 404
    return jsonify(book)

@app.route('/api/books', methods=['POST'])
def add_book():
    new_book = request.json
    new_id = max(books.keys()) + 1
    new_book['id'] = new_id
    books[new_id] = new_book
    return jsonify(new_book), 201

@app.route('/api/books/<int:book_id>', methods=['PUT'])
def update_book(book_id):
    if book_id not in books:
        return jsonify({"error": "Book not found"}), 404
    books[book_id].update(request.json)
    return jsonify(books[book_id])

@app.route('/api/books/<int:book_id>', methods=['DELETE'])
def delete_book(book_id):
    if book_id not in books:
        return jsonify({"error": "Book not found"}), 404
    del books[book_id]
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 13: Best Practices for REST API Design

Designing effective REST APIs involves following best practices to ensure consistency, scalability, and ease of use. Some key best practices include using appropriate HTTP methods, implementing proper error handling, versioning your API, and providing clear documentation.

```python
from flask import Flask, jsonify, request
from functools import wraps

app = Flask(__name__)

def validate_json(f):
    @wraps(f)
    def wrapper(*args, **kw):
        try:
            request.json
        except:
            return jsonify({"error": "Invalid JSON"}), 400
        return f(*args, **kw)
    return wrapper

@app.route('/api/v1/users', methods=['POST'])
@validate_json
def create_user():
    data = request.json
    # Process user creation
    return jsonify({"message": "User created successfully"}), 201

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 14: API Documentation with Swagger/OpenAPI

Proper documentation is crucial for the adoption and usability of your REST API. Swagger (now part of the OpenAPI Initiative) is a popular tool for describing and documenting RESTful APIs. This example demonstrates how to integrate Swagger documentation into a Flask application using the flask-swagger-ui extension.

```python
from flask import Flask, jsonify
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)

SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Sample API"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/api/hello', methods=['GET'])
def hello():
    """
    This is a sample API endpoint
    ---
    responses:
      200:
        description: Successful response
    """
    return jsonify({"message": "Hello, World!"})

if __name__ == '__main__':
    app.run(debug=True)

# Note: You would need to create a swagger.json file in the static folder
# to define your API specifications.
```

Slide 15: Additional Resources

For further exploration of REST API concepts and best practices, consider the following resources:

1.  RESTful Web Services Cookbook by Subbu Allamaraju (O'Reilly Media)
2.  "Architectural Styles and the Design of Network-based Software Architectures" by Roy Fielding (Ph.D. dissertation)
3.  "REST API Design Rulebook" by Mark Masse (O'Reilly Media)
4.  Flask documentation: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
5.  OpenAPI Specification: [https://swagger.io/specification/](https://swagger.io/specification/)

These resources provide in-depth information on REST API design, implementation, and best practices. Remember to always refer to the official documentation of the tools and frameworks you're using for the most up-to-date information.

