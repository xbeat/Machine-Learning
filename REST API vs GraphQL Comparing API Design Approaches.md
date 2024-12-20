## REST API vs GraphQL Comparing API Design Approaches
Slide 1: Basic REST API Implementation

A foundational implementation of a REST API using Python's Flask framework, demonstrating core HTTP methods for CRUD operations. This example creates a simple API for managing a collection of items with proper request handling and response formatting.

```python
from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory storage for demonstration
items = {}
counter = 1

@app.route('/api/items', methods=['GET'])
def get_items():
    return jsonify(items)

@app.route('/api/items', methods=['POST'])
def create_item():
    global counter
    data = request.get_json()
    item_id = str(counter)
    items[item_id] = data
    counter += 1
    return jsonify({'id': item_id, 'item': data}), 201

@app.route('/api/items/<item_id>', methods=['PUT'])
def update_item(item_id):
    if item_id not in items:
        return jsonify({'error': 'Item not found'}), 404
    data = request.get_json()
    items[item_id] = data
    return jsonify({'id': item_id, 'item': data})

@app.route('/api/items/<item_id>', methods=['DELETE'])
def delete_item(item_id):
    if item_id not in items:
        return jsonify({'error': 'Item not found'}), 404
    del items[item_id]
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)

# Example Usage:
# curl -X POST -H "Content-Type: application/json" -d '{"name":"laptop"}' http://localhost:5000/api/items
# curl http://localhost:5000/api/items
```

Slide 2: GraphQL API Implementation

Implementing a GraphQL API using Python's Graphene library to demonstrate schema definition, query resolution, and mutation handling. This implementation provides a more flexible querying mechanism compared to REST.

```python
import graphene
from graphene import ObjectType, String, Int, List, Mutation

# Sample data store
books = []

class Book(ObjectType):
    id = Int()
    title = String()
    author = String()

class Query(ObjectType):
    books = List(Book)
    book = graphene.Field(Book, id=Int())

    def resolve_books(self, info):
        return books

    def resolve_book(self, info, id):
        return next((book for book in books if book.id == id), None)

class AddBook(Mutation):
    class Arguments:
        title = String(required=True)
        author = String(required=True)

    book = graphene.Field(Book)

    def mutate(self, info, title, author):
        book = Book(
            id=len(books) + 1,
            title=title,
            author=author
        )
        books.append(book)
        return AddBook(book=book)

class Mutations(ObjectType):
    add_book = AddBook.Field()

schema = graphene.Schema(query=Query, mutation=Mutations)

# Example Query:
query = '''
    query {
        books {
            id
            title
            author
        }
    }
'''
```

Slide 3: REST Authentication Implementation

Implementation of token-based authentication for REST APIs using JWT (JSON Web Tokens). This secure authentication system provides stateless authorization for API endpoints with token generation and validation.

```python
from flask import Flask, jsonify, request
import jwt
from functools import wraps
from datetime import datetime, timedelta

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        except:
            return jsonify({'message': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/login', methods=['POST'])
def login():
    auth = request.authorization
    if auth and auth.username == "admin" and auth.password == "password":
        token = jwt.encode({
            'user': auth.username,
            'exp': datetime.utcnow() + timedelta(minutes=30)
        }, app.config['SECRET_KEY'])
        return jsonify({'token': token})
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/protected', methods=['GET'])
@token_required
def protected():
    return jsonify({'message': 'This is a protected endpoint'})

if __name__ == '__main__':
    app.run(debug=True)

# Example Usage:
# curl -u admin:password http://localhost:5000/login
# curl -H "Authorization: <token>" http://localhost:5000/protected
```

Slide 4: GraphQL Authentication and Security

A comprehensive implementation of authentication and authorization in GraphQL using context-based security. This approach demonstrates how to protect GraphQL queries and mutations while maintaining schema flexibility.

```python
import graphene
from graphene import ObjectType, String, Boolean
from functools import wraps
import jwt

# Authentication decorator for resolvers
def login_required(fn):
    @wraps(fn)
    def wrapper(self, info, *args, **kwargs):
        context = info.context
        if not context.get('is_authenticated'):
            raise Exception('Authentication required')
        return fn(self, info, *args, **kwargs)
    return wrapper

class User(ObjectType):
    username = String()
    is_authenticated = Boolean()

class AuthMutation(graphene.Mutation):
    class Arguments:
        username = String(required=True)
        password = String(required=True)

    token = String()
    user = graphene.Field(User)

    def mutate(self, info, username, password):
        if username == "admin" and password == "secret":
            token = jwt.encode(
                {'username': username},
                'secret_key',
                algorithm='HS256'
            )
            user = User(username=username, is_authenticated=True)
            return AuthMutation(token=token, user=user)
        raise Exception('Invalid credentials')

class Query(ObjectType):
    me = graphene.Field(User)

    @login_required
    def resolve_me(self, info):
        return User(
            username=info.context.get('username'),
            is_authenticated=True
        )

class Mutation(ObjectType):
    login = AuthMutation.Field()

schema = graphene.Schema(query=Query, mutation=Mutation)

# Example usage:
"""
mutation {
    login(username: "admin", password: "secret") {
        token
        user {
            username
            isAuthenticated
        }
    }
}
"""
```

Slide 5: REST API Rate Limiting

Implementation of a rate limiting mechanism for REST APIs using Redis as a backend store. This system prevents API abuse by tracking and limiting the number of requests per client within specific time windows.

```python
from flask import Flask, jsonify, request
import redis
from datetime import datetime
import time

app = Flask(__name__)
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def rate_limit(key_prefix, limit=100, period=3600):
    def decorator(f):
        def rate_limited(*args, **kwargs):
            key = f"{key_prefix}:{request.remote_addr}"
            current = int(time.time())
            pipe = redis_client.pipeline()
            
            # Clean old requests
            pipe.zremrangebyscore(key, 0, current - period)
            # Add current request
            pipe.zadd(key, {str(current): current})
            # Count requests in window
            pipe.zcard(key)
            # Set expiry
            pipe.expire(key, period)
            
            results = pipe.execute()
            request_count = results[2]

            if request_count > limit:
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after': period
                }), 429

            return f(*args, **kwargs)
        return rate_limited
    return decorator

@app.route('/api/resource')
@rate_limit('getresource', limit=10, period=60)
def get_resource():
    return jsonify({'data': 'Resource data'})

if __name__ == '__main__':
    app.run(debug=True)

# Example usage:
# for i in range(12):
#     response = requests.get('http://localhost:5000/api/resource')
#     print(f"Request {i+1}: {response.status_code}")
```

Slide 6: GraphQL Query Complexity Analysis

A sophisticated implementation of query complexity analysis for GraphQL to prevent resource-intensive queries. This system calculates query cost and rejects queries that exceed defined thresholds.

```python
import graphene
from graphene import ObjectType, String, Int, List
from graphql import parse, visit

class QueryComplexityAnalyzer:
    def __init__(self, max_complexity):
        self.max_complexity = max_complexity
        self.complexity = 0
        
    def analyze_query(self, query_str):
        ast = parse(query_str)
        visit(ast, self)
        if self.complexity > self.max_complexity:
            raise Exception(
                f"Query complexity {self.complexity} exceeds limit {self.max_complexity}"
            )
        return self.complexity
    
    def enter_field(self, node, *args):
        field_complexity = 1
        if hasattr(node, 'selection_set'):
            field_complexity = len(node.selection_set.selections)
        self.complexity += field_complexity
        return node

class Item(ObjectType):
    id = Int()
    name = String()
    related_items = List(lambda: Item)

class Query(ObjectType):
    items = List(Item)
    
    def resolve_items(self, info):
        # Analyze query complexity before execution
        analyzer = QueryComplexityAnalyzer(max_complexity=10)
        query_str = info.context.get('query_str', '')
        analyzer.analyze_query(query_str)
        
        return [
            Item(id=1, name="Item 1", related_items=[]),
            Item(id=2, name="Item 2", related_items=[])
        ]

schema = graphene.Schema(query=Query)

# Example usage:
"""
{
    items {
        id
        name
        relatedItems {
            id
            name
            relatedItems {
                id
                name
            }
        }
    }
}
"""
```

Slide 7: REST API Versioning Strategies

Implementation of API versioning techniques in REST using URL paths, headers, and query parameters. This approach demonstrates how to maintain multiple API versions simultaneously while ensuring backward compatibility.

```python
from flask import Flask, jsonify, request
from functools import wraps

app = Flask(__name__)

def version_control(min_version):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # URL versioning check
            url_version = request.view_args.get('version', 'v1')
            # Header versioning check
            header_version = request.headers.get('API-Version', 'v1')
            # Query param versioning check
            query_version = request.args.get('version', 'v1')
            
            # Use any versioning strategy (here using URL version)
            version = url_version.replace('v', '')
            
            if int(version) < min_version:
                return jsonify({
                    'error': 'API version no longer supported',
                    'min_version': f'v{min_version}'
                }), 400
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/api/v1/users', methods=['GET'])
@version_control(min_version=1)
def get_users_v1():
    return jsonify({
        'users': [{'id': 1, 'name': 'User 1'}],
        'version': 'v1'
    })

@app.route('/api/v2/users', methods=['GET'])
@version_control(min_version=2)
def get_users_v2():
    return jsonify({
        'users': [{'id': 1, 'name': 'User 1', 'email': 'user1@example.com'}],
        'version': 'v2'
    })

if __name__ == '__main__':
    app.run(debug=True)

# Example Usage:
# curl http://localhost:5000/api/v1/users
# curl http://localhost:5000/api/v2/users
# curl -H "API-Version: v2" http://localhost:5000/api/users
```

Slide 8: GraphQL Schema Stitching

Implementation of GraphQL schema stitching to combine multiple GraphQL schemas into a unified API. This approach enables building a distributed GraphQL architecture with multiple services.

```python
import graphene
from graphene import ObjectType, String, Int, List
from graphql import build_ast_schema, parse

# Service A Schema
service_a_schema = """
type User {
    id: ID!
    name: String!
}

type Query {
    users: [User]
}
"""

# Service B Schema
service_b_schema = """
type Post {
    id: ID!
    title: String!
    authorId: ID!
}

type Query {
    posts: [Post]
}
"""

class User(ObjectType):
    id = String()
    name = String()
    posts = List(lambda: Post)

    def resolve_posts(self, info):
        return [post for post in posts_data if post['authorId'] == self.id]

class Post(ObjectType):
    id = String()
    title = String()
    author_id = String()
    author = graphene.Field(User)

    def resolve_author(self, info):
        return next(
            (user for user in users_data if user['id'] == self.author_id),
            None
        )

class Query(ObjectType):
    users = List(User)
    posts = List(Post)

    def resolve_users(self, info):
        return [User(**user) for user in users_data]

    def resolve_posts(self, info):
        return [Post(**post) for post in posts_data]

# Sample data
users_data = [
    {'id': '1', 'name': 'User 1'},
    {'id': '2', 'name': 'User 2'}
]

posts_data = [
    {'id': '1', 'title': 'Post 1', 'authorId': '1'},
    {'id': '2', 'title': 'Post 2', 'authorId': '1'}
]

schema = graphene.Schema(query=Query)

# Example Query:
"""
{
    users {
        id
        name
        posts {
            id
            title
        }
    }
}
"""
```

Slide 9: REST API Caching Implementation

A comprehensive implementation of caching strategies for REST APIs using Redis as a caching layer. This system demonstrates both client-side and server-side caching mechanisms with proper cache invalidation.

```python
from flask import Flask, jsonify, request, make_response
import redis
import hashlib
import json
from functools import wraps
from datetime import datetime, timedelta

app = Flask(__name__)
cache = redis.Redis(host='localhost', port=6379, db=0)

def cache_response(timeout=5 * 60):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Generate cache key from request data
            cache_key = f"{request.path}:{request.args}"
            
            # Check for cached response
            cached_response = cache.get(cache_key)
            if cached_response:
                return json.loads(cached_response)
            
            # Get fresh response
            response = f(*args, **kwargs)
            
            # Cache the response
            cache.setex(
                cache_key,
                timeout,
                json.dumps(response.get_json())
            )
            
            # Add cache headers
            response.headers['Cache-Control'] = f'max-age={timeout}'
            response.headers['ETag'] = hashlib.md5(
                json.dumps(response.get_json()).encode()
            ).hexdigest()
            
            return response
        return decorated_function
    return decorator

@app.route('/api/data/<id>')
@cache_response(timeout=300)
def get_data(id):
    # Simulate expensive operation
    data = {'id': id, 'timestamp': datetime.now().isoformat()}
    return jsonify(data)

@app.route('/api/data/<id>', methods=['PUT'])
def update_data(id):
    # Invalidate cache on update
    cache_key = f"/api/data/{id}:"
    cache.delete(cache_key)
    return jsonify({'status': 'updated'})

if __name__ == '__main__':
    app.run(debug=True)

# Example Usage:
# curl -I http://localhost:5000/api/data/1
# curl -H "If-None-Match: <etag>" http://localhost:5000/api/data/1
```

Slide 10: GraphQL Batching and DataLoader

Implementation of efficient data loading in GraphQL using DataLoader pattern to prevent N+1 queries. This optimization technique batches multiple database queries into a single operation.

```python
from collections import defaultdict
from promise import Promise
from promise.dataloader import DataLoader
import graphene
from graphene import ObjectType, String, Int, List

class UserLoader(DataLoader):
    def batch_load_fn(self, user_ids):
        # Simulate database batch query
        users = fetch_users_in_batch(user_ids)
        user_map = {user['id']: user for user in users}
        return Promise.resolve([
            user_map.get(user_id) for user_id in user_ids
        ])

def fetch_users_in_batch(ids):
    # Simulate database query
    return [
        {'id': id, 'name': f'User {id}'}
        for id in ids
    ]

class Post(ObjectType):
    id = Int()
    title = String()
    author_id = Int()
    author = graphene.Field(lambda: User)

    def resolve_author(self, info):
        return info.context.user_loader.load(self.author_id)

class User(ObjectType):
    id = Int()
    name = String()
    posts = List(Post)

class Query(ObjectType):
    posts = List(Post)
    
    def resolve_posts(self, info):
        # Simulate fetching posts
        posts_data = [
            {'id': 1, 'title': 'Post 1', 'author_id': 1},
            {'id': 2, 'title': 'Post 2', 'author_id': 1},
            {'id': 3, 'title': 'Post 3', 'author_id': 2}
        ]
        return [Post(**post) for post in posts_data]

schema = graphene.Schema(query=Query)

# Example context setup:
context = {
    'user_loader': UserLoader()
}

# Example Query:
"""
{
    posts {
        id
        title
        author {
            id
            name
        }
    }
}
"""
```

Slide 11: REST API Error Handling and Response Standards

Implementation of comprehensive error handling and standardized response formatting for REST APIs. This system provides consistent error reporting and proper HTTP status code usage across all endpoints.

```python
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException
import traceback
from functools import wraps
import time

app = Flask(__name__)

def standard_response(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            response_data = {
                'success': True,
                'data': result,
                'metadata': {
                    'timestamp': time.time(),
                    'execution_time': time.time() - start_time
                }
            }
            return jsonify(response_data), 200
        except Exception as e:
            return handle_error(e)
    return decorated_function

def handle_error(error):
    if isinstance(error, HTTPException):
        response = {
            'success': False,
            'error': {
                'code': error.code,
                'message': error.description,
                'type': error.__class__.__name__
            }
        }
        return jsonify(response), error.code
    
    response = {
        'success': False,
        'error': {
            'code': 500,
            'message': str(error),
            'type': 'InternalServerError',
            'trace': traceback.format_exc() if app.debug else None
        }
    }
    return jsonify(response), 500

class APIError(Exception):
    def __init__(self, message, code=400, error_type='BadRequest'):
        self.message = message
        self.code = code
        self.error_type = error_type

@app.route('/api/resource/<id>')
@standard_response
def get_resource(id):
    if not id.isdigit():
        raise APIError('Invalid resource ID', 400)
    # Simulate resource fetch
    return {'id': id, 'name': f'Resource {id}'}

@app.errorhandler(Exception)
def handle_exception(e):
    return handle_error(e)

if __name__ == '__main__':
    app.run(debug=True)

# Example Usage:
# curl http://localhost:5000/api/resource/123
# curl http://localhost:5000/api/resource/invalid
```

Slide 12: GraphQL Error Handling and Extensions

Advanced implementation of GraphQL error handling with custom error types and extensions for detailed error reporting and monitoring.

```python
import graphene
from graphene import ObjectType, String, Int
import time
from functools import wraps

class CustomError(Exception):
    def __init__(self, message, code=None, path=None):
        super().__init__(message)
        self.code = code
        self.path = path
        self.extensions = {
            'code': code,
            'timestamp': time.time()
        }

def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CustomError as e:
            return None, e
        except Exception as e:
            return None, CustomError(
                str(e),
                code='INTERNAL_ERROR',
                path=func.__name__
            )
    return wrapper

class User(ObjectType):
    id = Int()
    name = String()
    email = String()

    @error_handler
    def resolve_email(self, info):
        if not info.context.get('is_authenticated'):
            raise CustomError(
                'Authentication required',
                code='AUTH_REQUIRED'
            )
        return f'user{self.id}@example.com'

class Query(ObjectType):
    user = graphene.Field(User, id=Int(required=True))

    @error_handler
    def resolve_user(self, info, id):
        if id <= 0:
            raise CustomError(
                'Invalid user ID',
                code='INVALID_ID',
                path=['user']
            )
        return User(id=id, name=f'User {id}')

schema = graphene.Schema(
    query=Query,
    # Enable error masking
    middleware=[
        lambda next, root, info, **args: next(root, info, **args)
    ]
)

# Example Query:
"""
{
    user(id: 1) {
        id
        name
        email
    }
}
"""
```

Slide 13: Real-World REST API Integration Use Case

A complete implementation of a REST API that integrates with a payment processing system, demonstrating error handling, retries, and transaction management in a production environment.

```python
from flask import Flask, jsonify, request
import requests
import jwt
import time
from functools import wraps
import uuid
from datetime import datetime, timedelta

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

class PaymentGateway:
    BASE_URL = 'https://api.payment-gateway.example'
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def create_payment(self, amount, currency, description):
        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.post(
                    f'{self.BASE_URL}/payments',
                    json={
                        'amount': amount,
                        'currency': currency,
                        'description': description,
                        'idempotency_key': str(uuid.uuid4())
                    },
                    timeout=5
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise PaymentError(f"Payment failed: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff

class PaymentError(Exception):
    pass

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        try:
            jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/api/payments', methods=['POST'])
@requires_auth
def create_payment():
    try:
        data = request.get_json()
        required_fields = ['amount', 'currency', 'description']
        if not all(field in data for field in required_fields):
            return jsonify({
                'error': 'Missing required fields'
            }), 400
        
        payment_gateway = PaymentGateway('your-api-key')
        result = payment_gateway.create_payment(
            amount=data['amount'],
            currency=data['currency'],
            description=data['description']
        )
        
        # Store payment record in database
        payment_record = {
            'id': str(uuid.uuid4()),
            'gateway_id': result['id'],
            'amount': data['amount'],
            'currency': data['currency'],
            'status': result['status'],
            'created_at': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'payment': payment_record
        }), 201
        
    except PaymentError as e:
        return jsonify({
            'error': str(e),
            'type': 'payment_error'
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'type': 'server_error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)

# Example Usage:
"""
curl -X POST \
  http://localhost:5000/api/payments \
  -H 'Authorization: Bearer your-token' \
  -H 'Content-Type: application/json' \
  -d '{
    "amount": 1000,
    "currency": "USD",
    "description": "Premium subscription"
}'
"""
```

Slide 14: Production GraphQL API Performance Monitoring

A comprehensive implementation of GraphQL performance monitoring and metrics collection, demonstrating how to track query complexity, response times, and error rates in a production environment.

```python
import graphene
from graphene import ObjectType, String, Int, List
import time
import json
from dataclasses import dataclass
from typing import Dict, List as TypeList
import asyncio
from prometheus_client import Counter, Histogram

# Metrics
QUERY_LATENCY = Histogram(
    'graphql_query_latency_seconds',
    'GraphQL query latency in seconds',
    ['operation_name', 'complexity']
)

ERROR_COUNTER = Counter(
    'graphql_errors_total',
    'Total GraphQL errors',
    ['error_type']
)

@dataclass
class QueryMetrics:
    operation_name: str
    start_time: float
    complexity: int
    depth: int
    fields: TypeList[str]

class MetricsMiddleware:
    def __init__(self):
        self.metrics: Dict[str, QueryMetrics] = {}

    def calculate_complexity(self, node, complexity=1, depth=1):
        if hasattr(node, 'selection_set'):
            selections = node.selection_set.selections
            return sum(
                self.calculate_complexity(selection, complexity, depth + 1)
                for selection in selections
            )
        return complexity * depth

    async def resolve(self, next, root, info, **args):
        operation_name = info.operation.name.value if info.operation.name else 'anonymous'
        request_id = id(info.context)

        if root is None:  # Only track top-level queries
            metrics = QueryMetrics(
                operation_name=operation_name,
                start_time=time.time(),
                complexity=self.calculate_complexity(info.operation),
                depth=0,
                fields=[]
            )
            self.metrics[request_id] = metrics

        try:
            result = await next(root, info, **args)
            
            if root is None:  # Query completed
                metrics = self.metrics.pop(request_id)
                duration = time.time() - metrics.start_time
                
                QUERY_LATENCY.labels(
                    operation_name=metrics.operation_name,
                    complexity=metrics.complexity
                ).observe(duration)
                
            return result
            
        except Exception as e:
            ERROR_COUNTER.labels(
                error_type=e.__class__.__name__
            ).inc()
            raise

class User(ObjectType):
    id = Int()
    name = String()
    posts = List(lambda: Post)

class Post(ObjectType):
    id = Int()
    title = String()
    content = String()

class Query(ObjectType):
    user = graphene.Field(User, id=Int(required=True))
    users = List(User)

    async def resolve_user(self, info, id):
        # Simulate database query
        await asyncio.sleep(0.1)
        return User(id=id, name=f"User {id}")

    async def resolve_users(self, info):
        await asyncio.sleep(0.2)
        return [
            User(id=i, name=f"User {i}")
            for i in range(1, 5)
        ]

schema = graphene.Schema(
    query=Query,
    middleware=[MetricsMiddleware()]
)

# Example Query:
"""
query GetUserWithPosts($userId: Int!) {
    user(id: $userId) {
        id
        name
        posts {
            id
            title
            content
        }
    }
}
"""
```

Slide 15: Additional Resources

*   Building Production-Ready GraphQL APIs
    *   [https://arxiv.org/abs/2105.xxxxx](https://arxiv.org/abs/2105.xxxxx) \[Modern GraphQL Architecture Patterns\]
    *   [https://arxiv.org/abs/2106.xxxxx](https://arxiv.org/abs/2106.xxxxx) \[Performance Optimization in GraphQL Systems\]
    *   [https://arxiv.org/abs/2107.xxxxx](https://arxiv.org/abs/2107.xxxxx) \[Security Best Practices for GraphQL APIs\]
*   REST API Design and Implementation
    *   [https://www.rfc-editor.org/rfc/rfc7231](https://www.rfc-editor.org/rfc/rfc7231)
    *   [https://www.rfc-editor.org/rfc/rfc7232](https://www.rfc-editor.org/rfc/rfc7232)
    *   [https://www.ics.uci.edu/~fielding/pubs/dissertation/rest\_arch\_style.htm](https://www.ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm)
*   Additional Search Terms:
    *   "REST API design patterns"
    *   "GraphQL schema design best practices"
    *   "API security implementation patterns"
    *   "Microservices architecture patterns"

