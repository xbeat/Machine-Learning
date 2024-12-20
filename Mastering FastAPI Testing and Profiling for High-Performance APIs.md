## Mastering FastAPI Testing and Profiling for High-Performance APIs
Slide 1: Comprehensive Testing with pytest & HTTPX

Testing is crucial for ensuring the reliability and correctness of FastAPI applications. Pytest and HTTPX are powerful tools for writing and executing tests. Let's explore how to use them effectively.

```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_item():
    response = client.post(
        "/items/",
        json={"name": "Foo", "price": 45.2}
    )
    assert response.status_code == 200
    assert response.json() == {
        "name": "Foo",
        "price": 45.2,
        "id": 1
    }
```

Slide 2: Asynchronous Testing with HTTPX

HTTPX allows for asynchronous testing, which is particularly useful for FastAPI's asynchronous nature. Here's how to implement async tests:

```python
import asyncio
import pytest
from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_async_read_main():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

@pytest.mark.asyncio
async def test_async_create_item():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/items/",
            json={"name": "Bar", "price": 30.5}
        )
    assert response.status_code == 200
    assert response.json() == {
        "name": "Bar",
        "price": 30.5,
        "id": 2
    }
```

Slide 3: Debugging Tips for FastAPI

Effective debugging is key to maintaining a robust API. Let's look at some debugging techniques specific to FastAPI:

```python
from fastapi import FastAPI, Request
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.debug(f"Outgoing response: Status {response.status_code}")
    return response

@app.get("/debug")
async def debug_endpoint():
    logger.debug("This is a debug message")
    return {"message": "Check your console for debug output"}
```

Slide 4: Using pdb for Interactive Debugging

The Python Debugger (pdb) is a powerful tool for interactive debugging. Here's how to use it in a FastAPI application:

```python
from fastapi import FastAPI
import pdb

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    # Set a breakpoint
    pdb.set_trace()
    
    # Simulate some processing
    item = {"id": item_id, "name": f"Item {item_id}"}
    
    return item

# To use: Run your FastAPI app, then make a request to /items/1
# The debugger will pause execution at the breakpoint
```

Slide 5: Profiling Techniques: cProfile

Profiling is essential for optimizing performance. Let's use cProfile to profile a FastAPI endpoint:

```python
import cProfile
import pstats
from fastapi import FastAPI

app = FastAPI()

def expensive_operation():
    return sum(i * i for i in range(10**6))

@app.get("/profile")
def profile_endpoint():
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = expensive_operation()
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(10)  # Print top 10 time-consuming functions
    
    return {"result": result}

# To use: Run your FastAPI app and make a request to /profile
# Check your console for profiling output
```

Slide 6: Memory Profiling with memory\_profiler

To identify memory-intensive operations, we can use the memory\_profiler library:

```python
from fastapi import FastAPI
from memory_profiler import profile

app = FastAPI()

@profile
def memory_intensive_operation():
    return [i * i for i in range(10**6)]

@app.get("/memory-profile")
def memory_profile_endpoint():
    result = memory_intensive_operation()
    return {"result_length": len(result)}

# To use: Install memory_profiler, run your FastAPI app with:
# mprof run your_app.py
# Then make a request to /memory-profile and analyze the results with:
# mprof plot
```

Slide 7: Achieving High Test Coverage

High test coverage ensures that most of your code is tested. Here's how to measure and improve coverage:

```python
# Install pytest-cov: pip install pytest-cov

# Run tests with coverage:
# pytest --cov=your_app_directory tests/

# Example test file: test_main.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_item():
    response = client.post(
        "/items/",
        json={"name": "Test", "price": 10.5}
    )
    assert response.status_code == 200
    assert response.json() == {
        "name": "Test",
        "price": 10.5,
        "id": 1
    }

# Add more tests to cover different scenarios and edge cases
```

Slide 8: Maintaining Clean Code: Linting with flake8

Clean code is easier to maintain and less prone to bugs. Let's use flake8 for linting:

```python
# Install flake8: pip install flake8

# Run flake8: flake8 your_app_directory

# Example of fixing linting issues:

# Before:
def some_function( x,y ):
    if x==y:
        return True
    else:
        return False

# After:
def some_function(x, y):
    return x == y

# To automate: Add a pre-commit hook that runs flake8
# Create .pre-commit-config.yaml:
repos:
  - repo: https://github.com/pycqa/flake8
    rev: 3.9.2
    hooks:
      - id: flake8
```

Slide 9: Code Formatting with Black

Consistent code formatting improves readability. Black is an opinionated formatter that ensures consistency:

```python
# Install black: pip install black

# Run black: black your_app_directory

# Example of Black formatting:

# Before:
def long_function_name(
    var_one, var_two, var_three,
    var_four):
    print(var_one)

# After:
def long_function_name(
    var_one,
    var_two,
    var_three,
    var_four,
):
    print(var_one)

# To automate: Add Black to your pre-commit hooks
# In .pre-commit-config.yaml:
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
```

Slide 10: Real-Life Example: API Rate Limiting

Implementing rate limiting is crucial for protecting your API from abuse. Here's a practical example using FastAPI:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time
from collections import defaultdict

app = FastAPI()

# Simple in-memory rate limiter
class RateLimiter:
    def __init__(self, calls=5, period=60):
        self.calls = calls
        self.period = period
        self.records = defaultdict(list)

    def is_allowed(self, key):
        now = time.time()
        self.records[key] = [t for t in self.records[key] if now - t < self.period]
        if len(self.records[key]) >= self.calls:
            return False
        self.records[key].append(now)
        return True

limiter = RateLimiter()

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if not limiter.is_allowed(request.client.host):
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})
    return await call_next(request)

@app.get("/")
async def root():
    return {"message": "Hello World"}

# To test: Run the app and make repeated requests to the root endpoint
```

Slide 11: Real-Life Example: Data Validation with Pydantic

Proper data validation is essential for maintaining data integrity. Let's use Pydantic with FastAPI:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
import re

app = FastAPI()

class User(BaseModel):
    username: str
    email: str
    age: int
    hobbies: List[str] = []

    @validator('username')
    def username_alphanumeric(cls, v):
        assert v.isalnum(), 'must be alphanumeric'
        return v

    @validator('email')
    def email_valid(cls, v):
        regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(regex, v):
            raise ValueError('Invalid email format')
        return v

    @validator('age')
    def age_valid(cls, v):
        if v < 0 or v > 120:
            raise ValueError('Age must be between 0 and 120')
        return v

@app.post("/users/")
async def create_user(user: User):
    # Here you would typically save the user to a database
    return {"message": "User created successfully", "user": user}

# To test: Send a POST request to /users/ with valid and invalid data
```

Slide 12: Performance Optimization: Asynchronous Database Queries

Optimizing database queries is crucial for API performance. Here's an example using asyncpg with FastAPI:

```python
from fastapi import FastAPI
import asyncpg
from typing import List

app = FastAPI()

async def get_db_connection():
    return await asyncpg.connect(user='user', password='password',
                                 database='database', host='127.0.0.1')

@app.on_event("startup")
async def startup():
    app.state.pool = await asyncpg.create_pool(user='user', password='password',
                                               database='database', host='127.0.0.1')

@app.on_event("shutdown")
async def shutdown():
    await app.state.pool.close()

@app.get("/users", response_model=List[dict])
async def get_users():
    async with app.state.pool.acquire() as connection:
        users = await connection.fetch("SELECT * FROM users LIMIT 100")
    return [dict(user) for user in users]

@app.get("/user/{user_id}")
async def get_user(user_id: int):
    async with app.state.pool.acquire() as connection:
        user = await connection.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
    if user:
        return dict(user)
    return {"error": "User not found"}

# To use: Set up a PostgreSQL database and adjust the connection details
```

Slide 13: Continuous Integration and Deployment (CI/CD) for FastAPI

Implementing CI/CD ensures consistent testing and deployment. Here's a sample GitHub Actions workflow:

```yaml
name: FastAPI CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: pytest
    - name: Run linter
      run: flake8 .

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Heroku
      uses: akhileshns/heroku-deploy@v3.12.12
      with:
        heroku_api_key: ${{secrets.HEROKU_API_KEY}}
        heroku_app_name: "your-app-name"
        heroku_email: "your-email@example.com"

# To use: Set up a GitHub repository, create a Heroku app, and add your Heroku API key to GitHub secrets
```

Slide 14: Additional Resources

For further learning and advanced techniques in FastAPI testing and profiling:

1.  FastAPI Documentation: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
2.  Pytest Documentation: [https://docs.pytest.org/](https://docs.pytest.org/)
3.  HTTPX Documentation: [https://www.python-httpx.org/](https://www.python-httpx.org/)
4.  Python Profilers: [https://docs.python.org/3/library/profile.html](https://docs.python.org/3/library/profile.html)
5.  Pydantic Documentation: [https://pydantic-docs.helpmanual.io/](https://pydantic-docs.helpmanual.io/)
6.  Asyncpg Documentation: [https://magicstack.github.io/asyncpg/current/](https://magicstack.github.io/asyncpg/current/)
7.  GitHub Actions Documentation: [https://docs.github.com/en/actions](https://docs.github.com/en/actions)

These resources provide in-depth information on the tools and techniques discussed in this presentation.

