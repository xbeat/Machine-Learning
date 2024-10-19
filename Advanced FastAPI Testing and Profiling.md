## Advanced FastAPI Testing and Profiling

Slide 1: Advanced FastAPI Testing and Profiling

FastAPI is a modern, fast web framework for building APIs with Python. To ensure the quality and performance of your FastAPI applications, it's crucial to implement comprehensive testing and profiling strategies. This presentation will cover advanced techniques for testing, debugging, and optimizing FastAPI projects.

```python
import pytest
from httpx import AsyncClient

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}

@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, FastAPI!"}
```

Slide 2: Setting Up a Testing Environment

To begin testing your FastAPI application, set up a testing environment using pytest and HTTPX. These tools allow you to write and run asynchronous tests for your API endpoints.

```python
fastapi==0.68.0
pytest==6.2.5
pytest-asyncio==0.15.1
httpx==0.19.0

# test_main.py
import pytest
from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_read_main():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, FastAPI!"}
```

Slide 3: Writing Comprehensive Tests

Create tests for various scenarios, including happy paths, edge cases, and error handling. Use parameterized tests to cover multiple input variations efficiently.

```python
from fastapi import HTTPException
from main import app, get_item

@pytest.mark.asyncio
@pytest.mark.parametrize("item_id, expected_name", [
    (1, "Item 1"),
    (2, "Item 2"),
    (3, "Item 3"),
])
async def test_get_item_success(item_id, expected_name):
    result = await get_item(item_id)
    assert result["name"] == expected_name

@pytest.mark.asyncio
async def test_get_item_not_found():
    with pytest.raises(HTTPException) as exc_info:
        await get_item(999)
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Item not found"
```

Slide 4: Mocking External Dependencies

When testing endpoints that rely on external services or databases, use mocking to isolate your tests and control the behavior of dependencies.

```python

@pytest.mark.asyncio
async def test_create_user():
    mock_db = MagicMock()
    mock_db.add_user.return_value = {"id": 1, "username": "testuser"}

    with patch("main.get_db", return_value=mock_db):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/users/", json={"username": "testuser", "password": "password123"})

    assert response.status_code == 201
    assert response.json() == {"id": 1, "username": "testuser"}
    mock_db.add_user.assert_called_once_with("testuser", "password123")
```

Slide 5: Test Coverage Analysis

Use coverage tools to measure and improve your test coverage. Aim for at least 90% coverage to ensure most of your code is tested.

```python
# pytest --cov=main --cov-report=term-missing

from coverage import Coverage

cov = Coverage()
cov.start()

# Run your tests here

cov.stop()
cov.save()

cov.report()
cov.html_report(directory='htmlcov')
```

Slide 6: Debugging FastAPI Applications

Use FastAPI's built-in debugging tools and Python's pdb module to troubleshoot issues in your application.

```python
import pdb

app = FastAPI(debug=True)

@app.get("/debug")
async def debug_endpoint(request: Request):
    # Set a breakpoint
    pdb.set_trace()
    
    # Inspect request details
    headers = request.headers
    query_params = request.query_params
    
    return {"message": "Debugging information", "headers": headers, "query_params": query_params}

# Run the application with:
# uvicorn main:app --reload --port 8000
```

Slide 7: Profiling FastAPI Performance

Use profiling tools to identify performance bottlenecks in your FastAPI application. The cProfile module helps measure execution time of different parts of your code.

```python
import pstats
from fastapi import FastAPI

app = FastAPI()

def profile(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative').print_stats(10)
        return result
    return wrapper

@app.get("/profile")
@profile
async def profiled_endpoint():
    # Simulating some work
    result = 0
    for i in range(1000000):
        result += i
    return {"result": result}

# Run the application and access /profile to see profiling results
```

Slide 8: Optimizing Database Queries

Improve the performance of database operations by optimizing queries and using async database clients.

```python
from databases import Database

app = FastAPI()
database = Database("postgresql://user:password@localhost/dbname")

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.get("/users")
async def get_users():
    query = "SELECT id, name FROM users LIMIT 100"
    results = await database.fetch_all(query)
    return results

# Optimized query with indexing and limiting results
@app.get("/posts")
async def get_posts():
    query = """
    SELECT p.id, p.title, u.name as author
    FROM posts p
    JOIN users u ON p.author_id = u.id
    WHERE p.published = TRUE
    ORDER BY p.created_at DESC
    LIMIT 10
    """
    results = await database.fetch_all(query)
    return results
```

Slide 9: Implementing Caching

Use caching to improve response times for frequently accessed data. Here's an example using Redis for caching.

```python
from redis import Redis
import json

app = FastAPI()
redis = Redis(host='localhost', port=6379, db=0)

def get_cached_data(key):
    data = redis.get(key)
    return json.loads(data) if data else None

def set_cached_data(key, data, expiration=3600):
    redis.setex(key, expiration, json.dumps(data))

@app.get("/cached-data/{item_id}")
async def get_data(item_id: int):
    cache_key = f"item:{item_id}"
    cached_data = get_cached_data(cache_key)
    
    if cached_data:
        return {"data": cached_data, "source": "cache"}
    
    # Simulating data fetch from database
    data = {"id": item_id, "name": f"Item {item_id}", "description": "Fetched from database"}
    set_cached_data(cache_key, data)
    
    return {"data": data, "source": "database"}
```

Slide 10: Asynchronous Background Tasks

Improve API responsiveness by offloading time-consuming tasks to background workers using FastAPI's BackgroundTasks.

```python
import time

app = FastAPI()

def process_data(data: dict):
    # Simulate a time-consuming task
    time.sleep(5)
    print(f"Processed data: {data}")

@app.post("/process")
async def process_endpoint(data: dict, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_data, data)
    return {"message": "Data processing started"}

# In a real-world scenario, you might use Celery or other task queues
# for more robust background task processing
```

Slide 11: Real-life Example: User Authentication System

Implement a secure user authentication system using FastAPI, with password hashing and JWT token generation.

```python
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

app = FastAPI()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Simulated user database
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": pwd_context.hash("secret123"),
    }
}

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    user = fake_users_db.get(username)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

Slide 12: Real-life Example: API Rate Limiting

Implement rate limiting to prevent abuse and ensure fair usage of your API.

```python
from fastapi.responses import JSONResponse
import time
from collections import defaultdict

app = FastAPI()

# Rate limiting configuration
RATE_LIMIT = 5  # requests
TIME_WINDOW = 60  # seconds

# Store for tracking request counts
request_counts = defaultdict(list)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    current_time = time.time()
    
    # Remove old requests outside the time window
    request_counts[client_ip] = [t for t in request_counts[client_ip] if current_time - t < TIME_WINDOW]
    
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Please try again later."}
        )
    
    # Add current request timestamp
    request_counts[client_ip].append(current_time)
    
    response = await call_next(request)
    return response

@app.get("/")
async def root():
    return {"message": "Hello, rate-limited world!"}

# Test the rate limiting by sending multiple requests quickly
```

Slide 13: Maintaining Clean, High-Quality Code

Adopt best practices for code organization, documentation, and linting to ensure your FastAPI project remains maintainable and scalable.

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="My FastAPI App",
    description="A clean and well-organized FastAPI application",
    version="1.0.0"
)

class Item(BaseModel):
    """
    Represents an item in the inventory.
    
    Attributes:
        id (int): The unique identifier for the item.
        name (str): The name of the item.
        description (str): A brief description of the item.
        price (float): The price of the item.
    """
    id: int
    name: str
    description: str
    price: float

@app.get("/items", response_model=List[Item])
async def get_items():
    """
    Retrieve a list of all items in the inventory.
    
    Returns:
        List[Item]: A list of Item objects representing the inventory.
    """
    # In a real application, this would fetch data from a database
    return [
        Item(id=1, name="Widget", description="A useful widget", price=9.99),
        Item(id=2, name="Gadget", description="A handy gadget", price=19.99)
    ]

@app.post("/items", response_model=Item)
async def create_item(item: Item):
    """
    Create a new item in the inventory.
    
    Args:
        item (Item): The item to be created.
    
    Returns:
        Item: The created item with its assigned ID.
    """
    # In a real application, this would add the item to a database
    return item

# To maintain code quality:
# 1. Use a linter (e.g., flake8) to enforce coding standards
# 2. Format code with tools like Black
# 3. Use type hints consistently
# 4. Write comprehensive docstrings
# 5. Organize your project into modules and packages
```

Slide 14: Additional Resources

To further enhance your FastAPI development skills, consider exploring these resources:

1. FastAPI Documentation: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
2. Starlette (ASGI framework used by FastAPI): [https://www.starlette.io/](https://www.starlette.io/)
3. Pydantic (Data validation library): [https://pydantic-docs.helpmanual.io/](https://pydantic-docs.helpmanual.io/)
4. SQLAlchemy (Database toolkit): [https://www.sqlalchemy.org/](https://www.sqlalchemy.org/)
5. AsyncIO in Python: [https://docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html)
6. ArXiv paper on API Design Best Practices: [https://arxiv.org/abs/2105.11120](https://arxiv.org/abs/2105.11120)

These resources will help you dive deeper into advanced FastAPI concepts, asynchronous programming, and API design principles.


