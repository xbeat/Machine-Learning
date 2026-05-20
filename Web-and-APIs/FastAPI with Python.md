## FastAPI with Python
Slide 1: Introduction to FastAPI

FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints. It's designed to be easy to use, fast to code, ready for production, and capable of handling high loads.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# To run: uvicorn main:app --reload
```

Slide 2: Key Features of FastAPI

FastAPI offers automatic API documentation, data validation, serialization, and more. It's built on top of Starlette for the web parts and Pydantic for the data parts, combining speed and simplicity.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None

@app.post("/items/")
async def create_item(item: Item):
    return {"item_name": item.name, "item_price": item.price}
```

Slide 3: Path Parameters

Path parameters allow you to capture values from the URL. FastAPI automatically validates and converts the parameters to the specified type.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

# Example URL: http://localhost:8000/items/42
# Result: {"item_id": 42}
```

Slide 4: Query Parameters

Query parameters are key-value pairs in the URL after the ? symbol. FastAPI automatically parses and validates these parameters.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

# Example URL: http://localhost:8000/items/?skip=20&limit=50
# Result: {"skip": 20, "limit": 50}
```

Slide 5: Request Body

FastAPI uses Pydantic models to define the structure of request bodies, providing automatic validation and serialization.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    username: str
    email: str
    full_name: str = None

@app.post("/users/")
async def create_user(user: User):
    return user

# Example request body:
# {
#     "username": "johndoe",
#     "email": "johndoe@example.com",
#     "full_name": "John Doe"
# }
```

Slide 6: Response Models

Response models define the structure of the API responses, ensuring type safety and automatic documentation.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    return item

# This ensures the response matches the Item model
```

Slide 7: Dependency Injection

FastAPI's dependency injection system allows you to declare shared logic or data processing as dependencies, promoting code reuse and separation of concerns.

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_decode_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user

@app.get("/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user

# This example demonstrates a simple authentication dependency
```

Slide 8: Background Tasks

FastAPI allows you to define background tasks that run after returning a response, ideal for operations that don't need to block the response.

```python
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()

def write_notification(email: str, message=""):
    with open("log.txt", mode="w") as email_file:
        content = f"notification for {email}: {message}"
        email_file.write(content)

@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_notification, email, message="some notification")
    return {"message": "Notification sent in the background"}
```

Slide 9: WebSockets

FastAPI supports WebSockets, allowing real-time bidirectional communication between the client and the server.

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")

# This creates a WebSocket endpoint that echoes received messages
```

Slide 10: Error Handling

FastAPI provides a straightforward way to handle errors and exceptions, allowing you to return custom error responses.

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

items = {"foo": "The Foo Wrestlers"}

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item": items[item_id]}

# This returns a 404 error if the item is not found
```

Slide 11: Middleware

Middleware allows you to add custom functionality to the request/response cycle, such as CORS, authentication, or logging.

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def main():
    return {"message": "Hello World"}

# This adds CORS middleware to allow all origins
```

Slide 12: Testing FastAPI Applications

FastAPI is built on top of Starlette, which provides a TestClient for easy testing of your API endpoints.

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/")
async def read_main():
    return {"msg": "Hello World"}

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}

# This sets up a simple test for the root endpoint
```

Slide 13: Real-Life Example: Task Management API

This example demonstrates a simple task management API using FastAPI, showcasing CRUD operations.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class Task(BaseModel):
    id: Optional[int] = None
    title: str
    description: Optional[str] = None
    completed: bool = False

tasks = []

@app.post("/tasks/", response_model=Task)
async def create_task(task: Task):
    task.id = len(tasks) + 1
    tasks.append(task)
    return task

@app.get("/tasks/", response_model=List[Task])
async def read_tasks():
    return tasks

@app.get("/tasks/{task_id}", response_model=Task)
async def read_task(task_id: int):
    task = next((task for task in tasks if task.id == task_id), None)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.put("/tasks/{task_id}", response_model=Task)
async def update_task(task_id: int, updated_task: Task):
    task = next((task for task in tasks if task.id == task_id), None)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    task.title = updated_task.title
    task.description = updated_task.description
    task.completed = updated_task.completed
    return task

@app.delete("/tasks/{task_id}", response_model=Task)
async def delete_task(task_id: int):
    task = next((task for task in tasks if task.id == task_id), None)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    tasks.remove(task)
    return task

# This API allows creating, reading, updating, and deleting tasks
```

Slide 14: Real-Life Example: Weather Information API

This example showcases a simple weather information API using FastAPI, demonstrating how to work with external data sources and caching.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from cachetools import TTLCache

app = FastAPI()

# Simple in-memory cache with a 30-minute TTL
cache = TTLCache(maxsize=100, ttl=1800)

class WeatherInfo(BaseModel):
    city: str
    temperature: float
    description: str

@app.get("/weather/{city}", response_model=WeatherInfo)
async def get_weather(city: str):
    if city in cache:
        return cache[city]

    # Simulating an external API call
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/weather/{city}")
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="City not found")
        data = response.json()

    weather_info = WeatherInfo(
        city=city,
        temperature=data["temperature"],
        description=data["description"]
    )

    # Cache the result
    cache[city] = weather_info
    return weather_info

# This API fetches weather information for a given city and caches the results
```

Slide 15: Additional Resources

For more information on FastAPI and related topics, consider exploring these resources:

1. FastAPI Official Documentation: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
2. Starlette Documentation: [https://www.starlette.io/](https://www.starlette.io/)
3. Pydantic Documentation: [https://pydantic-docs.helpmanual.io/](https://pydantic-docs.helpmanual.io/)
4. "Asynchronous Web APIs with FastAPI" on ArXiv: [https://arxiv.org/abs/2108.03261](https://arxiv.org/abs/2108.03261)
5. "A Comparative Analysis of FastAPI and Flask for Building RESTful APIs" on ArXiv: [https://arxiv.org/abs/2204.10584](https://arxiv.org/abs/2204.10584)

These resources provide in-depth information on FastAPI, its underlying technologies, and comparisons with other frameworks.

