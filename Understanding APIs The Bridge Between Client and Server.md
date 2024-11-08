## Understanding APIs The Bridge Between Client and Server
Slide 1: API Core Components and Setup

Modern APIs frequently require secure authentication, proper request formatting, and efficient response handling. A robust API client class encapsulates these core functionalities while providing a clean interface for making requests and processing responses.

```python
import requests
import json
from typing import Dict, Any, Optional

class APIClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def make_request(self, endpoint: str, method: str = 'GET', 
                    data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.request(method, url, json=data)
        response.raise_for_status()
        return response.json()

# Example usage
api = APIClient('https://api.example.com', 'your_api_key')
try:
    response = api.make_request('/users', method='POST', 
                              data={'name': 'John Doe'})
    print(f"Created user: {response}")
except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")
```

Slide 2: Request Rate Limiting and Retry Logic

API clients must handle rate limits and transient failures gracefully. Implementing exponential backoff and retry mechanisms ensures robust operation under real-world conditions while respecting API provider constraints.

```python
import time
from functools import wraps
from typing import Callable, Any

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        raise e
                    delay = base_delay * (2 ** (retries - 1))
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class RateLimitedAPIClient:
    def __init__(self, requests_per_second: float):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
    
    @retry_with_backoff(max_retries=3)
    def make_request(self, endpoint: str) -> Dict[str, Any]:
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_interval:
            time.sleep(self.min_interval - time_since_last_request)
        
        self.last_request_time = time.time()
        # Actual request logic here
        return {'status': 'success'}
```

Slide 3: API Response Caching

Implementing a caching mechanism reduces unnecessary API calls, improves response times, and minimizes server load. This implementation uses an LRU cache with TTL (Time To Live) for optimal resource utilization.

```python
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional, Tuple

class TTLCache:
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        if datetime.now() - timestamp > timedelta(seconds=self.ttl):
            del self.cache[key]
            return None
            
        self.cache.move_to_end(key)
        return value
    
    def put(self, key: str, value: Any) -> None:
        self.cache[key] = (value, datetime.now())
        self.cache.move_to_end(key)
        
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

# Example usage
cache = TTLCache(max_size=2, ttl_seconds=5)
cache.put('key1', 'value1')
print(f"Cached value: {cache.get('key1')}")
time.sleep(6)
print(f"After TTL: {cache.get('key1')}")  # Returns None
```

Slide 4: Asynchronous API Client

Modern applications benefit from asynchronous API calls to improve performance and resource utilization. This implementation uses Python's asyncio and aiohttp libraries for non-blocking API interactions.

```python
import asyncio
import aiohttp
from typing import List, Dict

class AsyncAPIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {api_key}'}
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            raise_for_status=True
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_all(self, endpoints: List[str]) -> List[Dict[str, Any]]:
        tasks = [
            self.fetch_one(endpoint) for endpoint in endpoints
        ]
        return await asyncio.gather(*tasks)
    
    async def fetch_one(self, endpoint: str) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        async with self.session.get(url) as response:
            return await response.json()

# Example usage
async def main():
    endpoints = ['/users/1', '/users/2', '/users/3']
    async with AsyncAPIClient('https://api.example.com', 'api_key') as client:
        results = await client.fetch_all(endpoints)
        print(f"Fetched data: {results}")

asyncio.run(main())
```

Slide 5: API Response Data Validation

Data validation ensures API responses conform to expected schemas and types. Using Pydantic models provides strong typing, automatic validation, and clear error messages for malformed API responses.

```python
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from datetime import datetime
import requests

class UserData(BaseModel):
    id: int
    username: str
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    created_at: datetime
    last_login: Optional[datetime] = None
    active: bool = True

class APIResponseValidator:
    def __init__(self, response_model: type[BaseModel]):
        self.response_model = response_model
    
    def validate_response(self, data: dict) -> BaseModel:
        try:
            return self.response_model.parse_obj(data)
        except ValidationError as e:
            raise ValueError(f"Invalid API response format: {e.errors()}")

# Example usage
api_response = {
    "id": 1,
    "username": "john_doe",
    "email": "john@example.com",
    "created_at": "2024-01-01T12:00:00Z",
    "active": True
}

validator = APIResponseValidator(UserData)
try:
    validated_data = validator.validate_response(api_response)
    print(f"Validated user data: {validated_data.dict()}")
except ValueError as e:
    print(f"Validation error: {e}")
```

Slide 6: API Pagination Handler

APIs often return large datasets in paginated form. This implementation handles automatic pagination, providing a clean iterator interface for accessing all available data while managing rate limits.

```python
from typing import Iterator, Optional
from dataclasses import dataclass

@dataclass
class PaginationMetadata:
    total_items: int
    items_per_page: int
    current_page: int
    total_pages: int

class PaginatedAPIClient:
    def __init__(self, base_url: str, items_per_page: int = 100):
        self.base_url = base_url
        self.items_per_page = items_per_page
        self.session = requests.Session()
    
    def get_paginated_data(self, endpoint: str) -> Iterator[dict]:
        page = 1
        while True:
            response = self.session.get(
                f"{self.base_url}/{endpoint}",
                params={
                    'page': page,
                    'per_page': self.items_per_page
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Yield each item from current page
            for item in data['items']:
                yield item
            
            # Check if we've reached the last page
            metadata = PaginationMetadata(**data['metadata'])
            if page >= metadata.total_pages:
                break
                
            page += 1

# Example usage
client = PaginatedAPIClient('https://api.example.com')
try:
    for item in client.get_paginated_data('users'):
        print(f"Processing user: {item['id']}")
except requests.exceptions.RequestException as e:
    print(f"Failed to fetch data: {e}")
```

Slide 7: API Webhook Handler

Webhooks allow APIs to push real-time updates to clients. This implementation provides a secure webhook receiver with signature validation and event processing capabilities.

```python
from flask import Flask, request, abort
import hmac
import hashlib
from typing import Callable, Dict
from functools import wraps

class WebhookHandler:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode('utf-8')
        self.event_handlers: Dict[str, Callable] = {}
        self.app = Flask(__name__)
        
        @self.app.route('/webhook', methods=['POST'])
        def webhook_endpoint():
            return self._handle_webhook_request()
    
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        computed = hmac.new(
            self.secret_key,
            payload,
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(computed, signature)
    
    def event_handler(self, event_type: str):
        def decorator(func: Callable):
            self.event_handlers[event_type] = func
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def _handle_webhook_request(self):
        payload = request.get_data()
        signature = request.headers.get('X-Webhook-Signature')
        
        if not signature or not self.verify_signature(payload, signature):
            abort(401)
        
        event_data = request.get_json()
        event_type = event_data.get('event_type')
        
        if event_type in self.event_handlers:
            self.event_handlers[event_type](event_data)
            return {'status': 'success'}, 200
        
        abort(400, f"Unknown event type: {event_type}")

# Example usage
webhook_handler = WebhookHandler('your_secret_key')

@webhook_handler.event_handler('user.created')
def handle_user_created(event_data: dict):
    user_id = event_data['user']['id']
    print(f"New user created: {user_id}")

if __name__ == '__main__':
    webhook_handler.app.run(port=5000)
```

Slide 8: API Error Handling and Custom Exceptions

Robust error handling is crucial for API interactions. This implementation provides custom exceptions and comprehensive error handling for various API failure scenarios.

```python
class APIError(Exception):
    def __init__(self, message: str, status_code: int = None, 
                 response: requests.Response = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)

class APIClientError(APIError):
    """4xx client errors"""
    pass

class APIServerError(APIError):
    """5xx server errors"""
    pass

class APIRateLimitError(APIClientError):
    """Rate limit exceeded"""
    pass

class APIErrorHandler:
    ERROR_CLASSES = {
        429: APIRateLimitError,
        404: APIClientError,
        500: APIServerError
    }
    
    @classmethod
    def handle_error_response(cls, response: requests.Response):
        try:
            error_data = response.json()
        except ValueError:
            error_data = {'message': response.text}
        
        error_class = cls.ERROR_CLASSES.get(
            response.status_code,
            APIError if response.status_code >= 500 else APIClientError
        )
        
        raise error_class(
            message=error_data.get('message', 'Unknown error'),
            status_code=response.status_code,
            response=response
        )

# Example usage
def make_api_call(url: str) -> dict:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if e.response is not None:
            APIErrorHandler.handle_error_response(e.response)
        raise APIError(f"Request failed: {str(e)}")

# Usage example
try:
    data = make_api_call('https://api.example.com/resource')
except APIRateLimitError as e:
    print(f"Rate limit exceeded: {e.message}")
except APIClientError as e:
    print(f"Client error: {e.message}")
except APIServerError as e:
    print(f"Server error: {e.message}")
```

Slide 9: API Authentication Manager

Implementing secure authentication management, handling token refresh, and maintaining session state across requests. This implementation supports multiple authentication methods and automatic token renewal.

```python
from datetime import datetime, timedelta
import jwt
from typing import Optional, Tuple

class AuthenticationManager:
    def __init__(self, client_id: str, client_secret: str, auth_url: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_url = auth_url
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
    
    def get_valid_token(self) -> str:
        if not self._is_token_valid():
            if self.refresh_token:
                self._refresh_access_token()
            else:
                self._fetch_new_tokens()
        return self.access_token
    
    def _is_token_valid(self) -> bool:
        if not self.access_token or not self.token_expiry:
            return False
        return datetime.now() < self.token_expiry - timedelta(minutes=5)
    
    def _fetch_new_tokens(self) -> None:
        response = requests.post(
            self.auth_url,
            data={
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
        )
        response.raise_for_status()
        self._update_tokens(response.json())
    
    def _refresh_access_token(self) -> None:
        response = requests.post(
            self.auth_url,
            data={
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token,
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
        )
        response.raise_for_status()
        self._update_tokens(response.json())
    
    def _update_tokens(self, token_data: dict) -> None:
        self.access_token = token_data['access_token']
        self.refresh_token = token_data.get('refresh_token')
        expires_in = token_data.get('expires_in', 3600)
        self.token_expiry = datetime.now() + timedelta(seconds=expires_in)

# Example usage
auth_manager = AuthenticationManager(
    client_id='your_client_id',
    client_secret='your_client_secret',
    auth_url='https://api.example.com/oauth/token'
)

try:
    token = auth_manager.get_valid_token()
    print(f"Valid token obtained: {token[:10]}...")
except requests.exceptions.RequestException as e:
    print(f"Authentication failed: {e}")
```

Slide 10: API Request Batching

Optimizing API usage by implementing request batching to reduce network overhead and improve performance when dealing with multiple API calls.

```python
from collections import deque
from typing import List, Any, Callable
import asyncio
import time

class BatchRequestHandler:
    def __init__(self, 
                 batch_size: int = 50,
                 batch_interval: float = 1.0,
                 max_retries: int = 3):
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.max_retries = max_retries
        self.queue = deque()
        self.processing = False
        self._last_batch_time = 0
    
    async def add_request(self, 
                         endpoint: str, 
                         payload: dict) -> asyncio.Future:
        future = asyncio.Future()
        self.queue.append((endpoint, payload, future))
        
        if not self.processing:
            asyncio.create_task(self._process_queue())
        
        return future
    
    async def _process_queue(self) -> None:
        self.processing = True
        
        while self.queue:
            current_time = time.time()
            time_since_last_batch = current_time - self._last_batch_time
            
            if time_since_last_batch < self.batch_interval:
                await asyncio.sleep(
                    self.batch_interval - time_since_last_batch
                )
            
            batch = []
            futures = []
            
            while self.queue and len(batch) < self.batch_size:
                endpoint, payload, future = self.queue.popleft()
                batch.append((endpoint, payload))
                futures.append(future)
            
            if batch:
                try:
                    results = await self._execute_batch(batch)
                    for future, result in zip(futures, results):
                        future.set_result(result)
                except Exception as e:
                    for future in futures:
                        future.set_exception(e)
            
            self._last_batch_time = time.time()
        
        self.processing = False
    
    async def _execute_batch(self, 
                           batch: List[Tuple[str, dict]]
                           ) -> List[Any]:
        retries = 0
        while retries < self.max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    tasks = [
                        self._make_request(session, endpoint, payload)
                        for endpoint, payload in batch
                    ]
                    return await asyncio.gather(*tasks)
            except Exception as e:
                retries += 1
                if retries == self.max_retries:
                    raise e
                await asyncio.sleep(2 ** retries)
        return []
    
    async def _make_request(self,
                           session: aiohttp.ClientSession,
                           endpoint: str,
                           payload: dict) -> dict:
        async with session.post(endpoint, json=payload) as response:
            response.raise_for_status()
            return await response.json()

# Example usage
async def main():
    batch_handler = BatchRequestHandler()
    
    # Create multiple requests
    requests = [
        batch_handler.add_request(
            'https://api.example.com/users',
            {'name': f'User {i}'}
        )
        for i in range(100)
    ]
    
    # Wait for all requests to complete
    results = await asyncio.gather(*requests)
    print(f"Processed {len(results)} requests in batches")

if __name__ == '__main__':
    asyncio.run(main())
```

Slide 11: API Response Transformation Pipeline

A flexible system for transforming API responses through a series of data processing steps, allowing for consistent data normalization, enrichment, and formatting across different API endpoints.

```python
from abc import ABC, abstractmethod
from typing import Any, List, Dict
from dataclasses import dataclass
import pandas as pd

class TransformationStep(ABC):
    @abstractmethod
    def transform(self, data: Any) -> Any:
        pass

class DateNormalizer(TransformationStep):
    def transform(self, data: Dict) -> Dict:
        for key, value in data.items():
            if isinstance(value, str) and 'date' in key.lower():
                try:
                    data[key] = pd.to_datetime(value).isoformat()
                except ValueError:
                    pass
        return data

class DataFlattener(TransformationStep):
    def transform(self, data: Dict) -> Dict:
        result = {}
        def flatten(item, prefix=''):
            for key, value in item.items():
                new_key = f"{prefix}_{key}" if prefix else key
                if isinstance(value, dict):
                    flatten(value, new_key)
                else:
                    result[new_key] = value
        flatten(data)
        return result

class ResponseTransformer:
    def __init__(self, steps: List[TransformationStep]):
        self.steps = steps
    
    def transform(self, data: Any) -> Any:
        result = data
        for step in self.steps:
            result = step.transform(result)
        return result

# Example usage
class CustomFieldMapper(TransformationStep):
    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping
    
    def transform(self, data: Dict) -> Dict:
        return {
            self.mapping.get(k, k): v 
            for k, v in data.items()
        }

# Create transformation pipeline
transformer = ResponseTransformer([
    DateNormalizer(),
    DataFlattener(),
    CustomFieldMapper({
        'user_id': 'id',
        'user_name': 'name'
    })
])

# Example API response
api_response = {
    'user_id': 123,
    'user_name': 'John Doe',
    'created_date': '2024-01-15T10:30:00',
    'metadata': {
        'last_login': '2024-02-01',
        'status': 'active'
    }
}

transformed_data = transformer.transform(api_response)
print(f"Transformed data: {transformed_data}")
```

Slide 12: Real-world Example - Weather API Integration

Implementation of a complete weather API client with caching, rate limiting, and error handling for production use.

```python
from dataclasses import dataclass
from typing import Optional, List, Dict
import json
from datetime import datetime, timedelta

@dataclass
class WeatherData:
    temperature: float
    humidity: float
    wind_speed: float
    description: str
    timestamp: datetime

class WeatherAPIClient:
    def __init__(self, api_key: str, cache_duration: int = 1800):
        self.api_key = api_key
        self.cache_duration = cache_duration
        self.cache: Dict[str, tuple[WeatherData, datetime]] = {}
        self.rate_limiter = RateLimitedAPIClient(requests_per_second=2)
    
    async def get_weather(self, 
                         city: str, 
                         country_code: str) -> WeatherData:
        cache_key = f"{city.lower()},{country_code.lower()}"
        
        # Check cache
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_duration):
                return data
        
        # Fetch new data
        try:
            weather_data = await self._fetch_weather_data(city, country_code)
            self.cache[cache_key] = (weather_data, datetime.now())
            return weather_data
        except Exception as e:
            # Log error and raise custom exception
            logger.error(f"Weather API error: {str(e)}")
            raise WeatherAPIError(f"Failed to fetch weather data: {str(e)}")
    
    async def _fetch_weather_data(self, 
                                city: str, 
                                country_code: str) -> WeatherData:
        url = (
            f"https://api.weatherapi.com/v1/current.json"
            f"?key={self.api_key}&q={city},{country_code}"
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                
                return WeatherData(
                    temperature=data['current']['temp_c'],
                    humidity=data['current']['humidity'],
                    wind_speed=data['current']['wind_kph'],
                    description=data['current']['condition']['text'],
                    timestamp=datetime.now()
                )
    
    async def get_forecast(self, 
                          city: str, 
                          country_code: str, 
                          days: int = 5) -> List[WeatherData]:
        if not 1 <= days <= 7:
            raise ValueError("Forecast days must be between 1 and 7")
        
        url = (
            f"https://api.weatherapi.com/v1/forecast.json"
            f"?key={self.api_key}&q={city},{country_code}&days={days}"
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                
                return [
                    WeatherData(
                        temperature=day['day']['avgtemp_c'],
                        humidity=day['day']['avghumidity'],
                        wind_speed=day['day']['maxwind_kph'],
                        description=day['day']['condition']['text'],
                        timestamp=datetime.strptime(
                            day['date'], 
                            '%Y-%m-%d'
                        )
                    )
                    for day in data['forecast']['forecastday']
                ]

# Example usage
async def main():
    client = WeatherAPIClient('your_api_key')
    
    try:
        # Get current weather
        weather = await client.get_weather('London', 'UK')
        print(f"Current weather in London: {weather}")
        
        # Get forecast
        forecast = await client.get_forecast('London', 'UK', days=3)
        print("\nForecast for next 3 days:")
        for day in forecast:
            print(f"{day.timestamp.date()}: {day.description}, "
                  f"{day.temperature}°C")
    
    except WeatherAPIError as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    asyncio.run(main())
```

Slide 13: Real-world Example - E-commerce API Integration

A comprehensive implementation of an e-commerce API client handling product management, inventory updates, and order processing with proper error handling and validation.

```python
from typing import Optional, List, Dict, Union
from decimal import Decimal
from enum import Enum
from pydantic import BaseModel, Field

class OrderStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class Product(BaseModel):
    id: str
    name: str
    price: Decimal
    stock: int
    sku: str
    category: str
    
class Order(BaseModel):
    id: str
    customer_id: str
    items: List[Dict[str, Union[str, int]]]
    total_amount: Decimal
    status: OrderStatus
    created_at: datetime

class EcommerceAPI:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        return session
    
    async def get_product(self, product_id: str) -> Product:
        url = f"{self.base_url}/products/{product_id}"
        async with self.session.get(url) as response:
            response.raise_for_status()
            data = await response.json()
            return Product(**data)
    
    async def update_inventory(self, 
                             product_id: str, 
                             quantity: int) -> Product:
        url = f"{self.base_url}/products/{product_id}/inventory"
        payload = {'quantity': quantity}
        
        async with self.session.patch(url, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return Product(**data)
    
    async def create_order(self, 
                          customer_id: str, 
                          items: List[Dict[str, Union[str, int]]]
                          ) -> Order:
        url = f"{self.base_url}/orders"
        payload = {
            'customer_id': customer_id,
            'items': items
        }
        
        async with self.session.post(url, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return Order(**data)
    
    async def update_order_status(self, 
                                order_id: str, 
                                status: OrderStatus) -> Order:
        url = f"{self.base_url}/orders/{order_id}/status"
        payload = {'status': status.value}
        
        async with self.session.patch(url, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return Order(**data)

# Example usage
async def process_order():
    api = EcommerceAPI(
        api_key='your_api_key',
        base_url='https://api.ecommerce.com/v1'
    )
    
    try:
        # Create new order
        order_items = [
            {'product_id': 'PROD123', 'quantity': 2},
            {'product_id': 'PROD456', 'quantity': 1}
        ]
        
        order = await api.create_order(
            customer_id='CUST789',
            items=order_items
        )
        print(f"Order created: {order.id}")
        
        # Update inventory for ordered items
        for item in order_items:
            product = await api.get_product(item['product_id'])
            new_quantity = product.stock - item['quantity']
            
            updated_product = await api.update_inventory(
                product_id=item['product_id'],
                quantity=new_quantity
            )
            print(f"Updated inventory for {updated_product.name}: "
                  f"{updated_product.stock} remaining")
        
        # Update order status
        updated_order = await api.update_order_status(
            order_id=order.id,
            status=OrderStatus.PROCESSING
        )
        print(f"Order status updated to: {updated_order.status.value}")
        
    except requests.exceptions.RequestException as e:
        print(f"API error: {str(e)}")
        # Implement rollback mechanism here
        
if __name__ == '__main__':
    asyncio.run(process_order())
```

Slide 14: Additional Resources

*   Representational State Transfer (REST) APIs: A Comprehensive Study [https://arxiv.org/abs/2104.12678](https://arxiv.org/abs/2104.12678)
*   Design Patterns for Modern Web APIs: A Systematic Review [https://arxiv.org/abs/2106.15592](https://arxiv.org/abs/2106.15592)
*   Security Considerations in API Design and Implementation [https://arxiv.org/abs/2103.09343](https://arxiv.org/abs/2103.09343)
*   Performance Optimization Techniques for Web APIs: A Survey [https://arxiv.org/abs/2105.11343](https://arxiv.org/abs/2105.11343)
*   Machine Learning Applications in API Design and Management [https://arxiv.org/abs/2202.15821](https://arxiv.org/abs/2202.15821)

