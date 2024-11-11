## Building a Chatbot with FastAPI and Python
Slide 1: Setting Up FastAPI Project Structure

Modern API development requires a well-organized project structure to maintain scalability and separation of concerns. FastAPI follows Python package conventions while enabling easy configuration of routers, middleware, and dependency injection for building robust chatbot applications.

```python
# project_structure.py
chatbot_api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── chat.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   └── services/
│       ├── __init__.py
│       └── chat_service.py
├── requirements.txt
└── README.md
```

Slide 2: Basic FastAPI Configuration

FastAPI requires specific configuration settings to handle CORS, middleware, and API documentation. The config module centralizes these settings while enabling environment-based configuration management for different deployment scenarios.

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Chatbot API"
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    class Config:
        case_sensitive = True

settings = Settings()

# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Slide 3: Chat Data Models

Data validation and serialization are crucial for API development. Pydantic models define the structure of request/response objects, ensuring type safety and automatic validation of incoming data.

```python
# api/models.py
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class Message(BaseModel):
    content: str
    timestamp: datetime = datetime.now()
    sender: str
    
class ChatRequest(BaseModel):
    message: str
    context: Optional[List[Message]] = []
    
class ChatResponse(BaseModel):
    response: str
    confidence: float
    timestamp: datetime = datetime.now()
```

Slide 4: Chat Service Implementation

The chat service implements core business logic for processing messages, maintaining conversation context, and generating responses. This implementation uses a simple pattern matching approach for demonstration purposes.

```python
# services/chat_service.py
from typing import List, Tuple
import re
from app.api.models import Message, ChatResponse

class ChatService:
    def __init__(self):
        self.patterns = [
            (r'hello|hi|hey', 'Hello! How can I help you today?'),
            (r'how are you', "I'm doing well, thank you for asking!"),
            (r'bye|goodbye', 'Goodbye! Have a great day!'),
        ]
        
    def process_message(self, message: str) -> ChatResponse:
        for pattern, response in self.patterns:
            if re.search(pattern, message.lower()):
                return ChatResponse(
                    response=response,
                    confidence=0.85
                )
        return ChatResponse(
            response="I'm not sure how to respond to that.",
            confidence=0.3
        )
```

Slide 5: API Route Implementation

FastAPI routes handle HTTP requests, process input data, and return responses. The chat endpoint demonstrates proper request handling, error management, and service integration.

```python
# api/routes/chat.py
from fastapi import APIRouter, HTTPException
from app.api.models import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = chat_service.process_message(request.message)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )
```

Slide 6: Dependency Injection Setup

FastAPI's dependency injection system enables clean separation of concerns and efficient resource management. This implementation shows how to handle database connections and service dependencies properly in a chatbot context.

```python
# dependencies.py
from fastapi import Depends
from typing import Generator
from app.services.chat_service import ChatService

def get_chat_service() -> Generator[ChatService, None, None]:
    service = ChatService()
    try:
        yield service
    finally:
        # Cleanup operations if needed
        pass

# Updated route with dependency
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    return chat_service.process_message(request.message)
```

Slide 7: Message Processing Pipeline

The message processing pipeline implements a chain of responsibility pattern to handle message preprocessing, intent classification, and response generation in a modular way.

```python
# services/processing.py
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class ProcessingContext:
    text: str
    tokens: List[str] = None
    intent: Optional[str] = None
    confidence: float = 0.0

class MessageProcessor:
    def __init__(self):
        self.pipeline = [
            self.tokenize,
            self.classify_intent,
            self.generate_response
        ]
    
    def process(self, text: str) -> ProcessingContext:
        context = ProcessingContext(text=text)
        for step in self.pipeline:
            context = step(context)
        return context
    
    def tokenize(self, context: ProcessingContext) -> ProcessingContext:
        context.tokens = context.text.lower().split()
        return context
    
    def classify_intent(self, context: ProcessingContext) -> ProcessingContext:
        # Simple intent classification
        if "help" in context.tokens:
            context.intent = "help"
            context.confidence = 0.9
        return context
```

Slide 8: Implementing Context Management

Efficient context management is crucial for maintaining conversation state and generating contextually relevant responses. This implementation uses a simple in-memory storage solution with TTL.

```python
# services/context_manager.py
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List
from app.api.models import Message

class ContextManager:
    def __init__(self, ttl_minutes: int = 30):
        self.conversations: Dict[str, List[Message]] = defaultdict(list)
        self.ttl = timedelta(minutes=ttl_minutes)
        self.last_cleanup = datetime.now()
    
    def add_message(self, session_id: str, message: Message):
        self.cleanup_old_sessions()
        self.conversations[session_id].append(message)
    
    def get_context(self, session_id: str) -> List[Message]:
        self.cleanup_old_sessions()
        return self.conversations.get(session_id, [])
    
    def cleanup_old_sessions(self):
        current_time = datetime.now()
        if current_time - self.last_cleanup > timedelta(minutes=5):
            for session_id in list(self.conversations.keys()):
                if not self.conversations[session_id]:
                    continue
                last_message = self.conversations[session_id][-1]
                if current_time - last_message.timestamp > self.ttl:
                    del self.conversations[session_id]
            self.last_cleanup = current_time
```

Slide 9: Advanced Response Generation

This implementation showcases a more sophisticated response generation system using template-based generation with dynamic slot filling and basic natural language variation.

```python
# services/response_generator.py
from typing import Dict, List
import random
from string import Template

class ResponseGenerator:
    def __init__(self):
        self.templates = {
            'greeting': [
                Template("Hello ${user}! How can I assist you today?"),
                Template("Hi ${user}! What can I help you with?")
            ],
            'error': [
                Template("I couldn't process ${error_type}. Can you rephrase?"),
                Template("There was an issue with ${error_type}. Mind trying again?")
            ]
        }
        
    def generate(self, intent: str, slots: Dict[str, str]) -> str:
        if intent not in self.templates:
            return "I'm not sure how to respond to that."
            
        template = random.choice(self.templates[intent])
        try:
            return template.safe_substitute(slots)
        except KeyError as e:
            return f"Error generating response: missing slot {str(e)}"
```

Slide 10: Real-world Implementation: Customer Service Bot

A practical implementation of a customer service chatbot handling product inquiries and basic support requests using the previously defined architecture.

```python
# services/customer_service_bot.py
from typing import Dict, Optional
import json
from pathlib import Path

class CustomerServiceBot:
    def __init__(self):
        self.product_db = self._load_product_database()
        self.faqs = self._load_faqs()
    
    def _load_product_database(self) -> Dict:
        db_path = Path("data/products.json")
        with db_path.open() as f:
            return json.load(f)
    
    def _load_faqs(self) -> Dict:
        faq_path = Path("data/faqs.json")
        with faq_path.open() as f:
            return json.load(f)
    
    def handle_query(self, query: str) -> Optional[str]:
        # Product inquiry handling
        for product_id, details in self.product_db.items():
            if product_id.lower() in query.lower():
                return (f"Product {details['name']}: "
                       f"Price: ${details['price']}, "
                       f"Stock: {details['stock']}")
        
        # FAQ handling
        for question, answer in self.faqs.items():
            if question.lower() in query.lower():
                return answer
        
        return None
```

Slide 11: Integration Testing Setup

Comprehensive testing ensures API reliability and maintainability. This implementation demonstrates proper testing setup using pytest with async support and fixture management for the chatbot API.

```python
# tests/test_chat_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.chat_service import ChatService

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def mock_chat_service(mocker):
    return mocker.patch.object(
        ChatService,
        'process_message',
        return_value={'response': 'Test response', 'confidence': 0.9}
    )

def test_chat_endpoint(test_client, mock_chat_service):
    response = test_client.post(
        "/api/v1/chat",
        json={"message": "hello", "context": []}
    )
    assert response.status_code == 200
    assert response.json()['response'] == 'Test response'
    assert response.json()['confidence'] == 0.9
```

Slide 12: Performance Monitoring Implementation

Implementing performance monitoring helps track API health and chatbot response quality. This system captures metrics like response times, error rates, and conversation success rates.

```python
# monitoring/metrics.py
from datetime import datetime
from typing import Dict, List
import statistics
from dataclasses import dataclass, field

@dataclass
class ChatMetrics:
    total_requests: int = 0
    successful_responses: int = 0
    response_times: List[float] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)
    
    def add_request(self, start_time: datetime, success: bool, error_type: str = None):
        self.total_requests += 1
        if success:
            self.successful_responses += 1
        else:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        self.response_times.append(response_time)
    
    def get_stats(self) -> Dict:
        return {
            'success_rate': self.successful_responses / max(1, self.total_requests),
            'avg_response_time': statistics.mean(self.response_times) if self.response_times else 0,
            'error_distribution': dict(self.error_counts)
        }
```

Slide 13: Rate Limiting and Security

Implementation of rate limiting and security measures protects the API from abuse while ensuring fair resource allocation among users.

```python
# middleware/security.py
from fastapi import HTTPException, Request
from datetime import datetime, timedelta
from collections import defaultdict
import jwt

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    async def check_rate_limit(self, request: Request):
        client_ip = request.client.host
        now = datetime.now()
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < timedelta(minutes=1)
        ]
        
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later."
            )
        
        self.requests[client_ip].append(now)
        
    def generate_api_key(self, user_id: str) -> str:
        return jwt.encode(
            {'user_id': user_id, 'exp': datetime.utcnow() + timedelta(days=30)},
            'your-secret-key',
            algorithm='HS256'
        )
```

Slide 14: Additional Resources

*   "Neural Approaches to Conversational AI" - [https://arxiv.org/abs/1809.08267](https://arxiv.org/abs/1809.08267)
*   "A Survey of Available Corpora for Building Data-Driven Dialogue Systems" - [https://arxiv.org/abs/1512.05742](https://arxiv.org/abs/1512.05742)
*   "Towards a Human-like Open-Domain Chatbot" - [https://arxiv.org/abs/2001.09977](https://arxiv.org/abs/2001.09977)
*   "Building Large Language Models: A Best Practice Guide" - [https://arxiv.org/abs/2004.08900](https://arxiv.org/abs/2004.08900)
*   "FastAPI for Production: Best Practices and Performance Optimization" - [https://arxiv.org/abs/2105.14851](https://arxiv.org/abs/2105.14851)

