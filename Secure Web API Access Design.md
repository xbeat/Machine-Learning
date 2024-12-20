## Secure Web API Access Design
Slide 1: Introduction to Web API Authentication

Authentication is a critical security measure that verifies the identity of users or systems making API requests. It ensures that only authorized entities can access protected resources and prevents unauthorized access through request validation.

```python
from enum import Enum
from typing import Dict, Optional

class AuthType(Enum):
    TOKEN = "token"
    HMAC = "hmac"

class AuthenticationManager:
    def __init__(self, auth_type: AuthType):
        self.auth_type = auth_type
        self.tokens: Dict[str, dict] = {}  # Store active tokens
        self.api_keys: Dict[str, str] = {} # Store API keys
```

Slide 2: Token-Based Authentication Implementation

Token-based authentication generates temporary access tokens after validating user credentials. These tokens are included in subsequent API requests instead of sending credentials repeatedly, reducing security risks and improving performance.

```python
import jwt
from datetime import datetime, timedelta
import secrets

class TokenAuth:
    def __init__(self, secret_key: str = secrets.token_hex(32)):
        self.secret_key = secret_key
    
    def generate_token(self, user_id: str, expires_in: int = 3600) -> str:
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def validate_token(self, token: str) -> Optional[dict]:
        try:
            return jwt.decode(token, self.secret_key, algorithms=['HS256'])
        except jwt.InvalidTokenError:
            return None
```

Slide 3: HMAC Authentication Core

HMAC authentication uses cryptographic hashing to verify request authenticity. By combining a secret key with request parameters, it creates a unique signature that proves the request's legitimacy and detects any tampering.

```python
import hmac
import hashlib
import time
from typing import Dict, Any

class HMACAuth:
    def __init__(self, api_key: str, app_id: str):
        self.api_key = api_key.encode('utf-8')
        self.app_id = app_id
        
    def generate_signature(self, params: Dict[str, Any]) -> str:
        # Sort parameters to ensure consistent ordering
        ordered_params = dict(sorted(params.items()))
        
        # Create string to sign
        message = '&'.join(f"{k}={v}" for k, v in ordered_params.items())
        
        # Generate HMAC signature
        signature = hmac.new(
            self.api_key,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
```

Slide 4: Request Parameters Processing

Processing request parameters is crucial for HMAC authentication as it ensures consistent signature generation across client and server. This implementation handles various parameter types and maintains ordering.

```python
def prepare_request_params(self, endpoint: str, method: str, 
                         params: Dict[str, Any]) -> Dict[str, str]:
    # Add required authentication parameters
    auth_params = {
        'app_id': self.app_id,
        'timestamp': str(int(time.time())),
        'method': method.upper(),
        'endpoint': endpoint
    }
    
    # Combine with request parameters
    all_params = {**auth_params, **params}
    
    # Convert all values to strings
    return {k: str(v) for k, v in all_params.items()}
```

Slide 5: API Request Handler

The request handler manages both token and HMAC authentication methods, implementing security best practices and providing a unified interface for API access verification.

```python
class APIRequestHandler:
    def __init__(self):
        self.token_auth = TokenAuth()
        self.hmac_auth = {}  # Dict of app_id to HMACAuth instances
    
    def verify_request(self, request_headers: Dict[str, str], 
                      request_data: Dict[str, Any]) -> bool:
        auth_type = request_headers.get('Auth-Type')
        
        if auth_type == 'token':
            token = request_headers.get('Authorization')
            return self.verify_token(token)
        elif auth_type == 'hmac':
            return self.verify_hmac(request_headers, request_data)
        
        return False
```

Slide 6: Token Verification Implementation

Token verification ensures the authenticity and validity of access tokens by checking digital signatures, expiration times, and token claims. This implementation includes revocation checking and JWT validation.

```python
def verify_token(self, token: str) -> bool:
    if not token or not token.startswith('Bearer '):
        return False
        
    token = token.split('Bearer ')[1]
    try:
        # Verify token and check expiration
        payload = jwt.decode(
            token,
            self.secret_key,
            algorithms=['HS256'],
            options={'verify_exp': True}
        )
        
        # Check if token is revoked
        if self.is_token_revoked(payload['jti']):
            return False
            
        return True
    except jwt.InvalidTokenError:
        return False
```

Slide 7: HMAC Signature Verification

HMAC signature verification compares client-generated and server-generated signatures to ensure request integrity. This process includes timestamp validation to prevent replay attacks.

```python
class HMACVerifier:
    def __init__(self, max_timestamp_diff: int = 300):
        self.max_timestamp_diff = max_timestamp_diff
    
    def verify_signature(self, received_signature: str, 
                        computed_signature: str,
                        timestamp: int) -> bool:
        # Verify timestamp freshness
        current_time = int(time.time())
        if abs(current_time - timestamp) > self.max_timestamp_diff:
            return False
            
        # Compare signatures using constant-time comparison
        return hmac.compare_digest(
            received_signature.encode('utf-8'),
            computed_signature.encode('utf-8')
        )
```

Slide 8: API Response Security Headers

Security headers are crucial for protecting API responses from various attacks. This implementation adds essential headers to prevent XSS, clickjacking, and other common web vulnerabilities.

```python
class SecurityHeadersMiddleware:
    def __init__(self, api_version: str):
        self.api_version = api_version
        
    def add_security_headers(self, response: dict) -> dict:
        headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'none'",
            'X-API-Version': self.api_version
        }
        response['headers'] = headers
        return response
```

Slide 9: Rate Limiting Implementation

Rate limiting protects APIs from abuse by controlling request frequency. This implementation uses a sliding window algorithm and supports multiple rate limit tiers based on authentication type.

```python
from collections import deque
from dataclasses import dataclass
from time import time

@dataclass
class RateLimitConfig:
    requests: int
    window: int  # seconds

class RateLimiter:
    def __init__(self, config: Dict[str, RateLimitConfig]):
        self.config = config
        self.requests = {}  # client_id -> deque of timestamps
    
    def is_allowed(self, client_id: str, auth_type: str) -> bool:
        if client_id not in self.requests:
            self.requests[client_id] = deque()
            
        current_time = time()
        config = self.config[auth_type]
        
        # Remove old requests
        while (self.requests[client_id] and 
               current_time - self.requests[client_id][0] > config.window):
            self.requests[client_id].popleft()
        
        # Check rate limit
        if len(self.requests[client_id]) >= config.requests:
            return False
            
        self.requests[client_id].append(current_time)
        return True
```

Slide 10: Secure Password Storage for Token Authentication

Implementing secure password storage is essential for token-based authentication. This implementation uses Argon2 for password hashing and includes salt generation and verification.

```python
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
import secrets

class PasswordManager:
    def __init__(self):
        self.ph = PasswordHasher(
            time_cost=2,      # Number of iterations
            memory_cost=65536, # Memory usage in KiB
            parallelism=4,    # Number of parallel threads
            hash_len=32,      # Length of the hash in bytes
            salt_len=16       # Length of the salt in bytes
        )
    
    def hash_password(self, password: str) -> str:
        return self.ph.hash(password)
    
    def verify_password(self, stored_hash: str, password: str) -> bool:
        try:
            return self.ph.verify(stored_hash, password)
        except VerifyMismatchError:
            return False
```

Slide 11: Real-World Implementation - E-commerce API Authentication

This implementation showcases a complete e-commerce API authentication system combining both token and HMAC authentication methods for different endpoints, with practical error handling and logging.

```python
import logging
from typing import Optional, Tuple
from datetime import datetime

class ECommerceAPIAuth:
    def __init__(self):
        self.logger = logging.getLogger('ecommerce_auth')
        self.token_auth = TokenAuth()
        self.hmac_auth = HMACAuth(api_key="your_secret_key", app_id="store_frontend")
        self.rate_limiter = RateLimiter({
            'token': RateLimitConfig(1000, 3600),
            'hmac': RateLimitConfig(10000, 3600)
        })
    
    def authenticate_request(self, 
                           endpoint: str,
                           method: str,
                           headers: Dict[str, str],
                           body: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        try:
            client_id = headers.get('X-Client-ID')
            if not client_id:
                return False, "Missing client ID"

            if not self.rate_limiter.is_allowed(client_id, 
                                              headers.get('Auth-Type', 'token')):
                return False, "Rate limit exceeded"

            if endpoint.startswith('/api/v1/orders'):
                return self._authenticate_order_request(headers, body)
            elif endpoint.startswith('/api/v1/products'):
                return self._authenticate_product_request(headers)
            
            return False, "Invalid endpoint"
            
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
            return False, "Internal authentication error"
```

Slide 12: Source Code for E-commerce API Authentication

```python
def _authenticate_order_request(self, headers: Dict[str, str], 
                              body: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    # Orders always require HMAC authentication for better security
    if headers.get('Auth-Type') != 'hmac':
        return False, "Orders require HMAC authentication"
    
    app_id = headers.get('X-App-ID')
    timestamp = headers.get('X-Timestamp')
    received_signature = headers.get('X-Signature')
    
    if not all([app_id, timestamp, received_signature]):
        return False, "Missing HMAC authentication parameters"
    
    # Prepare parameters for signature verification
    params = {
        'app_id': app_id,
        'timestamp': timestamp,
        'total_amount': str(body.get('total_amount', '0')),
        'order_items': ','.join(str(item['id']) for item in body.get('items', []))
    }
    
    computed_signature = self.hmac_auth.generate_signature(params)
    
    if not self.hmac_auth.verify_signature(received_signature, 
                                         computed_signature,
                                         int(timestamp)):
        self.logger.warning(f"Invalid HMAC signature for order from app_id: {app_id}")
        return False, "Invalid signature"
    
    return True, None

def _authenticate_product_request(self, headers: Dict[str, str]) -> Tuple[bool, Optional[str]]:
    # Product endpoints accept both token and HMAC
    auth_type = headers.get('Auth-Type')
    
    if auth_type == 'token':
        token = headers.get('Authorization')
        if not token:
            return False, "Missing token"
        
        if not self.token_auth.validate_token(token):
            return False, "Invalid or expired token"
            
    elif auth_type == 'hmac':
        # Similar HMAC verification as order requests but with different parameters
        return self._verify_hmac_product_request(headers)
    else:
        return False, "Invalid authentication type"
    
    return True, None
```

Slide 13: Advanced Security Features Implementation

This implementation adds advanced security features including request replay protection, IP-based rate limiting, and automated security scanning for suspicious patterns.

```python
class AdvancedSecurityManager:
    def __init__(self):
        self.nonce_store = set()  # Store used nonces
        self.ip_tracker = defaultdict(list)  # Track IP request patterns
        self.suspicious_patterns = compile_security_patterns()
        
    def validate_request_security(self, request: Dict[str, Any], 
                                client_ip: str) -> Tuple[bool, Optional[str]]:
        # Check for replay attacks using nonce
        nonce = request.headers.get('X-Nonce')
        if not nonce or nonce in self.nonce_store:
            return False, "Invalid or reused nonce"
        self.nonce_store.add(nonce)
        
        # Implement IP-based rate limiting with behavioral analysis
        if not self._validate_ip_behavior(client_ip, request):
            return False, "Suspicious IP behavior detected"
        
        # Scan for malicious patterns
        if self._detect_security_threats(request):
            return False, "Security threat detected"
        
        return True, None

    def _validate_ip_behavior(self, ip: str, request: Dict[str, Any]) -> bool:
        current_time = time.time()
        self.ip_tracker[ip].append(current_time)
        
        # Remove old requests
        self.ip_tracker[ip] = [t for t in self.ip_tracker[ip] 
                              if current_time - t < 3600]
        
        # Analyze request patterns
        return self._analyze_request_patterns(ip)
```

Slide 14: Results Analysis and Monitoring

This implementation provides comprehensive monitoring and analysis capabilities for API authentication events, helping detect security issues and maintain system health through metrics collection.

```python
from dataclasses import dataclass
from collections import Counter
import json
from typing import List, Dict

@dataclass
class AuthEvent:
    timestamp: float
    auth_type: str
    success: bool
    client_id: str
    ip_address: str
    endpoint: str
    error_message: Optional[str] = None

class AuthenticationMonitor:
    def __init__(self):
        self.events: List[AuthEvent] = []
        self.alert_threshold = 10  # Failed attempts threshold
        
    def record_event(self, event: AuthEvent) -> None:
        self.events.append(event)
        self._analyze_security_patterns()
        
    def _analyze_security_patterns(self) -> None:
        recent_events = [e for e in self.events 
                        if time.time() - e.timestamp < 300]  # Last 5 minutes
        
        # Analyze failed attempts per IP
        failed_by_ip = Counter(
            e.ip_address for e in recent_events 
            if not e.success
        )
        
        # Check for potential attacks
        for ip, count in failed_by_ip.items():
            if count >= self.alert_threshold:
                self._trigger_security_alert({
                    'type': 'excessive_failures',
                    'ip_address': ip,
                    'count': count,
                    'timestamp': time.time()
                })
```

Slide 15: Source Code for Results Analysis and Monitoring

```python
def generate_security_report(self) -> Dict[str, Any]:
    current_time = time.time()
    window = 3600  # Last hour
    recent_events = [e for e in self.events 
                    if current_time - e.timestamp < window]
    
    success_rate = len([e for e in recent_events if e.success]) / \
                  len(recent_events) if recent_events else 0
    
    auth_type_stats = Counter(e.auth_type for e in recent_events)
    endpoint_stats = Counter(e.endpoint for e in recent_events)
    
    return {
        'timestamp': current_time,
        'total_requests': len(recent_events),
        'success_rate': success_rate,
        'auth_type_distribution': dict(auth_type_stats),
        'endpoint_distribution': dict(endpoint_stats),
        'recent_failures': [
            {
                'timestamp': e.timestamp,
                'client_id': e.client_id,
                'ip_address': e.ip_address,
                'error': e.error_message
            }
            for e in recent_events if not e.success
        ][-10:]  # Last 10 failures
    }

def _trigger_security_alert(self, alert_data: Dict[str, Any]) -> None:
    alert_json = json.dumps(alert_data, indent=2)
    logging.warning(f"Security Alert:\n{alert_json}")
    
    # Example webhook implementation
    webhook_url = os.getenv('SECURITY_WEBHOOK_URL')
    if webhook_url:
        requests.post(webhook_url, json=alert_data, timeout=5)
```

Slide 16: Additional Resources

*   Complex API Authorization Using HMAC Authentication: [https://arxiv.org/abs/2203.08215](https://arxiv.org/abs/2203.08215)
*   Security Analysis of Token-based Authentication in Web APIs: [https://arxiv.org/abs/2104.12279](https://arxiv.org/abs/2104.12279)
*   Preventing API Abuse: Advanced Rate Limiting Strategies: [https://arxiv.org/abs/2201.09177](https://arxiv.org/abs/2201.09177)
*   Cryptographic Authentication Methods for RESTful APIs: [https://arxiv.org/abs/2202.11753](https://arxiv.org/abs/2202.11753)
*   Real-time Threat Detection in API Authentication Systems: [https://arxiv.org/abs/2205.14332](https://arxiv.org/abs/2205.14332)

