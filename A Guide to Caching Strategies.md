## A Guide to Caching Strategies
Slide 1: Cache Through Implementation - Read Pattern

A cache-through read pattern ensures data consistency by first checking the cache for requested data. If not found, it retrieves from the database and updates the cache atomically, providing subsequent reads with cached data while maintaining synchronization between storage layers.

```python
from typing import Any, Optional
import redis
import pymongo
from datetime import timedelta

class CacheThroughRead:
    def __init__(self, redis_host: str, mongo_uri: str):
        self.cache = redis.Redis(host=redis_host, decode_responses=True)
        self.db = pymongo.MongoClient(mongo_uri).get_database('products')
        self.cache_ttl = timedelta(hours=1)
    
    def get_product(self, product_id: str) -> Optional[dict]:
        # Check cache first
        cached_data = self.cache.get(f"product:{product_id}")
        if cached_data:
            return eval(cached_data)  # Convert string to dict
        
        # If not in cache, get from database
        product = self.db.products.find_one({"_id": product_id})
        if product:
            # Update cache with new data
            self.cache.setex(
                f"product:{product_id}",
                self.cache_ttl,
                str(product)
            )
        return product

# Usage Example
cache_through = CacheThroughRead(redis_host='localhost', mongo_uri='mongodb://localhost:27017/')
product = cache_through.get_product("12345")
```

Slide 2: Cache Through Implementation - Write Pattern

The write-through caching pattern maintains strong consistency by simultaneously updating both cache and database. This implementation ensures data integrity but may introduce higher latency due to the requirement of successful writes to both systems before confirming the operation.

```python
from typing import Dict, Any
import redis
import pymongo
from datetime import timedelta

class CacheThroughWrite:
    def __init__(self, redis_host: str, mongo_uri: str):
        self.cache = redis.Redis(host=redis_host, decode_responses=True)
        self.db = pymongo.MongoClient(mongo_uri).get_database('products')
        self.cache_ttl = timedelta(hours=1)
    
    def update_product(self, product_id: str, data: Dict[str, Any]) -> bool:
        try:
            # Update database first
            result = self.db.products.update_one(
                {"_id": product_id},
                {"$set": data},
                upsert=True
            )
            
            # If database update successful, update cache
            if result.modified_count > 0 or result.upserted_id:
                self.cache.setex(
                    f"product:{product_id}",
                    self.cache_ttl,
                    str(data)
                )
                return True
            return False
            
        except Exception as e:
            print(f"Error updating product: {e}")
            return False

# Usage Example
cache_through = CacheThroughWrite(redis_host='localhost', mongo_uri='mongodb://localhost:27017/')
success = cache_through.update_product("12345", {"name": "New Product", "price": 99.99})
```

Slide 3: Cache Aside Implementation - Read Pattern

Cache-aside reading implements a lazy loading strategy where the application first checks the cache, and only queries the database on cache misses. This approach optimizes for frequently accessed data while minimizing unnecessary cache population.

```python
import redis
import pymongo
from typing import Optional, Dict, Any
from datetime import timedelta

class CacheAsideRead:
    def __init__(self, redis_host: str, mongo_uri: str):
        self.cache = redis.Redis(host=redis_host, decode_responses=True)
        self.db = pymongo.MongoClient(mongo_uri).get_database('products')
        self.cache_ttl = timedelta(hours=1)
    
    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        # Try to get from cache first
        cache_key = f"product:{product_id}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return eval(cached_data)
        
        # On cache miss, get from database
        product = self.db.products.find_one({"_id": product_id})
        
        if product:
            # Populate cache with found data
            self.cache.setex(
                cache_key,
                self.cache_ttl,
                str(product)
            )
            
        return product

# Usage Example with performance tracking
import time

cache_aside = CacheAsideRead(redis_host='localhost', mongo_uri='mongodb://localhost:27017/')

start_time = time.time()
product = cache_aside.get_product("12345")  # First access (cache miss)
first_access = time.time() - start_time

start_time = time.time()
product = cache_aside.get_product("12345")  # Second access (cache hit)
second_access = time.time() - start_time

print(f"Cache miss time: {first_access:.4f}s")
print(f"Cache hit time: {second_access:.4f}s")
```

Slide 4: Cache Aside Implementation - Write Pattern

The cache-aside write pattern focuses on database updates while managing cache invalidation. This implementation provides better write performance by updating the database directly and either invalidating or updating the cache, preventing stale data scenarios.

```python
from typing import Dict, Any
import redis
import pymongo
from datetime import timedelta

class CacheAsideWrite:
    def __init__(self, redis_host: str, mongo_uri: str):
        self.cache = redis.Redis(host=redis_host, decode_responses=True)
        self.db = pymongo.MongoClient(mongo_uri).get_database('products')
        self.cache_ttl = timedelta(hours=1)
    
    def update_product(self, product_id: str, data: Dict[str, Any]) -> bool:
        try:
            # Update database
            result = self.db.products.update_one(
                {"_id": product_id},
                {"$set": data},
                upsert=True
            )
            
            # If database update successful, invalidate cache
            if result.modified_count > 0 or result.upserted_id:
                self.cache.delete(f"product:{product_id}")
                return True
                
            return False
            
        except Exception as e:
            print(f"Error updating product: {e}")
            return False
    
    def write_through_option(self, product_id: str, data: Dict[str, Any]) -> bool:
        """Alternative approach: Update cache instead of invalidating"""
        try:
            result = self.db.products.update_one(
                {"_id": product_id},
                {"$set": data},
                upsert=True
            )
            
            if result.modified_count > 0 or result.upserted_id:
                self.cache.setex(
                    f"product:{product_id}",
                    self.cache_ttl,
                    str(data)
                )
                return True
                
            return False
            
        except Exception as e:
            print(f"Error updating product: {e}")
            return False

# Usage Example
cache_aside = CacheAsideWrite(redis_host='localhost', mongo_uri='mongodb://localhost:27017/')
success = cache_aside.update_product("12345", {"name": "Updated Product", "price": 199.99})
```

Slide 5: Cache Ahead Implementation - Predictive Loading

Cache-ahead implements predictive data loading based on access patterns. This implementation uses a background task to analyze access patterns and preload frequently accessed data, optimizing read performance for anticipated requests.

```python
import redis
import pymongo
from typing import List, Dict, Any
import asyncio
from datetime import datetime, timedelta
import logging

class CacheAhead:
    def __init__(self, redis_host: str, mongo_uri: str):
        self.cache = redis.Redis(host=redis_host, decode_responses=True)
        self.db = pymongo.MongoClient(mongo_uri).get_database('products')
        self.cache_ttl = timedelta(hours=1)
        self.access_threshold = 5
        
    async def monitor_access_patterns(self):
        """Background task to monitor product access patterns"""
        while True:
            access_counts = self.cache.hgetall("access_counts")
            
            for product_id, count in access_counts.items():
                if int(count) >= self.access_threshold:
                    await self.preload_product(product_id.split(':')[1])
                    
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def preload_product(self, product_id: str):
        """Preload product and related items"""
        product = self.db.products.find_one({"_id": product_id})
        if product:
            # Cache main product
            self.cache.setex(
                f"product:{product_id}",
                self.cache_ttl,
                str(product)
            )
            
            # Preload related products
            related_products = self.db.products.find({
                "category": product["category"],
                "_id": {"$ne": product_id}
            }).limit(5)
            
            for related in related_products:
                self.cache.setex(
                    f"product:{related['_id']}",
                    self.cache_ttl,
                    str(related)
                )
    
    def record_access(self, product_id: str):
        """Record product access for pattern analysis"""
        self.cache.hincrby("access_counts", f"product:{product_id}", 1)

# Usage Example
cache_ahead = CacheAhead(redis_host='localhost', mongo_uri='mongodb://localhost:27017/')

# Start monitoring in background
async def main():
    await cache_ahead.monitor_access_patterns()

# Record some accesses
cache_ahead.record_access("12345")
```

Slide 6: Cache Behind Implementation - Write Deferral

Cache-behind strategy implements asynchronous write operations by immediately updating the cache and queuing database updates for batch processing. This implementation optimizes write performance while managing potential data consistency challenges through careful error handling.

```python
import redis
import pymongo
from typing import Dict, Any, List
import asyncio
from datetime import datetime
import json
from collections import defaultdict

class CacheBehind:
    def __init__(self, redis_host: str, mongo_uri: str):
        self.cache = redis.Redis(host=redis_host, decode_responses=True)
        self.db = pymongo.MongoClient(mongo_uri).get_database('products')
        self.write_queue = defaultdict(list)
        self.batch_size = 50
        self.flush_interval = 60  # seconds
        
    async def write_product(self, product_id: str, data: Dict[str, Any]) -> bool:
        try:
            # Immediately update cache
            self.cache.set(f"product:{product_id}", json.dumps(data))
            
            # Queue database update
            self.write_queue[product_id].append({
                'data': data,
                'timestamp': datetime.utcnow().timestamp()
            })
            
            # Trigger flush if queue is large enough
            if len(self.write_queue) >= self.batch_size:
                await self.flush_writes()
            
            return True
            
        except Exception as e:
            print(f"Error in write operation: {e}")
            return False
    
    async def flush_writes(self):
        """Batch process queued writes to database"""
        try:
            bulk_operations = []
            
            for product_id, updates in self.write_queue.items():
                # Take the latest update for each product
                latest_update = max(updates, key=lambda x: x['timestamp'])
                
                bulk_operations.append(
                    pymongo.UpdateOne(
                        {"_id": product_id},
                        {"$set": latest_update['data']},
                        upsert=True
                    )
                )
            
            if bulk_operations:
                result = self.db.products.bulk_write(bulk_operations)
                self.write_queue.clear()
                return result.modified_count
                
        except Exception as e:
            print(f"Error in flush operation: {e}")
            return 0
    
    async def background_flush(self):
        """Periodic flush of queued writes"""
        while True:
            await asyncio.sleep(self.flush_interval)
            await self.flush_writes()

# Usage Example
async def main():
    cache_behind = CacheBehind(redis_host='localhost', mongo_uri='mongodb://localhost:27017/')
    
    # Start background flush task
    asyncio.create_task(cache_behind.background_flush())
    
    # Example writes
    updates = [
        ("12345", {"name": "Product A", "price": 99.99}),
        ("67890", {"name": "Product B", "price": 149.99}),
    ]
    
    for product_id, data in updates:
        await cache_behind.write_product(product_id, data)

# Run example
asyncio.run(main())
```

Slide 7: Distributed Cache Implementation

Implementation of a distributed caching system using Redis Cluster for horizontal scalability. This approach ensures high availability and fault tolerance while maintaining consistent hashing for data distribution across multiple cache nodes.

```python
from redis.cluster import RedisCluster
from typing import Optional, Dict, Any
import json
import hashlib
from datetime import timedelta

class DistributedCache:
    def __init__(self, startup_nodes: list, retry_max: int = 3):
        self.cache = RedisCluster(
            startup_nodes=startup_nodes,
            decode_responses=True,
            skip_full_coverage_check=True,
            retry_on_timeout=True,
            max_retry=retry_max
        )
        self.ttl = timedelta(hours=1)
    
    def _get_shard_key(self, key: str) -> str:
        """Generate consistent hash for key distribution"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def set_value(self, key: str, value: Any) -> bool:
        try:
            shard_key = self._get_shard_key(key)
            return self.cache.setex(
                shard_key,
                self.ttl,
                json.dumps(value)
            )
        except Exception as e:
            print(f"Error setting value: {e}")
            return False
    
    def get_value(self, key: str) -> Optional[Any]:
        try:
            shard_key = self._get_shard_key(key)
            value = self.cache.get(shard_key)
            return json.loads(value) if value else None
        except Exception as e:
            print(f"Error getting value: {e}")
            return None
    
    def delete_value(self, key: str) -> bool:
        try:
            shard_key = self._get_shard_key(key)
            return bool(self.cache.delete(shard_key))
        except Exception as e:
            print(f"Error deleting value: {e}")
            return False
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about cluster nodes and slots"""
        try:
            nodes = self.cache.cluster_nodes()
            slots = self.cache.cluster_slots()
            return {
                "nodes": len(nodes),
                "slots": slots,
                "keyspace": self.cache.info("keyspace")
            }
        except Exception as e:
            print(f"Error getting cluster info: {e}")
            return {}

# Usage Example
startup_nodes = [
    {"host": "127.0.0.1", "port": "7000"},
    {"host": "127.0.0.1", "port": "7001"},
    {"host": "127.0.0.1", "port": "7002"}
]

cache = DistributedCache(startup_nodes)

# Store and retrieve data
cache.set_value("user:123", {"name": "John", "age": 30})
user = cache.get_value("user:123")

# Get cluster information
cluster_info = cache.get_cluster_info()
print(f"Cluster Status: {json.dumps(cluster_info, indent=2)}")
```

Slide 8: Cache Eviction Strategy Implementation

A sophisticated cache eviction strategy implementation that combines LRU (Least Recently Used) with TTL (Time To Live) and memory pressure monitoring. This approach optimizes cache utilization while preventing memory overflow.

```python
from typing import Dict, Any, Optional
import time
from collections import OrderedDict
import threading
import psutil

class AdvancedCache:
    def __init__(self, max_size: int = 1000, max_memory_percent: float = 75.0):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._max_memory_percent = max_memory_percent
        self._lock = threading.Lock()
        
        # Start memory monitor
        self._monitor_thread = threading.Thread(
            target=self._monitor_memory,
            daemon=True
        )
        self._monitor_thread.start()
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        with self._lock:
            try:
                # Check if we need to evict
                self._evict_if_needed()
                
                # Store value with metadata
                self._cache[key] = {
                    'value': value,
                    'accessed_at': time.time(),
                    'expires_at': time.time() + ttl if ttl else None
                }
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return True
                
            except Exception as e:
                print(f"Error setting value: {e}")
                return False
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
                
            item = self._cache[key]
            current_time = time.time()
            
            # Check if expired
            if item['expires_at'] and current_time > item['expires_at']:
                del self._cache[key]
                return None
            
            # Update access time and move to end
            item['accessed_at'] = current_time
            self._cache.move_to_end(key)
            
            return item['value']
    
    def _evict_if_needed(self):
        """Evict items based on size, memory pressure, and expiration"""
        current_time = time.time()
        
        # Remove expired items
        expired_keys = [
            k for k, v in self._cache.items()
            if v['expires_at'] and current_time > v['expires_at']
        ]
        for key in expired_keys:
            del self._cache[key]
        
        # Evict based on size
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)  # Remove least recently used
    
    def _monitor_memory(self):
        """Monitor system memory usage and evict if needed"""
        while True:
            memory_percent = psutil.Process().memory_percent()
            
            if memory_percent > self._max_memory_percent:
                with self._lock:
                    # Remove 25% of items when memory pressure is high
                    items_to_remove = len(self._cache) // 4
                    for _ in range(items_to_remove):
                        if self._cache:
                            self._cache.popitem(last=False)
            
            time.sleep(5)  # Check every 5 seconds
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'memory_usage': psutil.Process().memory_percent(),
                'max_memory_percent': self._max_memory_percent
            }

# Usage Example
cache = AdvancedCache(max_size=100, max_memory_percent=75.0)

# Set values with different TTLs
cache.set("key1", "value1", ttl=60)  # Expires in 60 seconds
cache.set("key2", "value2")  # No expiration

# Get statistics
stats = cache.get_stats()
print(f"Cache Stats: {stats}")
```

Slide 9: Real-World Example - E-commerce Product Cache

Implementation of a comprehensive product caching system for an e-commerce platform. This system handles product details, inventory updates, and price changes while maintaining cache consistency across distributed systems.

```python
import redis
import pymongo
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime, timedelta

class EcommerceCache:
    def __init__(self, redis_host: str, mongo_uri: str):
        self.cache = redis.Redis(
            host=redis_host,
            decode_responses=True,
            socket_timeout=5
        )
        self.db = pymongo.MongoClient(mongo_uri).get_database('ecommerce')
        self.cache_ttl = timedelta(hours=24)
        
    async def get_product_details(self, product_id: str) -> Optional[Dict[str, Any]]:
        cache_key = f"product:details:{product_id}"
        
        # Try cache first
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
        
        # Cache miss - get from database
        product = self.db.products.find_one({"_id": product_id})
        if product:
            # Add real-time inventory check
            inventory = await self._get_inventory(product_id)
            product['current_stock'] = inventory
            
            # Cache the complete data
            self.cache.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(product)
            )
            
            return product
        return None
    
    async def update_product_price(self, product_id: str, new_price: float) -> bool:
        try:
            # Update database
            result = self.db.products.update_one(
                {"_id": product_id},
                {"$set": {
                    "price": new_price,
                    "price_updated_at": datetime.utcnow()
                }}
            )
            
            if result.modified_count > 0:
                # Invalidate cache
                self.cache.delete(f"product:details:{product_id}")
                
                # Add to price update stream for other services
                self.cache.xadd(
                    "price_updates",
                    {
                        "product_id": product_id,
                        "price": str(new_price),
                        "timestamp": str(time.time())
                    }
                )
                return True
            return False
            
        except Exception as e:
            print(f"Error updating price: {e}")
            return False
    
    async def _get_inventory(self, product_id: str) -> int:
        """Get real-time inventory from warehouse system"""
        inventory_key = f"inventory:{product_id}"
        
        cached_inventory = self.cache.get(inventory_key)
        if cached_inventory:
            return int(cached_inventory)
            
        # Simulate warehouse system call
        inventory = self.db.inventory.find_one(
            {"product_id": product_id},
            {"quantity": 1}
        )
        
        if inventory:
            # Cache inventory with short TTL
            self.cache.setex(
                inventory_key,
                timedelta(minutes=5),  # Short TTL for inventory
                str(inventory['quantity'])
            )
            return inventory['quantity']
        return 0
    
    async def get_category_products(self, category: str, page: int = 1, per_page: int = 20) -> List[Dict[str, Any]]:
        """Get cached category product listings with pagination"""
        cache_key = f"category:{category}:page:{page}"
        
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
        
        # Get from database with pagination
        skip = (page - 1) * per_page
        products = list(self.db.products.find(
            {"category": category},
            {"_id": 1, "name": 1, "price": 1, "thumbnail": 1}
        ).skip(skip).limit(per_page))
        
        if products:
            self.cache.setex(
                cache_key,
                timedelta(hours=1),  # Shorter TTL for listings
                json.dumps(products)
            )
        
        return products

# Usage Example
async def main():
    cache = EcommerceCache(
        redis_host='localhost',
        mongo_uri='mongodb://localhost:27017/'
    )
    
    # Get product details
    product = await cache.get_product_details("PROD123")
    
    # Update price
    success = await cache.update_product_price("PROD123", 99.99)
    
    # Get category listings
    electronics = await cache.get_category_products("electronics", page=1)
    
    print(f"Product Details: {product}")
    print(f"Price Update Success: {success}")
    print(f"Category Products: {len(electronics)} items")

# Run example
import asyncio
asyncio.run(main())
```

Slide 10: Real-World Example - User Session Cache

Implementation of a high-performance user session caching system handling authentication, permissions, and user preferences with automatic synchronization across multiple application instances.

```python
import redis
from typing import Dict, Any, Optional, List
import json
import jwt
import hashlib
from datetime import datetime, timedelta
import asyncio

class UserSessionCache:
    def __init__(self, redis_host: str, secret_key: str):
        self.cache = redis.Redis(
            host=redis_host,
            decode_responses=True
        )
        self.secret_key = secret_key
        self.session_ttl = timedelta(hours=24)
        self.permission_ttl = timedelta(minutes=30)
    
    def create_session(self, user_id: str, user_data: Dict[str, Any]) -> Optional[str]:
        try:
            # Generate session token
            session_token = jwt.encode(
                {
                    'user_id': user_id,
                    'created_at': datetime.utcnow().timestamp(),
                    'exp': (datetime.utcnow() + self.session_ttl).timestamp()
                },
                self.secret_key,
                algorithm='HS256'
            )
            
            # Store session data
            session_key = f"session:{user_id}"
            session_data = {
                'token': session_token,
                'user_data': user_data,
                'last_access': datetime.utcnow().timestamp()
            }
            
            self.cache.setex(
                session_key,
                self.session_ttl,
                json.dumps(session_data)
            )
            
            # Store token mapping for validation
            self.cache.setex(
                f"token:{session_token}",
                self.session_ttl,
                user_id
            )
            
            return session_token
            
        except Exception as e:
            print(f"Error creating session: {e}")
            return None
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        try:
            # Verify token signature
            payload = jwt.decode(
                session_token,
                self.secret_key,
                algorithms=['HS256']
            )
            
            user_id = payload.get('user_id')
            if not user_id:
                return None
            
            # Check if token is still valid in cache
            token_key = f"token:{session_token}"
            if not self.cache.exists(token_key):
                return None
            
            # Get session data
            session_key = f"session:{user_id}"
            session_data = self.cache.get(session_key)
            
            if session_data:
                data = json.loads(session_data)
                
                # Update last access time
                data['last_access'] = datetime.utcnow().timestamp()
                self.cache.setex(
                    session_key,
                    self.session_ttl,
                    json.dumps(data)
                )
                
                return data['user_data']
                
            return None
            
        except jwt.ExpiredSignatureError:
            self.invalidate_session(session_token)
            return None
        except Exception as e:
            print(f"Error validating session: {e}")
            return None
    
    def invalidate_session(self, session_token: str) -> bool:
        try:
            # Get user_id from token mapping
            token_key = f"token:{session_token}"
            user_id = self.cache.get(token_key)
            
            if user_id:
                # Remove session and token
                self.cache.delete(f"session:{user_id}")
                self.cache.delete(token_key)
                return True
            return False
            
        except Exception as e:
            print(f"Error invalidating session: {e}")
            return False
    
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """Get cached user permissions with automatic refresh"""
        permissions_key = f"permissions:{user_id}"
        
        # Try cache first
        cached_permissions = self.cache.get(permissions_key)
        if cached_permissions:
            return json.loads(cached_permissions)
        
        # Simulate permission fetch from auth service
        permissions = await self._fetch_user_permissions(user_id)
        
        if permissions:
            self.cache.setex(
                permissions_key,
                self.permission_ttl,
                json.dumps(permissions)
            )
        
        return permissions
    
    async def _fetch_user_permissions(self, user_id: str) -> List[str]:
        """Simulate permission fetch from auth service"""
        # In real implementation, this would call your auth service
        await asyncio.sleep(0.1)  # Simulate network delay
        return ["read", "write", "delete"]

# Usage Example
async def main():
    session_cache = UserSessionCache(
        redis_host='localhost',
        secret_key='your-secret-key'
    )
    
    # Create session
    user_data = {
        "id": "123",
        "name": "John Doe",
        "email": "john@example.com"
    }
    
    session_token = session_cache.create_session("123", user_data)
    
    # Validate session
    session = session_cache.validate_session(session_token)
    
    # Get permissions
    permissions = await session_cache.get_user_permissions("123")
    
    print(f"Session Token: {session_token}")
    print(f"Validated Session: {session}")
    print(f"User Permissions: {permissions}")

# Run example
asyncio.run(main())
```

Slide 11: Performance Analysis and Monitoring

Implementation of a comprehensive cache monitoring system that tracks hit rates, latency, memory usage, and eviction patterns. This system helps identify cache efficiency and potential bottlenecks in real-time.

```python
import redis
import time
from typing import Dict, Any, List
from collections import deque
import threading
import statistics
import psutil
from datetime import datetime, timedelta

class CacheMonitor:
    def __init__(self, redis_host: str):
        self.cache = redis.Redis(host=redis_host, decode_responses=True)
        self.metrics: Dict[str, deque] = {
            'hit_rate': deque(maxlen=1000),
            'latency': deque(maxlen=1000),
            'memory_usage': deque(maxlen=60),
            'evictions': deque(maxlen=1000)
        }
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start background monitoring threads"""
        threading.Thread(
            target=self._monitor_memory,
            daemon=True
        ).start()
        
        threading.Thread(
            target=self._monitor_evictions,
            daemon=True
        ).start()
    
    def record_operation(self, operation_type: str, hit: bool, latency: float):
        """Record cache operation metrics"""
        self.metrics['hit_rate'].append(1 if hit else 0)
        self.metrics['latency'].append(latency)
        
        # Store detailed metrics in Redis
        timestamp = datetime.utcnow().timestamp()
        self.cache.xadd(
            'cache_metrics',
            {
                'operation': operation_type,
                'hit': str(hit),
                'latency': str(latency),
                'timestamp': str(timestamp)
            },
            maxlen=10000
        )
    
    def _monitor_memory(self):
        """Monitor Redis memory usage"""
        while True:
            try:
                info = self.cache.info()
                used_memory = int(info['used_memory']) / 1024 / 1024  # MB
                self.metrics['memory_usage'].append(used_memory)
                
                # Store memory metrics
                self.cache.xadd(
                    'memory_metrics',
                    {
                        'used_memory_mb': str(used_memory),
                        'timestamp': str(datetime.utcnow().timestamp())
                    },
                    maxlen=1000
                )
                
            except Exception as e:
                print(f"Error monitoring memory: {e}")
            
            time.sleep(60)  # Check every minute
    
    def _monitor_evictions(self):
        """Monitor cache evictions"""
        last_evictions = 0
        while True:
            try:
                info = self.cache.info()
                current_evictions = int(info['evicted_keys'])
                new_evictions = current_evictions - last_evictions
                
                if new_evictions > 0:
                    self.metrics['evictions'].append(new_evictions)
                    
                    # Store eviction metrics
                    self.cache.xadd(
                        'eviction_metrics',
                        {
                            'count': str(new_evictions),
                            'timestamp': str(datetime.utcnow().timestamp())
                        },
                        maxlen=1000
                    )
                
                last_evictions = current_evictions
                
            except Exception as e:
                print(f"Error monitoring evictions: {e}")
            
            time.sleep(5)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current cache performance metrics"""
        try:
            hit_rate = statistics.mean(self.metrics['hit_rate']) * 100
            avg_latency = statistics.mean(self.metrics['latency']) * 1000  # ms
            memory_usage = self.metrics['memory_usage'][-1] if self.metrics['memory_usage'] else 0
            eviction_rate = sum(self.metrics['evictions']) / len(self.metrics['evictions']) if self.metrics['evictions'] else 0
            
            return {
                'hit_rate_percent': round(hit_rate, 2),
                'avg_latency_ms': round(avg_latency, 2),
                'memory_usage_mb': round(memory_usage, 2),
                'eviction_rate': round(eviction_rate, 2),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {}
    
    def get_historical_metrics(self, 
                             metric_type: str, 
                             start_time: float,
                             end_time: float) -> List[Dict[str, Any]]:
        """Get historical metrics from Redis streams"""
        try:
            stream_name = f"{metric_type}_metrics"
            
            # Get metrics within time range
            results = self.cache.xrange(
                stream_name,
                min=start_time,
                max=end_time
            )
            
            return [
                {
                    'timestamp': float(item[1]['timestamp']),
                    'value': float(list(item[1].values())[0])
                }
                for item in results
            ]
            
        except Exception as e:
            print(f"Error getting historical metrics: {e}")
            return []

# Usage Example
def monitor_cache_performance():
    monitor = CacheMonitor(redis_host='localhost')
    
    # Simulate some cache operations
    for _ in range(100):
        start_time = time.time()
        hit = bool(time.time() % 2)  # Simulate hits/misses
        latency = time.time() - start_time
        
        monitor.record_operation('get', hit, latency)
        time.sleep(0.1)
    
    # Get current metrics
    metrics = monitor.get_current_metrics()
    print("Current Cache Metrics:")
    print(json.dumps(metrics, indent=2))
    
    # Get historical metrics
    end_time = time.time()
    start_time = end_time - 3600  # Last hour
    
    history = monitor.get_historical_metrics(
        'cache_metrics',
        start_time,
        end_time
    )
    print(f"\nHistorical Metrics (last hour): {len(history)} entries")

if __name__ == "__main__":
    monitor_cache_performance()
```

Slide 12: Cache Consistency Patterns

Implementation of different cache consistency patterns including Read-Through, Write-Through, and Write-Behind, with mechanisms to handle network partitions and eventual consistency.

```python
import redis
import pymongo
from typing import Dict, Any, Optional, List
import json
import asyncio
from datetime import datetime, timedelta
from enum import Enum

class ConsistencyPattern(Enum):
    READ_THROUGH = "read_through"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"

class ConsistentCache:
    def __init__(self, 
                 redis_host: str,
                 mongo_uri: str,
                 pattern: ConsistencyPattern = ConsistencyPattern.READ_THROUGH):
        self.cache = redis.Redis(host=redis_host, decode_responses=True)
        self.db = pymongo.MongoClient(mongo_uri).get_database('application')
        self.pattern = pattern
        self.write_queue = asyncio.Queue()
        self.cache_ttl = timedelta(hours=1)
        
        if pattern == ConsistencyPattern.WRITE_BEHIND:
            asyncio.create_task(self._process_write_queue())
    
    async def get(self, key: str) -> Optional[Any]:
        try:
            # Try cache first
            cached_value = self.cache.get(key)
            
            if cached_value:
                return json.loads(cached_value)
            
            # Cache miss - implement read-through
            if self.pattern == ConsistencyPattern.READ_THROUGH:
                value = await self._read_from_db(key)
                
                if value:
                    self.cache.setex(
                        key,
                        self.cache_ttl,
                        json.dumps(value)
                    )
                return value
                
            return None
            
        except Exception as e:
            print(f"Error getting value: {e}")
            return None
    
    async def set(self, key: str, value: Any) -> bool:
        try:
            # Different write patterns
            if self.pattern == ConsistencyPattern.WRITE_THROUGH:
                # Update both cache and DB atomically
                success = await self._write_through(key, value)
                
            elif self.pattern == ConsistencyPattern.WRITE_BEHIND:
                # Update cache immediately and queue DB update
                self.cache.setex(
                    key,
                    self.cache_ttl,
                    json.dumps(value)
                )
                await self.write_queue.put({
                    'key': key,
                    'value': value,
                    'timestamp': datetime.utcnow().timestamp()
                })
                success = True
                
            else:
                # Cache aside - update DB and invalidate cache
                success = await self._write_to_db(key, value)
                if success:
                    self.cache.delete(key)
            
            return success
            
        except Exception as e:
            print(f"Error setting value: {e}")
            return False
    
    async def _read_from_db(self, key: str) -> Optional[Any]:
        """Read value from database"""
        try:
            document = self.db.data.find_one({'_id': key})
            return document['value'] if document else None
        except Exception as e:
            print(f"Error reading from database: {e}")
            return None
    
    async def _write_to_db(self, key: str, value: Any) -> bool:
        """Write value to database"""
        try:
            result = self.db.data.update_one(
                {'_id': key},
                {'$set': {
                    'value': value,
                    'updated_at': datetime.utcnow()
                }},
                upsert=True
            )
            return result.modified_count > 0 or result.upserted_id is not None
        except Exception as e:
            print(f"Error writing to database: {e}")
            return False
    
    async def _write_through(self, key: str, value: Any) -> bool:
        """Implement write-through pattern"""
        try:
            # Start database write
            db_success = await self._write_to_db(key, value)
            
            if db_success:
                # If database write succeeds, update cache
                self.cache.setex(
                    key,
                    self.cache_ttl,
                    json.dumps(value)
                )
                return True
            return False
            
        except Exception as e:
            print(f"Error in write-through: {e}")
            return False
    
    async def _process_write_queue(self):
        """Process queued writes for write-behind pattern"""
        while True:
            try:
                # Get batch of writes (up to 10)
                batch = []
                for _ in range(10):
                    try:
                        item = self.write_queue.get_nowait()
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        break
                
                if batch:
                    # Process batch
                    operations = [
                        pymongo.UpdateOne(
                            {'_id': item['key']},
                            {'$set': {
                                'value': item['value'],
                                'updated_at': datetime.fromtimestamp(item['timestamp'])
                            }},
                            upsert=True
                        )
                        for item in batch
                    ]
                    
                    result = self.db.data.bulk_write(operations)
                    print(f"Processed {len(batch)} queued writes")
                
            except Exception as e:
                print(f"Error processing write queue: {e}")
            
            await asyncio.sleep(1)

# Usage Example
async def main():
    # Initialize cache with different patterns
    read_through_cache = ConsistentCache(
        redis_host='localhost',
        mongo_uri='mongodb://localhost:27017/',
        pattern=ConsistencyPattern.READ_THROUGH
    )
    
    write_behind_cache = ConsistentCache(
        redis_host='localhost',
        mongo_uri='mongodb://localhost:27017/',
        pattern=ConsistencyPattern.WRITE_BEHIND
    )
    
    # Test read-through pattern
    value = await read_through_cache.get("test_key")
    success = await read_through_cache.set("test_key", {"data": "test"})
    
    # Test write-behind pattern
    await write_behind_cache.set("async_key", {"data": "async"})
    value = await write_behind_cache.get("async_key")
    
    print("Operations completed")

if __name__ == "__main__":
    asyncio.run(main())
```

Slide 13: Cache Network Topology Implementation

Implementation of a distributed cache network with support for replication, sharding, and failover mechanisms. This system ensures high availability and data consistency across multiple cache nodes.

```python
import redis
from redis.sentinel import Sentinel
from typing import Dict, Any, List, Optional, Set
import hashlib
import json
from datetime import datetime
import asyncio
import random

class CacheNetworkNode:
    def __init__(self, 
                 host: str, 
                 port: int,
                 role: str = 'primary',
                 shard_id: int = 0):
        self.host = host
        self.port = port
        self.role = role
        self.shard_id = shard_id
        self.connection = redis.Redis(
            host=host,
            port=port,
            decode_responses=True
        )
        
    def __str__(self) -> str:
        return f"Node({self.host}:{self.port}, {self.role}, shard={self.shard_id})"

class DistributedCacheNetwork:
    def __init__(self, 
                 sentinel_hosts: List[tuple],
                 num_shards: int = 3):
        self.sentinel = Sentinel(sentinel_hosts)
        self.num_shards = num_shards
        self.nodes: Dict[int, List[CacheNetworkNode]] = {}
        self.init_network()
    
    def init_network(self):
        """Initialize cache network topology"""
        for shard_id in range(self.num_shards):
            # Get primary and replica nodes for each shard
            primary_host, primary_port = self.sentinel.discover_master(f'shard{shard_id}')
            replica_hosts = self.sentinel.discover_slaves(f'shard{shard_id}')
            
            # Initialize nodes
            nodes = [
                CacheNetworkNode(primary_host, primary_port, 'primary', shard_id)
            ]
            
            for host, port in replica_hosts:
                nodes.append(
                    CacheNetworkNode(host, port, 'replica', shard_id)
                )
            
            self.nodes[shard_id] = nodes
    
    def _get_shard(self, key: str) -> int:
        """Determine shard for key using consistent hashing"""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_value % self.num_shards
    
    def _get_node(self, shard_id: int, role: str = 'primary') -> Optional[CacheNetworkNode]:
        """Get node from specific shard by role"""
        nodes = self.nodes.get(shard_id, [])
        for node in nodes:
            if node.role == role:
                return node
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value with replication"""
        try:
            shard_id = self._get_shard(key)
            primary_node = self._get_node(shard_id, 'primary')
            
            if not primary_node:
                return False
            
            # Set on primary
            success = primary_node.connection.setex(
                key,
                ttl,
                json.dumps(value)
            )
            
            if success:
                # Replicate to secondary nodes
                replica_nodes = [
                    node for node in self.nodes[shard_id]
                    if node.role == 'replica'
                ]
                
                replication_tasks = [
                    self._replicate_data(node, key, value, ttl)
                    for node in replica_nodes
                ]
                
                await asyncio.gather(*replication_tasks)
                
            return bool(success)
            
        except Exception as e:
            print(f"Error setting value: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value with read distribution"""
        try:
            shard_id = self._get_shard(key)
            nodes = self.nodes.get(shard_id, [])
            
            if not nodes:
                return None
            
            # Randomly select node for read distribution
            node = random.choice([
                node for node in nodes
                if node.connection.ping()  # Check if node is alive
            ])
            
            if not node:
                return None
            
            value = node.connection.get(key)
            return json.loads(value) if value else None
            
        except Exception as e:
            print(f"Error getting value: {e}")
            return None
    
    async def _replicate_data(self, 
                             node: CacheNetworkNode, 
                             key: str, 
                             value: Any, 
                             ttl: int) -> bool:
        """Replicate data to a specific node"""
        try:
            return bool(
                node.connection.setex(
                    key,
                    ttl,
                    json.dumps(value)
                )
            )
        except Exception as e:
            print(f"Error replicating to {node}: {e}")
            return False
    
    async def monitor_health(self) -> Dict[int, Dict[str, Any]]:
        """Monitor health of all cache nodes"""
        health_status = {}
        
        for shard_id, nodes in self.nodes.items():
            shard_health = {
                'total_nodes': len(nodes),
                'healthy_nodes': 0,
                'primary_healthy': False,
                'replica_healthy': False
            }
            
            for node in nodes:
                try:
                    if node.connection.ping():
                        shard_health['healthy_nodes'] += 1
                        if node.role == 'primary':
                            shard_health['primary_healthy'] = True
                        else:
                            shard_health['replica_healthy'] = True
                except:
                    continue
            
            health_status[shard_id] = shard_health
        
        return health_status
    
    async def failover(self, shard_id: int) -> bool:
        """Handle failover for a specific shard"""
        try:
            # Get current primary
            current_primary = self._get_node(shard_id, 'primary')
            
            if not current_primary:
                return False
            
            # Check if primary is down
            try:
                current_primary.connection.ping()
                return True  # Primary is healthy
            except:
                pass
            
            # Select new primary from replicas
            replicas = [
                node for node in self.nodes[shard_id]
                if node.role == 'replica' and node.connection.ping()
            ]
            
            if not replicas:
                return False
            
            # Promote first healthy replica
            new_primary = replicas[0]
            new_primary.role = 'primary'
            current_primary.role = 'replica'
            
            # Update sentinel configuration
            self.sentinel.failover(f'shard{shard_id}')
            
            return True
            
        except Exception as e:
            print(f"Error during failover: {e}")
            return False

# Usage Example
async def main():
    # Initialize cache network
    sentinel_hosts = [
        ('localhost', 26379),
        ('localhost', 26380),
        ('localhost', 26381)
    ]
    
    cache_network = DistributedCacheNetwork(
        sentinel_hosts=sentinel_hosts,
        num_shards=3
    )
    
    # Set and get values
    await cache_network.set("key1", {"data": "test1"})
    await cache_network.set("key2", {"data": "test2"})
    
    value1 = await cache_network.get("key1")
    value2 = await cache_network.get("key2")
    
    # Monitor health
    health_status = await cache_network.monitor_health()
    print("Network Health:", json.dumps(health_status, indent=2))
    
    # Simulate failover
    await cache_network.failover(0)

if __name__ == "__main__":
    asyncio.run(main())
```

Slide 14: Additional Resources

*   "Consistency Models in Distributed Caching Systems" [https://arxiv.org/abs/2203.15472](https://arxiv.org/abs/2203.15472)
*   "Performance Analysis of Distributed Caching Architectures" [https://arxiv.org/abs/2106.09347](https://arxiv.org/abs/2106.09347)
*   "Optimal Cache Replacement Policies for Modern Storage Systems" [https://arxiv.org/abs/2104.12920](https://arxiv.org/abs/2104.12920)
*   "Dynamic Cache Partitioning and Replication Strategies" [https://arxiv.org/abs/2201.08374](https://arxiv.org/abs/2201.08374)
*   "Cache Coherence Protocols in Distributed Systems" [https://arxiv.org/abs/2202.11583](https://arxiv.org/abs/2202.11583)

