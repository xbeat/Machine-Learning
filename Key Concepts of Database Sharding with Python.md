## Key Concepts of Database Sharding with Python

Slide 1: Introduction to Database Sharding

Database sharding is a technique used to horizontally partition data across multiple databases or servers. It's a crucial strategy for handling large-scale applications and improving database performance.

```python

class DatabaseShard:
    def __init__(self, shard_key):
        self.shard_key = shard_key
        self.data = {}

    def insert(self, key, value):
        self.data[key] = value

class ShardedDatabase:
    def __init__(self, num_shards):
        self.shards = [DatabaseShard(i) for i in range(num_shards)]

    def insert(self, key, value):
        shard = self.get_shard(key)
        shard.insert(key, value)

    def get_shard(self, key):
        shard_key = hash(key) % len(self.shards)
        return self.shards[shard_key]

# Usage
db = ShardedDatabase(3)
db.insert("user1", {"name": "Alice", "age": 30})
db.insert("user2", {"name": "Bob", "age": 25})
```

Slide 2: Shard Key Selection

The shard key is a crucial element in database sharding. It determines how data is distributed across shards and affects query performance.

```python
    def __init__(self, user_id, name, country):
        self.user_id = user_id
        self.name = name
        self.country = country

class CountryShardedDatabase:
    def __init__(self):
        self.shards = {}

    def insert(self, user):
        if user.country not in self.shards:
            self.shards[user.country] = []
        self.shards[user.country].append(user)

    def get_users_by_country(self, country):
        return self.shards.get(country, [])

# Usage
db = CountryShardedDatabase()
db.insert(User(1, "Alice", "USA"))
db.insert(User(2, "Bob", "Canada"))
db.insert(User(3, "Charlie", "USA"))

usa_users = db.get_users_by_country("USA")
print(f"Users in USA: {len(usa_users)}")  # Output: Users in USA: 2
```

Slide 3: Shard Types: Vertical vs. Horizontal

Vertical sharding involves splitting different tables across multiple servers, while horizontal sharding distributes rows of a single table across multiple servers.

```python
class UserDatabase:
    def __init__(self):
        self.users = {}

class OrderDatabase:
    def __init__(self):
        self.orders = {}

# Horizontal Sharding
class UserShard:
    def __init__(self, shard_key):
        self.shard_key = shard_key
        self.users = {}

class HorizontallyShardedUserDatabase:
    def __init__(self, num_shards):
        self.shards = [UserShard(i) for i in range(num_shards)]

    def insert_user(self, user_id, user_data):
        shard = self.get_shard(user_id)
        shard.users[user_id] = user_data

    def get_shard(self, user_id):
        shard_key = hash(user_id) % len(self.shards)
        return self.shards[shard_key]

# Usage
vertical_user_db = UserDatabase()
vertical_order_db = OrderDatabase()

horizontal_user_db = HorizontallyShardedUserDatabase(3)
horizontal_user_db.insert_user(1, {"name": "Alice", "age": 30})
horizontal_user_db.insert_user(2, {"name": "Bob", "age": 25})
```

Slide 4: Consistent Hashing

Consistent hashing is a technique used to distribute data across shards in a way that minimizes reorganization when shards are added or removed.

```python

class ConsistentHash:
    def __init__(self, nodes, virtual_nodes=100):
        self.nodes = nodes
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self._build_ring()

    def _build_ring(self):
        for node in self.nodes:
            for i in range(self.virtual_nodes):
                key = self._hash(f"{node}:{i}")
                self.ring[key] = node

    def _hash(self, key):
        return hashlib.md5(key.encode()).hexdigest()

    def get_node(self, key):
        if not self.ring:
            return None
        hash_key = self._hash(key)
        for ring_key in sorted(self.ring.keys()):
            if ring_key >= hash_key:
                return self.ring[ring_key]
        return self.ring[sorted(self.ring.keys())[0]]

# Usage
nodes = ["shard1", "shard2", "shard3"]
ch = ConsistentHash(nodes)
print(ch.get_node("user1"))  # Output: shard2
print(ch.get_node("user2"))  # Output: shard1
```

Slide 5: Data Distribution Strategies

Different strategies can be used to distribute data across shards, such as range-based, hash-based, or directory-based sharding.

```python

class RangeBasedSharding:
    def __init__(self, ranges):
        self.ranges = ranges

    def get_shard(self, key):
        for shard, (start, end) in enumerate(self.ranges):
            if start <= key < end:
                return f"shard{shard + 1}"
        return None

class HashBasedSharding:
    def __init__(self, num_shards):
        self.num_shards = num_shards

    def get_shard(self, key):
        return f"shard{hash(key) % self.num_shards + 1}"

class DirectoryBasedSharding:
    def __init__(self):
        self.directory = {}

    def set_shard(self, key, shard):
        self.directory[key] = shard

    def get_shard(self, key):
        return self.directory.get(key, None)

# Usage
range_sharding = RangeBasedSharding([(0, 100), (100, 200), (200, 300)])
hash_sharding = HashBasedSharding(3)
dir_sharding = DirectoryBasedSharding()

print(range_sharding.get_shard(150))  # Output: shard2
print(hash_sharding.get_shard("user1"))  # Output: shard3
dir_sharding.set_shard("user1", "shard1")
print(dir_sharding.get_shard("user1"))  # Output: shard1
```

Slide 6: Handling Cross-Shard Queries

Cross-shard queries are a challenge in sharded databases. Strategies like scatter-gather or using a query router can help manage these queries efficiently.

```python
    def __init__(self, shards):
        self.shards = shards

    def execute_query(self, query):
        results = []
        for shard in self.shards:
            shard_result = shard.execute_query(query)
            results.extend(shard_result)
        return results

class Shard:
    def __init__(self, shard_id, data):
        self.shard_id = shard_id
        self.data = data

    def execute_query(self, query):
        # Simulate query execution
        return [item for item in self.data if query(item)]

# Usage
shard1 = Shard(1, [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
shard2 = Shard(2, [{"id": 3, "name": "Charlie"}, {"id": 4, "name": "David"}])

router = QueryRouter([shard1, shard2])
result = router.execute_query(lambda x: "a" in x["name"].lower())
print(result)  # Output: [{'id': 1, 'name': 'Alice'}, {'id': 3, 'name': 'Charlie'}, {'id': 4, 'name': 'David'}]
```

Slide 7: Rebalancing Shards

As data grows or access patterns change, it may be necessary to rebalance shards to maintain optimal performance.

```python

class ShardRebalancer:
    def __init__(self, shards):
        self.shards = shards

    def rebalance(self, threshold):
        total_size = sum(len(shard.data) for shard in self.shards)
        target_size = total_size // len(self.shards)

        for source_shard in self.shards:
            while len(source_shard.data) > target_size + threshold:
                dest_shard = min(self.shards, key=lambda s: len(s.data))
                if dest_shard == source_shard:
                    break
                key_to_move = random.choice(list(source_shard.data.keys()))
                dest_shard.data[key_to_move] = source_shard.data.pop(key_to_move)

class Shard:
    def __init__(self, shard_id):
        self.shard_id = shard_id
        self.data = {}

# Usage
shards = [Shard(i) for i in range(3)]
for i in range(100):
    shard = random.choice(shards)
    shard.data[f"key{i}"] = f"value{i}"

rebalancer = ShardRebalancer(shards)
rebalancer.rebalance(threshold=5)

for shard in shards:
    print(f"Shard {shard.shard_id} size: {len(shard.data)}")
```

Slide 8: Handling Shard Failures

Implementing fault tolerance and recovery mechanisms is crucial for maintaining data integrity in a sharded database system.

```python

class ShardedDatabaseWithReplication:
    def __init__(self, num_shards, replication_factor):
        self.shards = [Shard(i) for i in range(num_shards)]
        self.replication_factor = replication_factor

    def insert(self, key, value):
        primary_shard = self.get_primary_shard(key)
        primary_shard.insert(key, value)
        
        # Replicate to secondary shards
        secondary_shards = self.get_secondary_shards(key)
        for shard in secondary_shards:
            shard.insert(key, value)

    def get(self, key):
        primary_shard = self.get_primary_shard(key)
        if primary_shard.is_healthy():
            return primary_shard.get(key)
        
        # Fallback to secondary shards
        secondary_shards = self.get_secondary_shards(key)
        for shard in secondary_shards:
            if shard.is_healthy():
                return shard.get(key)
        
        raise Exception("Data unavailable")

    def get_primary_shard(self, key):
        return self.shards[hash(key) % len(self.shards)]

    def get_secondary_shards(self, key):
        all_shards = set(self.shards)
        primary_shard = self.get_primary_shard(key)
        all_shards.remove(primary_shard)
        return random.sample(list(all_shards), min(self.replication_factor - 1, len(all_shards)))

class Shard:
    def __init__(self, shard_id):
        self.shard_id = shard_id
        self.data = {}
        self.healthy = True

    def insert(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)

    def is_healthy(self):
        return self.healthy

# Usage
db = ShardedDatabaseWithReplication(num_shards=5, replication_factor=3)
db.insert("user1", {"name": "Alice", "age": 30})
print(db.get("user1"))  # Output: {'name': 'Alice', 'age': 30}

# Simulate primary shard failure
primary_shard = db.get_primary_shard("user1")
primary_shard.healthy = False

# Data is still accessible from secondary shards
print(db.get("user1"))  # Output: {'name': 'Alice', 'age': 30}
```

Slide 9: Sharding in NoSQL Databases

NoSQL databases often have built-in sharding capabilities, making it easier to scale horizontally.

```python

# Simulate a MongoDB sharded cluster
class MongoShardedCluster:
    def __init__(self, connection_string):
        self.client = MongoClient(connection_string)
        self.db = self.client.get_database("my_sharded_db")
        self.collection = self.db.get_collection("users")

    def insert_user(self, user):
        return self.collection.insert_one(user)

    def find_user(self, query):
        return self.collection.find_one(query)

# Usage
cluster = MongoShardedCluster("mongodb://localhost:27017,localhost:27018,localhost:27019")

# Insert a document
result = cluster.insert_user({"name": "Alice", "age": 30, "city": "New York"})
print(f"Inserted document ID: {result.inserted_id}")

# Find a document
user = cluster.find_user({"name": "Alice"})
print(f"Found user: {user}")

# Note: In a real MongoDB setup, you would need to configure sharding on the cluster
# and choose appropriate shard keys for your collections.
```

Slide 10: Sharding and Indexing

Proper indexing is crucial for maintaining performance in a sharded database environment. Sharded indexes distribute index data across multiple shards, allowing for faster query execution.

```python
    def __init__(self, num_shards):
        self.shards = [[] for _ in range(num_shards)]

    def insert(self, key, value):
        shard = self._get_shard(key)
        bisect.insort(self.shards[shard], (key, value))

    def search(self, key):
        shard = self._get_shard(key)
        index = bisect.bisect_left(self.shards[shard], (key, None))
        if index < len(self.shards[shard]) and self.shards[shard][index][0] == key:
            return self.shards[shard][index][1]
        return None

    def _get_shard(self, key):
        return hash(key) % len(self.shards)

# Usage
index = ShardedIndex(num_shards=3)
index.insert("user1", {"name": "Alice", "age": 30})
index.insert("user2", {"name": "Bob", "age": 25})
index.insert("user3", {"name": "Charlie", "age": 35})

print(index.search("user2"))  # Output: {'name': 'Bob', 'age': 25}
print(index.search("user4"))  # Output: None
```

Slide 11: Real-life Example: Social Media Platform

A social media platform can benefit from sharding to handle large amounts of user data and posts efficiently.

```python

class SocialMediaPlatform:
    def __init__(self, num_shards):
        self.user_shards = [dict() for _ in range(num_shards)]
        self.post_shards = [dict() for _ in range(num_shards)]

    def add_user(self, user_id, user_data):
        shard = self._get_user_shard(user_id)
        self.user_shards[shard][user_id] = user_data

    def add_post(self, post_id, post_data):
        shard = self._get_post_shard(post_id)
        self.post_shards[shard][post_id] = post_data

    def get_user(self, user_id):
        shard = self._get_user_shard(user_id)
        return self.user_shards[shard].get(user_id)

    def get_post(self, post_id):
        shard = self._get_post_shard(post_id)
        return self.post_shards[shard].get(post_id)

    def _get_user_shard(self, user_id):
        return hash(user_id) % len(self.user_shards)

    def _get_post_shard(self, post_id):
        return hash(post_id) % len(self.post_shards)

# Usage
platform = SocialMediaPlatform(num_shards=5)

# Add users
platform.add_user("user1", {"name": "Alice", "followers": 1000})
platform.add_user("user2", {"name": "Bob", "followers": 500})

# Add posts
platform.add_post("post1", {"author": "user1", "content": "Hello, world!"})
platform.add_post("post2", {"author": "user2", "content": "Sharding is cool!"})

# Retrieve data
print(platform.get_user("user1"))  # Output: {'name': 'Alice', 'followers': 1000}
print(platform.get_post("post2"))  # Output: {'author': 'user2', 'content': 'Sharding is cool!'}
```

Slide 12: Sharding Challenges and Considerations

While sharding offers significant benefits, it also introduces complexities that need to be carefully managed.

```python
    @staticmethod
    def demonstrate_data_distribution():
        # Simulating uneven data distribution
        shards = [[] for _ in range(3)]
        for i in range(1000):
            shard = hash(f"key{i}") % 3
            shards[shard].append(i)
        
        for i, shard in enumerate(shards):
            print(f"Shard {i} size: {len(shard)}")

    @staticmethod
    def simulate_cross_shard_query():
        # Simulating a cross-shard query
        def query_multiple_shards(shards, condition):
            results = []
            for shard in shards:
                results.extend([item for item in shard if condition(item)])
            return results

        shards = [
            [{"id": 1, "value": 10}, {"id": 2, "value": 20}],
            [{"id": 3, "value": 30}, {"id": 4, "value": 40}],
            [{"id": 5, "value": 50}, {"id": 6, "value": 60}]
        ]

        result = query_multiple_shards(shards, lambda x: x["value"] > 25)
        print(f"Items with value > 25: {result}")

# Usage
ShardingChallenges.demonstrate_data_distribution()
ShardingChallenges.simulate_cross_shard_query()
```

Slide 13: Sharding Best Practices

Following best practices can help ensure a successful implementation of database sharding.

```python
    @staticmethod
    def choose_shard_key():
        # Demonstrating shard key selection
        class User:
            def __init__(self, user_id, country):
                self.user_id = user_id
                self.country = country

        def shard_by_user_id(user):
            return hash(user.user_id) % 3

        def shard_by_country(user):
            country_to_shard = {"USA": 0, "Canada": 1, "Mexico": 2}
            return country_to_shard.get(user.country, 0)

        users = [
            User(1, "USA"),
            User(2, "Canada"),
            User(3, "Mexico"),
            User(4, "USA")
        ]

        for user in users:
            print(f"User {user.user_id} - Shard by ID: {shard_by_user_id(user)}, Shard by Country: {shard_by_country(user)}")

    @staticmethod
    def plan_for_growth():
        # Simulating database growth and resharding
        class ShardedDatabase:
            def __init__(self, initial_shards):
                self.shards = initial_shards

            def add_shard(self):
                new_shard = len(self.shards)
                self.shards.append(new_shard)
                print(f"Added new shard: {new_shard}")

            def rebalance(self):
                print("Rebalancing data across shards...")

        db = ShardedDatabase([0, 1])
        print(f"Initial shards: {db.shards}")
        
        # Simulate growth
        for _ in range(3):
            db.add_shard()
            db.rebalance()

        print(f"Final shards: {db.shards}")

# Usage
ShardingBestPractices.choose_shard_key()
ShardingBestPractices.plan_for_growth()
```

Slide 14: Real-life Example: E-commerce Platform

An e-commerce platform can use sharding to manage large product catalogs and handle high transaction volumes.

```python

class EcommercePlatform:
    def __init__(self, num_shards):
        self.product_shards = [dict() for _ in range(num_shards)]
        self.order_shards = [dict() for _ in range(num_shards)]

    def add_product(self, product_id, product_data):
        shard = self._get_product_shard(product_id)
        self.product_shards[shard][product_id] = product_data

    def place_order(self, order_id, order_data):
        shard = self._get_order_shard(order_id)
        self.order_shards[shard][order_id] = order_data

    def get_product(self, product_id):
        shard = self._get_product_shard(product_id)
        return self.product_shards[shard].get(product_id)

    def get_order(self, order_id):
        shard = self._get_order_shard(order_id)
        return self.order_shards[shard].get(order_id)

    def _get_product_shard(self, product_id):
        return hash(product_id) % len(self.product_shards)

    def _get_order_shard(self, order_id):
        return hash(order_id) % len(self.order_shards)

# Usage
ecommerce = EcommercePlatform(num_shards=3)

# Add products
ecommerce.add_product("prod1", {"name": "Laptop", "price": 999.99})
ecommerce.add_product("prod2", {"name": "Smartphone", "price": 599.99})

# Place orders
ecommerce.place_order("order1", {"user": "user1", "products": ["prod1"], "total": 999.99})
ecommerce.place_order("order2", {"user": "user2", "products": ["prod2"], "total": 599.99})

# Retrieve data
print(ecommerce.get_product("prod1"))  # Output: {'name': 'Laptop', 'price': 999.99}
print(ecommerce.get_order("order2"))   # Output: {'user': 'user2', 'products': ['prod2'], 'total': 599.99}
```

Slide 15: Additional Resources

For more in-depth information on database sharding and related topics, consider exploring the following resources:

1. "Designing Data-Intensive Applications" by Martin Kleppmann
   * A comprehensive guide to modern database systems, including sharding techniques.
2. "Database Internals" by Alex Petrov
   * Provides insights into the inner workings of databases, including distributed systems.
3. "Scalability, Availability & Stability Patterns" by Michael T. Nygard
   * Discusses patterns for building scalable systems, including sharding strategies.
4. ArXiv.org papers on database sharding:
   * "A Survey of Sharding in NoSQL Databases" (arXiv:2101.00274)
   * "ShardFS: A Scalable Distributed File System" (arXiv:1910.05801)

These resources offer deeper insights into the concepts and practical implementations of database sharding and related distributed systems technologies.


