## Understanding Apache Kafka with Python

Slide 1: Introduction to Apache Kafka

Apache Kafka is a distributed streaming platform that allows for high-throughput, fault-tolerant, and scalable data streaming. It's designed to handle real-time data feeds and can process millions of messages per second. Kafka uses a publish-subscribe model where data is organized into topics, and producers write data to topics while consumers read from them.

```python

# Create a producer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'])

# Create a consumer
consumer = KafkaConsumer('my_topic', bootstrap_servers=['localhost:9092'])

# Produce a message
producer.send('my_topic', b'Hello, Kafka!')

# Consume messages
for message in consumer:
    print(message.value.decode('utf-8'))
```

Slide 2: Kafka Architecture

Kafka's architecture consists of brokers, topics, partitions, and replicas. Brokers are servers that store and manage topics. Topics are categories or feed names to which records are published. Partitions allow topics to be distributed across multiple brokers for scalability. Replicas are copies of partitions for fault tolerance.

```python

# List topics
admin_client = kafka.KafkaAdminClient(bootstrap_servers=['localhost:9092'])
topics = admin_client.list_topics()
print(f"Available topics: {topics}")

# Get topic details
topic_metadata = admin_client.describe_topics(['my_topic'])
print(f"Topic metadata: {topic_metadata}")
```

Slide 3: Producing Messages

Producers are client applications that publish (write) events to Kafka topics. When sending messages, producers can choose to receive acknowledgments of data writes.

```python

# Create a producer with acknowledgment settings
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         acks='all',
                         retries=3)

# Send a message and get the metadata
future = producer.send('my_topic', b'Important message')
metadata = future.get(timeout=10)
print(f"Message sent to topic {metadata.topic}, partition {metadata.partition}, offset {metadata.offset}")
```

Slide 4: Consuming Messages

Consumers read data from topics. They can be part of consumer groups, allowing for load balancing and fault tolerance.

```python

# Create a consumer with a specific group id
consumer = KafkaConsumer('my_topic',
                         bootstrap_servers=['localhost:9092'],
                         group_id='my_group',
                         auto_offset_reset='earliest')

# Consume messages
for message in consumer:
    print(f"Received message: {message.value.decode('utf-8')} from partition {message.partition} at offset {message.offset}")
```

Slide 5: Working with Consumer Groups

Consumer groups allow multiple consumers to read from the same topic, distributing the load across instances. Each partition is read by only one consumer in the group.

```python
import threading

def consume(consumer_id):
    consumer = KafkaConsumer('my_topic',
                             bootstrap_servers=['localhost:9092'],
                             group_id='my_group')
    for message in consumer:
        print(f"Consumer {consumer_id} received: {message.value.decode('utf-8')}")

# Start multiple consumers in separate threads
for i in range(3):
    threading.Thread(target=consume, args=(i,)).start()
```

Slide 6: Handling Serialization and Deserialization

Kafka deals with byte arrays by default. Serialization and deserialization are crucial for working with complex data types.

```python
from kafka import KafkaProducer, KafkaConsumer

# Custom JSON serializer
def json_serializer(data):
    return json.dumps(data).encode('utf-8')

# Custom JSON deserializer
def json_deserializer(data):
    return json.loads(data.decode('utf-8'))

# Producer with JSON serializer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=json_serializer)

# Consumer with JSON deserializer
consumer = KafkaConsumer('my_topic',
                         bootstrap_servers=['localhost:9092'],
                         value_deserializer=json_deserializer)

# Produce a JSON message
producer.send('my_topic', {'key': 'value'})

# Consume JSON messages
for message in consumer:
    print(f"Received: {message.value}")
```

Slide 7: Handling Partitions

Partitions allow Kafka to distribute data across multiple brokers. You can control which partition a message goes to using a key.

```python

# Create a producer with a custom partitioner
def custom_partitioner(key, all_partitions, available):
    return hash(key) % len(all_partitions)

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         partitioner=custom_partitioner)

# Send messages with different keys
producer.send('my_topic', key=b'user1', value=b'Message for user1')
producer.send('my_topic', key=b'user2', value=b'Message for user2')

# Flush to ensure all messages are sent
producer.flush()
```

Slide 8: Handling Offsets

Offsets keep track of the consumer's position in each partition. Managing offsets is crucial for ensuring exactly-once processing.

```python

consumer = KafkaConsumer(bootstrap_servers=['localhost:9092'],
                         group_id='my_group',
                         enable_auto_commit=False)

# Manually assign partitions
partition = TopicPartition('my_topic', 0)
consumer.assign([partition])

# Seek to a specific offset
consumer.seek(partition, 100)

# Consume messages and manually commit offsets
for message in consumer:
    print(f"Received: {message.value.decode('utf-8')}")
    consumer.commit()
```

Slide 9: Error Handling and Retries

Robust Kafka applications need to handle various errors, such as network issues or broker failures. Implementing proper error handling and retries is essential.

```python
from kafka.errors import KafkaError

# Create a producer with retries
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         retries=5,
                         retry_backoff_ms=1000)

# Send a message with error handling
try:
    future = producer.send('my_topic', b'Important message')
    record_metadata = future.get(timeout=10)
except KafkaError as e:
    print(f"Failed to send message: {str(e)}")
else:
    print(f"Message sent successfully to {record_metadata.topic}")
```

Slide 10: Monitoring and Metrics

Monitoring Kafka clusters and applications is crucial for maintaining performance and identifying issues. Python's Kafka client provides various metrics.

```python
import time

producer = KafkaProducer(bootstrap_servers=['localhost:9092'])

# Send some messages
for i in range(1000):
    producer.send('my_topic', f'Message {i}'.encode('utf-8'))

# Get metrics
metrics = producer.metrics()

# Print some interesting metrics
print(f"Number of in-flight requests: {metrics['producer-metrics']['request-in-flight']}")
print(f"Average request latency: {metrics['producer-metrics']['request-latency-avg']}")
print(f"Number of record sends: {metrics['producer-metrics']['record-send-total']}")
```

Slide 11: Real-life Example: Log Aggregation

Kafka is commonly used for log aggregation in distributed systems. Here's a simple example of how to use Kafka for collecting and processing logs from multiple sources.

```python
from kafka import KafkaProducer, KafkaConsumer
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a Kafka producer for sending logs
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# Function to send a log message
def send_log(service_name, message):
    log_entry = {
        'service': service_name,
        'message': message,
        'timestamp': int(time.time())
    }
    producer.send('logs', log_entry)
    logger.info(f"Sent log: {log_entry}")

# Simulate logs from different services
send_log('web_server', 'GET request received')
send_log('database', 'Query executed in 15ms')
send_log('auth_service', 'User authenticated')

# Consumer for processing logs
consumer = KafkaConsumer('logs',
                         bootstrap_servers=['localhost:9092'],
                         value_deserializer=lambda v: json.loads(v.decode('utf-8')))

# Process incoming logs
for message in consumer:
    log = message.value
    print(f"Processed log: Service: {log['service']}, Message: {log['message']}")
```

Slide 12: Real-life Example: IoT Data Processing

Kafka is ideal for processing data from IoT devices due to its ability to handle high-throughput data streams. Here's a simple example of how to use Kafka for collecting and processing sensor data.

```python
import json
import random
import time

# Simulate IoT sensor data
def generate_sensor_data():
    return {
        'device_id': f'sensor_{random.randint(1, 100)}',
        'temperature': round(random.uniform(20, 30), 2),
        'humidity': round(random.uniform(30, 70), 2),
        'timestamp': int(time.time())
    }

# Create a Kafka producer for sending sensor data
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# Send simulated sensor data
for _ in range(10):
    data = generate_sensor_data()
    producer.send('iot_data', data)
    print(f"Sent data: {data}")
    time.sleep(1)

# Consumer for processing sensor data
consumer = KafkaConsumer('iot_data',
                         bootstrap_servers=['localhost:9092'],
                         value_deserializer=lambda v: json.loads(v.decode('utf-8')))

# Process incoming sensor data
for message in consumer:
    data = message.value
    if data['temperature'] > 28:
        print(f"High temperature alert: Device {data['device_id']} - {data['temperature']}°C")
    if data['humidity'] > 60:
        print(f"High humidity alert: Device {data['device_id']} - {data['humidity']}%")
```

Slide 13: Kafka Streams with Python

While Kafka Streams is primarily a Java library, you can achieve similar functionality in Python using the confluent-kafka library and some custom processing logic.

```python
import json

# Initialize consumer and producer
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'my_stream_group',
    'auto.offset.reset': 'earliest'
})
producer = Producer({'bootstrap.servers': 'localhost:9092'})

consumer.subscribe(['input_topic'])

# Simple stream processing
while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    if msg.error():
        print(f"Consumer error: {msg.error()}")
        continue
    
    # Process the message
    data = json.loads(msg.value().decode('utf-8'))
    processed_data = {
        'original_value': data['value'],
        'doubled_value': data['value'] * 2
    }
    
    # Produce the processed result
    producer.produce('output_topic', json.dumps(processed_data).encode('utf-8'))
    producer.flush()

consumer.close()
```

Slide 14: Kafka Connect with Python

Kafka Connect is a tool for streaming data between Kafka and other systems. While it's typically used with Java, you can create custom connectors in Python using the Kafka Connect API.

```python
import json

class PythonSourceConnector:
    def __init__(self, config):
        self.topic = config['topic']
        self.producer = Producer({'bootstrap.servers': config['bootstrap.servers']})

    def run(self):
        while True:
            # Fetch data from your source system
            data = self.fetch_data()
            
            # Send data to Kafka
            self.producer.produce(self.topic, json.dumps(data).encode('utf-8'))
            self.producer.flush()

    def fetch_data(self):
        # Implement your logic to fetch data from the source system
        return {'key': 'value'}

# Usage
config = {
    'bootstrap.servers': 'localhost:9092',
    'topic': 'source_data'
}
connector = PythonSourceConnector(config)
connector.run()
```

Slide 15: Additional Resources

For more in-depth information on Apache Kafka and its usage with Python, consider exploring the following resources:

1. Apache Kafka Documentation: [https://kafka.apache.org/documentation/](https://kafka.apache.org/documentation/)
2. confluent-kafka Python Client: [https://github.com/confluentinc/confluent-kafka-python](https://github.com/confluentinc/confluent-kafka-python)
3. Kafka: The Definitive Guide (O'Reilly book)
4. "Designing Event-Driven Systems" by Ben Stopford (free e-book from Confluent)
5. Kafka Summit conference recordings: [https://www.confluent.io/resources/kafka-summit-recordings/](https://www.confluent.io/resources/kafka-summit-recordings/)

Remember to verify these resources and their availability, as they may change over time.


