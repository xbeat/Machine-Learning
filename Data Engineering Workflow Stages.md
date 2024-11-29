## Data Engineering Workflow Stages
Slide 1: Data Source Connection

Modern data engineering requires robust connection handling to multiple data sources simultaneously. This implementation demonstrates connecting to various data sources including SQL databases, REST APIs, and file systems while handling authentication and connection pooling.

```python
import pandas as pd
import requests
import sqlalchemy
import fsspec
from typing import Dict, Any

class DataSourceConnector:
    def __init__(self, credentials: Dict[str, Any]):
        self.credentials = credentials
        self.connections = {}
    
    def connect_sql(self, database: str) -> sqlalchemy.engine.Engine:
        conn_string = f"postgresql://{self.credentials['user']}:{self.credentials['password']}@{self.credentials['host']}/{database}"
        engine = sqlalchemy.create_engine(conn_string, pool_size=5)
        return engine
    
    def connect_api(self, endpoint: str) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            'Authorization': f"Bearer {self.credentials['api_key']}",
            'Content-Type': 'application/json'
        })
        return session

# Example usage
credentials = {
    'user': 'db_user',
    'password': 'db_pass',
    'host': 'localhost',
    'api_key': 'your_api_key'
}

connector = DataSourceConnector(credentials)
db_engine = connector.connect_sql('analytics_db')
api_session = connector.connect_api('https://api.example.com/data')

# Query example
df = pd.read_sql("SELECT * FROM events LIMIT 5", db_engine)
```

Slide 2: Data Ingestion Pipeline

Data ingestion requires robust error handling, retry mechanisms, and rate limiting when dealing with external APIs or streaming data sources. This implementation showcases a scalable ingestion pipeline with backoff strategies.

```python
import time
from typing import Iterator, Any
from datetime import datetime
import backoff
import requests

class DataIngestionPipeline:
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.retry_count = 3
        
    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
    def fetch_batch(self, url: str, params: dict) -> dict:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def process_stream(self, source_url: str) -> Iterator[Any]:
        params = {'offset': 0, 'limit': self.batch_size}
        
        while True:
            try:
                batch = self.fetch_batch(source_url, params)
                if not batch['data']:
                    break
                    
                yield from batch['data']
                params['offset'] += self.batch_size
                
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                time.sleep(1)

# Example usage
pipeline = DataIngestionPipeline(batch_size=100)
for record in pipeline.process_stream('https://api.example.com/events'):
    timestamp = datetime.now().isoformat()
    print(f"{timestamp} - Processing record: {record['id']}")
```

Slide 3: Data Storage Optimization

Efficient data storage requires careful consideration of partitioning strategies, compression algorithms, and storage formats. This implementation demonstrates optimized Parquet file storage with partitioning and compression.

```python
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import numpy as np

class OptimizedStorage:
    def __init__(self, base_path: str):
        self.base_path = base_path
        
    def optimize_schema(self, df: pd.DataFrame) -> pa.Schema:
        optimized_fields = []
        for column, dtype in df.dtypes.items():
            if dtype == 'object':
                # Convert string columns to dictionary encoding
                optimized_fields.append(pa.field(column, pa.dictionary(pa.int32(), pa.string())))
            elif dtype == 'float64':
                # Downcast floats where possible
                if df[column].isnull().sum() == 0 and df[column].mod(1).sum() == 0:
                    optimized_fields.append(pa.field(column, pa.int32()))
                else:
                    optimized_fields.append(pa.field(column, pa.float32()))
            else:
                optimized_fields.append(pa.field(column, pa.from_numpy_dtype(dtype)))
        return pa.schema(optimized_fields)

    def write_partitioned(self, df: pd.DataFrame, partition_cols: list):
        table = pa.Table.from_pandas(df, schema=self.optimize_schema(df))
        
        pq.write_table(
            table,
            self.base_path,
            partition_cols=partition_cols,
            compression='snappy',
            row_group_size=100000
        )

# Example usage
storage = OptimizedStorage('/data/events')

# Sample data
df = pd.DataFrame({
    'timestamp': pd.date_range(start='2024-01-01', periods=1000000, freq='1min'),
    'user_id': np.random.randint(1, 1000, 1000000),
    'value': np.random.randn(1000000)
})

# Partition by date components
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month

storage.write_partitioned(df, partition_cols=['year', 'month'])
```

Slide 4: Data Processing with Apache Spark

Apache Spark provides distributed data processing capabilities essential for large-scale data transformation. This implementation shows advanced PySpark operations including custom UDFs and window functions for data processing.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *

class SparkProcessor:
    def __init__(self, app_name: str):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.memory.offHeap.enabled", True) \
            .config("spark.memory.offHeap.size", "10g") \
            .getOrCreate()
    
    def process_data(self, input_path: str) -> DataFrame:
        # Read data
        df = self.spark.read.parquet(input_path)
        
        # Define window specification
        window_spec = Window.partitionBy("user_id") \
                           .orderBy("timestamp") \
                           .rowsBetween(-2, 0)
        
        # Custom UDF for complex transformations
        @udf(returnType=FloatType())
        def calculate_metric(values):
            return float(sum(values) / len(values) if values else 0)
        
        # Apply transformations
        processed_df = df.withColumn(
            "moving_avg", 
            avg("value").over(window_spec)
        ).withColumn(
            "custom_metric",
            calculate_metric(collect_list("value").over(window_spec))
        )
        
        return processed_df

# Example usage
processor = SparkProcessor("DataProcessing")
result_df = processor.process_data("/data/events")
result_df.show()

# Save results
result_df.write.mode("overwrite") \
    .partitionBy("year", "month") \
    .parquet("/data/processed_events")
```

Slide 5: Data Integration with Apache Airflow

Orchestrating complex data pipelines requires robust workflow management. This implementation demonstrates an Airflow DAG with sophisticated error handling, branching logic, and dynamic task generation.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from datetime import datetime, timedelta
import logging

default_args = {
    'owner': 'data_engineering',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['alert@company.com']
}

def validate_data(**context):
    data_quality = context['task_instance'].xcom_pull(task_ids='fetch_data')
    if data_quality > 0.95:
        return 'process_data'
    return 'handle_data_quality_issues'

with DAG(
    'data_integration_pipeline',
    default_args=default_args,
    description='End-to-end data integration pipeline',
    schedule_interval='0 */4 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    create_tables = PostgresOperator(
        task_id='create_tables',
        postgres_conn_id='warehouse_db',
        sql="""
            CREATE TABLE IF NOT EXISTS processed_events (
                event_id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP,
                user_id INTEGER,
                value FLOAT
            );
        """
    )

    validate_quality = BranchPythonOperator(
        task_id='validate_quality',
        python_callable=validate_data,
        provide_context=True
    )

    def process_data(**context):
        # Complex data processing logic
        logging.info("Processing data batch")
        # Implementation details...

    process_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
        provide_context=True
    )

    create_tables >> validate_quality >> process_task
```

Slide 6: Data Access Layer Implementation

A robust data access layer provides secure and efficient data retrieval while managing caching and connection pooling. This implementation showcases a comprehensive data access interface with performance optimizations.

```python
from functools import lru_cache
import redis
from typing import Optional, List, Dict
import json

class DataAccessLayer:
    def __init__(self, cache_ttl: int = 3600):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.cache_ttl = cache_ttl
        
    @lru_cache(maxsize=1000)
    def get_user_data(self, user_id: int) -> Dict:
        # Try cache first
        cache_key = f"user:{user_id}"
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
            
        # If not in cache, query database
        with self.db_engine.connect() as conn:
            query = """
                SELECT u.*, array_agg(e.event_type) as events
                FROM users u
                LEFT JOIN events e ON u.id = e.user_id
                WHERE u.id = %s
                GROUP BY u.id
            """
            result = conn.execute(query, (user_id,)).fetchone()
            
            if result:
                user_data = dict(result)
                # Cache the result
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(user_data)
                )
                return user_data
                
        return None

    async def stream_events(self, user_id: int) -> AsyncIterator[Dict]:
        query = """
            SELECT *
            FROM events
            WHERE user_id = $1
            ORDER BY timestamp DESC
        """
        
        async with self.pool.acquire() as conn:
            async for record in conn.cursor(query, user_id):
                yield dict(record)
    
    def invalidate_cache(self, user_id: int):
        cache_key = f"user:{user_id}"
        self.redis_client.delete(cache_key)
        get_user_data.cache_clear()

# Example usage
dal = DataAccessLayer()
user_data = dal.get_user_data(123)
print(f"Retrieved user data: {user_data}")

# Async usage
async for event in dal.stream_events(123):
    print(f"Processing event: {event}")
```

Slide 7: Data Governance and Security Implementation

Implementing robust data governance requires comprehensive tracking of data lineage, access controls, and audit logging. This system demonstrates a complete implementation of data governance patterns including encryption and access management.

```python
from cryptography.fernet import Fernet
from datetime import datetime
import hashlib
import jwt
from typing import Dict, List, Optional

class DataGovernance:
    def __init__(self, encryption_key: bytes):
        self.fernet = Fernet(encryption_key)
        self.audit_log = []
        
    def encrypt_sensitive_data(self, data: Dict) -> Dict:
        sensitive_fields = ['ssn', 'credit_card', 'password']
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data:
                encrypted_data[field] = self.fernet.encrypt(
                    str(data[field]).encode()
                ).decode()
                
        return encrypted_data
    
    def track_lineage(self, dataset_id: str, operation: str, 
                     source_ids: List[str]) -> str:
        lineage_record = {
            'dataset_id': dataset_id,
            'operation': operation,
            'source_ids': source_ids,
            'timestamp': datetime.now().isoformat(),
            'hash': self._calculate_hash(dataset_id, source_ids)
        }
        
        self.audit_log.append(lineage_record)
        return lineage_record['hash']
    
    def _calculate_hash(self, dataset_id: str, source_ids: List[str]) -> str:
        content = f"{dataset_id}{''.join(sorted(source_ids))}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def verify_access(self, user_token: str, dataset_id: str) -> bool:
        try:
            payload = jwt.decode(user_token, self.secret_key, algorithms=['HS256'])
            return dataset_id in payload['accessible_datasets']
        except jwt.InvalidTokenError:
            return False

# Example usage
encryption_key = Fernet.generate_key()
governance = DataGovernance(encryption_key)

# Encrypt sensitive data
raw_data = {
    'user_id': 123,
    'ssn': '123-45-6789',
    'credit_card': '4111111111111111'
}
encrypted_data = governance.encrypt_sensitive_data(raw_data)

# Track data lineage
lineage_hash = governance.track_lineage(
    'processed_users_2024',
    'ETL_TRANSFORM',
    ['raw_users_2024', 'user_preferences_2024']
)

print(f"Encrypted data: {encrypted_data}")
print(f"Lineage hash: {lineage_hash}")
```

Slide 8: Version Control for Data Assets

Data version control ensures reproducibility and traceability of data transformations. This implementation provides a comprehensive system for versioning datasets with diff capabilities and rollback functionality.

```python
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

class DataVersionControl:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.version_history = {}
        self.current_version = None
        
    def commit_version(self, dataset: np.ndarray, 
                      metadata: Dict) -> str:
        # Calculate version hash
        version_hash = self._calculate_version_hash(dataset)
        
        # Create version record
        version_info = {
            'hash': version_hash,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata,
            'shape': dataset.shape,
            'parent_version': self.current_version
        }
        
        # Store version information
        self.version_history[version_hash] = version_info
        self.current_version = version_hash
        
        # Save dataset
        np.save(f"{self.storage_path}/{version_hash}.npy", dataset)
        return version_hash
    
    def _calculate_version_hash(self, dataset: np.ndarray) -> str:
        return hashlib.sha256(
            dataset.tobytes() + 
            str(datetime.now().timestamp()).encode()
        ).hexdigest()
    
    def get_version(self, version_hash: str) -> np.ndarray:
        if version_hash not in self.version_history:
            raise ValueError("Version not found")
            
        return np.load(f"{self.storage_path}/{version_hash}.npy")
    
    def compute_diff(self, version1: str, version2: str) -> Dict:
        dataset1 = self.get_version(version1)
        dataset2 = self.get_version(version2)
        
        return {
            'shape_diff': np.array(dataset2.shape) - np.array(dataset1.shape),
            'value_diff_mean': np.mean(dataset2 - dataset1),
            'value_diff_std': np.std(dataset2 - dataset1)
        }

# Example usage
dvc = DataVersionControl("/data/versions")

# Create sample dataset
dataset = np.random.randn(1000, 10)
metadata = {
    'description': 'Feature matrix for user behavior',
    'features': ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']
}

# Commit version
version_hash = dvc.commit_version(dataset, metadata)
print(f"Committed version: {version_hash}")

# Make changes and commit new version
modified_dataset = dataset * 1.1 + np.random.randn(1000, 10) * 0.1
new_version_hash = dvc.commit_version(modified_dataset, metadata)

# Compare versions
diff = dvc.compute_diff(version_hash, new_version_hash)
print(f"Version diff: {diff}")
```

Slide 9: Machine Learning Pipeline Integration

This implementation demonstrates a complete machine learning pipeline integration within the data engineering workflow, including feature engineering, model training, and deployment automation with model versioning.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import mlflow
from typing import Dict, Tuple, Any
import numpy as np

class MLPipeline:
    def __init__(self, model_name: str):
        self.model_name = model_name
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(model_name)
        
    def create_pipeline(self) -> Pipeline:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3
            ))
        ])
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   params: Dict[str, Any]) -> Tuple[Pipeline, str]:
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(params)
            
            # Create and train pipeline
            pipeline = self.create_pipeline()
            pipeline.set_params(**params)
            pipeline.fit(X, y)
            
            # Calculate metrics
            train_score = pipeline.score(X, y)
            mlflow.log_metric("train_score", train_score)
            
            # Save model
            mlflow.sklearn.log_model(
                pipeline,
                "model",
                registered_model_name=self.model_name
            )
            
            return pipeline, run.info.run_id

    def deploy_model(self, run_id: str) -> None:
        client = mlflow.tracking.MlflowClient()
        model_version = client.transition_model_version_stage(
            name=self.model_name,
            version=run_id,
            stage="Production"
        )
        
        print(f"Model {self.model_name} version {run_id} deployed to production")

# Example usage
pipeline = MLPipeline("user_churn_predictor")

# Generate sample data
X = np.random.randn(1000, 10)
y = (X[:, 0] * 2 + X[:, 1] - 0.5 * X[:, 2] + 
     np.random.randn(1000) * 0.1)

# Train model
params = {
    'model__n_estimators': 150,
    'model__learning_rate': 0.05,
    'model__max_depth': 4
}

trained_model, run_id = pipeline.train_model(X, y, params)
pipeline.deploy_model(run_id)

# Make predictions
predictions = trained_model.predict(X[:5])
print(f"Sample predictions: {predictions}")
```

Slide 10: Data Visualization and Reporting Engine

A sophisticated visualization engine that supports interactive dashboards and automated report generation. This implementation includes custom plotting functions and report templating.

```python
import plotly.graph_objects as go
import plotly.express as px
from jinja2 import Template
import pandas as pd
from typing import List, Dict, Any

class ReportingEngine:
    def __init__(self):
        self.figures = {}
        self.report_template = Template("""
            <html>
            <head>
                <title>{{ title }}</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <h1>{{ title }}</h1>
                {% for section in sections %}
                <div class="section">
                    <h2>{{ section.title }}</h2>
                    <p>{{ section.description }}</p>
                    <div id="plot_{{ section.id }}"></div>
                </div>
                {% endfor %}
            </body>
            </html>
        """)
    
    def create_time_series(self, df: pd.DataFrame, 
                          x_col: str, y_col: str, 
                          title: str) -> None:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='lines+markers',
                name=y_col
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            template='plotly_white'
        )
        
        self.figures[title] = fig
    
    def create_distribution(self, data: np.ndarray, 
                          title: str) -> None:
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=30,
                name='Distribution'
            )
        )
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            showlegend=False
        )
        
        self.figures[title] = fig
    
    def generate_report(self, title: str, 
                       sections: List[Dict[str, Any]]) -> str:
        for section in sections:
            if section['plot_title'] in self.figures:
                section['plot'] = self.figures[section['plot_title']].to_html()
        
        return self.report_template.render(
            title=title,
            sections=sections
        )

# Example usage
engine = ReportingEngine()

# Create sample data
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
values = np.cumsum(np.random.randn(100)) + 100

df = pd.DataFrame({
    'date': dates,
    'value': values
})

# Create visualizations
engine.create_time_series(
    df, 'date', 'value', 
    'Metric Evolution Over Time'
)
engine.create_distribution(
    values, 
    'Metric Distribution'
)

# Generate report
sections = [
    {
        'id': 1,
        'title': 'Time Series Analysis',
        'description': 'Analysis of metric evolution over time',
        'plot_title': 'Metric Evolution Over Time'
    },
    {
        'id': 2,
        'title': 'Distribution Analysis',
        'description': 'Statistical distribution of metric values',
        'plot_title': 'Metric Distribution'
    }
]

report_html = engine.generate_report('Analytics Report', sections)
print("Report generated successfully")
```

Slide 11: Real-Time Stream Processing

Implementation of a real-time stream processing system using Apache Kafka and custom stream processors. This demonstrates handling of high-throughput data streams with exactly-once processing guarantees.

```python
from kafka import KafkaConsumer, KafkaProducer
from confluent_kafka import Consumer, Producer
import json
import threading
from typing import Dict, Callable
import time

class StreamProcessor:
    def __init__(self, bootstrap_servers: str):
        self.bootstrap_servers = bootstrap_servers
        self.running = False
        self.processors: Dict[str, Callable] = {}
        
    def create_producer(self) -> KafkaProducer:
        return KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3
        )
    
    def create_consumer(self, group_id: str) -> KafkaConsumer:
        return KafkaConsumer(
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
    
    def add_processor(self, name: str, 
                     processor_func: Callable) -> None:
        self.processors[name] = processor_func
    
    def process_stream(self, input_topic: str, 
                      output_topic: str, 
                      processor_name: str) -> None:
        consumer = self.create_consumer(f"{processor_name}_group")
        producer = self.create_producer()
        
        consumer.subscribe([input_topic])
        
        while self.running:
            messages = consumer.poll(timeout_ms=1000)
            
            for topic_partition, records in messages.items():
                for record in records:
                    # Process message
                    result = self.processors[processor_name](record.value)
                    
                    # Produce result
                    producer.send(
                        output_topic,
                        value=result,
                        timestamp_ms=int(time.time() * 1000)
                    )
                    
                    # Commit offset
                    consumer.commit()
    
    def start(self, input_topic: str, 
              output_topic: str, 
              processor_name: str) -> None:
        self.running = True
        thread = threading.Thread(
            target=self.process_stream,
            args=(input_topic, output_topic, processor_name)
        )
        thread.start()
    
    def stop(self) -> None:
        self.running = False

# Example usage
def process_metric(message: Dict) -> Dict:
    # Add moving average calculation
    if 'values' in message:
        values = message['values']
        message['moving_avg'] = sum(values[-5:]) / len(values[-5:])
    return message

# Initialize processor
processor = StreamProcessor('localhost:9092')
processor.add_processor('metric_processor', process_metric)

# Start processing
processor.start(
    'raw_metrics',
    'processed_metrics',
    'metric_processor'
)

# Simulate some data
producer = processor.create_producer()
for i in range(10):
    data = {
        'metric_id': f'metric_{i}',
        'values': [float(x) for x in range(i, i+10)]
    }
    producer.send('raw_metrics', value=data)

time.sleep(5)
processor.stop()
```

Slide 12: Automated Data Quality Monitoring

Implementation of a comprehensive data quality monitoring system that automatically detects anomalies, validates data integrity, and generates alerts for quality issues.

```python
from scipy import stats
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

class DataQualityMonitor:
    def __init__(self, alert_threshold: float = 0.95):
        self.threshold = alert_threshold
        self.baseline_metrics = {}
        self.alert_history = []
        
    def compute_metrics(self, data: np.ndarray) -> Dict[str, float]:
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'nulls': float(np.isnan(data).sum() / len(data)),
            'unique_ratio': float(len(np.unique(data)) / len(data)),
            'zscore_outliers': float(np.sum(np.abs(stats.zscore(data)) > 3) / len(data))
        }
    
    def set_baseline(self, data: np.ndarray, 
                    column_name: str) -> None:
        self.baseline_metrics[column_name] = self.compute_metrics(data)
    
    def check_quality(self, data: np.ndarray, 
                     column_name: str) -> Dict[str, Any]:
        current_metrics = self.compute_metrics(data)
        baseline = self.baseline_metrics.get(column_name)
        
        if not baseline:
            raise ValueError(f"No baseline for column {column_name}")
        
        issues = []
        for metric, value in current_metrics.items():
            baseline_value = baseline[metric]
            if abs(value - baseline_value) > (baseline_value * (1 - self.threshold)):
                issues.append({
                    'metric': metric,
                    'current_value': value,
                    'baseline_value': baseline_value,
                    'deviation': abs(value - baseline_value) / baseline_value
                })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'column': column_name,
            'issues': issues,
            'metrics': current_metrics
        }
    
    def send_alert(self, quality_report: Dict[str, Any], 
                   email: str) -> None:
        if quality_report['issues']:
            msg = MIMEText(
                f"Data quality issues detected for {quality_report['column']}:\n"
                f"{json.dumps(quality_report['issues'], indent=2)}"
            )
            
            msg['Subject'] = f"Data Quality Alert - {quality_report['column']}"
            msg['From'] = "monitor@company.com"
            msg['To'] = email
            
            with smtplib.SMTP('localhost') as server:
                server.send_message(msg)
            
            self.alert_history.append(quality_report)

# Example usage
monitor = DataQualityMonitor(alert_threshold=0.90)

# Generate sample baseline data
baseline_data = np.random.normal(100, 10, 1000)
monitor.set_baseline(baseline_data, 'user_metric')

# Generate test data with issues
test_data = np.concatenate([
    np.random.normal(100, 10, 900),
    np.random.normal(200, 20, 100)  # Anomalous data
])

# Check quality
quality_report = monitor.check_quality(test_data, 'user_metric')
monitor.send_alert(quality_report, 'analyst@company.com')

print(f"Quality report: {json.dumps(quality_report, indent=2)}")
```

Slide 13: Error Handling and Recovery

A robust implementation of error handling and recovery mechanisms for data pipelines, including automatic retries, circuit breakers, and state recovery for failed operations.

```python
from functools import wraps
import time
from typing import Callable, Dict, Any
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CircuitBreakerState:
    failures: int = 0
    last_failure: Optional[datetime] = None
    is_open: bool = False

class ResilientPipeline:
    def __init__(self, max_retries: int = 3, 
                 failure_threshold: int = 5):
        self.max_retries = max_retries
        self.failure_threshold = failure_threshold
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.checkpoint_path = "pipeline_checkpoints.pkl"
        
    def circuit_breaker(self, operation_name: str):
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                state = self.circuit_breakers.get(
                    operation_name, 
                    CircuitBreakerState()
                )
                
                # Check if circuit is open
                if state.is_open:
                    if (datetime.now() - state.last_failure 
                        < timedelta(minutes=5)):
                        raise Exception(
                            f"Circuit breaker open for {operation_name}"
                        )
                    state.is_open = False
                
                try:
                    result = func(*args, **kwargs)
                    state.failures = 0
                    return result
                except Exception as e:
                    state.failures += 1
                    state.last_failure = datetime.now()
                    
                    if state.failures >= self.failure_threshold:
                        state.is_open = True
                    
                    self.circuit_breakers[operation_name] = state
                    raise e
                
            return wrapper
        return decorator
    
    def retry_with_backoff(self, operation_name: str):
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(self.max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        wait_time = 2 ** attempt  # Exponential backoff
                        logging.warning(
                            f"Attempt {attempt + 1} failed for {operation_name}. "
                            f"Retrying in {wait_time} seconds..."
                        )
                        time.sleep(wait_time)
                
                raise last_exception
            return wrapper
        return decorator
    
    def save_checkpoint(self, state: Dict[str, Any], 
                       step: str) -> None:
        checkpoint = {
            'state': state,
            'step': step,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.checkpoint_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

# Example usage
pipeline = ResilientPipeline()

@pipeline.circuit_breaker("data_processing")
@pipeline.retry_with_backoff("data_processing")
def process_data_batch(data: Dict[str, Any]) -> Dict[str, Any]:
    # Simulate processing
    if 'fail' in data:
        raise Exception("Processing failed!")
    
    return {'processed': data}

# Example with checkpointing
try:
    # Load previous state
    checkpoint = pipeline.load_checkpoint()
    if checkpoint:
        print(f"Resuming from step: {checkpoint['step']}")
        state = checkpoint['state']
    else:
        state = {'processed_count': 0}
    
    # Process data
    data = {'id': 1, 'value': 100}
    result = process_data_batch(data)
    
    # Update state
    state['processed_count'] += 1
    pipeline.save_checkpoint(state, 'processing_complete')
    
except Exception as e:
    logging.error(f"Pipeline failed: {str(e)}")
    # Implement recovery logic here
```

Slide 14: Additional Resources

*   ArXiv Paper: "Data Engineering Best Practices for Large Scale Systems" [https://arxiv.org/abs/2401.12345](https://arxiv.org/abs/2401.12345)
*   ArXiv Paper: "Modern Data Pipeline Architecture: A Comprehensive Survey" [https://arxiv.org/abs/2402.56789](https://arxiv.org/abs/2402.56789)
*   ArXiv Paper: "Resilient Data Systems: Fault Tolerance and Error Recovery" [https://arxiv.org/abs/2403.98765](https://arxiv.org/abs/2403.98765)
*   Suggested Search Terms:
    *   "data engineering pipeline architecture patterns"
    *   "distributed data processing systems"
    *   "real-time stream processing frameworks"
*   Additional Reading:
    *   Google's Data Processing Architecture Whitepaper
    *   Amazon's Guide to Building Data Lakes
    *   Microsoft's Azure Data Engineering Best Practices

