## Building End-to-End Data Pipelines with Python
Slide 1: 
Introduction to Data Pipelines

A data pipeline is a series of steps that move and transform data from various sources to a destination where it can be analyzed or used. Python offers powerful libraries and tools for building efficient and scalable data pipelines. This presentation will guide you through the process of creating end-to-end data pipelines using Python.

```python
# Basic structure of a data pipeline
def extract_data():
    # Code to extract data from source
    pass

def transform_data(raw_data):
    # Code to clean and transform data
    pass

def load_data(transformed_data):
    # Code to load data into destination
    pass

def run_pipeline():
    raw_data = extract_data()
    transformed_data = transform_data(raw_data)
    load_data(transformed_data)

run_pipeline()
```

Slide 2: 
Setting Up the Environment

Before we begin building our data pipeline, we need to set up our Python environment. We'll use virtual environments to isolate our project dependencies and install necessary libraries.

```python
# Create and activate a virtual environment
python -m venv data_pipeline_env
source data_pipeline_env/bin/activate  # On Windows: data_pipeline_env\Scripts\activate

# Install required libraries
pip install pandas numpy requests sqlalchemy

# Verify installations
import pandas as pd
import numpy as np
import requests
from sqlalchemy import create_engine

print("Environment setup complete!")
```

Slide 3: 
Data Extraction: API Requests

The first step in our data pipeline is extracting data from various sources. Let's start by fetching data from an API using the requests library.

```python
import requests

def extract_data_from_api(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status code: {response.status_code}")

# Example usage
api_url = "https://api.example.com/data"
raw_data = extract_data_from_api(api_url)
print(f"Extracted {len(raw_data)} records from API")
```

Slide 4: 
Data Extraction: Reading from Files

Another common source of data is files. Let's use pandas to read data from CSV and Excel files.

```python
import pandas as pd

def extract_data_from_csv(file_path):
    return pd.read_csv(file_path)

def extract_data_from_excel(file_path):
    return pd.read_excel(file_path)

# Example usage
csv_data = extract_data_from_csv("data.csv")
excel_data = extract_data_from_excel("data.xlsx")

print(f"CSV data shape: {csv_data.shape}")
print(f"Excel data shape: {excel_data.shape}")
```

Slide 5: 
Data Transformation: Cleaning and Preprocessing

After extracting data, we often need to clean and preprocess it. This step involves handling missing values, removing duplicates, and formatting data types.

```python
import pandas as pd

def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    # Convert date columns to datetime
    date_columns = ['date_column1', 'date_column2']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    
    return df

# Example usage
raw_data = pd.read_csv("raw_data.csv")
cleaned_data = clean_data(raw_data)
print(f"Cleaned data shape: {cleaned_data.shape}")
```

Slide 6: 
Data Transformation: Feature Engineering

Feature engineering involves creating new features or modifying existing ones to improve the quality of our data. This process can significantly impact the performance of downstream analyses or machine learning models.

```python
import pandas as pd

def engineer_features(df):
    # Create a new feature
    df['total_amount'] = df['quantity'] * df['price']
    
    # Extract components from datetime
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Bin a continuous variable
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 30, 50, 100], labels=['0-18', '19-30', '31-50', '51+'])
    
    return df

# Example usage
data = pd.read_csv("sales_data.csv")
data['date'] = pd.to_datetime(data['date'])
engineered_data = engineer_features(data)
print(engineered_data.head())
```

Slide 7: 
Data Loading: Saving to CSV

After transforming our data, we need to load it into a destination where it can be used for analysis or further processing. Let's start with a simple example of saving the data to a CSV file.

```python
import pandas as pd

def save_to_csv(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

# Example usage
transformed_data = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['x', 'y', 'z']
})
save_to_csv(transformed_data, "output_data.csv")
```

Slide 8: 
Data Loading: Writing to a Database

For more robust data storage and retrieval, we can load our data into a database. We'll use SQLAlchemy to connect to and write data to a SQL database.

```python
from sqlalchemy import create_engine
import pandas as pd

def load_to_database(df, table_name, connection_string):
    engine = create_engine(connection_string)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"Data loaded to table: {table_name}")

# Example usage
connection_string = "sqlite:///my_database.db"
data_to_load = pd.DataFrame({
    'id': [1, 2, 3],
    'value': [10, 20, 30]
})
load_to_database(data_to_load, "my_table", connection_string)
```

Slide 9: 
Parallel Processing with multiprocessing

To improve the performance of our data pipeline, we can use parallel processing to execute tasks concurrently. Python's multiprocessing module allows us to leverage multiple CPU cores.

```python
import multiprocessing
import pandas as pd

def process_chunk(chunk):
    # Perform some computations on the chunk
    return chunk.apply(lambda x: x ** 2)

def parallel_processing(df, num_processes=4):
    pool = multiprocessing.Pool(processes=num_processes)
    chunks = np.array_split(df, num_processes)
    results = pool.map(process_chunk, chunks)
    return pd.concat(results)

# Example usage
data = pd.DataFrame({'A': range(1000000)})
processed_data = parallel_processing(data)
print(processed_data.head())
```

Slide 10: 
Error Handling and Logging

Robust data pipelines should include error handling and logging to help diagnose and fix issues. Let's implement these features in our pipeline.

```python
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

@error_handler
def risky_operation(x):
    if x == 0:
        raise ValueError("Cannot divide by zero")
    return 10 / x

# Example usage
for i in range(-1, 2):
    try:
        result = risky_operation(i)
        logging.info(f"Result: {result}")
    except Exception:
        logging.info("Moving to next iteration")
```

Slide 11: 
Data Validation with Great Expectations

Ensuring data quality is crucial in data pipelines. Great Expectations is a powerful library for validating, documenting, and profiling your data.

```python
import great_expectations as ge

def validate_data(df):
    ge_df = ge.from_pandas(df)
    
    # Define expectations
    ge_df.expect_column_values_to_be_between("age", min_value=0, max_value=120)
    ge_df.expect_column_values_to_not_be_null("name")
    ge_df.expect_column_values_to_be_in_set("gender", ["M", "F", "Other"])
    
    # Run validation
    results = ge_df.validate()
    return results

# Example usage
data = pd.DataFrame({
    'name': ['Alice', 'Bob', None],
    'age': [25, 40, -5],
    'gender': ['F', 'M', 'Unknown']
})

validation_results = validate_data(data)
print(f"Validation successful: {validation_results.success}")
print(f"Number of expectations: {len(validation_results.results)}")
```

Slide 12: 
Scheduling with Apache Airflow

For complex data pipelines that need to run on a schedule, Apache Airflow provides a powerful framework for orchestrating and monitoring workflows.

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 20),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'my_data_pipeline',
    default_args=default_args,
    description='A simple data pipeline DAG',
    schedule_interval=timedelta(days=1),
)

def extract():
    # Data extraction code here
    pass

def transform():
    # Data transformation code here
    pass

def load():
    # Data loading code here
    pass

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load,
    dag=dag,
)

extract_task >> transform_task >> load_task
```

Slide 13: 
Monitoring and Alerting

Monitoring your data pipeline's performance and setting up alerts for potential issues is crucial for maintaining reliability. Here's an example of how to implement basic monitoring and alerting using the smtplib library for sending email notifications.

```python
import time
import smtplib
from email.mime.text import MIMEText

def monitor_pipeline(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        if execution_time > 300:  # Alert if execution takes more than 5 minutes
            send_alert(f"Pipeline {func.__name__} took {execution_time:.2f} seconds to execute")
        
        return result
    return wrapper

def send_alert(message):
    sender = "alert@example.com"
    recipient = "admin@example.com"
    msg = MIMEText(message)
    msg['Subject'] = "Data Pipeline Alert"
    msg['From'] = sender
    msg['To'] = recipient
    
    with smtplib.SMTP('smtp.example.com', 587) as server:
        server.starttls()
        server.login(sender, "password")
        server.send_message(msg)

@monitor_pipeline
def run_pipeline():
    # Your pipeline code here
    time.sleep(310)  # Simulating a long-running pipeline

run_pipeline()
```

Slide 14: 
Putting It All Together

Now that we've covered various aspects of building data pipelines, let's combine these concepts into a complete end-to-end pipeline.

```python
import pandas as pd
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_data(file_path):
    logging.info(f"Extracting data from {file_path}")
    return pd.read_csv(file_path)

def transform_data(df):
    logging.info("Transforming data")
    df['total'] = df['quantity'] * df['price']
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df

def load_data(df, table_name, connection_string):
    logging.info(f"Loading data to {table_name}")
    engine = create_engine(connection_string)
    df.to_sql(table_name, engine, if_exists='replace', index=False)

def run_pipeline(input_file, output_table, db_connection):
    try:
        raw_data = extract_data(input_file)
        transformed_data = transform_data(raw_data)
        load_data(transformed_data, output_table, db_connection)
        logging.info("Pipeline completed successfully")
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")

# Example usage
input_file = "sales_data.csv"
output_table = "processed_sales"
db_connection = "sqlite:///sales_database.db"

run_pipeline(input_file, output_table, db_connection)
```

Slide 15: 
Additional Resources

To further enhance your understanding of data pipelines and related topics, consider exploring these peer-reviewed articles from arXiv.org:

1. "A Survey on Data Pipeline Management: Concepts, Taxonomies, and Systems" (arXiv:2107.05766) [https://arxiv.org/abs/2107.05766](https://arxiv.org/abs/2107.05766)
2. "Automated Machine Learning: State-of-The-Art and Open Challenges" (arXiv:1906.02287) [https://arxiv.org/abs/1906.02287](https://arxiv.org/abs/1906.02287)
3. "A Survey of Deep Learning Techniques for Neural Machine Translation" (arXiv:2002.07526) [https://arxiv.org/abs/2002.07526](https://arxiv.org/abs/2002.07526)

These resources provide in-depth insights into various aspects of data pipeline management, automated machine learning, and deep learning techniques that can be applied to enhance your data processing workflows.

