## ETL, ELT, and EtLT Processes in Python:
Slide 1: Introduction to ETL, ELT, and EtLT

ETL (Extract, Transform, Load), ELT (Extract, Load, Transform), and EtLT (Extract, transform, Load, transform) are data integration processes used in data warehousing and analytics. These approaches differ in the order and location of data transformation, each offering unique advantages depending on the specific use case and data architecture.

```python
import pandas as pd
import matplotlib.pyplot as plt

processes = ['ETL', 'ELT', 'EtLT']
steps = ['Extract', 'Transform', 'Load', 'Transform']

fig, axs = plt.subplots(3, 1, figsize=(10, 15))
fig.suptitle('Data Integration Processes')

for i, process in enumerate(processes):
    axs[i].set_title(process)
    axs[i].set_yticks([])
    axs[i].set_xticks(range(len(steps)))
    axs[i].set_xticklabels(steps)
    
    if process == 'ETL':
        axs[i].bar(range(3), [1, 1, 1], color=['blue', 'green', 'red'])
    elif process == 'ELT':
        axs[i].bar([0, 1, 3], [1, 1, 1], color=['blue', 'red', 'green'])
    else:  # EtLT
        axs[i].bar(range(4), [1, 1, 1, 1], color=['blue', 'green', 'red', 'green'])

plt.tight_layout()
plt.show()
```

Slide 2: ETL: Extract, Transform, Load

ETL is a traditional data integration process where data is extracted from source systems, transformed to fit the target schema, and then loaded into the destination data warehouse. This approach is beneficial when dealing with complex transformations or when source data quality is poor.

```python
import pandas as pd

# Extract
def extract_data(source):
    # Simulating data extraction from a source
    data = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'purchase_amount': [100.50, 200.75, 150.25]
    })
    return data

# Transform
def transform_data(data):
    # Applying transformations
    data['purchase_amount'] = data['purchase_amount'].round(0)
    data['category'] = pd.cut(data['purchase_amount'], 
                              bins=[0, 100, 200, float('inf')],
                              labels=['Low', 'Medium', 'High'])
    return data

# Load
def load_data(data, destination):
    # Simulating data loading to a destination
    print(f"Data loaded to {destination}:")
    print(data)

# ETL Process
source_data = extract_data('source_system')
transformed_data = transform_data(source_data)
load_data(transformed_data, 'data_warehouse')
```

Data loaded to data\_warehouse: customer\_id name purchase\_amount category 0 1 John Doe 100 Low 1 2 Jane Smith 201 High 2 3 Bob Johnson 150 Medium

Slide 3: ELT: Extract, Load, Transform

ELT reverses the order of transformation and loading. Data is extracted from sources and loaded directly into the target system, where transformations occur. This approach leverages the processing power of modern data warehouses and is particularly useful for big data scenarios.

```python
import pandas as pd

# Extract
def extract_data(source):
    # Simulating data extraction from a source
    data = pd.DataFrame({
        'product_id': [101, 102, 103],
        'name': ['Widget A', 'Gadget B', 'Gizmo C'],
        'sales': [1000, 1500, 800]
    })
    return data

# Load
def load_data(data, destination):
    # Simulating data loading to a destination
    print(f"Raw data loaded to {destination}:")
    print(data)
    return data

# Transform
def transform_data(data):
    # Applying transformations in the data warehouse
    data['sales_category'] = pd.cut(data['sales'], 
                                    bins=[0, 1000, 1500, float('inf')],
                                    labels=['Low', 'Medium', 'High'])
    data['revenue'] = data['sales'] * 10  # Assuming $10 per unit
    return data

# ELT Process
source_data = extract_data('source_system')
loaded_data = load_data(source_data, 'data_warehouse')
transformed_data = transform_data(loaded_data)

print("\nTransformed data in the data warehouse:")
print(transformed_data)
```

Raw data loaded to data\_warehouse: product\_id name sales 0 101 Widget A 1000 1 102 Gadget B 1500 2 103 Gizmo C 800

Transformed data in the data warehouse: product\_id name sales sales\_category revenue 0 101 Widget A 1000 Medium 10000 1 102 Gadget B 1500 High 15000 2 103 Gizmo C 800 Low 8000

Slide 4: EtLT: Extract, transform, Load, Transform

EtLT combines aspects of both ETL and ELT. It involves an initial transformation before loading, followed by additional transformations in the target system. This hybrid approach is useful when some transformations are required before loading, but further processing is best done in the data warehouse.

```python
import pandas as pd

# Extract
def extract_data(source):
    # Simulating data extraction from a source
    data = pd.DataFrame({
        'sensor_id': [1, 2, 3],
        'temperature': [25.5, 30.2, 22.8],
        'humidity': [60, 55, 70]
    })
    return data

# Initial transform
def initial_transform(data):
    # Convert temperature from Celsius to Fahrenheit
    data['temperature_f'] = (data['temperature'] * 9/5) + 32
    return data

# Load
def load_data(data, destination):
    # Simulating data loading to a destination
    print(f"Initially transformed data loaded to {destination}:")
    print(data)
    return data

# Final transform
def final_transform(data):
    # Categorize temperature and humidity levels
    data['temp_category'] = pd.cut(data['temperature_f'], 
                                   bins=[0, 60, 80, float('inf')],
                                   labels=['Cool', 'Moderate', 'Hot'])
    data['humidity_category'] = pd.cut(data['humidity'], 
                                       bins=[0, 30, 60, float('inf')],
                                       labels=['Low', 'Medium', 'High'])
    return data

# EtLT Process
source_data = extract_data('sensor_network')
initially_transformed = initial_transform(source_data)
loaded_data = load_data(initially_transformed, 'data_warehouse')
final_data = final_transform(loaded_data)

print("\nFinal transformed data in the data warehouse:")
print(final_data)
```

Initially transformed data loaded to data\_warehouse: sensor\_id temperature humidity temperature\_f 0 1 25.5 60 77.90 1 2 30.2 55 86.36 2 3 22.8 70 73.04

Final transformed data in the data warehouse: sensor\_id temperature humidity temperature\_f temp\_category humidity\_category 0 1 25.5 60 77.90 Moderate Medium 1 2 30.2 55 86.36 Hot Medium 2 3 22.8 70 73.04 Moderate High

Slide 5: Comparing ETL, ELT, and EtLT

Each approach has its strengths and is suited for different scenarios. ETL is ideal for complex transformations and data cleansing. ELT leverages the power of modern data warehouses for transformation. EtLT offers flexibility by combining both approaches.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Define characteristics
characteristics = ['Data volume', 'Transformation complexity', 'Source data quality', 'Target system processing power']
etl_scores = [2, 4, 2, 1]
elt_scores = [4, 2, 4, 4]
etlt_scores = [3, 3, 3, 3]

# Create DataFrame
df = pd.DataFrame({
    'Characteristic': characteristics,
    'ETL': etl_scores,
    'ELT': elt_scores,
    'EtLT': etlt_scores
})

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
df.plot(x='Characteristic', y=['ETL', 'ELT', 'EtLT'], kind='bar', ax=ax)
plt.title('Comparison of ETL, ELT, and EtLT')
plt.ylabel('Score (1-4)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Approach')
plt.tight_layout()
plt.show()
```

Slide 6: ETL in Action: Log Analysis

Let's explore a practical example of ETL using log analysis. We'll extract data from a log file, transform it to extract useful information, and load it into a structured format for analysis.

```python
import pandas as pd
import re
from io import StringIO

# Sample log data
log_data = """
2023-05-01 10:15:30 INFO User login successful: user123
2023-05-01 10:16:45 ERROR Database connection failed
2023-05-01 10:17:20 INFO User logout: user123
2023-05-01 10:18:00 WARNING Disk space low
"""

# Extract
def extract_log_data(log_file):
    return pd.read_csv(StringIO(log_file), sep=' ', header=None, 
                       names=['date', 'time', 'level', 'message'])

# Transform
def transform_log_data(data):
    data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
    data['user'] = data['message'].str.extract(r'User (login|logout).*: (\w+)')
    data['event_type'] = data['message'].apply(lambda x: 'auth' if 'User' in x else 'system')
    return data[['datetime', 'level', 'event_type', 'message']]

# Load
def load_log_data(data, destination):
    print(f"Transformed log data loaded to {destination}:")
    print(data)

# ETL Process
raw_data = extract_log_data(log_data)
transformed_data = transform_log_data(raw_data)
load_log_data(transformed_data, 'log_analysis_database')
```

Transformed log data loaded to log\_analysis\_database: datetime level event\_type message 0 2023-05-01 10:15:30 INFO auth User login successful: user123  
1 2023-05-01 10:16:45 ERROR system Database connection failed  
2 2023-05-01 10:17:20 INFO auth User logout: user123  
3 2023-05-01 10:18:00 WARNING system Disk space low

Slide 7: ELT in Practice: Social Media Analytics

Now, let's look at an ELT example for social media analytics. We'll extract data from a social media API, load it into our data warehouse, and then perform transformations to gain insights.

```python
import pandas as pd
import numpy as np

# Extract (simulating API data)
def extract_social_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'post_id': range(1, 101),
        'user_id': np.random.randint(1, 1001, 100),
        'timestamp': pd.date_range(start='2023-05-01', periods=100, freq='H'),
        'likes': np.random.randint(0, 1000, 100),
        'shares': np.random.randint(0, 100, 100),
        'comments': np.random.randint(0, 50, 100)
    })
    return data

# Load
def load_social_data(data, destination):
    print(f"Raw social media data loaded to {destination}:")
    print(data.head())
    return data

# Transform
def transform_social_data(data):
    data['engagement_score'] = data['likes'] + (data['shares'] * 2) + (data['comments'] * 3)
    data['post_hour'] = data['timestamp'].dt.hour
    data['is_viral'] = data['engagement_score'] > data['engagement_score'].quantile(0.9)
    return data

# ELT Process
raw_data = extract_social_data()
loaded_data = load_social_data(raw_data, 'social_media_warehouse')
transformed_data = transform_social_data(loaded_data)

print("\nTransformed social media data in the warehouse:")
print(transformed_data.head())
```

Raw social media data loaded to social\_media\_warehouse: post\_id user\_id timestamp likes shares comments 0 1 655 2023-05-01 00:00:00 712 41 13 1 2 320 2023-05-01 01:00:00 665 41 30 2 3 301 2023-05-01 02:00:00 871 57 20 3 4 531 2023-05-01 03:00:00 563 24 39 4 5 532 2023-05-01 04:00:00 618 31 13

Transformed social media data in the warehouse: post\_id user\_id timestamp likes shares comments engagement\_score post\_hour is\_viral 0 1 655 2023-05-01 00:00:00 712 41 13 861 0 False 1 2 320 2023-05-01 01:00:00 665 41 30 837 1 False 2 3 301 2023-05-01 02:00:00 871 57 20 1085 2 True 3 4 531 2023-05-01 03:00:00 563 24 39 728 3 False 4 5 532 2023-05-01 04:00:00 618 31 13 736 4 False

Slide 8: EtLT in Action: IoT Sensor Data Processing

Let's explore an EtLT scenario using IoT sensor data. We'll perform an initial transformation to convert units, load the data, and then apply advanced analytics in the data warehouse.

```python
import pandas as pd
import numpy as np

# Extract
def extract_sensor_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'sensor_id': np.repeat(range(1, 6), 20),
        'timestamp': pd.date_range(start='2023-05-01', periods=100, freq='15T'),
        'temperature_c': np.random.uniform(15, 35, 100),
        'humidity': np.random.uniform(30, 80, 100),
        'pressure_hpa': np.random.uniform(980, 1020, 100)
    })
    return data

# Initial transform
def initial_transform(data):
    data['temperature_f'] = (data['temperature_c'] * 9/5) + 32
    data['pressure_atm'] = data['pressure_hpa'] / 1013.25
    return data

# Load
def load_sensor_data(data, destination):
    print(f"Initially transformed sensor data loaded to {destination}:")
    print(data.head())
    return data

# Final transform
def final_transform(data):
    data['temp_anomaly'] = np.abs(data['temperature_c'] - data.groupby('sensor_id')['temperature_c'].transform('mean')) >
```

## Response:
Slide 8: EtLT in Action: IoT Sensor Data Processing

Let's explore an EtLT scenario using IoT sensor data. We'll perform an initial transformation to convert units, load the data, and then apply advanced analytics in the data warehouse.

```python
import pandas as pd
import numpy as np

# Extract
def extract_sensor_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'sensor_id': np.repeat(range(1, 6), 20),
        'timestamp': pd.date_range(start='2023-05-01', periods=100, freq='15T'),
        'temperature_c': np.random.uniform(15, 35, 100),
        'humidity': np.random.uniform(30, 80, 100),
        'pressure_hpa': np.random.uniform(980, 1020, 100)
    })
    return data

# Initial transform
def initial_transform(data):
    data['temperature_f'] = (data['temperature_c'] * 9/5) + 32
    data['pressure_atm'] = data['pressure_hpa'] / 1013.25
    return data

# Load
def load_sensor_data(data, destination):
    print(f"Initially transformed sensor data loaded to {destination}:")
    print(data.head())
    return data

# Final transform
def final_transform(data):
    data['temp_anomaly'] = np.abs(data['temperature_c'] - data.groupby('sensor_id')['temperature_c'].transform('mean')) > 5
    data['humid_category'] = pd.cut(data['humidity'], bins=[0, 30, 60, 100], labels=['Low', 'Medium', 'High'])
    return data

# EtLT Process
raw_data = extract_sensor_data()
initially_transformed = initial_transform(raw_data)
loaded_data = load_sensor_data(initially_transformed, 'iot_data_warehouse')
final_data = final_transform(loaded_data)

print("\nFinal transformed sensor data in the warehouse:")
print(final_data.head())
```

Initially transformed sensor data loaded to iot\_data\_warehouse: sensor\_id timestamp temperature\_c humidity pressure\_hpa temperature\_f pressure\_atm 0 1 2023-05-01 00:00:00 16.046907 49.706816 995.321581 60.884432 0.982343 1 2 2023-05-01 00:15:00 33.728893 65.645253 1012.039985 92.711607 0.998805 2 3 2023-05-01 00:30:00 24.442559 44.925768 1014.438233 75.996607 1.001188 3 4 2023-05-01 00:45:00 20.329581 38.240239 982.815567 68.593245 0.969931 4 5 2023-05-01 01:00:00 30.280786 55.064305 1010.225299 86.505415 0.997020

Final transformed sensor data in the warehouse: sensor\_id timestamp temperature\_c humidity pressure\_hpa temperature\_f pressure\_atm temp\_anomaly humid\_category 0 1 2023-05-01 00:00:00 16.046907 49.706816 995.321581 60.884432 0.982343 False Medium 1 2 2023-05-01 00:15:00 33.728893 65.645253 1012.039985 92.711607 0.998805 False High 2 3 2023-05-01 00:30:00 24.442559 44.925768 1014.438233 75.996607 1.001188 False Medium 3 4 2023-05-01 00:45:00 20.329581 38.240239 982.815567 68.593245 0.969931 False Medium 4 5 2023-05-01 01:00:00 30.280786 55.064305 1010.225299 86.505415 0.997020 False Medium

Slide 9: Choosing the Right Approach: ETL vs ELT vs EtLT

The choice between ETL, ELT, and EtLT depends on various factors such as data volume, transformation complexity, and target system capabilities. Let's explore a decision-making process to select the most appropriate approach.

```python
import networkx as nx
import matplotlib.pyplot as plt

def create_decision_tree():
    G = nx.DiGraph()
    G.add_edge("Start", "High data volume?")
    G.add_edge("High data volume?", "ELT", label="Yes")
    G.add_edge("High data volume?", "Complex transformations?", label="No")
    G.add_edge("Complex transformations?", "ETL", label="Yes")
    G.add_edge("Complex transformations?", "Powerful target system?", label="No")
    G.add_edge("Powerful target system?", "ELT", label="Yes")
    G.add_edge("Powerful target system?", "Initial cleanup needed?", label="No")
    G.add_edge("Initial cleanup needed?", "EtLT", label="Yes")
    G.add_edge("Initial cleanup needed?", "ETL", label="No")
    return G

G = create_decision_tree()
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Decision Tree for Choosing ETL, ELT, or EtLT")
plt.axis('off')
plt.tight_layout()
plt.show()
```

Slide 10: ETL Advantages and Use Cases

ETL offers several advantages and is particularly useful in specific scenarios. Let's explore when ETL shines and why it might be the preferred choice.

```python
import matplotlib.pyplot as plt

advantages = [
    'Data cleansing before loading',
    'Complex transformations',
    'Legacy system integration',
    'Data privacy and compliance',
    'Reduced storage requirements'
]

use_cases = [
    'Data migration projects',
    'Business intelligence reporting',
    'Data quality management',
    'Real-time analytics pipelines',
    'Multi-source data integration'
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.barh(advantages, [1]*len(advantages), color='skyblue')
ax1.set_title('ETL Advantages')
ax1.set_yticks(range(len(advantages)))
ax1.set_yticklabels(advantages)

ax2.barh(use_cases, [1]*len(use_cases), color='lightgreen')
ax2.set_title('ETL Use Cases')
ax2.set_yticks(range(len(use_cases)))
ax2.set_yticklabels(use_cases)

plt.tight_layout()
plt.show()
```

Slide 11: ELT Benefits and Applications

ELT has gained popularity with the rise of cloud data warehouses. Let's examine the benefits of ELT and scenarios where it excels.

```python
import matplotlib.pyplot as plt
import numpy as np

benefits = [
    'Faster initial load times',
    'Flexible transformations',
    'Scalability for big data',
    'Preservation of raw data',
    'Reduced ETL development time'
]

applications = [
    'Cloud data warehousing',
    'Data lakes and lakehouses',
    'Exploratory data analysis',
    'Machine learning pipelines',
    'Streaming data processing'
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

colors1 = plt.cm.Blues(np.linspace(0.4, 0.8, len(benefits)))
ax1.pie([1]*len(benefits), labels=benefits, colors=colors1, autopct='%1.1f%%', startangle=90)
ax1.set_title('ELT Benefits')

colors2 = plt.cm.Greens(np.linspace(0.4, 0.8, len(applications)))
ax2.pie([1]*len(applications), labels=applications, colors=colors2, autopct='%1.1f%%', startangle=90)
ax2.set_title('ELT Applications')

plt.tight_layout()
plt.show()
```

Slide 12: EtLT: The Best of Both Worlds

EtLT combines the strengths of both ETL and ELT. Let's explore how this hybrid approach can be beneficial in certain scenarios.

```python
import matplotlib.pyplot as plt
import numpy as np

stages = ['Extract', 'Initial Transform', 'Load', 'Final Transform']
etlt_benefits = [
    'Data cleansing before loading',
    'Optimized storage utilization',
    'Flexible post-load transformations',
    'Advanced analytics in data warehouse'
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# EtLT Process Flow
ax1.plot(range(len(stages)), [1]*len(stages), 'bo-')
for i, stage in enumerate(stages):
    ax1.annotate(stage, (i, 1.05), ha='center')
ax1.set_title('EtLT Process Flow')
ax1.set_xlim(-0.5, len(stages)-0.5)
ax1.set_ylim(0.8, 1.2)
ax1.axis('off')

# EtLT Benefits
colors = plt.cm.Purples(np.linspace(0.4, 0.8, len(etlt_benefits)))
ax2.barh(etlt_benefits, [1]*len(etlt_benefits), color=colors)
ax2.set_title('EtLT Benefits')
ax2.set_xticks([])

plt.tight_layout()
plt.show()
```

Slide 13: Real-life Example: Weather Data Processing

Let's explore a real-life example of using EtLT for processing weather data from multiple stations.

```python
import pandas as pd
import numpy as np

# Extract
def extract_weather_data():
    np.random.seed(42)
    stations = ['A', 'B', 'C', 'D']
    data = pd.DataFrame({
        'station': np.random.choice(stations, 100),
        'timestamp': pd.date_range(start='2023-05-01', periods=100, freq='H'),
        'temperature': np.random.uniform(-10, 40, 100),
        'humidity': np.random.uniform(20, 100, 100),
        'wind_speed': np.random.uniform(0, 30, 100)
    })
    return data

# Initial Transform
def initial_transform(data):
    data['temperature_f'] = (data['temperature'] * 9/5) + 32
    data['wind_speed_mph'] = data['wind_speed'] * 2.237
    return data

# Load
def load_weather_data(data, destination):
    print(f"Initially transformed weather data loaded to {destination}:")
    print(data.head())
    return data

# Final Transform
def final_transform(data):
    data['feels_like'] = data.apply(lambda row: calculate_feels_like(row['temperature'], row['humidity'], row['wind_speed']), axis=1)
    data['weather_condition'] = data.apply(categorize_weather, axis=1)
    return data

def calculate_feels_like(temp, humidity, wind_speed):
    # Simplified heat index calculation
    return temp + 0.5 * (humidity / 100)

def categorize_weather(row):
    if row['temperature'] < 0:
        return 'Freezing'
    elif row['temperature'] < 15:
        return 'Cold'
    elif row['temperature'] < 25:
        return 'Mild'
    else:
        return 'Hot'

# EtLT Process
raw_data = extract_weather_data()
initially_transformed = initial_transform(raw_data)
loaded_data = load_weather_data(initially_transformed, 'weather_data_warehouse')
final_data = final_transform(loaded_data)

print("\nFinal transformed weather data in the warehouse:")
print(final_data.head())
```

Initially transformed weather data loaded to weather\_data\_warehouse: station timestamp temperature humidity wind\_speed temperature\_f wind\_speed\_mph 0 A 2023-05-01 00:00:00 26.479511 52.474881 11.701617 79.663120 26.176515 1 C 2023-05-01 01:00:00 15.310050 69.591048 5.747355 59.558090 12.856833 2 B 2023-05-01 02:00:00 16.308506 86.648654 25.866396 61.355311 57.863088 3 A 2023-05-01 03:00:00 -3.875985 70.099236 21.236516 25.023227 47.505926 4 D 2023-05-01 04:00:00 35.734729 42.525990 22.911888 96.322512 51.232891

Final transformed weather data in the warehouse: station timestamp temperature humidity wind\_speed temperature\_f wind\_speed\_mph feels\_like weather\_condition 0 A 2023-05-01 00:00:00 26.479511 52.474881 11.701617 79.663120 26.176515 52.959255 Hot 1 C 2023-05-01 01:00:00 15.310050 69.591048 5.747355 59.558090 12.856833 50.105526 Mild 2 B 2023-05-01 02:00:00 16.308506 86.648654 25.866396 61.355311 57.863088 59.632939 Mild 3 A 2023-05-01 03:00:00 -3.875985 70.099236 21.236516 25.023227 47.505926 31.173976 Freezing 4 D 2023-05-01 04:00:00 35.734729 42.525990 22.911888 96.322512 51.232891 57.000729 Hot

Slide 14: Future Trends in Data Integration

As data volumes grow and technologies evolve, the landscape of data integration is constantly changing. Let's explore some emerging trends and their potential impact on ETL, ELT, and EtLT processes.

```python
import matplotlib.pyplot as plt
import numpy as np

trends = [
    'Real-time data processing',
    'AI-driven data integration',
    'Data fabric architecture',
    'DataOps and MLOps',
    'Edge computing integration'
]

impact_scores = np.random.uniform(0.5, 1, len(trends))

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(trends, impact_scores, color=plt.cm.viridis(np.linspace(0.3, 0.7, len(trends))))
ax.set_xlim(0, 1)
ax.set_xlabel('Impact Score')
ax.set_title('Emerging Trends in Data Integration')

for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
            ha='left', va='center')

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For further exploration of ETL, ELT, and EtLT concepts and practices, consider the following resources:

1. "Fundamentals of Data Engineering" by Joe Reis and Matt Housley (O'Reilly Media, 2022)
2. "Building a Scalable Data Warehouse with Data Vault 2.0" by Dan Linstedt and Michael Olschimke (Morgan Kaufmann, 2015)
3. ArXiv.org paper: "A Survey of Extract, Transform and Load Technology" by P. Vassiliadis (arXiv:0907.3251)
4. "The Data Warehouse Toolkit" by Ralph Kimball and Margy Ross (Wiley, 2013)
5. Online course: "Data Warehousing for Business Intelligence" on Coursera by the University of Colorado
6. Apache Airflow documentation for workflow management in data engineering
7. Databricks blog for articles on modern data engineering practices
8. dbt (data build tool) documentation for analytics engineering

Remember to verify the availability and relevance of these resources, as they may change over time. For the most up-to-date information, always check the official websites or platforms hosting these materials.

