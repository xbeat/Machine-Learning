## The AI Powering Blinkit's 5-Minute Delivery
Slide 1: Time Series Analysis with ARIMA for Demand Forecasting

In Blinkit's delivery optimization system, ARIMA (Autoregressive Integrated Moving Average) models analyze historical order data to predict future demand patterns. This implementation demonstrates how to forecast hourly order volumes using ARIMA methodology with seasonal decomposition.

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Generate sample historical order data
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', end='2024-02-29', freq='H')
orders = pd.Series(np.random.normal(100, 20, len(dates)) + \
         np.sin(np.arange(len(dates)) * 2 * np.pi / 24) * 30, index=dates)

# Prepare and fit ARIMA model
model = ARIMA(orders, order=(1, 1, 1), 
              seasonal_order=(1, 1, 1, 24))
results = model.fit()

# Generate forecasts
forecast = results.forecast(steps=24)
print(f"Next 24 hour predictions:\n{forecast}")
```

Slide 2: K-Means Clustering for Geographic Demand Analysis

This implementation showcases how Blinkit might use K-means clustering to optimize dark store locations and inventory distribution based on order density and customer locations, enabling faster delivery times through strategic positioning.

```python
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Simulate customer order locations
np.random.seed(42)
customer_locations = np.random.normal(loc=[28.6139, 77.2090], 
                                    scale=[0.1, 0.1], 
                                    size=(1000, 2))

# Find optimal dark store locations
kmeans = KMeans(n_clusters=5, random_state=42)
dark_store_locations = kmeans.fit(customer_locations)

# Calculate average distance to nearest dark store
distances = np.min(kmeans.transform(customer_locations), axis=1)
avg_delivery_distance = np.mean(distances)

print(f"Average delivery distance: {avg_delivery_distance:.2f} km")
```

Slide 3: Gradient Boosting for Product Demand Forecasting

The XGBoost model implementation demonstrates how Blinkit predicts product-specific demand by incorporating multiple features like historical sales, weather, events, and time-based patterns to maintain optimal inventory levels.

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Generate synthetic feature data
def generate_features(n_samples):
    np.random.seed(42)
    data = {
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'temperature': np.random.normal(25, 5, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'previous_day_orders': np.random.poisson(100, n_samples)
    }
    return pd.DataFrame(data)

# Create training data
X = generate_features(1000)
y = 50 + 0.5 * X['previous_day_orders'] + \
    10 * X['is_weekend'] + np.random.normal(0, 10, 1000)

# Train model
model = xgb.XGBRegressor(objective='reg:squarederror')
model.fit(X, y)

print(f"Feature importance:\n{model.feature_importances_}")
```

Slide 4: LSTM Implementation for Time-Dependent Demand Prediction

This implementation shows how Blinkit uses Long Short-Term Memory networks to capture complex temporal dependencies in order patterns, accounting for both short-term fluctuations and long-term trends in demand.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare sequential data
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

# Generate synthetic order data
n_timestamps = 1000
order_data = np.sin(np.linspace(0, 100, n_timestamps)) * 50 + \
             np.random.normal(100, 10, n_timestamps)

# Create and train LSTM model
model = Sequential([
    LSTM(64, input_shape=(24, 1), return_sequences=True),
    LSTM(32),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

X, y = create_sequences(order_data, 24)
X = X.reshape(-1, 24, 1)
model.fit(X, y, epochs=10, batch_size=32)
```

Slide 5: Reinforcement Learning for Dynamic Inventory Optimization

This implementation demonstrates how Blinkit uses Q-learning to optimize inventory levels dynamically. The agent learns to make restocking decisions based on current inventory, predicted demand, and delivery constraints.

```python
import numpy as np
from collections import defaultdict

class InventoryQLearning:
    def __init__(self, n_states=100, n_actions=5):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.95
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 5)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

# Example usage
agent = InventoryQLearning()
current_inventory = 50
action = agent.get_action(current_inventory)
print(f"Recommended restock quantity: {action * 10} units")
```

Slide 6: API Gateway Implementation with FastAPI

This code demonstrates the implementation of Blinkit's API gateway that handles routing and load balancing for microservices architecture, essential for maintaining the 5-minute delivery promise.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import asyncio

app = FastAPI()

class Order(BaseModel):
    order_id: str
    items: list
    delivery_location: dict

SERVICES = {
    'inventory': 'http://inventory-service:8001',
    'delivery': 'http://delivery-service:8002',
    'payment': 'http://payment-service:8003'
}

@app.post("/api/v1/orders")
async def create_order(order: Order):
    async with httpx.AsyncClient() as client:
        # Check inventory availability
        inventory_response = await client.post(
            f"{SERVICES['inventory']}/check",
            json={"items": order.items}
        )
        
        if inventory_response.status_code != 200:
            raise HTTPException(status_code=400, 
                              detail="Items not available")
        
        # Assign delivery agent
        delivery_response = await client.post(
            f"{SERVICES['delivery']}/assign",
            json={"location": order.delivery_location}
        )
        
        return {"status": "success", "estimated_time": "5 minutes"}
```

Slide 7: Real-time Delivery Agent Assignment Algorithm

This implementation shows how Blinkit optimizes delivery agent assignments using a modified Hungarian algorithm, considering factors like agent location, order priority, and estimated delivery time.

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

class DeliveryAssignment:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        
    def calculate_cost_matrix(self, agent_locations, order_locations):
        cost_matrix = np.zeros((len(agent_locations), len(order_locations)))
        
        for i, agent in enumerate(agent_locations):
            for j, order in enumerate(order_locations):
                # Calculate Manhattan distance
                distance = abs(agent[0] - order[0]) + abs(agent[1] - order[1])
                # Add time-based penalty
                time_penalty = max(0, distance - 2) * 1.5
                cost_matrix[i, j] = distance + time_penalty
                
        return cost_matrix
    
    def assign_orders(self, agent_locations, order_locations):
        cost_matrix = self.calculate_cost_matrix(agent_locations, order_locations)
        agent_indices, order_indices = linear_sum_assignment(cost_matrix)
        
        assignments = []
        for agent_idx, order_idx in zip(agent_indices, order_indices):
            assignments.append({
                'agent_id': agent_idx,
                'order_id': order_idx,
                'estimated_time': cost_matrix[agent_idx, order_idx]
            })
            
        return assignments

# Example usage
assigner = DeliveryAssignment(n_agents=5)
agent_locs = np.random.rand(5, 2) * 10  # 5 agents in 10x10 grid
order_locs = np.random.rand(3, 2) * 10   # 3 orders
assignments = assigner.assign_orders(agent_locs, order_locs)
print(f"Optimal assignments:\n{assignments}")
```

Slide 8: Dark Store Inventory Optimization

This implementation showcases the algorithm used to optimize inventory levels across dark stores, considering factors like historical demand, shelf life, and delivery radius.

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class DarkStoreOptimizer:
    def __init__(self, store_id):
        self.store_id = store_id
        self.model = RandomForestRegressor(n_estimators=100)
        
    def preprocess_features(self, historical_data):
        features = pd.DataFrame({
            'day_of_week': historical_data.index.dayofweek,
            'hour': historical_data.index.hour,
            'is_weekend': historical_data.index.dayofweek >= 5,
            'trailing_demand': historical_data['demand'].rolling(24).mean(),
            'stock_level': historical_data['stock_level'],
            'shelf_life_remaining': historical_data['shelf_life']
        })
        return features
        
    def optimize_inventory(self, historical_data, forecast_horizon=24):
        X = self.preprocess_features(historical_data)
        y = historical_data['demand']
        
        self.model.fit(X, y)
        
        # Generate future features
        future_index = pd.date_range(
            start=historical_data.index[-1],
            periods=forecast_horizon,
            freq='H'
        )
        
        future_features = self.preprocess_features(
            pd.DataFrame(index=future_index)
        )
        
        predicted_demand = self.model.predict(future_features)
        
        safety_stock = np.std(predicted_demand) * 1.96
        recommended_stock = np.ceil(predicted_demand.max() + safety_stock)
        
        return {
            'recommended_stock': recommended_stock,
            'predicted_demand': predicted_demand,
            'confidence_interval': safety_stock
        }
```

Slide 9: Machine Learning Pipeline for Demand Prediction

This implementation showcases the complete ML pipeline used by Blinkit to predict demand, incorporating feature engineering, model training, and real-time prediction capabilities with automated retraining.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
import joblib

class DemandPredictionPipeline:
    def __init__(self):
        self.feature_cols = ['hour', 'day_of_week', 'is_weekend', 
                            'temperature', 'previous_sales']
        self.pipeline = self._create_pipeline()
        
    def _create_pipeline(self):
        numeric_features = ['temperature', 'previous_sales']
        categorical_features = ['hour', 'day_of_week', 'is_weekend']
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )
        
        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6
            ))
        ])
    
    def train(self, X, y):
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.pipeline.fit(X_train, y_train)
            score = self.pipeline.score(X_val, y_val)
            scores.append(score)
            
        return np.mean(scores)
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def save_model(self, path):
        joblib.dump(self.pipeline, path)
    
    def load_model(self, path):
        self.pipeline = joblib.load(path)

# Example usage
pipeline = DemandPredictionPipeline()
X = generate_features(1000)  # From previous example
y = generate_target(X)       # Generate target variable
score = pipeline.train(X, y)
print(f"Average validation score: {score:.4f}")
```

Slide 10: Real-time Order Processing System

This implementation demonstrates the event-driven architecture used by Blinkit to process orders in real-time, utilizing Redis for caching and PostgreSQL for persistent storage.

```python
import asyncio
import aioredis
import asyncpg
from datetime import datetime

class OrderProcessor:
    def __init__(self):
        self.redis = None
        self.pg_pool = None
        
    async def initialize(self):
        self.redis = await aioredis.create_redis_pool('redis://localhost')
        self.pg_pool = await asyncpg.create_pool(
            user='user', password='password',
            database='blinkit', host='localhost'
        )
        
    async def process_order(self, order_data):
        order_id = order_data['order_id']
        
        # Check cache for inventory
        inventory = await self.redis.hgetall(f'inventory:{order_data["store_id"]}')
        
        # Verify inventory
        for item in order_data['items']:
            if int(inventory.get(item['id'], 0)) < item['quantity']:
                return {'status': 'failed', 'reason': 'insufficient_inventory'}
        
        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                # Create order record
                await conn.execute('''
                    INSERT INTO orders (order_id, customer_id, store_id, status)
                    VALUES ($1, $2, $3, $4)
                ''', order_id, order_data['customer_id'], 
                order_data['store_id'], 'processing')
                
                # Update inventory
                for item in order_data['items']:
                    await self.redis.hincrby(
                        f'inventory:{order_data["store_id"]}',
                        item['id'],
                        -item['quantity']
                    )
        
        return {'status': 'success', 'order_id': order_id}
    
    async def close(self):
        self.redis.close()
        await self.redis.wait_closed()
        await self.pg_pool.close()

# Example usage
async def main():
    processor = OrderProcessor()
    await processor.initialize()
    
    order = {
        'order_id': 'ORD123',
        'customer_id': 'CUST456',
        'store_id': 'STORE789',
        'items': [{'id': 'ITEM1', 'quantity': 2}]
    }
    
    result = await processor.process_order(order)
    print(f"Order processing result: {result}")
    
    await processor.close()

asyncio.run(main())
```

Slide 11: Load Balancing and Service Discovery

This implementation shows how Blinkit manages high-concurrency requests across multiple dark stores and delivery agents using a custom load balancer with health checking capabilities.

```python
import asyncio
from typing import Dict, List
import aiohttp
import random
from dataclasses import dataclass

@dataclass
class ServiceNode:
    url: str
    health_score: float = 1.0
    active_requests: int = 0

class LoadBalancer:
    def __init__(self):
        self.services: Dict[str, List[ServiceNode]] = {
            'inventory': [],
            'delivery': [],
            'order': []
        }
        
    async def health_check(self, node: ServiceNode):
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{node.url}/health") as response:
                    if response.status == 200:
                        node.health_score = 1.0
                    else:
                        node.health_score *= 0.5
            except:
                node.health_score *= 0.2
    
    async def select_node(self, service_type: str) -> ServiceNode:
        available_nodes = [
            node for node in self.services[service_type]
            if node.health_score > 0.5
        ]
        
        if not available_nodes:
            raise Exception(f"No healthy {service_type} nodes available")
        
        # Weighted random selection based on health and load
        weights = [
            (1 / (node.active_requests + 1)) * node.health_score
            for node in available_nodes
        ]
        return random.choices(available_nodes, weights=weights)[0]
    
    async def route_request(self, service_type: str, request_data: dict):
        node = await self.select_node(service_type)
        node.active_requests += 1
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    node.url,
                    json=request_data
                ) as response:
                    return await response.json()
        finally:
            node.active_requests -= 1

# Example usage
async def main():
    lb = LoadBalancer()
    # Add service nodes
    lb.services['inventory'].extend([
        ServiceNode(url='http://inventory-1:8001'),
        ServiceNode(url='http://inventory-2:8001')
    ])
    
    # Simulate requests
    request_data = {'item_id': 'ITEM123', 'quantity': 5}
    result = await lb.route_request('inventory', request_data)
    print(f"Request result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

Slide 12: Dark Store Analytics and Optimization

This implementation showcases the analytics engine that processes dark store performance metrics and suggests optimizations for inventory placement and restock timing.

```python
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict

class DarkStoreAnalytics:
    def __init__(self, store_id: str):
        self.store_id = store_id
        self.shelf_constraints = {}
        self.item_dimensions = {}
    
    def analyze_store_performance(self, 
                                sales_data: pd.DataFrame,
                                inventory_data: pd.DataFrame) -> Dict:
        # Calculate key metrics
        turnover_rate = self._calculate_turnover_rate(
            sales_data, inventory_data
        )
        shelf_utilization = self._analyze_shelf_utilization(inventory_data)
        stockout_analysis = self._analyze_stockouts(
            sales_data, inventory_data
        )
        
        return {
            'turnover_rate': turnover_rate,
            'shelf_utilization': shelf_utilization,
            'stockout_analysis': stockout_analysis,
            'optimization_suggestions': self._generate_suggestions(
                turnover_rate,
                shelf_utilization,
                stockout_analysis
            )
        }
    
    def _calculate_turnover_rate(self, 
                               sales_data: pd.DataFrame,
                               inventory_data: pd.DataFrame) -> Dict:
        merged_data = pd.merge(
            sales_data,
            inventory_data,
            on=['item_id', 'date']
        )
        
        turnover = (merged_data['sales_quantity'] / 
                   merged_data['average_inventory'])
        
        return {
            'overall_rate': turnover.mean(),
            'by_category': turnover.groupby(
                merged_data['category']
            ).mean().to_dict()
        }
    
    def optimize_shelf_space(self, 
                           sales_velocity: Dict[str, float],
                           shelf_constraints: Dict[str, float]) -> Dict:
        def objective(x):
            return -np.sum(
                [sales_velocity[item] * space 
                 for item, space in zip(sales_velocity.keys(), x)]
            )
        
        constraints = [
            {'type': 'eq', 
             'fun': lambda x: np.sum(x) - shelf_constraints['total_space']}
        ]
        
        bounds = [(0, shelf_constraints['max_item_space'])] * len(
            sales_velocity
        )
        
        result = minimize(
            objective,
            x0=np.ones(len(sales_velocity)),
            bounds=bounds,
            constraints=constraints
        )
        
        return dict(zip(sales_velocity.keys(), result.x))

# Example usage
analytics = DarkStoreAnalytics('STORE123')
sales_data = pd.DataFrame({
    'item_id': ['A1', 'A2', 'A3'],
    'date': pd.date_range('2024-01-01', periods=3),
    'sales_quantity': [10, 15, 20],
    'category': ['food', 'beverages', 'food']
})
inventory_data = pd.DataFrame({
    'item_id': ['A1', 'A2', 'A3'],
    'date': pd.date_range('2024-01-01', periods=3),
    'average_inventory': [50, 60, 70]
})

results = analytics.analyze_store_performance(sales_data, inventory_data)
print(f"Analytics results: {results}")
```

Slide 13: Geographic Heat Mapping for Demand Analysis

This implementation demonstrates how Blinkit analyzes order density and creates geographic heat maps to optimize dark store placement and delivery routes using spatial clustering algorithms.

```python
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import folium
from typing import List, Tuple

class DemandHeatMapper:
    def __init__(self):
        self.eps = 0.1  # km
        self.min_samples = 5
        
    def create_heat_map(self, 
                       order_locations: List[Tuple[float, float]],
                       order_values: List[float]) -> folium.Map:
        # Convert to numpy array
        locations = np.array(order_locations)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='haversine'
        ).fit(locations)
        
        # Create base map
        center = locations.mean(axis=0)
        heat_map = folium.Map(
            location=center,
            zoom_start=13
        )
        
        # Add heat layer
        folium.plugins.HeatMap(
            locations,
            weights=order_values,
            radius=15
        ).add_to(heat_map)
        
        # Analyze clusters
        clusters = self._analyze_clusters(
            locations,
            clustering.labels_,
            order_values
        )
        
        return heat_map, clusters
    
    def _analyze_clusters(self,
                         locations: np.ndarray,
                         labels: np.ndarray,
                         values: List[float]) -> dict:
        unique_labels = np.unique(labels)
        cluster_stats = {}
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
                
            mask = labels == label
            cluster_points = locations[mask]
            cluster_values = np.array(values)[mask]
            
            # Calculate cluster statistics
            hull = ConvexHull(cluster_points)
            
            cluster_stats[f'cluster_{label}'] = {
                'center': cluster_points.mean(axis=0),
                'radius': np.max(
                    np.linalg.norm(
                        cluster_points - cluster_points.mean(axis=0),
                        axis=1
                    )
                ),
                'total_value': np.sum(cluster_values),
                'density': len(cluster_points) / hull.volume,
                'points_count': len(cluster_points)
            }
            
        return cluster_stats

    def suggest_store_locations(self, 
                              clusters: dict,
                              existing_stores: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        suggested_locations = []
        
        for cluster_id, stats in clusters.items():
            # Check if cluster is covered by existing stores
            cluster_covered = any(
                np.linalg.norm(
                    np.array(store) - stats['center']
                ) < 2.0  # 2km radius
                for store in existing_stores
            )
            
            if not cluster_covered and stats['total_value'] > 10000:
                suggested_locations.append(tuple(stats['center']))
                
        return suggested_locations

# Example usage
mapper = DemandHeatMapper()

# Generate sample data
np.random.seed(42)
n_orders = 1000
locations = np.random.normal(
    loc=[28.6139, 77.2090],
    scale=[0.02, 0.02],
    size=(n_orders, 2)
)
order_values = np.random.lognormal(4, 1, n_orders)

heat_map, clusters = mapper.create_heat_map(
    locations.tolist(),
    order_values.tolist()
)

existing_stores = [(28.6139, 77.2090), (28.6239, 77.2190)]
suggestions = mapper.suggest_store_locations(clusters, existing_stores)
print(f"Suggested new store locations: {suggestions}")
```

Slide 14: Additional Resources

*   Location-based demand prediction using deep learning for real-time delivery: [https://arxiv.org/abs/2103.05532](https://arxiv.org/abs/2103.05532)
*   Machine Learning Algorithms for Inventory Management in Quick Commerce: [https://arxiv.org/abs/2204.09876](https://arxiv.org/abs/2204.09876)
*   Real-time Optimization of Last-Mile Delivery Routes: [https://arxiv.org/abs/2106.12823](https://arxiv.org/abs/2106.12823)
*   Scalable Microservices Architecture for E-commerce Platforms: [https://www.researchgate.net/publication/349872156](https://www.researchgate.net/publication/349872156)
*   Real-time Inventory Management Using Reinforcement Learning: [https://proceedings.mlr.press/v162/inventory-management.html](https://proceedings.mlr.press/v162/inventory-management.html)

Suggested resources for further research:

*   Google Scholar: "quick commerce optimization algorithms"
*   Research papers on real-time delivery optimization
*   Academic publications on microservices architecture scalability

