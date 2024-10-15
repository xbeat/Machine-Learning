## Recency Frequency Magnitude Analysis with Python
Slide 1: Introduction to Recency, Frequency, Magnitude (RFM) Analysis

RFM analysis is a customer segmentation technique used in marketing and customer relationship management. It evaluates customers based on three key metrics: Recency (how recently they made a purchase), Frequency (how often they make purchases), and Magnitude (how much they spend). This powerful tool helps businesses understand customer behavior and tailor their marketing strategies accordingly.

```python
import pandas as pd
import numpy as np

# Sample customer data
data = {
    'customer_id': [1, 2, 3, 4, 5],
    'last_purchase_date': ['2023-08-15', '2023-07-01', '2023-08-30', '2023-06-10', '2023-08-25'],
    'purchase_frequency': [5, 2, 10, 1, 7],
    'total_spend': [500, 100, 1000, 50, 750]
}

df = pd.DataFrame(data)
df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])

print(df.head())
```

Slide 2: Recency: The 'R' in RFM

Recency measures how recently a customer has made a purchase. It's based on the idea that customers who have purchased more recently are more likely to respond to marketing efforts. Recency is typically calculated as the number of days since the customer's last purchase.

```python
import datetime

# Calculate recency
current_date = datetime.date(2023, 9, 1)
df['recency'] = (current_date - df['last_purchase_date'].dt.date).dt.days

print(df[['customer_id', 'last_purchase_date', 'recency']])
```

Slide 3: Frequency: The 'F' in RFM

Frequency refers to how often a customer makes purchases. It's based on the assumption that customers who buy more frequently are more engaged and likely to continue purchasing. Frequency is usually calculated as the total number of purchases made by a customer over a specific period.

```python
# Frequency is already in our dataset as 'purchase_frequency'
# Let's calculate the average frequency
avg_frequency = df['purchase_frequency'].mean()

print(f"Average purchase frequency: {avg_frequency:.2f}")
print(df[['customer_id', 'purchase_frequency']])
```

Slide 4: Magnitude: The 'M' in RFM

Magnitude, also known as Monetary value, represents how much a customer spends. It's based on the principle that customers who spend more are more valuable to the business. Magnitude is typically calculated as the total amount spent by a customer over a specific period.

```python
# Magnitude is already in our dataset as 'total_spend'
# Let's calculate the average spend
avg_spend = df['total_spend'].mean()

print(f"Average total spend: ${avg_spend:.2f}")
print(df[['customer_id', 'total_spend']])
```

Slide 5: Scoring RFM Components

To perform RFM analysis, we need to score each component. A common approach is to divide customers into quintiles (five equal groups) for each metric, assigning scores from 1 to 5, where 5 represents the highest value.

```python
def assign_rfm_scores(df, column, ascending=False):
    return pd.qcut(df[column], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

df['R_score'] = assign_rfm_scores(df, 'recency', ascending=True)
df['F_score'] = assign_rfm_scores(df, 'purchase_frequency')
df['M_score'] = assign_rfm_scores(df, 'total_spend')

print(df[['customer_id', 'R_score', 'F_score', 'M_score']])
```

Slide 6: Combining RFM Scores

After scoring each component, we combine them into a single RFM score. This is typically done by concatenating the individual scores or by calculating a weighted average.

```python
# Concatenate scores
df['RFM_score'] = df['R_score'].astype(str) + df['F_score'].astype(str) + df['M_score'].astype(str)

# Calculate weighted average (example weights: R=100, F=10, M=1)
df['RFM_weighted'] = (100 * df['R_score'] + 10 * df['F_score'] + df['M_score'])

print(df[['customer_id', 'RFM_score', 'RFM_weighted']])
```

Slide 7: Customer Segmentation

Based on RFM scores, we can segment customers into different groups. Common segments include "Best Customers," "Loyal Customers," "Big Spenders," "Lost Customers," and more. These segments help in tailoring marketing strategies.

```python
def segment_customers(row):
    if row['RFM_weighted'] > 450:
        return 'Best Customers'
    elif row['RFM_weighted'] > 400:
        return 'Loyal Customers'
    elif row['RFM_weighted'] > 300:
        return 'Big Spenders'
    else:
        return 'Lost Customers'

df['Customer_Segment'] = df.apply(segment_customers, axis=1)

print(df[['customer_id', 'RFM_weighted', 'Customer_Segment']])
```

Slide 8: Visualizing RFM Analysis

Visualization is crucial for understanding RFM analysis results. Let's create a scatter plot to visualize the relationship between Recency and Frequency, with point size representing Magnitude.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['recency'], df['purchase_frequency'], s=df['total_spend']/10, alpha=0.5)
plt.xlabel('Recency (days)')
plt.ylabel('Frequency (number of purchases)')
plt.title('RFM Analysis Visualization')
for i, txt in enumerate(df['customer_id']):
    plt.annotate(txt, (df['recency'].iloc[i], df['purchase_frequency'].iloc[i]))
plt.show()
```

Slide 9: Real-Life Example: E-commerce Website

An e-commerce website selling electronics uses RFM analysis to segment its customer base. They identify a group of "Best Customers" who have made purchases within the last 30 days, have bought more than 5 times in the past year, and have spent over $1000 in total. These customers receive personalized product recommendations and early access to new product launches.

```python
# Sample data for e-commerce example
ecommerce_data = {
    'customer_id': [101, 102, 103, 104, 105],
    'last_purchase_date': ['2023-08-25', '2023-07-15', '2023-08-30', '2023-06-01', '2023-08-28'],
    'purchase_count': [7, 3, 10, 2, 8],
    'total_spend': [1200, 500, 2000, 300, 1500]
}

edf = pd.DataFrame(ecommerce_data)
edf['last_purchase_date'] = pd.to_datetime(edf['last_purchase_date'])
edf['recency'] = (pd.Timestamp('2023-09-01') - edf['last_purchase_date']).dt.days

best_customers = edf[(edf['recency'] <= 30) & (edf['purchase_count'] > 5) & (edf['total_spend'] > 1000)]
print("Best Customers:")
print(best_customers)
```

Slide 10: Real-Life Example: Subscription-Based Service

A subscription-based streaming service uses RFM analysis to identify at-risk customers. They focus on users with high recency (haven't used the service in over 60 days), low frequency (used less than 5 times in the past month), and low magnitude (only subscribed to the basic plan). These customers are targeted with re-engagement campaigns and special offers.

```python
# Sample data for subscription service example
subscription_data = {
    'user_id': [201, 202, 203, 204, 205],
    'last_login_date': ['2023-06-15', '2023-08-30', '2023-07-01', '2023-08-25', '2023-05-01'],
    'logins_last_month': [2, 10, 4, 8, 1],
    'subscription_tier': ['basic', 'premium', 'basic', 'premium', 'basic']
}

sdf = pd.DataFrame(subscription_data)
sdf['last_login_date'] = pd.to_datetime(sdf['last_login_date'])
sdf['recency'] = (pd.Timestamp('2023-09-01') - sdf['last_login_date']).dt.days

at_risk_customers = sdf[(sdf['recency'] > 60) & (sdf['logins_last_month'] < 5) & (sdf['subscription_tier'] == 'basic')]
print("At-Risk Customers:")
print(at_risk_customers)
```

Slide 11: Implementing RFM Analysis: Step-by-Step

Let's walk through the process of implementing RFM analysis on a dataset:

1. Prepare the data
2. Calculate RFM metrics
3. Score RFM components
4. Combine scores
5. Segment customers

```python
import pandas as pd
import numpy as np
from datetime import datetime

# Step 1: Prepare the data
data = {
    'customer_id': range(1, 11),
    'last_purchase_date': ['2023-08-15', '2023-07-01', '2023-08-30', '2023-06-10', '2023-08-25',
                           '2023-08-20', '2023-07-05', '2023-08-28', '2023-06-15', '2023-08-22'],
    'purchase_frequency': [5, 2, 10, 1, 7, 6, 3, 8, 2, 5],
    'total_spend': [500, 100, 1000, 50, 750, 600, 200, 900, 150, 550]
}

df = pd.DataFrame(data)
df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])

# Step 2: Calculate RFM metrics
current_date = datetime(2023, 9, 1)
df['recency'] = (current_date - df['last_purchase_date']).dt.days
df['frequency'] = df['purchase_frequency']
df['magnitude'] = df['total_spend']

# Step 3: Score RFM components
for metric in ['recency', 'frequency', 'magnitude']:
    col_name = f'{metric[0].upper()}_score'
    df[col_name] = pd.qcut(df[metric], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    if metric == 'recency':
        df[col_name] = 6 - df[col_name]  # Invert recency score

# Step 4: Combine scores
df['RFM_score'] = df['R_score'].astype(str) + df['F_score'].astype(str) + df['M_score'].astype(str)

# Step 5: Segment customers
def segment_customers(row):
    score = int(row['RFM_score'])
    if score >= 444:
        return 'Best Customers'
    elif score >= 434:
        return 'Loyal Customers'
    elif score >= 333:
        return 'Potential Churners'
    else:
        return 'Lost Customers'

df['Customer_Segment'] = df.apply(segment_customers, axis=1)

print(df[['customer_id', 'RFM_score', 'Customer_Segment']])
```

Slide 12: Challenges and Considerations in RFM Analysis

While RFM analysis is powerful, it's important to consider its limitations and challenges:

1. Seasonality: Some businesses may have seasonal patterns that affect RFM metrics.
2. Product lifecycle: The nature of products (e.g., durable goods vs. consumables) can impact purchase frequency.
3. Customer lifecycle: New customers might be undervalued in RFM analysis.
4. Data quality: Accurate and comprehensive data is crucial for meaningful analysis.

```python
import random

# Simulate seasonal data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
seasonal_data = []

for date in dates:
    # Simulate higher sales in summer and winter
    season_factor = 1.5 if date.month in [6, 7, 8, 12, 1, 2] else 1.0
    
    # Generate random sales data
    sales = int(random.normalvariate(100 * season_factor, 20))
    seasonal_data.append({'date': date, 'sales': sales})

seasonal_df = pd.DataFrame(seasonal_data)

# Plot seasonal data
plt.figure(figsize=(12, 6))
plt.plot(seasonal_df['date'], seasonal_df['sales'])
plt.title('Simulated Seasonal Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

print("Mean sales in summer/winter:", seasonal_df[seasonal_df['date'].dt.month.isin([6, 7, 8, 12, 1, 2])]['sales'].mean())
print("Mean sales in spring/fall:", seasonal_df[~seasonal_df['date'].dt.month.isin([6, 7, 8, 12, 1, 2])]['sales'].mean())
```

Slide 13: Advanced RFM Techniques

As businesses evolve, so do RFM techniques. Some advanced approaches include:

1. Time-weighted RFM: Giving more weight to recent purchases.
2. Predictive RFM: Using machine learning to predict future customer behavior.
3. Multi-dimensional RFM: Incorporating additional variables beyond the basic three.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Generate sample data
np.random.seed(42)
n_samples = 1000
recency = np.random.randint(1, 365, n_samples)
frequency = np.random.randint(1, 20, n_samples)
magnitude = np.random.randint(10, 1000, n_samples)

# Create target variable (e.g., likelihood of next purchase)
next_purchase_likelihood = 0.7 * (1 / recency) + 0.2 * frequency + 0.1 * (magnitude / 1000)
next_purchase_likelihood += np.random.normal(0, 0.1, n_samples)  # Add some noise

# Prepare data for model
X = np.column_stack((recency, frequency, magnitude))
y = next_purchase_likelihood

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Print feature importances
feature_importance = rf_model.feature_importances_
print("Feature Importances:")
print("Recency:", feature_importance[0])
print("Frequency:", feature_importance[1])
print("Magnitude:", feature_importance[2])
```

Slide 14: Conclusion and Best Practices

RFM analysis is a powerful tool for customer segmentation and targeted marketing. To make the most of it:

1. Regularly update your RFM analysis to capture changing customer behavior.
2. Combine RFM with other data sources for a more comprehensive view.
3. Use RFM insights to personalize marketing campaigns and improve customer experience.
4. Continuously test and refine your RFM model and segmentation strategy.

Remember, while RFM provides valuable insights


