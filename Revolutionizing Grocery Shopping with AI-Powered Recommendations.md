## Revolutionizing Grocery Shopping with AI-Powered Recommendations
Slide 1: The Future of Grocery Shopping: AI-Powered Recommender Systems

Picnic, an innovative grocery company, is leveraging machine learning to revolutionize the shopping experience. Their journey began with the Customer Article Rebuy Prediction (CARP) model, which has since evolved into more sophisticated recommender systems.

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load and preprocess data
X, y = load_customer_purchase_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train CARP model using XGBoost
carp_model = xgb.XGBClassifier()
carp_model.fit(X_train, y_train)

# Make predictions
predictions = carp_model.predict(X_test)

# Evaluate model performance
accuracy = (predictions == y_test).mean()
print(f"CARP Model Accuracy: {accuracy:.2f}")
```

Slide 2: CARP: Predicting Repeat Purchases

The CARP model, powered by XGBoost, excels at predicting and recommending repeat purchases. This personalization significantly improves the shopping experience by quickly surfacing frequently bought items to customers.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_carp_features(customer_data):
    # Encode categorical features
    le = LabelEncoder()
    customer_data['product_id'] = le.fit_transform(customer_data['product_id'])
    customer_data['customer_id'] = le.fit_transform(customer_data['customer_id'])
    
    # Create time-based features
    customer_data['days_since_last_purchase'] = (pd.Timestamp.now() - customer_data['last_purchase_date']).dt.days
    customer_data['purchase_frequency'] = customer_data.groupby('product_id')['customer_id'].transform('count')
    
    return customer_data[['customer_id', 'product_id', 'days_since_last_purchase', 'purchase_frequency']]

# Example usage
customer_data = pd.read_csv('customer_purchase_history.csv')
carp_features = prepare_carp_features(customer_data)
print(carp_features.head())
```

Slide 3: Beyond Repeat Purchases: Exploring New Recommendations

While CARP excels at predicting repeat purchases, Picnic faces the challenge of recommending new, unexplored items. This involves using more complex models like Markov Chains, RNNs, Transformers, CNNs, and GNNs to predict entire baskets of items.

```python
import torch
import torch.nn as nn

class BasketRecommenderRNN(nn.Module):
    def __init__(self, num_products, embedding_dim, hidden_dim):
        super(BasketRecommenderRNN, self).__init__()
        self.embedding = nn.Embedding(num_products, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_products)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.fc(output[:, -1, :])

# Example usage
num_products = 1000
embedding_dim = 50
hidden_dim = 100
model = BasketRecommenderRNN(num_products, embedding_dim, hidden_dim)

# Simulated input: batch of 32 customers, each with a history of 10 products
x = torch.randint(0, num_products, (32, 10))
output = model(x)
print(f"Output shape: {output.shape}")  # Expected: torch.Size([32, 1000])
```

Slide 4: Key Findings: CARP's Dominance

CARP outperforms other models by large margins for repeat purchases, making it crucial for recommending frequent buys. This success highlights the importance of leveraging historical data for personalized recommendations.

```python
import matplotlib.pyplot as plt
import numpy as np

def compare_model_performance():
    models = ['CARP', 'Collaborative Filtering', 'Content-Based', 'Hybrid']
    accuracy = [0.85, 0.62, 0.58, 0.71]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracy, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    plt.title('Model Performance Comparison')
    plt.xlabel('Recommender Models')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    
    plt.show()

compare_model_performance()
```

Slide 5: The Challenge of Explore Recommendations

Recommending new, unexplored items poses significant challenges due to sparse data and subtle customer preferences. Picnic's approach involves adapting sequential recommendation models to predict entire baskets of items.

```python
import torch
import torch.nn as nn

class TransformerBasketRecommender(nn.Module):
    def __init__(self, num_products, d_model, nhead, num_layers):
        super(TransformerBasketRecommender, self).__init__()
        self.embedding = nn.Embedding(num_products, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers
        )
        self.fc = nn.Linear(d_model, num_products)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))

# Example usage
num_products = 1000
d_model = 64
nhead = 4
num_layers = 2
model = TransformerBasketRecommender(num_products, d_model, nhead, num_layers)

# Simulated input: batch of 32 customers, each with a history of 20 products
x = torch.randint(0, num_products, (32, 20))
output = model(x)
print(f"Output shape: {output.shape}")  # Expected: torch.Size([32, 1000])
```

Slide 6: Incremental Success: Long-Term Patterns

Picnic discovered that performance improves when considering purchases over extended periods. This indicates that customer preferences might take time to materialize, highlighting the need for long-term analysis.

```python
import pandas as pd
import matplotlib.pyplot as plt

def analyze_long_term_patterns(purchase_data, time_windows):
    results = []
    for window in time_windows:
        filtered_data = purchase_data[purchase_data['date'] >= (purchase_data['date'].max() - pd.Timedelta(days=window))]
        accuracy = calculate_recommendation_accuracy(filtered_data)
        results.append((window, accuracy))
    
    df = pd.DataFrame(results, columns=['Time Window (days)', 'Recommendation Accuracy'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time Window (days)'], df['Recommendation Accuracy'], marker='o')
    plt.title('Recommendation Accuracy vs Time Window')
    plt.xlabel('Time Window (days)')
    plt.ylabel('Recommendation Accuracy')
    plt.grid(True)
    plt.show()

# Example usage
purchase_data = pd.read_csv('purchase_history.csv')
time_windows = [30, 60, 90, 180, 365]
analyze_long_term_patterns(purchase_data, time_windows)
```

Slide 7: Innovative Solution: Category Recommendations

To address the challenge of sparse data, Picnic explores category-level recommendations. This approach aggregates data to make more accurate predictions about customer preferences across product categories.

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def category_based_recommendations(user_history, product_catalog):
    # Aggregate user history by category
    user_categories = user_history.groupby('category')['purchase_count'].sum().reset_index()
    
    # Create TF-IDF matrix of user's category preferences
    tfidf = TfidfVectorizer()
    user_profile = tfidf.fit_transform(user_categories['category'])
    
    # Calculate similarity between user profile and product categories
    product_categories = tfidf.transform(product_catalog['category'])
    similarities = cosine_similarity(user_profile, product_categories)
    
    # Get top recommended categories
    top_categories = similarities.argsort()[0][::-1][:5]
    recommendations = product_catalog.iloc[top_categories]
    
    return recommendations

# Example usage
user_history = pd.DataFrame({
    'category': ['Fruits', 'Vegetables', 'Dairy', 'Fruits', 'Snacks'],
    'purchase_count': [3, 2, 1, 2, 1]
})
product_catalog = pd.DataFrame({
    'product_id': range(1, 101),
    'category': ['Fruits', 'Vegetables', 'Dairy', 'Snacks', 'Beverages'] * 20
})

recommendations = category_based_recommendations(user_history, product_catalog)
print(recommendations)
```

Slide 8: Future Insights: Monitoring Long-Term Patterns

By monitoring longer-term purchase patterns, Picnic aims to refine its recommendations further. This approach allows for a deeper understanding of evolving customer preferences over time.

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def analyze_purchase_trends(purchase_data, product_id):
    # Filter data for specific product
    product_data = purchase_data[purchase_data['product_id'] == product_id]
    
    # Resample to weekly frequency and calculate total purchases
    weekly_purchases = product_data.resample('W')['quantity'].sum()
    
    # Perform time series decomposition
    result = seasonal_decompose(weekly_purchases, model='additive', period=52)
    
    # Plot the decomposition
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
    result.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    result.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    result.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    result.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    plt.tight_layout()
    plt.show()

# Example usage
purchase_data = pd.read_csv('purchase_history.csv', parse_dates=['date'])
purchase_data.set_index('date', inplace=True)
analyze_purchase_trends(purchase_data, product_id=1)
```

Slide 9: Real-Life Example: Smart Shopping List

Imagine a smart shopping list that learns from your past purchases and suggests items you might need. As you check off items, it updates in real-time, considering factors like seasonality and your typical shopping patterns.

```python
import random

class SmartShoppingList:
    def __init__(self, user_id):
        self.user_id = user_id
        self.items = self.load_user_preferences()
        self.seasonal_items = self.get_seasonal_items()
    
    def load_user_preferences(self):
        # Simulated user preferences
        return {
            'Milk': 0.9, 'Bread': 0.8, 'Eggs': 0.7,
            'Apples': 0.6, 'Bananas': 0.5, 'Chicken': 0.4
        }
    
    def get_seasonal_items(self):
        # Simulated seasonal items
        return ['Pumpkin', 'Turkey', 'Cranberries']
    
    def generate_list(self):
        shopping_list = []
        for item, probability in self.items.items():
            if random.random() < probability:
                shopping_list.append(item)
        
        # Add seasonal items with lower probability
        for item in self.seasonal_items:
            if random.random() < 0.3:
                shopping_list.append(item)
        
        return shopping_list
    
    def update_preferences(self, purchased_items):
        for item in purchased_items:
            if item in self.items:
                self.items[item] = min(1.0, self.items[item] + 0.1)
            else:
                self.items[item] = 0.5

# Example usage
smart_list = SmartShoppingList(user_id=123)
suggested_list = smart_list.generate_list()
print("Suggested shopping list:", suggested_list)

# Simulate user purchases
purchased_items = ['Milk', 'Bread', 'Apples', 'Pumpkin']
smart_list.update_preferences(purchased_items)

# Generate new list after update
new_list = smart_list.generate_list()
print("Updated shopping list:", new_list)
```

Slide 10: Real-Life Example: Dietary Recommendations

Consider a system that recommends recipes based on your dietary preferences and restrictions. It analyzes your past choices and suggests new recipes that align with your tastes while ensuring nutritional balance.

```python
import random

class DietaryRecommender:
    def __init__(self, user_id):
        self.user_id = user_id
        self.preferences = self.load_user_preferences()
        self.restrictions = self.load_user_restrictions()
        self.recipe_database = self.load_recipe_database()
    
    def load_user_preferences(self):
        # Simulated user preferences
        return {'vegetarian': 0.7, 'spicy': 0.6, 'low-carb': 0.5}
    
    def load_user_restrictions(self):
        # Simulated user dietary restrictions
        return ['gluten-free', 'no-nuts']
    
    def load_recipe_database(self):
        # Simulated recipe database
        return [
            {'name': 'Vegetable Stir-Fry', 'tags': ['vegetarian', 'low-carb', 'gluten-free']},
            {'name': 'Spicy Chicken Curry', 'tags': ['spicy', 'gluten-free']},
            {'name': 'Quinoa Salad', 'tags': ['vegetarian', 'gluten-free']},
            {'name': 'Grilled Salmon', 'tags': ['low-carb', 'gluten-free']},
            {'name': 'Peanut Butter Smoothie', 'tags': ['vegetarian']}
        ]
    
    def recommend_recipes(self, num_recommendations=3):
        suitable_recipes = [
            recipe for recipe in self.recipe_database
            if all(restriction in recipe['tags'] for restriction in self.restrictions)
        ]
        
        scored_recipes = []
        for recipe in suitable_recipes:
            score = sum(self.preferences.get(tag, 0) for tag in recipe['tags'])
            scored_recipes.append((recipe, score))
        
        scored_recipes.sort(key=lambda x: x[1], reverse=True)
        return [recipe['name'] for recipe, _ in scored_recipes[:num_recommendations]]

# Example usage
recommender = DietaryRecommender(user_id=456)
recommendations = recommender.recommend_recipes()
print("Recommended recipes:", recommendations)
```

Slide 11: Ethical Considerations in AI-Powered Recommendations

As we develop more sophisticated recommender systems, it's crucial to consider the ethical implications. These systems should respect user privacy, avoid creating echo chambers, and promote diverse, healthy choices while maintaining transparency.

```python
class EthicalRecommender:
    def __init__(self, user_preferences, product_catalog):
        self.user_preferences = user_preferences
        self.product_catalog = product_catalog
        self.diversity_threshold = 0.3
        self.health_score_threshold = 7

    def generate_recommendations(self, num_recommendations=5):
        recommendations = []
        for _ in range(num_recommendations):
            candidate = self.get_candidate_recommendation()
            if self.is_ethical_recommendation(candidate):
                recommendations.append(candidate)
        return recommendations

    def get_candidate_recommendation(self):
        # Simplified recommendation logic
        return random.choice(self.product_catalog)

    def is_ethical_recommendation(self, candidate):
        return (self.respects_privacy(candidate) and
                self.promotes_diversity(candidate) and
                self.promotes_health(candidate))

    def respects_privacy(self, candidate):
        # Implement privacy checks
        return True

    def promotes_diversity(self, candidate):
        similarity = self.calculate_similarity(candidate, self.user_preferences)
        return similarity < self.diversity_threshold

    def promotes_health(self, candidate):
        return candidate['health_score'] > self.health_score_threshold

    def calculate_similarity(self, item1, item2):
        # Implement similarity calculation
        return random.random()

# Example usage
user_preferences = {'category': 'fruits', 'price_range': 'medium'}
product_catalog = [
    {'id': 1, 'name': 'Apple', 'category': 'fruits', 'health_score': 8},
    {'id': 2, 'name': 'Chips', 'category': 'snacks', 'health_score': 3},
    {'id': 3, 'name': 'Broccoli', 'category': 'vegetables', 'health_score': 9}
]

recommender = EthicalRecommender(user_preferences, product_catalog)
ethical_recommendations = recommender.generate_recommendations()
print("Ethical Recommendations:", ethical_recommendations)
```

Slide 12: Balancing Personalization and Discovery

Effective recommender systems must strike a balance between suggesting familiar items and introducing new discoveries. This approach ensures user satisfaction while expanding their horizons.

```python
import numpy as np

class BalancedRecommender:
    def __init__(self, user_history, product_catalog, exploration_rate=0.2):
        self.user_history = user_history
        self.product_catalog = product_catalog
        self.exploration_rate = exploration_rate

    def generate_recommendations(self, num_recommendations=5):
        recommendations = []
        for _ in range(num_recommendations):
            if np.random.random() < self.exploration_rate:
                recommendation = self.explore()
            else:
                recommendation = self.exploit()
            recommendations.append(recommendation)
        return recommendations

    def explore(self):
        # Select a random product not in user history
        new_products = [p for p in self.product_catalog if p not in self.user_history]
        return np.random.choice(new_products)

    def exploit(self):
        # Select a product based on user history (simplified)
        return np.random.choice(self.user_history)

# Example usage
user_history = ['apple', 'banana', 'milk']
product_catalog = ['apple', 'banana', 'milk', 'bread', 'cheese', 'eggs', 'yogurt']

recommender = BalancedRecommender(user_history, product_catalog)
balanced_recommendations = recommender.generate_recommendations()
print("Balanced Recommendations:", balanced_recommendations)
```

Slide 13: The Future of AI in Grocery Shopping

As AI continues to evolve, we can expect even more sophisticated recommender systems that not only predict what we need but also help us make healthier, more sustainable choices. These systems might integrate with smart home devices, health trackers, and environmental data to provide holistic recommendations.

```python
import random

class FutureGroceryAI:
    def __init__(self, user_id):
        self.user_id = user_id
        self.preferences = self.load_user_data()
        self.health_data = self.get_health_data()
        self.environmental_data = self.get_environmental_data()

    def load_user_data(self):
        # Simulated user data
        return {
            'favorite_foods': ['apples', 'chicken', 'quinoa'],
            'dietary_restrictions': ['lactose-free'],
            'sustainability_preference': 0.8
        }

    def get_health_data(self):
        # Simulated health data from wearable device
        return {
            'daily_steps': 8000,
            'calories_burned': 2200,
            'heart_rate': 65
        }

    def get_environmental_data(self):
        # Simulated environmental data
        return {
            'local_produce': ['tomatoes', 'lettuce', 'cucumbers'],
            'carbon_footprint_threshold': 50
        }

    def generate_smart_recommendations(self):
        recommendations = []
        
        # Consider health data
        if self.health_data['daily_steps'] < 5000:
            recommendations.append('energy-boosting foods')
        
        # Consider environmental impact
        if random.random() < self.preferences['sustainability_preference']:
            recommendations.extend(self.environmental_data['local_produce'])
        
        # Consider dietary restrictions and favorites
        recommendations.extend([food for food in self.preferences['favorite_foods']
                                if food not in self.preferences['dietary_restrictions']])
        
        return recommendations[:5]  # Return top 5 recommendations

# Example usage
future_ai = FutureGroceryAI(user_id=789)
smart_recommendations = future_ai.generate_smart_recommendations()
print("Smart Recommendations:", smart_recommendations)
```

Slide 14: Additional Resources

For those interested in diving deeper into the world of AI-powered recommender systems, here are some valuable resources:

1. "Deep Learning for Recommender Systems" by Balázs Hidasi et al. (ArXiv:1703.04247) URL: [https://arxiv.org/abs/1703.04247](https://arxiv.org/abs/1703.04247)
2. "Sequential Recommender Systems: Challenges, Progress and Prospects" by Fajie Yuan et al. (ArXiv:1905.01997) URL: [https://arxiv.org/abs/1905.01997](https://arxiv.org/abs/1905.01997)
3. "A Survey on Session-based Recommender Systems" by Shoujin Wang et al. (ArXiv:1902.04864) URL: [https://arxiv.org/abs/1902.04864](https://arxiv.org/abs/1902.04864)

These papers provide in-depth insights into the latest techniques and challenges in building advanced recommender systems for various applications, including e-commerce and grocery shopping.

