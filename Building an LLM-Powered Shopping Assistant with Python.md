## Building an LLM-Powered Shopping Assistant with Python
Slide 1: Introduction to LLM Shopping Copilots

Building an LLM Shopping Copilot combines natural language processing with e-commerce to enhance the shopping experience. This AI-powered assistant helps users find products, compare options, and make informed decisions. Let's explore how to create such a system using Python and popular NLP libraries.

```python
import openai
from transformers import pipeline

class ShoppingCopilot:
    def __init__(self):
        self.llm = openai.ChatCompletion()
        self.sentiment_analyzer = pipeline("sentiment-analysis")

    def process_query(self, user_input):
        # Main logic for processing user queries
        pass
```

Slide 2: Setting Up the Environment

To begin, we need to set up our Python environment with the necessary libraries. We'll use OpenAI's GPT model for natural language understanding and the Transformers library for additional NLP tasks.

```python
# Install required libraries
!pip install openai transformers torch

# Import essential modules
import openai
from transformers import pipeline
import os

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

print("Environment set up successfully!")
```

Slide 3: Defining the Copilot Class

Our ShoppingCopilot class will encapsulate the main functionality of our assistant. It will handle user queries, process them, and generate appropriate responses.

```python
class ShoppingCopilot:
    def __init__(self):
        self.conversation_history = []

    def process_query(self, user_input):
        self.conversation_history.append({"role": "user", "content": user_input})
        response = self.generate_response()
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def generate_response(self):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.conversation_history
        )
        return response.choices[0].message['content']

copilot = ShoppingCopilot()
print(copilot.process_query("What are some popular laptop brands?"))
```

Slide 4: Implementing Product Search

A crucial feature of our shopping copilot is the ability to search for products. We'll implement a simple product search function using a mock database.

```python
import random

class ProductDatabase:
    def __init__(self):
        self.products = [
            {"name": "Laptop X", "brand": "TechCo", "price": 999},
            {"name": "Smartphone Y", "brand": "Gadget Inc", "price": 699},
            {"name": "Tablet Z", "brand": "ElectronicsPro", "price": 399},
        ]

    def search(self, query):
        return [p for p in self.products if query.lower() in p['name'].lower()]

class ShoppingCopilot:
    def __init__(self):
        self.db = ProductDatabase()

    def search_products(self, query):
        results = self.db.search(query)
        if results:
            return f"Found {len(results)} products: " + ", ".join([p['name'] for p in results])
        else:
            return "No products found matching your query."

copilot = ShoppingCopilot()
print(copilot.search_products("laptop"))
```

Slide 5: Implementing Price Comparison

Price comparison is an essential feature for any shopping assistant. Let's add a method to compare prices of similar products.

```python
class ShoppingCopilot:
    def __init__(self):
        self.db = ProductDatabase()

    def compare_prices(self, product_type):
        products = self.db.search(product_type)
        if not products:
            return f"No {product_type} found in our database."
        
        sorted_products = sorted(products, key=lambda x: x['price'])
        comparison = f"Price comparison for {product_type}:\n"
        for product in sorted_products:
            comparison += f"{product['name']} ({product['brand']}): ${product['price']}\n"
        return comparison

copilot = ShoppingCopilot()
print(copilot.compare_prices("laptop"))
```

Slide 6: Implementing Sentiment Analysis

To provide better recommendations, we can analyze user sentiment towards products. We'll use the Transformers library for this purpose.

```python
from transformers import pipeline

class ShoppingCopilot:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")

    def analyze_sentiment(self, text):
        result = self.sentiment_analyzer(text)[0]
        sentiment = result['label']
        score = result['score']
        return f"Sentiment: {sentiment} (confidence: {score:.2f})"

copilot = ShoppingCopilot()
review = "I love this laptop! It's fast and has great battery life."
print(copilot.analyze_sentiment(review))
```

Slide 7: Implementing Product Recommendations

Based on user preferences and previous interactions, our copilot can provide personalized product recommendations.

```python
import random

class ShoppingCopilot:
    def __init__(self):
        self.db = ProductDatabase()
        self.user_preferences = {}

    def update_preferences(self, user_id, product_type):
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = set()
        self.user_preferences[user_id].add(product_type)

    def recommend_products(self, user_id):
        if user_id not in self.user_preferences:
            return "We don't have enough information to make recommendations yet."
        
        preferred_types = list(self.user_preferences[user_id])
        recommended_type = random.choice(preferred_types)
        products = self.db.search(recommended_type)
        
        if products:
            recommended_product = random.choice(products)
            return f"Based on your preferences, we recommend: {recommended_product['name']} (${recommended_product['price']})"
        else:
            return f"We couldn't find any products matching your preferences for {recommended_type}."

copilot = ShoppingCopilot()
copilot.update_preferences("user123", "laptop")
print(copilot.recommend_products("user123"))
```

Slide 8: Implementing Natural Language Understanding

To make our copilot more user-friendly, we'll implement natural language understanding to interpret user queries more accurately.

```python
import openai

class ShoppingCopilot:
    def __init__(self):
        self.conversation_history = []

    def process_query(self, user_input):
        self.conversation_history.append({"role": "user", "content": user_input})
        response = self.generate_response()
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def generate_response(self):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.conversation_history + [
                {"role": "system", "content": "You are a helpful shopping assistant. Provide concise and relevant information about products."}
            ]
        )
        return response.choices[0].message['content']

copilot = ShoppingCopilot()
print(copilot.process_query("What features should I look for in a gaming laptop?"))
```

Slide 9: Implementing Multi-turn Conversations

To make our copilot more interactive, let's implement multi-turn conversations that can maintain context across multiple user queries.

```python
class ShoppingCopilot:
    def __init__(self):
        self.conversation_history = []

    def chat(self, user_input):
        self.conversation_history.append({"role": "user", "content": user_input})
        response = self.generate_response()
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def generate_response(self):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.conversation_history + [
                {"role": "system", "content": "You are a helpful shopping assistant. Maintain context across multiple queries."}
            ]
        )
        return response.choices[0].message['content']

copilot = ShoppingCopilot()
print(copilot.chat("I'm looking for a new smartphone."))
print(copilot.chat("What are some good options under $500?"))
print(copilot.chat("Which one has the best camera?"))
```

Slide 10: Implementing Product Comparison

Let's add a feature to compare multiple products side by side, helping users make informed decisions.

```python
class ShoppingCopilot:
    def __init__(self):
        self.db = ProductDatabase()

    def compare_products(self, product_names):
        products = [p for p in self.db.products if p['name'] in product_names]
        if not products:
            return "No matching products found."
        
        comparison = "Product Comparison:\n"
        headers = ["Name", "Brand", "Price"]
        comparison += " | ".join(headers) + "\n"
        comparison += "-" * (sum(len(h) for h in headers) + len(headers) - 1) + "\n"
        
        for product in products:
            comparison += f"{product['name']} | {product['brand']} | ${product['price']}\n"
        
        return comparison

copilot = ShoppingCopilot()
print(copilot.compare_products(["Laptop X", "Smartphone Y"]))
```

Slide 11: Implementing User Profiles

To provide personalized recommendations, let's implement user profiles that store preferences and shopping history.

```python
class UserProfile:
    def __init__(self, user_id):
        self.user_id = user_id
        self.preferences = set()
        self.purchase_history = []

    def add_preference(self, preference):
        self.preferences.add(preference)

    def add_purchase(self, product):
        self.purchase_history.append(product)

class ShoppingCopilot:
    def __init__(self):
        self.users = {}

    def get_or_create_user(self, user_id):
        if user_id not in self.users:
            self.users[user_id] = UserProfile(user_id)
        return self.users[user_id]

    def update_user_preference(self, user_id, preference):
        user = self.get_or_create_user(user_id)
        user.add_preference(preference)
        return f"Added {preference} to your preferences."

    def get_user_profile(self, user_id):
        user = self.get_or_create_user(user_id)
        return f"User {user_id}:\nPreferences: {', '.join(user.preferences)}\nPurchases: {len(user.purchase_history)}"

copilot = ShoppingCopilot()
print(copilot.update_user_preference("user123", "electronics"))
print(copilot.get_user_profile("user123"))
```

Slide 12: Implementing Intent Recognition

To better understand user queries, let's implement intent recognition to classify user intentions and respond accordingly.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class IntentClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()
        self.intents = ['search', 'compare', 'recommend', 'buy']
        self.train_data = [
            "find a product", "search for items", "look for a gadget",
            "compare prices", "which one is better", "difference between products",
            "recommend a good laptop", "suggest a smartphone", "what should I buy",
            "purchase this item", "add to cart", "buy now"
        ]
        self.train_labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        self.train()

    def train(self):
        X = self.vectorizer.fit_transform(self.train_data)
        self.classifier.fit(X, self.train_labels)

    def predict_intent(self, query):
        X = self.vectorizer.transform([query])
        intent_id = self.classifier.predict(X)[0]
        return self.intents[intent_id]

class ShoppingCopilot:
    def __init__(self):
        self.intent_classifier = IntentClassifier()

    def process_query(self, query):
        intent = self.intent_classifier.predict_intent(query)
        return f"Detected intent: {intent}\nQuery: {query}"

copilot = ShoppingCopilot()
print(copilot.process_query("Can you find me a good laptop?"))
print(copilot.process_query("Which smartphone is better, A or B?"))
```

Slide 13: Implementing a Simple Web Interface

To make our shopping copilot more accessible, let's create a basic web interface using Flask.

```python
from flask import Flask, render_template, request, jsonify
from shopping_copilot import ShoppingCopilot

app = Flask(__name__)
copilot = ShoppingCopilot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    response = copilot.process_query(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

# In a separate file named 'index.html' in the 'templates' folder:
'''
<!DOCTYPE html>
<html>
<head>
    <title>Shopping Copilot</title>
</head>
<body>
    <h1>Shopping Copilot</h1>
    <div id="chat-container"></div>
    <input type="text" id="user-input" placeholder="Ask me anything...">
    <button onclick="sendMessage()">Send</button>

    <script>
        function sendMessage() {
            var userInput = document.getElementById('user-input').value;
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({message: userInput}),
            })
            .then(response => response.json())
            .then(data => {
                var chatContainer = document.getElementById('chat-container');
                chatContainer.innerHTML += '<p><strong>You:</strong> ' + userInput + '</p>';
                chatContainer.innerHTML += '<p><strong>Copilot:</strong> ' + data.response + '</p>';
                document.getElementById('user-input').value = '';
            });
        }
    </script>
</body>
</html>
'''
```

Slide 14: Real-life Example: Restaurant Menu Assistant

Let's adapt our shopping copilot to help users navigate a restaurant menu, demonstrating its versatility in different domains.

```python
class MenuItem:
    def __init__(self, name, category, price, description):
        self.name = name
        self.category = category
        self.price = price
        self.description = description

class RestaurantMenuCopilot:
    def __init__(self):
        self.menu = [
            MenuItem("Margherita Pizza", "Main Course", 12.99, "Classic tomato and mozzarella pizza"),
            MenuItem("Caesar Salad", "Appetizer", 8.99, "Romaine lettuce with Caesar dressing and croutons"),
            MenuItem("Tiramisu", "Dessert", 6.99, "Italian coffee-flavored dessert"),
        ]

    def search_menu(self, query):
        results = [item for item in self.menu if query.lower() in item.name.lower() or query.lower() in item.category.lower()]
        if results:
            return "\n".join([f"{item.name} (${item.price}): {item.description}" for item in results])
        else:
            return "No menu items found matching your query."

    def recommend_dish(self, preference):
        matching_items = [item for item in self.menu if preference.lower() in item.description.lower()]
        if matching_items:
            recommended = random.choice(matching_items)
            return f"Based on your preference for {preference}, we recommend: {recommended.name} (${recommended.price})"
        else:
            return f"We couldn't find a dish matching your preference for {preference}."

copilot = RestaurantMenuCopilot()
print(copilot.search_menu("pizza"))
print(copilot.recommend_dish("Italian"))
```

Slide 15: Real-life Example: Grocery List Assistant

Now let's create a grocery list assistant to help users manage their shopping lists and suggest recipes based on available ingredients.

```python
import random

class GroceryItem:
    def __init__(self, name, category, unit_price):
        self.name = name
        self.category = category
        self.unit_price = unit_price

class GroceryListAssistant:
    def __init__(self):
        self.grocery_items = [
            GroceryItem("Apple", "Fruit", 0.50),
            GroceryItem("Bread", "Bakery", 2.99),
            GroceryItem("Milk", "Dairy", 3.49),
            GroceryItem("Chicken", "Meat", 5.99),
        ]
        self.shopping_list = []

    def add_to_list(self, item_name, quantity):
        item = next((i for i in self.grocery_items if i.name.lower() == item_name.lower()), None)
        if item:
            self.shopping_list.append((item, quantity))
            return f"Added {quantity} {item.name}(s) to your list."
        else:
            return f"Sorry, {item_name} is not available in our store."

    def view_list(self):
        if not self.shopping_list:
            return "Your shopping list is empty."
        total_cost = sum(item.unit_price * quantity for item, quantity in self.shopping_list)
        list_view = "Your shopping list:\n"
        for item, quantity in self.shopping_list:
            list_view += f"{quantity}x {item.name} (${item.unit_price * quantity:.2f})\n"
        list_view += f"\nTotal estimated cost: ${total_cost:.2f}"
        return list_view

    def suggest_recipe(self):
        available_ingredients = [item.name.lower() for item, _ in self.shopping_list]
        recipes = {
            "Fruit Salad": ["apple"],
            "Chicken Sandwich": ["bread", "chicken"],
            "Cereal with Milk": ["milk"]
        }
        possible_recipes = [name for name, ingredients in recipes.items() 
                            if all(ing in available_ingredients for ing in ingredients)]
        if possible_recipes:
            return f"Based on your shopping list, you could make: {random.choice(possible_recipes)}"
        else:
            return "We couldn't suggest a recipe based on your current shopping list."

assistant = GroceryListAssistant()
print(assistant.add_to_list("Apple", 3))
print(assistant.add_to_list("Bread", 1))
print(assistant.view_list())
print(assistant.suggest_recipe())
```

Slide 16: Additional Resources

For those interested in diving deeper into building LLM-based assistants and natural language processing, here are some valuable resources:

1. "Attention Is All You Need" by Vaswani et al. (2017) - The original transformer paper: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018): [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Language Models are Few-Shot Learners" by Brown et al. (2020) - Introduces GPT-3: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
4. "HuggingFace Transformers Library Documentation": [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
5. "OpenAI API Documentation": [https://platform.openai.com/docs/](https://platform.openai.com/docs/)

These resources provide a solid foundation for understanding the underlying technologies and implementing more advanced features in your LLM shopping copilot.

