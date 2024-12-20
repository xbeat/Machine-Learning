## Understanding RAG vs. Agentic RAG with Python
Slide 1: Understanding RAG: Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is a powerful technique that combines the strengths of large language models with external knowledge retrieval. It enhances the model's ability to generate accurate and contextually relevant responses by retrieving information from a knowledge base.

```python
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# Initialize RAG components
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

# Generate text using RAG
input_text = "What is the capital of France?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
generated = model.generate(input_ids)
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
```

Slide 2: The RAG Process

RAG operates in two main steps: retrieval and generation. First, it retrieves relevant information from a knowledge base using the input query. Then, it combines this retrieved information with the input to generate a response using a language model.

```python
import faiss
import numpy as np

# Simulating a knowledge base
knowledge_base = [
    "Paris is the capital of France.",
    "The Eiffel Tower is located in Paris.",
    "France is a country in Western Europe."
]

# Create a simple vector index
dimension = 64
index = faiss.IndexFlatL2(dimension)

# Add vectors to the index (simplified)
for i, text in enumerate(knowledge_base):
    vector = np.random.random(dimension).astype('float32')
    index.add(np.array([vector]))

# Retrieval step
query_vector = np.random.random(dimension).astype('float32')
k = 1  # Number of nearest neighbors to retrieve
distances, indices = index.search(np.array([query_vector]), k)

print(f"Retrieved: {knowledge_base[indices[0][0]]}")
```

Slide 3: Agentic RAG: Enhancing RAG with Agency

Agentic RAG takes the concept of RAG a step further by incorporating agency. This means the system can make decisions about when and how to retrieve information, and can even update its knowledge base dynamically.

```python
import random

class AgenticRAG:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.confidence_threshold = 0.7

    def retrieve(self, query):
        # Simplified retrieval logic
        relevant_info = random.choice(self.knowledge_base)
        confidence = random.random()
        return relevant_info, confidence

    def generate(self, query):
        retrieved_info, confidence = self.retrieve(query)
        if confidence > self.confidence_threshold:
            return f"Based on retrieved information: {retrieved_info}"
        else:
            return "I don't have enough confidence in the retrieved information to answer."

    def update_knowledge(self, new_info):
        self.knowledge_base.append(new_info)

# Usage
arag = AgenticRAG(["Earth is the third planet from the Sun.", "The Moon orbits the Earth."])
print(arag.generate("Tell me about Earth"))
arag.update_knowledge("Mars is called the Red Planet.")
print(arag.generate("What do you know about Mars?"))
```

Slide 4: Key Differences: RAG vs. Agentic RAG

The main difference between RAG and Agentic RAG lies in the level of autonomy and decision-making capabilities. While RAG follows a fixed retrieval-generation process, Agentic RAG can adapt its behavior based on the context and its confidence in the retrieved information.

```python
import random

def simulate_rag_vs_arag(query, knowledge_base):
    # Simulating RAG
    rag_result = random.choice(knowledge_base)
    
    # Simulating Agentic RAG
    arag = AgenticRAG(knowledge_base)
    arag_result = arag.generate(query)
    
    return rag_result, arag_result

# Example usage
knowledge_base = [
    "The Great Wall of China is visible from space.",
    "The Great Wall of China is not easily visible from space with the naked eye.",
    "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states."
]

query = "Is the Great Wall of China visible from space?"
rag_answer, arag_answer = simulate_rag_vs_arag(query, knowledge_base)

print(f"RAG: {rag_answer}")
print(f"Agentic RAG: {arag_answer}")
```

Slide 5: Advantages of Agentic RAG

Agentic RAG offers several advantages over traditional RAG, including improved accuracy, adaptability, and the ability to handle ambiguous or conflicting information more effectively. It can also learn and update its knowledge base over time.

```python
class ImprovedAgenticRAG(AgenticRAG):
    def __init__(self, knowledge_base):
        super().__init__(knowledge_base)
        self.learning_rate = 0.1

    def generate(self, query):
        retrieved_info, confidence = self.retrieve(query)
        if confidence > self.confidence_threshold:
            response = f"Based on retrieved information: {retrieved_info}"
            self.update_confidence(retrieved_info, 1)  # Reinforce correct information
        else:
            response = "I'm not confident about this. Let me learn more and try again."
            self.learn_new_information(query)
        return response

    def update_confidence(self, info, factor):
        # Simplified confidence update
        index = self.knowledge_base.index(info)
        self.confidences[index] += self.learning_rate * factor

    def learn_new_information(self, query):
        # Simplified learning process
        new_info = f"New information about: {query}"
        self.update_knowledge(new_info)
        print(f"Learned: {new_info}")

# Usage
improved_arag = ImprovedAgenticRAG(["The speed of light is approximately 299,792,458 meters per second."])
print(improved_arag.generate("What is the speed of light?"))
print(improved_arag.generate("What is the speed of sound?"))
```

Slide 6: Implementing RAG in Python

Let's implement a basic version of RAG using Python. We'll use a simplified retrieval mechanism and a mock language model for generation.

```python
import random

class SimpleRAG:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def retrieve(self, query):
        # Simple keyword-based retrieval
        relevant_docs = [doc for doc in self.knowledge_base if any(word in doc.lower() for word in query.lower().split())]
        return random.choice(relevant_docs) if relevant_docs else ""

    def generate(self, query):
        retrieved_info = self.retrieve(query)
        # Simple generation by combining query and retrieved info
        return f"Query: {query}\nRelevant info: {retrieved_info}\nGenerated response: Based on the retrieved information, {retrieved_info}"

# Example usage
knowledge_base = [
    "Python is a high-level programming language.",
    "Python was created by Guido van Rossum.",
    "Python is known for its simplicity and readability."
]

rag = SimpleRAG(knowledge_base)
query = "Tell me about Python"
response = rag.generate(query)
print(response)
```

Slide 7: Implementing Agentic RAG in Python

Now, let's implement a basic version of Agentic RAG, which includes decision-making capabilities and the ability to update its knowledge base.

```python
import random

class AgenticRAG:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.confidence_threshold = 0.7

    def retrieve(self, query):
        relevant_docs = [doc for doc in self.knowledge_base if any(word in doc.lower() for word in query.lower().split())]
        if relevant_docs:
            return random.choice(relevant_docs), random.random()
        return "", 0

    def generate(self, query):
        retrieved_info, confidence = self.retrieve(query)
        if confidence > self.confidence_threshold:
            return f"Confident response: Based on the retrieved information, {retrieved_info}"
        elif retrieved_info:
            return f"Low confidence response: I'm not entirely sure, but {retrieved_info}"
        else:
            new_info = self.learn_new_information(query)
            return f"I've learned something new: {new_info}"

    def learn_new_information(self, query):
        new_info = f"New information about {query} (to be verified)"
        self.knowledge_base.append(new_info)
        return new_info

# Example usage
arag = AgenticRAG([
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning is a type of machine learning based on artificial neural networks."
])

queries = ["What is machine learning?", "Explain quantum computing", "Define deep learning"]
for query in queries:
    print(f"\nQuery: {query}")
    print(arag.generate(query))

print("\nUpdated knowledge base:")
print(arag.knowledge_base)
```

Slide 8: Real-life Example: Content Recommendation System

Let's explore how RAG and Agentic RAG can be applied to a content recommendation system for a streaming platform.

```python
import random

class ContentRecommender:
    def __init__(self, content_database):
        self.content_database = content_database
        self.user_preferences = {}

    def rag_recommend(self, user_id, query):
        # Simple RAG-based recommendation
        relevant_content = [c for c in self.content_database if query.lower() in c['description'].lower()]
        return random.choice(relevant_content) if relevant_content else None

    def agentic_rag_recommend(self, user_id, query):
        # Agentic RAG-based recommendation
        relevant_content = [c for c in self.content_database if query.lower() in c['description'].lower()]
        if user_id in self.user_preferences:
            # Consider user preferences
            preferred_genre = self.user_preferences[user_id]
            relevant_content = [c for c in relevant_content if c['genre'] == preferred_genre] or relevant_content
        
        if relevant_content:
            recommendation = random.choice(relevant_content)
            self.update_user_preferences(user_id, recommendation['genre'])
            return recommendation
        else:
            return self.explore_new_content(query)

    def update_user_preferences(self, user_id, genre):
        self.user_preferences[user_id] = genre

    def explore_new_content(self, query):
        # Simulate exploring new content
        return {'title': f"New content related to {query}", 'description': f"Exploring new areas based on {query}", 'genre': 'Exploration'}

# Example usage
content_database = [
    {'title': "The Space Odyssey", 'description': "A journey through space and time", 'genre': 'Sci-Fi'},
    {'title': "Mystery Manor", 'description': "A thrilling detective story", 'genre': 'Mystery'},
    {'title': "Laugh Out Loud", 'description': "A hilarious comedy special", 'genre': 'Comedy'}
]

recommender = ContentRecommender(content_database)

print("RAG Recommendation:")
print(recommender.rag_recommend(user_id=1, query="space"))

print("\nAgentic RAG Recommendation (First time):")
print(recommender.agentic_rag_recommend(user_id=1, query="space"))

print("\nAgentic RAG Recommendation (Second time, with learned preference):")
print(recommender.agentic_rag_recommend(user_id=1, query="thriller"))
```

Slide 9: Real-life Example: Intelligent Customer Support System

Let's implement a simple intelligent customer support system using RAG and Agentic RAG principles.

```python
import random

class CustomerSupportSystem:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.confidence_threshold = 0.7
        self.escalation_count = 0

    def rag_response(self, query):
        relevant_info = [info for info in self.knowledge_base if any(word in info.lower() for word in query.lower().split())]
        if relevant_info:
            return random.choice(relevant_info)
        return "I'm sorry, I don't have information about that."

    def agentic_rag_response(self, query):
        relevant_info = [info for info in self.knowledge_base if any(word in info.lower() for word in query.lower().split())]
        confidence = random.random()
        
        if confidence > self.confidence_threshold and relevant_info:
            return random.choice(relevant_info)
        elif relevant_info:
            self.escalation_count += 1
            return f"I'm not entirely sure, but here's what I found: {random.choice(relevant_info)}. Would you like me to escalate this to a human agent?"
        else:
            self.escalation_count += 1
            return "I don't have information about that. Let me escalate this to a human agent."

    def get_escalation_stats(self):
        return f"Total escalations: {self.escalation_count}"

# Example usage
knowledge_base = [
    "Our return policy allows returns within 30 days of purchase.",
    "You can track your order using the tracking number in your confirmation email.",
    "We offer free shipping on orders over $50."
]

support_system = CustomerSupportSystem(knowledge_base)

queries = ["What's your return policy?", "How can I track my order?", "Do you have any discounts?"]

print("RAG Responses:")
for query in queries:
    print(f"Q: {query}")
    print(f"A: {support_system.rag_response(query)}\n")

print("Agentic RAG Responses:")
for query in queries:
    print(f"Q: {query}")
    print(f"A: {support_system.agentic_rag_response(query)}\n")

print(support_system.get_escalation_stats())
```

Slide 10: Challenges in Implementing Agentic RAG

Implementing Agentic RAG comes with its own set of challenges, including maintaining accuracy, managing computational resources, and ensuring ethical decision-making.

```python
import random
import time

class AgenticRAGChallenges:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.accuracy_threshold = 0.8
        self.max_runtime = 2  # seconds

    def retrieve_and_generate(self, query):
        start_time = time.time()
        accuracy = random.random()
        
        while accuracy < self.accuracy_threshold and (time.time() - start_time) < self.max_runtime:
            accuracy += random.random() * 0.1
            time.sleep(0.1)
        
        runtime = time.time() - start_time
        
        if accuracy >= self.accuracy_threshold:
            result = random.choice(self.knowledge_base)
            return f"Result: {result}", accuracy, runtime
        else:
            return "Unable to generate a sufficiently accurate response", accuracy, runtime

challenges = AgenticRAGChallenges([
    "Agentic RAG can adapt to new information dynamically.",
    "Implementing Agentic RAG requires careful balance of accuracy and efficiency.",
    "Ethical considerations are crucial in Agentic RAG systems."
])

queries = ["Tell me about Agentic RAG", "What are the challenges of Agentic RAG?", "How does Agentic RAG work?"]

for query in queries:
    result, accuracy, runtime = challenges.retrieve_and_generate(query)
    print(f"Query: {query}")
    print(f"Response: {result}")
    print(f"Accuracy: {accuracy:.2f}, Runtime: {runtime:.2f}s\n")
```

Slide 11: Ethical Considerations in Agentic RAG

Agentic RAG systems raise important ethical considerations, particularly in terms of decision-making autonomy, data privacy, and potential biases in information retrieval and generation.

```python
class EthicalAgenticRAG:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.bias_threshold = 0.3
        self.privacy_sensitive_words = ["personal", "private", "confidential"]

    def check_bias(self, text):
        # Simplified bias check (in practice, use more sophisticated methods)
        bias_score = sum(1 for word in text.split() if len(word) > 7) / len(text.split())
        return bias_score > self.bias_threshold

    def check_privacy(self, query):
        return any(word in query.lower() for word in self.privacy_sensitive_words)

    def generate_response(self, query):
        if self.check_privacy(query):
            return "I cannot process queries involving personal or confidential information."
        
        relevant_info = [info for info in self.knowledge_base if query.lower() in info.lower()]
        if not relevant_info:
            return "I don't have relevant information to answer this query."
        
        response = relevant_info[0]
        if self.check_bias(response):
            return "The generated response may contain biases. Please interpret with caution."
        
        return response

ethical_rag = EthicalAgenticRAG([
    "Climate change is a global challenge requiring immediate action.",
    "Renewable energy sources are becoming increasingly cost-effective.",
    "Sustainable practices can help mitigate environmental impact."
])

queries = [
    "Tell me about climate change",
    "What is your personal opinion on politics?",
    "How can we address environmental challenges?"
]

for query in queries:
    print(f"Query: {query}")
    print(f"Response: {ethical_rag.generate_response(query)}\n")
```

Slide 12: Future Directions for Agentic RAG

The future of Agentic RAG holds exciting possibilities, including improved decision-making capabilities, integration with multimodal data, and enhanced adaptability to user needs.

```python
import random

class FutureAgenticRAG:
    def __init__(self):
        self.knowledge_types = ["text", "image", "audio", "video"]
        self.adaptation_level = 0

    def retrieve_multimodal(self, query):
        knowledge_type = random.choice(self.knowledge_types)
        return f"Retrieved {knowledge_type} data relevant to: {query}"

    def adaptive_generation(self, query, user_profile):
        self.adaptation_level += 0.1
        adapted_response = f"Adapting to user needs (level {self.adaptation_level:.1f}): {query}"
        return adapted_response

    def decision_making(self, query, context):
        decision = random.choice(["retrieve", "generate", "ask for clarification"])
        return f"Decision for '{query}' in context '{context}': {decision}"

future_rag = FutureAgenticRAG()

queries = ["Explain quantum computing", "Show me cat pictures", "What's the weather like?"]
user_profile = "tech-savvy, visual learner"
context = "educational setting"

for query in queries:
    print(f"Query: {query}")
    print(f"Multimodal Retrieval: {future_rag.retrieve_multimodal(query)}")
    print(f"Adaptive Generation: {future_rag.adaptive_generation(query, user_profile)}")
    print(f"Decision Making: {future_rag.decision_making(query, context)}\n")
```

Slide 13: Comparing RAG and Agentic RAG: A Summary

This slide provides a concise comparison between traditional RAG and Agentic RAG, highlighting their key differences and use cases.

```python
def compare_rag_and_agentic_rag(query, knowledge_base):
    class SimpleRAG:
        def generate(self, q):
            return random.choice([info for info in knowledge_base if q.lower() in info.lower()])

    class AgenticRAG:
        def generate(self, q):
            relevant_info = [info for info in knowledge_base if q.lower() in info.lower()]
            if not relevant_info:
                return "I need to learn more about this topic."
            return f"Based on analysis: {random.choice(relevant_info)}"

    rag = SimpleRAG()
    arag = AgenticRAG()

    print(f"Query: {query}")
    print(f"RAG Response: {rag.generate(query)}")
    print(f"Agentic RAG Response: {arag.generate(query)}")

knowledge_base = [
    "RAG retrieves information and generates responses.",
    "Agentic RAG can make decisions and adapt its behavior.",
    "RAG is simpler but less flexible than Agentic RAG.",
    "Agentic RAG is more complex but can handle uncertain situations better."
]

queries = ["What is RAG?", "How does Agentic RAG work?", "Compare RAG and Agentic RAG"]

for q in queries:
    compare_rag_and_agentic_rag(q, knowledge_base)
    print()
```

Slide 14: Additional Resources

For those interested in delving deeper into RAG and Agentic RAG, here are some valuable resources:

1. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020) ArXiv link: [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
2. "Self-Supervised Learning for Neural Information Retrieval" (Chang et al., 2020) ArXiv link: [https://arxiv.org/abs/2006.05542](https://arxiv.org/abs/2006.05542)
3. "Language Models are Few-Shot Learners" (Brown et al., 2020) ArXiv link: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

These papers provide in-depth insights into the foundations and advanced concepts of retrieval-augmented generation and related techniques.

