## Building and Deploying an LLM Agent in Python A Step-by-Step Guide
Slide 1: Introduction to LLM Agents

Building an LLM (Large Language Model) agent in Python involves creating a system that can understand and generate human-like text, and perform tasks based on natural language instructions. This process combines natural language processing, machine learning, and software engineering principles to create an intelligent, interactive system.

```python
import transformers
import torch

# Initialize a pre-trained LLM
model_name = "gpt2-medium"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

# Example of generating text
input_text = "Hello, I'm an LLM agent."
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=50)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

Slide 2: Setting Up the Environment

Before we start building our LLM agent, we need to set up our Python environment. This involves installing necessary libraries and ensuring we have the required dependencies.

```python
# Create a virtual environment
python -m venv llm_agent_env

# Activate the virtual environment
source llm_agent_env/bin/activate  # On Unix or MacOS
llm_agent_env\Scripts\activate.bat  # On Windows

# Install required libraries
pip install transformers torch numpy

# Verify installations
import transformers
import torch
import numpy as np

print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
```

Slide 3: Choosing and Loading a Pre-trained Model

Selecting an appropriate pre-trained model is crucial for our LLM agent. We'll use the Hugging Face Transformers library to load a pre-trained model.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2-medium"  # You can choose other models like "facebook/opt-350m"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"Model {model_name} loaded successfully")
print(f"Vocabulary size: {len(tokenizer)}")
print(f"Model parameters: {model.num_parameters()}")
```

Slide 4: Creating the LLM Agent Class

We'll create a Python class to encapsulate our LLM agent's functionality. This class will handle text generation and manage the agent's state.

```python
import torch

class LLMAgent:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.conversation_history = []

    def generate_response(self, user_input, max_length=100):
        # Add user input to conversation history
        self.conversation_history.append(f"User: {user_input}")
        
        # Prepare input for the model
        full_input = " ".join(self.conversation_history)
        input_ids = self.tokenizer.encode(full_input, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=max_length, num_return_sequences=1)
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract the newly generated part
        new_response = response[len(full_input):].strip()
        
        # Add agent's response to conversation history
        self.conversation_history.append(f"Agent: {new_response}")
        
        return new_response

# Usage example
agent = LLMAgent("gpt2-medium")
response = agent.generate_response("What's the weather like today?")
print(response)
```

Slide 5: Implementing Context Management

To make our LLM agent more coherent and context-aware, we need to implement effective context management. This involves maintaining a conversation history and using it to inform the agent's responses.

```python
class LLMAgent:
    def __init__(self, model_name, max_history=5):
        # ... (previous initialization code) ...
        self.max_history = max_history

    def generate_response(self, user_input, max_length=100):
        # Add user input to conversation history
        self.conversation_history.append(f"User: {user_input}")
        
        # Limit conversation history to max_history entries
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
        
        # Prepare input for the model
        full_input = " ".join(self.conversation_history)
        input_ids = self.tokenizer.encode(full_input, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=len(input_ids[0]) + max_length, num_return_sequences=1)
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract the newly generated part
        new_response = response[len(full_input):].strip()
        
        # Add agent's response to conversation history
        self.conversation_history.append(f"Agent: {new_response}")
        
        return new_response

# Usage example
agent = LLMAgent("gpt2-medium", max_history=3)
print(agent.generate_response("Hi, how are you?"))
print(agent.generate_response("What's your favorite color?"))
print(agent.generate_response("Can you remember what I asked first?"))
```

Slide 6: Adding Task-specific Capabilities

To make our LLM agent more versatile, we can add task-specific capabilities. This involves defining specific functions for different tasks and integrating them with the agent's language understanding.

```python
import datetime
import requests

class LLMAgent:
    # ... (previous code) ...

    def get_current_time(self):
        return datetime.datetime.now().strftime("%H:%M:%S")

    def get_weather(self, city):
        # Note: You would need to sign up for an API key and use a real weather API
        api_key = "your_api_key_here"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
        response = requests.get(url)
        data = response.json()
        return f"The weather in {city} is {data['weather'][0]['description']}"

    def process_task(self, user_input):
        if "time" in user_input.lower():
            return f"The current time is {self.get_current_time()}"
        elif "weather" in user_input.lower():
            city = user_input.split("in")[-1].strip()
            return self.get_weather(city)
        else:
            return self.generate_response(user_input)

# Usage example
agent = LLMAgent("gpt2-medium")
print(agent.process_task("What time is it?"))
print(agent.process_task("What's the weather in London?"))
print(agent.process_task("Tell me a joke"))
```

Slide 7: Implementing Error Handling and Robustness

To make our LLM agent more reliable, we need to implement proper error handling and add robustness to our code. This involves catching exceptions, validating inputs, and gracefully handling unexpected situations.

```python
class LLMAgent:
    # ... (previous code) ...

    def generate_response(self, user_input, max_length=100):
        try:
            # Add user input to conversation history
            self.conversation_history.append(f"User: {user_input}")
            
            # Prepare input for the model
            full_input = " ".join(self.conversation_history)
            input_ids = self.tokenizer.encode(full_input, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(input_ids, max_length=len(input_ids[0]) + max_length, num_return_sequences=1)
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract the newly generated part
            new_response = response[len(full_input):].strip()
            
            # Add agent's response to conversation history
            self.conversation_history.append(f"Agent: {new_response}")
            
            return new_response
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return "I'm sorry, I encountered an error while processing your request."

    def process_task(self, user_input):
        if not isinstance(user_input, str):
            return "Invalid input. Please provide a string."
        
        try:
            if "time" in user_input.lower():
                return f"The current time is {self.get_current_time()}"
            elif "weather" in user_input.lower():
                city = user_input.split("in")[-1].strip()
                if not city:
                    return "Please specify a city for the weather information."
                return self.get_weather(city)
            else:
                return self.generate_response(user_input)
        except Exception as e:
            print(f"An error occurred while processing the task: {str(e)}")
            return "I'm sorry, I encountered an error while processing your request."

# Usage example
agent = LLMAgent("gpt2-medium")
print(agent.process_task("What time is it?"))
print(agent.process_task("What's the weather in"))  # Invalid input
print(agent.process_task(123))  # Invalid input type
```

Slide 8: Enhancing Response Quality

To improve the quality of our LLM agent's responses, we can implement techniques like temperature control, top-k sampling, and response filtering.

```python
import re

class LLMAgent:
    # ... (previous code) ...

    def generate_response(self, user_input, max_length=100, temperature=0.7, top_k=50):
        try:
            # ... (previous code for preparing input) ...
            
            # Generate response with temperature and top-k sampling
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=len(input_ids[0]) + max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            new_response = response[len(full_input):].strip()
            
            # Filter out inappropriate content (simple example)
            inappropriate_words = ["offensive", "explicit", "violent"]
            if any(word in new_response.lower() for word in inappropriate_words):
                new_response = "I apologize, but I don't feel comfortable responding to that."
            
            # Ensure the response is not too short
            if len(new_response.split()) < 3:
                new_response = self.generate_response(user_input, max_length, temperature, top_k)
            
            # Add agent's response to conversation history
            self.conversation_history.append(f"Agent: {new_response}")
            
            return new_response
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return "I'm sorry, I encountered an error while processing your request."

# Usage example
agent = LLMAgent("gpt2-medium")
print(agent.generate_response("Tell me about the importance of kindness.", temperature=0.8, top_k=40))
```

Slide 9: Implementing Memory and Long-term Learning

To make our LLM agent more intelligent over time, we can implement a form of memory and long-term learning. This involves storing important information and using it in future interactions.

```python
import json

class LLMAgent:
    def __init__(self, model_name, memory_file="agent_memory.json"):
        # ... (previous initialization code) ...
        self.memory_file = memory_file
        self.long_term_memory = self.load_memory()

    def load_memory(self):
        try:
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.long_term_memory, f)

    def remember(self, key, value):
        self.long_term_memory[key] = value
        self.save_memory()

    def recall(self, key):
        return self.long_term_memory.get(key, "I don't have any information about that.")

    def generate_response(self, user_input, max_length=100):
        # Check if the input is asking about remembered information
        if user_input.lower().startswith("what do you remember about"):
            topic = user_input[27:].strip()
            return self.recall(topic)

        # ... (previous generation code) ...

        # After generating the response, check if it contains important information to remember
        if "remember that" in user_input.lower():
            info_to_remember = user_input.split("remember that")[-1].strip()
            self.remember(info_to_remember[:20], info_to_remember)

        return new_response

# Usage example
agent = LLMAgent("gpt2-medium")
agent.generate_response("Remember that the capital of France is Paris.")
print(agent.generate_response("What do you remember about the capital of France?"))
```

Slide 10: Adding Multi-modal Capabilities

To enhance our LLM agent's abilities, we can add multi-modal capabilities, allowing it to process and generate not just text, but also images or other types of data.

```python
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

class LLMAgent:
    # ... (previous code) ...

    def process_image(self, image_url):
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        
        # Perform some basic image analysis (this is a placeholder for more advanced image processing)
        width, height = img.size
        format = img.format
        mode = img.mode
        
        return f"This is a {width}x{height} {format} image in {mode} mode."

    def generate_image(self, prompt):
        # This is a placeholder for actual image generation
        # In a real implementation, you would use a model like DALL-E or Stable Diffusion
        plt.figure(figsize=(5,5))
        plt.text(0.5, 0.5, prompt, ha='center', va='center', wrap=True)
        plt.axis('off')
        plt.show()
        return "Image generated based on the prompt."

    def process_task(self, user_input):
        if user_input.lower().startswith("analyze image:"):
            image_url = user_input.split(":", 1)[1].strip()
            return self.process_image(image_url)
        elif user_input.lower().startswith("generate image:"):
            prompt = user_input.split(":", 1)[1].strip()
            return self.generate_image(prompt)
        else:
            return self.generate_response(user_input)

# Usage example
agent = LLMAgent("gpt2-medium")
print(agent.process_task("Analyze image: https://example.com/image.jpg"))
print(agent.process_task("Generate image: A cat sitting on a rainbow"))
```

Slide 11: Implementing Conversational Flow Control

To make our LLM agent more engaging and interactive, we can implement conversational flow control. This involves managing turn-taking, handling interruptions, and maintaining coherent dialogue.

```python
import time

class LLMAgent:
    def __init__(self, model_name):
        # ... (previous initialization code) ...
        self.conversation_state = "waiting"
        self.interruption_buffer = []

    def start_conversation(self):
        self.conversation_state = "active"
        return "Hello! How can I assist you today?"

    def end_conversation(self):
        self.conversation_state = "waiting"
        return "Thank you for chatting. Goodbye!"

    def handle_interruption(self, user_input):
        self.interruption_buffer.append(user_input)
        return "I see you have something to add. Let me finish my thought, and then we'll address that."

    def generate_response(self, user_input):
        if self.conversation_state == "waiting":
            return self.start_conversation()

        if user_input.lower() == "goodbye":
            return self.end_conversation()

        if self.conversation_state == "speaking":
            return self.handle_interruption(user_input)

        self.conversation_state = "thinking"
        response = super().generate_response(user_input)
        self.conversation_state = "speaking"

        time.sleep(len(response) * 0.05)  # Simulate typing time
        self.conversation_state = "waiting"

        if self.interruption_buffer:
            interruption = self.interruption_buffer.pop(0)
            return f"{response}\nNow, addressing your earlier point: {self.generate_response(interruption)}"

        return response

# Usage example
agent = LLMAgent("gpt2-medium")
print(agent.generate_response("Hello!"))
print(agent.generate_response("What's the weather like?"))
print(agent.generate_response("Wait, before you answer..."))
print(agent.generate_response("Actually, can you tell me the time?"))
print(agent.generate_response("Goodbye"))
```

Slide 12: Implementing Ethical Guidelines

Ensuring our LLM agent behaves ethically is crucial. We can implement a system of ethical guidelines to filter out inappropriate content and promote responsible AI behavior.

```python
class EthicalFilter:
    def __init__(self):
        self.inappropriate_topics = ["violence", "hate speech", "explicit content"]
        self.sensitive_topics = ["politics", "religion", "personal information"]

    def filter_content(self, text):
        lower_text = text.lower()
        for topic in self.inappropriate_topics:
            if topic in lower_text:
                return "I apologize, but I can't discuss that topic."
        for topic in self.sensitive_topics:
            if topic in lower_text:
                return f"The topic of {topic} can be sensitive. I'll try to provide a balanced and respectful perspective."
        return text

class LLMAgent:
    def __init__(self, model_name):
        # ... (previous initialization code) ...
        self.ethical_filter = EthicalFilter()

    def generate_response(self, user_input):
        filtered_input = self.ethical_filter.filter_content(user_input)
        if filtered_input != user_input:
            return filtered_input
        
        response = super().generate_response(user_input)
        return self.ethical_filter.filter_content(response)

# Usage example
agent = LLMAgent("gpt2-medium")
print(agent.generate_response("Tell me about history"))
print(agent.generate_response("What do you think about politics?"))
```

Slide 13: Implementing Continuous Learning

While true continuous learning is challenging for LLM agents, we can implement a system that allows the agent to accumulate new information over time and use it in future interactions.

```python
import json
from datetime import datetime

class LLMAgent:
    def __init__(self, model_name, knowledge_file="agent_knowledge.json"):
        # ... (previous initialization code) ...
        self.knowledge_file = knowledge_file
        self.learned_knowledge = self.load_knowledge()

    def load_knowledge(self):
        try:
            with open(self.knowledge_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_knowledge(self):
        with open(self.knowledge_file, 'w') as f:
            json.dump(self.learned_knowledge, f)

    def learn(self, topic, information):
        self.learned_knowledge[topic] = {
            "info": information,
            "timestamp": datetime.now().isoformat()
        }
        self.save_knowledge()

    def generate_response(self, user_input):
        if user_input.lower().startswith("learn:"):
            parts = user_input[6:].split(":", 1)
            if len(parts) == 2:
                topic, information = parts
                self.learn(topic.strip(), information.strip())
                return f"I've learned new information about {topic}."

        for topic, data in self.learned_knowledge.items():
            if topic.lower() in user_input.lower():
                return f"I recall learning this about {topic}: {data['info']} (Learned on: {data['timestamp']})"

        return super().generate_response(user_input)

# Usage example
agent = LLMAgent("gpt2-medium")
print(agent.generate_response("Learn: Python: Python is a high-level programming language."))
print(agent.generate_response("What can you tell me about Python?"))
```

Slide 14: Real-Life Example - Personal Assistant

Let's implement a simple personal assistant using our LLM agent. This example demonstrates how our agent can handle various tasks in a practical scenario.

```python
import datetime
import requests

class PersonalAssistant(LLMAgent):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)
        return f"Task added: {task}"

    def list_tasks(self):
        return "Your tasks:\n" + "\n".join(f"- {task}" for task in self.tasks)

    def get_weather(self, city):
        # Note: You would need to sign up for an API key and use a real weather API
        api_key = "your_api_key_here"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
        response = requests.get(url)
        data = response.json()
        return f"The weather in {city} is {data['weather'][0]['description']}"

    def process_command(self, user_input):
        if user_input.lower().startswith("add task:"):
            return self.add_task(user_input[9:].strip())
        elif user_input.lower() == "list tasks":
            return self.list_tasks()
        elif user_input.lower().startswith("weather in"):
            city = user_input[11:].strip()
            return self.get_weather(city)
        elif user_input.lower() == "time":
            return f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}"
        else:
            return self.generate_response(user_input)

# Usage example
assistant = PersonalAssistant("gpt2-medium")
print(assistant.process_command("Add task: Buy groceries"))
print(assistant.process_command("Add task: Call mom"))
print(assistant.process_command("List tasks"))
print(assistant.process_command("Weather in New York"))
print(assistant.process_command("Time"))
print(assistant.process_command("Tell me a joke"))
```

Slide 15: Additional Resources

For further exploration of LLM agents and advanced natural language processing techniques, consider the following resources:

1. "Attention Is All You Need" by Vaswani et al. (2017) - The original transformer paper: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018): [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Language Models are Few-Shot Learners" by Brown et al. (2020) - Introduces GPT-3: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
4. "Hugging Face Transformers Library Documentation": [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
5. "The Natural Language Decathlon: Multitask Learning as Question Answering" by McCann et al. (2018): [https://arxiv.org/abs/1806.08730](https://arxiv.org/abs/1806.08730)

These resources provide a solid foundation for understanding the underlying principles and advanced techniques in building LLM agents. Remember to verify the information and check for the most recent developments in this rapidly evolving field.

Slide 16: Creating a Web Interface

To make our LLM agent accessible via a web browser, we'll create a simple web interface using Flask, a lightweight Python web framework.

```python
from flask import Flask, render_template, request, jsonify
from llm_agent import LLMAgent

app = Flask(__name__)
agent = LLMAgent("gpt2-medium")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response = agent.generate_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
```

Create a templates folder and add an index.html file:

```html
<!DOCTYPE html>
<html>
<head>
    <title>LLM Agent Chat</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
</head>
<body>
    <h1>Chat with LLM Agent</h1>
    <div id="chat-log"></div>
    <input type="text" id="user-input" placeholder="Type your message...">
    <button onclick="sendMessage()">Send</button>

    <script>
        function sendMessage() {
            var userInput = $('#user-input').val();
            $('#chat-log').append('<p><strong>You:</strong> ' + userInput + '</p>');
            $('#user-input').val('');

            $.ajax({
                url: '/chat',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({message: userInput}),
                success: function(response) {
                    $('#chat-log').append('<p><strong>Agent:</strong> ' + response.response + '</p>');
                }
            });
        }
    </script>
</body>
</html>
```

Slide 17: Containerizing the Application

To make our application easier to deploy, we'll containerize it using Docker. This ensures consistency across different environments.

Create a Dockerfile in your project root:

```dockerfile
FROM python:3.8-slim-buster

WORKDIR /app

 requirements.txt requirements.txt
RUN pip install -r requirements.txt

 . .

CMD ["python", "app.py"]
```

Create a requirements.txt file:

```
flask
transformers
torch
```

Build and run the Docker container:

```bash
docker build -t llm-agent .
docker run -p 5000:5000 llm-agent
```

Slide 18: Deploying to a Cloud Service (AWS)

We'll deploy our containerized application to Amazon Web Services (AWS) using Elastic Container Service (ECS).

1. Push your Docker image to Amazon Elastic Container Registry (ECR):

```bash
aws ecr create-repository --repository-name llm-agent
aws ecr get-login-password --region region | docker login --username AWS --password-stdin account-id.dkr.ecr.region.amazonaws.com
docker tag llm-agent:latest account-id.dkr.ecr.region.amazonaws.com/llm-agent:latest
docker push account-id.dkr.ecr.region.amazonaws.com/llm-agent:latest
```

2. Create an ECS cluster:

```bash
aws ecs create-cluster --cluster-name llm-agent-cluster
```

3. Create a task definition (task-definition.json):

```json
{
  "family": "llm-agent-task",
  "containerDefinitions": [
    {
      "name": "llm-agent",
      "image": "account-id.dkr.ecr.region.amazonaws.com/llm-agent:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "hostPort": 5000,
          "protocol": "tcp"
        }
      ],
      "memory": 512,
      "cpu": 256
    }
  ],
  "requiresCompatibilities": [
    "FARGATE"
  ],
  "networkMode": "awsvpc",
  "memory": "512",
  "cpu": "256"
}
```

Register the task definition:

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

4. Create and run an ECS service:

```bash
aws ecs create-service --cluster llm-agent-cluster --service-name llm-agent-service --task-definition llm-agent-task --desired-count 1 --launch-type FARGATE --network-configuration "awsvpcConfiguration={subnets=[subnet-12345678],securityGroups=[sg-12345678],assignPublicIp=ENABLED}"
```

Slide 19: Scaling and Monitoring

To ensure our LLM agent can handle increased traffic and maintain performance, we'll implement scaling and monitoring.

1. Set up Auto Scaling for ECS:

```bash
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/llm-agent-cluster/llm-agent-service \
  --min-capacity 1 \
  --max-capacity 10

aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/llm-agent-cluster/llm-agent-service \
  --policy-name cpu-tracking-scaling-policy \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

Create a scaling-policy.json file:

```json
{
  "TargetValue": 75.0,
  "PredefinedMetricSpecification": {
    "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
  },
  "ScaleOutCooldown": 60,
  "ScaleInCooldown": 60
}
```

2. Set up CloudWatch for monitoring:

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name llm-agent-cpu-high \
  --alarm-description "CPU utilization high" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 60 \
  --threshold 70 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=ClusterName,Value=llm-agent-cluster Name=ServiceName,Value=llm-agent-service \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:region:account-id:llm-agent-alerts
```

These additional slides cover the process of creating a web interface for the LLM agent, containerizing the application, deploying it to AWS, and setting up scaling and monitoring. This provides a comprehensive guide to taking the LLM agent from a local Python script to a scalable, cloud-hosted web application.
