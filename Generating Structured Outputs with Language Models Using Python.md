## Generating Structured Outputs with Language Models Using Python
Slide 1: Introduction to Structured LLM Outputs

Structured LLM outputs allow us to generate specific formats of text from language models, making it easier to parse and use the results in various applications. This approach enhances the usability of LLM responses in tasks requiring structured data.

```python
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
prompt = "Generate a JSON object with name and age:"
response = generator(prompt, max_length=50)

print(response[0]['generated_text'])
```

Slide 2: Using JSON Templates

JSON templates provide a structured format for LLM outputs, allowing for consistent and easily parseable responses. By including a template in the prompt, we guide the model to generate data in the desired structure.

```python
import json
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
template = '{"name": "<NAME>", "age": <AGE>}'
prompt = f"Complete this JSON template: {template}"

response = generator(prompt, max_length=50)
generated_text = response[0]['generated_text']

# Extract the JSON part
json_start = generated_text.find('{')
json_end = generated_text.rfind('}') + 1
json_str = generated_text[json_start:json_end]

parsed_json = json.loads(json_str)
print(parsed_json)
```

Slide 3: Regex for Extracting Structured Data

Regular expressions can be used to extract specific patterns from LLM outputs, allowing for more flexible structured data generation. This approach is particularly useful when the output format is not strictly defined but follows a recognizable pattern.

```python
import re
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
prompt = "Generate a person's name and age in the format 'Name: John Doe, Age: 30'"
response = generator(prompt, max_length=50)

pattern = r"Name: (.*?), Age: (\d+)"
match = re.search(pattern, response[0]['generated_text'])

if match:
    name, age = match.groups()
    print(f"Extracted Name: {name}")
    print(f"Extracted Age: {age}")
else:
    print("Pattern not found in the generated text.")
```

Slide 4: Using Prompt Engineering for Structure

Carefully crafted prompts can guide LLMs to produce structured outputs without requiring additional post-processing. This technique relies on clear instructions and examples to shape the model's response.

```python
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
prompt = """
Generate a list of 3 fruits in the following format:
1. [Fruit Name]
2. [Fruit Name]
3. [Fruit Name]
"""

response = generator(prompt, max_length=100)
print(response[0]['generated_text'])
```

Slide 5: Leveraging Few-Shot Learning

Few-shot learning techniques can be employed to teach LLMs to generate structured outputs by providing examples in the prompt. This method is particularly effective for more complex structures or when consistency is crucial.

```python
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
prompt = """
Generate a product review in the following format:
Product: Smartphone X
Rating: 4/5
Pros: Fast processor, great camera
Cons: Average battery life
Summary: A solid choice for tech enthusiasts.

Now generate a review for a new laptop:
Product:"""

response = generator(prompt, max_length=200)
print(response[0]['generated_text'])
```

Slide 6: Implementing Output Parsers

Output parsers are custom functions designed to extract and structure the relevant information from LLM responses. They can handle various output formats and provide a consistent interface for working with generated data.

```python
from transformers import pipeline
import re

def parse_product_review(text):
    patterns = {
        'product': r'Product: (.+)',
        'rating': r'Rating: (\d+)/5',
        'pros': r'Pros: (.+)',
        'cons': r'Cons: (.+)',
        'summary': r'Summary: (.+)'
    }
    
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            result[key] = match.group(1)
    return result

generator = pipeline('text-generation', model='gpt2')
prompt = "Write a product review for a smartwatch:"
response = generator(prompt, max_length=200)

parsed_review = parse_product_review(response[0]['generated_text'])
print(parsed_review)
```

Slide 7: Using Delimiters for Structured Outputs

Delimiters can be used to separate different parts of the LLM output, making it easier to extract structured information. This technique is particularly useful when generating multiple distinct pieces of information in a single response.

```python
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
prompt = """
Generate a movie recommendation with the following information, separated by '|||':
Title ||| Genre ||| Release Year ||| Brief Description
"""

response = generator(prompt, max_length=100)
generated_text = response[0]['generated_text']

# Split the generated text by the delimiter
parts = generated_text.split('|||')
if len(parts) == 4:
    title, genre, year, description = [part.strip() for part in parts]
    print(f"{title}")
    print(f"Genre: {genre}")
    print(f"Year: {year}")
    print(f"{description}")
else:
    print("Unable to parse the generated text.")
```

Slide 8: Implementing a Custom Tokenizer

Custom tokenizers can be created to handle specific structured output formats. This approach allows for fine-grained control over how the LLM generates and processes structured data.

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

class StructuredTokenizer(GPT2Tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_special_tokens({'additional_special_tokens': ['<START>', '<END>', '<FIELD>']})

tokenizer = StructuredTokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

prompt = "<START>Name<FIELD>John Doe<END><START>Age<FIELD>"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
print(generated_text)
```

Slide 9: Using Type Hints for Structured Outputs

Type hints can be used to define the expected structure of LLM outputs, improving code readability and enabling static type checking. This approach is particularly useful when integrating structured LLM outputs into larger applications.

```python
from typing import TypedDict, List
from transformers import pipeline

class Person(TypedDict):
    name: str
    age: int
    hobbies: List[str]

def generate_person() -> Person:
    generator = pipeline('text-generation', model='gpt2')
    prompt = "Generate a person's name, age, and hobbies:"
    response = generator(prompt, max_length=100)
    
    # In a real scenario, you'd parse the response here
    # This is a simplified example
    return {
        "name": "John Doe",
        "age": 30,
        "hobbies": ["reading", "hiking", "photography"]
    }

person = generate_person()
print(f"Name: {person['name']}")
print(f"Age: {person['age']}")
print(f"Hobbies: {', '.join(person['hobbies'])}")
```

Slide 10: Implementing a Structured Output Pipeline

Creating a custom pipeline for generating structured outputs can streamline the process of working with LLMs. This approach encapsulates the generation, parsing, and validation steps into a reusable component.

```python
from transformers import pipeline
import json

class StructuredOutputPipeline:
    def __init__(self, model_name='gpt2'):
        self.generator = pipeline('text-generation', model=model_name)
    
    def generate(self, prompt, output_format):
        response = self.generator(prompt, max_length=200)
        generated_text = response[0]['generated_text']
        return self._parse_output(generated_text, output_format)
    
    def _parse_output(self, text, output_format):
        # Implement parsing logic based on output_format
        # This is a simplified example
        return json.loads(text)

pipeline = StructuredOutputPipeline()
prompt = "Generate a JSON object with a person's details:"
output_format = {"name": str, "age": int, "city": str}

result = pipeline.generate(prompt, output_format)
print(result)
```

Slide 11: Handling Multiple Output Formats

When working with various structured output formats, it's useful to implement a flexible system that can adapt to different requirements. This approach allows for greater versatility in generating structured LLM outputs.

```python
from typing import Dict, Any, Callable
from transformers import pipeline

class MultiFormatOutputGenerator:
    def __init__(self, model_name='gpt2'):
        self.generator = pipeline('text-generation', model=model_name)
        self.format_handlers: Dict[str, Callable] = {
            'json': self._handle_json,
            'csv': self._handle_csv,
            'xml': self._handle_xml
        }
    
    def generate(self, prompt: str, output_format: str) -> Any:
        response = self.generator(prompt, max_length=200)
        generated_text = response[0]['generated_text']
        
        handler = self.format_handlers.get(output_format.lower())
        if handler:
            return handler(generated_text)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _handle_json(self, text: str) -> Dict:
        # Implement JSON parsing logic
        pass
    
    def _handle_csv(self, text: str) -> List[List[str]]:
        # Implement CSV parsing logic
        pass
    
    def _handle_xml(self, text: str) -> Any:
        # Implement XML parsing logic
        pass

generator = MultiFormatOutputGenerator()
result = generator.generate("Generate a person's details", 'json')
print(result)
```

Slide 12: Validating Structured Outputs

Implementing validation for structured LLM outputs ensures that the generated data meets the required format and constraints. This step is crucial for maintaining data integrity and consistency in applications that rely on LLM-generated structured data.

```python
from pydantic import BaseModel, validator
from transformers import pipeline

class Person(BaseModel):
    name: str
    age: int
    email: str

    @validator('name')
    def name_must_contain_space(cls, v):
        if ' ' not in v:
            raise ValueError('Name must contain a space')
        return v

    @validator('age')
    def age_must_be_reasonable(cls, v):
        if v < 0 or v > 120:
            raise ValueError('Age must be between 0 and 120')
        return v

    @validator('email')
    def email_must_contain_at(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v

def generate_person_data():
    generator = pipeline('text-generation', model='gpt2')
    prompt = "Generate a person's name, age, and email:"
    response = generator(prompt, max_length=100)
    
    # In a real scenario, you'd parse the response here
    # This is a simplified example
    return {
        "name": "John Doe",
        "age": 30,
        "email": "john.doe@example.com"
    }

try:
    person_data = generate_person_data()
    validated_person = Person(**person_data)
    print(validated_person)
except ValueError as e:
    print(f"Validation error: {e}")
```

Slide 13: Combining LLMs with Structured Data Sources

Integrating LLMs with structured data sources, such as databases or APIs, allows for the generation of outputs that combine pre-existing structured data with dynamically generated content. This approach enables more contextually relevant and accurate structured outputs.

```python
import sqlite3
from transformers import pipeline

# Simulating a database connection
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE products
                  (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL)''')
cursor.execute("INSERT INTO products VALUES (1, 'Smartphone X', 'Electronics', 799.99)")
conn.commit()

def generate_product_description(product_id):
    # Fetch product data from the database
    cursor.execute("SELECT name, category, price FROM products WHERE id = ?", (product_id,))
    product = cursor.fetchone()
    
    if product:
        name, category, price = product
        
        # Generate description using LLM
        generator = pipeline('text-generation', model='gpt2')
        prompt = f"Write a product description for {name}, a {category} product priced at ${price}:"
        response = generator(prompt, max_length=150)
        
        return {
            "name": name,
            "category": category,
            "price": price,
            "description": response[0]['generated_text']
        }
    else:
        return None

product_info = generate_product_description(1)
print(product_info)

conn.close()
```

Slide 14: Additional Resources

For further exploration of structured LLM outputs and related topics, consider the following resources:

1. "Language Models are Few-Shot Learners" (Brown et al., 2020) arXiv: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
2. "Grounded Language Learning Fast and Slow" (Hill et al., 2020) arXiv: [https://arxiv.org/abs/2009.01719](https://arxiv.org/abs/2009.01719)
3. "Learning to Summarize from Human Feedback" (Stiennon et al., 2020) arXiv: [https://arxiv.org/abs/2009.01325](https://arxiv.org/abs/2009.01325)

These papers provide insights into advanced techniques for working with language models and structured outputs.

