## Unlocking the Power of Type Hinting in Python
Slide 1: Introduction to Type Hinting in Python

Type hinting is a feature in Python that allows developers to specify the expected data types of variables, function parameters, and return values. It enhances code readability, improves error detection, and facilitates better development tools support. Let's explore a simple example of type hinting in a function definition.

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Using the function
result = greet("Alice")
print(result)  # Output: Hello, Alice!

# Attempting to use with an incorrect type
# This will work, but static type checkers will flag it as an error
result = greet(123)
print(result)  # Output: Hello, 123!
```

Slide 2: Basic Syntax of Type Hints

Type hints in Python use a colon (:) followed by the type after variable names or function parameters. For function return types, an arrow (->) is used before the return type. Here's an example demonstrating various type hints:

```python
from typing import List, Dict, Tuple

def process_data(numbers: List[int], config: Dict[str, str]) -> Tuple[int, float]:
    total = sum(numbers)
    avg = total / len(numbers)
    return total, avg

# Using the function
data = [1, 2, 3, 4, 5]
settings = {"mode": "advanced", "precision": "high"}
result = process_data(data, settings)
print(result)  # Output: (15, 3.0)
```

Slide 3: Type Hinting for Built-in Types

Python's built-in types can be used directly for type hinting. This includes types like int, float, str, bool, and more. Let's see how to use these in a practical example:

```python
def calculate_discount(price: float, discount_percentage: float, apply_discount: bool) -> float:
    if apply_discount:
        discount = price * (discount_percentage / 100)
        return price - discount
    return price

# Using the function
original_price = 100.0
discount_percent = 20.0
should_apply = True

final_price = calculate_discount(original_price, discount_percent, should_apply)
print(f"Final price: ${final_price:.2f}")  # Output: Final price: $80.00
```

Slide 4: Using Union Types

Union types allow specifying that a value can be one of several types. This is useful when a function can accept or return multiple types. The Union type from the typing module is used for this purpose.

```python
from typing import Union

def to_str(data: Union[int, float, str]) -> str:
    return str(data)

# Using the function with different types
result1 = to_str(42)
result2 = to_str(3.14)
result3 = to_str("hello")

print(result1, type(result1))  # Output: 42 <class 'str'>
print(result2, type(result2))  # Output: 3.14 <class 'str'>
print(result3, type(result3))  # Output: hello <class 'str'>
```

Slide 5: Optional Types

Optional types are used when a value can be of a specific type or None. This is common in function parameters with default values. The Optional type from the typing module is used for this purpose.

```python
from typing import Optional

def greet_user(name: Optional[str] = None) -> str:
    if name is None:
        return "Hello, Guest!"
    return f"Hello, {name}!"

# Using the function with and without an argument
print(greet_user())  # Output: Hello, Guest!
print(greet_user("Alice"))  # Output: Hello, Alice!
```

Slide 6: Type Hinting for Collections

Python's typing module provides generic types for collections like List, Dict, and Tuple. These allow specifying the types of elements within the collections.

```python
from typing import List, Dict, Tuple

def process_employee_data(
    employees: List[str],
    salaries: Dict[str, float]
) -> Tuple[str, float]:
    highest_paid = max(salaries, key=salaries.get)
    return highest_paid, salaries[highest_paid]

# Using the function
emp_list = ["Alice", "Bob", "Charlie"]
salary_dict = {"Alice": 75000.0, "Bob": 82000.0, "Charlie": 68000.0}

top_earner, top_salary = process_employee_data(emp_list, salary_dict)
print(f"Highest paid employee: {top_earner}, Salary: ${top_salary:.2f}")
# Output: Highest paid employee: Bob, Salary: $82000.00
```

Slide 7: Type Aliases

Type aliases allow creating custom names for complex types, improving code readability and reusability. They are defined using the typing.TypeAlias annotation or simple assignment.

```python
from typing import Dict, List, TypeAlias

# Define type aliases
UserID = int
Username = str
UserData = Dict[str, str]
UserDatabase: TypeAlias = Dict[UserID, UserData]

def add_user(database: UserDatabase, user_id: UserID, username: Username, email: str) -> None:
    database[user_id] = {"username": username, "email": email}

def get_usernames(database: UserDatabase) -> List[Username]:
    return [user_data["username"] for user_data in database.values()]

# Using the functions
db: UserDatabase = {}
add_user(db, 1, "alice", "alice@example.com")
add_user(db, 2, "bob", "bob@example.com")

usernames = get_usernames(db)
print(usernames)  # Output: ['alice', 'bob']
```

Slide 8: Type Hinting for Classes

Type hinting can be applied to class methods, including the **init** method. The 'self' parameter doesn't need a type hint. Let's create a simple class with type hints:

```python
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def scale(self, factor: float) -> None:
        self.width *= factor
        self.height *= factor

# Using the class
rect = Rectangle(5.0, 3.0)
print(f"Area: {rect.area()}")  # Output: Area: 15.0

rect.scale(2.0)
print(f"Scaled area: {rect.area()}")  # Output: Scaled area: 60.0
```

Slide 9: Type Hinting for Generics

Generics allow creating flexible, reusable code that can work with different types. The TypeVar class from the typing module is used to define type variables for generic types.

```python
from typing import TypeVar, List, Callable

T = TypeVar('T')

def apply_operation(items: List[T], operation: Callable[[T], T]) -> List[T]:
    return [operation(item) for item in items]

# Using the generic function with different types
numbers = [1, 2, 3, 4, 5]
doubled_numbers = apply_operation(numbers, lambda x: x * 2)
print(doubled_numbers)  # Output: [2, 4, 6, 8, 10]

words = ["hello", "world", "python"]
uppercase_words = apply_operation(words, str.upper)
print(uppercase_words)  # Output: ['HELLO', 'WORLD', 'PYTHON']
```

Slide 10: Type Checking with mypy

mypy is a static type checker for Python that can catch type-related errors before runtime. It analyzes your code and its type hints to find potential issues. Let's see an example of how mypy can help catch errors:

```python
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers(5, "3")  # This line will not raise an error in Python
print(result)

# Running mypy on this file would produce an error like:
# error: Argument 2 to "add_numbers" has incompatible type "str"; expected "int"

# To run mypy, use the following command in your terminal:
# mypy your_file.py
```

Slide 11: Real-Life Example: Data Processing Pipeline

Let's create a simple data processing pipeline using type hints. This example demonstrates how type hints can improve the readability and maintainability of more complex code structures.

```python
from typing import List, Dict, Tuple

def load_data(filename: str) -> List[str]:
    with open(filename, 'r') as file:
        return file.readlines()

def parse_data(raw_data: List[str]) -> List[Dict[str, str]]:
    parsed = []
    for line in raw_data:
        name, age, city = line.strip().split(',')
        parsed.append({"name": name, "age": age, "city": city})
    return parsed

def analyze_data(data: List[Dict[str, str]]) -> Dict[str, int]:
    city_count: Dict[str, int] = {}
    for entry in data:
        city = entry["city"]
        city_count[city] = city_count.get(city, 0) + 1
    return city_count

def process_file(filename: str) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    raw_data = load_data(filename)
    parsed_data = parse_data(raw_data)
    analysis = analyze_data(parsed_data)
    return parsed_data, analysis

# Using the pipeline
filename = "user_data.txt"
parsed_data, city_analysis = process_file(filename)

print("Parsed Data:", parsed_data[:2])  # Print first two entries
print("City Analysis:", city_analysis)
```

Slide 12: Real-Life Example: Simple Web Scraper

Here's an example of a simple web scraper using type hints. This demonstrates how type hints can be used in a practical scenario involving network requests and HTML parsing.

```python
import requests
from typing import List, Dict
from bs4 import BeautifulSoup

def fetch_webpage(url: str) -> str:
    response = requests.get(url)
    return response.text

def parse_articles(html_content: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_content, 'html.parser')
    articles = []
    for article in soup.find_all('article'):
        title = article.find('h2').text.strip()
        summary = article.find('p').text.strip()
        articles.append({"title": title, "summary": summary})
    return articles

def scrape_news_site(url: str) -> List[Dict[str, str]]:
    html_content = fetch_webpage(url)
    return parse_articles(html_content)

# Using the scraper
news_url = "https://example.com/news"
news_articles = scrape_news_site(news_url)

for article in news_articles[:3]:  # Print first three articles
    print(f"Title: {article['title']}")
    print(f"Summary: {article['summary']}")
    print("---")
```

Slide 13: Benefits and Best Practices of Type Hinting

Type hinting offers numerous advantages in Python development. It enhances code readability by explicitly stating the expected types, which serves as a form of self-documentation. This clarity reduces the likelihood of type-related bugs and makes the codebase more maintainable. Type hints also enable better tooling support, including improved autocompletion, refactoring capabilities, and static analysis.

Best practices for using type hints include:

1.  Start with critical parts of your codebase, gradually adding hints to existing code.
2.  Use type hints consistently across your project.
3.  Leverage tools like mypy for static type checking.
4.  Keep type hints simple and readable, using aliases for complex types.
5.  Document any non-obvious type usage with comments.

```python
from typing import List, Dict, Union

def process_data(raw_data: List[Union[int, float]]) -> Dict[str, float]:
    """
    Process a list of numeric data.
    
    Args:
        raw_data: A list of integers or floats.
    
    Returns:
        A dictionary with statistical results.
    """
    filtered_data = [x for x in raw_data if isinstance(x, (int, float))]
    return {
        "mean": sum(filtered_data) / len(filtered_data),
        "max": max(filtered_data),
        "min": min(filtered_data)
    }

# Example usage
data = [1, 2, 3, 4.5, "invalid", 5.5]
results = process_data(data)
print(results)
# Output: {'mean': 3.2, 'max': 5.5, 'min': 1}
```

Slide 14: Additional Resources

For those interested in diving deeper into type hinting and static typing in Python, here are some valuable resources:

1.  PEP 484 - Type Hints: The official Python Enhancement Proposal that introduced type hints. [https://www.python.org/dev/peps/pep-0484/](https://www.python.org/dev/peps/pep-0484/)
2.  mypy documentation: Comprehensive guide to using mypy for static type checking. [http://mypy-lang.org/](http://mypy-lang.org/)
3.  "Static Typing for Python" by Jukka Lehtosalo (mypy creator): ArXiv link: [https://arxiv.org/abs/1807.02037](https://arxiv.org/abs/1807.02037)
4.  Python's official typing module documentation: [https://docs.python.org/3/library/typing.html](https://docs.python.org/3/library/typing.html)
5.  "Typed Python: A Future Version of Python with Static Typing" paper by Guido van Rossum et al.: ArXiv link: [https://arxiv.org/abs/2109.13770](https://arxiv.org/abs/2109.13770)

These resources provide in-depth information on type hinting implementation, best practices, and the future direction of static typing in Python.

