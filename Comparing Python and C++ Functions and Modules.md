## Comparing Python and C++ Functions and Modules

Slide 1: Python - Basic Function Definition

Python - Defining a Simple Function

Functions in Python are defined using the 'def' keyword, followed by the function name and parameters.

```python
def greet(name):
    return f"Hello, {name}!"

# Using the function
user = "Alice"
message = greet(user)
print(message)  # Output: Hello, Alice!
```

Slide 2: C++ - Basic Function Definition

C++ - Defining a Simple Function

C++ functions are typically declared in header files and defined in source files.

```cpp
#include <iostream>
#include <string>

std::string greet(const std::string& name) {
    return "Hello, " + name + "!";
}

int main() {
    std::string user = "Alice";
    std::string message = greet(user);
    std::cout << message << std::endl;  // Output: Hello, Alice!
    return 0;
}
```

Slide 3: Python - Function Parameters

Python - Function Parameters and Default Values

Python supports flexible parameter handling, including default values and keyword arguments.

```python
def calculate_price(item, quantity=1, discount=0):
    base_price = 10  # Assume $10 per item
    total = base_price * quantity * (1 - discount)
    return total

# Different ways to call the function
print(calculate_price("widget"))  # Output: 10.0
print(calculate_price("gadget", 3))  # Output: 30.0
print(calculate_price("gizmo", discount=0.1))  # Output: 9.0
print(calculate_price("doohickey", 2, 0.2))  # Output: 16.0
```

Slide 4: C++ - Function Parameters

C++ - Function Parameters and Default Values

C++ also supports default parameter values, but with some restrictions compared to Python.

```cpp
#include <iostream>

double calculate_price(const char* item, int quantity = 1, double discount = 0) {
    double base_price = 10;  // Assume $10 per item
    return base_price * quantity * (1 - discount);
}

int main() {
    std::cout << calculate_price("widget") << std::endl;  // Output: 10
    std::cout << calculate_price("gadget", 3) << std::endl;  // Output: 30
    std::cout << calculate_price("gizmo", 1, 0.1) << std::endl;  // Output: 9
    return 0;
}
```

Slide 5: Python - Return Values

Python - Multiple Return Values

Python functions can easily return multiple values using tuples.

```python
def analyze_text(text):
    word_count = len(text.split())
    char_count = len(text)
    avg_word_length = char_count / word_count if word_count > 0 else 0
    return word_count, char_count, avg_word_length

sample = "Python is awesome!"
words, chars, avg_length = analyze_text(sample)
print(f"Words: {words}, Characters: {chars}, Avg Length: {avg_length:.2f}")
# Output: Words: 3, Characters: 20, Avg Length: 6.67
```

Slide 6: C++ - Return Values

C++ - Multiple Return Values

C++ traditionally uses output parameters or structs for multiple returns, but modern C++ offers std::tuple.

```cpp
#include <iostream>
#include <string>
#include <tuple>
#include <sstream>

std::tuple<int, int, double> analyze_text(const std::string& text) {
    int word_count = 0;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) ++word_count;
    
    int char_count = text.length();
    double avg_word_length = word_count > 0 ? static_cast<double>(char_count) / word_count : 0;
    
    return {word_count, char_count, avg_word_length};
}

int main() {
    std::string sample = "C++ is powerful!";
    auto [words, chars, avg_length] = analyze_text(sample);
    std::cout << "Words: " << words << ", Characters: " << chars 
              << ", Avg Length: " << avg_length << std::endl;
    // Output: Words: 3, Characters: 18, Avg Length: 6
    return 0;
}
```

Slide 7: Python - Lambda Functions

Python - Lambda Functions

Python supports small anonymous functions called lambda functions.

```python
# Using lambda with built-in functions
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(squared)  # Output: [1, 4, 9, 16, 25]

# Using lambda with custom sorting
fruits = ["apple", "banana", "cherry", "date"]
sorted_fruits = sorted(fruits, key=lambda x: len(x))
print(sorted_fruits)  # Output: ['date', 'apple', 'banana', 'cherry']
```

Slide 8: C++ - Lambda Functions

C++ - Lambda Functions

C++ also supports lambda functions with a slightly different syntax.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::vector<int> squared;
    
    // Using lambda with std::transform
    std::transform(numbers.begin(), numbers.end(), std::back_inserter(squared),
                   [](int x) { return x * x; });
    
    for (int num : squared) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // Output: 1 4 9 16 25

    std::vector<std::string> fruits = {"apple", "banana", "cherry", "date"};
    
    // Using lambda with std::sort
    std::sort(fruits.begin(), fruits.end(), 
              [](const std::string& a, const std::string& b) {
                  return a.length() < b.length();
              });
    
    for (const auto& fruit : fruits) {
        std::cout << fruit << " ";
    }
    std::cout << std::endl;  // Output: date apple banana cherry
    
    return 0;
}
```

Slide 9: Python - Modules

Python - Creating and Using Modules

Python modules are files containing Python definitions and statements. They allow code organization and reuse.

```python
# File: math_operations.py
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

# File: main.py
import math_operations

result1 = math_operations.add(5, 3)
result2 = math_operations.multiply(4, 2)

print(f"5 + 3 = {result1}")  # Output: 5 + 3 = 8
print(f"4 * 2 = {result2}")  # Output: 4 * 2 = 8

# Alternative import style
from math_operations import multiply
print(f"6 * 7 = {multiply(6, 7)}")  # Output: 6 * 7 = 42
```

Slide 10: C++ - Header Files and Implementation Files

C++ - Header Files and Implementation Files

C++ typically separates function declarations (in header files) from their implementations (in source files).

```cpp
// File: math_operations.h
#ifndef MATH_OPERATIONS_H
#define MATH_OPERATIONS_H

int add(int a, int b);
int multiply(int a, int b);

#endif

// File: math_operations.cpp
#include "math_operations.h"

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

// File: main.cpp
#include <iostream>
#include "math_operations.h"

int main() {
    int result1 = add(5, 3);
    int result2 = multiply(4, 2);
    
    std::cout << "5 + 3 = " << result1 << std::endl;  // Output: 5 + 3 = 8
    std::cout << "4 * 2 = " << result2 << std::endl;  // Output: 4 * 2 = 8
    
    return 0;
}
```

Slide 11: Python - Function Decorators

Python - Function Decorators

Decorators in Python allow modifying or enhancing functions without changing their code.

```python
def log_function_call(func):
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} returned: {result}")
        return result
    return wrapper

@log_function_call
def calculate_area(radius):
    return 3.14 * radius ** 2

area = calculate_area(5)
# Output:
# Calling function: calculate_area
# Function calculate_area returned: 78.5
print(f"Area: {area}")  # Output: Area: 78.5
```

Slide 12: C++ - Function Pointers and std::function

C++ - Function Pointers and std::function

C++ uses function pointers and std::function for similar functionality to Python's first-class functions.

```cpp
#include <iostream>
#include <functional>

double calculate_area(double radius) {
    return 3.14 * radius * radius;
}

double calculate_circumference(double radius) {
    return 2 * 3.14 * radius;
}

void process_circle(double radius, std::function<double(double)> operation) {
    std::cout << "Result: " << operation(radius) << std::endl;
}

int main() {
    double radius = 5.0;
    
    process_circle(radius, calculate_area);  // Output: Result: 78.5
    process_circle(radius, calculate_circumference);  // Output: Result: 31.4
    
    // Using lambda
    process_circle(radius, [](double r) { return r * r; });  // Output: Result: 25
    
    return 0;
}
```

Slide 13: Python - Variable Scope and Closures

Python - Variable Scope and Closures

Python's scope rules and closures allow for powerful function factories and data encapsulation.

```python
def create_multiplier(factor):
    def multiplier(x):
        return x * factor
    return multiplier

double = create_multiplier(2)
triple = create_multiplier(3)

print(double(5))  # Output: 10
print(triple(5))  # Output: 15

# Closure with mutable state
def counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

my_counter = counter()
print(my_counter())  # Output: 1
print(my_counter())  # Output: 2
print(my_counter())  # Output: 3
```

Slide 14: C++ - Variable Scope and Lambdas with Captures

C++ - Variable Scope and Lambdas with Captures

C++ uses lambda captures to achieve similar functionality to Python's closures.

```cpp
#include <iostream>
#include <functional>

std::function<int(int)> create_multiplier(int factor) {
    return [factor](int x) { return x * factor; };
}

int main() {
    auto double_func = create_multiplier(2);
    auto triple_func = create_multiplier(3);
    
    std::cout << double_func(5) << std::endl;  // Output: 10
    std::cout << triple_func(5) << std::endl;  // Output: 15
    
    // Lambda with mutable state
    auto counter = [count = 0]() mutable {
        return ++count;
    };
    
    std::cout << counter() << std::endl;  // Output: 1
    std::cout << counter() << std::endl;  // Output: 2
    std::cout << counter() << std::endl;  // Output: 3
    
    return 0;
}
```

Slide 15: Wrap-up - Python vs C++ Functions and Modules

```
| Feature                | Python                                   | C++                                       |
|------------------------|------------------------------------------|-------------------------------------------|
| Function Definition    | def function_name(parameters):           | return_type function_name(parameters) {   |
|                        |     # function body                      |     // function body                      |
|                        |                                          | }                                         |
|------------------------|------------------------------------------|-------------------------------------------|
| Default Parameters     | def func(param1, param2=default):        | int func(int param1, int param2 = default)|
|------------------------|------------------------------------------|-------------------------------------------|
| Multiple Return Values | return value1, value2, value3            | std::tuple<T1, T2, T3> or struct          |
|------------------------|------------------------------------------|-------------------------------------------|
| Lambda Functions       | lambda arguments: expression             | [capture](parameters) { body }            |
|------------------------|------------------------------------------|-------------------------------------------|
| Modules                | import module_name                       | #include "header_file.h"                  |
|------------------------|------------------------------------------|-------------------------------------------|
| Function Decorators    | @decorator                               | No direct equivalent (use templates)      |
|                        | def function():                          |                                           |
|------------------------|------------------------------------------|-------------------------------------------|
| Closures               | Supported with nested functions          | Achieved using lambdas with captures      |
|------------------------|------------------------------------------|-------------------------------------------|
| Variable Scope         | Global, local, nonlocal                  | Global, local, static                     |
```

