## Comparing Python and C++ Key Concepts and Differences

Slide 1: Introduction to Python and C++

Python and C++ are powerful programming languages with distinct features. This slideshow compares their syntax and usage for various programming concepts.

Slide 2: Python Variables

Variables in Python are dynamically typed and can change type during execution.

```python
# Storing user information
name = "Alice"
age = 30
height = 1.75  # in meters

print(f"{name} is {age} years old and {height}m tall.")
# Output: Alice is 30 years old and 1.75m tall.

# Changing variable types
age = "thirty"  # Now age is a string
print(f"{name}'s age: {age}")
# Output: Alice's age: thirty
```

Slide 3: C++ Variables

C++ variables are statically typed and must be declared with their type.

```cpp
#include <iostream>
#include <string>

int main() {
    std::string name = "Bob";
    int age = 25;
    double height = 1.80;  // in meters

    std::cout << name << " is " << age << " years old and "
              << height << "m tall." << std::endl;
    // Output: Bob is 25 years old and 1.8m tall.

    // Changing variable types requires explicit conversion
    age = static_cast<int>(height);
    std::cout << name << "'s new age: " << age << std::endl;
    // Output: Bob's new age: 1
    
    return 0;
}
```

Slide 4: Python Functions

Python functions are defined using the `def` keyword and use indentation for scope.

```python
def calculate_bmi(weight, height):
    """Calculate BMI given weight (kg) and height (m)."""
    return weight / (height ** 2)

# Using the function
weight = 70  # kg
height = 1.75  # m
bmi = calculate_bmi(weight, height)
print(f"BMI: {bmi:.2f}")
# Output: BMI: 22.86
```

Slide 5: C++ Functions

C++ functions require explicit return types and use braces for scope.

```cpp
#include <iostream>

double calculateBMI(double weight, double height) {
    // Calculate BMI given weight (kg) and height (m)
    return weight / (height * height);
}

int main() {
    double weight = 70;  // kg
    double height = 1.75;  // m
    double bmi = calculateBMI(weight, height);
    std::cout << "BMI: " << std::fixed << std::setprecision(2) << bmi << std::endl;
    // Output: BMI: 22.86
    return 0;
}
```

Slide 6: Python Lists

Python lists are dynamic and can hold multiple data types.

```python
# Creating a shopping list
shopping_list = ["apples", "bread", "cheese", 5, True]

# Adding items
shopping_list.append("milk")

# Accessing items
print(f"First item: {shopping_list[0]}")
# Output: First item: apples

# Slicing
print(f"First three items: {shopping_list[:3]}")
# Output: First three items: ['apples', 'bread', 'cheese']

# List comprehension
prices = [2.5, 1.8, 3.2, 4.0, 1.5, 2.0]
expensive_items = [price for price in prices if price > 2.5]
print(f"Expensive items: {expensive_items}")
# Output: Expensive items: [3.2, 4.0]
```

Slide 7: C++ Vectors

C++ vectors are dynamic arrays that can grow or shrink in size.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    // Creating a shopping list
    std::vector<std::string> shoppingList = {"apples", "bread", "cheese"};

    // Adding items
    shoppingList.push_back("milk");

    // Accessing items
    std::cout << "First item: " << shoppingList[0] << std::endl;
    // Output: First item: apples

    // Iterating through the vector
    std::cout << "Shopping list: ";
    for (const auto& item : shoppingList) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    // Output: Shopping list: apples bread cheese milk

    // Using algorithms
    std::vector<double> prices = {2.5, 1.8, 3.2, 4.0, 1.5, 2.0};
    std::vector<double> expensiveItems;
    std::_if(prices.begin(), prices.end(), std::back_inserter(expensiveItems),
                 [](double price) { return price > 2.5; });

    std::cout << "Expensive items: ";
    for (const auto& price : expensiveItems) {
        std::cout << price << " ";
    }
    std::cout << std::endl;
    // Output: Expensive items: 3.2 4

    return 0;
}
```

Slide 8: Python Classes

Python classes use a simple syntax for object-oriented programming.

```python
class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.speed = 0

    def accelerate(self, amount):
        self.speed += amount
        print(f"The {self.make} {self.model} is now going {self.speed} km/h")

# Creating and using a Car object
my_car = Car("Toyota", "Corolla", 2022)
my_car.accelerate(30)
my_car.accelerate(20)
# Output:
# The Toyota Corolla is now going 30 km/h
# The Toyota Corolla is now going 50 km/h
```

Slide 9: C++ Classes

C++ classes separate declaration and implementation, and use access specifiers.

```cpp
#include <iostream>
#include <string>

class Car {
private:
    std::string make;
    std::string model;
    int year;
    int speed;

public:
    Car(std::string m, std::string mod, int y) : make(m), model(mod), year(y), speed(0) {}

    void accelerate(int amount) {
        speed += amount;
        std::cout << "The " << make << " " << model << " is now going " 
                  << speed << " km/h" << std::endl;
    }
};

int main() {
    Car myCar("Honda", "Civic", 2023);
    myCar.accelerate(40);
    myCar.accelerate(20);
    // Output:
    // The Honda Civic is now going 40 km/h
    // The Honda Civic is now going 60 km/h
    return 0;
}
```

Slide 10: Python File I/O

Python offers simple file handling with context managers.

```python
# Writing to a file
with open("example.txt", "w") as file:
    file.write("Hello, Python!\n")
    file.write("File I/O is easy.")

# Reading from a file
with open("example.txt", "r") as file:
    content = file.read()
    print(content)

# Output:
# Hello, Python!
# File I/O is easy.
```

Slide 11: C++ File I/O

C++ uses streams for file input and output operations.

```cpp
#include <iostream>
#include <fstream>
#include <string>

int main() {
    // Writing to a file
    std::ofstream outFile("example.txt");
    if (outFile.is_open()) {
        outFile << "Hello, C++!\n";
        outFile << "File I/O requires more setup.";
        outFile.close();
    }

    // Reading from a file
    std::ifstream inFile("example.txt");
    if (inFile.is_open()) {
        std::string line;
        while (getline(inFile, line)) {
            std::cout << line << std::endl;
        }
        inFile.close();
    }

    // Output:
    // Hello, C++!
    // File I/O requires more setup.

    return 0;
}
```

Slide 12: Python Error Handling

Python uses try-except blocks for error handling.

```python
def divide_numbers(a, b):
    try:
        result = a / b
        print(f"Result: {result}")
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")
    except TypeError:
        print("Error: Invalid input types!")
    finally:
        print("Division operation attempted.")

divide_numbers(10, 2)  # Result: 5.0
divide_numbers(10, 0)  # Error: Cannot divide by zero!
divide_numbers("10", 2)  # Error: Invalid input types!

# Output for all calls:
# Division operation attempted.
```

Slide 13: C++ Error Handling

C++ uses try-catch blocks for exception handling.

```cpp
#include <iostream>
#include <stdexcept>

double divideNumbers(double a, double b) {
    if (b == 0) {
        throw std::runtime_error("Cannot divide by zero!");
    }
    return a / b;
}

int main() {
    try {
        std::cout << "Result: " << divideNumbers(10, 2) << std::endl;
        std::cout << "Result: " << divideNumbers(10, 0) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    std::cout << "Program continues after error handling." << std::endl;

    // Output:
    // Result: 5
    // Error: Cannot divide by zero!
    // Program continues after error handling.

    return 0;
}
```

Slide 14: Wrap-up - Python vs C++ Comparison

```
| Feature          | Python                            | C++                                     |
|------------------|-----------------------------------|-----------------------------------------|
| Typing           | Dynamic                           | Static                                  |
| Syntax           | Indentation-based                 | Curly braces and semicolons             |
| Memory Mgmt      | Automatic (garbage collection)    | Manual (with smart pointers)            |
| Performance      | Interpreted, generally slower     | Compiled, generally faster              |
| Use Cases        | Web, data science, AI, scripting  | Systems, games, performance-critical    |
| Learning Curve   | Easier for beginners              | Steeper learning curve                  |
| OOP              | Everything is an object           | Supports both OOP and procedural        |
| Standard Library | Extensive built-in functionality  | Smaller, but growing (C++20)            |
| Portability      | Highly portable                   | Portable, but may require recompilation |
```

This concludes our comparison of Python and C++. Each language has its strengths and is suited for different types of projects. Choose based on your specific needs and requirements.

