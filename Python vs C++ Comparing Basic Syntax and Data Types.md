## Python vs C++ Comparing Basic Syntax and Data Types

Slide 1: Python Variables

Python Variables: Dynamic and Flexible

Variables in Python are dynamically typed and can change types during execution.

```python
# Storing user information
name = "Alice"
age = 30
height = 1.65  # in meters

print(f"{name} is {age} years old and {height}m tall.")
# Output: Alice is 30 years old and 1.65m tall.

# Changing variable types
age = "thirty"  # age is now a string
print(f"{name}'s age: {age}")
# Output: Alice's age: thirty
```

Slide 2: C++ Variables

C++ Variables: Statically Typed and Explicit

C++ variables must be declared with a specific type before use.

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

Slide 3: Python Data Types - Numbers

Python Numbers: Flexible and Intuitive

Python supports integers, floating-point numbers, and complex numbers.

```python
# Temperature converter
celsius = 25
fahrenheit = (celsius * 9/5) + 32
kelvin = celsius + 273.15

print(f"{celsius}°C is {fahrenheit}°F or {kelvin}K")
# Output: 25°C is 77.0°F or 298.15K

# Complex numbers
z = 2 + 3j
print(f"Magnitude of {z} is {abs(z)}")
# Output: Magnitude of (2+3j) is 3.605551275463989
```

Slide 4: C++ Data Types - Numbers

C++ Numbers: Precise Control over Types

C++ offers various integer and floating-point types with specific sizes.

```cpp
#include <iostream>
#include <cmath>
#include <complex>

int main() {
    // Temperature converter
    int celsius = 25;
    double fahrenheit = (celsius * 9.0/5.0) + 32;
    double kelvin = celsius + 273.15;

    std::cout << celsius << "°C is " << fahrenheit << "°F or "
              << kelvin << "K" << std::endl;
    // Output: 25°C is 77°F or 298.15K

    // Complex numbers
    std::complex<double> z(2, 3);
    std::cout << "Magnitude of " << z << " is " << abs(z) << std::endl;
    // Output: Magnitude of (2,3) is 3.60555

    return 0;
}
```

Slide 5: Python Strings

Python Strings: Versatile and Easy to Manipulate

Python strings are immutable sequences of Unicode characters.

```python
# String operations
greeting = "Hello, World!"
name = "Alice"

# Concatenation and slicing
message = greeting[:-1] + " " + name + "!"
print(message)  # Output: Hello, Alice!

# String methods
print(message.upper())  # Output: HELLO, ALICE!
print(message.replace("Alice", "Bob"))  # Output: Hello, Bob!

# Formatting
age = 30
print(f"{name} is {age} years old.")  # Output: Alice is 30 years old.
```

Slide 6: C++ Strings

C++ Strings: Efficient and Mutable

C++ strings are mutable and offer both C-style and object-oriented approaches.

```cpp
#include <iostream>
#include <string>

int main() {
    // String operations
    std::string greeting = "Hello, World!";
    std::string name = "Alice";

    // Concatenation and substr
    std::string message = greeting.substr(0, greeting.length() - 1) + " " + name + "!";
    std::cout << message << std::endl;  // Output: Hello, Alice!

    // String methods
    for (char &c : message) c = toupper(c);
    std::cout << message << std::endl;  // Output: HELLO, ALICE!

    size_t pos = message.find("ALICE");
    if (pos != std::string::npos) {
        message.replace(pos, 5, "BOB");
    }
    std::cout << message << std::endl;  // Output: HELLO, BOB!

    // Formatting
    int age = 30;
    std::cout << name << " is " << age << " years old." << std::endl;
    // Output: Alice is 30 years old.

    return 0;
}
```

Slide 7: Python Lists

Python Lists: Dynamic and Versatile

Python lists are mutable sequences that can hold mixed data types.

```python
# Shopping list manager
shopping_list = ["apples", "bread", "milk"]

# Adding items
shopping_list.append("eggs")
shopping_list.extend(["cheese", "yogurt"])

# Removing items
if "bread" in shopping_list:
    shopping_list.remove("bread")

# Accessing and modifying
shopping_list[0] = "oranges"

print("Updated list:", shopping_list)
# Output: Updated list: ['oranges', 'milk', 'eggs', 'cheese', 'yogurt']

# List comprehension
prices = [1.5, 2.0, 3.5, 2.5, 1.0]
total = sum([price for price in prices if price > 2])
print(f"Total for expensive items: ${total}")
# Output: Total for expensive items: $6.0
```

Slide 8: C++ Vectors

C++ Vectors: Dynamic Arrays with Type Safety

Vectors in C++ are dynamic arrays that can grow or shrink in size.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

int main() {
    // Shopping list manager
    std::vector<std::string> shopping_list = {"apples", "bread", "milk"};

    // Adding items
    shopping_list.push_back("eggs");
    shopping_list.insert(shopping_list.end(), {"cheese", "yogurt"});

    // Removing items
    auto it = std::find(shopping_list.begin(), shopping_list.end(), "bread");
    if (it != shopping_list.end()) {
        shopping_list.erase(it);
    }

    // Accessing and modifying
    shopping_list[0] = "oranges";

    std::cout << "Updated list: ";
    for (const auto& item : shopping_list) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    // Output: Updated list: oranges milk eggs cheese yogurt

    // Using algorithms
    std::vector<double> prices = {1.5, 2.0, 3.5, 2.5, 1.0};
    double total = std::accumulate(prices.begin(), prices.end(), 0.0,
        [](double sum, double price) { return price > 2 ? sum + price : sum; });
    std::cout << "Total for expensive items: $" << total << std::endl;
    // Output: Total for expensive items: $6

    return 0;
}
```

Slide 9: Python Dictionaries

Python Dictionaries: Flexible Key-Value Pairs

Dictionaries in Python store key-value pairs with quick lookup.

```python
# Student grade tracker
grades = {
    "Alice": {"Math": 85, "Science": 92, "English": 78},
    "Bob": {"Math": 90, "Science": 88, "English": 85}
}

# Adding a new student
grades["Charlie"] = {"Math": 78, "Science": 80, "English": 92}

# Updating grades
grades["Alice"]["Math"] = 87

# Calculating average grade
def average_grade(student):
    return sum(grades[student].values()) / len(grades[student])

for student, subjects in grades.items():
    avg = average_grade(student)
    print(f"{student}'s average grade: {avg:.2f}")

# Output:
# Alice's average grade: 85.67
# Bob's average grade: 87.67
# Charlie's average grade: 83.33

# Dictionary comprehension
high_performers = {name: avg for name, avg in 
                   ((s, average_grade(s)) for s in grades) if avg > 85}
print("High performers:", high_performers)
# Output: High performers: {'Bob': 87.67}
```

Slide 10: C++ Maps

C++ Maps: Efficient Associative Containers

Maps in C++ store key-value pairs with fast key-based access.

```cpp
#include <iostream>
#include <map>
#include <string>
#include <numeric>

// Student grade tracker
using GradeMap = std::map<std::string, int>;
using StudentMap = std::map<std::string, GradeMap>;

double average_grade(const GradeMap& grades) {
    return std::accumulate(grades.begin(), grades.end(), 0.0,
        [](double sum, const auto& pair) { return sum + pair.second; }) / grades.size();
}

int main() {
    StudentMap grades = {
        {"Alice", {{"Math", 85}, {"Science", 92}, {"English", 78}}},
        {"Bob", {{"Math", 90}, {"Science", 88}, {"English", 85}}}
    };

    // Adding a new student
    grades["Charlie"] = {{"Math", 78}, {"Science", 80}, {"English", 92}};

    // Updating grades
    grades["Alice"]["Math"] = 87;

    // Calculating and displaying average grades
    for (const auto& [student, subjects] : grades) {
        double avg = average_grade(subjects);
        std::cout << student << "'s average grade: " << avg << std::endl;
    }

    // Output:
    // Alice's average grade: 85.6667
    // Bob's average grade: 87.6667
    // Charlie's average grade: 83.3333

    // Finding high performers
    std::map<std::string, double> high_performers;
    for (const auto& [student, subjects] : grades) {
        double avg = average_grade(subjects);
        if (avg > 85) {
            high_performers[student] = avg;
        }
    }

    std::cout << "High performers: ";
    for (const auto& [student, avg] : high_performers) {
        std::cout << student << " (" << avg << ") ";
    }
    std::cout << std::endl;
    // Output: High performers: Bob (87.6667)

    return 0;
}
```

Slide 11: Python Control Flow

Python Control Flow: Clean and Intuitive

Python uses indentation to define code blocks, making it readable.

```python
# Temperature classifier
def classify_temperature(temp):
    if temp < 0:
        return "Freezing"
    elif 0 <= temp < 10:
        return "Cold"
    elif 10 <= temp < 20:
        return "Cool"
    elif 20 <= temp < 30:
        return "Warm"
    else:
        return "Hot"

# Testing the classifier
temperatures = [-5, 2, 15, 28, 35]

for temp in temperatures:
    category = classify_temperature(temp)
    print(f"{temp}°C is {category}")

# Output:
# -5°C is Freezing
# 2°C is Cold
# 15°C is Cool
# 28°C is Warm
# 35°C is Hot

# Using a while loop for user input
while True:
    user_temp = input("Enter a temperature (or 'q' to quit): ")
    if user_temp.lower() == 'q':
        break
    try:
        user_temp = float(user_temp)
        print(f"{user_temp}°C is {classify_temperature(user_temp)}")
    except ValueError:
        print("Please enter a valid number or 'q' to quit.")
```

Slide 12: C++ Control Flow

C++ Control Flow: Explicit and Flexible

C++ uses curly braces to define code blocks and offers various control structures.

```cpp
#include <iostream>
#include <string>
#include <vector>

// Temperature classifier
std::string classify_temperature(double temp) {
    if (temp < 0) {
        return "Freezing";
    } else if (temp < 10) {
        return "Cold";
    } else if (temp < 20) {
        return "Cool";
    } else if (temp < 30) {
        return "Warm";
    } else {
        return "Hot";
    }
}

int main() {
    // Testing the classifier
    std::vector<double> temperatures = {-5, 2, 15, 28, 35};

    for (double temp : temperatures) {
        std::string category = classify_temperature(temp);
        std::cout << temp << "°C is " << category << std::endl;
    }

    // Output:
    // -5°C is Freezing
    // 2°C is Cold
    // 15°C is Cool
    // 28°C is Warm
    // 35°C is Hot

    // Using a while loop for user input
    while (true) {
        std::string user_input;
        std::cout << "Enter a temperature (or 'q' to quit): ";
        std::cin >> user_input;

        if (user_input == "q" || user_input == "Q") {
            break;
        }

        try {
            double user_temp = std::stod(user_input);
            std::cout << user_temp << "°C is " << classify_temperature(user_temp) << std::endl;
        } catch (const std::invalid_argument&) {
            std::cout << "Please enter a valid number or 'q' to quit." << std::endl;
        }
    }

    return 0;
}
```

Slide 13: Python Functions

Python Functions: Flexible and Powerful

Python functions support default arguments, keyword arguments, and variable-length arguments.

```python
# Flexible function for calculating discounted prices
def calculate_discount(price, discount_percent=10, max_discount=50):
    discount = min(price * (discount_percent / 100), max_discount)
    return price - discount

# Using the function with different arguments
original_price = 100

# Default discount
print(f"Default discount: ${calculate_discount(original_price):.2f}")
# Output: Default discount: $90.00

# Custom discount percentage
print(f"20% discount: ${calculate_discount(original_price, 20):.2f}")
# Output: 20% discount: $80.00

# Custom discount with maximum limit
print(f"30% discount (max $25): ${calculate_discount(original_price, 30, 25):.2f}")
# Output: 30% discount (max $25): $75.00

# Function with variable arguments
def calculate_total(*prices, tax_rate=0.08):
    subtotal = sum(prices)
    tax = subtotal * tax_rate
    return subtotal + tax

# Calculate total with variable number of items
total = calculate_total(10.99, 24.50, 5.95, 7.75)
print(f"Total (including 8% tax): ${total:.2f}")
# Output: Total (including 8% tax): $53.25
```

Slide 14: C++ Functions

C++ Functions: Type-Safe with Overloading

C++ functions are strongly typed and support function overloading.

```cpp
#include <iostream>
#include <vector>

// Function to calculate discounted price
double calculate_discount(double price, double discount_percent = 10, double max_discount = 50) {
    double discount = std::min(price * (discount_percent / 100), max_discount);
    return price - discount;
}

// Function overloading for integer prices
int calculate_discount(int price, int discount_percent = 10, int max_discount = 50) {
    int discount = std::min(price * discount_percent / 100, max_discount);
    return price - discount;
}

// Function with variable arguments using std::vector
double calculate_total(const std::vector<double>& prices, double tax_rate = 0.08) {
    double subtotal = 0;
    for (double price : prices) {
        subtotal += price;
    }
    return subtotal * (1 + tax_rate);
}

int main() {
    double original_price = 100.0;

    std::cout << "Default discount: $" << calculate_discount(original_price) << std::endl;
    std::cout << "20% discount: $" << calculate_discount(original_price, 20) << std::endl;
    std::cout << "30% discount (max $25): $" << calculate_discount(original_price, 30, 25) << std::endl;

    std::vector<double> items = {10.99, 24.50, 5.95, 7.75};
    double total = calculate_total(items);
    std::cout << "Total (including 8% tax): $" << total << std::endl;

    return 0;
}
```

Slide 15: Python Classes and Objects

Python Classes: Simple and Intuitive

Python classes use a straightforward syntax for object-oriented programming.

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(f"Deposited ${amount}. New balance: ${self.balance}")

    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
            print(f"Withdrew ${amount}. New balance: ${self.balance}")
        else:
            print("Insufficient funds!")

    def __str__(self):
        return f"{self.owner}'s account. Balance: ${self.balance}"

# Using the BankAccount class
account = BankAccount("Alice", 1000)
print(account)  # Output: Alice's account. Balance: $1000

account.deposit(500)  # Output: Deposited $500. New balance: $1500
account.withdraw(200)  # Output: Withdrew $200. New balance: $1300
account.withdraw(2000)  # Output: Insufficient funds!
```

Slide 16: C++ Classes and Objects

C++ Classes: Powerful and Efficient

C++ classes offer fine-grained control over member access and behavior.

```cpp
#include <iostream>
#include <string>

class BankAccount {
private:
    std::string owner;
    double balance;

public:
    BankAccount(const std::string& owner, double balance = 0)
        : owner(owner), balance(balance) {}

    void deposit(double amount) {
        balance += amount;
        std::cout << "Deposited $" << amount << ". New balance: $" << balance << std::endl;
    }

    void withdraw(double amount) {
        if (amount <= balance) {
            balance -= amount;
            std::cout << "Withdrew $" << amount << ". New balance: $" << balance << std::endl;
        } else {
            std::cout << "Insufficient funds!" << std::endl;
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const BankAccount& account) {
        os << account.owner << "'s account. Balance: $" << account.balance;
        return os;
    }
};

int main() {
    BankAccount account("Alice", 1000);
    std::cout << account << std::endl;  // Output: Alice's account. Balance: $1000

    account.deposit(500);  // Output: Deposited $500. New balance: $1500
    account.withdraw(200);  // Output: Withdrew $200. New balance: $1300
    account.withdraw(2000);  // Output: Insufficient funds!

    return 0;
}
```

Slide 17: Wrap-up: Python vs C++ Syntax and Data Types

```
| Feature       | Python                                   | C++                                       |
|---------------|------------------------------------------|-------------------------------------------|
| Variables     | Dynamic typing                           | Static typing                             |
| Numbers       | int, float, complex                      | int, float, double, long, etc.            |
| Strings       | Immutable, Unicode                       | std::string, mutable                      |
| Lists/Arrays  | Dynamic lists                            | Fixed arrays, std::vector                 |
| Dictionaries  | Built-in dict type                       | std::map, std::unordered_map              |
| Control Flow  | Indentation-based blocks                 | Curly brace-based blocks                  |
| Functions     | Flexible arguments, easy lambda          | Strict typing, function overloading       |
| Classes       | Simple syntax, dynamic attributes        | More verbose, better encapsulation        |
| Memory Mgmt   | Automatic (garbage collection)           | Manual (RAII, smart pointers)             |
| Performance   | Generally slower                         | Generally faster                          |
| Ease of Use   | More beginner-friendly                   | Steeper learning curve                    |
```

This wrap-up slide summarizes the key differences in syntax and data types between Python and C++, providing a quick reference for comparison.

