## Python vs C++ Exception Handling and Debugging
Slide 1: Python - Basic Exception Handling

Python: Try-Except Basics

Python uses try-except blocks to handle exceptions gracefully.

Code:

```python
# Handling a potential divide-by-zero error
def safe_divide(a, b):
    try:
        result = a / b
        print(f"{a} divided by {b} is {result}")
    except ZeroDivisionError:
        print("Error: Cannot divide by zero")

safe_divide(10, 2)  # Output: 10 divided by 2 is 5.0
safe_divide(10, 0)  # Output: Error: Cannot divide by zero
```

Slide 2: C++ - Basic Exception Handling

C++: Try-Catch Basics

C++ uses try-catch blocks for exception handling.

Code:

```cpp
#include <iostream>
#include <stdexcept>

// Handling a potential out-of-range error
void access_array(int index) {
    int numbers[] = {1, 2, 3, 4, 5};
    try {
        if (index < 0 || index >= 5) {
            throw std::out_of_range("Index out of bounds");
        }
        std::cout << "Value at index " << index << ": " << numbers[index] << std::endl;
    } catch (const std::out_of_range& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main() {
    access_array(2);  // Output: Value at index 2: 3
    access_array(10); // Output: Error: Index out of bounds
    return 0;
}
```

Slide 3: Python - Multiple Exception Handling

Python: Handling Multiple Exceptions

Python can handle multiple exception types in a single try-except block.

Code:

```python
# Handling different types of exceptions
def process_data(data):
    try:
        result = int(data) * 2
        print(f"Processed result: {result}")
    except ValueError:
        print("Error: Invalid input. Please enter a number.")
    except TypeError:
        print("Error: Input must be a string or a number.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

process_data("10")    # Output: Processed result: 20
process_data("hello") # Output: Error: Invalid input. Please enter a number.
process_data(None)    # Output: Error: Input must be a string or a number.
```

Slide 4: C++ - Multiple Exception Handling

C++: Handling Multiple Exceptions

C++ can catch multiple exception types using multiple catch blocks.

Code:

```cpp
#include <iostream>
#include <stdexcept>
#include <string>

// Handling different types of exceptions
void process_data(const std::string& data) {
    try {
        int value = std::stoi(data);
        std::cout << "Processed result: " << (value * 2) << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: Invalid input. Please enter a number." << std::endl;
    } catch (const std::out_of_range& e) {
        std::cerr << "Error: Number out of range." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
    }
}

int main() {
    process_data("10");    // Output: Processed result: 20
    process_data("hello"); // Output: Error: Invalid input. Please enter a number.
    process_data("999999999999999999"); // Output: Error: Number out of range.
    return 0;
}
```

Slide 5: Python - Custom Exceptions

Python: Creating Custom Exceptions

Python allows defining custom exception classes for specific error scenarios.

Code:

```python
class InsufficientFundsError(Exception):
    pass

class BankAccount:
    def __init__(self, balance):
        self.balance = balance

    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError("Not enough funds in the account")
        self.balance -= amount
        print(f"Withdrawn ${amount}. New balance: ${self.balance}")

account = BankAccount(100)
try:
    account.withdraw(50)   # Output: Withdrawn $50. New balance: $50
    account.withdraw(100)  # Raises InsufficientFundsError
except InsufficientFundsError as e:
    print(f"Error: {str(e)}")  # Output: Error: Not enough funds in the account
```

Slide 6: C++ - Custom Exceptions

C++: Creating Custom Exceptions

C++ supports creating custom exception classes for specific error handling.

Code:

```cpp
#include <iostream>
#include <stdexcept>

class InsufficientFundsError : public std::runtime_error {
public:
    InsufficientFundsError(const char* msg) : std::runtime_error(msg) {}
};

class BankAccount {
private:
    double balance;
public:
    BankAccount(double initial_balance) : balance(initial_balance) {}

    void withdraw(double amount) {
        if (amount > balance) {
            throw InsufficientFundsError("Not enough funds in the account");
        }
        balance -= amount;
        std::cout << "Withdrawn $" << amount << ". New balance: $" << balance << std::endl;
    }
};

int main() {
    BankAccount account(100);
    try {
        account.withdraw(50);  // Output: Withdrawn $50. New balance: $50
        account.withdraw(100); // Throws InsufficientFundsError
    } catch (const InsufficientFundsError& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
```

Slide 7: Python - Exception Hierarchy

Python: Exception Hierarchy

Python has a rich hierarchy of built-in exceptions, all inheriting from BaseException.

Code:

```python
# Demonstrating Python's exception hierarchy
def show_exception_hierarchy():
    try:
        # Intentionally cause different types of errors
        1 / 0  # ZeroDivisionError
    except ArithmeticError as e:
        print(f"Caught ArithmeticError: {type(e).__name__}")
        print(f"Is it a subclass of Exception? {issubclass(type(e), Exception)}")
        print(f"Is it a subclass of BaseException? {issubclass(type(e), BaseException)}")

show_exception_hierarchy()
# Output:
# Caught ArithmeticError: ZeroDivisionError
# Is it a subclass of Exception? True
# Is it a subclass of BaseException? True
```

Slide 8: C++ - Exception Hierarchy

C++: Exception Hierarchy

C++ standard library provides a hierarchy of exception classes, with std::exception as the base.

Code:

```cpp
#include <iostream>
#include <stdexcept>

void show_exception_hierarchy() {
    try {
        // Intentionally cause a runtime error
        throw std::runtime_error("A runtime error occurred");
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
        std::cout << "Is it a std::runtime_error? " 
                  << (dynamic_cast<const std::runtime_error*>(&e) != nullptr) << std::endl;
        std::cout << "Is it a std::exception? " 
                  << (dynamic_cast<const std::exception*>(&e) != nullptr) << std::endl;
    }
}

int main() {
    show_exception_hierarchy();
    return 0;
}
// Output:
// Caught exception: A runtime error occurred
// Is it a std::runtime_error? 1
// Is it a std::exception? 1
```

Slide 9: Python - Debugging with pdb

Python: Debugging with pdb

Python's built-in debugger, pdb, allows interactive debugging of code.

Code:

```python
import pdb

def complex_calculation(x, y):
    result = x * y
    pdb.set_trace()  # Debugger will pause here
    if result > 50:
        return "Large result"
    else:
        return "Small result"

print(complex_calculation(5, 12))

# When the debugger pauses, you can:
# - Use 'n' to step to the next line
# - Use 'p result' to print the value of result
# - Use 'c' to continue execution
# - Use 'q' to quit the debugger
```

Slide 10: C++ - Debugging with GDB

C++: Debugging with GDB

GDB (GNU Debugger) is a powerful tool for debugging C++ programs.

Code:

```cpp
#include <iostream>

int complex_calculation(int x, int y) {
    int result = x * y;
    // Set a breakpoint here in GDB
    if (result > 50) {
        return 1;  // Large result
    } else {
        return 0;  // Small result
    }
}

int main() {
    std::cout << complex_calculation(5, 12) << std::endl;
    return 0;
}

// Compile with debug symbols: g++ -g program.cpp -o program
// In GDB:
// - Use 'break complex_calculation' to set a breakpoint
// - Use 'run' to start the program
// - Use 'next' to step to the next line
// - Use 'print result' to display the value of result
// - Use 'continue' to resume execution
// - Use 'quit' to exit GDB
```

Slide 11: Python - Context Managers for Exception Handling

Python: Context Managers for Exception Handling

Python's 'with' statement provides a clean way to handle exceptions and resource management.

Code:

```python
class DatabaseConnection:
    def __enter__(self):
        print("Opening database connection")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Closing database connection")
        if exc_type is not None:
            print(f"An error occurred: {exc_value}")
        return False  # Propagate the exception

    def query(self, sql):
        if "DROP" in sql.upper():
            raise ValueError("DROP statements are not allowed")
        print(f"Executing query: {sql}")

# Using the context manager
with DatabaseConnection() as db:
    db.query("SELECT * FROM users")
    db.query("DROP TABLE users")  # This will raise an exception

# Output:
# Opening database connection
# Executing query: SELECT * FROM users
# An error occurred: DROP statements are not allowed
# Closing database connection
# ValueError: DROP statements are not allowed
```

Slide 12: C++ - RAII for Exception Safety

C++: RAII for Exception Safety

C++ uses RAII (Resource Acquisition Is Initialization) for exception-safe resource management.

Code:

```cpp
#include <iostream>
#include <stdexcept>
#include <memory>

class DatabaseConnection {
public:
    DatabaseConnection() {
        std::cout << "Opening database connection" << std::endl;
    }
    ~DatabaseConnection() {
        std::cout << "Closing database connection" << std::endl;
    }
    void query(const std::string& sql) {
        if (sql.find("DROP") != std::string::npos) {
            throw std::runtime_error("DROP statements are not allowed");
        }
        std::cout << "Executing query: " << sql << std::endl;
    }
};

void perform_database_operations() {
    std::unique_ptr<DatabaseConnection> db(new DatabaseConnection());
    db->query("SELECT * FROM users");
    db->query("DROP TABLE users");  // This will throw an exception
}

int main() {
    try {
        perform_database_operations();
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
    }
    return 0;
}

// Output:
// Opening database connection
// Executing query: SELECT * FROM users
// Closing database connection
// An error occurred: DROP statements are not allowed
```

Slide 13: Python vs C++ Exception Handling and Debugging Comparison

Python vs C++ Exception Handling and Debugging Comparison

A table comparing key aspects of exception handling and debugging in Python and C++.

Code:

```
| Feature                | Python                              | C++                                 |
|------------------------|-------------------------------------|-------------------------------------|
| Exception Syntax       | try-except-else-finally             | try-catch-throw                     |
| Built-in Exceptions    | Rich hierarchy (e.g., ValueError)   | std::exception hierarchy            |
| Custom Exceptions      | class CustomError(Exception):       | class CustomError : public std::exception |
| Multiple Exceptions    | except (TypeError, ValueError):     | catch (const std::exception&)       |
| Resource Management    | with statement (Context Managers)   | RAII (destructors)                  |
| Debugger               | pdb (Python Debugger)               | gdb (GNU Debugger)                  |
| Performance Overhead   | Higher (due to dynamic nature)      | Lower (static typing, optimizations)|
| Exception Specification| Not required                        | Optional (noexcept specifier)       |
| Stack Unwinding        | Automatic                           | Automatic (may be more complex)     |
| Standard Library       | Extensive built-in modules          | Comprehensive STL                   |
```

