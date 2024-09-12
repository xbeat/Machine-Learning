## Comparing Python and C++ Control Structures
Slide 1: Python - If Statement

Conditional execution based on a boolean expression.

```python
# Weather-based activity recommendation
temperature = 25  # in Celsius

if temperature > 30:
    print("It's hot! Go for a swim.")
elif temperature > 20:
    print("Nice weather! How about a picnic?")
else:
    print("It's a bit cool. Maybe stay indoors.")

# Output: Nice weather! How about a picnic?
```

Slide 2: C++ - If Statement

Conditional execution in C++ using if-else statements.

```cpp
#include <iostream>
using namespace std;

int main() {
    int temperature = 25;  // in Celsius

    if (temperature > 30) {
        cout << "It's hot! Go for a swim.";
    } else if (temperature > 20) {
        cout << "Nice weather! How about a picnic?";
    } else {
        cout << "It's a bit cool. Maybe stay indoors.";
    }

    // Output: Nice weather! How about a picnic?
    return 0;
}
```

Slide 3: Python - For Loop

Iterate over a sequence (list, tuple, string) or range of numbers.

```python
# Calculating total calories for a meal
foods = ["rice", "chicken", "vegetables", "dessert"]
calories = [200, 300, 100, 150]

total_calories = 0
for i in range(len(foods)):
    total_calories += calories[i]
    print(f"{foods[i].capitalize()}: {calories[i]} calories")

print(f"Total calories: {total_calories}")

# Output:
# Rice: 200 calories
# Chicken: 300 calories
# Vegetables: 100 calories
# Dessert: 150 calories
# Total calories: 750
```

Slide 4: C++ - For Loop

Traditional for loop in C++ for iterating over a range or array.

```cpp
#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main() {
    vector<string> foods = {"rice", "chicken", "vegetables", "dessert"};
    vector<int> calories = {200, 300, 100, 150};

    int total_calories = 0;
    for (int i = 0; i < foods.size(); i++) {
        total_calories += calories[i];
        cout << foods[i] << ": " << calories[i] << " calories" << endl;
    }

    cout << "Total calories: " << total_calories << endl;

    // Output:
    // rice: 200 calories
    // chicken: 300 calories
    // vegetables: 100 calories
    // dessert: 150 calories
    // Total calories: 750
    return 0;
}
```

Slide 5: Python - While Loop

Execute a block of code repeatedly while a condition is true.

```python
# Simulating battery discharge
battery_level = 100
hours = 0

while battery_level > 0:
    print(f"Hour {hours}: Battery at {battery_level}%")
    battery_level -= 10
    hours += 1

print("Battery depleted!")

# Output:
# Hour 0: Battery at 100%
# Hour 1: Battery at 90%
# ...
# Hour 9: Battery at 10%
# Hour 10: Battery at 0%
# Battery depleted!
```

Slide 6: C++ - While Loop

Execute a block of code repeatedly while a condition is true in C++.

```cpp
#include <iostream>
using namespace std;

int main() {
    int battery_level = 100;
    int hours = 0;

    while (battery_level > 0) {
        cout << "Hour " << hours << ": Battery at " << battery_level << "%" << endl;
        battery_level -= 10;
        hours++;
    }

    cout << "Battery depleted!" << endl;

    // Output:
    // Hour 0: Battery at 100%
    // Hour 1: Battery at 90%
    // ...
    // Hour 9: Battery at 10%
    // Hour 10: Battery at 0%
    // Battery depleted!
    return 0;
}
```

Slide 7: Python - Break Statement

Exit a loop prematurely when a certain condition is met.

```python
# Finding the first prime number after a given number
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

start = 1000
current = start + 1

while True:
    if is_prime(current):
        print(f"The first prime number after {start} is {current}")
        break
    current += 1

# Output: The first prime number after 1000 is 1009
```

Slide 8: C++ - Break Statement

Exit a loop prematurely in C++ when a certain condition is met.

```cpp
#include <iostream>
#include <cmath>
using namespace std;

bool is_prime(int n) {
    if (n < 2) return false;
    for (int i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) return false;
    }
    return true;
}

int main() {
    int start = 1000;
    int current = start + 1;

    while (true) {
        if (is_prime(current)) {
            cout << "The first prime number after " << start << " is " << current << endl;
            break;
        }
        current++;
    }

    // Output: The first prime number after 1000 is 1009
    return 0;
}
```

Slide 9: Python - Continue Statement

Skip the rest of the current iteration and continue with the next one.

```python
# Printing even numbers and replacing odd numbers with "Odd"
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for num in numbers:
    if num % 2 != 0:
        continue
    print(num, end=" ")

# Output: 2 4 6 8 10
```

Slide 10: C++ - Continue Statement

Skip the rest of the current iteration and continue with the next one in C++.

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    for (int num : numbers) {
        if (num % 2 != 0) {
            continue;
        }
        cout << num << " ";
    }

    // Output: 2 4 6 8 10
    return 0;
}
```

Slide 11: Python - List Comprehension

Create new lists based on existing lists in a concise way.

```python
# Converting temperatures from Celsius to Fahrenheit
celsius = [0, 10, 20, 30, 40]
fahrenheit = [((9/5) * temp + 32) for temp in celsius]

print("Celsius:   ", celsius)
print("Fahrenheit:", fahrenheit)

# Output:
# Celsius:    [0, 10, 20, 30, 40]
# Fahrenheit: [32.0, 50.0, 68.0, 86.0, 104.0]
```

Slide 12: C++ - Range-based For Loop

Iterate over elements in a container or array without using indices.

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<string> fruits = {"apple", "banana", "cherry", "date"};

    for (const auto& fruit : fruits) {
        cout << "I like " << fruit << "!" << endl;
    }

    // Output:
    // I like apple!
    // I like banana!
    // I like cherry!
    // I like date!
    return 0;
}
```

Wrap-up: Python vs C++ Control Structures and Loops

```
| Feature           | Python                                   | C++                                      |
|-------------------|------------------------------------------|------------------------------------------|
| Syntax            | More readable, less verbose              | More verbose, requires semicolons        |
| Indentation       | Meaningful, defines code blocks          | Uses braces {} to define code blocks     |
| For loops         | for item in iterable:                    | for (int i = 0; i < n; i++) {}           |
| While loops       | while condition:                         | while (condition) {}                     |
| If statements     | if condition:                            | if (condition) {}                        |
| List comprehension| [expression for item in iterable]        | Not available (use std::transform)       |
| Range-based for   | for item in iterable:                    | for (auto& item : container) {}          |
| Switch statement  | Not available (use if-elif-else)         | switch (variable) { case value: ... }    |
| Break/Continue    | break and continue keywords              | break and continue keywords              |
| Error handling    | try-except blocks                        | try-catch blocks                         |
```

