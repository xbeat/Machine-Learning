## Comparing Python and C++ String Manipulation
Slide 1: Python - String Creation and Basic Operations

Python String Creation and Concatenation

In Python, strings are created easily and can be concatenated using the + operator.

Code:

```python
# Creating and concatenating strings
first_name = "John"
last_name = "Doe"
full_name = first_name + " " + last_name
print(full_name)  # Output: John Doe

# String repetition
cheer = "Hip " * 2 + "Hooray!"
print(cheer)  # Output: Hip Hip Hooray!
```

Slide 2: C++ - String Creation and Basic Operations

C++ String Creation and Concatenation

C++ strings require the string header and use the + operator for concatenation.

Code:

```cpp
#include <iostream>
#include <string>

int main() {
    // Creating and concatenating strings
    std::string first_name = "John";
    std::string last_name = "Doe";
    std::string full_name = first_name + " " + last_name;
    std::cout << full_name << std::endl;  // Output: John Doe

    // String repetition (no built-in operator)
    std::string cheer = "Hip Hip Hooray!";
    std::cout << cheer << std::endl;  // Output: Hip Hip Hooray!

    return 0;
}
```

Slide 3: Python - String Slicing

Python String Slicing

Python offers powerful string slicing capabilities for extracting substrings.

Code:

```python
# String slicing examples
text = "Python Programming"
print(text[0:6])    # Output: Python
print(text[7:])     # Output: Programming
print(text[::-1])   # Output: gnimmargorP nohtyP (reversed)

# Extracting every other character
print(text[::2])    # Output: Pto rgamn
```

Slide 4: C++ - String Substring Extraction

C++ Substring Extraction

C++ uses the substr() method for extracting substrings from strings.

Code:

```cpp
#include <iostream>
#include <string>

int main() {
    std::string text = "C++ Programming";
    std::cout << text.substr(0, 3) << std::endl;  // Output: C++
    std::cout << text.substr(4) << std::endl;     // Output: Programming

    // Reversing a string (no built-in method)
    std::string reversed(text.rbegin(), text.rend());
    std::cout << reversed << std::endl;  // Output: gnimmargorP ++C

    return 0;
}
```

Slide 5: Python - String Methods

Python String Methods

Python provides numerous built-in methods for string manipulation.

Code:

```python
# String method examples
text = "  Python is Awesome!  "
print(text.strip())         # Output: Python is Awesome!
print(text.lower())         # Output:   python is awesome!  
print(text.upper())         # Output:   PYTHON IS AWESOME!  
print(text.replace("Awesome", "Amazing"))  # Output:   Python is Amazing!  
print(text.split())         # Output: ['Python', 'is', 'Awesome!']
```

Slide 6: C++ - String Methods

C++ String Methods

C++ offers similar string manipulation methods through the string class.

Code:

```cpp
#include <iostream>
#include <string>
#include <algorithm>

int main() {
    std::string text = "  C++ is Powerful!  ";
    
    // Trimming whitespace (C++20)
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    text.erase(std::find_if(text.rbegin(), text.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), text.end());
    std::cout << text << std::endl;  // Output: C++ is Powerful!

    // Other methods
    std::transform(text.begin(), text.end(), text.begin(), ::tolower);
    std::cout << text << std::endl;  // Output: c++ is powerful!

    std::transform(text.begin(), text.end(), text.begin(), ::toupper);
    std::cout << text << std::endl;  // Output: C++ IS POWERFUL!

    size_t pos = text.find("POWERFUL");
    if (pos != std::string::npos) {
        text.replace(pos, 8, "AMAZING");
    }
    std::cout << text << std::endl;  // Output: C++ IS AMAZING!

    return 0;
}
```

Slide 7: Python - String Formatting

Python String Formatting

Python offers multiple ways to format strings, including f-strings for easy interpolation.

Code:

```python
name = "Alice"
age = 30
height = 1.75

# Old-style formatting
print("Name: %s, Age: %d, Height: %.2f" % (name, age, height))

# str.format() method
print("Name: {}, Age: {}, Height: {:.2f}".format(name, age, height))

# f-strings (Python 3.6+)
print(f"Name: {name}, Age: {age}, Height: {height:.2f}")

# Output for all: Name: Alice, Age: 30, Height: 1.75
```

Slide 8: C++ - String Formatting

C++ String Formatting

C++ uses iostream manipulators or the printf-style formatting for string interpolation.

Code:

```cpp
#include <iostream>
#include <iomanip>
#include <string>

int main() {
    std::string name = "Alice";
    int age = 30;
    double height = 1.75;

    // Using iostream manipulators
    std::cout << "Name: " << name << ", Age: " << age 
              << ", Height: " << std::fixed << std::setprecision(2) << height << std::endl;

    // Using printf-style formatting
    printf("Name: %s, Age: %d, Height: %.2f\n", name.c_str(), age, height);

    // Output for both: Name: Alice, Age: 30, Height: 1.75

    return 0;
}
```

Slide 9: Python - String Searching and Manipulation

Python String Searching and Manipulation

Python provides intuitive methods for searching and manipulating strings.

Code:

```python
text = "The quick brown fox jumps over the lazy dog"

# Searching
print(text.find("fox"))        # Output: 16
print(text.count("the"))       # Output: 2 (case-sensitive)

# Manipulation
words = text.split()
print(words)                   # Output: ['The', 'quick', 'brown', ...]
print(" ".join(reversed(words)))  # Output: dog lazy the over jumps fox brown quick The

# Case manipulation
print(text.title())            # Output: The Quick Brown Fox Jumps Over The Lazy Dog
```

Slide 10: C++ - String Searching and Manipulation

C++ String Searching and Manipulation

C++ offers methods for string searching and manipulation, often requiring additional algorithms.

Code:

```cpp
#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>
#include <vector>

int main() {
    std::string text = "The quick brown fox jumps over the lazy dog";

    // Searching
    std::cout << text.find("fox") << std::endl;  // Output: 16
    std::cout << std::count(text.begin(), text.end(), 'e') << std::endl;  // Output: 3

    // Manipulation
    std::istringstream iss(text);
    std::vector<std::string> words;
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }

    std::reverse(words.begin(), words.end());
    for (const auto& w : words) {
        std::cout << w << " ";
    }
    std::cout << std::endl;  // Output: dog lazy the over jumps fox brown quick The

    // Case manipulation (no built-in title case)
    std::transform(text.begin(), text.end(), text.begin(), ::tolower);
    if (!text.empty()) {
        text[0] = std::toupper(text[0]);
    }
    std::cout << text << std::endl;  // Output: The quick brown fox jumps over the lazy dog

    return 0;
}
```

Slide 11: Python - Regular Expressions

Python Regular Expressions

Python's re module provides powerful string pattern matching and manipulation capabilities.

Code:

```python
import re

text = "The email is john@example.com and the phone is 123-456-7890."

# Finding all email addresses
emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
print(emails)  # Output: ['john@example.com']

# Replacing phone numbers with a masked version
masked_text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', 'XXX-XXX-XXXX', text)
print(masked_text)  # Output: The email is john@example.com and the phone is XXX-XXX-XXXX.

# Splitting text by multiple delimiters
words = re.split(r'[ ,.]', "Hello, world. How are you?")
print(words)  # Output: ['Hello', '', 'world', '', 'How', 'are', 'you', '']
```

Slide 12: C++ - Regular Expressions

C++ Regular Expressions

C++11 introduced the <regex> library for pattern matching and string manipulation using regular expressions.

Code:

```cpp
#include <iostream>
#include <string>
#include <regex>

int main() {
    std::string text = "The email is john@example.com and the phone is 123-456-7890.";

    // Finding all email addresses
    std::regex email_regex(R"(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)");
    std::sregex_iterator it(text.begin(), text.end(), email_regex);
    std::sregex_iterator end;
    while (it != end) {
        std::cout << it->str() << std::endl;  // Output: john@example.com
        ++it;
    }

    // Replacing phone numbers with a masked version
    std::regex phone_regex(R"(\b\d{3}-\d{3}-\d{4}\b)");
    std::string masked_text = std::regex_replace(text, phone_regex, "XXX-XXX-XXXX");
    std::cout << masked_text << std::endl;  // Output: The email is john@example.com and the phone is XXX-XXX-XXXX.

    // Splitting text by multiple delimiters
    std::string split_text = "Hello, world. How are you?";
    std::regex split_regex("[ ,.]");
    std::sregex_token_iterator split_it(split_text.begin(), split_text.end(), split_regex, -1);
    std::sregex_token_iterator split_end;
    while (split_it != split_end) {
        std::cout << *split_it << std::endl;
    }
    // Output:
    // Hello
    // 
    // world
    // 
    // How
    // are
    // you
    // 

    return 0;
}
```

Slide 13: Python vs C++ String Manipulation Comparison

Python vs C++ String Manipulation Comparison

A summary of key differences in string manipulation between Python and C++.

Code:

```
| Feature                | Python                            | C++                               |
|------------------------|-----------------------------------|-----------------------------------|
| String Creation        | str = "Hello"                     | std::string str = "Hello";        |
| Concatenation          | result = str1 + str2              | result = str1 + str2;             |
| Substring              | sub = string[start:end]           | sub = string.substr(start, length)|
| Length                 | length = len(string)              | length = string.length();         |
| Case Conversion        | upper = string.upper()            | std::transform(begin, end, upper) |
| Searching              | index = string.find("substring")  | index = string.find("substring"); |
| Splitting              | parts = string.split()            | Use std::istringstream            |
| Joining                | result = " ".join(list_of_strings)| Use std::ostringstream            |
| Formatting             | f"Name: {name}, Age: {age}"       | sprintf or std::stringstream      |
| Regular Expressions    | import re                         | #include <regex>                  |
| Memory Management      | Automatic                         | Manual (C-style) or RAII (C++)    |
```

