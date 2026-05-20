## Comparing Python and C++ File I/O
Slide 1: Python - Opening a File

Python - Opening a File

Python uses the built-in open() function to create a file object.

Code:

```python
# Opening a file for reading
with open('example.txt', 'r') as file:
    content = file.read()
    print(f"File contents: {content}")

# Opening a file for writing
with open('new_file.txt', 'w') as file:
    file.write("Hello, world!")
```

Slide 2: C++ - Opening a File

C++ - Opening a File

C++ uses the fstream library for file operations.

Code:

```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    // Opening a file for reading
    ifstream inFile("example.txt");
    string content;
    getline(inFile, content);
    cout << "File contents: " << content << endl;
    inFile.close();

    // Opening a file for writing
    ofstream outFile("new_file.txt");
    outFile << "Hello, world!";
    outFile.close();

    return 0;
}
```

Slide 3: Python - Reading from a File

Python - Reading from a File

Python offers multiple methods to read file contents.

Code:

```python
# Reading entire file
with open('recipe.txt', 'r') as file:
    content = file.read()
    print(f"Full recipe: {content}")

# Reading line by line
with open('ingredients.txt', 'r') as file:
    for line in file:
        print(f"Ingredient: {line.strip()}")

# Reading specific number of characters
with open('instructions.txt', 'r') as file:
    first_50_chars = file.read(50)
    print(f"First 50 characters: {first_50_chars}")
```

Slide 4: C++ - Reading from a File

C++ - Reading from a File

C++ provides various methods for reading file contents.

Code:

```cpp
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main() {
    ifstream file("recipe.txt");
    string line, content;

    // Reading entire file
    while (getline(file, line)) {
        content += line + "\n";
    }
    cout << "Full recipe: " << content << endl;

    // Reading line by line
    file.clear();
    file.seekg(0);
    while (getline(file, line)) {
        cout << "Ingredient: " << line << endl;
    }

    // Reading specific number of characters
    file.clear();
    file.seekg(0);
    char buffer[51];
    file.read(buffer, 50);
    buffer[50] = '\0';
    cout << "First 50 characters: " << buffer << endl;

    file.close();
    return 0;
}
```

Slide 5: Python - Writing to a File

Python - Writing to a File

Python provides simple methods for writing to files.

Code:

```python
# Writing a single string
with open('log.txt', 'w') as file:
    file.write("Application started at 09:00 AM")

# Writing multiple lines
daily_sales = [120, 145, 190, 170, 210]
with open('sales_report.txt', 'w') as file:
    for day, sale in enumerate(daily_sales, 1):
        file.write(f"Day {day}: ${sale}\n")

# Appending to a file
with open('events.txt', 'a') as file:
    file.write("New event: Product launch\n")
```

Slide 6: C++ - Writing to a File

C++ - Writing to a File

C++ uses stream operators for writing to files.

Code:

```cpp
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

int main() {
    // Writing a single string
    ofstream logFile("log.txt");
    logFile << "Application started at 09:00 AM";
    logFile.close();

    // Writing multiple lines
    vector<int> dailySales = {120, 145, 190, 170, 210};
    ofstream salesReport("sales_report.txt");
    for (int i = 0; i < dailySales.size(); ++i) {
        salesReport << "Day " << i+1 << ": $" << dailySales[i] << endl;
    }
    salesReport.close();

    // Appending to a file
    ofstream eventsFile("events.txt", ios::app);
    eventsFile << "New event: Product launch" << endl;
    eventsFile.close();

    return 0;
}
```

Slide 7: Python - File Modes

Python - File Modes

Python uses mode arguments to specify file operations.

Code:

```python
# Read mode ('r')
with open('data.txt', 'r') as file:
    content = file.read()

# Write mode ('w') - Overwrites existing content
with open('output.txt', 'w') as file:
    file.write("New content")

# Append mode ('a') - Adds to existing content
with open('log.txt', 'a') as file:
    file.write("New log entry\n")

# Read and write mode ('r+')
with open('config.txt', 'r+') as file:
    content = file.read()
    file.write("Updated config")

# Binary mode ('b') - For non-text files
with open('image.jpg', 'rb') as file:
    image_data = file.read()
```

Slide 8: C++ - File Modes

C++ - File Modes

C++ uses flags to specify file operation modes.

Code:

```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    // Read mode (ios::in)
    ifstream inFile("data.txt", ios::in);
    string content;
    getline(inFile, content);
    inFile.close();

    // Write mode (ios::out) - Overwrites existing content
    ofstream outFile("output.txt", ios::out);
    outFile << "New content";
    outFile.close();

    // Append mode (ios::app)
    ofstream appendFile("log.txt", ios::app);
    appendFile << "New log entry\n";
    appendFile.close();

    // Read and write mode (ios::in | ios::out)
    fstream rwFile("config.txt", ios::in | ios::out);
    rwFile << "Updated config";
    rwFile.close();

    // Binary mode (ios::binary) - For non-text files
    ifstream binFile("image.jpg", ios::binary);
    char buffer[1024];
    binFile.read(buffer, 1024);
    binFile.close();

    return 0;
}
```

Slide 9: Python - Error Handling

Python - Error Handling in File Operations

Python uses try-except blocks for file operation error handling.

Code:

```python
try:
    with open('nonexistent_file.txt', 'r') as file:
        content = file.read()
except FileNotFoundError:
    print("Error: File not found")
except PermissionError:
    print("Error: Permission denied")
except IOError as e:
    print(f"I/O error occurred: {e}")
else:
    print(f"File contents: {content}")
finally:
    print("File operation attempt completed")
```

Slide 10: C++ - Error Handling

C++ - Error Handling in File Operations

C++ uses exceptions and error state flags for file operation error handling.

Code:

```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    ifstream file;
    try {
        file.exceptions(ifstream::failbit | ifstream::badbit);
        file.open("nonexistent_file.txt");
        string content;
        getline(file, content);
        cout << "File contents: " << content << endl;
    }
    catch (const ifstream::failure& e) {
        if (!file.is_open()) {
            cerr << "Error: File not found" << endl;
        } else if (file.bad()) {
            cerr << "Error: I/O error occurred" << endl;
        } else if (file.eof()) {
            cerr << "Error: End of file reached" << endl;
        } else {
            cerr << "Error: " << e.what() << endl;
        }
    }
    if (file.is_open()) {
        file.close();
    }
    cout << "File operation attempt completed" << endl;
    return 0;
}
```

Slide 11: Python - Working with CSV Files

Python - Working with CSV Files

Python's csv module simplifies CSV file operations.

Code:

```python
import csv

# Writing to a CSV file
data = [
    ['Name', 'Age', 'City'],
    ['Alice', 28, 'New York'],
    ['Bob', 35, 'San Francisco'],
    ['Charlie', 42, 'Chicago']
]

with open('users.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

# Reading from a CSV file
with open('users.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(f"Name: {row[0]}, Age: {row[1]}, City: {row[2]}")
```

Slide 12: C++ - Working with CSV Files

C++ - Working with CSV Files

C++ requires manual parsing for CSV file operations.

Code:

```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;

int main() {
    // Writing to a CSV file
    vector<vector<string>> data = {
        {"Name", "Age", "City"},
        {"Alice", "28", "New York"},
        {"Bob", "35", "San Francisco"},
        {"Charlie", "42", "Chicago"}
    };

    ofstream outFile("users.csv");
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            outFile << row[i];
            if (i < row.size() - 1) outFile << ",";
        }
        outFile << endl;
    }
    outFile.close();

    // Reading from a CSV file
    ifstream inFile("users.csv");
    string line, field;
    while (getline(inFile, line)) {
        stringstream ss(line);
        vector<string> row;
        while (getline(ss, field, ',')) {
            row.push_back(field);
        }
        cout << "Name: " << row[0] << ", Age: " << row[1] << ", City: " << row[2] << endl;
    }
    inFile.close();

    return 0;
}
```

Slide 13: Python - File Position and Seeking

Python - File Position and Seeking

Python provides methods to manipulate file position.

Code:

```python
with open('sample.txt', 'r+') as file:
    # Read first 10 characters
    print(file.read(10))

    # Get current position
    print(f"Current position: {file.tell()}")

    # Move to start of file
    file.seek(0)

    # Move 5 characters from start
    file.seek(5)

    # Move 3 characters from current position
    file.seek(3, 1)

    # Move 5 characters from end
    file.seek(-5, 2)

    # Write at current position
    file.write("NEW")
```

Slide 14: C++ - File Position and Seeking

C++ - File Position and Seeking

C++ offers methods to control file pointer position.

Code:

```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    fstream file("sample.txt", ios::in | ios::out);

    // Read first 10 characters
    char buffer[11];
    file.read(buffer, 10);
    buffer[10] = '\0';
    cout << buffer << endl;

    // Get current position
    cout << "Current position: " << file.tellg() << endl;

    // Move to start of file
    file.seekg(0, ios::beg);

    // Move 5 characters from start
    file.seekg(5, ios::beg);

    // Move 3 characters from current position
    file.seekg(3, ios::cur);

    // Move 5 characters from end
    file.seekg(-5, ios::end);

    // Write at current position
    file << "NEW";

    file.close();
    return 0;
}
```

Slide 15: Wrap-up - Python vs C++ File I/O Comparison

Python vs C++ File I/O Comparison

Key differences in file I/O operations between Python and C++.

Code:

```
| Feature           | Python                             | C++                                |
|-------------------|------------------------------------|------------------------------------|
| File Opening      | open('file.txt', 'r')              | ifstream file("file.txt")          |
| Reading           | content = file.read()              | getline(file, content)             |
| Writing           | file.write("content")              | file << "content"                  |
| Closing           | Automatic with 'with' statement    | Explicit file.close()              |
| Error Handling    | try-except blocks                  | try-catch blocks & error flags     |
| CSV Handling      | Built-in csv module                | Manual parsing required            |
| File Modes        | 'r', 'w', 'a', 'r+', 'b'           | ios::in, ios::out, ios::app, etc.  |
| Seeking           | file.seek(offset, [whence])        | file.seekg(offset, ios::direction) |
| Ease of Use       | Generally simpler syntax           | More verbose, but flexible         |
| Performance       | Good for most tasks                | Potentially faster for large files |
```

