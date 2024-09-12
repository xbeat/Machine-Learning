## Comparing Python and C++ Data Structures:
Slide 1: Python Lists

Python Lists - Dynamic Arrays

Lists in Python are dynamic arrays that can store multiple items of different types.

Code:

```python
# Shopping list
groceries = ["apples", "milk", "bread"]
groceries.append("cheese")
print(f"Items to buy: {groceries}")  # Output: Items to buy: ['apples', 'milk', 'bread', 'cheese']

# Accessing elements
print(f"First item: {groceries[0]}")  # Output: First item: apples

# List comprehension
prices = [2.5, 3.0, 1.5, 4.0]
tax_rate = 0.08
total_cost = sum([price * (1 + tax_rate) for price in prices])
print(f"Total cost with tax: ${total_cost:.2f}")  # Output: Total cost with tax: $11.88
```

Slide 2: C++ Arrays

C++ Arrays - Fixed-Size Collections

Arrays in C++ are fixed-size collections of elements of the same type.

Code:

```cpp
#include <iostream>
using namespace std;

int main() {
    // Fixed-size array of grades
    int grades[5] = {85, 92, 78, 90, 88};
    
    // Calculating average grade
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        sum += grades[i];
    }
    double average = static_cast<double>(sum) / 5;
    
    cout << "Average grade: " << average << endl;  // Output: Average grade: 86.6
    
    // Accessing elements
    cout << "Highest grade: " << grades[1] << endl;  // Output: Highest grade: 92
    
    return 0;
}
```

Slide 3: Python Dictionaries

Python Dictionaries - Key-Value Pairs

Dictionaries in Python store key-value pairs for efficient data retrieval.

Code:

```python
# Student information
student = {
    "name": "Emma",
    "age": 20,
    "major": "Computer Science",
    "gpa": 3.8
}

# Accessing and modifying data
print(f"Student: {student['name']}, Major: {student['major']}")
student['gpa'] = 3.9

# Adding new key-value pair
student['graduation_year'] = 2025

# Iterating through dictionary
for key, value in student.items():
    print(f"{key}: {value}")

# Output:
# Student: Emma, Major: Computer Science
# name: Emma
# age: 20
# major: Computer Science
# gpa: 3.9
# graduation_year: 2025
```

Slide 4: C++ Maps

C++ Maps - Ordered Key-Value Pairs

Maps in C++ store ordered key-value pairs, allowing efficient data retrieval.

Code:

```cpp
#include <iostream>
#include <map>
#include <string>
using namespace std;

int main() {
    // Car inventory
    map<string, int> carInventory;
    
    // Adding items to the map
    carInventory["Sedan"] = 15;
    carInventory["SUV"] = 10;
    carInventory["Hatchback"] = 8;
    
    // Accessing and modifying data
    cout << "Sedans in stock: " << carInventory["Sedan"] << endl;
    carInventory["SUV"] += 5;
    
    // Iterating through the map
    for (const auto& pair : carInventory) {
        cout << pair.first << ": " << pair.second << endl;
    }
    
    // Output:
    // Sedans in stock: 15
    // Hatchback: 8
    // Sedan: 15
    // SUV: 15
    
    return 0;
}
```

Slide 5: Python Sets

Python Sets - Unique Unordered Elements

Sets in Python store unique, unordered elements, ideal for removing duplicates and membership testing.

Code:

```python
# Unique website visitors
visitors = {"Alice", "Bob", "Charlie", "David", "Alice"}

print(f"Unique visitors: {visitors}")  # Output: Unique visitors: {'Bob', 'Alice', 'David', 'Charlie'}

# Adding and removing elements
visitors.add("Eve")
visitors.remove("Bob")

# Set operations
employees = {"Alice", "Charlie", "Frank"}
print(f"Visitors who are employees: {visitors & employees}")  # Output: {'Alice', 'Charlie'}

# Membership testing
print("Is David a visitor?", "David" in visitors)  # Output: Is David a visitor? True
```

Slide 6: C++ Sets

C++ Sets - Ordered Unique Elements

Sets in C++ store unique elements in a specific order, providing fast insertion and lookup.

Code:

```cpp
#include <iostream>
#include <set>
#include <string>
using namespace std;

int main() {
    // Unique book titles in a library
    set<string> bookTitles;
    
    // Adding elements
    bookTitles.insert("1984");
    bookTitles.insert("To Kill a Mockingbird");
    bookTitles.insert("Pride and Prejudice");
    bookTitles.insert("1984");  // Duplicate, won't be added
    
    // Iterating through the set
    cout << "Books in the library:" << endl;
    for (const auto& title : bookTitles) {
        cout << "- " << title << endl;
    }
    
    // Checking if an element exists
    string searchTitle = "1984";
    if (bookTitles.find(searchTitle) != bookTitles.end()) {
        cout << searchTitle << " is available." << endl;
    }
    
    // Output:
    // Books in the library:
    // - 1984
    // - Pride and Prejudice
    // - To Kill a Mockingbird
    // 1984 is available.
    
    return 0;
}
```

Slide 7: Python Tuples

Python Tuples - Immutable Sequences

Tuples in Python are immutable sequences, often used for fixed collections of data.

Code:

```python
# Geographic coordinates (latitude, longitude)
location = (40.7128, -74.0060)

print(f"New York City coordinates: {location}")

# Unpacking tuple values
latitude, longitude = location
print(f"Latitude: {latitude}, Longitude: {longitude}")

# Tuples in a list (multiple locations)
cities = [
    ("New York", 40.7128, -74.0060),
    ("Los Angeles", 34.0522, -118.2437),
    ("Chicago", 41.8781, -87.6298)
]

# Accessing tuple elements in a list
for city, lat, lon in cities:
    print(f"{city}: {lat:.4f}, {lon:.4f}")

# Output:
# New York City coordinates: (40.7128, -74.006)
# Latitude: 40.7128, Longitude: -74.006
# New York: 40.7128, -74.0060
# Los Angeles: 34.0522, -118.2437
# Chicago: 41.8781, -87.6298
```

Slide 8: C++ Tuples

C++ Tuples - Fixed-Size Heterogeneous Collections

Tuples in C++ are fixed-size collections that can hold elements of different types.

Code:

```cpp
#include <iostream>
#include <tuple>
#include <string>
using namespace std;

int main() {
    // Student record: (name, age, GPA)
    tuple<string, int, double> student("Alice", 20, 3.8);
    
    // Accessing tuple elements
    cout << "Student: " << get<0>(student) << endl;
    cout << "Age: " << get<1>(student) << endl;
    cout << "GPA: " << get<2>(student) << endl;
    
    // Modifying tuple elements
    get<2>(student) = 3.9;
    
    // Unpacking tuple values
    string name;
    int age;
    double gpa;
    tie(name, age, gpa) = student;
    
    cout << "Updated GPA: " << gpa << endl;
    
    // Output:
    // Student: Alice
    // Age: 20
    // GPA: 3.8
    // Updated GPA: 3.9
    
    return 0;
}
```

Slide 9: Python Linked Lists

Python Linked Lists - Custom Implementation

Linked lists in Python are typically implemented as custom classes, as there's no built-in linked list data structure.

Code:

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return elements

# Using the linked list
todo_list = LinkedList()
todo_list.append("Buy groceries")
todo_list.append("Clean house")
todo_list.append("Pay bills")

print("To-Do List:", todo_list.display())
# Output: To-Do List: ['Buy groceries', 'Clean house', 'Pay bills']
```

Slide 10: C++ Linked Lists

C++ Linked Lists - Custom Implementation

Linked lists in C++ are typically implemented as custom classes, offering dynamic memory allocation.

Code:

```cpp
#include <iostream>
#include <string>
using namespace std;

class Node {
public:
    string data;
    Node* next;
    Node(string d) : data(d), next(nullptr) {}
};

class LinkedList {
private:
    Node* head;

public:
    LinkedList() : head(nullptr) {}

    void append(string data) {
        Node* new_node = new Node(data);
        if (!head) {
            head = new_node;
            return;
        }
        Node* current = head;
        while (current->next) {
            current = current->next;
        }
        current->next = new_node;
    }

    void display() {
        Node* current = head;
        while (current) {
            cout << current->data << " -> ";
            current = current->next;
        }
        cout << "nullptr" << endl;
    }
};

int main() {
    LinkedList playlist;
    playlist.append("Song 1");
    playlist.append("Song 2");
    playlist.append("Song 3");

    cout << "Playlist: ";
    playlist.display();
    // Output: Playlist: Song 1 -> Song 2 -> Song 3 -> nullptr

    return 0;
}
```

Slide 11: Python Stacks

Python Stacks - Using Lists

Stacks in Python can be implemented using lists, following the Last-In-First-Out (LIFO) principle.

Code:

```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def peek(self):
        if not self.is_empty():
            return self.items[-1]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

# Using the stack for browser history
browser_history = Stack()
browser_history.push("google.com")
browser_history.push("wikipedia.org")
browser_history.push("stackoverflow.com")

print("Current page:", browser_history.peek())
print("Going back:", browser_history.pop())
print("New current page:", browser_history.peek())

# Output:
# Current page: stackoverflow.com
# Going back: stackoverflow.com
# New current page: wikipedia.org
```

Slide 12: C++ Stacks

C++ Stacks - Using std::stack

C++ provides a built-in stack container adapter in the Standard Template Library (STL).

Code:

```cpp
#include <iostream>
#include <stack>
#include <string>
using namespace std;

int main() {
    stack<string> book_stack;

    // Adding books to the stack
    book_stack.push("The Great Gatsby");
    book_stack.push("To Kill a Mockingbird");
    book_stack.push("1984");

    cout << "Top book: " << book_stack.top() << endl;
    cout << "Stack size: " << book_stack.size() << endl;

    // Removing books from the stack
    book_stack.pop();
    cout << "After removing top book:" << endl;
    cout << "New top book: " << book_stack.top() << endl;

    // Checking if stack is empty
    while (!book_stack.empty()) {
        cout << "Removing: " << book_stack.top() << endl;
        book_stack.pop();
    }

    cout << "Is stack empty? " << (book_stack.empty() ? "Yes" : "No") << endl;

    // Output:
    // Top book: 1984
    // Stack size: 3
    // After removing top book:
    // New top book: To Kill a Mockingbird
    // Removing: To Kill a Mockingbird
    // Removing: The Great Gatsby
    // Is stack empty? Yes

    return 0;
}
```

Slide 13: Python Queues

Python Queues - Using collections.deque

Queues in Python can be efficiently implemented using collections.deque, following the First-In-First-Out (FIFO) principle.

Code:

```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()

    def front(self):
        if not self.is_empty():
            return self.items[0]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

# Using the queue for a print job queue
print_queue = Queue()
print_queue.enqueue("Document1.pdf")
print_queue.enqueue("Image.jpg")
print_queue.enqueue("Report.docx")

print("Next print job:", print_queue.front())
print("Processing:", print_queue.dequeue())
print("New next print job:", print_queue.front())

# Output:
# Next print job: Document1.pdf
# Processing: Document1.pdf
# New next print job: Image.jpg
```

Slide 14: C++ Queues

C++ Queues - Using std::queue

C++ provides a built-in queue container adapter in the Standard Template Library (STL).

Code:

```cpp
#include <iostream>
#include <queue>
#include <string>
using namespace std;

int main() {
    queue<string> customer_service;

    // Adding customers to the queue
    customer_service.push("Alice");
    customer_service.push("Bob");
    customer_service.push("Charlie");

    cout << "First in line: " << customer_service.front() << endl;
    cout << "Queue size: " << customer_service.size() << endl;

    // Serving customers
    customer_service.pop();
    cout << "After serving first customer:" << endl;
    cout << "New first in line: " << customer_service.front() << endl;

    // Serving all customers
    while (!customer_service.empty()) {
        cout << "Serving: " << customer_service.front() << endl;
        customer_service.pop();
    }

    cout << "Is queue empty? " << (customer_service.empty() ? "Yes" : "No") << endl;

    return 0;
}

// Output:
// First in line: Alice
// Queue size: 3
// After serving first customer:
// New first in line: Bob
// Serving: Bob
// Serving: Charlie
// Is queue empty? Yes
```

Slide 15: Python Heaps

Python Heaps - Using heapq Module

Python's heapq module implements a min-heap, useful for priority queues and sorting.

Code:

```python
import heapq

# Creating a min-heap
task_queue = []
heapq.heappush(task_queue, (3, "Low priority task"))
heapq.heappush(task_queue, (1, "High priority task"))
heapq.heappush(task_queue, (2, "Medium priority task"))

# Popping items from the heap
while task_queue:
    priority, task = heapq.heappop(task_queue)
    print(f"Executing: {task} (Priority: {priority})")

# Output:
# Executing: High priority task (Priority: 1)
# Executing: Medium priority task (Priority: 2)
# Executing: Low priority task (Priority: 3)

# Converting a list to a heap
numbers = [5, 2, 8, 1, 9]
heapq.heapify(numbers)
print("Heapified list:", numbers)
# Output: Heapified list: [1, 2, 8, 5, 9]
```

Slide 16: C++ Priority Queues

C++ Priority Queues - Using std::priority\_queue

C++ provides a priority\_queue container adapter in the STL, implementing a max-heap by default.

Code:

```cpp
#include <iostream>
#include <queue>
#include <vector>
using namespace std;

int main() {
    // Max-heap (default)
    priority_queue<int> max_heap;
    
    // Min-heap
    priority_queue<int, vector<int>, greater<int>> min_heap;

    // Adding elements
    for (int num : {3, 1, 4, 1, 5, 9, 2, 6}) {
        max_heap.push(num);
        min_heap.push(num);
    }

    cout << "Max-heap top: " << max_heap.top() << endl;
    cout << "Min-heap top: " << min_heap.top() << endl;

    // Popping elements from max-heap
    cout << "Max-heap elements:" << endl;
    while (!max_heap.empty()) {
        cout << max_heap.top() << " ";
        max_heap.pop();
    }
    cout << endl;

    // Output:
    // Max-heap top: 9
    // Min-heap top: 1
    // Max-heap elements:
    // 9 6 5 4 3 2 1 1

    return 0;
}
```

Slide 17: Python Trees

Python Trees - Custom Binary Tree Implementation

Trees in Python are typically implemented as custom classes, with nodes containing data and references to child nodes.

Code:

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if not self.root:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._insert_recursive(node.right, value)

    def inorder_traversal(self):
        return self._inorder_recursive(self.root)

    def _inorder_recursive(self, node):
        if node is None:
            return []
        return (self._inorder_recursive(node.left) +
                [node.value] +
                self._inorder_recursive(node.right))

# Using the binary tree
tree = BinaryTree()
for value in [5, 3, 7, 1, 4, 6, 8]:
    tree.insert(value)

print("Inorder traversal:", tree.inorder_traversal())
# Output: Inorder traversal: [1, 3, 4, 5, 6, 7, 8]
```

Slide 18: C++ Trees

C++ Trees - Custom Binary Search Tree Implementation

Trees in C++ are typically implemented as custom classes, with nodes containing data and pointers to child nodes.

Code:

```cpp
#include <iostream>
#include <vector>
using namespace std;

struct TreeNode {
    int value;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int val) : value(val), left(nullptr), right(nullptr) {}
};

class BinarySearchTree {
private:
    TreeNode* root;

    void insertRecursive(TreeNode*& node, int value) {
        if (node == nullptr) {
            node = new TreeNode(value);
        } else if (value < node->value) {
            insertRecursive(node->left, value);
        } else {
            insertRecursive(node->right, value);
        }
    }

    void inorderTraversalRecursive(TreeNode* node, vector<int>& result) {
        if (node == nullptr) return;
        inorderTraversalRecursive(node->left, result);
        result.push_back(node->value);
        inorderTraversalRecursive(node->right, result);
    }

public:
    BinarySearchTree() : root(nullptr) {}

    void insert(int value) {
        insertRecursive(root, value);
    }

    vector<int> inorderTraversal() {
        vector<int> result;
        inorderTraversalRecursive(root, result);
        return result;
    }
};

int main() {
    BinarySearchTree bst;
    for (int value : {5, 3, 7, 1, 4, 6, 8}) {
        bst.insert(value);
    }

    vector<int> inorder = bst.inorderTraversal();
    cout << "Inorder traversal: ";
    for (int value : inorder) {
        cout << value << " ";
    }
    cout << endl;

    // Output: Inorder traversal: 1 3 4 5 6 7 8

    return 0;
}
```

Slide 19: Python Graphs

Python Graphs - Adjacency List Implementation

Graphs in Python can be implemented using dictionaries to represent adjacency lists.

Code:

```python
class Graph:
    def __init__(self):
        self.graph = {}

    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []

    def add_edge(self, vertex1, vertex2):
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)
        self.graph[vertex1].append(vertex2)
        self.graph[vertex2].append(vertex1)  # For undirected graph

    def display(self):
        for vertex, neighbors in self.graph.items():
            print(f"{vertex}: {neighbors}")

# Using the graph
social_network = Graph()
social_network.add_edge("Alice", "Bob")
social_network.add_edge("Alice", "Charlie")
social_network.add_edge("Bob", "David")
social_network.add_edge("Charlie", "David")
social_network.add_edge("Eve", "Alice")

print("Social Network Connections:")
social_network.display()

# Output:
# Social Network Connections:
# Alice: ['Bob', 'Charlie', 'Eve']
# Bob: ['Alice', 'David']
# Charlie: ['Alice', 'David']
# David: ['Bob', 'Charlie']
# Eve: ['Alice']
```

Slide 20: C++ Graphs

C++ Graphs - Adjacency List Implementation

Graphs in C++ can be implemented using vectors and pairs to represent adjacency lists.

Code:

```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
using namespace std;

class Graph {
private:
    unordered_map<string, vector<string>> adjacencyList;

public:
    void addEdge(const string& vertex1, const string& vertex2) {
        adjacencyList[vertex1].push_back(vertex2);
        adjacencyList[vertex2].push_back(vertex1);  // For undirected graph
    }

    void display() {
        for (const auto& pair : adjacencyList) {
            cout << pair.first << ": ";
            for (const string& neighbor : pair.second) {
                cout << neighbor << " ";
            }
            cout << endl;
        }
    }
};

int main() {
    Graph socialNetwork;
    socialNetwork.addEdge("Alice", "Bob");
    socialNetwork.addEdge("Alice", "Charlie");
    socialNetwork.addEdge("Bob", "David");
    socialNetwork.addEdge("Charlie", "David");
    socialNetwork.addEdge("Eve", "Alice");

    cout << "Social Network Connections:" << endl;
    socialNetwork.display();

    // Output:
    // Social Network Connections:
    // Alice: Bob Charlie Eve
    // Bob: Alice David
    // Charlie: Alice David
    // David: Bob Charlie
    // Eve: Alice

    return 0;
}
```

Slide 21: Wrap-up - Python vs C++ Data Structures

Python vs C++ Data Structures Comparison

Key differences between Python and C++ data structures implementation and usage.

Code:

```
| Feature          | Python                                   | C++                                      |
|------------------|------------------------------------------|------------------------------------------|
| Arrays/Lists     | Dynamic (list)                           | Fixed-size (array) or dynamic (vector)   |
| Dictionaries/Maps| Built-in (dict)                          | std::map or std::unordered_map           |
| Sets             | Built-in (set)                           | std::set or std::unordered_set           |
| Tuples           | Built-in, immutable                      | std::tuple, fixed-size                   |
| Linked Lists     | Custom implementation                    | Custom implementation or std::list       |
| Stacks           | List or collections.deque                | std::stack                               |
| Queues           | collections.deque or queue.Queue         | std::queue                               |
| Priority Queues  | heapq module (min-heap)                  | std::priority_queue (max-heap by default)|
| Trees            | Custom implementation                    | Custom implementation                    |
| Graphs           | Custom (dict, list, or set)              | Custom (vector, list, or set)            |
| Memory Management| Automatic (garbage collection)           | Manual (new/delete) or smart pointers    |
| Performance      | Generally slower, easier to use          | Generally faster, more control           |
| Syntax           | More concise, easier to read             | More verbose, requires explicit typing   |
| Standard Library | Rich built-in data structures            | Comprehensive STL                        |
```

This wrap-up slide provides a concise comparison of data structures in Python and C++, highlighting key differences in implementation, performance, and usage.

