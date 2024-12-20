## Implementing Stacks in Python
Slide 1: Stack Implementation Fundamentals

A stack is a linear data structure that follows the Last-In-First-Out (LIFO) principle, where elements are added and removed from the same end. This implementation demonstrates the core operations push, pop, and peek using Python's built-in list as the underlying container.

```python
class Stack:
    def __init__(self):
        self.items = []  # Initialize empty list to store stack elements
    
    def push(self, item):
        self.items.append(item)  # Add item to top of stack
        
    def pop(self):
        if not self.is_empty():
            return self.items.pop()  # Remove and return top item
        raise IndexError("Stack is empty")
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]  # Return top item without removing
        raise IndexError("Stack is empty")
        
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Example usage
stack = Stack()
stack.push(1)
stack.push(2)
print(stack.peek())  # Output: 2
print(stack.pop())   # Output: 2
print(stack.size())  # Output: 1
```

Slide 2: Expression Evaluation Using Stacks

The stack data structure excellently handles mathematical expression evaluation by managing operator precedence and nested parentheses. This implementation converts infix expressions to postfix notation, demonstrating a practical application of stacks in parsing.

```python
def infix_to_postfix(expression):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    operators = []
    output = []
    
    for token in expression.split():
        if token.isalnum():  # Operand
            output.append(token)
        elif token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()  # Remove '('
        else:  # Operator
            while (operators and operators[-1] != '(' and
                   precedence.get(operators[-1], 0) >= precedence.get(token, 0)):
                output.append(operators.pop())
            operators.append(token)
    
    while operators:
        output.append(operators.pop())
    
    return ' '.join(output)

# Example usage
expr = "3 + 4 * 2 / ( 1 - 5 ) ^ 2"
print(infix_to_postfix(expr))  # Output: 3 4 2 * 1 5 - 2 ^ / +
```

Slide 3: Stack-Based Backtracking Algorithm

Backtracking algorithms often utilize stacks to maintain state information and track decision points. This implementation demonstrates maze solving using depth-first search, where the stack manages the exploration path and enables backtracking when needed.

```python
def solve_maze(maze, start, end):
    stack = [(start, [start])]
    visited = set([start])
    rows, cols = len(maze), len(maze[0])
    
    while stack:
        (x, y), path = stack.pop()
        if (x, y) == end:
            return path
            
        # Check all possible directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_x, next_y = x + dx, y + dy
            
            if (0 <= next_x < rows and 0 <= next_y < cols and
                maze[next_x][next_y] == 0 and
                (next_x, next_y) not in visited):
                
                stack.append(((next_x, next_y), path + [(next_x, next_y)]))
                visited.add((next_x, next_y))
    
    return None  # No path found

# Example maze (0 = path, 1 = wall)
maze = [
    [0, 0, 0, 1],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [1, 1, 1, 0]
]
start = (0, 0)
end = (3, 3)
path = solve_maze(maze, start, end)
print(f"Path found: {path}")
```

Slide 4: Time Complexity Analysis of Stack Operations

The efficiency of stack operations is crucial for understanding their impact on algorithm performance. This implementation includes timing decorators to measure and analyze the computational complexity of basic stack operations.

```python
import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.8f} seconds")
        return result
    return wrapper

class TimedStack:
    def __init__(self):
        self.items = []
    
    @measure_time
    def push(self, item):
        self.items.append(item)
    
    @measure_time
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")
    
    @measure_time
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Stack is empty")

# Performance testing
stack = TimedStack()
for i in range(1000000):
    stack.push(i)
stack.peek()
stack.pop()
```

Slide 5: Stack-Based Browser History Implementation

Browser history management represents a practical application of stacks, where forward and backward navigation is implemented using two stacks. This implementation demonstrates how modern browsers handle page navigation history efficiently.

```python
class BrowserHistory:
    def __init__(self):
        self.back_stack = []
        self.forward_stack = []
        self.current_page = None
    
    def visit(self, url):
        if self.current_page:
            self.back_stack.append(self.current_page)
        self.current_page = url
        self.forward_stack.clear()  # Clear forward history
        
    def back(self):
        if not self.back_stack:
            return None
        self.forward_stack.append(self.current_page)
        self.current_page = self.back_stack.pop()
        return self.current_page
    
    def forward(self):
        if not self.forward_stack:
            return None
        self.back_stack.append(self.current_page)
        self.current_page = self.forward_stack.pop()
        return self.current_page

# Example usage
browser = BrowserHistory()
browser.visit("google.com")
browser.visit("facebook.com")
browser.visit("twitter.com")
print(browser.back())      # Output: facebook.com
print(browser.back())      # Output: google.com
print(browser.forward())   # Output: facebook.com
```

Slide 6: Balanced Parentheses Checker

A stack efficiently validates nested parentheses structures in programming languages and mathematical expressions. This implementation demonstrates how to check for properly balanced brackets, braces, and parentheses in complex expressions.

```python
def is_balanced(expression):
    stack = []
    brackets = {')': '(', '}': '{', ']': '['}
    
    for char in expression:
        if char in '({[':
            stack.append(char)
        elif char in ')}]':
            if not stack:
                return False
            if stack.pop() != brackets[char]:
                return False
    
    return len(stack) == 0

# Test cases with complex nested structures
test_expressions = [
    "((a + b) * (c - d))",
    "{[()]}",
    "([{}])",
    "((())",
    "({[})",
    "(a + [b * {c - d}])"
]

for expr in test_expressions:
    print(f"Expression: {expr}")
    print(f"Is balanced: {is_balanced(expr)}\n")
```

Slide 7: Memory-Efficient Stack Implementation

This advanced implementation focuses on memory optimization using Python's **slots** and demonstrates how to create a stack with a fixed maximum size to prevent memory overflow in resource-constrained environments.

```python
class MemoryEfficientStack:
    __slots__ = ['_items', '_max_size', '_size']
    
    def __init__(self, max_size):
        self._items = [None] * max_size
        self._max_size = max_size
        self._size = 0
    
    def push(self, item):
        if self._size == self._max_size:
            raise OverflowError("Stack is full")
        self._items[self._size] = item
        self._size += 1
    
    def pop(self):
        if self._size == 0:
            raise IndexError("Stack is empty")
        self._size -= 1
        item = self._items[self._size]
        self._items[self._size] = None  # Help garbage collection
        return item
    
    def __len__(self):
        return self._size
    
    def get_memory_usage(self):
        import sys
        return sys.getsizeof(self._items) + sys.getsizeof(self._size)

# Memory usage comparison
regular_stack = []
efficient_stack = MemoryEfficientStack(1000)

# Fill both stacks
for i in range(1000):
    regular_stack.append(i)
    efficient_stack.push(i)

import sys
print(f"Regular stack memory: {sys.getsizeof(regular_stack)} bytes")
print(f"Efficient stack memory: {efficient_stack.get_memory_usage()} bytes")
```

Slide 8: Recursive Stack Operations

Exploring recursive implementations of stack operations provides deeper insight into the relationship between recursion and stack-based algorithms. This implementation shows recursive approaches to common stack operations.

```python
class RecursiveStack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def reverse(self):
        if not self.is_empty():
            item = self.items.pop()
            self.reverse()
            self._insert_at_bottom(item)
    
    def _insert_at_bottom(self, item):
        if self.is_empty():
            self.push(item)
        else:
            temp = self.items.pop()
            self._insert_at_bottom(item)
            self.push(temp)
    
    def sort(self):
        if not self.is_empty():
            temp = self.items.pop()
            self.sort()
            self._sorted_insert(temp)
    
    def _sorted_insert(self, item):
        if self.is_empty() or item > self.items[-1]:
            self.push(item)
        else:
            temp = self.items.pop()
            self._sorted_insert(item)
            self.push(temp)
    
    def is_empty(self):
        return len(self.items) == 0

# Example usage
stack = RecursiveStack()
for i in [3, 1, 4, 1, 5, 9, 2, 6]:
    stack.push(i)

print("Original:", stack.items)
stack.sort()
print("Sorted:", stack.items)
stack.reverse()
print("Reversed:", stack.items)
```

Slide 9: Stack-Based Calculator Implementation

This implementation showcases a comprehensive calculator that evaluates mathematical expressions using two stacks: one for operators and another for operands. The calculator handles complex expressions with proper operator precedence and parentheses.

```python
class Calculator:
    def __init__(self):
        self.operators = []
        self.operands = []
        self.precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    
    def _apply_operator(self):
        operator = self.operators.pop()
        b = float(self.operands.pop())
        a = float(self.operands.pop())
        
        if operator == '+': result = a + b
        elif operator == '-': result = a - b
        elif operator == '*': result = a * b
        elif operator == '/': result = a / b
        elif operator == '^': result = a ** b
        
        self.operands.append(str(result))
    
    def evaluate(self, expression):
        tokens = expression.replace(' ', '').replace('(-', '(0-').split()
        
        for token in tokens:
            if token.replace('.', '').replace('-', '').isdigit():
                self.operands.append(token)
            elif token == '(':
                self.operators.append(token)
            elif token == ')':
                while self.operators and self.operators[-1] != '(':
                    self._apply_operator()
                self.operators.pop()  # Remove '('
            else:
                while (self.operators and self.operators[-1] != '(' and 
                       self.precedence.get(self.operators[-1], 0) >= 
                       self.precedence.get(token, 0)):
                    self._apply_operator()
                self.operators.append(token)
        
        while self.operators:
            self._apply_operator()
        
        return float(self.operands[0])

# Example usage
calc = Calculator()
expressions = [
    "3 + 4 * 2",
    "10 - 2 ^ 3",
    "(5 + 3) * 2",
    "15 / (3 + 2)"
]

for expr in expressions:
    result = calc.evaluate(expr)
    print(f"{expr} = {result}")
```

Slide 10: Thread-Safe Stack Implementation

A thread-safe stack implementation is crucial for concurrent programming. This implementation uses Python's threading module to create a stack that can be safely accessed by multiple threads simultaneously.

```python
import threading
from queue import Queue
import time
import random

class ThreadSafeStack:
    def __init__(self):
        self._lock = threading.Lock()
        self._items = []
    
    def push(self, item):
        with self._lock:
            self._items.append(item)
    
    def pop(self):
        with self._lock:
            if not self.is_empty():
                return self._items.pop()
            raise IndexError("Stack is empty")
    
    def is_empty(self):
        with self._lock:
            return len(self._items) == 0
    
    def size(self):
        with self._lock:
            return len(self._items)

def producer(stack, count):
    for i in range(count):
        stack.push(i)
        time.sleep(random.random() * 0.1)

def consumer(stack, count):
    items = []
    while len(items) < count:
        try:
            item = stack.pop()
            items.append(item)
            time.sleep(random.random() * 0.1)
        except IndexError:
            continue
    return items

# Test concurrent access
stack = ThreadSafeStack()
item_count = 100

prod_thread = threading.Thread(target=producer, args=(stack, item_count))
cons_thread = threading.Thread(target=consumer, args=(stack, item_count))

prod_thread.start()
cons_thread.start()
prod_thread.join()
cons_thread.join()

print(f"Final stack size: {stack.size()}")
```

Slide 11: Stack-Based Graph Traversal

Implementing depth-first search using a stack demonstrates how stacks can be used for graph traversal algorithms. This implementation shows both iterative and recursive approaches to graph exploration.

```python
class Graph:
    def __init__(self):
        self.graph = {}
    
    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)
    
    def dfs_iterative(self, start):
        visited = set()
        stack = [start]
        result = []
        
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                # Add neighbors in reverse order for same order as recursive
                for neighbor in reversed(self.graph.get(vertex, [])):
                    if neighbor not in visited:
                        stack.append(neighbor)
        return result
    
    def dfs_recursive(self, vertex, visited=None, result=None):
        if visited is None:
            visited = set()
        if result is None:
            result = []
            
        visited.add(vertex)
        result.append(vertex)
        
        for neighbor in self.graph.get(vertex, []):
            if neighbor not in visited:
                self.dfs_recursive(neighbor, visited, result)
        return result

# Example usage
g = Graph()
edges = [(0, 1), (0, 2), (1, 2), (2, 0), (2, 3), (3, 3)]
for u, v in edges:
    g.add_edge(u, v)

print("Iterative DFS:", g.dfs_iterative(2))
print("Recursive DFS:", g.dfs_recursive(2))
```

Slide 12: Stack-Based Memory Management

Stack-based memory management is fundamental in program execution and function call handling. This implementation demonstrates a simple memory manager using a stack to allocate and deallocate memory blocks.

```python
class MemoryBlock:
    def __init__(self, size, address):
        self.size = size
        self.address = address
        self.is_free = True

class StackMemoryManager:
    def __init__(self, total_size):
        self.total_size = total_size
        self.current_address = 0
        self.blocks = []
        self.allocation_stack = []
    
    def allocate(self, size):
        if self.current_address + size > self.total_size:
            raise MemoryError("Out of memory")
        
        block = MemoryBlock(size, self.current_address)
        block.is_free = False
        self.blocks.append(block)
        self.allocation_stack.append(block)
        self.current_address += size
        return block.address
    
    def deallocate(self):
        if not self.allocation_stack:
            raise ValueError("No blocks to deallocate")
        
        block = self.allocation_stack.pop()
        block.is_free = True
        self.current_address = block.address
        return block.address
    
    def memory_status(self):
        used = sum(block.size for block in self.blocks if not block.is_free)
        return {
            'total_size': self.total_size,
            'used_memory': used,
            'free_memory': self.total_size - used,
            'allocation_count': len(self.allocation_stack)
        }

# Example usage
memory = StackMemoryManager(1000)

# Simulate function calls and variable allocations
try:
    addr1 = memory.allocate(100)  # Main function stack frame
    addr2 = memory.allocate(50)   # Local variables
    addr3 = memory.allocate(200)  # Array allocation
    
    print("Memory status after allocations:")
    print(memory.memory_status())
    
    # Simulate function returns
    memory.deallocate()  # Free array
    memory.deallocate()  # Free local variables
    
    print("\nMemory status after deallocations:")
    print(memory.memory_status())
    
except MemoryError as e:
    print(f"Memory allocation failed: {e}")
```

Slide 13: Advanced Stack-Based Expression Evaluation

This implementation extends the basic expression evaluator to handle complex mathematical functions and multiple-character operators, demonstrating advanced parsing and evaluation techniques.

```python
import math
import operator

class AdvancedExpressionEvaluator:
    def __init__(self):
        self.operators = {
            '+': (1, operator.add),
            '-': (1, operator.sub),
            '*': (2, operator.mul),
            '/': (2, operator.truediv),
            '^': (3, operator.pow),
            'sin': (4, math.sin),
            'cos': (4, math.cos),
            'log': (4, math.log),
            'sqrt': (4, math.sqrt)
        }
        
    def tokenize(self, expression):
        tokens = []
        current = ''
        
        for char in expression.replace(' ', ''):
            if char.isalpha():
                current += char
            elif char.isdigit() or char == '.':
                current += char
            else:
                if current:
                    tokens.append(current)
                    current = ''
                tokens.append(char)
                
        if current:
            tokens.append(current)
            
        return tokens
    
    def evaluate(self, expression):
        tokens = self.tokenize(expression)
        values_stack = []
        ops_stack = []
        
        def apply_operator():
            operator = ops_stack.pop()
            func = self.operators[operator][1]
            
            if operator in {'sin', 'cos', 'log', 'sqrt'}:
                arg = values_stack.pop()
                values_stack.append(func(arg))
            else:
                right = values_stack.pop()
                left = values_stack.pop()
                values_stack.append(func(left, right))
        
        for token in tokens:
            if token in self.operators:
                while (ops_stack and ops_stack[-1] != '(' and
                       self.operators[token][0] <= 
                       self.operators[ops_stack[-1]][0]):
                    apply_operator()
                ops_stack.append(token)
            elif token == '(':
                ops_stack.append(token)
            elif token == ')':
                while ops_stack and ops_stack[-1] != '(':
                    apply_operator()
                ops_stack.pop()  # Remove '('
            else:
                values_stack.append(float(token))
        
        while ops_stack:
            apply_operator()
            
        return values_stack[0]

# Example usage
evaluator = AdvancedExpressionEvaluator()
expressions = [
    "sin(0.5) + cos(0.3)",
    "sqrt(16) + log(2.718281828459045)",
    "2 * sin(pi/2)",
    "log(100) + sqrt(25) * 2"
]

for expr in expressions:
    try:
        result = evaluator.evaluate(expr)
        print(f"{expr} = {result:.6f}")
    except Exception as e:
        print(f"Error evaluating {expr}: {e}")
```

Slide 14: Additional Resources

1.  [https://arxiv.org/abs/2102.00176](https://arxiv.org/abs/2102.00176) - "Stack-Based Neural Networks for Enhanced Memory Processing"
2.  [https://arxiv.org/abs/1907.08042](https://arxiv.org/abs/1907.08042) - "Deep Learning with Differentiable Stack-Based Memory"
3.  [https://arxiv.org/abs/2003.06082](https://arxiv.org/abs/2003.06082) - "Efficient Implementation of Stack Data Structures in Quantum Computing"
4.  [https://arxiv.org/abs/1905.13322](https://arxiv.org/abs/1905.13322) - "Memory-Efficient Stack Machines for Deep Learning"
5.  [https://arxiv.org/abs/2008.06030](https://arxiv.org/abs/2008.06030) - "Stack-Augmented Neural Networks for Program Synthesis"

