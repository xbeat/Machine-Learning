## Python Namespace Types and Scopes
Slide 1: Python Namespaces Overview

Python namespaces are containers that hold mappings of names to objects. There are four types of namespaces in Python: Built-in, Global, Enclosing, and Local. Each namespace has a specific scope and lifetime, determining when and where names are accessible in your code.

```python
# Demonstrating different namespaces
x = 10  # Global namespace

def outer_function():
    y = 20  # Enclosing namespace
    
    def inner_function():
        z = 30  # Local namespace
        print(f"Built-in function id() in action: {id(z)}")
        print(f"Local z: {z}, Enclosing y: {y}, Global x: {x}")
    
    inner_function()

outer_function()
```

Slide 2: Built-in Namespace

The Built-in namespace contains names for Python's built-in functions and exceptions. These are always available and don't need to be imported. Examples include `print()`, `len()`, and `range()`.

```python
# Exploring the built-in namespace
import builtins

print("Some built-in functions:")
for name in dir(builtins)[:10]:  # Printing first 10 for brevity
    if callable(getattr(builtins, name)):
        print(name)

# Using a built-in function
numbers = [1, 2, 3, 4, 5]
print(f"Sum of numbers: {sum(numbers)}")
```

Slide 3: Global Namespace

The Global namespace includes names defined at the top level of a script or module. These are available throughout the entire program unless shadowed by a local name.

```python
# Global namespace example
global_var = "I'm global"

def print_globals():
    print(f"Inside function: {global_var}")

print(f"Outside function: {global_var}")
print_globals()

# Listing global names
print("\nGlobal names:")
print(list(globals().keys())[:5])  # Printing first 5 for brevity
```

Slide 4: Enclosing Namespace

The Enclosing namespace occurs in nested functions. It contains names from the outer (enclosing) function that are accessible to the inner function.

```python
def outer():
    x = "outer x"
    
    def inner():
        print(f"Accessing enclosing x: {x}")
    
    inner()
    
    # Demonstrating nonlocal
    def inner_modifier():
        nonlocal x
        x = "modified x"
    
    inner_modifier()
    print(f"After modification: {x}")

outer()
```

Slide 5: Local Namespace

The Local namespace contains names defined within a function. These are only accessible within that function's scope.

```python
def demonstrate_local():
    local_var = "I'm local"
    print(f"Inside function: {local_var}")
    
    # Listing local names
    print("Local names:")
    print(list(locals().keys()))

demonstrate_local()

# This will raise a NameError
try:
    print(local_var)
except NameError as e:
    print(f"Error: {e}")
```

Slide 6: Namespace Lookup Order

Python follows the LEGB rule when looking up names: Local, Enclosing, Global, Built-in. This determines the order in which Python searches for a name.

```python
x = "global x"

def outer():
    x = "outer x"
    
    def inner():
        x = "inner x"
        print(f"Inner x: {x}")
    
    inner()
    print(f"Outer x: {x}")

outer()
print(f"Global x: {x}")
```

Slide 7: The `global` Keyword

The `global` keyword allows you to modify a global variable from within a function's local scope.

```python
count = 0

def increment():
    global count
    count += 1
    print(f"Count is now: {count}")

increment()
increment()
print(f"Final count: {count}")
```

Slide 8: The `nonlocal` Keyword

The `nonlocal` keyword is used in nested functions to refer to variables in the nearest enclosing scope, excluding the global scope.

```python
def outer():
    x = 0
    
    def inner():
        nonlocal x
        x += 1
        print(f"Inner x: {x}")
    
    inner()
    inner()
    print(f"Outer x: {x}")

outer()
```

Slide 9: Namespace Collision Resolution

When names collide across different namespaces, Python resolves them based on the LEGB rule. Local names take precedence over global names.

```python
x = "global"

def test_collision():
    x = "local"
    print(f"Inside function: {x}")
    
    # Accessing global x
    print(f"Global x: {globals()['x']}")

test_collision()
print(f"Outside function: {x}")
```

Slide 10: Dynamic Nature of Namespaces

Python namespaces are dynamic and can be modified at runtime. This allows for flexible programming but requires careful management.

```python
# Dynamically adding to global namespace
globals()['dynamic_var'] = "I'm dynamic"
print(dynamic_var)

def add_to_locals():
    locals()['local_dynamic'] = "Local dynamic"
    print(local_dynamic)

add_to_locals()

# This will raise a NameError
try:
    print(local_dynamic)
except NameError as e:
    print(f"Error: {e}")
```

Slide 11: Namespace and Module Imports

When you import a module, Python creates a new namespace for that module. This helps prevent naming conflicts between different modules.

```python
# file: my_module.py
# x = 100

# In main script:
import my_module

print(my_module.x)  # Accessing x from my_module's namespace

x = 200  # This doesn't affect my_module.x
print(f"Local x: {x}")
print(f"my_module.x: {my_module.x}")
```

Slide 12: Real-life Example: Configuration Management

Namespaces can be used for managing configuration settings in a large application, allowing for easy access and modification of settings across different parts of the program.

```python
class Config:
    DEBUG = False
    DATABASE_URI = "sqlite:///example.db"

def run_app():
    if Config.DEBUG:
        print("Running in debug mode")
    print(f"Connecting to database: {Config.DATABASE_URI}")

# In a different part of the application
def enable_debug():
    Config.DEBUG = True

enable_debug()
run_app()
```

Slide 13: Real-life Example: Game State Management

Namespaces can be utilized in game development to manage different states and scores, providing a clean way to organize and access game-related data.

```python
class GameState:
    score = 0
    level = 1
    player_health = 100

def play_game():
    print(f"Current level: {GameState.level}")
    print(f"Player health: {GameState.player_health}")
    
    # Simulate game action
    GameState.score += 50
    GameState.player_health -= 10
    
    print(f"Updated score: {GameState.score}")
    print(f"Updated player health: {GameState.player_health}")

play_game()
```

Slide 14: Namespace Best Practices

When working with namespaces, it's important to follow best practices to maintain clean and readable code. Avoid excessive use of global variables, use clear and descriptive names, and be mindful of namespace pollution.

```python
# Bad practice: using many globals
x = 0
y = 0
z = 0

def update_position(dx, dy, dz):
    global x, y, z
    x += dx
    y += dy
    z += dz

# Better practice: using a class
class Position:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def update(self, dx, dy, dz):
        self.x += dx
        self.y += dy
        self.z += dz

pos = Position()
pos.update(1, 2, 3)
print(f"New position: ({pos.x}, {pos.y}, {pos.z})")
```

Slide 15: Additional Resources

For further exploration of Python namespaces and related concepts, consider the following resources:

1. Python Documentation: [https://docs.python.org/3/tutorial/classes.html#python-scopes-and-namespaces](https://docs.python.org/3/tutorial/classes.html#python-scopes-and-namespaces)
2. Real Python - Python Scope & the LEGB Rule: [https://realpython.com/python-scope-legb-rule/](https://realpython.com/python-scope-legb-rule/)
3. ArXiv.org - "A Formal Specification of the Python Virtual Machine" by Mark Shannon: [https://arxiv.org/abs/2108.11242](https://arxiv.org/abs/2108.11242)

These resources provide in-depth explanations and advanced topics related to Python namespaces and scope.

