## Shallow vs Deep  in Python
Slide 1: Understanding Shallow and Deep  in Python

In Python, ing objects can be more complex than it seems. This presentation will explore the concepts of shallow and deep ing, their differences, and how to implement them effectively.

```python
# Let's start with a simple list
original = [1, [2, 3], 4]
print(f"Original: {original}")

# We'll explore different ways to  this list
```

Slide 2: Assignment is Not ing

When we assign a variable to another, we're creating a new reference to the same object, not a new .

```python
original = [1, [2, 3], 4]
assigned = original

original[0] = 99
print(f"Original: {original}")
print(f"Assigned: {assigned}")

# Output:
# Original: [99, [2, 3], 4]
# Assigned: [99, [2, 3], 4]
```

Slide 3: Shallow : Introduction

A shallow  creates a new object but references the same nested objects as the original.

```python
import 

original = [1, [2, 3], 4]
shallow = .(original)

print(f"Original: {original}")
print(f"Shallow : {shallow}")
print(f"Is original == shallow? {original == shallow}")
print(f"Is original is shallow? {original is shallow}")

# Output:
# Original: [1, [2, 3], 4]
# Shallow : [1, [2, 3], 4]
# Is original == shallow? True
# Is original is shallow? False
```

Slide 4: Shallow : Modifying Nested Objects

Changes to nested mutable objects in a shallow  affect the original and vice versa.

```python
original = [1, [2, 3], 4]
shallow = .(original)

shallow[1][0] = 99
print(f"Original after modifying shallow : {original}")
print(f"Shallow  after modification: {shallow}")

# Output:
# Original after modifying shallow : [1, [99, 3], 4]
# Shallow  after modification: [1, [99, 3], 4]
```

Slide 5: Deep : Introduction

A deep  creates a new object and recursively copies all nested objects, creating independent copies.

```python
import 

original = [1, [2, 3], 4]
deep = .deep(original)

print(f"Original: {original}")
print(f"Deep : {deep}")
print(f"Is original == deep? {original == deep}")
print(f"Is original is deep? {original is deep}")

# Output:
# Original: [1, [2, 3], 4]
# Deep : [1, [2, 3], 4]
# Is original == deep? True
# Is original is deep? False
```

Slide 6: Deep : Modifying Nested Objects

Changes to nested mutable objects in a deep  do not affect the original.

```python
original = [1, [2, 3], 4]
deep = .deep(original)

deep[1][0] = 99
print(f"Original after modifying deep : {original}")
print(f"Deep  after modification: {deep}")

# Output:
# Original after modifying deep : [1, [2, 3], 4]
# Deep  after modification: [1, [99, 3], 4]
```

Slide 7: Real-Life Example: Modifying User Profiles

Imagine a scenario where we need to create temporary user profiles for testing.

```python
def create_test_profile(base_profile):
    test_profile = .deep(base_profile)
    test_profile['id'] += '_test'
    test_profile['permissions'].append('test_access')
    return test_profile

base_profile = {
    'id': 'user123',
    'name': 'John Doe',
    'permissions': ['read', 'write']
}

test_profile = create_test_profile(base_profile)
print(f"Base profile: {base_profile}")
print(f"Test profile: {test_profile}")

# Output:
# Base profile: {'id': 'user123', 'name': 'John Doe', 'permissions': ['read', 'write']}
# Test profile: {'id': 'user123_test', 'name': 'John Doe', 'permissions': ['read', 'write', 'test_access']}
```

Slide 8: Shallow vs Deep : Performance Considerations

Shallow copies are generally faster and use less memory, but deep copies ensure complete independence.

```python
import 
import timeit

def shallow__test():
    original = [1, [2, 3], 4] * 1000
    shallow = .(original)

def deep__test():
    original = [1, [2, 3], 4] * 1000
    deep = .deep(original)

shallow_time = timeit.timeit(shallow__test, number=1000)
deep_time = timeit.timeit(deep__test, number=1000)

print(f"Shallow  time: {shallow_time:.6f} seconds")
print(f"Deep  time: {deep_time:.6f} seconds")

# Output may vary, but deep  will be significantly slower
# Shallow  time: 0.012345 seconds
# Deep  time: 0.678901 seconds
```

Slide 9: ing Custom Objects

For custom objects, the `` module uses the `____` and `__deep__` methods if defined.

```python
import 

class CustomObject:
    def __init__(self, x):
        self.x = x
    
    def ____(self):
        return CustomObject(self.x)
    
    def __deep__(self, memo):
        return CustomObject(.deep(self.x, memo))

obj = CustomObject([1, 2, 3])
shallow = .(obj)
deep = .deep(obj)

print(f"Original: {obj.x}")
print(f"Shallow : {shallow.x}")
print(f"Deep : {deep.x}")

# Output:
# Original: [1, 2, 3]
# Shallow : [1, 2, 3]
# Deep : [1, 2, 3]
```

Slide 10: Real-Life Example: Game State Management

In game development, managing game state often requires ing objects.

```python
import 

class GameState:
    def __init__(self, player_pos, enemies, items):
        self.player_pos = player_pos
        self.enemies = enemies
        self.items = items

def save_game_state(current_state):
    return .deep(current_state)

current_state = GameState([0, 0], [['Enemy1', [1, 1]], ['Enemy2', [2, 2]]], ['Sword', 'Shield'])
saved_state = save_game_state(current_state)

# Player moves and picks up an item
current_state.player_pos = [1, 1]
current_state.items.append('Potion')

print(f"Current state: {current_state.__dict__}")
print(f"Saved state: {saved_state.__dict__}")

# Output:
# Current state: {'player_pos': [1, 1], 'enemies': [['Enemy1', [1, 1]], ['Enemy2', [2, 2]]], 'items': ['Sword', 'Shield', 'Potion']}
# Saved state: {'player_pos': [0, 0], 'enemies': [['Enemy1', [1, 1]], ['Enemy2', [2, 2]]], 'items': ['Sword', 'Shield']}
```

Slide 11: Shallow  with Slicing

List slicing creates a shallow  of the list.

```python
original = [1, [2, 3], 4]
sliced = original[:]

print(f"Original: {original}")
print(f"Sliced (shallow ): {sliced}")
print(f"Is original == sliced? {original == sliced}")
print(f"Is original is sliced? {original is sliced}")

sliced[1][0] = 99
print(f"Original after modifying sliced : {original}")
print(f"Sliced  after modification: {sliced}")

# Output:
# Original: [1, [2, 3], 4]
# Sliced (shallow ): [1, [2, 3], 4]
# Is original == sliced? True
# Is original is sliced? False
# Original after modifying sliced : [1, [99, 3], 4]
# Sliced  after modification: [1, [99, 3], 4]
```

Slide 12: ing Immutable Objects

Immutable objects like strings, tuples, and frozensets don't need deep ing.

```python
import 

immutable = (1, 2, 'hello')
shallow = .(immutable)
deep = .deep(immutable)

print(f"Original: {immutable}")
print(f"Shallow : {shallow}")
print(f"Deep : {deep}")
print(f"Are all objects the same? {immutable is shallow is deep}")

# Output:
# Original: (1, 2, 'hello')
# Shallow : (1, 2, 'hello')
# Deep : (1, 2, 'hello')
# Are all objects the same? True
```

Slide 13: ing in Numpy Arrays

Numpy, a popular library for scientific computing, has its own ing mechanisms.

```python
import numpy as np

arr = np.array([1, 2, 3])
view = arr.view()
 = arr.()

arr[0] = 99

print(f"Original: {arr}")
print(f"View: {view}")
print(f": {}")

# Output:
# Original: [99  2  3]
# View: [99  2  3]
# : [1 2 3]
```

Slide 14: Best Practices and Considerations

When deciding between shallow and deep :

* Use shallow  for simple data structures with immutable elements
* Use deep  for complex nested structures or when complete independence is required
* Consider performance implications, especially for large data structures
* Implement custom `____` and `__deep__` methods for better control over ing behavior in custom classes
* Be aware of the differences when working with different data types and libraries

Slide 15: Additional Resources

For more information on shallow and deep ing in Python, consider exploring these resources:

1. Python official documentation on the  module: [https://docs.python.org/3/library/.html](https://docs.python.org/3/library/.html)
2. "The Perils of Deep ing" by Raymond Hettinger: [https://rhettinger.wordpress.com/2013/06/05/the-perils-of-deep-ing/](https://rhettinger.wordpress.com/2013/06/05/the-perils-of-deep-ing/)
3. "Understanding Deep and Shallow  in Python" by Real Python: [https://realpython.com/ing-python-objects/](https://realpython.com/ing-python-objects/)

These resources provide in-depth explanations and advanced use cases for ing in Python.

