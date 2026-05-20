## 15 Very helpful Programming Concepts for Python
Slide 1: Thunk - Delayed Execution

Thunks allow us to postpone computation until it's absolutely necessary. This concept is particularly useful for optimizing performance in scenarios where expensive calculations may not always be needed.

```python
def expensive_calculation():
    print("Performing expensive calculation...")
    return sum(range(1000000))

# Create a thunk
thunk = lambda: expensive_calculation()

# The calculation is not performed yet
print("Thunk created, but not executed")

# Execute the thunk when needed
result = thunk()
print(f"Result: {result}")
```

Slide 2: Monad - Handling Uncertain Operations

Monads provide a way to chain operations while safely handling potential errors or uncertain outcomes. They're particularly useful in scenarios involving asynchronous operations or when dealing with nullable values.

```python
from typing import Callable, Optional

class Maybe:
    def __init__(self, value: Optional[int]):
        self.value = value

    def bind(self, func: Callable[[int], 'Maybe']) -> 'Maybe':
        if self.value is None:
            return Maybe(None)
        return func(self.value)

def divide_by_two(x: int) -> Maybe:
    return Maybe(x // 2 if x % 2 == 0 else None)

result = Maybe(16).bind(divide_by_two).bind(divide_by_two)
print(f"Result: {result.value}")  # Output: Result: 4

result = Maybe(15).bind(divide_by_two).bind(divide_by_two)
print(f"Result: {result.value}")  # Output: Result: None
```

Slide 3: Closure - Functions with Memory

Closures allow functions to "remember" and access variables from their outer scope, even after the outer function has finished executing. This concept is fundamental in creating function factories and maintaining state.

```python
def counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

counter1 = counter()
print(counter1())  # Output: 1
print(counter1())  # Output: 2

counter2 = counter()
print(counter2())  # Output: 1
print(counter1())  # Output: 3
```

Slide 4: Memoization - Caching Function Results

Memoization is an optimization technique that stores the results of expensive function calls and returns the cached result when the same inputs occur again. This can significantly improve performance for recursive or computationally intensive functions.

```python
def memoize(func):
    cache = {}
    def memoized(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return memoized

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(100))  # Calculates quickly due to memoization
```

Slide 5: Continuation - Saving and Resuming Execution

Continuations allow us to capture the state of a computation at a certain point and resume it later. This concept is particularly useful in implementing complex control flows, especially in asynchronous programming.

```python
def simple_coroutine():
    print("Coroutine started")
    x = yield "First yield"
    print(f"Received: {x}")
    y = yield "Second yield"
    print(f"Received: {y}")

cr = simple_coroutine()
print(next(cr))  # Start the coroutine
print(cr.send("Hello"))  # Resume and send value
print(cr.send("World"))  # Resume and send value

# Output:
# Coroutine started
# First yield
# Received: Hello
# Second yield
# Received: World
# StopIteration
```

Slide 6: Idempotence - Consistency in Repeated Operations

Idempotence ensures that an operation produces the same result regardless of how many times it's applied. This property is crucial in designing robust systems, especially in distributed computing and RESTful APIs.

```python
class IdempotentCounter:
    def __init__(self):
        self.count = 0
        self.processed = set()

    def increment(self, operation_id):
        if operation_id not in self.processed:
            self.count += 1
            self.processed.add(operation_id)
        return self.count

counter = IdempotentCounter()
print(counter.increment("op1"))  # Output: 1
print(counter.increment("op1"))  # Output: 1 (no change)
print(counter.increment("op2"))  # Output: 2
```

Slide 7: Quine - Self-Replicating Code

A quine is a program that produces its own source code as output. While not typically used in practical applications, quines demonstrate interesting properties of programming languages and self-reference.

```python
s = 's = {!r}\nprint(s.format(s))'
print(s.format(s))

# Output:
# s = 's = {!r}\nprint(s.format(s))'
# print(s.format(s))
```

Slide 8: Zipper - Efficient Data Structure Navigation

Zippers provide a way to efficiently navigate and modify hierarchical data structures, such as trees. They allow for local updates without the need to recreate the entire structure.

```python
class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

class Zipper:
    def __init__(self, focus, path):
        self.focus = focus
        self.path = path

    def go_left(self):
        if self.focus.left:
            return Zipper(self.focus.left, ('left', self.focus, self.focus.right, self.path))
        return None

    def go_right(self):
        if self.focus.right:
            return Zipper(self.focus.right, ('right', self.focus, self.focus.left, self.path))
        return None

    def go_up(self):
        if not self.path:
            return None
        direction, parent, sibling, grandparent = self.path
        if direction == 'left':
            parent.left = self.focus
        else:
            parent.right = self.focus
        return Zipper(parent, grandparent)

# Example usage
root = TreeNode(1, TreeNode(2), TreeNode(3))
zipper = Zipper(root, None)
zipper = zipper.go_left()
zipper.focus.value = 4  # Modify left child
zipper = zipper.go_up()
zipper = zipper.go_right()
zipper.focus.value = 5  # Modify right child

print(root.value)  # 1
print(root.left.value)  # 4
print(root.right.value)  # 5
```

Slide 9: Functor - Mapping Over Containers

Functors are structures that can be mapped over, preserving the structure while applying a function to their contents. This concept is fundamental in functional programming and allows for flexible data transformations.

```python
from typing import Callable, TypeVar, Generic

T = TypeVar('T')
U = TypeVar('U')

class Maybe(Generic[T]):
    def __init__(self, value: T | None):
        self.value = value

    def map(self, func: Callable[[T], U]) -> 'Maybe[U]':
        if self.value is None:
            return Maybe(None)
        return Maybe(func(self.value))

# Example usage
def double(x: int) -> int:
    return x * 2

maybe_5 = Maybe(5)
maybe_10 = maybe_5.map(double)
print(maybe_10.value)  # Output: 10

maybe_none = Maybe(None)
still_none = maybe_none.map(double)
print(still_none.value)  # Output: None
```

Slide 10: Tail Call Optimization - Efficient Recursion

Tail Call Optimization (TCO) is a technique that optimizes recursive functions to prevent stack overflow. While Python doesn't natively support TCO, understanding the concept is crucial for writing efficient recursive algorithms.

```python
def factorial(n, acc=1):
    if n == 0:
        return acc
    return factorial(n - 1, n * acc)

# This function is tail-recursive, but Python doesn't optimize it
# In languages with TCO, this would not cause a stack overflow for large n
print(factorial(5))  # Output: 120

# To simulate TCO in Python, we can use a loop:
def factorial_tco(n):
    acc = 1
    while n > 0:
        acc *= n
        n -= 1
    return acc

print(factorial_tco(5))  # Output: 120
```

Slide 11: Currying - Function Decomposition

Currying is the technique of converting a function that takes multiple arguments into a sequence of functions, each taking a single argument. This allows for partial application and more flexible function composition.

```python
from functools import partial

def add(x, y, z):
    return x + y + z

# Manual currying
def curried_add(x):
    def curry_y(y):
        def curry_z(z):
            return x + y + z
        return curry_z
    return curry_y

# Usage of manually curried function
add_5 = curried_add(5)
add_5_10 = add_5(10)
result = add_5_10(15)
print(result)  # Output: 30

# Using partial for a similar effect
add_5 = partial(add, 5)
add_5_10 = partial(add_5, 10)
result = add_5_10(15)
print(result)  # Output: 30
```

Slide 12: Lazy Evaluation - Computing Only When Necessary

Lazy evaluation delays the evaluation of an expression until its value is actually needed. This can lead to performance improvements and allows for working with potentially infinite data structures.

```python
class LazyRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        current = self.start
        while current < self.end:
            yield current
            current += 1

# Using the lazy range
lazy_range = LazyRange(1, 1000000)
print("Lazy range created")

# Only compute values when needed
for num in lazy_range:
    if num > 5:
        break
    print(num)

print("Iteration complete")

# Output:
# Lazy range created
# 1
# 2
# 3
# 4
# 5
# Iteration complete
```

Slide 13: Side Effects - Beyond Local Scope

Side effects occur when a function modifies state outside its local environment. While sometimes necessary, understanding and managing side effects is crucial for writing predictable and maintainable code.

```python
global_var = 10

def function_with_side_effect(x):
    global global_var
    global_var += x
    return global_var

print(f"Before: {global_var}")
result = function_with_side_effect(5)
print(f"After: {global_var}")
print(f"Result: {result}")

# Output:
# Before: 10
# After: 15
# Result: 15

# A pure function alternative
def pure_function(x, y):
    return x + y

initial_value = 10
result = pure_function(initial_value, 5)
print(f"Initial value: {initial_value}")
print(f"Result: {result}")

# Output:
# Initial value: 10
# Result: 15
```

Slide 14: Hoisting - Declaration Lifting

While hoisting is a JavaScript concept, understanding it can help Python developers appreciate Python's scope rules. In Python, names are resolved using the LEGB rule (Local, Enclosing, Global, Built-in).

```python
# Python doesn't hoist, but it's good to understand scope rules

x = 10

def outer():
    # This prints the global x
    print("Outer x:", x)
    
    def inner():
        # This raises an UnboundLocalError
        print("Inner x:", x)
        x = 20
    
    inner()

outer()

# Output:
# Outer x: 10
# UnboundLocalError: local variable 'x' referenced before assignment

# To fix this, use 'nonlocal' or 'global' keyword
def outer_fixed():
    x = 10
    def inner():
        nonlocal x
        print("Inner x before:", x)
        x = 20
        print("Inner x after:", x)
    inner()
    print("Outer x after inner call:", x)

outer_fixed()

# Output:
# Inner x before: 10
# Inner x after: 20
# Outer x after inner call: 20
```

Slide 15: Monoid - Combining Data Consistently

A monoid is an algebraic structure with an associative binary operation and an identity element. In programming, monoids provide a consistent way to combine data, which is particularly useful in parallel and distributed computing.

```python
from typing import TypeVar, List, Callable

T = TypeVar('T')

class Monoid:
    def __init__(self, combine: Callable[[T, T], T], identity: T):
        self.combine = combine
        self.identity = identity

    def reduce(self, items: List[T]) -> T:
        return reduce(self.combine, items, self.identity)

# Example: Sum monoid
sum_monoid = Monoid(lambda x, y: x + y, 0)
numbers = [1, 2, 3, 4, 5]
result = sum_monoid.reduce(numbers)
print(f"Sum: {result}")  # Output: Sum: 15

# Example: String concatenation monoid
concat_monoid = Monoid(lambda x, y: x + y, "")
words = ["Hello", " ", "World", "!"]
result = concat_monoid.reduce(words)
print(f"Concatenated: {result}")  # Output: Concatenated: Hello World!

# Monoids allow for easy parallelization
def parallel_reduce(monoid: Monoid, items: List[T], chunk_size: int) -> T:
    chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
    partial_results = [monoid.reduce(chunk) for chunk in chunks]
    return monoid.reduce(partial_results)

large_list = list(range(1000000))
parallel_sum = parallel_reduce(sum_monoid, large_list, 1000)
print(f"Parallel Sum: {parallel_sum}")  # Output: Parallel Sum: 499999500000
```

Slide 16: Additional Resources

For those interested in delving deeper into these programming concepts, here are some valuable resources:

1. "Structure and Interpretation of Computer Programs" by Abelson, Sussman, and Sussman
2. "Concepts, Techniques, and Models of Computer Programming" by Van Roy and Haridi
3. ArXiv.org papers:
   * "A Tutorial on the Universality and Expressiveness of Fold" (arXiv:0903.2813)
   * "Monads for functional programming" (arXiv:1809.06289)

These resources provide in-depth explanations and formal treatments of many of the concepts we've discussed, as well as related topics in computer science and programming language theory.

