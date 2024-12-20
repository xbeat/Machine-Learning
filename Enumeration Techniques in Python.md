## Enumeration Techniques in Python
Slide 1: Introduction to Enumeration

Enumeration is a fundamental concept in combinatorics that involves counting distinct objects or arrangements. In this course, we'll explore various enumeration techniques using Python to implement and visualize these concepts.

```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

# Example: Calculate 5!
result = factorial(5)
print(f"5! = {result}")  # Output: 5! = 120
```

Slide 2: Permutations

Permutations are arrangements of objects where order matters. We'll use Python to calculate and generate permutations.

```python
import itertools

def calculate_permutations(items, r):
    return list(itertools.permutations(items, r))

# Example: Generate all permutations of 'ABC'
items = ['A', 'B', 'C']
perms = calculate_permutations(items, len(items))
print(f"Permutations of {items}: {perms}")
print(f"Number of permutations: {len(perms)}")
```

Slide 3: Combinations

Combinations are selections of objects where order doesn't matter. Let's implement a function to calculate combinations.

```python
import math

def calculate_combinations(n, r):
    return math.comb(n, r)

# Example: Calculate C(5,2)
n, r = 5, 2
result = calculate_combinations(n, r)
print(f"C({n},{r}) = {result}")  # Output: C(5,2) = 10
```

Slide 4: Binomial Coefficients

Binomial coefficients represent the number of ways to choose k items from n items. We'll use Python to calculate and visualize Pascal's triangle.

```python
def pascal_triangle(n):
    triangle = [[1]]
    for _ in range(n - 1):
        row = [1]
        for j in range(1, len(triangle[-1])):
            row.append(triangle[-1][j-1] + triangle[-1][j])
        row.append(1)
        triangle.append(row)
    return triangle

# Generate and print Pascal's triangle (first 5 rows)
for row in pascal_triangle(5):
    print(" ".join(map(str, row)).center(20))
```

Slide 5: Stirling Numbers of the Second Kind

Stirling numbers of the second kind count the number of ways to partition a set of n objects into k non-empty subsets.

```python
def stirling_second_kind(n, k):
    if k == 1 or k == n:
        return 1
    return k * stirling_second_kind(n - 1, k) + stirling_second_kind(n - 1, k - 1)

# Example: Calculate S(5,3)
n, k = 5, 3
result = stirling_second_kind(n, k)
print(f"S({n},{k}) = {result}")  # Output: S(5,3) = 25
```

Slide 6: Bell Numbers

Bell numbers count the number of ways to partition a set. We'll implement a function to calculate Bell numbers using Stirling numbers.

```python
def bell_number(n):
    return sum(stirling_second_kind(n, k) for k in range(1, n + 1))

# Calculate and print the first 5 Bell numbers
for i in range(1, 6):
    print(f"B({i}) = {bell_number(i)}")
```

Slide 7: Catalan Numbers

Catalan numbers appear in various counting problems. Let's implement a function to calculate Catalan numbers and explore their applications.

```python
def catalan_number(n):
    return math.comb(2*n, n) // (n + 1)

# Calculate and print the first 5 Catalan numbers
for i in range(5):
    print(f"C({i}) = {catalan_number(i)}")

# Example: Number of valid parentheses expressions
n = 3
print(f"Number of valid parentheses expressions with {n} pairs: {catalan_number(n)}")
```

Slide 8: Generating Functions

Generating functions are powerful tools in enumeration. We'll use SymPy to work with generating functions.

```python
from sympy import symbols, expand

x = symbols('x')

# Example: Generating function for the sequence 1, 1, 1, 1, ...
g = 1 / (1 - x)
print("Expansion of 1 / (1 - x):")
print(expand(g, nterms=5))  # Print first 5 terms
```

Slide 9: Recurrence Relations

Recurrence relations are equations that define a sequence recursively. We'll implement and solve simple recurrence relations.

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Generate and print the first 10 Fibonacci numbers
fib_sequence = [fibonacci(i) for i in range(10)]
print("First 10 Fibonacci numbers:", fib_sequence)
```

Slide 10: Inclusion-Exclusion Principle

The Inclusion-Exclusion Principle is used to calculate the number of elements in the union of multiple sets.

```python
from itertools import combinations

def inclusion_exclusion(universal, *sets):
    result = len(universal)
    for r in range(1, len(sets) + 1):
        for combo in combinations(sets, r):
            intersection = set.intersection(*combo)
            result += (-1)**r * len(intersection)
    return result

# Example: Students taking different subjects
students = set(range(1, 101))  # 100 students
math = set(range(1, 71))       # 70 students take math
physics = set(range(1, 51))    # 50 students take physics
chemistry = set(range(1, 41))  # 40 students take chemistry

result = inclusion_exclusion(students, math, physics, chemistry)
print(f"Number of students taking at least one subject: {result}")
```

Slide 11: Partitions

A partition of a positive integer n is a way of writing n as a sum of positive integers. Let's implement a function to generate all partitions of a number.

```python
def generate_partitions(n):
    def partition(n, max_val, prefix):
        if n == 0:
            yield prefix
        for i in range(min(max_val, n), 0, -1):
            yield from partition(n - i, i, prefix + [i])

    return list(partition(n, n, []))

# Example: Generate all partitions of 5
partitions = generate_partitions(5)
print("Partitions of 5:")
for p in partitions:
    print(f"{5} = {' + '.join(map(str, p))}")
```

Slide 12: Burnside's Lemma

Burnside's Lemma is used in group theory to count the number of orbits of a group action. We'll implement a simple example.

```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def burnside_necklaces(n, k):
    total = sum(k ** (gcd(i, n)) for i in range(1, n + 1))
    return total // n

# Example: Count the number of distinct necklaces with 4 beads and 3 colors
n, k = 4, 3
result = burnside_necklaces(n, k)
print(f"Number of distinct necklaces with {n} beads and {k} colors: {result}")
```

Slide 13: Real-life Example: Seating Arrangements

Let's consider a real-life example of enumeration: seating arrangements at a dinner party.

```python
def dinner_party_seatings(n):
    return factorial(n - 1)

# Example: Calculate seating arrangements for 8 people at a round table
guests = 8
arrangements = dinner_party_seatings(guests)
print(f"Number of seating arrangements for {guests} people: {arrangements}")

# Bonus: Generate a random seating arrangement
import random

def random_seating(guests):
    seats = list(range(1, guests))
    random.shuffle(seats)
    return [1] + seats  # Host always sits at position 1

print("Random seating arrangement:", random_seating(guests))
```

Slide 14: Real-life Example: Password Combinations

Another practical application of enumeration is in calculating the number of possible passwords.

```python
def password_combinations(length, char_types):
    return sum(math.comb(length, i) * (char_types ** i) for i in range(1, length + 1))

# Example: Calculate password combinations for a 8-character password
# with at least one lowercase, one uppercase, one digit, and one special character
length = 8
char_types = 4  # lowercase, uppercase, digits, special characters

combinations = password_combinations(length, char_types)
print(f"Number of possible {length}-character passwords: {combinations:,}")

# Estimate time to brute-force (assuming 1 billion attempts per second)
time_seconds = combinations / 1_000_000_000
print(f"Time to brute-force: {time_seconds:.2f} seconds")
```

Slide 15: Additional Resources

For further exploration of enumeration techniques and combinatorics, consider these resources:

1. "Enumerative Combinatorics" by Richard P. Stanley (Cambridge University Press)
2. "A Course in Enumeration" by Martin Aigner (Springer)
3. "Concrete Mathematics: A Foundation for Computer Science" by Ronald L. Graham, Donald E. Knuth, and Oren Patashnik (Addison-Wesley)
4. ArXiv.org: "Combinatorial Species and Tree-like Structures" by F. Bergeron, G. Labelle, and P. Leroux ([https://arxiv.org/abs/math/9805066](https://arxiv.org/abs/math/9805066))
5. ArXiv.org: "Analytic Combinatorics: A Calculus of Discrete Structures" by Philippe Flajolet and Robert Sedgewick ([https://arxiv.org/abs/1005.0260](https://arxiv.org/abs/1005.0260))

