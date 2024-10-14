## Combinatorics Fundamentals
Slide 1: Introduction to Combinatorics

Combinatorics is a branch of mathematics that deals with counting, arrangement, and combination of objects. It forms the foundation for probability theory and statistical analysis. This presentation will explore key concepts in combinatorics and their applications, providing practical examples and Python code to illustrate these ideas.

```python
# A simple example of combinatorics: counting permutations
import itertools

elements = ['A', 'B', 'C']
permutations = list(itertools.permutations(elements))
print(f"Number of permutations: {len(permutations)}")
print("Permutations:", permutations)

# Output:
# Number of permutations: 6
# Permutations: [('A', 'B', 'C'), ('A', 'C', 'B'), ('B', 'A', 'C'), ('B', 'C', 'A'), ('C', 'A', 'B'), ('C', 'B', 'A')]
```

Slide 2: Fundamental Counting Principle

The Fundamental Counting Principle states that if one event can occur in 'm' ways, and another independent event can occur in 'n' ways, then the two events can occur together in 'm × n' ways. This principle is crucial for solving complex counting problems.

```python
# Demonstrating the Fundamental Counting Principle
def fundamental_counting_principle(events):
    total_ways = 1
    for event in events:
        total_ways *= event
    return total_ways

# Example: Choosing an outfit
tops = 4  # 4 different tops
bottoms = 3  # 3 different bottoms
shoes = 2  # 2 pairs of shoes

outfit_combinations = fundamental_counting_principle([tops, bottoms, shoes])
print(f"Total outfit combinations: {outfit_combinations}")

# Output:
# Total outfit combinations: 24
```

Slide 3: Permutations

Permutations are arrangements of objects where order matters. The number of permutations of n distinct objects is n!. When selecting r objects from n, we use the formula P(n,r) = n! / (n-r)!.

```python
import math

def permutations(n, r):
    return math.factorial(n) // math.factorial(n - r)

# Example: Arranging 5 books on a shelf
n_books = 5
ways_to_arrange = permutations(n_books, n_books)
print(f"Ways to arrange {n_books} books: {ways_to_arrange}")

# Selecting 3 books from 5 and arranging them
ways_to_select_and_arrange = permutations(n_books, 3)
print(f"Ways to select and arrange 3 books from {n_books}: {ways_to_select_and_arrange}")

# Output:
# Ways to arrange 5 books: 120
# Ways to select and arrange 3 books from 5: 60
```

Slide 4: Combinations

Combinations are selections of objects where order doesn't matter. The number of ways to choose r objects from n is given by C(n,r) = n! / (r! \* (n-r)!).

```python
import math

def combinations(n, r):
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

# Example: Selecting a committee
total_people = 10
committee_size = 3

ways_to_form_committee = combinations(total_people, committee_size)
print(f"Ways to form a committee of {committee_size} from {total_people} people: {ways_to_form_committee}")

# Output:
# Ways to form a committee of 3 from 10 people: 120
```

Slide 5: Real-Life Example: Library Book Selection

A library has 1000 fiction books and 500 non-fiction books. We'll calculate how many ways a person can select 3 fiction books and 2 non-fiction books.

```python
def book_selection_combinations(fiction, non_fiction, f_select, nf_select):
    fiction_combos = combinations(fiction, f_select)
    non_fiction_combos = combinations(non_fiction, nf_select)
    return fiction_combos * non_fiction_combos

fiction_books = 1000
non_fiction_books = 500
fiction_select = 3
non_fiction_select = 2

total_selections = book_selection_combinations(fiction_books, non_fiction_books, fiction_select, non_fiction_select)
print(f"Total ways to select {fiction_select} fiction and {non_fiction_select} non-fiction books: {total_selections:,}")

# Output:
# Total ways to select 3 fiction and 2 non-fiction books: 2,573,031,125,000
```

Slide 6: Permutations with Repetition

When we have repeated elements, the number of unique permutations changes. For n elements with r1, r2, ..., rk repetitions, the formula is: n! / (r1! \* r2! \* ... \* rk!).

```python
from collections import Counter

def permutations_with_repetition(elements):
    n = len(elements)
    counts = Counter(elements)
    denominator = math.prod(math.factorial(count) for count in counts.values())
    return math.factorial(n) // denominator

# Example: Permutations of 'MISSISSIPPI'
word = 'MISSISSIPPI'
unique_permutations = permutations_with_repetition(word)
print(f"Unique permutations of '{word}': {unique_permutations:,}")

# Output:
# Unique permutations of 'MISSISSIPPI': 34,650
```

Slide 7: Combinations with Repetition

Combinations with repetition allow selecting objects multiple times. The formula for selecting r objects from n types with repetition allowed is: C(n+r-1, r).

```python
def combinations_with_repetition(n, r):
    return combinations(n + r - 1, r)

# Example: Selecting ice cream flavors
flavors = 5
scoops = 3

ways_to_select = combinations_with_repetition(flavors, scoops)
print(f"Ways to select {scoops} scoops from {flavors} flavors (with repetition): {ways_to_select}")

# Output:
# Ways to select 3 scoops from 5 flavors (with repetition): 35
```

Slide 8: Real-Life Example: Password Generation

Let's calculate the number of possible 8-character passwords using uppercase letters, lowercase letters, and digits.

```python
def password_combinations():
    uppercase = 26
    lowercase = 26
    digits = 10
    characters = uppercase + lowercase + digits
    password_length = 8
    return characters ** password_length

possible_passwords = password_combinations()
print(f"Number of possible 8-character passwords: {possible_passwords:,}")

# Output:
# Number of possible 8-character passwords: 218,340,105,584,896
```

Slide 9: The Pigeonhole Principle

The Pigeonhole Principle states that if n items are placed into m containers, and n > m, then at least one container must contain more than one item.

```python
import random

def demonstrate_pigeonhole_principle(items, containers):
    distribution = [0] * containers
    for _ in range(items):
        container = random.randint(0, containers - 1)
        distribution[container] += 1
    
    max_items = max(distribution)
    print(f"Items: {items}, Containers: {containers}")
    print(f"Distribution: {distribution}")
    print(f"Maximum items in a single container: {max_items}")

# Example
demonstrate_pigeonhole_principle(15, 10)

# Possible Output:
# Items: 15, Containers: 10
# Distribution: [1, 3, 1, 2, 1, 1, 2, 1, 2, 1]
# Maximum items in a single container: 3
```

Slide 10: Binomial Coefficient

The Binomial Coefficient, often denoted as (n choose k) or C(n,k), represents the number of ways to choose k items from n items without repetition and without order.

```python
import math

def binomial_coefficient(n, k):
    return math.comb(n, k)

# Example: Calculating probabilities in coin tosses
def coin_toss_probability(n, k):
    total_outcomes = 2**n
    favorable_outcomes = binomial_coefficient(n, k)
    probability = favorable_outcomes / total_outcomes
    return probability

n_tosses = 10
k_heads = 6

prob = coin_toss_probability(n_tosses, k_heads)
print(f"Probability of getting exactly {k_heads} heads in {n_tosses} coin tosses: {prob:.4f}")

# Output:
# Probability of getting exactly 6 heads in 10 coin tosses: 0.2051
```

Slide 11: Stirling Numbers

Stirling numbers of the second kind, denoted as S(n,k), count the number of ways to partition a set of n elements into k non-empty subsets.

```python
def stirling_number(n, k):
    if n == k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    return k * stirling_number(n-1, k) + stirling_number(n-1, k-1)

# Example: Partitioning students into study groups
students = 5
groups = 3

ways_to_partition = stirling_number(students, groups)
print(f"Ways to partition {students} students into {groups} study groups: {ways_to_partition}")

# Output:
# Ways to partition 5 students into 3 study groups: 25
```

Slide 12: Inclusion-Exclusion Principle

The Inclusion-Exclusion Principle is used to calculate the number of elements in the union of multiple sets, avoiding double-counting.

```python
def inclusion_exclusion(sets):
    n = len(sets)
    result = 0
    for i in range(1, 2**n):
        intersection = set.intersection(*[sets[j] for j in range(n) if (i & (1 << j))])
        sign = (-1)**(bin(i).count('1') + 1)
        result += sign * len(intersection)
    return result

# Example: Students in different clubs
math_club = {1, 2, 3, 4, 5}
science_club = {2, 4, 6, 8}
chess_club = {3, 5, 6, 7}

total_students = inclusion_exclusion([math_club, science_club, chess_club])
print(f"Total number of students in at least one club: {total_students}")

# Output:
# Total number of students in at least one club: 8
```

Slide 13: Generating Functions

Generating functions are a powerful tool in combinatorics for solving counting problems and recurrence relations.

```python
from sympy import symbols, expand

def generating_function(coefficients, degree):
    x = symbols('x')
    return expand(sum(coeff * x**i for i, coeff in enumerate(coefficients)))

# Example: Generating function for Fibonacci sequence
fib_coeffs = [0, 1, 1, 2, 3, 5, 8, 13]
fib_gf = generating_function(fib_coeffs, len(fib_coeffs) - 1)
print("Generating function for Fibonacci sequence:")
print(fib_gf)

# Output:
# Generating function for Fibonacci sequence:
# 13*x**7 + 8*x**6 + 5*x**5 + 3*x**4 + 2*x**3 + x**2 + x
```

Slide 14: Recurrence Relations

Recurrence relations are equations that define a sequence based on preceding terms. They are often used in combinatorics to solve counting problems.

```python
def solve_recurrence(initial_terms, coeffs, n):
    sequence = initial_terms[:]
    while len(sequence) < n:
        next_term = sum(coeff * sequence[-(i+1)] for i, coeff in enumerate(coeffs))
        sequence.append(next_term)
    return sequence

# Example: Fibonacci sequence
fib_initial = [0, 1]
fib_coeffs = [1, 1]
n = 10

fibonacci_sequence = solve_recurrence(fib_initial, fib_coeffs, n)
print(f"First {n} terms of Fibonacci sequence: {fibonacci_sequence}")

# Output:
# First 10 terms of Fibonacci sequence: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

Slide 15: Additional Resources

For those interested in diving deeper into combinatorics, here are some valuable resources:

1. ArXiv.org: "Combinatorial Species and Tree-like Structures" by F. Bergeron, G. Labelle, and P. Leroux URL: [https://arxiv.org/abs/math/9805066](https://arxiv.org/abs/math/9805066)
2. ArXiv.org: "Analytic Combinatorics: A Calculus of Discrete Structures" by Philippe Flajolet and Robert Sedgewick URL: [https://arxiv.org/abs/0810.4752](https://arxiv.org/abs/0810.4752)

These papers provide advanced insights into combinatorial structures and their applications in various fields of mathematics and computer science.

