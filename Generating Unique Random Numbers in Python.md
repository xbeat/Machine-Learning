## Generating Unique Random Numbers in Python
Slide 1: Basic Random Number Generation Without Duplicates Using Lists

Understanding how to generate unique random numbers is crucial for various applications, from sampling to simulation. The simplest approach uses Python's random module with a list comprehension to create a pool of numbers and randomly select from it.

```python
import random

def generate_unique_numbers(start, end, count):
    # Create a pool of numbers and randomly sample from it
    number_pool = list(range(start, end + 1))
    unique_numbers = random.sample(number_pool, count)
    return sorted(unique_numbers)

# Example usage
result = generate_unique_numbers(1, 100, 10)
print(f"10 unique random numbers between 1 and 100: {result}")
```

Slide 2: Set-Based Random Number Generation

Sets provide an alternative approach to generating unique random numbers, leveraging their inherent property of storing unique elements. This method is particularly efficient when dealing with large ranges and preventing duplicates.

```python
import random

def generate_unique_with_set(start, end, count):
    unique_numbers = set()
    while len(unique_numbers) < count:
        unique_numbers.add(random.randint(start, end))
    return sorted(list(unique_numbers))

# Example usage
result = generate_unique_with_set(1, 50, 10)
print(f"10 unique random numbers between 1 and 50: {result}")
```

Slide 3: Cryptographic Random Number Generation

For applications requiring cryptographic security, we use the secrets module to generate unique random numbers. This approach is essential for security-critical applications like token generation or encryption keys.

```python
import secrets
import random

def crypto_unique_numbers(start, end, count):
    number_pool = list(range(start, end + 1))
    return sorted(secrets.SystemRandom().sample(number_pool, count))

# Example usage
result = crypto_unique_numbers(1, 100, 5)
print(f"5 cryptographically secure unique numbers: {result}")
```

Slide 4: Fisher-Yates Shuffle Implementation

The Fisher-Yates shuffle algorithm provides an efficient method for generating unique random numbers through array shuffling. This implementation demonstrates the algorithm's core concepts and its application.

```python
def fisher_yates_selection(start, end, count):
    # Create array of numbers
    arr = list(range(start, end + 1))
    n = len(arr)
    
    # Perform Fisher-Yates shuffle for 'count' elements
    for i in range(count):
        j = random.randint(i, n-1)
        arr[i], arr[j] = arr[j], arr[i]
    
    return sorted(arr[:count])

# Example usage
result = fisher_yates_selection(1, 30, 8)
print(f"8 unique random numbers using Fisher-Yates: {result}")
```

Slide 5: Generator-Based Random Number Implementation

Using Python generators provides a memory-efficient way to generate unique random numbers, especially useful when dealing with large ranges or when numbers need to be generated lazily.

```python
def unique_number_generator(start, end, count):
    seen = set()
    while len(seen) < count:
        num = random.randint(start, end)
        if num not in seen:
            seen.add(num)
            yield num

# Example usage
generator = unique_number_generator(1, 20, 5)
result = sorted(list(generator))
print(f"5 unique random numbers using generator: {result}")
```

Slide 6: Weighted Random Number Generation

In real-world applications, we often need to generate unique random numbers with specific weights or probabilities. This implementation demonstrates how to achieve weighted random selection while maintaining uniqueness.

```python
import numpy as np

def weighted_unique_random(start, end, count, weights=None):
    numbers = list(range(start, end + 1))
    if weights is None:
        weights = [1] * len(numbers)
    
    selected = []
    weights = np.array(weights)
    
    for _ in range(count):
        # Normalize weights
        weights_norm = weights / weights.sum()
        # Select number
        chosen_idx = np.random.choice(len(numbers), p=weights_norm)
        selected.append(numbers[chosen_idx])
        # Remove selected number and its weight
        numbers.pop(chosen_idx)
        weights = np.delete(weights, chosen_idx)
    
    return sorted(selected)

# Example usage
weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
result = weighted_unique_random(1, 10, 5, weights)
print(f"5 weighted unique random numbers: {result}")
```

Slide 7: Time-Based Seeding for Reproducibility

Implementing reproducible random number generation is crucial for testing and validation. This approach uses time-based seeding to create reproducible sequences of unique random numbers.

```python
import time

def time_seeded_unique_random(start, end, count, seed=None):
    if seed is None:
        seed = int(time.time())
    
    random.seed(seed)
    numbers = random.sample(range(start, end + 1), count)
    
    return {
        'numbers': sorted(numbers),
        'seed': seed
    }

# Example usage
result1 = time_seeded_unique_random(1, 100, 5, seed=42)
result2 = time_seeded_unique_random(1, 100, 5, seed=42)
print(f"First generation: {result1}")
print(f"Second generation (same seed): {result2}")
```

Slide 8: Binary Search Tree-Based Generation

A binary search tree implementation provides an interesting approach to generating unique random numbers while maintaining ordering properties and ensuring uniqueness through tree structure.

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BSTRandomGenerator:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        if not self.root:
            self.root = Node(value)
            return True
        
        current = self.root
        while True:
            if value < current.value:
                if not current.left:
                    current.left = Node(value)
                    return True
                current = current.left
            elif value > current.value:
                if not current.right:
                    current.right = Node(value)
                    return True
                current = current.right
            else:
                return False

    def generate_unique_numbers(self, start, end, count):
        numbers = []
        while len(numbers) < count:
            num = random.randint(start, end)
            if self.insert(num):
                numbers.append(num)
        return sorted(numbers)

# Example usage
bst_generator = BSTRandomGenerator()
result = bst_generator.generate_unique_numbers(1, 50, 10)
print(f"10 unique numbers using BST: {result}")
```

Slide 9: Prime Number Random Generation

Generating unique random prime numbers presents a specialized case of unique random number generation, useful in cryptography and number theory applications.

```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def generate_unique_primes(start, end, count):
    primes = [num for num in range(start, end + 1) if is_prime(num)]
    if len(primes) < count:
        raise ValueError("Not enough primes in the given range")
    return sorted(random.sample(primes, count))

# Example usage
try:
    result = generate_unique_primes(1, 100, 5)
    print(f"5 unique random prime numbers: {result}")
except ValueError as e:
    print(f"Error: {e}")
```

Slide 10: Real-World Application: Monte Carlo Simulation

This implementation demonstrates how unique random number generation can be applied to Monte Carlo simulation for estimating π using random point generation in a square.

```python
import math

def monte_carlo_pi(points):
    inside_circle = 0
    unique_points = set()
    
    while len(unique_points) < points:
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        point = (round(x, 6), round(y, 6))
        
        if point not in unique_points:
            unique_points.add(point)
            if math.sqrt(x*x + y*y) <= 1:
                inside_circle += 1
    
    pi_estimate = 4 * inside_circle / points
    return pi_estimate, unique_points

# Example usage
points = 1000
estimated_pi, unique_points = monte_carlo_pi(points)
print(f"π estimate using {points} unique points: {estimated_pi}")
print(f"Actual π: {math.pi}")
print(f"Error: {abs(estimated_pi - math.pi)}")
```

Slide 11: Sliding Window Random Generation

This advanced technique implements a sliding window approach to generate unique random numbers within a moving range, useful for time-series analysis and streaming data applications.

```python
class SlidingWindowRandomGenerator:
    def __init__(self, window_size):
        self.window_size = window_size
        self.history = set()
        
    def generate(self, start, count):
        window_start = start
        window_end = start + self.window_size
        result = []
        
        while len(result) < count:
            # Generate candidate within current window
            candidate = random.randint(window_start, window_end)
            
            # Check if number is unique in history
            if candidate not in self.history:
                self.history.add(candidate)
                result.append(candidate)
                
                # Slide window if needed
                if len(result) % (self.window_size // 2) == 0:
                    window_start += self.window_size // 2
                    window_end += self.window_size // 2
                    
        return sorted(result)

# Example usage
generator = SlidingWindowRandomGenerator(window_size=20)
result = generator.generate(start=1, count=10)
print(f"10 unique numbers using sliding window: {result}")
```

Slide 12: Permutation-Based Random Generation

A permutation-based approach that generates unique random numbers by creating controlled permutations of existing sequences, particularly useful for maintaining specific distribution properties.

```python
def permutation_based_random(start, end, count):
    # Create base sequence
    base_sequence = list(range(start, end + 1))
    
    # Generate permutation parameters
    permutation_params = {
        'shift': random.randint(1, len(base_sequence)),
        'multiply': random.choice([num for num in range(1, len(base_sequence)) 
                                 if math.gcd(num, len(base_sequence)) == 1])
    }
    
    # Apply permutation
    result = []
    used = set()
    current = start
    
    while len(result) < count:
        # Generate next number using permutation formula
        next_num = ((current * permutation_params['multiply'] + 
                     permutation_params['shift']) % (end - start + 1)) + start
        
        if next_num not in used:
            used.add(next_num)
            result.append(next_num)
        current = next_num
    
    return sorted(result)

# Example usage
result = permutation_based_random(1, 50, 15)
print(f"15 unique numbers using permutation: {result}")
```

Slide 13: Real-World Application: Random Sampling for Cross-Validation

This implementation demonstrates how to use unique random number generation for creating cross-validation folds in machine learning applications.

```python
import numpy as np

def create_cv_folds(dataset_size, n_folds):
    # Generate unique indices for the entire dataset
    all_indices = list(range(dataset_size))
    fold_size = dataset_size // n_folds
    
    # Create folds using unique random sampling
    folds = []
    remaining_indices = set(all_indices)
    
    for fold in range(n_folds - 1):
        # Sample indices for current fold
        fold_indices = set(random.sample(list(remaining_indices), fold_size))
        folds.append(sorted(list(fold_indices)))
        remaining_indices -= fold_indices
    
    # Add remaining indices to last fold
    folds.append(sorted(list(remaining_indices)))
    
    return folds

# Example usage with synthetic dataset
dataset_size = 100
n_folds = 5

cv_folds = create_cv_folds(dataset_size, n_folds)
for i, fold in enumerate(cv_folds):
    print(f"Fold {i+1} size: {len(fold)}")
    print(f"First 5 indices in fold {i+1}: {fold[:5]}")
```

Slide 14: Sequence Pattern Generation

Implementation of a pattern-based unique random number generator that creates sequences following specific mathematical patterns while maintaining randomness and uniqueness.

```python
def pattern_based_generator(start, end, count, pattern_type='fibonacci'):
    def fibonacci_pattern(n):
        return int((1 + math.sqrt(5))**n / math.sqrt(5))
    
    def geometric_pattern(n):
        return int(2**n)
    
    patterns = {
        'fibonacci': fibonacci_pattern,
        'geometric': geometric_pattern
    }
    
    pattern_func = patterns.get(pattern_type, fibonacci_pattern)
    
    # Generate pattern-based numbers
    candidates = set()
    n = 0
    while len(candidates) < count * 2:  # Generate extra numbers for flexibility
        num = pattern_func(n) % (end - start + 1) + start
        if start <= num <= end:
            candidates.add(num)
        n += 1
    
    # Randomly select required number of values
    return sorted(random.sample(list(candidates), count))

# Example usage
fib_result = pattern_based_generator(1, 100, 8, 'fibonacci')
geo_result = pattern_based_generator(1, 100, 8, 'geometric')
print(f"Fibonacci pattern: {fib_result}")
print(f"Geometric pattern: {geo_result}")
```

Slide 15: Additional Resources

*   "Random Number Generation Techniques in Machine Learning" - [https://arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)
*   "Analysis of Random Number Generation Methods" - [https://scholar.google.com/](https://scholar.google.com/)
*   "Cryptographic Random Number Generation: A Comprehensive Survey" - [https://scholar.google.com/](https://scholar.google.com/)
*   Suggested search terms for Google Scholar:
    *   "Unique random number generation algorithms"
    *   "Random sampling without replacement methods"
    *   "Efficient unique random number generation"

