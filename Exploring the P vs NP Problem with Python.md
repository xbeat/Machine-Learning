## Exploring the P vs NP Problem with Python
Slide 1: What is P vs NP?

P vs NP is one of the most important unsolved problems in computer science and mathematics. It asks whether every problem whose solution can be quickly verified by a computer can also be solved quickly by a computer.

```python
def is_solution(problem, solution):
    # Quickly verify if the solution is correct
    return verify(problem, solution)

def find_solution(problem):
    # This is the hard part - can we solve it quickly?
    for possible_solution in all_possible_solutions(problem):
        if is_solution(problem, possible_solution):
            return possible_solution
    return None
```

Slide 2: The P Class

P stands for "Polynomial time". Problems in P can be solved by an algorithm in polynomial time relative to the input size.

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Example usage
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = bubble_sort(arr)
print(sorted_arr)
```

Slide 3: The NP Class

NP stands for "Nondeterministic Polynomial time". Problems in NP can have their solutions verified in polynomial time.

```python
import itertools

def is_clique(graph, vertices):
    for v1, v2 in itertools.combinations(vertices, 2):
        if not graph[v1][v2]:
            return False
    return True

def verify_clique(graph, k, clique):
    return len(clique) == k and is_clique(graph, clique)

# Example usage
graph = {
    0: {1: True, 2: True, 3: False},
    1: {0: True, 2: True, 3: False},
    2: {0: True, 1: True, 3: False},
    3: {0: False, 1: False, 2: False}
}
k = 3
clique = [0, 1, 2]
print(verify_clique(graph, k, clique))
```

Slide 4: P ⊆ NP

All problems in P are also in NP, because if we can solve a problem quickly, we can certainly verify its solution quickly.

```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def verify_prime(n, proof):
    # In this case, the proof is just the number itself
    return is_prime(n)

# Example usage
n = 97
proof = n
print(verify_prime(n, proof))
```

Slide 5: NP-Complete Problems

NP-Complete problems are the hardest problems in NP. If any NP-Complete problem can be solved in polynomial time, then P = NP.

```python
def satisfiable(formula):
    variables = set(var for clause in formula for var in clause if var[0] != '!')
    for assignment in itertools.product([True, False], repeat=len(variables)):
        var_dict = dict(zip(variables, assignment))
        if all(any(var_dict.get(var[1:], var[0] != '!') for var in clause) for clause in formula):
            return True
    return False

# Example usage
formula = [{'x', 'y'}, {'!x', 'z'}, {'!y', '!z'}]
print(satisfiable(formula))
```

Slide 6: The Importance of P vs NP

The P vs NP problem has significant implications for cryptography, optimization, and artificial intelligence.

```python
import random
import sympy

def generate_rsa_keys(p, q):
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537  # Commonly used public exponent
    d = sympy.mod_inverse(e, phi)
    return ((n, e), (n, d))

def encrypt(message, public_key):
    n, e = public_key
    return pow(message, e, n)

def decrypt(ciphertext, private_key):
    n, d = private_key
    return pow(ciphertext, d, n)

# Example usage
p, q = sympy.randprime(2**1023, 2**1024), sympy.randprime(2**1023, 2**1024)
public_key, private_key = generate_rsa_keys(p, q)
message = random.randint(1, public_key[0] - 1)
ciphertext = encrypt(message, public_key)
decrypted = decrypt(ciphertext, private_key)
print(f"Original: {message}")
print(f"Decrypted: {decrypted}")
print(f"Successful decryption: {message == decrypted}")
```

Slide 7: Real-Life Example: Traveling Salesman Problem

The Traveling Salesman Problem (TSP) is a classic NP-hard problem with real-world applications in logistics and route planning.

```python
import itertools

def tsp(cities):
    shortest_route = None
    min_distance = float('inf')
    for route in itertools.permutations(cities):
        distance = sum(distance(route[i], route[i+1]) for i in range(len(route)-1))
        distance += distance(route[-1], route[0])  # Return to start
        if distance < min_distance:
            shortest_route = route
            min_distance = distance
    return shortest_route, min_distance

def distance(city1, city2):
    return ((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)**0.5

# Example usage
cities = [(0, 0), (1, 5), (2, 2), (3, 3), (5, 1)]
route, distance = tsp(cities)
print(f"Shortest route: {route}")
print(f"Total distance: {distance}")
```

Slide 8: Real-Life Example: Integer Factorization

Integer factorization is believed to be a hard problem and forms the basis of many cryptographic systems.

```python
def trial_division(n):
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
        if d * d > n:
            if n > 1:
                factors.append(n)
            break
    return factors

# Example usage
n = 1234567890
factors = trial_division(n)
print(f"Factors of {n}: {factors}")
print(f"Verification: {n == reduce(lambda x, y: x*y, factors)}")
```

Slide 9: P vs NP and Optimization

If P = NP, many optimization problems could be solved efficiently, revolutionizing fields like operations research and artificial intelligence.

```python
import pulp

def optimize_production(products, materials, constraints):
    prob = pulp.LpProblem("Production Optimization", pulp.LpMaximize)
    
    # Decision variables
    x = pulp.LpVariable.dicts("product", products, lowBound=0, cat='Integer')
    
    # Objective function
    prob += pulp.lpSum([products[i]['profit'] * x[i] for i in products])
    
    # Constraints
    for m in materials:
        prob += pulp.lpSum([products[i]['materials'][m] * x[i] for i in products]) <= constraints[m]
    
    # Solve the problem
    prob.solve()
    
    return {i: x[i].varValue for i in products}

# Example usage
products = {
    'A': {'profit': 20, 'materials': {'wood': 2, 'metal': 1}},
    'B': {'profit': 30, 'materials': {'wood': 1, 'metal': 3}}
}
materials = ['wood', 'metal']
constraints = {'wood': 100, 'metal': 80}

result = optimize_production(products, materials, constraints)
print("Optimal production quantities:")
for product, quantity in result.items():
    print(f"{product}: {quantity}")
```

Slide 10: Approximation Algorithms

When exact solutions are infeasible, approximation algorithms can provide near-optimal solutions in polynomial time.

```python
import random

def approx_vertex_cover(graph):
    cover = set()
    edges = list(graph.items())
    random.shuffle(edges)
    
    for u, neighbors in edges:
        if u not in cover:
            cover.update([u] + list(neighbors))
    
    return cover

# Example usage
graph = {
    0: {1, 2},
    1: {0, 2, 3},
    2: {0, 1, 3, 4},
    3: {1, 2, 4},
    4: {2, 3}
}

cover = approx_vertex_cover(graph)
print(f"Approximate vertex cover: {cover}")
```

Slide 11: Heuristics and Meta-heuristics

Heuristics and meta-heuristics can find good (but not necessarily optimal) solutions to NP-hard problems in reasonable time.

```python
import random

def simulated_annealing(initial_state, energy, neighbor, T, cooling_rate, iterations):
    current_state = initial_state
    current_energy = energy(current_state)
    
    for _ in range(iterations):
        T *= cooling_rate
        new_state = neighbor(current_state)
        new_energy = energy(new_state)
        
        if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / T):
            current_state = new_state
            current_energy = new_energy
    
    return current_state, current_energy

# Example: Traveling Salesman Problem
def tsp_energy(route):
    return sum(distance(route[i], route[(i+1)%len(route)]) for i in range(len(route)))

def tsp_neighbor(route):
    i, j = random.sample(range(len(route)), 2)
    new_route = route[:]
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

cities = [(0, 0), (1, 5), (2, 2), (3, 3), (5, 1)]
initial_route = cities[:]
random.shuffle(initial_route)

best_route, best_distance = simulated_annealing(initial_route, tsp_energy, tsp_neighbor, T=100, cooling_rate=0.995, iterations=10000)
print(f"Best route found: {best_route}")
print(f"Total distance: {best_distance}")
```

Slide 12: Quantum Computing and P vs NP

Quantum computing offers new approaches to solving NP-hard problems, potentially changing our understanding of computational complexity.

```python
from qiskit import QuantumCircuit, Aer, execute

def grover_search(n, oracle):
    qc = QuantumCircuit(n, n)
    
    # Initialize superposition
    qc.h(range(n))
    
    # Apply Grover operator
    qc.append(oracle, range(n))
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n-1)
    qc.mct(list(range(n-1)), n-1)
    qc.h(n-1)
    qc.x(range(n))
    qc.h(range(n))
    
    # Measure
    qc.measure(range(n), range(n))
    
    # Simulate and get results
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=1000)
    result = job.result()
    counts = result.get_counts(qc)
    return max(counts, key=counts.get)

# Example oracle for f(x) = 1 if x == 11, else 0
def oracle():
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    return qc

result = grover_search(2, oracle())
print(f"Search result: {result}")
```

Slide 13: The Future of P vs NP

As we continue to explore the boundaries of computation, new insights into the P vs NP problem may emerge, potentially reshaping our understanding of efficient problem-solving.

```python
import networkx as nx
import matplotlib.pyplot as plt

def visualize_complexity_classes():
    G = nx.DiGraph()
    G.add_edge("P", "NP")
    G.add_edge("P", "co-NP")
    G.add_edge("NP-complete", "NP")
    G.add_edge("NP-hard", "NP-complete")
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=10, font_weight='bold')
    
    plt.title("Complexity Classes Hierarchy")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_complexity_classes()
```

Slide 14: Additional Resources

For further exploration of the P vs NP problem and computational complexity theory:

1. Arora, S., & Barak, B. (2009). Computational Complexity: A Modern Approach. Cambridge University Press. ArXiv: [https://arxiv.org/abs/0903.4732](https://arxiv.org/abs/0903.4732)
2. Fortnow, L. (2009). The Status of the P Versus NP Problem. Communications of the ACM, 52(9), 78-86. ArXiv: [https://arxiv.org/abs/0904.3004](https://arxiv.org/abs/0904.3004)
3. Aaronson, S. (2017). P ≟ NP. Electronic Colloquium on Computational Complexity, Report No. 4. ArXiv: [https://arxiv.org/abs/1708.04420](https://arxiv.org/abs/1708.04420)

