## Solving the Traveling Salesman Problem
Slide 1: Foundations of TSP Representation

A fundamental aspect of solving the Traveling Salesman Problem is representing cities and distances efficiently. We implement a City class to store coordinates and calculate Euclidean distances, forming the basis for our optimization algorithms.

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import random

@dataclass
class City:
    x: float
    y: float
    id: int

    def distance_to(self, other: 'City') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

# Example usage
cities = [City(random.uniform(0, 100), random.uniform(0, 100), i) for i in range(5)]
distance_matrix = np.zeros((len(cities), len(cities)))

for i, city1 in enumerate(cities):
    for j, city2 in enumerate(cities):
        distance_matrix[i][j] = city1.distance_to(city2)

print("Distance Matrix:")
print(distance_matrix)
```

Slide 2: Path Representation and Evaluation

The path representation in TSP requires efficient data structures to store and manipulate city sequences. The total path length calculation is crucial for evaluating solution quality and guiding optimization algorithms.

```python
class TSPPath:
    def __init__(self, cities: List[City]):
        self.cities = cities
        self.path = list(range(len(cities)))
        
    def calculate_total_distance(self) -> float:
        total_distance = 0
        for i in range(len(self.path)):
            city1 = self.cities[self.path[i]]
            city2 = self.cities[self.path[(i + 1) % len(self.path)]]
            total_distance += city1.distance_to(city2)
        return total_distance
    
    def get_path(self) -> List[int]:
        return self.path.copy()

# Example usage
path = TSPPath(cities)
print(f"Initial path length: {path.calculate_total_distance():.2f}")
```

Slide 3: Pheromone Trail Implementation

The pheromone trail matrix is essential for ACO, representing the learned desirability of paths between cities. The implementation includes initialization and update mechanisms for pheromone levels.

```python
class PheromoneMatrix:
    def __init__(self, num_cities: int, initial_pheromone: float = 1.0):
        self.pheromone = np.full((num_cities, num_cities), initial_pheromone)
        self.evaporation_rate = 0.1
        
    def update(self, path: List[int], quality: float):
        for i in range(len(path)):
            current_city = path[i]
            next_city = path[(i + 1) % len(path)]
            self.pheromone[current_city][next_city] += quality
            self.pheromone[next_city][current_city] += quality
            
    def evaporate(self):
        self.pheromone *= (1 - self.evaporation_rate)

# Example usage
pheromone = PheromoneMatrix(len(cities))
print("Initial pheromone levels:\n", pheromone.pheromone)
```

Slide 4: Ant Implementation

Individual ants in ACO make probabilistic decisions based on pheromone levels and heuristic information. Each ant constructs a complete tour through all cities using a selection mechanism.

```python
class Ant:
    def __init__(self, num_cities: int, alpha: float = 1.0, beta: float = 2.0):
        self.num_cities = num_cities
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Distance importance
        self.tour = []
        self.unvisited = set(range(num_cities))
        
    def select_next_city(self, current_city: int, 
                        pheromone: np.ndarray, 
                        distances: np.ndarray) -> int:
        if not self.unvisited:
            return None
            
        probabilities = []
        for city in self.unvisited:
            pheromone_value = pheromone[current_city][city] ** self.alpha
            distance_value = (1.0 / distances[current_city][city]) ** self.beta
            probabilities.append((city, pheromone_value * distance_value))
            
        total = sum(prob[1] for prob in probabilities)
        normalized_probs = [(city, prob/total) for city, prob in probabilities]
        
        # Roulette wheel selection
        r = random.random()
        cumsum = 0
        for city, prob in normalized_probs:
            cumsum += prob
            if r <= cumsum:
                return city
        return normalized_probs[-1][0]
```

Slide 5: ACO Algorithm Core Implementation

The core ACO algorithm coordinates multiple ants over several iterations, managing pheromone updates and maintaining the best solution found. This implementation showcases the main loop and solution construction process.

```python
class ACO:
    def __init__(self, cities: List[City], num_ants: int = 20, 
                 num_iterations: int = 100):
        self.cities = cities
        self.num_cities = len(cities)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.pheromone = PheromoneMatrix(self.num_cities)
        self.best_path = None
        self.best_distance = float('inf')
        
        # Precompute distance matrix
        self.distances = np.zeros((self.num_cities, self.num_cities))
        for i, city1 in enumerate(cities):
            for j, city2 in enumerate(cities):
                self.distances[i][j] = city1.distance_to(city2)
    
    def solve(self):
        for iteration in range(self.num_iterations):
            ant_solutions = []
            
            # Construct solutions
            for _ in range(self.num_ants):
                ant = Ant(self.num_cities)
                solution = self.construct_solution(ant)
                distance = self.calculate_path_distance(solution)
                ant_solutions.append((solution, distance))
                
                # Update best solution
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = solution.copy()
            
            # Update pheromones
            self.pheromone.evaporate()
            for solution, distance in ant_solutions:
                self.pheromone.update(solution, 1.0/distance)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best distance: {self.best_distance:.2f}")
```

Slide 6: Solution Construction and Evaluation

The construction of solutions by individual ants requires careful implementation of selection probabilities and path building. Here we implement the detailed mechanism for building complete tours.

```python
class ACO:  # continuation
    def construct_solution(self, ant: Ant) -> List[int]:
        current_city = random.randint(0, self.num_cities - 1)
        solution = [current_city]
        unvisited = set(range(self.num_cities)) - {current_city}
        
        while unvisited:
            next_city = self._select_next_city(current_city, unvisited)
            solution.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
            
        return solution
    
    def _select_next_city(self, current_city: int, unvisited: set) -> int:
        probabilities = []
        for city in unvisited:
            pheromone = self.pheromone.pheromone[current_city][city]
            distance = 1.0 / max(self.distances[current_city][city], 1e-10)
            probability = (pheromone ** self.alpha) * (distance ** self.beta)
            probabilities.append((city, probability))
        
        # Normalize probabilities
        total = sum(p[1] for p in probabilities)
        if total == 0:
            return random.choice(list(unvisited))
            
        r = random.random() * total
        cumsum = 0
        for city, prob in probabilities:
            cumsum += prob
            if cumsum >= r:
                return city
        return probabilities[-1][0]
```

Slide 7: Local Search Enhancement

Local search procedures improve ACO solutions through systematic neighborhood exploration. The 2-opt local search is implemented to enhance the quality of solutions found by the ants.

```python
def two_opt_improvement(path: List[int], distances: np.ndarray) -> Tuple[List[int], float]:
    improved = True
    best_distance = calculate_path_distance(path, distances)
    
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path)):
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                new_distance = calculate_path_distance(new_path, distances)
                
                if new_distance < best_distance:
                    path = new_path
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
                
    return path, best_distance

def calculate_path_distance(path: List[int], distances: np.ndarray) -> float:
    return sum(distances[path[i]][path[i+1]] 
              for i in range(len(path)-1)) + distances[path[-1]][path[0]]
```

Slide 8: Real-world Application: City Routing

Implementation of ACO for solving a real-world routing problem using actual city coordinates. This example demonstrates data preprocessing and solution visualization.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Sample data for major US cities
cities_data = {
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'latitude': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484],
    'longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740]
}

def prepare_city_data(data: dict) -> List[City]:
    df = pd.DataFrame(data)
    cities = []
    for idx, row in df.iterrows():
        cities.append(City(row['longitude'], row['latitude'], idx))
    return cities

# Initialize and solve
real_cities = prepare_city_data(cities_data)
aco_solver = ACO(real_cities, num_ants=20, num_iterations=100)
solution = aco_solver.solve()

# Visualize solution
def plot_solution(cities: List[City], path: List[int]):
    plt.figure(figsize=(10, 6))
    x = [city.x for city in cities]
    y = [city.y for city in cities]
    
    # Plot cities
    plt.scatter(x, y, c='red', s=100)
    
    # Plot path
    for i in range(len(path)):
        city1 = cities[path[i]]
        city2 = cities[path[(i + 1) % len(path)]]
        plt.plot([city1.x, city2.x], [city1.y, city2.y], 'b-')
    
    plt.title('ACO Solution for City Routing')
    plt.show()

plot_solution(real_cities, aco_solver.best_path)
```

Slide 9: Performance Metrics Implementation

The evaluation of ACO performance requires comprehensive metrics including solution quality, convergence rate, and computational efficiency. This implementation provides tools for measuring and comparing algorithm performance.

```python
class ACOMetrics:
    def __init__(self):
        self.iteration_history = []
        self.convergence_times = []
        self.solution_quality = []
        self.start_time = None
        
    def track_iteration(self, iteration: int, best_distance: float, 
                       current_distance: float):
        if self.start_time is None:
            self.start_time = time.time()
            
        self.iteration_history.append({
            'iteration': iteration,
            'best_distance': best_distance,
            'current_distance': current_distance,
            'time_elapsed': time.time() - self.start_time
        })
        
    def calculate_convergence_rate(self) -> float:
        improvements = [
            i for i in range(1, len(self.iteration_history))
            if self.iteration_history[i]['best_distance'] < 
               self.iteration_history[i-1]['best_distance']
        ]
        return len(improvements) / len(self.iteration_history)
        
    def plot_convergence(self):
        data = pd.DataFrame(self.iteration_history)
        plt.figure(figsize=(10, 6))
        plt.plot(data['iteration'], data['best_distance'], 'b-', 
                label='Best Distance')
        plt.plot(data['iteration'], data['current_distance'], 'r--', 
                label='Current Distance')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.title('ACO Convergence Plot')
        plt.legend()
        plt.grid(True)
        plt.show()
```

Slide 10: Parallel ACO Implementation

Parallel processing can significantly improve ACO performance for large-scale problems. This implementation uses Python's multiprocessing to distribute ant colony operations across multiple cores.

```python
from multiprocessing import Pool, cpu_count
import time

class ParallelACO(ACO):
    def __init__(self, *args, n_processes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_processes = n_processes or cpu_count()
        
    def parallel_ant_solutions(self, ant_batch: int) -> List[Tuple[List[int], float]]:
        ant = Ant(self.num_cities)
        solutions = []
        for _ in range(ant_batch):
            solution = self.construct_solution(ant)
            distance = self.calculate_path_distance(solution)
            solutions.append((solution, distance))
        return solutions
    
    def solve(self):
        with Pool(self.n_processes) as pool:
            ants_per_process = self.num_ants // self.n_processes
            
            for iteration in range(self.num_iterations):
                # Parallel solution construction
                batch_results = pool.map(
                    self.parallel_ant_solutions,
                    [ants_per_process] * self.n_processes
                )
                
                # Combine results
                ant_solutions = []
                for batch in batch_results:
                    ant_solutions.extend(batch)
                
                # Update best solution
                for solution, distance in ant_solutions:
                    if distance < self.best_distance:
                        self.best_distance = distance
                        self.best_path = solution.copy()
                
                # Update pheromones
                self.pheromone.evaporate()
                for solution, distance in ant_solutions:
                    self.pheromone.update(solution, 1.0/distance)
                    
        return self.best_path, self.best_distance
```

Slide 11: Advanced Pheromone Strategies

Enhanced pheromone management strategies can improve convergence and solution quality. This implementation includes max-min pheromone limits and rank-based updates.

```python
class AdvancedPheromoneMatrix(PheromoneMatrix):
    def __init__(self, num_cities: int, 
                 min_pheromone: float = 0.1,
                 max_pheromone: float = 5.0):
        super().__init__(num_cities)
        self.min_pheromone = min_pheromone
        self.max_pheromone = max_pheromone
        
    def rank_based_update(self, solutions: List[Tuple[List[int], float]], 
                         max_rank: int = 6):
        # Sort solutions by quality
        ranked_solutions = sorted(solutions, 
                                key=lambda x: x[1])[:max_rank]
        
        # Apply ranked updates
        for rank, (solution, distance) in enumerate(ranked_solutions):
            weight = max_rank - rank
            self.update(solution, weight/distance)
            
        # Enforce pheromone limits
        self.pheromone = np.clip(
            self.pheromone,
            self.min_pheromone,
            self.max_pheromone
        )
        
    def adaptive_evaporation(self, iteration: int, max_iterations: int):
        # Dynamic evaporation rate
        self.evaporation_rate = 0.1 + 0.1 * (iteration / max_iterations)
        self.evaporate()
```

Slide 12: Real-world Application: Warehouse Routing

Implementation of ACO for optimizing picking routes in a warehouse environment, demonstrating practical application with obstacles and restricted movements.

```python
class WarehouseEnvironment:
    def __init__(self, width: int, height: int, obstacles: List[Tuple[int, int]]):
        self.width = width
        self.height = height
        self.obstacles = set(obstacles)
        self.grid = np.zeros((height, width))
        for x, y in obstacles:
            self.grid[y][x] = 1
            
    def get_valid_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.width and 
                0 <= new_y < self.height and 
                (new_x, new_y) not in self.obstacles):
                neighbors.append((new_x, new_y))
        return neighbors

class WarehouseACO(ACO):
    def __init__(self, environment: WarehouseEnvironment, 
                 pickup_points: List[Tuple[int, int]], *args, **kwargs):
        self.env = environment
        self.pickup_points = pickup_points
        super().__init__([City(x, y, i) for i, (x, y) in enumerate(pickup_points)],
                        *args, **kwargs)
        
    def calculate_path_distance(self, path: List[int]) -> float:
        total_distance = 0
        for i in range(len(path) - 1):
            start = self.pickup_points[path[i]]
            end = self.pickup_points[path[i + 1]]
            distance = self._calculate_grid_distance(start, end)
            total_distance += distance
        return total_distance
    
    def _calculate_grid_distance(self, start: Tuple[int, int], 
                               end: Tuple[int, int]) -> float:
        # A* pathfinding implementation
        from heapq import heappush, heappop
        
        def heuristic(a, b):
            return abs(b[0] - a[0]) + abs(b[1] - a[1])
        
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heappop(frontier)[1]
            
            if current == end:
                break
                
            for next_pos in self.env.get_valid_neighbors(current):
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(next_pos, end)
                    heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        return cost_so_far.get(end, float('inf'))

# Example usage
warehouse = WarehouseEnvironment(20, 20, [(5, 5), (5, 6), (6, 5), (6, 6)])
pickup_points = [(1, 1), (18, 1), (1, 18), (18, 18), (10, 10)]
solver = WarehouseACO(warehouse, pickup_points, num_ants=20, num_iterations=100)
solution = solver.solve()
```

Slide 13: Results Analysis and Visualization

A comprehensive suite of visualization and analysis tools for understanding ACO performance and solution characteristics across different problem instances.

```python
class ACOAnalyzer:
    def __init__(self, solutions_history: List[Tuple[List[int], float]]):
        self.solutions = solutions_history
        self.best_solutions = self._extract_best_solutions()
        
    def _extract_best_solutions(self):
        best_distance = float('inf')
        best_solutions = []
        
        for solution, distance in self.solutions:
            if distance < best_distance:
                best_distance = distance
                best_solutions.append({
                    'solution': solution,
                    'distance': distance,
                    'iteration': len(best_solutions)
                })
        return best_solutions
    
    def plot_solution_distribution(self):
        distances = [sol[1] for sol in self.solutions]
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=50, density=True)
        plt.axvline(min(distances), color='r', linestyle='--',
                   label='Best Solution')
        plt.xlabel('Solution Distance')
        plt.ylabel('Density')
        plt.title('Distribution of Solution Quality')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def generate_report(self):
        best_solution = min(self.solutions, key=lambda x: x[1])
        avg_distance = np.mean([sol[1] for sol in self.solutions])
        std_distance = np.std([sol[1] for sol in self.solutions])
        
        report = {
            'best_distance': best_solution[1],
            'average_distance': avg_distance,
            'std_distance': std_distance,
            'total_solutions': len(self.solutions),
            'improvement_rate': len(self.best_solutions) / len(self.solutions)
        }
        
        return report

# Example usage
analyzer = ACOAnalyzer(solver.solution_history)
analyzer.plot_solution_distribution()
print("Performance Report:", analyzer.generate_report())
```

Slide 14: Additional Resources

*   ArXiv:2104.09844 - "Recent Advances in Ant Colony Optimization for Solving the Traveling Salesman Problem" [https://arxiv.org/abs/2104.09844](https://arxiv.org/abs/2104.09844)
*   ArXiv:1904.08694 - "A Comprehensive Survey of Ant Colony Optimization Methods for Routing Problems" [https://arxiv.org/abs/1904.08694](https://arxiv.org/abs/1904.08694)
*   ArXiv:2003.05702 - "Hybrid Ant Colony Optimization: A Review of Current Trends and Future Directions" [https://arxiv.org/abs/2003.05702](https://arxiv.org/abs/2003.05702)
*   ArXiv:1910.07793 - "Performance Analysis of Different ACO Variants for Dynamic TSP" [https://arxiv.org/abs/1910.07793](https://arxiv.org/abs/1910.07793)
*   ArXiv:2012.15740 - "Multi-objective Ant Colony Optimization: A Systematic Review" [https://arxiv.org/abs/2012.15740](https://arxiv.org/abs/2012.15740)

