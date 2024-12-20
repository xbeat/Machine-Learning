## Genetic Algorithms for Image Blending in Python
Slide 1: Introduction to Genetic Algorithms and Image Combination

Genetic algorithms are optimization techniques inspired by natural selection. In the context of image combination, they can be used to evolve and blend multiple images into a single, unique result. This process involves encoding image properties as genes, creating a population of potential solutions, and iteratively improving them through selection, crossover, and mutation operations.

```python
import numpy as np
from PIL import Image

def load_image(path):
    return np.array(Image.open(path))

def save_image(array, path):
    Image.fromarray(array.astype('uint8')).save(path)

# Load two images
image1 = load_image('image1.jpg')
image2 = load_image('image2.jpg')

# Example of a simple combination (average)
combined = (image1 + image2) // 2

save_image(combined, 'combined.jpg')
```

Slide 2: Encoding Images as Chromosomes

To apply genetic algorithms to image combination, we need to represent images as chromosomes. One approach is to flatten the image array into a 1D vector, where each pixel's RGB values become genes in our chromosome.

```python
def image_to_chromosome(image):
    return image.flatten()

def chromosome_to_image(chromosome, shape):
    return chromosome.reshape(shape)

# Convert images to chromosomes
chromosome1 = image_to_chromosome(image1)
chromosome2 = image_to_chromosome(image2)

# Create an initial population
population_size = 50
population = [np.random.randint(0, 256, size=chromosome1.shape) for _ in range(population_size)]
```

Slide 3: Fitness Function

The fitness function evaluates how well each individual in the population meets our desired criteria. For image combination, we might want to preserve features from both original images while creating a visually appealing result.

```python
def fitness(chromosome, target1, target2):
    # Calculate similarity to both target images
    similarity1 = np.sum(np.abs(chromosome - target1))
    similarity2 = np.sum(np.abs(chromosome - target2))
    
    # Combine similarities (lower is better)
    return -(similarity1 + similarity2)

# Evaluate fitness for each individual in the population
fitnesses = [fitness(ind, chromosome1, chromosome2) for ind in population]
```

Slide 4: Selection

Selection chooses individuals from the population to be parents for the next generation. We'll use tournament selection, where we randomly choose a subset of individuals and select the best one.

```python
def tournament_selection(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        tournament = np.random.choice(len(population), tournament_size, replace=False)
        winner = max(tournament, key=lambda i: fitnesses[i])
        selected.append(population[winner])
    return selected

# Select parents for the next generation
parents = tournament_selection(population, fitnesses)
```

Slide 5: Crossover

Crossover combines genetic information from two parents to create offspring. We'll implement a simple single-point crossover.

```python
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Create offspring through crossover
offspring = []
for i in range(0, len(parents), 2):
    child1, child2 = crossover(parents[i], parents[i+1])
    offspring.extend([child1, child2])
```

Slide 6: Mutation

Mutation introduces small random changes to individuals, maintaining genetic diversity and preventing premature convergence.

```python
def mutate(chromosome, mutation_rate=0.01):
    mask = np.random.random(chromosome.shape) < mutation_rate
    mutation = np.random.randint(0, 256, size=chromosome.shape)
    return np.where(mask, mutation, chromosome)

# Apply mutation to offspring
mutated_offspring = [mutate(child) for child in offspring]
```

Slide 7: Replacement

After creating a new generation, we replace the old population with the new one. We can also implement elitism, where the best individuals from the previous generation are preserved.

```python
def elitism(population, fitnesses, elite_size):
    elite_indices = np.argsort(fitnesses)[-elite_size:]
    return [population[i] for i in elite_indices]

# Implement elitism and create the new generation
elite_size = 2
elite = elitism(population, fitnesses, elite_size)
new_population = elite + mutated_offspring[:len(population)-elite_size]
```

Slide 8: Main Genetic Algorithm Loop

Now we'll put all the components together into the main genetic algorithm loop.

```python
def genetic_algorithm(image1, image2, generations=100):
    shape = image1.shape
    chromosome1 = image_to_chromosome(image1)
    chromosome2 = image_to_chromosome(image2)
    
    population = [np.random.randint(0, 256, size=chromosome1.shape) for _ in range(50)]
    
    for generation in range(generations):
        fitnesses = [fitness(ind, chromosome1, chromosome2) for ind in population]
        parents = tournament_selection(population, fitnesses)
        offspring = []
        for i in range(0, len(parents), 2):
            child1, child2 = crossover(parents[i], parents[i+1])
            offspring.extend([child1, child2])
        mutated_offspring = [mutate(child) for child in offspring]
        elite = elitism(population, fitnesses, 2)
        population = elite + mutated_offspring[:len(population)-2]
    
    best_individual = max(population, key=lambda ind: fitness(ind, chromosome1, chromosome2))
    return chromosome_to_image(best_individual, shape)

# Run the genetic algorithm
result = genetic_algorithm(image1, image2)
save_image(result, 'result.jpg')
```

Slide 9: Visualizing the Evolution

To better understand the genetic algorithm's progress, we can visualize intermediate results throughout the generations.

```python
import matplotlib.pyplot as plt

def visualize_evolution(image1, image2, generations=100, interval=10):
    shape = image1.shape
    chromosome1 = image_to_chromosome(image1)
    chromosome2 = image_to_chromosome(image2)
    
    population = [np.random.randint(0, 256, size=chromosome1.shape) for _ in range(50)]
    
    fig, axes = plt.subplots(1, generations//interval + 1, figsize=(20, 4))
    axes[0].imshow(image1)
    axes[0].set_title('Original 1')
    
    for generation in range(generations):
        fitnesses = [fitness(ind, chromosome1, chromosome2) for ind in population]
        parents = tournament_selection(population, fitnesses)
        offspring = []
        for i in range(0, len(parents), 2):
            child1, child2 = crossover(parents[i], parents[i+1])
            offspring.extend([child1, child2])
        mutated_offspring = [mutate(child) for child in offspring]
        elite = elitism(population, fitnesses, 2)
        population = elite + mutated_offspring[:len(population)-2]
        
        if generation % interval == 0:
            best_individual = max(population, key=lambda ind: fitness(ind, chromosome1, chromosome2))
            axes[generation//interval + 1].imshow(chromosome_to_image(best_individual, shape))
            axes[generation//interval + 1].set_title(f'Gen {generation}')
    
    plt.tight_layout()
    plt.show()

visualize_evolution(image1, image2)
```

Slide 10: Real-Life Example: Blending Landscapes

Genetic algorithms for image combination can be used to create unique landscape images by blending elements from multiple photographs. This technique is particularly useful in digital art and graphic design.

```python
def blend_landscapes(landscape1, landscape2):
    # Load landscape images
    img1 = load_image(landscape1)
    img2 = load_image(landscape2)
    
    # Run genetic algorithm
    blended = genetic_algorithm(img1, img2, generations=200)
    
    # Save and display result
    save_image(blended, 'blended_landscape.jpg')
    plt.imshow(blended)
    plt.title('Blended Landscape')
    plt.axis('off')
    plt.show()

blend_landscapes('mountain.jpg', 'forest.jpg')
```

Slide 11: Real-Life Example: Character Design

In game development or animation, genetic algorithms can be used to generate new character designs by combining features from existing characters.

```python
def evolve_character(character1, character2):
    # Load character images
    char1 = load_image(character1)
    char2 = load_image(character2)
    
    # Define a custom fitness function for character design
    def character_fitness(chromosome, target1, target2):
        similarity = np.sum(np.abs(chromosome - target1)) + np.sum(np.abs(chromosome - target2))
        uniqueness = np.sum(np.abs(chromosome - target1) * np.abs(chromosome - target2))
        return -similarity + uniqueness * 0.5
    
    # Run genetic algorithm with custom fitness function
    evolved = genetic_algorithm(char1, char2, generations=300)
    
    # Save and display result
    save_image(evolved, 'evolved_character.jpg')
    plt.imshow(evolved)
    plt.title('Evolved Character')
    plt.axis('off')
    plt.show()

evolve_character('character1.jpg', 'character2.jpg')
```

Slide 12: Optimizing Performance

As image sizes increase, the genetic algorithm can become computationally expensive. We can optimize performance using vectorized operations and parallel processing.

```python
import multiprocessing

def parallel_fitness(population, target1, target2):
    with multiprocessing.Pool() as pool:
        return pool.starmap(fitness, [(ind, target1, target2) for ind in population])

def vectorized_mutation(population, mutation_rate=0.01):
    mask = np.random.random(population.shape) < mutation_rate
    mutation = np.random.randint(0, 256, size=population.shape)
    return np.where(mask, mutation, population)

# Usage in the main loop
population = np.array(population)
fitnesses = parallel_fitness(population, chromosome1, chromosome2)
mutated_offspring = vectorized_mutation(np.array(offspring))
```

Slide 13: Extending the Algorithm

We can extend our genetic algorithm to handle more complex image combination tasks, such as style transfer or image inpainting.

```python
def style_transfer_fitness(chromosome, content, style):
    # Simplified style transfer fitness function
    content_loss = np.sum(np.abs(chromosome - content))
    style_loss = np.sum(np.abs(np.corrcoef(chromosome) - np.corrcoef(style)))
    return -(content_loss + style_loss * 0.001)

def inpainting_fitness(chromosome, target, mask):
    # Fitness function for image inpainting
    reconstruction_loss = np.sum(np.abs((chromosome * mask) - (target * mask)))
    coherence_loss = np.sum(np.abs(np.diff(chromosome)))
    return -(reconstruction_loss + coherence_loss * 0.1)

# These fitness functions can be used in place of the original fitness function
# in the genetic_algorithm function to perform style transfer or inpainting tasks
```

Slide 14: Additional Resources

For more information on genetic algorithms and their applications in image processing, refer to the following resources:

1. "A Survey of Evolutionary Algorithms for Image Enhancement" by S. Maity et al. (arXiv:1910.08141)
2. "Genetic Algorithms in Image Processing: A Survey" by K. S. Desale and R. Ade (arXiv:1508.00102)
3. "Deep Learning and Evolutionary Computation for Digital Art Creation" by A. Elgammal et al. (arXiv:1711.10957)

These papers provide in-depth discussions on various aspects of using genetic algorithms for image processing tasks and can serve as excellent starting points for further exploration of the topic.

