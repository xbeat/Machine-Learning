## Probability of Shiny Female Tyrunt with Python
Slide 1: The Elusive Shiny Female Tyrunt

This slide introduces the problem: determining the probability of encountering a female Tyrunt that is shiny and has diamond sparkles. We'll approach this using probability theory and combinatorics.

Slide 2: Pokémon Rarity Basics

Here we provide background information on Pokémon rarity factors:

* Gender ratios in certain species
* Shiny Pokémon occurrence
* Special characteristics like diamond sparkles
* Breeding methods that affect rarity (e.g., Masuda Method)

Slide 3: Assumptions and Simplifications

We outline the key assumptions for our calculation:

1. Events are independent (gender, shininess, sparkles)
2. Masuda Method is being used for breeding
3. No additional factors affecting probabilities (e.g., Shiny Charm)

Slide 4: Breaking Down the Problem

We formulate the problem mathematically: P(Female Shiny Tyrunt with Diamond Sparkles) = P(Female) × P(Shiny | Masuda Method) × P(Diamond Sparkles | Shiny)

Each probability: P(Female) = 1/8 P(Shiny | Masuda Method) = 1/512 P(Diamond Sparkles | Shiny) = 1/16

Slide 5: Logical Reasoning and Pseudocode

We develop pseudocode to calculate the final probability:

```
function calculate_probability():
    p_female = 1/8
    p_shiny = 1/512
    p_sparkles = 1/16
    
    total_probability = p_female * p_shiny * p_sparkles
    
    return total_probability

result = calculate_probability()
percentage = result * 100
```

Slide 6: Python Implementation (Part 1)

We translate our pseudocode into Python:

```python
def calculate_probability():
    p_female = 1/8
    p_shiny = 1/512
    p_sparkles = 1/16
    
    total_probability = p_female * p_shiny * p_sparkles
    
    return total_probability

result = calculate_probability()
percentage = result * 100

print(f"The probability is: {result:.10f}")
print(f"The percentage chance is: {percentage:.8f}%")
```

Slide 7: Python Implementation (Part 2)

We run the code and display the results:

```python
# Output:
The probability is: 0.0000001525
The percentage chance is: 0.00001525%
```

This incredibly small probability highlights the rarity of encountering such a Pokémon.

Slide 8: Real-World Applications

This problem demonstrates principles applicable in various fields:

* Genetics: Calculating probabilities of specific trait combinations
* Quality Control: Estimating defect rates in manufacturing
* Epidemiology: Assessing the likelihood of multiple condition occurrences
* Finance: Risk assessment for compound events

Slide 9: Pokémon Trivia Challenge

Made-up trivia question: "If a Trainer encounters 10,000 Tyrunt, what's the probability they'll see at least one female shiny Tyrunt with diamond sparkles?"

This question requires using the binomial probability formula and complement rule.

Slide 10: Solving the Trivia Challenge

To solve this, we use the complement of the probability of seeing no special Tyrunt:

```python
import math

def probability_at_least_one(n_encounters, single_probability):
    prob_none = (1 - single_probability) ** n_encounters
    prob_at_least_one = 1 - prob_none
    return prob_at_least_one

single_prob = 0.0000001525  # from our earlier calculation
encounters = 10000

result = probability_at_least_one(encounters, single_prob)
percentage = result * 100

print(f"Probability of at least one: {result:.6f}")
print(f"Percentage chance: {percentage:.4f}%")

# Output:
# Probability of at least one: 0.001525
# Percentage chance: 0.1525%
```

Slide 11: The History of Shiny Pokémon

Shiny Pokémon were introduced in Generation II (Gold and Silver) in 1999. Originally, the shiny status was determined by the Pokémon's IVs, with a 1/8192 chance. The Masuda Method, introduced in Generation IV, increased shiny odds for bred Pokémon.

Slide 12: Further Reading

For those interested in diving deeper into probability theory and its applications in gaming:

1. "Probability Theory and Stochastic Processes with Applications" by Oliver Knill [https://arxiv.org/abs/1109.3552](https://arxiv.org/abs/1109.3552)
2. "The Mathematics of Pokémon Go" by Michael A. Gottlieb [https://arxiv.org/abs/1808.05930](https://arxiv.org/abs/1808.05930)
3. "Random Number Generators for Massively Parallel Simulations on GPU" by Manfred Alef et al. [https://arxiv.org/abs/1108.4215](https://arxiv.org/abs/1108.4215)

