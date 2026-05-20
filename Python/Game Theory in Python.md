## Game Theory in Python

Slide 1: Game Theory in Python

Game theory is a mathematical framework used to analyze strategic interactions between rational decision-makers. It helps understand the behavior of individuals or groups in situations where their actions affect one another. Python provides a powerful toolset for implementing game theory concepts and simulating various scenarios.

Slide 2: Normal-Form Games

Normal-form games represent strategic situations where players make simultaneous decisions. They are typically represented using payoff matrices. The Python library, Nashpy, provides tools for analyzing and solving normal-form games.

Code Example:

```python
import nashpy as nash

# Define the payoff matrices
payoff_matrix_player_1 = [[3, 0], [5, 1]]
payoff_matrix_player_2 = [[3, 5], [0, 1]]

# Create the game
game = nash.Game(payoff_matrix_player_1, payoff_matrix_player_2)

# Find the Nash equilibria
equilibria = game.support_enumeration()
print(equilibria)
```

Slide 3: Extensive-Form Games

Extensive-form games represent situations where players make sequential decisions, with each player aware of the previous decisions made by other players. The Python library, Axelrod, provides tools for simulating and analyzing these types of games.

Code Example:

```python
import axelrod as axl

# Define the players
player_1 = axl.Cooperator()
player_2 = axl.Defector()

# Create the tournament
tournament = axl.Tournament(players=[player_1, player_2], turns=10, repetitions=5)

# Run the tournament
results = tournament.play()

# Print the results
for player_index, player_scores in results.scores.items():
    player = tournament.players[player_index]
    print(f"{player.name}: {player_scores}")
```

Slide 4: Prisoner's Dilemma

The Prisoner's Dilemma is a classic example of a game that demonstrates the conflict between individual and group interests. It involves two prisoners who must decide whether to cooperate or defect, and their payoffs depend on both players' choices.

Code Example:

```python
import numpy as np

# Define the payoff matrix
payoff_matrix = np.array([[3, 0], [5, 1]])

# Define the players' choices
choices = ['Cooperate', 'Defect']

# Simulate the game
player_1_choice = 'Cooperate'  # or 'Defect'
player_2_choice = 'Defect'  # or 'Cooperate'

player_1_payoff = payoff_matrix[choices.index(player_1_choice)][choices.index(player_2_choice)]
player_2_payoff = payoff_matrix[choices.index(player_2_choice)][choices.index(player_1_choice)]

print(f"Player 1's payoff: {player_1_payoff}")
print(f"Player 2's payoff: {player_2_payoff}")
```

Slide 5: Evolutionary Game Theory

Evolutionary game theory studies how strategies evolve over time in a population of individuals who interact strategically. It incorporates principles from biology and evolution to model the dynamics of strategy adoption.

Code Example:

```python
import numpy as np
import matplotlib.pyplot as plt

# Define the payoff matrix
payoff_matrix = np.array([[3, 0], [5, 1]])

# Define the population sizes and initial strategies
population_size = 100
initial_cooperators = 50
initial_defectors = population_size - initial_cooperators

# Simulate the evolutionary process
generations = 100
cooperators = [initial_cooperators]
defectors = [initial_defectors]

for generation in range(generations):
    # Calculate fitness scores
    fitness_cooperators = 3 * cooperators[-1] / population_size + 0 * defectors[-1] / population_size
    fitness_defectors = 5 * cooperators[-1] / population_size + 1 * defectors[-1] / population_size

    # Update population sizes
    new_cooperators = int(population_size * fitness_cooperators / (fitness_cooperators + fitness_defectors))
    new_defectors = population_size - new_cooperators

    cooperators.append(new_cooperators)
    defectors.append(new_defectors)

# Plot the results
plt.plot(cooperators, label='Cooperators')
plt.plot(defectors, label='Defectors')
plt.xlabel('Generation')
plt.ylabel('Population Size')
plt.legend()
plt.show()
```

Slide 6: Auction Theory

Auction theory is a branch of game theory that studies how people act in auction markets and explores optimal bidding strategies. Python provides libraries like PyStan and PyMC3 for implementing and simulating various auction models.

Code Example:

```python
import pystan

# Define the auction model
model_code = """
data {
    int<lower=1> N;  // Number of bidders
    vector[N] values;  // Bidders' values
}
parameters {
    vector<lower=0>[N] bids;  // Bidders' bids
}
model {
    for (i in 1:N) {
        bids[i] ~ uniform(0, values[i]);  // Bids are uniform between 0 and value
    }
}
generated quantities {
    int winner;
    real highest_bid;
    highest_bid = max(bids);
    winner = max_index(bids);
}
"""

# Define the data
bidders_values = [10, 15, 20, 25]
data = {'N': len(bidders_values), 'values': bidders_values}

# Instantiate and sample the model
sm = pystan.StanModel(model_code=model_code)
fit = sm.sampling(data=data, iter=1000, chains=4)

# Print the results
print(fit.summary())
```

Slide 7: Cooperative Game Theory

Cooperative game theory focuses on how groups of players can form coalitions and how they should distribute the resulting payoffs. The Python library, PyCooperativeGame, provides tools for analyzing and solving cooperative games.

Code Example:

```python
from cooperativegame import CooperativeGame

# Define the characteristic function
characteristic_function = {
    frozenset({}): 0,
    frozenset({1}): 10,
    frozenset({2}): 20,
    frozenset({3}): 30,
    frozenset({1, 2}): 45,
    frozenset({1, 3}): 55,
    frozenset({2, 3}): 65,
    frozenset({1, 2, 3}): 80
}

# Create the cooperative game
game = CooperativeGame(characteristic_function)

# Find the Shapley value
shapley_value = game.shapley_value()
print(shapley_value)
```

Slide 8: Bargaining Theory

Bargaining theory focuses on the study of situations where two or more parties need to cooperate and reach an agreement, but their interests may differ. It analyzes strategies and outcomes in various bargaining scenarios.

Code Example:

```python
import numpy as np

# Define the players' utility functions
player_1_utility = lambda x, y: 2 * x + y
player_2_utility = lambda x, y: x + 3 * y

# Define the bargaining set
X = np.linspace(0, 10, 101)
Y = np.linspace(0, 10, 101)

# Find the Pareto-optimal frontier
pareto_frontier = []
for x in X:
    for y in Y:
        if player_1_utility(x, y) >= player_1_utility(10, 0) and player_2_utility(x, y) >= player_2_utility(0, 10):
            pareto_frontier.append((x, y))

# Print the Pareto-optimal frontier
print("Pareto-optimal frontier:")
for point in pareto_frontier:
    print(point)
```

Slide 9: Mechanism Design

Mechanism design is a subdiscipline of game theory that explores how to implement desirable system-wide objectives by designing appropriate incentive structures. It involves creating rules and protocols that incentivize rational agents to behave in a way that achieves the desired outcome.

Code Example:

```python
import numpy as np

# Define the agents' valuations
valuations = [10, 15, 20]

# Define the mechanism (second-price auction)
def second_price_auction(bids):
    max_bid = max(bids)
    winner = bids.index(max_bid)
    second_highest_bid = sorted(bids, reverse=True)[1]
    
    payments = [0] * len(bids)
    payments[winner] = second_highest_bid
    
    return winner, payments

# Simulate the auction
bids = [8, 12, 18]  # Agents' bids
winner, payments = second_price_auction(bids)

print(f"Winner: Agent {winner + 1}")
print(f"Payments: {payments}")
```

Slide 10: Voting Theory

Voting theory is a subset of game theory that studies and analyzes various voting systems and their properties, such as fairness, strategy-proofness, and paradoxes. It helps understand the implications of different voting rules and their potential vulnerabilities.

Code Example:

```python
from itertools import product

# Define the voters and their preferences
voters = [
    [1, 2, 3],  # Voter 1's preferences
    [2, 3, 1],  # Voter 2's preferences
    [3, 1, 2]   # Voter 3's preferences
]
candidates = set(range(1, 4))  # Candidates {1, 2, 3}

# Implement the Plurality voting rule
def plurality_vote(voters):
    scores = {candidate: 0 for candidate in candidates}
    for voter in voters:
        top_choice = voter[0]
        scores[top_choice] += 1
    winner = max(scores, key=scores.get)
    return winner

# Simulate the election
winner = plurality_vote(voters)
print(f"Winner (Plurality): {winner}")
```

Slide 11: Network Games

Network games are a class of games where players are represented as nodes in a network, and their payoffs depend not only on their own actions but also on the actions of their neighbors in the network. These games are useful for studying social and economic phenomena in networked systems.

Code Example:

```python
import networkx as nx

# Create a network
G = nx.Graph()
G.add_nodes_from(range(5))
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])

# Define the payoff function
def payoff(G, strategy):
    payoffs = {node: 0 for node in G.nodes()}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if strategy[node] == 1:  # Cooperate
            payoffs[node] += len(neighbors)
            for neighbor in neighbors:
                if strategy[neighbor] == 1:
                    payoffs[node] += 1
        else:  # Defect
            for neighbor in neighbors:
                if strategy[neighbor] == 1:
                    payoffs[node] += 2
    return payoffs

# Simulate a strategy
strategy = {0: 1, 1: 0, 2: 1, 3: 0, 4: 1}
payoffs = payoff(G, strategy)
print(payoffs)
```

Slide 12: Algorithmic Game Theory

Algorithmic game theory combines game theory with computational complexity theory and algorithm design. It focuses on developing efficient algorithms for solving game-theoretic problems, particularly in scenarios where the number of players or strategies is large.

Code Example:

```python
import numpy as np
from scipy.optimize import linprog

# Define the payoff matrices
payoff_matrix_player_1 = [[3, 0], [5, 1]]
payoff_matrix_player_2 = [[3, 5], [0, 1]]

# Convert payoff matrices to vectors
payoff_vector_player_1 = np.array(payoff_matrix_player_1).flatten()
payoff_vector_player_2 = np.array(payoff_matrix_player_2).flatten()

# Define the constraints
A_ineq = np.vstack((np.eye(4), -np.eye(4)))
b_ineq = np.zeros(8)

# Define the objective function
c = -np.concatenate((payoff_vector_player_1, payoff_vector_player_2))

# Solve the linear program
res = linprog(-c, A_ineq=A_ineq, b_ineq=b_ineq, bounds=(0, 1))

# Print the equilibrium strategies
equilibrium_strategy_player_1 = res.x[:2]
equilibrium_strategy_player_2 = res.x[2:]

print(f"Equilibrium strategy for Player 1: {equilibrium_strategy_player_1}")
print(f"Equilibrium strategy for Player 2: {equilibrium_strategy_player_2}")
```

This covers various topics in game theory, including normal-form games, extensive-form games, evolutionary game theory, auction theory, cooperative game theory, bargaining theory, mechanism design, voting theory, network games, and algorithmic game theory. Each slide provides a brief description of the topic, along with a Python code example to illustrate the concepts.

## Meta
Mastering Game Theory with Python: Unlocking Strategic Insights

Dive into the fascinating world of game theory and unravel the intricacies of strategic decision-making with Python. This comprehensive TikTok series explores key concepts, from the Prisoner's Dilemma to evolutionary game theory, auction theory, and beyond. Gain a deep understanding of game theory principles through concise explanations and practical code examples. Master the art of modeling strategic interactions, analyzing equilibria, and optimizing outcomes. Whether you're a student, researcher, or industry professional, this series will equip you with the tools to leverage game theory in various domains, from economics to computer science and beyond.

Hashtags: #GameTheory #Python #StrategicDecisionMaking #CodeExamples #DataScience #MachineLearning #Mathematics #Optimization #CompetitiveAdvantage #RationalChoice #InteractiveSituations #EquilibriumAnalysis #AuctionTheory #VotingTheory #NetworkGames #AlgorithmicGameTheory #AcademicExcellence #InstitutionalLearning

