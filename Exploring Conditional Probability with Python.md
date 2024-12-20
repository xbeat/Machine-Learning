## Exploring Conditional Probability with Python
Slide 1: Introduction to Conditional Probability

Conditional probability is a fundamental concept in statistics that allows us to calculate the likelihood of an event occurring given that another event has already occurred. This powerful tool helps us make informed decisions in various fields, from medicine to machine learning.

```python
# Simulating a simple conditional probability scenario
import random

def coin_flip():
    return random.choice(['Heads', 'Tails'])

def two_coin_flips():
    return (coin_flip(), coin_flip())

# Simulate 10000 trials of flipping two coins
trials = 10000
results = [two_coin_flips() for _ in range(trials)]

# Count the number of times we get at least one head
at_least_one_head = sum(1 for r in results if 'Heads' in r)

# Count the number of times we get two heads
two_heads = sum(1 for r in results if r == ('Heads', 'Heads'))

# Calculate P(Second coin is Heads | First coin is Heads)
p_second_head_given_first_head = two_heads / at_least_one_head

print(f"P(Second coin is Heads | First coin is Heads) ≈ {p_second_head_given_first_head:.4f}")
```

Slide 2: The Basics of Conditional Probability

Conditional probability is expressed as P(A|B), which reads as "the probability of event A occurring given that event B has occurred." This concept helps us update our beliefs based on new information, allowing for more accurate predictions and decision-making.

```python
# Visualizing conditional probability with a Venn diagram
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# Create a Venn diagram
plt.figure(figsize=(8, 6))
v = venn2(subsets=(300, 200, 100), set_labels=('Event A', 'Event B'))

# Add labels for probabilities
plt.text(0.2, 0.2, 'P(A|B)', fontsize=12, ha='center')
plt.text(-0.3, 0.2, 'P(A)', fontsize=12, ha='center')
plt.text(0.7, 0.2, 'P(B)', fontsize=12, ha='center')

plt.title('Conditional Probability: P(A|B)')
plt.show()
```

Slide 3: The Conditional Probability Formula

The formula for conditional probability is P(A|B) = P(A ∩ B) / P(B), where P(A ∩ B) is the probability of both events A and B occurring, and P(B) is the probability of event B occurring. This formula allows us to calculate the probability of an event given that another event has occurred.

```python
def conditional_probability(p_a_and_b, p_b):
    """
    Calculate conditional probability P(A|B)
    
    :param p_a_and_b: Probability of both A and B occurring
    :param p_b: Probability of B occurring
    :return: Conditional probability P(A|B)
    """
    if p_b == 0:
        return "Undefined (division by zero)"
    return p_a_and_b / p_b

# Example calculation
p_a_and_b = 0.3  # Probability of both A and B occurring
p_b = 0.5  # Probability of B occurring

result = conditional_probability(p_a_and_b, p_b)
print(f"P(A|B) = {result:.4f}")
```

Slide 4: Bayes' Theorem

Bayes' Theorem is a powerful extension of conditional probability that allows us to reverse the condition and calculate P(B|A) from P(A|B). The formula is: P(B|A) = P(A|B) \* P(B) / P(A). This theorem is crucial in many applications, including medical diagnosis and machine learning.

```python
def bayes_theorem(p_a_given_b, p_b, p_a):
    """
    Calculate P(B|A) using Bayes' Theorem
    
    :param p_a_given_b: P(A|B)
    :param p_b: P(B)
    :param p_a: P(A)
    :return: P(B|A)
    """
    return (p_a_given_b * p_b) / p_a

# Example: Medical diagnosis
p_disease = 0.01  # 1% of population has the disease
p_positive_given_disease = 0.95  # 95% true positive rate
p_positive_given_no_disease = 0.05  # 5% false positive rate

p_positive = p_positive_given_disease * p_disease + p_positive_given_no_disease * (1 - p_disease)
p_disease_given_positive = bayes_theorem(p_positive_given_disease, p_disease, p_positive)

print(f"Probability of having the disease given a positive test: {p_disease_given_positive:.4f}")
```

Slide 5: The Law of Total Probability

The Law of Total Probability is a fundamental rule that relates marginal probabilities to conditional probabilities. It states that for a partition of the sample space into events B1, B2, ..., Bn, the probability of an event A is the sum of the conditional probabilities of A given each Bi, weighted by the probability of Bi.

```python
import numpy as np
import matplotlib.pyplot as plt

def total_probability(p_a_given_b, p_b):
    """
    Calculate P(A) using the Law of Total Probability
    
    :param p_a_given_b: List of P(A|Bi) for each partition
    :param p_b: List of P(Bi) for each partition
    :return: P(A)
    """
    return np.sum(np.array(p_a_given_b) * np.array(p_b))

# Example: Weather forecast
weather_conditions = ['Sunny', 'Cloudy', 'Rainy']
p_picnic_given_weather = [0.9, 0.6, 0.1]  # P(Picnic|Weather)
p_weather = [0.5, 0.3, 0.2]  # P(Weather)

p_picnic = total_probability(p_picnic_given_weather, p_weather)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.bar(weather_conditions, p_picnic_given_weather, alpha=0.5, label='P(Picnic|Weather)')
plt.bar(weather_conditions, p_weather, alpha=0.5, label='P(Weather)')
plt.axhline(y=p_picnic, color='r', linestyle='--', label='P(Picnic)')
plt.legend()
plt.title('Law of Total Probability: Picnic Example')
plt.ylabel('Probability')
plt.show()

print(f"Overall probability of having a picnic: {p_picnic:.4f}")
```

Slide 6: Independence and Conditional Independence

Two events A and B are independent if the occurrence of one does not affect the probability of the other. Mathematically, A and B are independent if P(A|B) = P(A) or equivalently, if P(A ∩ B) = P(A) \* P(B). Conditional independence is a similar concept but in the context of a third event C.

```python
import numpy as np

def check_independence(p_a, p_b, p_a_and_b, tolerance=1e-6):
    """
    Check if events A and B are independent
    
    :param p_a: P(A)
    :param p_b: P(B)
    :param p_a_and_b: P(A ∩ B)
    :param tolerance: Tolerance for floating-point comparison
    :return: True if independent, False otherwise
    """
    return np.isclose(p_a_and_b, p_a * p_b, atol=tolerance)

# Example: Rolling two fair dice
p_even_first = 0.5  # P(First die is even)
p_sum_greater_than_7 = 21/36  # P(Sum is greater than 7)
p_even_first_and_sum_greater_than_7 = 9/36  # P(First die is even AND sum is greater than 7)

independent = check_independence(p_even_first, p_sum_greater_than_7, p_even_first_and_sum_greater_than_7)
print(f"Are the events independent? {independent}")
```

Slide 7: The Chain Rule of Probability

The Chain Rule of Probability allows us to calculate the joint probability of multiple events by decomposing it into a product of conditional probabilities. For events A, B, and C, P(A ∩ B ∩ C) = P(A) \* P(B|A) \* P(C|A ∩ B).

```python
def chain_rule(probabilities):
    """
    Calculate joint probability using the Chain Rule
    
    :param probabilities: List of probabilities [P(A), P(B|A), P(C|A∩B), ...]
    :return: Joint probability
    """
    return np.prod(probabilities)

# Example: Card drawing without replacement
p_ace = 4/52  # P(First card is Ace)
p_king_given_ace = 4/51  # P(Second card is King | First card is Ace)
p_queen_given_ace_king = 4/50  # P(Third card is Queen | First is Ace and Second is King)

joint_prob = chain_rule([p_ace, p_king_given_ace, p_queen_given_ace_king])
print(f"Probability of drawing Ace, King, Queen in that order: {joint_prob:.6f}")
```

Slide 8: Conditional Probability in Machine Learning

Conditional probability is a fundamental concept in many machine learning algorithms, particularly in Bayesian methods and probabilistic graphical models. One common application is in Naive Bayes classifiers, which use Bayes' theorem to predict the most likely class for a given input.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_split import train_test_split
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Naive Bayes Classifier Accuracy: {accuracy:.4f}")

# Get probability estimates for a single sample
sample = X_test[0].reshape(1, -1)
class_probabilities = nb_classifier.predict_proba(sample)
print(f"Class probabilities for the sample: {class_probabilities[0]}")
```

Slide 9: Conditional Probability in Natural Language Processing

Conditional probability plays a crucial role in various Natural Language Processing (NLP) tasks, such as language modeling, part-of-speech tagging, and named entity recognition. One common application is in n-gram language models, which use conditional probabilities to predict the next word in a sequence.

```python
import nltk
from nltk.util import ngrams
from collections import defaultdict, Counter

# Download required NLTK data
nltk.download('punkt')
nltk.download('brown')

# Load a corpus
from nltk.corpus import brown
sentences = brown.sents(categories='news')

# Create a bigram model
bigram_model = defaultdict(lambda: defaultdict(lambda: 0))
for sentence in sentences:
    for w1, w2 in ngrams(sentence, 2):
        bigram_model[w1][w2] += 1

# Convert counts to probabilities
for w1 in bigram_model:
    total_count = float(sum(bigram_model[w1].values()))
    for w2 in bigram_model[w1]:
        bigram_model[w1][w2] /= total_count

# Function to generate the next word
def generate_next_word(current_word):
    if current_word not in bigram_model:
        return "."
    return max(bigram_model[current_word], key=bigram_model[current_word].get)

# Generate a sequence of words
seed_word = "the"
sequence = [seed_word]
for _ in range(10):
    next_word = generate_next_word(sequence[-1])
    sequence.append(next_word)
    if next_word == ".":
        break

print(" ".join(sequence))
```

Slide 10: Monte Carlo Methods and Conditional Probability

Monte Carlo methods are computational algorithms that use repeated random sampling to obtain numerical results. These methods are particularly useful when dealing with complex probability distributions or high-dimensional problems. We can use Monte Carlo simulations to estimate conditional probabilities in scenarios where analytical solutions are difficult to obtain.

```python
import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_conditional_prob(num_samples, condition_func, event_func):
    """
    Estimate conditional probability using Monte Carlo simulation
    
    :param num_samples: Number of samples to generate
    :param condition_func: Function that checks if a sample meets the condition
    :param event_func: Function that checks if a sample belongs to the event
    :return: Estimated conditional probability
    """
    samples = np.random.uniform(0, 1, (num_samples, 2))
    condition_met = condition_func(samples)
    event_occurred = event_func(samples)
    
    conditional_prob = np.sum(event_occurred[condition_met]) / np.sum(condition_met)
    return conditional_prob

# Example: Estimating P(X + Y > 1.5 | X > 0.7) for X, Y ~ U(0, 1)
condition_func = lambda s: s[:, 0] > 0.7
event_func = lambda s: s[:, 0] + s[:, 1] > 1.5

estimated_prob = monte_carlo_conditional_prob(1000000, condition_func, event_func)
print(f"Estimated conditional probability: {estimated_prob:.4f}")

# Visualize the results
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axvline(x=0.7, color='r', linestyle='--', label='X = 0.7')
ax.plot([0, 1], [1.5, 0.5], 'g-', label='X + Y = 1.5')
ax.fill_between([0.7, 1], [0.8, 0.5], 1, alpha=0.3, color='b', label='Condition & Event')
ax.legend()
ax.set_title('Monte Carlo Estimation of Conditional Probability')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()
```

Slide 11: Conditional Probability in Bayesian Networks

Bayesian networks are graphical models that represent probabilistic relationships among a set of variables. They use conditional probability tables (CPTs) to define the relationships between connected nodes. Bayesian networks are powerful tools for reasoning under uncertainty and are widely used in artificial intelligence and expert systems.

```python
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure of the Bayesian network
model = BayesianNetwork([('Cloudy', 'Rain'), ('Cloudy', 'Sprinkler'), 
                         ('Rain', 'WetGrass'), ('Sprinkler', 'WetGrass')])

# Define the CPDs (Conditional Probability Distributions)
cpd_cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[0.5], [0.5]])
cpd_rain = TabularCPD(variable='Rain', variable_card=2, 
                      values=[[0.8, 0.2], [0.2, 0.8]],
                      evidence=['Cloudy'], evidence_card=[2])
cpd_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
                           values=[[0.4, 0.9], [0.6, 0.1]],
                           evidence=['Cloudy'], evidence_card=[2])
cpd_wetgrass = TabularCPD(variable='WetGrass', variable_card=2,
                          values=[[1.0, 0.1, 0.1, 0.01],
                                  [0.0, 0.9, 0.9, 0.99]],
                          evidence=['Sprinkler', 'Rain'],
                          evidence_card=[2, 2])

# Add CPDs to the model
model.add_cpds(cpd_cloudy, cpd_rain, cpd_sprinkler, cpd_wetgrass)

# Perform inference
inference = VariableElimination(model)
result = inference.query(variables=['WetGrass'], evidence={'Cloudy': 1})
print("Probability of wet grass given that it's cloudy:")
print(result)
```

Slide 12: Real-Life Example: Disease Diagnosis

Conditional probability is crucial in medical diagnosis. Let's consider a scenario where a doctor is trying to determine the probability of a patient having a certain disease given a positive test result. This example demonstrates the practical application of Bayes' theorem in a real-world context.

```python
def calculate_disease_probability(prevalence, sensitivity, specificity, test_result):
    """
    Calculate the probability of having a disease given a test result
    
    :param prevalence: Prior probability of having the disease
    :param sensitivity: True positive rate (P(positive test | disease))
    :param specificity: True negative rate (P(negative test | no disease))
    :param test_result: 'positive' or 'negative'
    :return: Probability of having the disease given the test result
    """
    if test_result == 'positive':
        p_test_given_disease = sensitivity
        p_test_given_no_disease = 1 - specificity
    else:
        p_test_given_disease = 1 - sensitivity
        p_test_given_no_disease = specificity
    
    p_test = (p_test_given_disease * prevalence) + (p_test_given_no_disease * (1 - prevalence))
    p_disease_given_test = (p_test_given_disease * prevalence) / p_test
    
    return p_disease_given_test

# Example: Rare disease diagnosis
prevalence = 0.01  # 1% of the population has the disease
sensitivity = 0.95  # 95% of diseased individuals test positive
specificity = 0.90  # 90% of healthy individuals test negative

p_disease_given_positive = calculate_disease_probability(prevalence, sensitivity, specificity, 'positive')
print(f"Probability of having the disease given a positive test: {p_disease_given_positive:.4f}")

p_disease_given_negative = calculate_disease_probability(prevalence, sensitivity, specificity, 'negative')
print(f"Probability of having the disease given a negative test: {p_disease_given_negative:.4f}")
```

Slide 13: Real-Life Example: Spam Email Classification

Email spam filters often use conditional probability techniques, such as Naive Bayes classifiers, to determine whether an incoming email is spam or not. This example demonstrates how to implement a simple spam filter using the Naive Bayes algorithm.

```python
from collections import defaultdict
import math

class NaiveBayesSpamFilter:
    def __init__(self):
        self.word_counts = {'spam': defaultdict(int), 'ham': defaultdict(int)}
        self.total_counts = {'spam': 0, 'ham': 0}
    
    def train(self, email, category):
        for word in email.split():
            self.word_counts[category][word] += 1
            self.total_counts[category] += 1
    
    def classify(self, email):
        words = email.split()
        spam_score = math.log(0.5)  # Prior probability of spam
        ham_score = math.log(0.5)   # Prior probability of ham
        
        for word in words:
            spam_prob = (self.word_counts['spam'][word] + 1) / (self.total_counts['spam'] + len(self.word_counts['spam']))
            ham_prob = (self.word_counts['ham'][word] + 1) / (self.total_counts['ham'] + len(self.word_counts['ham']))
            
            spam_score += math.log(spam_prob)
            ham_score += math.log(ham_prob)
        
        return 'spam' if spam_score > ham_score else 'ham'

# Example usage
spam_filter = NaiveBayesSpamFilter()

# Training
spam_filter.train("Buy cheap watches now", "spam")
spam_filter.train("Get rich quick", "spam")
spam_filter.train("Hello, how are you?", "ham")
spam_filter.train("Meeting at 3 PM", "ham")

# Classification
test_email = "Buy now and get rich"
result = spam_filter.classify(test_email)
print(f"The email '{test_email}' is classified as: {result}")
```

Slide 14: Conditional Probability in A/B Testing

A/B testing is a common technique used in marketing and web design to compare two versions of a webpage or app to determine which one performs better. Conditional probability plays a crucial role in analyzing the results of these tests and making data-driven decisions.

```python
import numpy as np
from scipy import stats

def ab_test(conversions_a, samples_a, conversions_b, samples_b, confidence_level=0.95):
    """
    Perform an A/B test and calculate the statistical significance
    
    :param conversions_a: Number of conversions in group A
    :param samples_a: Total number of samples in group A
    :param conversions_b: Number of conversions in group B
    :param samples_b: Total number of samples in group B
    :param confidence_level: Desired confidence level (default: 0.95)
    :return: Tuple (is_significant, p_value)
    """
    rate_a = conversions_a / samples_a
    rate_b = conversions_b / samples_b
    
    # Calculate the standard error
    se = np.sqrt(rate_a * (1 - rate_a) / samples_a + rate_b * (1 - rate_b) / samples_b)
    
    # Calculate the z-score
    z_score = (rate_b - rate_a) / se
    
    # Calculate the p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Determine if the result is statistically significant
    is_significant = p_value < (1 - confidence_level)
    
    return is_significant, p_value

# Example A/B test
conversions_a, samples_a = 180, 1000  # Control group
conversions_b, samples_b = 210, 1000  # Test group

is_significant, p_value = ab_test(conversions_a, samples_a, conversions_b, samples_b)

print(f"Conversion rate A: {conversions_a/samples_a:.2%}")
print(f"Conversion rate B: {conversions_b/samples_b:.2%}")
print(f"Is the difference statistically significant? {is_significant}")
print(f"P-value: {p_value:.4f}")
```

Slide 15: Additional Resources

For those interested in delving deeper into conditional probability and its applications, here are some valuable resources:

1. "Probabilistic Graphical Models: Principles and Techniques" by Daphne Koller and Nir Friedman (MIT Press, 2009)
2. "Pattern Recognition and Machine Learning" by Christopher M. Bishop (Springer, 2006)
3. "Introduction to Probability" by Joseph K. Blitzstein and Jessica Hwang (Chapman and Hall/CRC, 2019)
4. ArXiv.org papers:
   * "A Tutorial on Bayesian Optimization" by Peter I. Frazier (ArXiv:1807.02811)
   * "Probabilistic Machine Learning and Artificial Intelligence" by Zoubin Ghahramani (ArXiv:1502.05336)

These resources provide in-depth explanations and advanced topics related to conditional probability and its applications in various fields.

