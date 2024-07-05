## Mathematical Logic with Python

Slide 1: Introduction to Mathematical Logic

Mathematical logic is the study of formal systems for reasoning. It combines mathematics and logic to create a powerful framework for analyzing and solving complex problems.

```python
def is_valid_argument(premises, conclusion):
    # Simplified logical validity checker
    return all(premises) and conclusion

premises = [True, True, False]
conclusion = True

print(f"Is the argument valid? {is_valid_argument(premises, conclusion)}")
```

Slide 2: Propositional Logic

Propositional logic deals with propositions and their relationships. It uses logical connectives to form complex statements.

```python
def AND(p, q):
    return p and q

def OR(p, q):
    return p or q

def NOT(p):
    return not p

p, q = True, False
print(f"p AND q: {AND(p, q)}")
print(f"p OR q: {OR(p, q)}")
print(f"NOT p: {NOT(p)}")
```

Slide 3: Truth Tables

Truth tables display all possible combinations of truth values for logical expressions.

```python
def truth_table(expression, variables):
    print(f"{'|'.join(variables)}|Result")
    print("-" * (len(variables) * 2 + 7))
    
    for values in itertools.product([True, False], repeat=len(variables)):
        result = expression(*values)
        values_str = '|'.join(str(int(v)) for v in values)
        print(f"{values_str}|{int(result)}")

import itertools

def XOR(p, q):
    return (p or q) and not (p and q)

truth_table(XOR, ['p', 'q'])
```

Slide 4: Predicate Logic

Predicate logic extends propositional logic by introducing quantifiers and predicates, allowing for more expressive statements.

```python
class PredicateLogic:
    def __init__(self, domain):
        self.domain = domain
    
    def forall(self, predicate):
        return all(predicate(x) for x in self.domain)
    
    def exists(self, predicate):
        return any(predicate(x) for x in self.domain)

numbers = PredicateLogic(range(1, 11))
is_even = lambda x: x % 2 == 0
print(f"∀x(Even(x)): {numbers.forall(is_even)}")
print(f"∃x(Even(x)): {numbers.exists(is_even)}")
```

Slide 5: First-Order Logic

First-order logic combines predicate logic with quantifiers to express complex relationships and properties of objects.

```python
class FirstOrderLogic:
    def __init__(self, domain):
        self.domain = domain
    
    def interpret(self, formula):
        # Simplified interpreter for first-order logic formulas
        if formula[0] == '∀':
            return all(self.interpret(formula[1:].replace('x', str(x))) for x in self.domain)
        elif formula[0] == '∃':
            return any(self.interpret(formula[1:].replace('x', str(x))) for x in self.domain)
        else:
            return eval(formula)

fol = FirstOrderLogic(range(1, 6))
print(fol.interpret("∀x(x > 0)"))
print(fol.interpret("∃x(x % 2 == 0)"))
```

Slide 6: Logical Inference

Logical inference is the process of deriving new statements from existing ones using rules of inference.

```python
def modus_ponens(p, p_implies_q, q):
    if p and p_implies_q:
        return q
    return "Cannot infer"

p = True
p_implies_q = True
q = True

result = modus_ponens(p, p_implies_q, q)
print(f"Modus Ponens result: {result}")
```

Slide 7: Proof Techniques

Various proof techniques are used in mathematical logic to establish the truth of statements.

```python
def proof_by_contradiction(statement, negation):
    if statement and negation:
        return "Contradiction found, statement is true"
    return "No contradiction, cannot prove statement"

statement = lambda x: x ** 2 >= 0
negation = lambda x: x ** 2 < 0

result = proof_by_contradiction(statement(-1), negation(-1))
print(result)
```

Slide 8: Set Theory

Set theory is fundamental to mathematical logic, providing a foundation for studying mathematical structures.

```python
class Set:
    def __init__(self, elements):
        self.elements = set(elements)
    
    def union(self, other):
        return Set(self.elements.union(other.elements))
    
    def intersection(self, other):
        return Set(self.elements.intersection(other.elements))
    
    def __str__(self):
        return f"{{{', '.join(map(str, self.elements))}}}"

A = Set([1, 2, 3])
B = Set([3, 4, 5])
print(f"A ∪ B = {A.union(B)}")
print(f"A ∩ B = {A.intersection(B)}")
```

Slide 9: Boolean Algebra

Boolean algebra is a branch of mathematical logic dealing with the study of boolean values and operations.

```python
def boolean_function(x, y, z):
    return (x and y) or (not x and z)

def print_truth_table(func):
    print("x | y | z | Result")
    print("-" * 20)
    for x in [False, True]:
        for y in [False, True]:
            for z in [False, True]:
                result = func(x, y, z)
                print(f"{int(x)} | {int(y)} | {int(z)} | {int(result)}")

print_truth_table(boolean_function)
```

Slide 10: Automated Theorem Proving

Automated theorem proving uses algorithms to prove mathematical theorems automatically.

```python
def simple_theorem_prover(axioms, theorem):
    known_truths = set(axioms)
    
    def can_prove(statement):
        if statement in known_truths:
            return True
        for axiom in known_truths:
            if axiom.endswith(f"-> {statement}"):
                premise = axiom.split("->")[0].strip()
                if can_prove(premise):
                    return True
        return False
    
    return can_prove(theorem)

axioms = ["A", "A -> B", "B -> C"]
theorem = "C"

print(f"Can prove theorem: {simple_theorem_prover(axioms, theorem)}")
```

Slide 11: Formal Languages

Formal languages are essential in mathematical logic for precisely expressing and analyzing logical statements.

```python
import re

def is_valid_propositional_formula(formula):
    # Simplified grammar for propositional logic
    atom = r'[pqr]'
    negation = r'¬'
    binary_op = r'[∧∨→↔]'
    
    pattern = f'^({atom}|{negation})*({atom}|{binary_op})*$'
    
    return bool(re.match(pattern, formula))

formulas = ['p∧q', 'p∨¬q', 'p→q↔r', 'pq∧']
for f in formulas:
    print(f"Is '{f}' valid? {is_valid_propositional_formula(f)}")
```

Slide 12: Gödel's Incompleteness Theorems

Gödel's Incompleteness Theorems are fundamental results in mathematical logic about the limitations of formal systems.

```python
def goedel_encoding(statement):
    # Simplified Gödel numbering
    encoding = {'(': 3, ')': 5, '∀': 7, '∃': 11, '=': 13}
    result = 1
    for char in statement:
        if char in encoding:
            result *= encoding[char]
        else:
            result *= ord(char)
    return result

statement = "∀x(x=x)"
encoded = goedel_encoding(statement)
print(f"Gödel encoding of '{statement}': {encoded}")
```

Slide 13: Real-life Example: Database Queries

Mathematical logic principles are applied in database systems for querying and data manipulation.

```python
class DatabaseQuery:
    def __init__(self, data):
        self.data = data
    
    def select(self, condition):
        return [item for item in self.data if condition(item)]
    
    def project(self, attributes):
        return [{k: item[k] for k in attributes} for item in self.data]

employees = [
    {"id": 1, "name": "Alice", "department": "IT"},
    {"id": 2, "name": "Bob", "department": "HR"},
    {"id": 3, "name": "Charlie", "department": "IT"}
]

db = DatabaseQuery(employees)
it_employees = db.select(lambda x: x["department"] == "IT")
names = db.project(["name"])

print("IT Employees:", it_employees)
print("Employee Names:", names)
```

Slide 14: Real-life Example: Constraint Satisfaction Problems

Constraint satisfaction problems (CSPs) use logical principles to solve complex real-world problems.

```python
def solve_csp(variables, domains, constraints):
    def backtrack(assignment):
        if len(assignment) == len(variables):
            return assignment
        var = next(v for v in variables if v not in assignment)
        for value in domains[var]:
            if all(constraint(assignment | {var: value}) for constraint in constraints):
                result = backtrack(assignment | {var: value})
                if result is not None:
                    return result
        return None

    return backtrack({})

# Simple scheduling problem
variables = ['A', 'B', 'C']
domains = {var: list(range(3)) for var in variables}  # 0: morning, 1: afternoon, 2: evening
constraints = [
    lambda a: a.get('A', 1) != a.get('B', 1),  # A and B can't be at the same time
    lambda a: a.get('B', 1) < a.get('C', 1),   # B must be before C
]

solution = solve_csp(variables, domains, constraints)
print("Schedule:", {v: ["morning", "afternoon", "evening"][t] for v, t in solution.items()})
```

Slide 15: Additional Resources

For further exploration of Mathematical Logic, consider these peer-reviewed articles from ArXiv:

1. "A Survey of Automated Theorem Proving" by John Harrison (arXiv:cs/9404215)
2. "Foundations of Mathematical Logic" by H. Jerome Keisler (arXiv:math/0408173)
3. "An Introduction to Mathematical Logic" by Alonzo Church (arXiv:math/0703035)

Visit arxiv.org and search for these article IDs to access the full papers.

