## Introduction to Finite State Machines in Python

Slide 1: 
Introduction to Finite State Machines (FSMs) 
A Finite State Machine (FSM) is a computational model used to design systems that can be in one of a finite number of states. It transitions from one state to another based on specific inputs or events. FSMs are widely used in various domains, including computer science, electronics, and linguistics.

Slide 2: 
Components of an FSM 
An FSM consists of the following components:

1. States: The different conditions or situations the system can be in.
2. Transitions: The rules that define how the system moves from one state to another.
3. Start State: The initial state of the system when it begins operation.
4. Accept/Final States: The states that indicate successful completion of a task or sequence.

Slide 3: Simple FSM Example Let's start with a simple example of an FSM that models a turnstile system.

```python
from enum import Enum

class State(Enum):
    LOCKED = 0
    UNLOCKED = 1

class TurnstileFSM:
    def __init__(self):
        self.state = State.LOCKED

    def coin_inserted(self):
        if self.state == State.LOCKED:
            self.state = State.UNLOCKED
            print("Turnstile unlocked.")
        else:
            print("Turnstile already unlocked.")

    def person_passed(self):
        if self.state == State.UNLOCKED:
            self.state = State.LOCKED
            print("Turnstile locked.")
        else:
            print("Turnstile is locked.")
```

Slide 4: 
Using the Turnstile FSM 
Here's how you can use the TurnstileFS M:

```python
turnstile = TurnstileFS M()
turnstile.coin_inserted() # Output: Turnstile unlocked.
turnstile.person_passed() # Output: Turnstile locked.
turnstile.person_passed() # Output: Turnstile is locked.
```

Slide 5: Advantages of FSMs FSMs offer several advantages:

* Simplicity: FSMs are easy to understand and implement.
* Modular design: FSMs can be combined and composed to create more complex systems.
* Deterministic behavior: Given a specific input, an FSM will always transition to a well-defined state.
* Testing and verification: FSMs can be thoroughly tested and verified due to their finite nature.

Slide 6: 
Implementing FSMs in Python 
Python provides several ways to implement FSMs. One approach is to use classes and methods to represent states and transitions. Another approach is to use dictionaries or lookup tables to define state transitions.

Slide 7: 
Class-based FSM Implementation 
Here's an example of a class-based FSM implementation for a simple vending machine:

```python
class State:
    def run(self):
        pass

class IdleState(State):
    def run(self):
        print("Waiting for input...")

class CoinInsertedState(State):
    def run(self):
        print("Coin inserted, select item...")

class ItemDeliveredState(State):
    def run(self):
        print("Item delivered, thank you!")

class VendingMachine:
    def __init__(self):
        self.state = IdleState()

    def coin_inserted(self):
        self.state = CoinInsertedState()

    def item_selected(self):
        self.state = ItemDeliveredState()
        self.dispense_item()

    def dispense_item(self):
        print("Dispensing item...")

    def run(self):
        while True:
            self.state.run()
            input_value = input("Enter input: ")
            if input_value == "coin":
                self.coin_inserted()
            elif input_value == "select":
                self.item_selected()
```

Slide 8: 
Using the Vending Machine FSM 
Here's how you can use the VendingMachine FSM:

```python
vending_machine = VendingMachine()
vending_machine.run()
```

Input:

```
Enter input: coin
Enter input: select
```

Output:

```
Waiting for input...
Coin inserted, select item...
Item delivered, thank you!
Dispensing item...
```

Slide 9: 
Dictionary-based FSM Implementation 
Alternatively, you can implement an FSM using dictionaries or lookup tables to define state transitions. Here's an example:

```python
transitions = {}

def idle_state(input_value):
    if input_value == "coin":
        print("Coin inserted, select item...")
        return "coin_inserted"
    else:
        print("Waiting for input...")
        return "idle"

def coin_inserted_state(input_value):
    if input_value == "select":
        print("Item delivered, thank you!")
        return "idle"
    else:
        print("Invalid input, please try again.")
        return "coin_inserted"

transitions["idle"] = idle_state
transitions["coin_inserted"] = coin_inserted_state

def run_fsm():
    current_state = "idle"
    while True:
        input_value = input("Enter input: ")
        current_state = transitions[current_state](input_value)
```

Slide 10: 
Running the Dictionary-based FSM 
To run the dictionary-based FSM, simply call the `run_fsm()` function:

```python
run_fsm()
```

Input:

```
Enter input: coin
Enter input: select
Enter input: invalid
```

Output:

```
Waiting for input...
Coin inserted, select item...
Item delivered, thank you!
Waiting for input...
Invalid input, please try again.
```

Slide 11: 
FSM Design Considerations 
When designing FSMs, it's essential to consider the following:

* Identify all possible states and transitions.
* Define clear rules for state transitions based on inputs or events.
* Ensure that the FSM is deterministic and free from ambiguities.
* Consider error handling and invalid inputs or events.

Slide 12: 
FSM Applications 
FSMs have a wide range of applications, including:

* User interface design (e.g., modal dialogs, wizards)
* Protocol implementation (e.g., network protocols, communication protocols)
* Control systems (e.g., traffic lights, elevators)
* Natural language processing (e.g., parsing, text generation)
* Game development (e.g., character behavior, level design)

Slide 13: 
Advantages and Limitations of FSMs 
Advantages of FSMs:

* Simple and easy to understand
* Deterministic behavior
* Well-suited for modeling sequential processes

Limitations of FSMs:

* Can become complex for systems with many states and transitions
* Not suitable for modeling concurrent or parallel processes
* May require additional data structures for more complex scenarios

Slide 14: 
Resources and Further Reading 
Here are some resources for further reading on FSMs:

* "Finite State Machines in Python" (Python documentation)
* "Finite State Machines" (Wikipedia)
* "Implementing Finite State Machines in Python" (Real Python)
* "Finite State Machines: Theory and Implementation" (Michael Sipser)

