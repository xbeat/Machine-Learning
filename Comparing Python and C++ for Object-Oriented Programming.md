## Comparing Python and C++ for Object-Oriented Programming
Slide 1: Python Classes - Basic Structure

Title: Python Classes - Basic Structure

Description: Classes in Python define objects with attributes and methods.

```python
class Car:
    def __init__(self, make, model):
        self.make = make
        self.model = model
    
    def display_info(self):
        return f"{self.make} {self.model}"

# Creating an object
my_car = Car("Toyota", "Corolla")
print(my_car.display_info())  # Output: Toyota Corolla
```

Slide 2: C++ Classes - Basic Structure

Title: C++ Classes - Basic Structure

Description: C++ classes encapsulate data and functions into a single unit.

```cpp
#include <iostream>
#include <string>

class Car {
private:
    std::string make;
    std::string model;

public:
    Car(std::string m, std::string md) : make(m), model(md) {}
    
    std::string displayInfo() {
        return make + " " + model;
    }
};

int main() {
    Car myCar("Toyota", "Corolla");
    std::cout << myCar.displayInfo() << std::endl;  // Output: Toyota Corolla
    return 0;
}
```

Slide 3: Python Inheritance

Title: Python Inheritance

Description: Inheritance allows a class to inherit attributes and methods from another class.

```python
class Vehicle:
    def __init__(self, wheels):
        self.wheels = wheels
    
    def move(self):
        return "Moving..."

class Car(Vehicle):
    def __init__(self, make, model):
        super().__init__(4)  # Cars typically have 4 wheels
        self.make = make
        self.model = model
    
    def honk(self):
        return "Beep!"

my_car = Car("Honda", "Civic")
print(my_car.move())  # Output: Moving...
print(my_car.honk())  # Output: Beep!
```

Slide 4: C++ Inheritance

Title: C++ Inheritance

Description: C++ supports inheritance, allowing derived classes to inherit from base classes.

```cpp
#include <iostream>
#include <string>

class Vehicle {
protected:
    int wheels;
public:
    Vehicle(int w) : wheels(w) {}
    std::string move() { return "Moving..."; }
};

class Car : public Vehicle {
private:
    std::string make;
    std::string model;
public:
    Car(std::string m, std::string md) : Vehicle(4), make(m), model(md) {}
    std::string honk() { return "Beep!"; }
};

int main() {
    Car myCar("Honda", "Civic");
    std::cout << myCar.move() << std::endl;  // Output: Moving...
    std::cout << myCar.honk() << std::endl;  // Output: Beep!
    return 0;
}
```

Slide 5: Python Encapsulation

Title: Python Encapsulation

Description: Encapsulation in Python uses naming conventions to indicate access levels.

```python
class BankAccount:
    def __init__(self, balance):
        self._balance = balance  # Protected attribute
    
    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
    
    def get_balance(self):
        return self._balance

account = BankAccount(1000)
account.deposit(500)
print(account.get_balance())  # Output: 1500
# print(account._balance)  # Discouraged, but possible
```

Slide 6: C++ Encapsulation

Title: C++ Encapsulation

Description: C++ provides access specifiers for strict encapsulation control.

```cpp
#include <iostream>

class BankAccount {
private:
    double balance;

public:
    BankAccount(double initial_balance) : balance(initial_balance) {}
    
    void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
        }
    }
    
    double getBalance() const {
        return balance;
    }
};

int main() {
    BankAccount account(1000);
    account.deposit(500);
    std::cout << account.getBalance() << std::endl;  // Output: 1500
    // std::cout << account.balance;  // Compilation error
    return 0;
}
```

Slide 7: Python Polymorphism

Title: Python Polymorphism

Description: Polymorphism allows objects of different classes to be treated as objects of a common base class.

```python
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

def animal_sound(animal):
    print(animal.speak())

dog = Dog()
cat = Cat()

animal_sound(dog)  # Output: Woof!
animal_sound(cat)  # Output: Meow!
```

Slide 8: C++ Polymorphism

Title: C++ Polymorphism

Description: C++ implements polymorphism through virtual functions and pointers/references to base class.

```cpp
#include <iostream>

class Animal {
public:
    virtual std::string speak() = 0;  // Pure virtual function
};

class Dog : public Animal {
public:
    std::string speak() override { return "Woof!"; }
};

class Cat : public Animal {
public:
    std::string speak() override { return "Meow!"; }
};

void animalSound(Animal& animal) {
    std::cout << animal.speak() << std::endl;
}

int main() {
    Dog dog;
    Cat cat;
    
    animalSound(dog);  // Output: Woof!
    animalSound(cat);  // Output: Meow!
    return 0;
}
```

Slide 9: Python Multiple Inheritance

Title: Python Multiple Inheritance

Description: Python supports multiple inheritance, allowing a class to inherit from multiple base classes.

```python
class Flying:
    def fly(self):
        return "I can fly!"

class Swimming:
    def swim(self):
        return "I can swim!"

class Duck(Flying, Swimming):
    def quack(self):
        return "Quack!"

duck = Duck()
print(duck.fly())    # Output: I can fly!
print(duck.swim())   # Output: I can swim!
print(duck.quack())  # Output: Quack!
```

Slide 10: C++ Multiple Inheritance

Title: C++ Multiple Inheritance

Description: C++ also supports multiple inheritance, but it can lead to the diamond problem.

```cpp
#include <iostream>

class Flying {
public:
    std::string fly() { return "I can fly!"; }
};

class Swimming {
public:
    std::string swim() { return "I can swim!"; }
};

class Duck : public Flying, public Swimming {
public:
    std::string quack() { return "Quack!"; }
};

int main() {
    Duck duck;
    std::cout << duck.fly() << std::endl;    // Output: I can fly!
    std::cout << duck.swim() << std::endl;   // Output: I can swim!
    std::cout << duck.quack() << std::endl;  // Output: Quack!
    return 0;
}
```

Slide 11: Python Abstract Base Classes

Title: Python Abstract Base Classes

Description: Python uses the ABC module to define abstract base classes.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14 * self.radius ** 2

# shape = Shape()  # TypeError: Can't instantiate abstract class
circle = Circle(5)
print(f"Area: {circle.area()}")  # Output: Area: 78.5
```

Slide 12: C++ Abstract Classes

Title: C++ Abstract Classes

Description: C++ uses pure virtual functions to create abstract classes.

```cpp
#include <iostream>

class Shape {
public:
    virtual double area() = 0;  // Pure virtual function
};

class Circle : public Shape {
private:
    double radius;
public:
    Circle(double r) : radius(r) {}
    
    double area() override {
        return 3.14 * radius * radius;
    }
};

int main() {
    // Shape shape;  // Compilation error: cannot instantiate abstract class
    Circle circle(5);
    std::cout << "Area: " << circle.area() << std::endl;  // Output: Area: 78.5
    return 0;
}
```

Slide 13: Python Properties

Title: Python Properties

Description: Python uses properties to create getter, setter, and deleter methods for attributes.

```python
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def fahrenheit(self):
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5/9

temp = Temperature(25)
print(f"{temp.fahrenheit}°F")  # Output: 77.0°F
temp.fahrenheit = 68
print(f"{temp._celsius}°C")    # Output: 20.0°C
```

Slide 14: C++ Getters and Setters

Title: C++ Getters and Setters

Description: C++ uses explicit getter and setter methods for controlled access to class members.

```cpp
#include <iostream>

class Temperature {
private:
    double celsius;

public:
    Temperature(double c) : celsius(c) {}
    
    double getFahrenheit() const {
        return (celsius * 9.0/5.0) + 32;
    }
    
    void setFahrenheit(double f) {
        celsius = (f - 32) * 5.0/9.0;
    }
    
    double getCelsius() const {
        return celsius;
    }
};

int main() {
    Temperature temp(25);
    std::cout << temp.getFahrenheit() << "°F" << std::endl;  // Output: 77°F
    temp.setFahrenheit(68);
    std::cout << temp.getCelsius() << "°C" << std::endl;     // Output: 20°C
    return 0;
}
```

Slide 15: Wrap-up - Python vs C++ in OOP

Title: Wrap-up - Python vs C++ in OOP

Description: Key differences between Python and C++ in Object-Oriented Programming.

```
| Feature           | Python                                 | C++                                    |
|-------------------|----------------------------------------|----------------------------------------|
| Syntax            | Simple, less verbose                   | More complex, explicit                 |
| Type System       | Dynamic typing                         | Static typing                          |
| Encapsulation     | Convention-based (_private)            | Strict (private, protected, public)    |
| Multiple Inherit. | Supported, uses MRO                    | Supported, can lead to diamond problem |
| Abstract Classes  | Uses ABC module                        | Uses pure virtual functions            |
| Memory Management | Automatic (garbage collection)         | Manual (destructors, smart pointers)   |
| Performance       | Generally slower                       | Generally faster                       |
| Ease of Use       | Easier for beginners                   | Steeper learning curve                 |
| Compile/Interpret | Interpreted (also compiled to bytecode)| Compiled to machine code               |
```

