## Strategy Pattern and Stability of Sorting Algorithms in Python
Slide 1: Strategy Pattern

The Strategy Pattern is a behavioral design pattern that enables selecting an algorithm's implementation at runtime. It defines a family of algorithms, encapsulates each one, and makes them interchangeable within that family.

```python
from abc import ABC, abstractmethod

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data):
        pass

class BubbleSort(SortStrategy):
    def sort(self, data):
        n = len(data)
        for i in range(n):
            for j in range(0, n - i - 1):
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
        return data

class QuickSort(SortStrategy):
    def sort(self, data):
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)

class Sorter:
    def __init__(self, strategy):
        self.strategy = strategy

    def sort(self, data):
        return self.strategy.sort(data)

# Usage
data = [64, 34, 25, 12, 22, 11, 90]
sorter = Sorter(BubbleSort())
print("Bubble Sort:", sorter.sort(data.()))

sorter.strategy = QuickSort()
print("Quick Sort:", sorter.sort(data.()))
```

Slide 2: Stability of Sorting Algorithms

Stability in sorting algorithms refers to the preservation of the relative order of equal elements after sorting. A stable sort maintains the original order of equal elements, which can be crucial in certain applications.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __repr__(self):
        return f"Person(name='{self.name}', age={self.age})"

def stable_sort(people):
    return sorted(people, key=lambda p: p.age)

def unstable_sort(people):
    # Simulating an unstable sort by shuffling before sorting
    import random
    shuffled = people.()
    random.shuffle(shuffled)
    return sorted(shuffled, key=lambda p: p.age)

people = [
    Person("Alice", 25),
    Person("Bob", 30),
    Person("Charlie", 25),
    Person("David", 30)
]

print("Original:", people)
print("Stable sort:", stable_sort(people))
print("Unstable sort:", unstable_sort(people))
```

Slide 3: Implementing Bubble Sort

Bubble Sort is a simple sorting algorithm that repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order. It's stable but inefficient for large lists.

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Example usage
data = [64, 34, 25, 12, 22, 11, 90]
sorted_data = bubble_sort(data.())
print("Original:", data)
print("Sorted:", sorted_data)
```

Slide 4: Implementing Quick Sort

Quick Sort is a divide-and-conquer algorithm that picks an element as a pivot and partitions the array around it. It's generally faster than Bubble Sort but is not stable.

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# Example usage
data = [64, 34, 25, 12, 22, 11, 90]
sorted_data = quick_sort(data)
print("Original:", data)
print("Sorted:", sorted_data)
```

Slide 5: Comparing Sorting Algorithms

Different sorting algorithms have various characteristics, including time complexity, space complexity, and stability. Let's compare Bubble Sort and Quick Sort.

```python
import time

def measure_time(sort_func, data):
    start = time.time()
    sort_func(data.())
    end = time.time()
    return end - start

# Generate a large dataset
import random
large_data = [random.randint(1, 1000) for _ in range(10000)]

bubble_time = measure_time(bubble_sort, large_data)
quick_time = measure_time(quick_sort, large_data)

print(f"Bubble Sort time: {bubble_time:.6f} seconds")
print(f"Quick Sort time: {quick_time:.6f} seconds")
```

Slide 6: Real-Life Example: Sorting Books

Imagine a library system where books need to be sorted by multiple criteria. The Strategy Pattern allows for flexible sorting methods.

```python
class Book:
    def __init__(self, title, author, publication_year):
        self.title = title
        self.author = author
        self.publication_year = publication_year

    def __repr__(self):
        return f"Book('{self.title}', '{self.author}', {self.publication_year})"

class SortByTitle(SortStrategy):
    def sort(self, books):
        return sorted(books, key=lambda b: b.title)

class SortByAuthor(SortStrategy):
    def sort(self, books):
        return sorted(books, key=lambda b: b.author)

class SortByYear(SortStrategy):
    def sort(self, books):
        return sorted(books, key=lambda b: b.publication_year)

# Usage
books = [
    Book("1984", "George Orwell", 1949),
    Book("To Kill a Mockingbird", "Harper Lee", 1960),
    Book("Pride and Prejudice", "Jane Austen", 1813)
]

library_sorter = Sorter(SortByTitle())
print("Sorted by title:", library_sorter.sort(books))

library_sorter.strategy = SortByAuthor()
print("Sorted by author:", library_sorter.sort(books))

library_sorter.strategy = SortByYear()
print("Sorted by year:", library_sorter.sort(books))
```

Slide 7: Visualizing Sorting Algorithms

To better understand how sorting algorithms work, we can visualize their process using matplotlib.

```python
import matplotlib.pyplot as plt
import random

def visualize_bubble_sort(arr):
    n = len(arr)
    fig, ax = plt.subplots()
    
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
            
            ax.clear()
            ax.bar(range(len(arr)), arr)
            ax.set_title(f"Bubble Sort - Step {i*n + j + 1}")
            plt.pause(0.01)
    
    plt.show()

# Generate random data
data = [random.randint(1, 100) for _ in range(20)]
visualize_bubble_sort(data)
```

Slide 8: Implementing Merge Sort

Merge Sort is a divide-and-conquer algorithm that divides the input array into two halves, recursively sorts them, and then merges the two sorted halves. It's a stable sorting algorithm with O(n log n) time complexity.

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i, j = 0, 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Example usage
data = [38, 27, 43, 3, 9, 82, 10]
sorted_data = merge_sort(data)
print("Original:", data)
print("Sorted:", sorted_data)
```

Slide 9: Time Complexity Analysis

Understanding the time complexity of sorting algorithms helps in choosing the right algorithm for specific scenarios. Let's compare the time complexities of Bubble Sort, Quick Sort, and Merge Sort.

```python
import matplotlib.pyplot as plt
import time

def time_sort(sort_func, sizes):
    times = []
    for size in sizes:
        data = [random.randint(1, 1000) for _ in range(size)]
        start = time.time()
        sort_func(data)
        end = time.time()
        times.append(end - start)
    return times

sizes = [100, 500, 1000, 2000, 3000, 4000, 5000]
bubble_times = time_sort(bubble_sort, sizes)
quick_times = time_sort(quick_sort, sizes)
merge_times = time_sort(merge_sort, sizes)

plt.plot(sizes, bubble_times, label='Bubble Sort')
plt.plot(sizes, quick_times, label='Quick Sort')
plt.plot(sizes, merge_times, label='Merge Sort')
plt.xlabel('Input Size')
plt.ylabel('Time (seconds)')
plt.title('Sorting Algorithm Time Complexity')
plt.legend()
plt.show()
```

Slide 10: Stability in Practice

Let's demonstrate the importance of stability in sorting algorithms using a real-world example of sorting students by grade and then by name.

```python
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def __repr__(self):
        return f"Student('{self.name}', {self.grade})"

def stable_sort_students(students):
    # Sort by name first (stable)
    students_sorted_by_name = sorted(students, key=lambda s: s.name)
    # Then sort by grade (stable)
    return sorted(students_sorted_by_name, key=lambda s: s.grade, reverse=True)

students = [
    Student("Alice", 85),
    Student("Bob", 90),
    Student("Charlie", 85),
    Student("David", 80),
    Student("Eve", 90)
]

print("Original:", students)
print("Sorted (stable):", stable_sort_students(students))
```

Slide 11: Custom Sorting with Lambda Functions

Python's `sorted()` function and the `sort()` method of lists allow for custom sorting using lambda functions. This provides a flexible way to sort complex objects.

```python
class Product:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
    
    def __repr__(self):
        return f"Product('{self.name}', ${self.price}, stock: {self.stock})"

products = [
    Product("Laptop", 999.99, 10),
    Product("Mouse", 29.99, 100),
    Product("Keyboard", 59.99, 50),
    Product("Monitor", 199.99, 25)
]

# Sort by price (ascending)
by_price = sorted(products, key=lambda p: p.price)
print("Sorted by price:", by_price)

# Sort by stock (descending)
by_stock = sorted(products, key=lambda p: p.stock, reverse=True)
print("Sorted by stock:", by_stock)

# Sort by name length, then alphabetically
by_name = sorted(products, key=lambda p: (len(p.name), p.name))
print("Sorted by name length, then alphabetically:", by_name)
```

Slide 12: Real-Life Example: Playlist Sorting

Consider a music streaming application that allows users to sort their playlists using different criteria. The Strategy Pattern can be applied to implement various sorting options.

```python
class Song:
    def __init__(self, title, artist, duration, plays):
        self.title = title
        self.artist = artist
        self.duration = duration  # in seconds
        self.plays = plays
    
    def __repr__(self):
        return f"Song('{self.title}', '{self.artist}', {self.duration}s, {self.plays} plays)"

class SortByTitle(SortStrategy):
    def sort(self, songs):
        return sorted(songs, key=lambda s: s.title)

class SortByArtist(SortStrategy):
    def sort(self, songs):
        return sorted(songs, key=lambda s: s.artist)

class SortByPopularity(SortStrategy):
    def sort(self, songs):
        return sorted(songs, key=lambda s: s.plays, reverse=True)

# Usage
playlist = [
    Song("Bohemian Rhapsody", "Queen", 354, 1000000),
    Song("Stairway to Heaven", "Led Zeppelin", 482, 800000),
    Song("Imagine", "John Lennon", 183, 950000),
    Song("Like a Rolling Stone", "Bob Dylan", 369, 700000)
]

playlist_sorter = Sorter(SortByTitle())
print("Sorted by title:", playlist_sorter.sort(playlist))

playlist_sorter.strategy = SortByArtist()
print("Sorted by artist:", playlist_sorter.sort(playlist))

playlist_sorter.strategy = SortByPopularity()
print("Sorted by popularity:", playlist_sorter.sort(playlist))
```

Slide 13: Implementing Timsort

Timsort is a hybrid sorting algorithm derived from merge sort and insertion sort. It's the default sorting algorithm used in Python's `sorted()` function and `list.sort()` method.

```python
def insertion_sort(arr, left, right):
    for i in range(left + 1, right + 1):
        key_item = arr[i]
        j = i - 1
        while j >= left and arr[j] > key_item:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key_item

def merge(left, right):
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def timsort(arr):
    min_run = 32
    runs = []
    
    # Create runs
    for i in range(0, len(arr), min_run):
        runs.append(insertion_sort(arr[i:i+min_run]))
    
    # Merge runs
    while len(runs) > 1:
        runs = [merge(runs[i], runs[i+1]) for i in range(0, len(runs)-1, 2)]
    
    return runs[0] if runs else []

# Example usage
data = [38, 27, 43, 3, 9, 82, 10]
sorted_data = timsort(data)
print("Original:", data)
print("Sorted:", sorted_data)
```

Slide 14: Comparing Sorting Algorithms Performance

Let's compare the performance of different sorting algorithms we've implemented: Bubble Sort, Quick Sort, Merge Sort, and Timsort.

```python
import time
import random

def measure_sort_time(sort_func, data):
    start_time = time.time()
    sort_func(data.())
    end_time = time.time()
    return end_time - start_time

# Generate a large dataset
data_size = 10000
large_data = [random.randint(1, 1000000) for _ in range(data_size)]

# Measure sorting times
bubble_time = measure_sort_time(bubble_sort, large_data)
quick_time = measure_sort_time(quick_sort, large_data)
merge_time = measure_sort_time(merge_sort, large_data)
tim_time = measure_sort_time(timsort, large_data)
python_time = measure_sort_time(sorted, large_data)

print(f"Sorting {data_size} elements:")
print(f"Bubble Sort: {bubble_time:.6f} seconds")
print(f"Quick Sort: {quick_time:.6f} seconds")
print(f"Merge Sort: {merge_time:.6f} seconds")
print(f"Timsort: {tim_time:.6f} seconds")
print(f"Python's sorted(): {python_time:.6f} seconds")
```

Slide 15: Real-Life Example: Task Prioritization

Consider a task management system where tasks need to be sorted based on different criteria. The Strategy Pattern allows for flexible sorting methods.

```python
from datetime import datetime, timedelta

class Task:
    def __init__(self, title, priority, due_date):
        self.title = title
        self.priority = priority
        self.due_date = due_date
    
    def __repr__(self):
        return f"Task('{self.title}', priority={self.priority}, due={self.due_date.strftime('%Y-%m-%d')})"

class SortByPriority(SortStrategy):
    def sort(self, tasks):
        return sorted(tasks, key=lambda t: t.priority, reverse=True)

class SortByDueDate(SortStrategy):
    def sort(self, tasks):
        return sorted(tasks, key=lambda t: t.due_date)

# Create sample tasks
today = datetime.now()
tasks = [
    Task("Complete project", 3, today + timedelta(days=7)),
    Task("Review code", 2, today + timedelta(days=2)),
    Task("Update documentation", 1, today + timedelta(days=5)),
    Task("Fix critical bug", 4, today + timedelta(days=1))
]

# Sort tasks
task_sorter = Sorter(SortByPriority())
print("Sorted by priority:", task_sorter.sort(tasks))

task_sorter.strategy = SortByDueDate()
print("Sorted by due date:", task_sorter.sort(tasks))
```

Slide 16: Additional Resources

For those interested in diving deeper into sorting algorithms and the Strategy Pattern, here are some valuable resources:

1. "Introduction to Algorithms" by Cormen, Leiserson, Rivest, and Stein - A comprehensive book covering various sorting algorithms and their analysis.
2. "Design Patterns: Elements of Reusable Object-Oriented Software" by Gamma, Helm, Johnson, and Vlissides - The classic book on design patterns, including the Strategy Pattern.
3. "Timsort — the fastest sorting algorithm you've never heard of" by Noel Varanda ([https://arxiv.org/abs/2206.03521](https://arxiv.org/abs/2206.03521)) - An in-depth look at the Timsort algorithm used in Python and Java.
4. Python's official documentation on sorting ([https://docs.python.org/3/howto/sorting.html](https://docs.python.org/3/howto/sorting.html)) - A guide to sorting in Python, including the use of key functions and the `sorted()` built-in.

These resources provide a solid foundation for understanding sorting algorithms and design patterns in software development.

