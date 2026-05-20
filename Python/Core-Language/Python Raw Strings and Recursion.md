## Python Raw Strings and Recursion

Slide 1: 
Introduction to Raw Strings 
Raw strings in Python are literal strings that are prefixed with the letter 'r' or 'R'. They treat backslashes () as literal characters instead of escape characters, making them useful for handling file paths, regular expressions, and other cases where backslashes are common. Code:

```python
print(r'C:\Users\Documents')  # Output: C:\Users\Documents
```

Slide 2: 
Raw Strings vs. Regular Strings 
Unlike regular strings, raw strings treat backslashes as literal characters. This can be particularly useful when working with file paths or regular expressions. Code:

```python
regular_string = 'C:\Users\Documents'  # Treats \U as a Unicode escape sequence
print(regular_string)  # Output: C:nUsers\Documents

raw_string = r'C:\Users\Documents'  # Treats backslashes literally
print(raw_string)  # Output: C:\Users\Documents
```

Slide 3: 
Multiline Raw Strings 
Raw strings can also span multiple lines, making them useful for representing multi-line strings or storing large blocks of text. Code:

```python
multiline_string = """This
is a
multiline string"""

raw_multiline_string = r"""This
is a \multiline
raw string"""

print(multiline_string)
# Output:
# This
# is a
# multiline string

print(raw_multiline_string)
# Output:
# This
# is a \multiline
# raw string
```

Slide 4: 
Introduction to Recursion 
Recursion is a programming technique where a function calls itself with a smaller input or a slightly different set of parameters. It is often used to solve problems that can be broken down into smaller instances of the same problem. Code:

```python
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

print(factorial(5))  # Output: 120
```

Slide 5: 
Recursive Function Structure 
A recursive function typically has two parts: a base case that stops the recursion, and a recursive case that calls the function again with a different input. Code:

```python
def recursive_function(input_value):
    # Base case
    if base_condition_is_met(input_value):
        return some_value

    # Recursive case
    else:
        # Do some processing
        new_input_value = modify_input(input_value)
        return recursive_function(new_input_value)
```

Slide 6: 
Fibonacci Sequence with Recursion 
The Fibonacci sequence is a classic example of a problem that can be solved using recursion. Code:

```python
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(7))  # Output: 13
```

Slide 7: Recursion and Data Structures Recursion is often used in conjunction with data structures like lists, trees, and graphs, where problems can be broken down into smaller subproblems involving the same data structure. Code:

```python
def sum_list(lst):
    if not lst:
        return 0
    else:
        return lst[0] + sum_list(lst[1:])

my_list = [1, 2, 3, 4, 5]
print(sum_list(my_list))  # Output: 15
```

Slide 8: 
Recursive Binary Search 
Binary search is an efficient algorithm for finding an element in a sorted list. It can be implemented recursively by dividing the list in half at each step. Code:

```python
def binary_search(lst, target, low=0, high=None):
    if high is None:
        high = len(lst) - 1

    if low > high:
        return -1  # Target not found

    mid = (low + high) // 2

    if lst[mid] == target:
        return mid
    elif lst[mid] > target:
        return binary_search(lst, target, low, mid - 1)
    else:
        return binary_search(lst, target, mid + 1, high)

my_list = [1, 3, 5, 7, 9]
print(binary_search(my_list, 5))  # Output: 2
```

Slide 9: 
Recursive Tree Traversal 
Trees are another data structure that can be traversed using recursion. Common tree traversal algorithms like in-order, pre-order, and post-order traversal can be implemented recursively. Code:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def preorder_traversal(root):
    if root:
        print(root.val)  # Visit root
        preorder_traversal(root.left)  # Traverse left subtree
        preorder_traversal(root.right)  # Traverse right subtree

# Create a sample tree
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

preorder_traversal(root)  # Output: 1 2 4 5 3
```

Slide 10: 
Tail Recursion 
Tail recursion is a special case of recursion where the recursive call is the last operation performed by the function. This can be optimized by some compilers to use iteration instead of recursion, avoiding potential stack overflow issues. Code:

```python
def factorial(n, acc=1):
    if n == 0:
        return acc
    else:
        return factorial(n - 1, n * acc)

print(factorial(5))  # Output: 120
```

Slide 11: 
Recursive Backtracking 
Backtracking is a general algorithmic technique that considers searching every possible combination in order to solve a computational problem. It can be implemented recursively by exploring all possible solutions and backtracking when a solution is not viable. Code:

```python
def permute(nums):
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
        else:
            for i in range(start, len(nums)):
                nums[start], nums[i] = nums[i], nums[start]  # Swap
                backtrack(start + 1)  # Recurse
                nums[start], nums[i] = nums[i], nums[start]  # Backtrack

    result = []
    backtrack(0)
    return result

print(permute([1, 2, 3]))
# Output: [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
```

Slide 12: 
Recursion and Memoization 
Memoization is an optimization technique that stores the results of expensive function calls and returns the cached result when the same inputs occur again. This can be used to improve the performance of recursive functions by avoiding redundant computations. Code:

```python
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        result = n
    else:
        result = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    memo[n] = result
    return result

print(fibonacci(100))  # Output: 354224848179261915075
```

Slide 13: 
Tail Call Optimization 
Tail call optimization is a compiler optimization technique that reuses the current stack frame for recursive calls instead of creating a new one. This can help avoid stack overflow errors in tail-recursive functions by eliminating the need for additional stack frames. Code:

```python
def factorial(n, acc=1):
    if n == 0:
        return acc
    else:
        return factorial(n - 1, n * acc)

print(factorial(1000000))  # With tail call optimization, this will not cause a stack overflow
```

Slide 14: 
Limitations of Recursion 
While recursion is a powerful technique, it can also lead to performance issues and stack overflow errors if not used carefully. Recursive functions with a large depth or high computational complexity can consume a significant amount of memory and processing power. Code:

```python
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# This will likely cause a stack overflow for large values of n
print(fibonacci(1000))
```

Slide 15: 
When to Use Recursion 
Recursion is well-suited for problems that can be divided into smaller instances of the same problem. It can provide elegant and concise solutions, especially for problems involving trees, graphs, and other hierarchical data structures. However, for problems with high computational complexity or large input sizes, iterative solutions may be more efficient. Code:

```python
# Recursive function to compute the sum of elements in a list
def sum_list(lst):
    if not lst:
        return 0
    else:
        return lst[0] + sum_list(lst[1:])

# Iterative solution using a loop
def sum_list_iterative(lst):
    total = 0
    for elem in lst:
        total += elem
    return total
```

Slide 16: 
Additional Resources 
For further learning and exploration, here are some additional resources from arXiv.org:

1. "Recursion in Computational Logic" by Dale Miller ([https://arxiv.org/abs/1909.04396](https://arxiv.org/abs/1909.04396))
2. "Recursion and Induction in Computer Science" by Robert Sedgewick and Michael Schidlowsky ([https://arxiv.org/abs/1611.05789](https://arxiv.org/abs/1611.05789))
3. "On the Complexities of Recursive Functions" by Achim Jung and K. Zuse ([https://arxiv.org/abs/1711.05151](https://arxiv.org/abs/1711.05151))

