## Symmetric Indexing in Python Flexible Array Manipulation
Slide 1: Introduction to Symmetric Indexing

Symmetric indexing is a powerful technique in Python that allows for flexible and intuitive array manipulation. It enables users to access and modify array elements using both positive and negative indices, providing a seamless way to work with data from both ends of an array.

```python
# Example of symmetric indexing
arr = [1, 2, 3, 4, 5]
print(arr[1])   # Output: 2
print(arr[-1])  # Output: 5
print(arr[-2])  # Output: 4
```

Slide 2: Basic Syntax of Symmetric Indexing

In Python, array indices start at 0 for the first element and go up to n-1 for an array of n elements. Negative indices start from -1 for the last element and go down to -n for the first element. This allows for intuitive access to elements from both ends of the array.

```python
def demonstrate_symmetric_indexing(arr):
    print(f"Array: {arr}")
    print(f"First element (arr[0]): {arr[0]}")
    print(f"Last element (arr[-1]): {arr[-1]}")
    print(f"Second element (arr[1]): {arr[1]}")
    print(f"Second-to-last element (arr[-2]): {arr[-2]}")

sample_array = [10, 20, 30, 40, 50]
demonstrate_symmetric_indexing(sample_array)
```

Slide 3: Slicing with Symmetric Indexing

Symmetric indexing can be combined with slicing to extract subarrays. The syntax for slicing is arr\[start:end:step\], where all parameters are optional and can be positive or negative.

```python
def slice_array(arr):
    print(f"Original array: {arr}")
    print(f"First three elements (arr[:3]): {arr[:3]}")
    print(f"Last three elements (arr[-3:]): {arr[-3:]}")
    print(f"Every other element (arr[::2]): {arr[::2]}")
    print(f"Reverse array (arr[::-1]): {arr[::-1]}")

sample_array = [1, 2, 3, 4, 5, 6, 7, 8, 9]
slice_array(sample_array)
```

Slide 4: Modifying Arrays with Symmetric Indexing

Symmetric indexing allows for intuitive modification of array elements. You can assign new values to specific indices or replace entire slices of an array.

```python
def modify_array(arr):
    print(f"Original array: {arr}")
    
    arr[0] = 100  # Modify first element
    arr[-1] = 900  # Modify last element
    print(f"After modifying first and last elements: {arr}")
    
    arr[1:4] = [200, 300, 400]  # Replace a slice
    print(f"After replacing elements 1-3: {arr}")
    
    arr[:] = [i * 10 for i in range(1, 6)]  # Replace entire array
    print(f"After replacing entire array: {arr}")

sample_array = [1, 2, 3, 4, 5, 6, 7, 8, 9]
modify_array(sample_array)
```

Slide 5: Real-life Example: Text Processing

Symmetric indexing is particularly useful in text processing tasks, such as reversing words or extracting substrings. Here's an example of how it can be used to create a simple word reversal function.

```python
def reverse_words(sentence):
    words = sentence.split()
    reversed_words = [word[::-1] for word in words]
    return ' '.join(reversed_words)

sample_sentence = "Python is awesome"
reversed_sentence = reverse_words(sample_sentence)
print(f"Original sentence: {sample_sentence}")
print(f"Reversed words: {reversed_sentence}")
```

Slide 6: Advanced Slicing Techniques

Symmetric indexing allows for more complex slicing operations, such as stepping through an array with a specific interval or reversing only a portion of an array.

```python
def advanced_slicing(arr):
    print(f"Original array: {arr}")
    print(f"Every third element (arr[::3]): {arr[::3]}")
    print(f"Reverse slice (arr[3:7][::-1]): {arr[3:7][::-1]}")
    print(f"Last 5 elements in reverse (arr[-1:-6:-1]): {arr[-1:-6:-1]}")

sample_array = list(range(1, 11))
advanced_slicing(sample_array)
```

Slide 7: Symmetric Indexing with Numpy Arrays

Numpy, a popular library for numerical computing in Python, supports symmetric indexing and extends its capabilities to multi-dimensional arrays.

```python
import numpy as np

def numpy_symmetric_indexing():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"Original 2D array:\n{arr}")
    print(f"Last row (arr[-1]): {arr[-1]}")
    print(f"First column (arr[:, 0]): {arr[:, 0]}")
    print(f"Diagonal elements (arr.diagonal()): {arr.diagonal()}")
    print(f"Reverse both dimensions (arr[::-1, ::-1]):\n{arr[::-1, ::-1]}")

numpy_symmetric_indexing()
```

Slide 8: Performance Considerations

While symmetric indexing is powerful and intuitive, it's important to consider its performance implications, especially when working with large datasets.

```python
import timeit

def compare_indexing_performance(size):
    setup = f"import numpy as np; arr = np.arange({size})"
    
    positive_indexing = timeit.timeit("arr[size-1]", setup=setup, number=1000000)
    negative_indexing = timeit.timeit("arr[-1]", setup=setup, number=1000000)
    
    print(f"Array size: {size}")
    print(f"Positive indexing time: {positive_indexing:.6f} seconds")
    print(f"Negative indexing time: {negative_indexing:.6f} seconds")

compare_indexing_performance(1000000)
```

Slide 9: Error Handling in Symmetric Indexing

When using symmetric indexing, it's crucial to handle potential IndexError exceptions that may occur when accessing out-of-range indices.

```python
def safe_indexing(arr, index):
    try:
        return arr[index]
    except IndexError:
        return f"Index {index} is out of range for array of length {len(arr)}"

sample_array = [1, 2, 3, 4, 5]
print(safe_indexing(sample_array, 2))   # Valid index
print(safe_indexing(sample_array, 10))  # Out of range positive index
print(safe_indexing(sample_array, -10)) # Out of range negative index
```

Slide 10: Real-life Example: Image Processing

Symmetric indexing is particularly useful in image processing tasks, such as image rotation or flipping. Here's an example using the Pillow library to flip an image horizontally and vertically.

```python
from PIL import Image
import numpy as np

def flip_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Flip horizontally
    horizontal_flip = img_array[:, ::-1]
    
    # Flip vertically
    vertical_flip = img_array[::-1, :]
    
    # Convert back to images
    Image.fromarray(horizontal_flip).save('horizontal_flip.jpg')
    Image.fromarray(vertical_flip).save('vertical_flip.jpg')
    
    print("Images flipped and saved.")

# Note: Replace 'image.jpg' with an actual image file path
flip_image('image.jpg')
```

Slide 11: Symmetric Indexing in Custom Classes

You can implement symmetric indexing in your custom classes by defining the `__getitem__` and `__setitem__` methods. This allows your objects to behave like built-in sequences.

```python
class SymmetricList:
    def __init__(self, data):
        self.data = list(data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value
    
    def __len__(self):
        return len(self.data)
    
    def __str__(self):
        return str(self.data)

sym_list = SymmetricList([1, 2, 3, 4, 5])
print(f"Original: {sym_list}")
print(f"First element: {sym_list[0]}")
print(f"Last element: {sym_list[-1]}")
sym_list[-2] = 10
print(f"After modification: {sym_list}")
```

Slide 12: Symmetric Indexing in String Manipulation

Strings in Python are sequences, which means they support symmetric indexing. This feature is particularly useful for various string manipulation tasks.

```python
def string_manipulation(text):
    print(f"Original text: {text}")
    print(f"First character: {text[0]}")
    print(f"Last character: {text[-1]}")
    print(f"Reversed text: {text[::-1]}")
    print(f"Every other character: {text[::2]}")
    print(f"Last 5 characters: {text[-5:]}")
    print(f"Text without first and last character: {text[1:-1]}")

sample_text = "Python Symmetric Indexing"
string_manipulation(sample_text)
```

Slide 13: Symmetric Indexing in List Comprehensions

Symmetric indexing can be combined with list comprehensions to create powerful and concise data transformations.

```python
def list_comprehension_examples(data):
    print(f"Original data: {data}")
    
    # Reverse every other element
    result1 = [x[::-1] if i % 2 == 0 else x for i, x in enumerate(data)]
    print(f"Reverse every other element: {result1}")
    
    # Create pairs of adjacent elements
    result2 = [data[i:i+2] for i in range(0, len(data)-1, 2)]
    print(f"Pairs of adjacent elements: {result2}")
    
    # Interleave first half with reversed second half
    mid = len(data) // 2
    result3 = [x for pair in zip(data[:mid], data[:mid-1:-1]) for x in pair]
    print(f"Interleaved result: {result3}")

sample_data = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
list_comprehension_examples(sample_data)
```

Slide 14: Conclusion and Best Practices

Symmetric indexing is a powerful feature in Python that enhances readability and flexibility when working with sequences. To make the most of it:

1. Use negative indices for counting from the end of sequences.
2. Leverage slicing for efficient subarray operations.
3. Be mindful of performance implications, especially with large datasets.
4. Handle potential IndexError exceptions in your code.
5. Experiment with combining symmetric indexing and other Python features for concise and expressive code.

Remember, while symmetric indexing is intuitive, it's essential to write clear and maintainable code that others (including your future self) can easily understand.

Slide 15: Additional Resources

For those interested in diving deeper into symmetric indexing and related topics, here are some valuable resources:

1. Python's official documentation on sequence types: [https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range)
2. "Fluent Python" by Luciano Ramalho, which covers advanced Python concepts including sequence manipulation: [https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)
3. NumPy documentation on array indexing: [https://numpy.org/doc/stable/user/basics.indexing.html](https://numpy.org/doc/stable/user/basics.indexing.html)
4. ArXiv paper on efficient array operations in scientific computing: [https://arxiv.org/abs/1102.1523](https://arxiv.org/abs/1102.1523)

These resources provide in-depth information and advanced techniques related to symmetric indexing and sequence manipulation in Python.

