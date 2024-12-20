## Remembering Python Built-in Functions
Slide 1: Using dir() to Explore Class Methods

The dir() function in Python is a powerful tool for discovering available methods and attributes of an object or class. It's particularly useful when you can't recall a specific method name but remember the class it belongs to.

Slide 2: Source Code for Using dir() to Explore Class Methods

```python
# Example: Using dir() with a string object
my_string = "Hello, World!"
string_methods = dir(my_string)

print("Some string methods:")
for method in string_methods[:5]:  # Print first 5 methods
    print(method)

# Example: Using dir() with the str class itself
str_class_methods = dir(str)

print("\nSome str class methods:")
for method in str_class_methods[:5]:  # Print first 5 methods
    print(method)
```

Slide 3: Results for: Using dir() to Explore Class Methods

```
Some string methods:
__add__
__class__
__contains__
__delattr__
__dir__

Some str class methods:
__add__
__class__
__contains__
__delattr__
__dir__
```

Slide 4: Using **doc** for Method Documentation

The **doc** attribute provides access to the docstring of a method or function. This is invaluable when you need to quickly understand how a method works or what parameters it accepts.

Slide 5: Source Code for Using **doc** for Method Documentation

```python
# Example: Accessing docstring of the str.format_map() method
print(str.format_map.__doc__)

# Example: Accessing docstring of a custom function
def custom_function(x, y):
    """This function adds two numbers and returns the result."""
    return x + y

print(custom_function.__doc__)
```

Slide 6: Results for: Using **doc** for Method Documentation

```
S.format_map(mapping) -> str

Return a formatted version of S, using substitutions from mapping.
The substitutions are identified by braces ('{' and '}').

This function adds two numbers and returns the result.
```

Slide 7: Real-Life Example: Exploring List Methods

Let's explore the methods available for Python lists, which are commonly used in coding contests for managing collections of data.

Slide 8: Source Code for Real-Life Example: Exploring List Methods

```python
# Create a sample list
my_list = [1, 2, 3, 4, 5]

# Get all methods of the list
list_methods = dir(my_list)

# Print methods that don't start with '__'
print("List methods:")
for method in list_methods:
    if not method.startswith('__'):
        print(method)

# Get documentation for the 'append' method
print("\nDocumentation for append method:")
print(my_list.append.__doc__)
```

Slide 9: Results for: Real-Life Example: Exploring List Methods

```
List methods:
append
clear
copy
count
extend
index
insert
pop
remove
reverse
sort

Documentation for append method:
Append object to the end of the list.
```

Slide 10: Real-Life Example: String Manipulation

String manipulation is crucial in many coding challenges. Let's explore some string methods using dir() and **doc**.

Slide 11: Source Code for Real-Life Example: String Manipulation

```python
# Create a sample string
text = "Python is awesome!"

# Get all methods of the string
string_methods = dir(text)

# Print methods that don't start with '__'
print("String methods:")
for method in string_methods:
    if not method.startswith('__'):
        print(method)

# Get documentation for the 'split' method
print("\nDocumentation for split method:")
print(text.split.__doc__)

# Demonstrate usage of split method
words = text.split()
print("\nSplit result:", words)
```

Slide 12: Results for: Real-Life Example: String Manipulation

```
String methods:
capitalize
casefold
center
count
encode
endswith
expandtabs
find
format
format_map
index
isalnum
isalpha
isascii
isdecimal
isdigit
isidentifier
islower
isnumeric
isprintable
isspace
istitle
isupper
join
ljust
lower
lstrip
maketrans
partition
removeprefix
removesuffix
replace
rfind
rindex
rjust
rpartition
rsplit
rstrip
split
splitlines
startswith
strip
swapcase
title
translate
upper
zfill

Documentation for split method:
Return a list of the substrings in the string, using sep as the separator string.

  sep
    The separator used to split the string.
    When set to None (the default value), will split on any whitespace
    character (including \\n \\r \\t \\f and spaces) and will discard
    empty strings from the result.
  maxsplit
    Maximum number of splits (starting from the left).
    -1 (the default value) means no limit.

Note, str.split() is mainly useful for data that has been intentionally
delimited.  With natural text that includes punctuation, consider using
the regular expression module.

Split result: ['Python', 'is', 'awesome!']
```

Slide 13: Combining dir() and **doc** for Efficient Coding

By combining dir() and **doc**, you can quickly explore and understand available methods, enhancing your coding efficiency during contests or everyday programming tasks.

Slide 14: Source Code for Combining dir() and **doc** for Efficient Coding

```python
def explore_methods(obj, prefix=''):
    """Explore methods of an object that start with a given prefix."""
    methods = [method for method in dir(obj) if method.startswith(prefix)]
    
    for method in methods:
        print(f"Method: {method}")
        print(f"Documentation: {getattr(obj, method).__doc__}\n")

# Example usage with string methods starting with 'is'
explore_methods("", 'is')
```

Slide 15: Results for: Combining dir() and **doc** for Efficient Coding

```
Method: isalnum
Documentation: Return True if the string is an alpha-numeric string, False otherwise.

A string is alpha-numeric if all characters in the string are alpha-numeric and
there is at least one character in the string.

Method: isalpha
Documentation: Return True if the string is an alphabetic string, False otherwise.

A string is alphabetic if all characters in the string are alphabetic and there
is at least one character in the string.

Method: isascii
Documentation: Return True if all characters in the string are ASCII, False otherwise.

ASCII characters have code points in the range U+0000-U+007F.
Empty string is ASCII too.

Method: isdecimal
Documentation: Return True if the string is a decimal string, False otherwise.

A string is a decimal string if all characters in the string are decimal and
there is at least one character in the string.

Method: isdigit
Documentation: Return True if the string is a digit string, False otherwise.

A string is a digit string if all characters in the string are digits and there
is at least one character in the string.

Method: isidentifier
Documentation: Return True if the string is a valid Python identifier, False otherwise.

Call keyword.iskeyword(s) to test whether string s is a reserved identifier,
such as "def" or "class".

Method: islower
Documentation: Return True if the string is a lowercase string, False otherwise.

A string is lowercase if all cased characters in the string are lowercase and
there is at least one cased character in the string.

Method: isnumeric
Documentation: Return True if the string is a numeric string, False otherwise.

A string is numeric if all characters in the string are numeric and there is at
least one character in the string.

Method: isprintable
Documentation: Return True if the string is printable, False otherwise.

A string is printable if all of its characters are considered printable in
repr() or if it is empty.

Method: isspace
Documentation: Return True if the string is a whitespace string, False otherwise.

A string is whitespace if all characters in the string are whitespace and there
is at least one character in the string.

Method: istitle
Documentation: Return True if the string is a title-cased string, False otherwise.

In a title-cased string, upper- and title-case characters may only
follow uncased characters and lowercase characters only cased ones.

Method: isupper
Documentation: Return True if the string is an uppercase string, False otherwise.

A string is uppercase if all cased characters in the string are uppercase and
there is at least one cased character in the string.
```

Slide 16: Additional Resources

For more information on Python's built-in functions and object-oriented programming concepts:

1.  Python Official Documentation: [https://docs.python.org/3/](https://docs.python.org/3/)
2.  "Object-Oriented Programming in Python" by Goldwasser et al. (2013): arXiv:1303.6207 \[cs.PL\]

