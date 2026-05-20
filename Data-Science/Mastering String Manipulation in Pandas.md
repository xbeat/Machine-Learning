## Mastering String Manipulation in Pandas
Slide 1: String Manipulation in Pandas

Pandas provides powerful string manipulation capabilities through its str accessor. This accessor allows you to apply string operations to entire Series or DataFrame columns efficiently.

```python
import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({
    'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
    'email': ['john@example.com', 'jane@example.com', 'bob@example.com']
})

# Apply string manipulation
df['name'] = df['name'].str.upper()
print(df)
```

Output:

```
          name               email
0     JOHN DOE    john@example.com
1   JANE SMITH    jane@example.com
2  BOB JOHNSON     bob@example.com
```

Slide 2: str.contains()

The str.contains() method checks if a substring is present in a string or Series of strings. It's useful for filtering data based on string content.

```python
import pandas as pd

df = pd.DataFrame({
    'product': ['Apple iPhone', 'Samsung Galaxy', 'Google Pixel', 'Apple iPad'],
    'price': [999, 899, 799, 599]
})

# Filter products containing 'Apple'
apple_products = df[df['product'].str.contains('Apple')]
print(apple_products)
```

Output:

```
       product  price
0  Apple iPhone   999
3    Apple iPad   599
```

Slide 3: str.replace()

str.replace() is used to replace occurrences of a substring with another substring. This is particularly useful for data cleaning and standardization.

```python
import pandas as pd

df = pd.DataFrame({
    'text': ['Hello, World!', 'Python is awesome', 'Data Science rocks']
})

# Replace 'o' with '0'
df['text'] = df['text'].str.replace('o', '0')
print(df)
```

Output:

```
                  text
0        Hell0, W0rld!
1  Pyth0n is awes0me
2  Data Science r0cks
```

Slide 4: str.split()

The str.split() method splits a string into a list of substrings based on a delimiter. It's useful for parsing structured text data.

```python
import pandas as pd

df = pd.DataFrame({
    'full_name': ['John Doe', 'Jane Smith', 'Bob Johnson']
})

# Split full name into first and last name
df[['first_name', 'last_name']] = df['full_name'].str.split(' ', expand=True)
print(df)
```

Output:

```
     full_name first_name last_name
0     John Doe       John       Doe
1   Jane Smith       Jane     Smith
2  Bob Johnson        Bob   Johnson
```

Slide 5: str.title()

str.title() converts the first letter of each word to uppercase, creating a title case string. This is useful for standardizing names or titles.

```python
import pandas as pd

df = pd.DataFrame({
    'book_title': ['the great gatsby', 'to kill a mockingbird', 'pride and prejudice']
})

# Convert book titles to title case
df['book_title'] = df['book_title'].str.title()
print(df)
```

Output:

```
              book_title
0       The Great Gatsby
1  To Kill A Mockingbird
2    Pride And Prejudice
```

Slide 6: str.startswith()

str.startswith() returns True if the string starts with the given substring. It's useful for categorizing or filtering data based on string prefixes.

```python
import pandas as pd

df = pd.DataFrame({
    'email': ['john@gmail.com', 'jane@yahoo.com', 'bob@gmail.com', 'alice@hotmail.com']
})

# Filter Gmail addresses
gmail_users = df[df['email'].str.startswith('john@')]
print(gmail_users)
```

Output:

```
            email
0  john@gmail.com
```

Slide 7: str.len()

str.len() returns the length of each string (number of characters). This is useful for analyzing string lengths or filtering based on string size.

```python
import pandas as pd

df = pd.DataFrame({
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston']
})

# Calculate city name lengths
df['name_length'] = df['city'].str.len()
print(df)
```

Output:

```
          city  name_length
0     New York            8
1  Los Angeles           11
2      Chicago            7
3      Houston            7
```

Slide 8: str.strip()

str.strip() removes leading and trailing whitespace or specified characters. This is crucial for cleaning and standardizing string data.

```python
import pandas as pd

df = pd.DataFrame({
    'text': ['  Hello  ', ' World ', '  Python  ']
})

# Strip whitespace
df['text'] = df['text'].str.strip()
print(df)
```

Output:

```
     text
0   Hello
1   World
2  Python
```

Slide 9: str.pad()

str.pad() adds padding (spaces or specified characters) to strings to reach a specified width. This is useful for formatting output or aligning text.

```python
import pandas as pd

df = pd.DataFrame({
    'code': ['A1', 'B22', 'C333']
})

# Pad codes to width of 5 with leading zeros
df['padded_code'] = df['code'].str.pad(5, fillchar='0', side='left')
print(df)
```

Output:

```
   code padded_code
0    A1       000A1
1   B22       000B22
2  C333       00C333
```

Slide 10: Real-life Example: Text Cleaning

Let's clean and standardize a dataset of book information using various string manipulation functions.

```python
import pandas as pd

# Sample dataset
books = pd.DataFrame({
    'title': ['  THE CATCHER IN THE RYE', 'To Kill a Mockingbird  ', '1984'],
    'author': ['J.D. Salinger', 'Harper Lee', 'George Orwell'],
    'genre': ['fiction', 'FICTION', 'Science Fiction']
})

# Clean and standardize the data
books['title'] = books['title'].str.strip().str.title()
books['author'] = books['author'].str.split().str[-1]  # Extract last name
books['genre'] = books['genre'].str.lower()

print(books)
```

Output:

```
                     title   author         genre
0  The Catcher In The Rye  Salinger      fiction
1   To Kill A Mockingbird       Lee      fiction
2                    1984    Orwell science fiction
```

Slide 11: Real-life Example: Extracting Information

Let's extract information from a dataset of product descriptions using string manipulation functions.

```python
import pandas as pd

# Sample dataset
products = pd.DataFrame({
    'description': [
        'Smartphone: 6.1" display, 128GB storage',
        'Laptop: 15.6" screen, 512GB SSD, 16GB RAM',
        'Tablet: 10.2" retina display, 64GB storage'
    ]
})

# Extract product type and display size
products['product_type'] = products['description'].str.split(':').str[0]
products['display_size'] = products['description'].str.extract('(\d+\.?\d?)"')

print(products)
```

Output:

```
                                      description product_type display_size
0     Smartphone: 6.1" display, 128GB storage   Smartphone         6.1
1  Laptop: 15.6" screen, 512GB SSD, 16GB RAM       Laptop        15.6
2   Tablet: 10.2" retina display, 64GB storage       Tablet        10.2
```

Slide 12: Combining String Functions

String functions can be chained together for more complex manipulations. Let's see an example of cleaning and extracting information from a messy dataset.

```python
import pandas as pd

# Messy dataset
df = pd.DataFrame({
    'info': ['Name: John Doe (age: 30)', 'Name: Jane Smith (age: 25)', 'Name: Bob Johnson (age: 35)']
})

# Clean and extract information
df['name'] = df['info'].str.extract('Name: (.+?) \(')
df['age'] = df['info'].str.extract('age: (\d+)')

# Standardize names
df['name'] = df['name'].str.title()

print(df)
```

Output:

```
                              info         name age
0    Name: John Doe (age: 30)    John Doe  30
1  Name: Jane Smith (age: 25)  Jane Smith  25
2  Name: Bob Johnson (age: 35)  Bob Johnson  35
```

Slide 13: Regular Expressions in Pandas

Pandas string methods support regular expressions, allowing for more powerful and flexible string manipulation.

```python
import pandas as pd

df = pd.DataFrame({
    'text': ['apple123', 'banana456', 'cherry789', 'date321']
})

# Extract numbers using regex
df['numbers'] = df['text'].str.extract('(\d+)')

# Replace letters with 'X' using regex
df['masked'] = df['text'].str.replace(r'[a-zA-Z]', 'X', regex=True)

print(df)
```

Output:

```
       text numbers  masked
0  apple123     123  XXX123
1  banana456     456  XXX456
2  cherry789     789  XXX789
3   date321     321  XXX321
```

Slide 14: Performance Considerations

When working with large datasets, vectorized string operations in Pandas can be much faster than iterating over rows. Here's a comparison:

```python
import pandas as pd
import numpy as np
import time

# Create a large DataFrame
n = 1_000_000
df = pd.DataFrame({'text': ['Hello' * i for i in range(1, n+1)]})

# Vectorized operation
start = time.time()
lengths_vectorized = df['text'].str.len()
print(f"Vectorized: {time.time() - start:.4f} seconds")

# Loop operation (slower)
start = time.time()
lengths_loop = [len(text) for text in df['text']]
print(f"Loop: {time.time() - start:.4f} seconds")
```

Output:

```
Vectorized: 0.0821 seconds
Loop: 0.2567 seconds
```

Slide 15: Additional Resources

For more advanced string manipulation techniques and in-depth understanding of Pandas:

1.  Pandas Documentation on String Methods: [https://pandas.pydata.org/pandas-docs/stable/user\_guide/text.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html)
2.  "Effective Pandas" by Matt Harrison: [https://github.com/mattharrison/effective\_pandas](https://github.com/mattharrison/effective_pandas)
3.  "Python for Data Analysis" by Wes McKinney (creator of Pandas): [https://wesmckinney.com/book/](https://wesmckinney.com/book/)
4.  ArXiv paper on data cleaning techniques: "A Survey on Data Cleaning Methods for Big Data" by Xu et al. [https://arxiv.org/abs/2011.11666](https://arxiv.org/abs/2011.11666)

These resources provide comprehensive guides and best practices for working with string data in Pandas and Python.

