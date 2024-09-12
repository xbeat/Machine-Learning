## Introduction to LLM-based Text-to-SQL Integration
Slide 1: Introduction to LLM-based Text-to-SQL Integration

Text-to-SQL integration using Large Language Models (LLMs) is a powerful approach to bridge natural language queries with database operations. This technology allows users to interact with databases using everyday language, making data access more intuitive and accessible. In this presentation, we'll explore how to integrate LLM-based solutions into text-to-SQL systems using Python, covering key concepts, techniques, and practical implementations.

```python
import openai
import sqlite3

def natural_language_to_sql(query):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Convert this natural language query to SQL: {query}",
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Example usage
nl_query = "Show me all employees who joined in the last 3 months"
sql_query = natural_language_to_sql(nl_query)
print(f"Generated SQL: {sql_query}")
```

Slide 2: Setting Up the Environment

To begin integrating LLM-based solutions into text-to-SQL systems, we need to set up our Python environment. This involves installing necessary libraries and configuring API access for the chosen LLM service. In this example, we'll use OpenAI's GPT-3 model, but the concepts can be applied to other LLMs as well.

```python
# Install required libraries
!pip install openai sqlite3

# Import necessary modules
import openai
import sqlite3
import os

# Set up OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Test API connection
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt="Hello, world!",
    max_tokens=5
)
print("API test response:", response.choices[0].text.strip())
```

Slide 3: Creating a Sample Database

To demonstrate text-to-SQL integration, we'll create a sample SQLite database. This database will store information about books, including their titles, authors, and publication years. We'll use this database throughout our examples to showcase how natural language queries can be converted to SQL statements.

```python
# Create a sample SQLite database
conn = sqlite3.connect('books.db')
cursor = conn.cursor()

# Create a table for books
cursor.execute('''
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    year INTEGER
)
''')

# Insert sample data
sample_books = [
    ('To Kill a Mockingbird', 'Harper Lee', 1960),
    ('1984', 'George Orwell', 1949),
    ('Pride and Prejudice', 'Jane Austen', 1813),
    ('The Great Gatsby', 'F. Scott Fitzgerald', 1925)
]

cursor.executemany('INSERT INTO books (title, author, year) VALUES (?, ?, ?)', sample_books)
conn.commit()

# Verify data insertion
cursor.execute('SELECT * FROM books')
print("Sample database contents:")
for row in cursor.fetchall():
    print(row)

conn.close()
```

Slide 4: Implementing the Text-to-SQL Conversion

The core of our LLM-based text-to-SQL system is the function that converts natural language queries to SQL statements. We'll use OpenAI's GPT-3 model to perform this conversion. The function takes a natural language query as input and returns the corresponding SQL statement.

```python
def natural_language_to_sql(query):
    prompt = f"""
    Given the following SQLite table structure:
    
    CREATE TABLE books (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        author TEXT NOT NULL,
        year INTEGER
    )
    
    Convert this natural language query to SQL: {query}
    """
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        temperature=0.3
    )
    
    return response.choices[0].text.strip()

# Test the function
nl_query = "What are the titles of books published after 1950?"
sql_query = natural_language_to_sql(nl_query)
print(f"Natural Language Query: {nl_query}")
print(f"Generated SQL: {sql_query}")
```

Slide 5: Executing SQL Queries and Returning Results

Once we have converted the natural language query to SQL, we need to execute it against our database and return the results. We'll create a function that takes the generated SQL query, executes it, and returns the results in a formatted manner.

```python
def execute_sql_query(sql_query):
    conn = sqlite3.connect('books.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        # Format results as a list of dictionaries
        formatted_results = [dict(zip(columns, row)) for row in results]
        
        return formatted_results
    except sqlite3.Error as e:
        return f"An error occurred: {e}"
    finally:
        conn.close()

# Test the function
sql_query = "SELECT title, year FROM books WHERE year > 1950"
results = execute_sql_query(sql_query)
print("Query results:")
for result in results:
    print(result)
```

Slide 6: Combining Text-to-SQL Conversion and Query Execution

Now that we have implemented both the text-to-SQL conversion and SQL query execution, we can combine them into a single function. This function will take a natural language query, convert it to SQL, execute the query, and return the results.

```python
def process_natural_language_query(nl_query):
    sql_query = natural_language_to_sql(nl_query)
    results = execute_sql_query(sql_query)
    
    return {
        "natural_language_query": nl_query,
        "sql_query": sql_query,
        "results": results
    }

# Test the combined function
nl_query = "Which books were written by Jane Austen?"
response = process_natural_language_query(nl_query)

print(f"Natural Language Query: {response['natural_language_query']}")
print(f"Generated SQL: {response['sql_query']}")
print("Results:")
for result in response['results']:
    print(result)
```

Slide 7: Handling Complex Queries and Joins

LLM-based text-to-SQL systems can handle complex queries involving multiple tables and joins. To demonstrate this, let's add a new table for book categories and modify our conversion function to handle more complex scenarios.

```python
# Add a new table for book categories
conn = sqlite3.connect('books.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS book_categories (
    book_id INTEGER,
    category_id INTEGER,
    FOREIGN KEY (book_id) REFERENCES books (id),
    FOREIGN KEY (category_id) REFERENCES categories (id)
)
''')

# Insert sample data
categories = [('Fiction',), ('Non-fiction',), ('Classic',)]
cursor.executemany('INSERT INTO categories (name) VALUES (?)', categories)

book_categories = [(1, 1), (1, 3), (2, 1), (2, 3), (3, 1), (3, 3), (4, 1), (4, 3)]
cursor.executemany('INSERT INTO book_categories (book_id, category_id) VALUES (?, ?)', book_categories)

conn.commit()
conn.close()

# Update the natural_language_to_sql function to handle complex queries
def natural_language_to_sql(query):
    prompt = f"""
    Given the following SQLite table structure:
    
    CREATE TABLE books (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        author TEXT NOT NULL,
        year INTEGER
    )
    
    CREATE TABLE categories (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    )
    
    CREATE TABLE book_categories (
        book_id INTEGER,
        category_id INTEGER,
        FOREIGN KEY (book_id) REFERENCES books (id),
        FOREIGN KEY (category_id) REFERENCES categories (id)
    )
    
    Convert this natural language query to SQL: {query}
    """
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=200,
        temperature=0.3
    )
    
    return response.choices[0].text.strip()

# Test with a complex query
nl_query = "List all classic books with their authors and categories"
response = process_natural_language_query(nl_query)
print(f"Natural Language Query: {response['natural_language_query']}")
print(f"Generated SQL: {response['sql_query']}")
print("Results:")
for result in response['results']:
    print(result)
```

Slide 8: Handling Ambiguity and Clarification

Natural language queries can sometimes be ambiguous or lack necessary information. In such cases, we need to implement a system for seeking clarification from the user. Let's create a function that detects potential ambiguities and prompts the user for additional information.

```python
def detect_ambiguity(nl_query, sql_query):
    prompt = f"""
    Given the natural language query: "{nl_query}"
    And the generated SQL query: "{sql_query}"
    
    Is there any ambiguity or missing information? If yes, what clarification is needed?
    If no ambiguity, respond with "No ambiguity detected."
    """
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        temperature=0.3
    )
    
    return response.choices[0].text.strip()

def process_query_with_clarification(nl_query):
    sql_query = natural_language_to_sql(nl_query)
    ambiguity = detect_ambiguity(nl_query, sql_query)
    
    if ambiguity != "No ambiguity detected.":
        print(f"Clarification needed: {ambiguity}")
        clarification = input("Please provide additional information: ")
        nl_query += " " + clarification
        sql_query = natural_language_to_sql(nl_query)
    
    results = execute_sql_query(sql_query)
    return {
        "natural_language_query": nl_query,
        "sql_query": sql_query,
        "results": results
    }

# Test the ambiguity detection and clarification process
nl_query = "Show me all books published recently"
response = process_query_with_clarification(nl_query)
print(f"Final Natural Language Query: {response['natural_language_query']}")
print(f"Generated SQL: {response['sql_query']}")
print("Results:")
for result in response['results']:
    print(result)
```

Slide 9: Implementing Query Validation and Safety Checks

To ensure the safety and integrity of our database, it's crucial to implement validation and safety checks on the generated SQL queries. This helps prevent potential SQL injection attacks and ensures that the queries conform to our expected patterns.

```python
import re

def validate_sql_query(sql_query):
    # Check for potential SQL injection patterns
    dangerous_patterns = [
        r'\bDROP\b',
        r'\bDELETE\b',
        r'\bTRUNCATE\b',
        r'\bALTER\b',
        r'\bCREATE\b',
        r';--',
        r'1=1'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, sql_query, re.IGNORECASE):
            return False, f"Potentially dangerous pattern detected: {pattern}"
    
    # Ensure the query only references our known tables
    allowed_tables = ['books', 'categories', 'book_categories']
    table_pattern = r'\bFROM\s+(\w+)\b'
    referenced_tables = re.findall(table_pattern, sql_query, re.IGNORECASE)
    
    for table in referenced_tables:
        if table.lower() not in allowed_tables:
            return False, f"Query references unauthorized table: {table}"
    
    return True, "Query validated successfully"

# Modify the process_natural_language_query function to include validation
def process_natural_language_query(nl_query):
    sql_query = natural_language_to_sql(nl_query)
    is_valid, validation_message = validate_sql_query(sql_query)
    
    if not is_valid:
        return {
            "natural_language_query": nl_query,
            "sql_query": sql_query,
            "error": validation_message
        }
    
    results = execute_sql_query(sql_query)
    return {
        "natural_language_query": nl_query,
        "sql_query": sql_query,
        "results": results
    }

# Test the validation process
nl_query = "Show me all books and drop the categories table"
response = process_natural_language_query(nl_query)
print(f"Natural Language Query: {response['natural_language_query']}")
print(f"Generated SQL: {response['sql_query']}")
if 'error' in response:
    print(f"Error: {response['error']}")
else:
    print("Results:")
    for result in response['results']:
        print(result)
```

Slide 10: Implementing Query Optimization

LLM-generated SQL queries may not always be optimized for performance. We can implement a query optimization step to improve the efficiency of our generated queries. This involves analyzing the query structure and making adjustments to enhance performance.

```python
import sqlparse

def optimize_sql_query(sql_query):
    # Parse the SQL query
    parsed = sqlparse.parse(sql_query)[0]
    
    # Optimize SELECT * queries
    if parsed.get_type() == 'SELECT':
        select_tokens = [token for token in parsed.tokens if isinstance(token, sqlparse.sql.IdentifierList)]
        if any(token.value == '*' for token in select_tokens):
            # Replace SELECT * with specific column names
            table_name = [token.value for token in parsed.tokens if isinstance(token, sqlparse.sql.Identifier)][0]
            columns = get_table_columns(table_name)
            optimized_query = sql_query.replace('*', ', '.join(columns))
            return optimized_query
    
    # Add more optimization rules here
    
    return sql_query

def get_table_columns(table_name):
    conn = sqlite3.connect('books.db')
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()
    return columns

# Modify the process_natural_language_query function to include optimization
def process_natural_language_query(nl_query):
    sql_query = natural_language_to_sql(nl_query)
    is_valid, validation_message = validate_sql_query(sql_query)
    
    if not is_valid:
        return {
            "natural_language_query": nl_query,
            "sql_query": sql_query,
            "error": validation_message
        }
    
    optimized_query = optimize_sql_query(sql_query)
    results = execute_sql_query(optimized_query)
    
    return {
        "natural_language_query": nl_query,
        "original_sql_query": sql_query,
        "optimized_sql_query": optimized_query,
        "results": results
    }

# Test the optimization process
nl_query = "Show me all information about books published after 2000"
response = process_natural_language_query(nl_query)
print(f"Natural Language Query: {response['natural_language_query']}")
print(f"Original SQL: {response['original_sql_query']}")
print(f"Optimized SQL: {response['optimized_sql_query']}")
print("Results:")
for result in response['results']:
    print(result)
```

Slide 11: Implementing Query Optimization

Query optimization is crucial for enhancing the performance of LLM-generated SQL queries. We'll implement a basic optimization process that focuses on improving SELECT queries and adding appropriate indexes.

```python
import sqlparse

def optimize_sql_query(sql_query):
    parsed = sqlparse.parse(sql_query)[0]
    
    if parsed.get_type() == 'SELECT':
        # Optimize SELECT * queries
        if '*' in sql_query:
            table_name = extract_table_name(sql_query)
            columns = get_table_columns(table_name)
            sql_query = sql_query.replace('*', ', '.join(columns))
        
        # Add index hint for frequently used columns
        if 'WHERE' in sql_query:
            where_clause = sql_query.split('WHERE')[1]
            frequent_columns = ['year', 'author']
            for col in frequent_columns:
                if col in where_clause:
                    sql_query = f"SELECT /*+ INDEX(books idx_{col}) */ {sql_query.split('SELECT ')[1]}"
                    break
    
    return sql_query

def extract_table_name(sql_query):
    from_clause = sql_query.split('FROM')[1]
    return from_clause.split()[0]

def get_table_columns(table_name):
    # Implement this function to return column names for the given table
    # You can use SQLite's PRAGMA table_info or a similar method
    pass

# Test the optimization function
original_query = "SELECT * FROM books WHERE year > 2000"
optimized_query = optimize_sql_query(original_query)
print(f"Original query: {original_query}")
print(f"Optimized query: {optimized_query}")
```

Slide 12: Handling Multiple Database Dialects

LLM-based text-to-SQL systems often need to support multiple database dialects. Let's implement a function that can generate SQL for different database systems based on user input.

```python
def generate_dialect_specific_sql(nl_query, dialect):
    prompt = f"""
    Convert the following natural language query to {dialect} SQL:
    "{nl_query}"
    
    Ensure the generated SQL follows {dialect} syntax and conventions.
    """
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        temperature=0.3
    )
    
    return response.choices[0].text.strip()

# Test the dialect-specific SQL generation
nl_query = "Find all books published in the last 5 years"
dialects = ["SQLite", "MySQL", "PostgreSQL"]

for dialect in dialects:
    sql_query = generate_dialect_specific_sql(nl_query, dialect)
    print(f"{dialect} SQL: {sql_query}")
```

Slide 13: Implementing Error Handling and Logging

Robust error handling and logging are essential for maintaining and debugging LLM-based text-to-SQL systems. Let's implement a simple error handling and logging mechanism.

```python
import logging

logging.basicConfig(filename='text_to_sql.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def safe_execute_query(sql_query):
    try:
        results = execute_sql_query(sql_query)
        logging.info(f"Query executed successfully: {sql_query}")
        return results
    except sqlite3.Error as e:
        error_message = f"SQLite error occurred: {e}"
        logging.error(error_message)
        return {"error": error_message}
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        logging.error(error_message)
        return {"error": error_message}

# Modify the process_natural_language_query function to use safe_execute_query
def process_natural_language_query(nl_query):
    try:
        sql_query = natural_language_to_sql(nl_query)
        logging.info(f"Generated SQL query: {sql_query}")
        
        results = safe_execute_query(sql_query)
        
        return {
            "natural_language_query": nl_query,
            "sql_query": sql_query,
            "results": results
        }
    except Exception as e:
        error_message = f"Error processing query: {e}"
        logging.error(error_message)
        return {"error": error_message}

# Test error handling and logging
nl_query = "Show me all books published in the year 2100"
response = process_natural_language_query(nl_query)
print(response)
```

Slide 14: Real-Life Example: Library Management System

Let's apply our LLM-based text-to-SQL system to a library management scenario. We'll create a simple interface for librarians to query the book database using natural language.

```python
def library_management_system():
    print("Welcome to the Library Management System")
    print("You can ask questions about books in natural language.")
    print("Type 'exit' to quit.")
    
    while True:
        nl_query = input("\nEnter your query: ")
        
        if nl_query.lower() == 'exit':
            print("Thank you for using the Library Management System. Goodbye!")
            break
        
        response = process_natural_language_query(nl_query)
        
        if 'error' in response:
            print(f"Error: {response['error']}")
        else:
            print(f"\nSQL Query: {response['sql_query']}")
            print("\nResults:")
            for result in response['results']:
                print(result)

# Run the library management system
library_management_system()
```

Slide 15: Real-Life Example: Data Analysis Assistant

In this example, we'll create a data analysis assistant that helps users explore the book database using natural language queries and generates simple visualizations.

```python
import matplotlib.pyplot as plt

def generate_visualization(data, chart_type):
    if chart_type == 'bar':
        plt.figure(figsize=(10, 6))
        plt.bar(data.keys(), data.values())
        plt.title('Book Analysis')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('book_analysis.png')
        plt.close()
        return 'book_analysis.png'
    # Add more chart types as needed

def data_analysis_assistant():
    print("Welcome to the Data Analysis Assistant")
    print("You can ask questions about books and request visualizations.")
    print("Type 'exit' to quit.")
    
    while True:
        nl_query = input("\nEnter your query (include 'visualize' for charts): ")
        
        if nl_query.lower() == 'exit':
            print("Thank you for using the Data Analysis Assistant. Goodbye!")
            break
        
        response = process_natural_language_query(nl_query)
        
        if 'error' in response:
            print(f"Error: {response['error']}")
        else:
            print(f"\nSQL Query: {response['sql_query']}")
            print("\nResults:")
            for result in response['results']:
                print(result)
            
            if 'visualize' in nl_query.lower():
                # Example: Generate a bar chart of books per author
                data = {}
                for result in response['results']:
                    author = result.get('author', 'Unknown')
                    data[author] = data.get(author, 0) + 1
                
                chart_file = generate_visualization(data, 'bar')
                print(f"\nVisualization generated: {chart_file}")

# Run the data analysis assistant
data_analysis_assistant()
```

Slide 16: Additional Resources

For further exploration of LLM-based text-to-SQL integration, consider the following resources:

1. "Natural Language Interfaces to Databases" by Li Dong and Mirella Lapata (2018) ArXiv: [https://arxiv.org/abs/1804.00401](https://arxiv.org/abs/1804.00401)
2. "Towards Complex Text-to-SQL in Cross-Domain Database with Intermediate Representation" by Jiaqi Guo et al. (2019) ArXiv: [https://arxiv.org/abs/1905.08205](https://arxiv.org/abs/1905.08205)
3. "RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers" by Bailin Wang et al. (2020) ArXiv: [https://arxiv.org/abs/1911.04942](https://arxiv.org/abs/1911.04942)
4. "PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models" by Torsten Scholak et al. (2021) ArXiv: [https://arxiv.org/abs/2109.05093](https://arxiv.org/abs/2109.05093)

These papers provide in-depth discussions on advanced techniques and methodologies for improving text-to-SQL systems using machine learning and natural language processing.

