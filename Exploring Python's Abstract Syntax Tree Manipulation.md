## Exploring Python's Abstract Syntax Tree Manipulation
Slide 1: Introduction to Python AST Manipulation

Abstract Syntax Trees (ASTs) are tree-like representations of the structure of source code. Python provides powerful tools for working with ASTs, allowing developers to analyze, modify, and generate code programmatically. This slideshow will explore the fundamentals of AST manipulation in Python, providing practical examples and real-world applications.

```python
import ast

# Parse a simple Python expression into an AST
node = ast.parse("x + y")
print(ast.dump(node, indent=2))
```

Slide 2: Parsing Python Code into ASTs

Python's ast module provides the parse() function to convert source code into an AST. This is the first step in working with ASTs and allows us to examine the structure of our code programmatically.

```python
import ast

# Parse a more complex Python code snippet
code = """
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))
"""

tree = ast.parse(code)
print(ast.dump(tree, indent=2))
```

Slide 3: Traversing ASTs with NodeVisitor

The ast.NodeVisitor class allows us to walk through an AST and perform actions on specific node types. This is useful for analyzing code structure and gathering information about the code.

```python
import ast

class FunctionVisitor(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
        print(f"Found function: {node.name}")
        self.generic_visit(node)

tree = ast.parse(open("example.py").read())
FunctionVisitor().visit(tree)
```

Slide 4: Modifying ASTs with NodeTransformer

The ast.NodeTransformer class enables us to modify ASTs by replacing or altering nodes. This is powerful for code transformation tasks, such as optimizing or refactoring code automatically.

```python
import ast

class ConstantFolder(ast.NodeTransformer):
    def visit_BinOp(self, node):
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            if isinstance(node.op, ast.Add):
                return ast.Constant(node.left.value + node.right.value)
        return node

tree = ast.parse("2 + 3")
new_tree = ConstantFolder().visit(tree)
print(ast.unparse(new_tree))  # Output: 5
```

Slide 5: Generating Python Code from ASTs

After modifying an AST, we can convert it back into Python code using ast.unparse(). This allows us to programmatically generate or modify code and then execute it or write it to a file.

```python
import ast

# Create an AST for a simple function
func_ast = ast.FunctionDef(
    name='greet',
    args=ast.arguments(args=[ast.arg(arg='name')], posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
    body=[
        ast.Return(
            value=ast.Call(
                func=ast.Attribute(value=ast.Str(s='Hello, {}!'), attr='format'),
                args=[ast.Name(id='name', ctx=ast.Load())],
                keywords=[]
            )
        )
    ],
    decorator_list=[]
)

# Wrap the function in a module
module = ast.Module(body=[func_ast], type_ignores=[])

# Generate Python code from the AST
generated_code = ast.unparse(module)
print(generated_code)

# Execute the generated code
exec(generated_code)
print(greet("World"))  # Output: Hello, World!
```

Slide 6: AST-based Code Analysis: Finding Function Calls

ASTs can be used to analyze code structure and gather information. Here's an example of finding all function calls in a piece of code.

```python
import ast

class FunctionCallFinder(ast.NodeVisitor):
    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        self.generic_visit(node)

code = """
def example():
    print("Hello")
    math.sqrt(16)
    [1, 2, 3].append(4)

example()
"""

tree = ast.parse(code)
finder = FunctionCallFinder()
finder.visit(tree)
print("Function calls found:", finder.calls)
```

Slide 7: AST-based Code Transformation: Adding Logging

ASTs allow us to automatically modify code. Here's an example of adding logging statements to all function definitions.

```python
import ast

class LoggingTransformer(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        log_stmt = ast.Expr(
            ast.Call(
                func=ast.Name(id='print', ctx=ast.Load()),
                args=[ast.Str(s=f"Calling function: {node.name}")],
                keywords=[]
            )
        )
        node.body.insert(0, log_stmt)
        return node

code = """
def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b
"""

tree = ast.parse(code)
new_tree = LoggingTransformer().visit(tree)
print(ast.unparse(new_tree))
```

Slide 8: Real-life Example: Custom Decorator Implementation

ASTs can be used to implement custom decorators that modify function behavior. Here's an example of a decorator that measures execution time.

```python
import ast
import time

class TimingDecorator(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        # Create AST nodes for timing logic
        start_time = ast.Assign(
            targets=[ast.Name(id='start_time', ctx=ast.Store())],
            value=ast.Call(func=ast.Attribute(value=ast.Name(id='time', ctx=ast.Load()), attr='time'), args=[], keywords=[])
        )
        
        end_time = ast.Assign(
            targets=[ast.Name(id='end_time', ctx=ast.Store())],
            value=ast.Call(func=ast.Attribute(value=ast.Name(id='time', ctx=ast.Load()), attr='time'), args=[], keywords=[])
        )
        
        print_stmt = ast.Expr(
            ast.Call(
                func=ast.Name(id='print', ctx=ast.Load()),
                args=[
                    ast.BinOp(
                        left=ast.Str(s=f"Function '{node.name}' took "),
                        op=ast.Add(),
                        right=ast.BinOp(
                            left=ast.BinOp(
                                left=ast.Name(id='end_time', ctx=ast.Load()),
                                op=ast.Sub(),
                                right=ast.Name(id='start_time', ctx=ast.Load())
                            ),
                            op=ast.Mult(),
                            right=ast.Constant(value=1000)
                        )
                    ),
                    ast.Str(s=" ms")
                ],
                keywords=[]
            )
        )
        
        # Insert timing logic
        node.body.insert(0, start_time)
        node.body.append(end_time)
        node.body.append(print_stmt)
        
        return node

# Example usage
code = """
def slow_function():
    import time
    time.sleep(1)
    return "Done"

result = slow_function()
print(result)
"""

tree = ast.parse(code)
new_tree = TimingDecorator().visit(tree)
exec(ast.unparse(new_tree))
```

Slide 9: AST-based Code Generation: Creating a Simple ORM

ASTs can be used to generate code dynamically. Here's an example of a simple Object-Relational Mapping (ORM) system that generates Python classes from a schema definition.

```python
import ast

def generate_orm_class(class_name, fields):
    class_body = []
    
    # Generate __init__ method
    init_args = [ast.arg(arg='self')] + [ast.arg(arg=field) for field in fields]
    init_body = [
        ast.Assign(
            targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr=field, ctx=ast.Store())],
            value=ast.Name(id=field, ctx=ast.Load())
        )
        for field in fields
    ]
    init_method = ast.FunctionDef(
        name='__init__',
        args=ast.arguments(args=init_args, posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=init_body,
        decorator_list=[]
    )
    class_body.append(init_method)
    
    # Generate the class
    class_def = ast.ClassDef(
        name=class_name,
        bases=[],
        keywords=[],
        body=class_body,
        decorator_list=[]
    )
    
    # Wrap in a module
    module = ast.Module(body=[class_def], type_ignores=[])
    
    return ast.unparse(module)

# Example usage
schema = {
    "User": ["id", "name", "email"]
}

for class_name, fields in schema.items():
    orm_class = generate_orm_class(class_name, fields)
    print(orm_class)
    exec(orm_class)

# Create an instance
user = User(1, "Alice", "alice@example.com")
print(f"User: {user.name}, Email: {user.email}")
```

Slide 10: AST-based Code Analysis: Cyclomatic Complexity

ASTs can be used to analyze code complexity. Here's an example that calculates the cyclomatic complexity of a function.

```python
import ast

class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.complexity = 1

    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.complexity += 1
        self.generic_visit(node)

def calculate_complexity(code):
    tree = ast.parse(code)
    visitor = ComplexityVisitor()
    visitor.visit(tree)
    return visitor.complexity

# Example usage
code = """
def complex_function(x, y):
    if x > 0:
        if y > 0:
            return x + y
        else:
            return x - y
    else:
        for i in range(y):
            try:
                x /= i
            except ZeroDivisionError:
                continue
    return x
"""

complexity = calculate_complexity(code)
print(f"Cyclomatic Complexity: {complexity}")
```

Slide 11: AST-based Code Transformation: Constant Folding

ASTs can be used to optimize code by performing constant folding, which evaluates constant expressions at compile-time.

```python
import ast
import operator

class ConstantFolder(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            op = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow
            }.get(type(node.op))
            if op:
                try:
                    result = op(node.left.value, node.right.value)
                    return ast.Constant(value=result)
                except:
                    pass
        return node

code = """
def calculate():
    return 2 + 3 * 4 - 1
"""

tree = ast.parse(code)
folder = ConstantFolder()
optimized_tree = folder.visit(tree)
print("Original:")
print(ast.unparse(tree))
print("\nOptimized:")
print(ast.unparse(optimized_tree))
```

Slide 12: Real-life Example: Custom Static Type Checker

ASTs can be used to implement custom static type checking. Here's a simple example that checks function argument types based on type hints.

```python
import ast
import typing

class TypeChecker(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
        if not node.returns:
            return
        
        for arg, annotation in zip(node.args.args, node.args.annotations):
            if not isinstance(annotation, ast.Name):
                continue
            
            expected_type = getattr(typing, annotation.id, None)
            if expected_type is None:
                continue
            
            print(f"Checking argument '{arg.arg}' of function '{node.name}'")
            print(f"Expected type: {expected_type}")

code = """
def greet(name: str, age: int) -> str:
    return f"Hello, {name}! You are {age} years old."

def calculate(x: float, y: float) -> float:
    return x + y
"""

tree = ast.parse(code)
TypeChecker().visit(tree)
```

Slide 13: AST-based Code Generation: Creating a Simple DSL

ASTs can be used to create domain-specific languages (DSLs). Here's an example of a simple DSL for creating HTML elements.

```python
import ast

class HTMLBuilder(ast.NodeTransformer):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in ['div', 'span', 'p']:
            tag = node.func.id
            attrs = []
            content = []
            
            for kw in node.keywords:
                if kw.arg == 'content':
                    content.append(kw.value)
                else:
                    attrs.append(f'{kw.arg}="{ast.literal_eval(kw.value)}"')
            
            attrs_str = " ".join(attrs)
            content_str = "".join([ast.unparse(c) for c in content])
            
            return ast.Str(s=f'<{tag} {attrs_str}>{content_str}</{tag}>')
        
        return node

# Example DSL code
dsl_code = """
def create_html():
    return div(
        content=span(content="Hello", class_="greeting") + p(content="World", id="message")
    )
"""

tree = ast.parse(dsl_code)
transformed = HTMLBuilder().visit(tree)
exec(ast.unparse(transformed))

result = create_html()
print(result)
```

Slide 14: Conclusion and Best Practices

Python's AST manipulation capabilities offer powerful tools for code analysis, transformation, and generation. When working with ASTs:

1. Always validate and sanitize input code to prevent security vulnerabilities.
2. Use ast.parse() with a specific mode (e.g., 'exec', 'eval', 'single') when appropriate.
3. Be cautious when executing generated code, especially from untrusted sources.
4. Consider using the astor library for more advanced AST operations and code generation.
5. Test your AST transformations thoroughly, as small changes can have significant impacts.
6. Keep your AST manipulations modular and composable for better maintainability.
7. Use type annotations and static type checkers to catch errors early in your AST manipulation code.

AST manipulation is a powerful technique that opens up numerous possibilities for metaprogramming, code analysis, and domain-specific language development in Python.

Slide 15: Additional Resources

For those interested in diving deeper into Python AST manipulation, here are some valuable resources:

1. Python AST module documentation: [https://docs.python.org/3/library/ast.html](https://docs.python.org/3/library/ast.html)
2. "Green Tree Snakes - the missing Python AST docs

