## Implementing Binary Tree Traversal Techniques
Slide 1: Binary Tree Implementation Fundamentals

A binary tree is a hierarchical data structure composed of nodes, where each node contains a value and has up to two children nodes - left and right. This implementation creates the foundation for more complex tree operations and traversals.

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
```

Slide 2: In-Order Traversal Implementation

In-order traversal visits nodes in the sequence: left subtree, root, right subtree. This approach is fundamental for binary search trees as it prints values in ascending order when the tree follows BST properties.

```python
def inorder_traversal(self, node, result=None):
    if result is None:
        result = []
    
    if node:
        # Traverse left subtree
        self.inorder_traversal(node.left, result)
        # Visit root
        result.append(node.value)
        # Traverse right subtree
        self.inorder_traversal(node.right, result)
    
    return result

# Example usage
tree = BinaryTree()
for value in [5, 3, 7, 1, 4, 6, 8]:
    tree.insert(value)
print("In-order traversal:", tree.inorder_traversal(tree.root))
# Output: [1, 3, 4, 5, 6, 7, 8]
```

Slide 3: Pre-Order Traversal Implementation

Pre-order traversal follows the pattern of visiting the root first, then traversing the left subtree, and finally the right subtree. This traversal is particularly useful for creating a copy of the tree or generating a prefix expression.

```python
def preorder_traversal(self, node, result=None):
    if result is None:
        result = []
    
    if node:
        # Visit root
        result.append(node.value)
        # Traverse left subtree
        self.preorder_traversal(node.left, result)
        # Traverse right subtree
        self.preorder_traversal(node.right, result)
    
    return result

# Example usage
tree = BinaryTree()
for value in [5, 3, 7, 1, 4, 6, 8]:
    tree.insert(value)
print("Pre-order traversal:", tree.preorder_traversal(tree.root))
# Output: [5, 3, 1, 4, 7, 6, 8]
```

Slide 4: Post-Order Traversal Implementation

Post-order traversal visits the left subtree, right subtree, and then the root. This pattern is essential for operations that require processing child nodes before their parents, such as deletion or calculating directory sizes.

```python
def postorder_traversal(self, node, result=None):
    if result is None:
        result = []
    
    if node:
        # Traverse left subtree
        self.postorder_traversal(node.left, result)
        # Traverse right subtree
        self.postorder_traversal(node.right, result)
        # Visit root
        result.append(node.value)
    
    return result

# Example usage
tree = BinaryTree()
for value in [5, 3, 7, 1, 4, 6, 8]:
    tree.insert(value)
print("Post-order traversal:", tree.postorder_traversal(tree.root))
# Output: [1, 4, 3, 6, 8, 7, 5]
```

Slide 5: Binary Tree Height and Depth Calculation

The height of a binary tree represents the longest path from root to leaf, while depth represents the distance from a node to the root. These metrics are crucial for analyzing tree balance and performance characteristics.

```python
def height(self, node):
    if not node:
        return 0
    
    # Calculate height of left and right subtrees
    left_height = self.height(node.left)
    right_height = self.height(node.right)
    
    # Return maximum height plus 1 for current node
    return max(left_height, right_height) + 1

def depth(self, node, target_value, current_depth=0):
    if not node:
        return -1
    
    if node.value == target_value:
        return current_depth
    
    left_depth = self.depth(node.left, target_value, current_depth + 1)
    if left_depth != -1:
        return left_depth
    
    return self.depth(node.right, target_value, current_depth + 1)
```

Slide 6: Level Order Traversal with Queue

Level order traversal processes nodes level by level from left to right, using a queue data structure to maintain the order of visitation. This traversal is crucial for applications requiring breadth-first exploration of tree structures.

```python
from collections import deque

def level_order_traversal(self, root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.value)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result

# Example usage
tree = BinaryTree()
for value in [5, 3, 7, 1, 4, 6, 8]:
    tree.insert(value)
print("Level order traversal:", tree.level_order_traversal(tree.root))
# Output: [[5], [3, 7], [1, 4, 6, 8]]
```

Slide 7: Binary Search Tree Validation

A binary search tree must maintain specific properties where all left subtree values are less than the node's value, and all right subtree values are greater. This implementation validates these properties recursively.

```python
def is_bst(self, node, min_value=float('-inf'), max_value=float('inf')):
    if not node:
        return True
    
    if not (min_value < node.value < max_value):
        return False
    
    return (self.is_bst(node.left, min_value, node.value) and 
            self.is_bst(node.right, node.value, max_value))

def is_balanced(self, node):
    if not node:
        return True, 0
    
    left_balanced, left_height = self.is_balanced(node.left)
    right_balanced, right_height = self.is_balanced(node.right)
    
    balanced = (left_balanced and right_balanced and 
               abs(left_height - right_height) <= 1)
    
    return balanced, max(left_height, right_height) + 1

# Example usage
tree = BinaryTree()
values = [5, 3, 7, 1, 4, 6, 8]
for value in values:
    tree.insert(value)
print("Is BST:", tree.is_bst(tree.root))
print("Is Balanced:", tree.is_balanced(tree.root)[0])
```

Slide 8: Binary Tree Serialization and Deserialization

Converting a binary tree to a string representation and back is crucial for storage and transmission. This implementation uses a level-order approach with special markers for null nodes.

```python
def serialize(self, root):
    if not root:
        return "[]"
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        if node:
            result.append(str(node.value))
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append("null")
            
    # Remove trailing nulls
    while result[-1] == "null":
        result.pop()
    
    return "[" + ",".join(result) + "]"

def deserialize(self, data):
    if data == "[]":
        return None
    
    values = data[1:-1].split(",")
    root = Node(int(values[0]))
    queue = deque([root])
    i = 1
    
    while queue and i < len(values):
        node = queue.popleft()
        
        if i < len(values) and values[i] != "null":
            node.left = Node(int(values[i]))
            queue.append(node.left)
        i += 1
        
        if i < len(values) and values[i] != "null":
            node.right = Node(int(values[i]))
            queue.append(node.right)
        i += 1
    
    return root
```

Slide 9: Lowest Common Ancestor Implementation

Finding the lowest common ancestor (LCA) of two nodes is a fundamental operation in binary trees, useful in various applications including network routing and phylogenetic trees.

```python
def lowest_common_ancestor(self, root, p_val, q_val):
    if not root:
        return None
    
    # If either p or q matches the root, we've found an ancestor
    if root.value == p_val or root.value == q_val:
        return root
    
    # Look for p and q in left and right subtrees
    left = self.lowest_common_ancestor(root.left, p_val, q_val)
    right = self.lowest_common_ancestor(root.right, p_val, q_val)
    
    # If p and q are found in different subtrees, current node is LCA
    if left and right:
        return root
    
    # Return the non-null node
    return left if left else right

# Example usage
tree = BinaryTree()
nodes = [5, 3, 7, 1, 4, 6, 8]
for node in nodes:
    tree.insert(node)
lca = tree.lowest_common_ancestor(tree.root, 1, 4)
print(f"LCA of 1 and 4: {lca.value}")  # Output: 3
```

Slide 10: Binary Tree Path Finding

This implementation explores all possible root-to-leaf paths in a binary tree, essential for analyzing tree structure and finding specific routes through the tree hierarchy.

```python
def find_all_paths(self, root):
    def dfs(node, current_path, all_paths):
        if not node:
            return
        
        current_path.append(node.value)
        
        # If leaf node, add path to results
        if not node.left and not node.right:
            all_paths.append(current_path.copy())
        else:
            dfs(node.left, current_path, all_paths)
            dfs(node.right, current_path, all_paths)
        
        current_path.pop()
    
    all_paths = []
    dfs(root, [], all_paths)
    return all_paths

# Example usage
tree = BinaryTree()
for value in [5, 3, 7, 1, 4, 6, 8]:
    tree.insert(value)
print("All root-to-leaf paths:", tree.find_all_paths(tree.root))
# Output: [[5, 3, 1], [5, 3, 4], [5, 7, 6], [5, 7, 8]]
```

Slide 11: Real-World Application - File System Directory Structure

Binary trees can model file system hierarchies, where each node represents a directory or file. This implementation includes size calculation and path resolution functionality.

```python
class FileSystemNode:
    def __init__(self, name, is_directory=False, size=0):
        self.name = name
        self.is_directory = is_directory
        self.size = size
        self.left = None  # Previous sibling
        self.right = None  # Next sibling
        self.child = None  # First child

class FileSystem:
    def __init__(self):
        self.root = FileSystemNode("/", True)
    
    def calculate_directory_size(self, node):
        if not node:
            return 0
        
        total_size = node.size
        if node.is_directory:
            # Add sizes of all children
            child = node.child
            while child:
                total_size += self.calculate_directory_size(child)
                child = child.right
                
        return total_size
    
    def find_path(self, path):
        components = path.split("/")[1:]  # Skip empty first component
        current = self.root
        
        for component in components:
            if not current or not current.is_directory:
                return None
            
            # Search children for component
            child = current.child
            found = False
            while child:
                if child.name == component:
                    current = child
                    found = True
                    break
                child = child.right
            
            if not found:
                return None
        
        return current

# Example usage
fs = FileSystem()
# Add some files and directories
root = fs.root
root.child = FileSystemNode("home", True)
root.child.child = FileSystemNode("user1", True)
root.child.child.child = FileSystemNode("document.txt", False, 1024)

print("Size of /home:", fs.calculate_directory_size(fs.find_path("/home")))
# Output: 1024
```

Slide 12: Real-World Application - Expression Tree Evaluator

Expression trees represent mathematical expressions where operators are internal nodes and operands are leaf nodes. This implementation includes evaluation and expression printing capabilities.

```python
class ExpressionNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class ExpressionTree:
    def __init__(self):
        self.root = None
    
    def evaluate(self, node):
        if not node:
            return 0
        
        # Leaf node (operand)
        if not node.left and not node.right:
            return float(node.value)
        
        # Evaluate left and right subtrees
        left_val = self.evaluate(node.left)
        right_val = self.evaluate(node.right)
        
        # Apply operator
        if node.value == '+':
            return left_val + right_val
        elif node.value == '-':
            return left_val - right_val
        elif node.value == '*':
            return left_val * right_val
        elif node.value == '/':
            return left_val / right_val
        
        return 0

# Example usage
expr_tree = ExpressionTree()
# Create expression tree for: (3 + 4) * 2
expr_tree.root = ExpressionNode('*')
expr_tree.root.left = ExpressionNode('+')
expr_tree.root.right = ExpressionNode('2')
expr_tree.root.left.left = ExpressionNode('3')
expr_tree.root.left.right = ExpressionNode('4')

result = expr_tree.evaluate(expr_tree.root)
print("Expression result:", result)
# Output: 14.0
```

Slide 13: Additional Resources

*   "The Art of Tree Traversal with Space-Optimal Analysis"
    *   [https://arxiv.org/abs/2102.05475](https://arxiv.org/abs/2102.05475)
*   "Optimal Binary Search Trees with Parallel Construction"
    *   [https://arxiv.org/abs/1904.06743](https://arxiv.org/abs/1904.06743)
*   "Self-Adjusting Binary Search Trees: Improvements and Applications"
    *   [https://arxiv.org/abs/2108.07054](https://arxiv.org/abs/2108.07054)
*   "Dynamic Optimality in Binary Search Trees - A Survey"
    *   [https://arxiv.org/abs/2001.00699](https://arxiv.org/abs/2001.00699)
*   "On the Height of Binary Search Trees"
    *   [https://arxiv.org/abs/1901.01961](https://arxiv.org/abs/1901.01961)

