## Manipulating Bits with Python's Bitwise Operators
Slide 1: Understanding Binary and Bitwise Operations in Python

Binary numbers form the foundation of all computer operations, representing data as sequences of 0s and 1s. In Python, integers are stored internally as binary, and bitwise operators allow direct manipulation of these individual bits for low-level operations.

```python
# Converting between decimal and binary representations
decimal_num = 42
binary_str = bin(decimal_num)  # Convert to binary string
binary_num = int('101010', 2)  # Convert from binary string to decimal

print(f"Decimal: {decimal_num}")  # Output: Decimal: 42
print(f"Binary: {binary_str}")    # Output: Binary: 0b101010
print(f"Back to decimal: {binary_num}")  # Output: Back to decimal: 42
```

Slide 2: Basic Bitwise Operators

Python provides six fundamental bitwise operators that perform operations at the bit level. These operators work by comparing corresponding bits in two numbers, producing a result based on specific rules for each operation.

```python
a = 60  # 0011 1100 in binary
b = 13  # 0000 1101 in binary

# Bitwise AND (&)
print(f"a & b = {a & b}")   # Output: 12 (0000 1100)

# Bitwise OR (|)
print(f"a | b = {a | b}")   # Output: 61 (0011 1101)

# Bitwise XOR (^)
print(f"a ^ b = {a ^ b}")   # Output: 49 (0011 0001)

# Bitwise NOT (~)
print(f"~a = {~a}")         # Output: -61

# Left shift (<<)
print(f"a << 2 = {a << 2}") # Output: 240 (1111 0000)

# Right shift (>>)
print(f"a >> 2 = {a >> 2}") # Output: 15 (0000 1111)
```

Slide 3: Bit Masking Techniques

Bit masking is a powerful technique for manipulating specific bits while leaving others unchanged. It's commonly used in embedded systems programming, flag management, and optimizing memory usage in applications.

```python
# Example of bit masking operations
def set_bit(num, position):
    """Set the bit at given position to 1."""
    mask = 1 << position
    return num | mask

def clear_bit(num, position):
    """Clear the bit at given position to 0."""
    mask = ~(1 << position)
    return num & mask

def toggle_bit(num, position):
    """Toggle the bit at given position."""
    mask = 1 << position
    return num ^ mask

# Example usage
number = 42  # 101010 in binary
print(f"Original number: {bin(number)}")
print(f"Set bit 2: {bin(set_bit(number, 2))}")
print(f"Clear bit 3: {bin(clear_bit(number, 3))}")
print(f"Toggle bit 1: {bin(toggle_bit(number, 1))}")
```

Slide 4: Bit Flags and State Management

Using bit flags is an efficient way to manage multiple boolean states in a single integer. This technique is commonly used in systems programming, game development, and permission systems.

```python
# Using bit flags for state management
class Permissions:
    NONE  = 0b00000000  # 0
    READ  = 0b00000001  # 1
    WRITE = 0b00000010  # 2
    EXEC  = 0b00000100  # 4
    ALL   = 0b00000111  # 7

def check_permission(user_perms, required_perm):
    return user_perms & required_perm == required_perm

# Example usage
user_permissions = Permissions.READ | Permissions.WRITE
print(f"Has read permission: {check_permission(user_permissions, Permissions.READ)}")
print(f"Has exec permission: {check_permission(user_permissions, Permissions.EXEC)}")
```

Slide 5: Efficient Data Packing

Bit manipulation enables efficient storage of multiple values in a single integer. This technique is particularly useful when working with limited memory or when optimizing data structures.

```python
class ColorPacker:
    @staticmethod
    def pack_rgb(r, g, b):
        """Pack RGB values (0-255) into a single integer."""
        return (r << 16) | (g << 8) | b
    
    @staticmethod
    def unpack_rgb(packed):
        """Extract RGB values from packed integer."""
        r = (packed >> 16) & 0xFF
        g = (packed >> 8) & 0xFF
        b = packed & 0xFF
        return r, g, b

# Example usage
color = ColorPacker.pack_rgb(255, 128, 0)  # Orange
print(f"Packed color: {hex(color)}")
r, g, b = ColorPacker.unpack_rgb(color)
print(f"Unpacked RGB: ({r}, {g}, {b})")
```

Slide 6: Binary Data Compression

Bitwise operations enable efficient data compression by manipulating individual bits. This implementation demonstrates a simple run-length encoding compression algorithm using bit manipulation for storing counts and values.

```python
def compress_binary(data):
    """
    Compress binary data using run-length encoding with bit manipulation.
    Each compressed value uses 6 bits for count (max 63) and 2 bits for value.
    """
    result = []
    count = 1
    current = data[0]
    
    for bit in data[1:]:
        if bit == current and count < 63:  # 6 bits max for count
            count += 1
        else:
            # Pack count and value into single byte
            packed = (count << 2) | current
            result.append(packed)
            current = bit
            count = 1
    
    # Handle last sequence
    packed = (count << 2) | current
    result.append(packed)
    return result

def decompress_binary(compressed):
    """Decompress data compressed with compress_binary."""
    result = []
    for packed in compressed:
        count = packed >> 2  # Extract count from upper 6 bits
        value = packed & 0b11  # Extract value from lower 2 bits
        result.extend([value] * count)
    return result

# Example usage
data = [1,1,1,1,0,0,0,1,1,1,0,0]
compressed = compress_binary(data)
decompressed = decompress_binary(compressed)
print(f"Original: {data}")
print(f"Compressed: {compressed}")
print(f"Decompressed: {decompressed}")
```

Slide 7: Error Detection with Parity Bits

Parity bits are commonly used for error detection in data transmission. This implementation shows how to calculate and verify parity using bitwise operations.

```python
def calculate_parity(data, width=8):
    """Calculate parity bit for data."""
    x = data
    while width > 1:
        width = width // 2
        x = x ^ (x >> width)
    return x & 1

def add_parity_bit(data):
    """Add even parity bit to 7-bit data."""
    parity = calculate_parity(data, 7)
    return (data << 1) | parity

def check_parity(data):
    """Check if 8-bit data has correct even parity."""
    return calculate_parity(data, 8) == 0

# Example usage
original_data = 0b1010111
with_parity = add_parity_bit(original_data)
print(f"Original: {bin(original_data)}")
print(f"With parity: {bin(with_parity)}")
print(f"Parity check: {check_parity(with_parity)}")

# Simulate error
corrupted = with_parity ^ (1 << 2)  # Flip one bit
print(f"Corrupted: {bin(corrupted)}")
print(f"Parity check on corrupted: {check_parity(corrupted)}")
```

Slide 8: Bit Manipulation for Image Processing

Bitwise operations can be used for efficient image processing operations. This example demonstrates basic image masking and pixel manipulation techniques.

```python
import numpy as np

def create_bit_mask(width, height, bit_position):
    """Create a bit mask for specific bit plane."""
    return np.full((height, width), 1 << bit_position, dtype=np.uint8)

def extract_bit_plane(image, bit_position):
    """Extract a specific bit plane from an image."""
    mask = create_bit_mask(image.shape[1], image.shape[0], bit_position)
    return (image & mask) >> bit_position

def modify_bit_plane(image, bit_position, new_plane):
    """Modify a specific bit plane in an image."""
    mask = create_bit_mask(image.shape[1], image.shape[0], bit_position)
    cleared_image = image & ~mask
    return cleared_image | ((new_plane << bit_position) & mask)

# Example usage (assuming you have a grayscale image as numpy array)
sample_image = np.random.randint(0, 256, (4, 4), dtype=np.uint8)
print("Original image:")
print(sample_image)

# Extract and modify bit plane 7 (MSB)
bit_plane = extract_bit_plane(sample_image, 7)
modified_image = modify_bit_plane(sample_image, 7, ~bit_plane)
print("\nModified image:")
print(modified_image)
```

Slide 9: Bitwise Operations for Cryptography

Bitwise operations are fundamental in cryptography for implementing encryption algorithms. This example demonstrates a simple XOR cipher implementation using bit manipulation techniques.

```python
def xor_cipher(data: bytes, key: bytes) -> bytes:
    """
    Implement simple XOR encryption/decryption.
    The same function works for both encryption and decryption.
    """
    if not data or not key:
        return b''
    
    # Create a key of the same length as data by repeating the key
    extended_key = key * (len(data) // len(key) + 1)
    key_bytes = extended_key[:len(data)]
    
    # Perform XOR operation on each byte
    return bytes(a ^ b for a, b in zip(data, key_bytes))

# Example usage
message = "Hello, World!".encode('utf-8')
key = b'SECRET'

# Encrypt
encrypted = xor_cipher(message, key)
print(f"Original: {message}")
print(f"Encrypted (hex): {encrypted.hex()}")

# Decrypt (using the same function)
decrypted = xor_cipher(encrypted, key)
print(f"Decrypted: {decrypted.decode('utf-8')}")
```

Slide 10: Custom Binary Protocol Implementation

Implementing binary protocols requires precise bit manipulation. This example shows how to create a custom binary protocol for efficient data transmission.

```python
class BinaryProtocol:
    @staticmethod
    def pack_message(msg_type: int, payload: bytes, priority: int) -> bytes:
        """
        Pack message into binary format:
        - Header (1 byte): 3 bits msg_type, 2 bits priority, 3 bits reserved
        - Length (2 bytes): payload length
        - Payload (variable)
        """
        if msg_type > 7 or priority > 3:  # 3 bits max for type, 2 for priority
            raise ValueError("Invalid message type or priority")
        
        header = (msg_type << 5) | (priority << 3)
        length = len(payload).to_bytes(2, 'big')
        
        return bytes([header]) + length + payload
    
    @staticmethod
    def unpack_message(data: bytes) -> tuple:
        """Unpack binary message into components."""
        if len(data) < 3:
            raise ValueError("Invalid message format")
        
        header = data[0]
        msg_type = header >> 5
        priority = (header >> 3) & 0b11
        
        length = int.from_bytes(data[1:3], 'big')
        payload = data[3:3+length]
        
        return msg_type, priority, payload

# Example usage
protocol = BinaryProtocol()
message = b"Important data"
packed = protocol.pack_message(msg_type=2, payload=message, priority=1)
print(f"Packed message (hex): {packed.hex()}")

msg_type, priority, payload = protocol.unpack_message(packed)
print(f"Unpacked - Type: {msg_type}, Priority: {priority}")
print(f"Payload: {payload.decode()}")
```

Slide 11: Bit Field Data Structure

A bit field allows efficient storage of multiple small-value integers or boolean flags in a single integer. This implementation shows how to create and manipulate bit fields.

```python
class BitField:
    def __init__(self, num_fields: int, bits_per_field: int):
        if num_fields * bits_per_field > 64:
            raise ValueError("Total bits exceeds 64-bit limit")
        
        self.num_fields = num_fields
        self.bits_per_field = bits_per_field
        self.mask = (1 << bits_per_field) - 1
        self.value = 0
    
    def set_field(self, field_index: int, value: int) -> None:
        """Set value for specific field."""
        if field_index >= self.num_fields:
            raise IndexError("Field index out of range")
        if value > self.mask:
            raise ValueError("Value too large for field")
        
        position = field_index * self.bits_per_field
        self.value &= ~(self.mask << position)  # Clear field
        self.value |= (value << position)       # Set new value
    
    def get_field(self, field_index: int) -> int:
        """Get value from specific field."""
        if field_index >= self.num_fields:
            raise IndexError("Field index out of range")
        
        position = field_index * self.bits_per_field
        return (self.value >> position) & self.mask

# Example usage
bit_field = BitField(num_fields=4, bits_per_field=4)  # 16 bits total
bit_field.set_field(0, 12)  # Set first field to 12
bit_field.set_field(1, 7)   # Set second field to 7
bit_field.set_field(2, 15)  # Set third field to 15

print(f"Field 0: {bit_field.get_field(0)}")  # Output: 12
print(f"Field 1: {bit_field.get_field(1)}")  # Output: 7
print(f"Field 2: {bit_field.get_field(2)}")  # Output: 15
print(f"Complete value: {bin(bit_field.value)}")
```

Slide 12: Real-time Bit Manipulation for Hardware Control

Bit manipulation is crucial for controlling hardware devices through registers. This example demonstrates how to control GPIO pins on a microcontroller using bitwise operations.

```python
class GPIOController:
    def __init__(self):
        # Simulated GPIO registers
        self.data_register = 0x00        # Pin values
        self.direction_register = 0x00    # Pin directions (0=input, 1=output)
        self.pull_up_register = 0x00     # Pull-up resistors
    
    def set_pin_mode(self, pin: int, mode: str) -> None:
        """Set pin mode (INPUT, OUTPUT, INPUT_PULLUP)."""
        if pin > 7:
            raise ValueError("Invalid pin number")
            
        mask = 1 << pin
        if mode == "INPUT":
            self.direction_register &= ~mask  # Clear bit
            self.pull_up_register &= ~mask
        elif mode == "OUTPUT":
            self.direction_register |= mask   # Set bit
        elif mode == "INPUT_PULLUP":
            self.direction_register &= ~mask
            self.pull_up_register |= mask
    
    def digital_write(self, pin: int, value: bool) -> None:
        """Set digital output value."""
        if pin > 7:
            raise ValueError("Invalid pin number")
            
        if value:
            self.data_register |= (1 << pin)
        else:
            self.data_register &= ~(1 << pin)
    
    def digital_read(self, pin: int) -> bool:
        """Read digital input value."""
        return bool(self.data_register & (1 << pin))

# Example usage
gpio = GPIOController()
gpio.set_pin_mode(0, "OUTPUT")
gpio.set_pin_mode(1, "INPUT_PULLUP")

gpio.digital_write(0, True)   # Set pin 0 HIGH
print(f"Pin 0 state: {gpio.digital_read(0)}")
print(f"Direction register: {bin(gpio.direction_register)}")
print(f"Pull-up register: {bin(gpio.pull_up_register)}")
```

Slide 13: Performance Analysis of Bitwise Operations

This slide demonstrates performance comparisons between traditional arithmetic operations and their bitwise equivalents, showing why bit manipulation can be more efficient in certain scenarios.

```python
import timeit
import random

def benchmark_operations(size: int = 1000000):
    """Compare performance of arithmetic vs bitwise operations."""
    numbers = [random.randint(0, 1000) for _ in range(size)]
    
    def test_multiply_by_2():
        return [n * 2 for n in numbers]
    
    def test_left_shift_1():
        return [n << 1 for n in numbers]
    
    def test_divide_by_2():
        return [n // 2 for n in numbers]
    
    def test_right_shift_1():
        return [n >> 1 for n in numbers]
    
    def test_modulo_2():
        return [n % 2 for n in numbers]
    
    def test_bitwise_and_1():
        return [n & 1 for n in numbers]
    
    results = {}
    operations = [
        ("Multiply by 2", test_multiply_by_2),
        ("Left shift 1", test_left_shift_1),
        ("Divide by 2", test_divide_by_2),
        ("Right shift 1", test_right_shift_1),
        ("Modulo 2", test_modulo_2),
        ("Bitwise AND 1", test_bitwise_and_1)
    ]
    
    for name, func in operations:
        time = timeit.timeit(func, number=100)
        results[name] = time
    
    return results

# Run benchmark and display results
results = benchmark_operations()
for operation, time in results.items():
    print(f"{operation}: {time:.6f} seconds")
```

Slide 14: Additional Resources

*   "Efficient Bit Manipulation Techniques for Modern Processors" - [https://arxiv.org/abs/1611.07612](https://arxiv.org/abs/1611.07612)
*   "A Survey of Binary Code Analysis and Transformation Techniques" - [https://arxiv.org/abs/2005.07258](https://arxiv.org/abs/2005.07258)
*   "Optimization Techniques Using Bit-Level Operations" - [https://arxiv.org/abs/1908.11459](https://arxiv.org/abs/1908.11459)
*   General resources:
    *   [https://wiki.python.org/moin/BitManipulation](https://wiki.python.org/moin/BitManipulation)
    *   [https://docs.python.org/3/reference/expressions.html#binary-bitwise-operations](https://docs.python.org/3/reference/expressions.html#binary-bitwise-operations)
    *   [https://realpython.com/python-bitwise-operators/](https://realpython.com/python-bitwise-operators/)

