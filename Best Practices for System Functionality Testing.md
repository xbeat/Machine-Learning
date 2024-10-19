## Best Practices for System Functionality Testing

Slide 1: Unit Testing

Unit testing is a fundamental practice in software development that focuses on testing individual components or functions of a system in isolation. It helps ensure that each unit of code performs as expected before integration with other parts of the system.

```python

def add_numbers(a, b):
    return a + b

class TestAddNumbers(unittest.TestCase):
    def test_add_positive_numbers(self):
        self.assertEqual(add_numbers(2, 3), 5)
    
    def test_add_negative_numbers(self):
        self.assertEqual(add_numbers(-1, -4), -5)

if __name__ == '__main__':
    unittest.main()
```

Slide 2: Integration Testing

Integration testing verifies that different components of a system work together correctly. It helps identify issues that may arise when individual units are combined and interact with each other.

```python

def get_user_data(user_id):
    response = requests.get(f"https://api.example.com/users/{user_id}")
    return response.json()

def process_user_data(user_data):
    return f"User: {user_data['name']}, Age: {user_data['age']}"

def integration_test():
    user_id = 123
    user_data = get_user_data(user_id)
    result = process_user_data(user_data)
    assert "User:" in result and "Age:" in result, "Integration test failed"

integration_test()
print("Integration test passed")
```

Slide 3: System Testing

System testing evaluates the entire system's functionality, performance, and compliance with specified requirements. It ensures that the system as a whole meets user expectations and operates reliably.

```python
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def system_test_login():
    driver = webdriver.Chrome()
    driver.get("https://example.com/login")
    
    username_field = driver.find_element_by_id("username")
    password_field = driver.find_element_by_id("password")
    
    username_field.send_keys("testuser")
    password_field.send_keys("testpassword")
    password_field.send_keys(Keys.RETURN)
    
    assert "Welcome, testuser" in driver.page_source, "Login failed"
    
    driver.quit()

system_test_login()
print("System test: Login functionality passed")
```

Slide 4: Load Testing

Load testing assesses a system's ability to handle high workloads and identifies performance bottlenecks. It helps ensure that the system can maintain stability and responsiveness under stress.

```python
import aiohttp

async def make_request(session, url):
    async with session.get(url) as response:
        return await response.text()

async def load_test(num_requests):
    url = "https://api.example.com/data"
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, url) for _ in range(num_requests)]
        responses = await asyncio.gather(*tasks)
    
    success_count = sum(1 for r in responses if r is not None)
    print(f"Successful requests: {success_count}/{num_requests}")

asyncio.run(load_test(1000))
```

Slide 5: Error Testing

Error testing evaluates how a system handles invalid inputs, unexpected conditions, and error scenarios. It helps improve the robustness and reliability of the software.

```python
    try:
        result = a / b
    except ZeroDivisionError:
        return "Error: Division by zero"
    except TypeError:
        return "Error: Invalid input types"
    else:
        return result

def test_error_handling():
    print(divide_numbers(10, 2))  # Expected: 5.0
    print(divide_numbers(10, 0))  # Expected: Error: Division by zero
    print(divide_numbers("10", 2))  # Expected: Error: Invalid input types

test_error_handling()
```

Slide 6: Test Automation

Test automation involves creating and running automated test scripts to improve efficiency, repeatability, and coverage of testing processes. It helps catch regressions and ensures consistent test execution.

```python

def calculate_area(length, width):
    return length * width

@pytest.mark.parametrize("length,width,expected", [
    (5, 3, 15),
    (2.5, 4, 10),
    (0, 10, 0),
])
def test_calculate_area(length, width, expected):
    assert calculate_area(length, width) == expected

# Run tests with: pytest test_file.py
```

Slide 7: Behavior-Driven Development (BDD)

BDD is an approach that emphasizes collaboration between developers, QA, and non-technical stakeholders. It uses natural language descriptions of software behaviors as the basis for test cases.

```python

@given('the user is on the login page')
def step_impl(context):
    context.browser.get('https://example.com/login')

@when('the user enters valid credentials')
def step_impl(context):
    context.browser.find_element_by_id('username').send_keys('testuser')
    context.browser.find_element_by_id('password').send_keys('testpassword')
    context.browser.find_element_by_id('login-button').click()

@then('the user should be redirected to the dashboard')
def step_impl(context):
    assert 'dashboard' in context.browser.current_url
```

Slide 8: Test-Driven Development (TDD)

TDD is a development process where tests are written before the actual code. This approach ensures that code is testable from the start and helps developers focus on meeting requirements.

```python

class TestStringMethods(unittest.TestCase):
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

if __name__ == '__main__':
    unittest.main()

# Implement the actual code after writing these tests
```

Slide 9: Continuous Integration and Continuous Deployment (CI/CD)

CI/CD practices automate the integration, testing, and deployment of code changes. This ensures that new code is regularly tested and can be safely deployed to production.

```python
name: CI/CD Pipeline

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.x'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: pytest

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to production
      run: |
        # Add deployment steps here
```

Slide 10: Performance Testing

Performance testing evaluates the speed, responsiveness, and stability of a system under various conditions. It helps identify bottlenecks and ensures the system meets performance requirements.

```python
import statistics

def measure_execution_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

execution_times = [measure_execution_time(fibonacci, 30) for _ in range(10)]
average_time = statistics.mean(execution_times)
print(f"Average execution time: {average_time:.4f} seconds")
```

Slide 11: Security Testing

Security testing identifies vulnerabilities and weaknesses in a system's defenses. It helps protect against potential threats and ensures that sensitive data remains secure.

```python
from bs4 import BeautifulSoup

def check_xss_vulnerability(url):
    payload = "<script>alert('XSS')</script>"
    response = requests.get(url, params={"input": payload})
    soup = BeautifulSoup(response.text, 'html.parser')
    
    if payload in str(soup):
        print(f"Potential XSS vulnerability found at {url}")
    else:
        print(f"No XSS vulnerability detected at {url}")

check_xss_vulnerability("https://example.com/search")
```

Slide 12: Usability Testing

Usability testing evaluates how easily users can interact with a system. It focuses on user experience, interface design, and overall satisfaction with the product.

```python
from tkinter import messagebox

def simulate_usability_test():
    def submit_form():
        name = name_entry.get()
        age = age_entry.get()
        if name and age:
            messagebox.showinfo("Success", f"Thank you, {name}!")
        else:
            messagebox.showerror("Error", "Please fill all fields")

    root = tk.Tk()
    root.title("Usability Test")

    tk.Label(root, text="Name:").grid(row=0, column=0)
    name_entry = tk.Entry(root)
    name_entry.grid(row=0, column=1)

    tk.Label(root, text="Age:").grid(row=1, column=0)
    age_entry = tk.Entry(root)
    age_entry.grid(row=1, column=1)

    submit_button = tk.Button(root, text="Submit", command=submit_form)
    submit_button.grid(row=2, column=1)

    root.mainloop()

simulate_usability_test()
```

Slide 13: Regression Testing

Regression testing ensures that new code changes do not adversely affect existing functionality. It helps maintain the stability and reliability of the system over time.

```python

class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()

    def test_add(self):
        self.assertEqual(self.calc.add(2, 3), 5)

    def test_subtract(self):
        self.assertEqual(self.calc.subtract(5, 3), 2)

    # New test for multiplication (not yet implemented)
    def test_multiply(self):
        self.assertEqual(self.calc.multiply(2, 3), 6)

if __name__ == '__main__':
    unittest.main()
```

Slide 14: Real-Life Example: E-commerce Website Testing

Consider testing an e-commerce website. We'll focus on the product search functionality, which is crucial for user experience and sales.

```python
from bs4 import BeautifulSoup

def test_product_search(base_url, search_term):
    search_url = f"{base_url}/search?q={search_term}"
    response = requests.get(search_url)
    
    if response.status_code != 200:
        print(f"Error: Unable to access {search_url}")
        return False
    
    soup = BeautifulSoup(response.text, 'html.parser')
    product_elements = soup.find_all('div', class_='product-item')
    
    if not product_elements:
        print(f"No products found for '{search_term}'")
        return False
    
    print(f"Found {len(product_elements)} products for '{search_term}'")
    for product in product_elements[:3]:  # Display first 3 products
        name = product.find('h2', class_='product-name').text.strip()
        price = product.find('span', class_='product-price').text.strip()
        print(f"- {name}: {price}")
    
    return True

# Test the search functionality
base_url = "https://example-ecommerce.com"
test_product_search(base_url, "laptop")
test_product_search(base_url, "nonexistent_product")
```

Slide 15: Real-Life Example: Weather App Testing

Let's test a weather application that provides current weather conditions and forecasts for different locations.

```python
import json

def test_weather_api(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print(f"Error: Unable to fetch weather data for {city}")
        return False
    
    data = json.loads(response.text)
    
    print(f"Weather in {city}:")
    print(f"Temperature: {data['main']['temp']}°C")
    print(f"Humidity: {data['main']['humidity']}%")
    print(f"Description: {data['weather'][0]['description']}")
    
    return True

# Test the weather API
api_key = "your_api_key_here"
test_weather_api(api_key, "London")
test_weather_api(api_key, "Tokyo")
test_weather_api(api_key, "NonexistentCity")
```

Slide 16: Additional Resources

For further exploration of system functionality testing, consider these resources:

1. "Software Testing: A Craftsman's Approach" by Paul C. Jorgensen
2. "Effective Software Testing: A Developer's Guide" by Mauricio Aniche
3. "The Art of Software Testing" by Glenford J. Myers, Corey Sandler, and Tom Badgett
4. Python unittest documentation: [https://docs.python.org/3/library/unittest.html](https://docs.python.org/3/library/unittest.html)
5. Pytest documentation: [https://docs.pytest.org/](https://docs.pytest.org/)

These resources provide in-depth information on various testing techniques, best practices, and tools to enhance your system functionality testing skills.


