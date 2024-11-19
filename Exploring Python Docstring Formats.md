## Exploring Python Docstring Formats
Slide 1: Python Docstring Basics - reStructuredText Style

The reStructuredText (reST) docstring format, originally developed for Python documentation, uses a semantic markup syntax to define documentation elements. It's the default format recognized by Sphinx and provides rich features for creating detailed API documentation.

```python
def calculate_mean(numbers: list) -> float:
    """Calculate the arithmetic mean of a list of numbers.

    :param numbers: A list of numeric values
    :type numbers: list
    :returns: The arithmetic mean
    :rtype: float
    :raises ValueError: If the input list is empty
    
    Example:
    >>> calculate_mean([1, 2, 3, 4, 5])
    3.0
    """
    if not numbers:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(numbers) / len(numbers)
```

Slide 2: Google Style Docstrings

Google's docstring format emphasizes readability through clear section headers and indentation. This style has gained popularity due to its clean structure and ease of writing, making it particularly suitable for projects where simplicity is valued.

```python
def process_text_data(text: str, max_length: int = 100) -> dict:
    """Processes raw text data and returns various text metrics.
    
    Args:
        text (str): The input text to process
        max_length (int, optional): Maximum length to consider. Defaults to 100.
    
    Returns:
        dict: A dictionary containing:
            - word_count (int): Number of words
            - char_count (int): Number of characters
            - sentence_count (int): Number of sentences
    
    Raises:
        ValueError: If text is empty or None
    """
    if not text:
        raise ValueError("Input text cannot be empty")
    
    metrics = {
        'word_count': len(text.split()),
        'char_count': len(text),
        'sentence_count': text.count('.') + text.count('!') + text.count('?')
    }
    return metrics
```

Slide 3: NumPy Style Documentation

The NumPy documentation style combines the best aspects of reST and Google styles, providing a structured format particularly well-suited for scientific computing and data analysis projects where detailed parameter documentation is crucial.

```python
def analyze_dataset(data: np.ndarray, features: list, 
                   normalize: bool = True) -> tuple:
    """
    Perform statistical analysis on a numerical dataset.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    features : list
        List of feature names corresponding to columns
    normalize : bool, optional
        Whether to normalize the data, by default True

    Returns
    -------
    tuple
        statistics : dict
            Dictionary containing mean, std, min, max for each feature
        normalized_data : np.ndarray
            Normalized dataset if normalize=True, else None

    Examples
    --------
    >>> data = np.array([[1, 2], [3, 4]])
    >>> features = ['A', 'B']
    >>> stats, norm_data = analyze_dataset(data, features)
    """
    statistics = {}
    for i, feature in enumerate(features):
        statistics[feature] = {
            'mean': np.mean(data[:, i]),
            'std': np.std(data[:, i]),
            'min': np.min(data[:, i]),
            'max': np.max(data[:, i])
        }
    
    if normalize:
        normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    else:
        normalized_data = None
        
    return statistics, normalized_data
```

Slide 4: Class Documentation with Epytext Style

Epytext style, derived from Epydoc, offers a format particularly well-suited for documenting complex class hierarchies and their interactions, making it valuable for object-oriented Python applications.

```python
class DataProcessor:
    """
    @brief: A class for processing and validating input data
    @author: Data Science Team
    
    @type data: pandas.DataFrame
    @ivar data: The input dataset to process
    
    @type config: dict
    @ivar config: Configuration parameters
    """
    
    def __init__(self, data, config=None):
        """
        @param data: Input dataset
        @type data: pandas.DataFrame
        
        @param config: Configuration dictionary
        @type config: dict
        
        @raise ValueError: If data is None or empty
        """
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty")
        self.data = data
        self.config = config or {}

    def validate(self):
        """
        @return: Validation results
        @rtype: dict
        """
        return {
            'rows': len(self.data),
            'columns': list(self.data.columns),
            'missing': self.data.isnull().sum().to_dict()
        }
```

Slide 5: Real-world Example - Data Analysis Documentation

This example demonstrates comprehensive docstring documentation in a real-world data analysis scenario, implementing a complete data preprocessing pipeline with proper documentation that follows industry best practices.

```python
from typing import Tuple, Optional
import pandas as pd
import numpy as np

class DataPreprocessor:
    """
    A comprehensive data preprocessing pipeline for machine learning tasks.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing raw data
    target_col : str
        Name of the target variable column
    categorical_threshold : int, optional
        Maximum number of unique values to consider a column categorical
        
    Attributes
    ----------
    X : pd.DataFrame
        Processed feature matrix
    y : pd.Series
        Processed target variable
    
    Examples
    --------
    >>> data = pd.read_csv('customer_data.csv')
    >>> preprocessor = DataPreprocessor(data, 'purchase_amount')
    >>> X_clean, y_clean = preprocessor.fit_transform()
    """
    
    def __init__(self, df: pd.DataFrame, 
                 target_col: str,
                 categorical_threshold: int = 10):
        self.df = df.copy()
        self.target_col = target_col
        self.categorical_threshold = categorical_threshold
        self._validate_input()
        
    def _validate_input(self) -> None:
        """Validates input data integrity."""
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found")
        if self.df.empty:
            raise ValueError("DataFrame cannot be empty")
            
    def fit_transform(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Executes the complete preprocessing pipeline.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Processed features and target variable
        """
        # Split features and target
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Handle missing values
        X = self._handle_missing(X)
        
        # Encode categorical variables
        X = self._encode_categorical(X)
        
        return X, y
        
    def _handle_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values in the dataset.
        
        Strategy:
        - Numeric: median imputation
        - Categorical: mode imputation
        """
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            X[col].fillna(X[col].median(), inplace=True)
            
        for col in categorical_cols:
            X[col].fillna(X[col].mode()[0], inplace=True)
            
        return X
        
    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Performs one-hot encoding on categorical variables."""
        categorical_cols = []
        
        for col in X.columns:
            if X[col].nunique() < self.categorical_threshold:
                categorical_cols.append(col)
                
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, 
                             drop_first=True)
        
        return X
```

Slide 6: Scientific Computing Documentation

This slide demonstrates documenting scientific computing functions with mathematical formulas and comprehensive parameter descriptions using NumPy style docstrings.

```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     weights: Optional[np.ndarray] = None) -> dict:
    """
    Calculate various regression and classification metrics.
    
    The following metrics are computed:
    $$MSE = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
    $$RMSE = \sqrt{MSE}$$
    $$MAE = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$$
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
    weights : np.ndarray, optional
        Sample weights for weighted metrics
        
    Returns
    -------
    dict
        Dictionary containing computed metrics:
        - mse: Mean Squared Error
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        
    Examples
    --------
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
    >>> metrics = calculate_metrics(y_true, y_pred)
    >>> print(f"RMSE: {metrics['rmse']:.3f}")
    RMSE: 0.158
    """
    if weights is None:
        weights = np.ones_like(y_true)
        
    # Ensure arrays have same shape
    assert y_true.shape == y_pred.shape == weights.shape
    
    # Calculate weighted metrics
    mse = np.average((y_true - y_pred) ** 2, weights=weights)
    rmse = np.sqrt(mse)
    mae = np.average(np.abs(y_true - y_pred), weights=weights)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }
```

Slide 7: Advanced Function Documentation

Deep learning model documentation requires precise parameter specifications and mathematical formulas for clarity. This example shows how to document complex neural network operations using NumPy style.

```python
import numpy as np
from typing import Union, List, Tuple

def neural_layer_forward(
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    activation: str = 'relu'
) -> Tuple[np.ndarray, dict]:
    """
    Implements forward propagation for a neural network layer.
    
    The computation follows:
    $$Z = XW + b$$
    $$A = g(Z)$$
    
    Where g(Z) is the activation function:
    For ReLU: $$g(Z) = max(0, Z)$$
    For Sigmoid: $$g(Z) = \frac{1}{1 + e^{-Z}}$$
    
    Parameters
    ----------
    X : np.ndarray
        Input data of shape (m, n_features)
    W : np.ndarray
        Weight matrix of shape (n_features, n_neurons)
    b : np.ndarray
        Bias vector of shape (1, n_neurons)
    activation : str
        Activation function: 'relu' or 'sigmoid'
        
    Returns
    -------
    A : np.ndarray
        Output of the activation function
    cache : dict
        Cache containing 'Z', 'W', 'b' for backpropagation
        
    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> W = np.array([[0.1, 0.2], [0.3, 0.4]])
    >>> b = np.array([[0.01, 0.01]])
    >>> A, cache = neural_layer_forward(X, W, b)
    """
    Z = np.dot(X, W) + b
    
    if activation == "relu":
        A = np.maximum(0, Z)
    elif activation == "sigmoid":
        A = 1 / (1 + np.exp(-Z))
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    
    cache = {
        'Z': Z,
        'W': W,
        'b': b
    }
    
    return A, cache
```

Slide 8: Documenting Custom Data Structures

This example shows how to document custom data structures with inheritance, demonstrating proper documentation of class hierarchies and their relationships.

```python
class DataNode:
    """
    Base class for data structure nodes.
    
    Parameters
    ----------
    value : Any
        Node value
    next_node : DataNode, optional
        Reference to the next node
        
    Attributes
    ----------
    value : Any
        Stored node value
    next : DataNode
        Reference to next node
    """
    
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node
        
class LinkedList:
    """
    Custom linked list implementation with comprehensive operation tracking.
    
    The list maintains the following invariants:
    1. Head points to first element or None if empty
    2. Tail points to last element or None if empty
    3. Size is always accurate
    
    Parameters
    ----------
    items : list, optional
        Initial items to populate the list
        
    Attributes
    ----------
    head : DataNode
        First node in the list
    tail : DataNode
        Last node in the list
    size : int
        Current number of elements
        
    Examples
    --------
    >>> lst = LinkedList([1, 2, 3])
    >>> lst.append(4)
    >>> print(lst.to_list())
    [1, 2, 3, 4]
    """
    
    def __init__(self, items=None):
        self.head = None
        self.tail = None
        self.size = 0
        
        if items:
            for item in items:
                self.append(item)
                
    def append(self, value) -> None:
        """
        Append a new value to the end of the list.
        
        Parameters
        ----------
        value : Any
            Value to append
        """
        new_node = DataNode(value)
        
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
            
        self.size += 1
        
    def to_list(self) -> list:
        """
        Convert linked list to Python list.
        
        Returns
        -------
        list
            List containing all elements
        """
        result = []
        current = self.head
        
        while current:
            result.append(current.value)
            current = current.next
            
        return result
```

Slide 9: Algorithm Implementation Documentation

This example demonstrates how to document complex algorithms with time complexity analysis and implementation details using Google style docstrings for maximum clarity.

```python
def quicksort(arr: list, low: int = None, high: int = None) -> list:
    """
    Implements the QuickSort algorithm with Hoare partition scheme.
    
    Time Complexity Analysis:
    Average Case: $$O(n \log n)$$
    Worst Case: $$O(n^2)$$
    Space Complexity: $$O(\log n)$$
    
    Args:
        arr (list): Input array to be sorted
        low (int, optional): Starting index of the partition
        high (int, optional): Ending index of the partition
    
    Returns:
        list: Sorted array
        
    Example:
        >>> data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        >>> quicksort(data)
        [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
    """
    def partition(low: int, high: int) -> int:
        """Helper function to partition the array."""
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    if low is None:
        low = 0
    if high is None:
        high = len(arr) - 1
        
    def _quicksort(low: int, high: int) -> None:
        """Recursive quicksort implementation."""
        if low < high:
            pi = partition(low, high)
            _quicksort(low, pi - 1)
            _quicksort(pi + 1, high)
    
    _quicksort(low, high)
    return arr
```

Slide 10: Database Interface Documentation

Documentation for database interfaces requires clear specification of connection parameters, transaction handling, and error cases, demonstrated here using reStructuredText style.

```python
class DatabaseManager:
    """
    Manages database connections and operations with proper transaction handling.
    
    :param host: Database host address
    :type host: str
    :param port: Database port number
    :type port: int
    :param database: Database name
    :type database: str
    :param user: Database username
    :type user: str
    :param password: Database password
    :type password: str
    :raises ConnectionError: If database connection fails
    
    Usage Example::
        
        with DatabaseManager('localhost', 5432, 'mydb', 'user', 'pass') as db:
            results = db.execute_query("SELECT * FROM users")
    """
    
    def __init__(self, host: str, port: int, database: str,
                 user: str, password: str):
        self.config = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.connection = None
        self.cursor = None
        
    def __enter__(self):
        """
        Establishes database connection with transaction management.
        
        :returns: Self for context manager usage
        :rtype: DatabaseManager
        """
        try:
            self.connection = self._connect()
            self.cursor = self.connection.cursor()
            return self
        except Exception as e:
            raise ConnectionError(f"Failed to connect: {str(e)}")
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Handles connection cleanup and transaction completion.
        
        :param exc_type: Exception type if error occurred
        :param exc_val: Exception value if error occurred
        :param exc_tb: Exception traceback if error occurred
        """
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
            
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            
    def execute_query(self, query: str, params: tuple = None) -> list:
        """
        Executes SQL query with parameter binding.
        
        :param query: SQL query string
        :type query: str
        :param params: Query parameters for binding
        :type params: tuple
        :returns: Query results
        :rtype: list
        """
        self.cursor.execute(query, params or ())
        return self.cursor.fetchall()
```

Slide 11: Machine Learning Model Documentation

This comprehensive example shows how to document machine learning models with proper parameter descriptions, mathematical formulas, and implementation details using NumPy style docstrings.

```python
class GradientBoostingClassifier:
    """
    Gradient Boosting Classifier implementation with custom loss functions.
    
    The model implements the gradient boosting algorithm:
    $$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$
    
    Where:
    $$\gamma_m = \arg\min_{\gamma} \sum_{i=1}^n L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$$
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages to perform
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree
    max_depth : int, default=3
        Maximum depth of the regression trees
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node
        
    Attributes
    ----------
    estimators_ : list
        The collection of fitted sub-estimators
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances (higher = more important)
        
    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4)
    >>> clf = GradientBoostingClassifier(n_estimators=10)
    >>> clf.fit(X, y)
    >>> y_pred = clf.predict(X)
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.estimators_ = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingClassifier':
        """
        Build a gradient boosting classifier from the training set (X, y).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Returns self.
        """
        self.n_classes_ = len(np.unique(y))
        self.estimators_ = []
        
        # Initialize with zeros
        F = np.zeros((X.shape[0], self.n_classes_))
        
        for i in range(self.n_estimators):
            # Calculate negative gradient
            negative_gradient = self._calculate_negative_gradient(y, F)
            
            # Fit a regression tree to the negative gradient
            tree = self._fit_regression_tree(X, negative_gradient)
            self.estimators_.append(tree)
            
            # Update model
            F += self.learning_rate * tree.predict(X)
            
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
            
        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class probabilities of the input samples
        """
        F = np.zeros((X.shape[0], self.n_classes_))
        
        for estimator in self.estimators_:
            F += self.learning_rate * estimator.predict(X)
            
        return self._softmax(F)
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """
        Compute softmax values for each set of scores in x.
        
        Parameters
        ----------
        x : array-like of shape (n_samples, n_classes)
            The input samples
            
        Returns
        -------
        p : array of shape (n_samples, n_classes)
            Softmax probabilities
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

Slide 12: API Documentation with Error Handling

This example demonstrates comprehensive API documentation with detailed error handling specifications and response schemas using NumPy style docstrings.

```python
from typing import Dict, Optional, Union, Any
import json
from datetime import datetime

class APIEndpoint:
    """
    RESTful API endpoint handler with comprehensive error handling.
    
    The endpoint follows REST principles:
    - Stateless communication
    - Resource-based URLs
    - Proper HTTP method usage
    
    Parameters
    ----------
    base_url : str
        Base URL for the API endpoint
    timeout : int, optional
        Request timeout in seconds, default 30
    retry_attempts : int, optional
        Number of retry attempts for failed requests
        
    Attributes
    ----------
    headers : dict
        HTTP headers for requests
    session : requests.Session
        Persistent session for requests
        
    Examples
    --------
    >>> api = APIEndpoint("https://api.example.com", timeout=60)
    >>> response = api.get("/users/123")
    >>> print(response['status'])
    'success'
    """
    
    def __init__(self, base_url: str, timeout: int = 30,
                 retry_attempts: int = 3):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
    def request(self, method: str, endpoint: str, 
                data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute HTTP request with automatic retry and error handling.
        
        Parameters
        ----------
        method : str
            HTTP method (GET, POST, PUT, DELETE)
        endpoint : str
            API endpoint path
        data : dict, optional
            Request payload for POST/PUT methods
            
        Returns
        -------
        dict
            Parsed JSON response
            
        Raises
        ------
        APIError
            If request fails after all retry attempts
        ValidationError
            If response fails schema validation
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        attempt = 0
        
        while attempt < self.retry_attempts:
            try:
                response = self._execute_request(method, url, data)
                self._validate_response(response)
                return self._parse_response(response)
            except Exception as e:
                attempt += 1
                if attempt == self.retry_attempts:
                    raise APIError(f"Request failed: {str(e)}")
                    
    def _execute_request(self, method: str, url: str,
                        data: Optional[Dict]) -> Dict[str, Any]:
        """
        Execute single HTTP request with logging.
        
        Parameters
        ----------
        method : str
            HTTP method
        url : str
            Full request URL
        data : dict, optional
            Request payload
            
        Returns
        -------
        dict
            Raw response data
        """
        request_id = self._generate_request_id()
        self._log_request(request_id, method, url, data)
        
        try:
            response = self._send_request(method, url, data)
            self._log_response(request_id, response)
            return response
        except Exception as e:
            self._log_error(request_id, e)
            raise
            
    def _generate_request_id(self) -> str:
        """
        Generate unique request ID for tracking.
        
        Returns
        -------
        str
            Unique request identifier
        """
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"req_{timestamp}"
```

Slide 13: Advanced Testing Documentation

This example shows how to document test cases and testing utilities with proper docstrings that explain test coverage and assertions.

```python
import unittest
from typing import Any, Callable

class TestCase(unittest.TestCase):
    """
    Base test case class with enhanced assertion and documentation capabilities.
    
    Provides utilities for:
    - Parameterized testing
    - Custom assertions
    - Test case documentation
    
    Examples
    --------
    >>> class UserTests(TestCase):
    ...     @parameterized([
    ...         ("valid_email", "user@example.com", True),
    ...         ("invalid_email", "invalid-email", False)
    ...     ])
    ...     def test_email_validation(self, name, email, expected):
    ...         self.assertEqual(validate_email(email), expected)
    """
    
    def assertValidResponse(self, response: dict,
                          schema: dict,
                          msg: Optional[str] = None) -> None:
        """
        Assert that API response matches expected schema.
        
        Parameters
        ----------
        response : dict
            API response to validate
        schema : dict
            Expected response schema
        msg : str, optional
            Custom assertion message
            
        Raises
        ------
        AssertionError
            If response doesn't match schema
        """
        try:
            self._validate_schema(response, schema)
        except Exception as e:
            raise AssertionError(msg or str(e))
            
    def parameterized(cases: list) -> Callable:
        """
        Decorator for parameterized tests.
        
        Parameters
        ----------
        cases : list
            List of test cases with parameters
            
        Returns
        -------
        callable
            Decorated test method
            
        Examples
        --------
        >>> @parameterized([
        ...     ("case1", 1, 2, 3),
        ...     ("case2", 4, 5, 9)
        ... ])
        ... def test_addition(self, name, a, b, expected):
        ...     self.assertEqual(a + b, expected)
        """
        def decorator(func):
            def wrapper(self):
                for case in cases:
                    with self.subTest(name=case[0]):
                        func(self, *case)
            return wrapper
        return decorator
```

Slide 14: Advanced Error Handling Documentation

This example demonstrates comprehensive error handling documentation with inheritance hierarchies and custom exception types using reStructuredText style.

```python
class BaseError(Exception):
    """
    Base error class for custom exception hierarchy.
    
    :param message: Error description
    :type message: str
    :param code: Error code
    :type code: int
    :param details: Additional error details
    :type details: dict
    
    Example:
        >>> try:
        ...     raise ValidationError("Invalid input", code=400)
        ... except BaseError as e:
        ...     print(f"{e.code}: {str(e)}")
        400: Invalid input
    """
    
    def __init__(self, message: str, code: int = 500,
                 details: dict = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        
    def to_dict(self) -> dict:
        """
        Convert error to dictionary format.
        
        :returns: Dictionary representation of error
        :rtype: dict
        """
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'code': self.code,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }

class ValidationError(BaseError):
    """
    Raised when input validation fails.
    
    :param message: Validation error description
    :type message: str
    :param field: Name of invalid field
    :type field: str
    
    Example:
        >>> try:
        ...     raise ValidationError("Invalid email", field="email")
        ... except ValidationError as e:
        ...     print(e.to_dict())
    """
    
    def __init__(self, message: str, field: str = None):
        super().__init__(
            message=message,
            code=400,
            details={'field': field} if field else None
        )

class AuthenticationError(BaseError):
    """
    Raised when authentication fails.
    
    :param message: Authentication error description
    :type message: str
    :param user_id: ID of user attempting authentication
    :type user_id: str
    
    Example:
        >>> try:
        ...     raise AuthenticationError("Invalid token")
        ... except AuthenticationError as e:
        ...     print(f"Status {e.code}: {str(e)}")
    """
    
    def __init__(self, message: str, user_id: str = None):
        super().__init__(
            message=message,
            code=401,
            details={'user_id': user_id} if user_id else None
        )
```

Slide 15: Additional Resources

*   Enhancing Code Documentation with Neural Language Models
    *   [https://arxiv.org/abs/2105.14079](https://arxiv.org/abs/2105.14079)
*   Automated Python Documentation Generation: A Survey
    *   [https://arxiv.org/abs/2010.12687](https://arxiv.org/abs/2010.12687)
*   Best Practices for Scientific Computing Documentation
    *   [https://arxiv.org/abs/1810.08055](https://arxiv.org/abs/1810.08055)
*   Useful search terms for finding more resources:
    *   "Python docstring automation"
    *   "Documentation generation best practices"
    *   "Code documentation metrics"
*   Recommended tools:
    *   Sphinx Documentation Generator
    *   pydocstyle for docstring validation
    *   doctest for testing code examples

