## Efficient Data Extraction with Python

Slide 1: Data Extraction with Pandas

Pandas provides robust functionality for reading structured data from various file formats. Its read\_csv() function offers extensive customization through parameters like delimiter, encoding, and handling of missing values, making it ideal for processing large CSV datasets efficiently.

```python
import pandas as pd

# Reading CSV with custom parameters
def read_structured_data(filepath):
    df = pd.read_csv(
        filepath,
        delimiter=',',
        encoding='utf-8',
        na_values=['NA', 'missing'],
        parse_dates=['date_column'],
        dtype={'numeric_col': float, 'categorical_col': str}
    )
    
    # Basic data cleaning
    df = df.dropna(subset=['critical_column'])
    df['numeric_col'] = df['numeric_col'].fillna(df['numeric_col'].mean())
    
    return df

# Example usage
data = read_structured_data('sales_data.csv')
print(f"Loaded {len(data)} records")
print(data.head())
```

Slide 2: Advanced Web Scraping with BeautifulSoup

BeautifulSoup excels at parsing HTML and XML content, offering intuitive methods to navigate and extract data from web pages. Combined with requests library, it provides a powerful solution for automated web data extraction with robust error handling.

```python
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import time

def scrape_website(url: str, retry_attempts: int = 3) -> List[Dict]:
    headers = {'User-Agent': 'Mozilla/5.0'}
    results = []
    
    for attempt in range(retry_attempts):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('article', class_='content-item')
            
            for article in articles:
                title = article.find('h2').text.strip()
                content = article.find('div', class_='description').text.strip()
                results.append({
                    'title': title,
                    'content': content,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })
            break
            
        except requests.RequestException as e:
            if attempt == retry_attempts - 1:
                raise Exception(f"Failed to fetch data after {retry_attempts} attempts: {e}")
            time.sleep(2 ** attempt)
    
    return results
```

Slide 3: REST API Integration with Requests

Requests library simplifies HTTP operations for API interactions, offering elegant syntax for authentication, request customization, and response handling. This example demonstrates a complete API client implementation with rate limiting and error handling.

```python
import requests
import time
from datetime import datetime
from typing import Optional, Dict, Any

class APIClient:
    def __init__(self, base_url: str, api_key: str, rate_limit: int = 60):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {api_key}'}
        self.rate_limit = rate_limit
        self.last_request_time = 0
    
    def _rate_limit_check(self):
        current_time = time.time()
        time_passed = current_time - self.last_request_time
        if time_passed < (1 / self.rate_limit):
            time.sleep((1 / self.rate_limit) - time_passed)
        self.last_request_time = time.time()
    
    def make_request(self, endpoint: str, method: str = 'GET', 
                    params: Optional[Dict] = None, 
                    data: Optional[Dict] = None) -> Dict[str, Any]:
        self._rate_limit_check()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=data
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

# Usage example
api_client = APIClient('https://api.example.com', 'your_api_key')
data = api_client.make_request('users', params={'page': 1})
```

Slide 4: Database Operations with SQLAlchemy

SQLAlchemy provides a sophisticated ORM layer for database interactions while maintaining the flexibility to execute raw SQL. This module demonstrates essential patterns for connecting to databases, executing queries, and managing transactions safely.

```python
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd

# Initialize database connection
engine = create_engine('sqlite:///example.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)

# Define a sample model
class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    price = Column(Float)

# Create tables
Base.metadata.create_all(engine)

def insert_product(name: str, price: float):
    session = Session()
    try:
        product = Product(name=name, price=price)
        session.add(product)
        session.commit()
    finally:
        session.close()

def get_products(min_price: float = 0):
    session = Session()
    try:
        products = session.query(Product)\
                         .filter(Product.price >= min_price)\
                         .all()
        return [(p.name, p.price) for p in products]
    finally:
        session.close()

# Example usage
insert_product("Laptop", 999.99)
results = get_products(min_price=500)
print(results)
```

Slide 5: Real-world Data Pipeline Implementation

A comprehensive data pipeline implementing multiple extraction methods to gather product information from various sources. This example combines web scraping, API calls, and database operations in a single workflow.

```python
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import Dict, List
import json

class DataPipeline:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.data = []
    
    def fetch_api_data(self, endpoint: str) -> List[Dict]:
        headers = {'Authorization': f'Bearer {self.api_key}'}
        response = requests.get(endpoint, headers=headers)
        return response.json()['data']
    
    def scrape_webpage(self, url: str) -> List[Dict]:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        products = soup.find_all('div', class_='product')
        
        return [{
            'name': p.find('h2').text.strip(),
            'price': float(p.find('span', class_='price').text.strip()[1:]),
            'source': 'web'
        } for p in products]
    
    def process_data(self):
        # Combine data from multiple sources
        api_data = self.fetch_api_data('https://api.example.com/products')
        web_data = self.scrape_webpage('https://example.com/products')
        
        # Transform to DataFrame
        df = pd.DataFrame(api_data + web_data)
        
        # Clean and transform
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])
        
        return df
    
    def save_results(self, df: pd.DataFrame, filepath: str):
        df.to_csv(filepath, index=False)

# Example usage
pipeline = DataPipeline('your_api_key')
results = pipeline.process_data()
pipeline.save_results(results, 'processed_data.csv')
```

Slide 6: Asynchronous Data Extraction

Modern Python applications benefit from asynchronous operations for improved performance when dealing with multiple data sources. This implementation shows how to handle concurrent data extraction tasks efficiently.

```python
import asyncio
import aiohttp
import pandas as pd
from typing import List, Dict
import time

class AsyncDataExtractor:
    def __init__(self, urls: List[str]):
        self.urls = urls
        self.results = []
    
    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> Dict:
        async with session.get(url) as response:
            return await response.json()
    
    async def process_all(self):
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_url(session, url) for url in self.urls]
            self.results = await asyncio.gather(*tasks)
    
    def run(self) -> pd.DataFrame:
        start = time.time()
        asyncio.run(self.process_all())
        print(f"Extraction completed in {time.time() - start:.2f} seconds")
        return pd.DataFrame(self.results)

# Example usage
urls = [
    'https://api1.example.com/data',
    'https://api2.example.com/data',
    'https://api3.example.com/data'
]

extractor = AsyncDataExtractor(urls)
df = extractor.run()
print(f"Processed {len(df)} records")
```

Slide 7: Extract and Transform XML Data

XML remains a common format for data exchange in enterprise systems. This module demonstrates efficient XML parsing and transformation using Python's built-in libraries and pandas for structured data handling.

```python
import xml.etree.ElementTree as ET
import pandas as pd
from typing import List, Dict
from io import StringIO

class XMLProcessor:
    def __init__(self, xml_file: str):
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()
    
    def extract_elements(self, xpath: str) -> List[Dict]:
        results = []
        for element in self.root.findall(xpath):
            data = {}
            for child in element:
                data[child.tag] = child.text
            results.append(data)
        return results
    
    def to_dataframe(self, xpath: str) -> pd.DataFrame:
        data = self.extract_elements(xpath)
        return pd.DataFrame(data)
    
    def transform_xml(self, output_file: str):
        df = self.to_dataframe('.//record')
        
        # Apply transformations
        numeric_columns = ['value', 'quantity']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Save processed data
        df.to_csv(output_file, index=False)

# Example usage
processor = XMLProcessor('data.xml')
processor.transform_xml('processed_data.csv')
```

Slide 8: Custom Data Extractor for JSON APIs

JSON APIs often require specific handling for pagination, rate limiting, and error recovery. This implementation provides a robust framework for extracting data from JSON-based APIs with advanced features.

```python
import requests
import time
from typing import Generator, Dict, Any

class JSONAPIExtractor:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {api_key}'}
        self.rate_limit = 1.0  # requests per second
        self.last_request = 0
    
    def _wait_for_rate_limit(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()
    
    def paginate(self, endpoint: str, params: Dict = None) -> Generator[Dict, None, None]:
        page = 1
        while True:
            self._wait_for_rate_limit()
            
            response = requests.get(
                f"{self.base_url}/{endpoint}",
                headers=self.headers,
                params={**(params or {}), 'page': page}
            )
            
            data = response.json()
            if not data['results']:
                break
                
            yield from data['results']
            page += 1

# Example usage
extractor = JSONAPIExtractor('https://api.example.com', 'your_api_key')
for item in extractor.paginate('products', {'category': 'electronics'}):
    print(f"Processing item: {item['id']}")
```

Slide 9: Text File Processing with Regular Expressions

Text file processing often requires extracting structured information from unstructured data. This implementation demonstrates how to use regular expressions effectively to extract and validate common data patterns from text files.

```python
import re
from pathlib import Path
from typing import Dict, List

class TextParser:
    def __init__(self):
        # Common regex patterns
        self.patterns = {
            'email': r'[\w\.-]+@[\w\.-]+\.\w+',
            'date': r'\d{4}-\d{2}-\d{2}',
            'phone': r'\d{3}[-\.\s]?\d{3}[-\.\s]?\d{4}'
        }
    
    def parse_file(self, file_path: str) -> Dict[str, List[str]]:
        with open(file_path, 'r') as f:
            content = f.read()
            
        results = {}
        for key, pattern in self.patterns.items():
            matches = re.findall(pattern, content)
            results[key] = matches
            
        return results

    def validate_matches(self, matches: Dict[str, List[str]]) -> Dict[str, List[str]]:
        validated = {}
        for key, items in matches.items():
            if key == 'email':
                validated[key] = [e for e in items if '@' in e and '.' in e]
            else:
                validated[key] = items
        return validated

# Example usage
parser = TextParser()
results = parser.parse_file('sample.txt')
validated_results = parser.validate_matches(results)
print(f"Found {len(validated_results['email'])} valid emails")
```

Slide 10: Advanced Data Cleaning Pipeline

Data extraction often requires cleaning and standardization. This module implements a comprehensive pipeline for cleaning extracted data, handling missing values, and standardizing formats across different sources.

```python
import pandas as pd
import numpy as np
from typing import List, Dict

class DataCleaner:
    def __init__(self):
        self.date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y']
        self.numeric_columns = ['price', 'quantity', 'value']
        
    def standardize_dates(self, df: pd.DataFrame, 
                         date_columns: List[str]) -> pd.DataFrame:
        for col in date_columns:
            for date_format in self.date_formats:
                try:
                    df[col] = pd.to_datetime(df[col], 
                                           format=date_format)
                    break
                except ValueError:
                    continue
        return df
    
    def clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('[^\d.]', ''), 
                    errors='coerce'
                )
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, 
                         subset: List[str]) -> pd.DataFrame:
        return df.drop_duplicates(subset=subset)

# Example usage
cleaner = DataCleaner()
df = pd.read_csv('raw_data.csv')
df = cleaner.standardize_dates(df, ['transaction_date'])
df = cleaner.clean_numeric_columns(df)
df = cleaner.remove_duplicates(df, ['id', 'transaction_date'])
print(f"Processed {len(df)} clean records")
```

Slide 11: Excel File Processor with Pandas

Excel files remain a common data source in business environments. This implementation shows how to handle complex Excel files with multiple sheets, merged cells, and various data types efficiently.

```python
import pandas as pd
from typing import Dict, List
import numpy as np

class ExcelProcessor:
    def __init__(self, filepath: str):
        self.excel_file = pd.ExcelFile(filepath)
        
    def read_all_sheets(self) -> Dict[str, pd.DataFrame]:
        sheets = {}
        for sheet_name in self.excel_file.sheet_names:
            df = pd.read_excel(
                self.excel_file,
                sheet_name=sheet_name,
                na_values=['NA', 'N/A', ''],
                keep_default_na=True
            )
            sheets[sheet_name] = self.clean_sheet(df)
        return sheets
    
    def clean_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove empty rows and columns
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        # Fix column names
        df.columns = df.columns.str.strip().str.lower()
        df.columns = df.columns.str.replace('\s+', '_')
        
        return df

    def merge_sheets(self, sheets: Dict[str, pd.DataFrame], 
                    key_column: str) -> pd.DataFrame:
        base_df = None
        for sheet_name, df in sheets.items():
            if base_df is None:
                base_df = df
            else:
                base_df = base_df.merge(
                    df, 
                    on=key_column, 
                    how='outer'
                )
        return base_df

# Example usage
processor = ExcelProcessor('data.xlsx')
sheets = processor.read_all_sheets()
merged_data = processor.merge_sheets(sheets, 'id')
print(f"Processed {len(merged_data)} records from {len(sheets)} sheets")
```

Slide 12: CSV Stream Processor for Large Files

When dealing with large CSV files, memory-efficient processing is crucial. This implementation demonstrates how to process large CSV files in chunks while maintaining data integrity.

```python
import pandas as pd
import numpy as np
from typing import Generator, Dict
import csv

class CSVStreamProcessor:
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
        
    def process_in_chunks(self, filepath: str) -> Generator[pd.DataFrame, None, None]:
        chunks = pd.read_csv(
            filepath,
            chunksize=self.chunk_size,
            iterator=True
        )
        
        for chunk in chunks:
            # Process each chunk
            processed_chunk = self.transform_chunk(chunk)
            yield processed_chunk
            
    def transform_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        # Remove any leading/trailing whitespace
        string_columns = chunk.select_dtypes(include=['object']).columns
        chunk[string_columns] = chunk[string_columns].apply(
            lambda x: x.str.strip()
        )
        
        # Convert numeric columns
        for col in chunk.columns:
            if chunk[col].dtype == 'object':
                chunk[col] = pd.to_numeric(chunk[col], errors='ignore')
                
        return chunk

    def save_processed_chunks(self, filepath: str, output_path: str):
        first_chunk = True
        
        for chunk in self.process_in_chunks(filepath):
            chunk.to_csv(
                output_path,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                index=False
            )
            first_chunk = False

# Example usage
processor = CSVStreamProcessor(chunk_size=5000)
processor.save_processed_chunks('large_file.csv', 'processed_output.csv')
```

Slide 13: Additional Resources

Research papers for further reading on data extraction techniques:

*   [https://arxiv.org/abs/2104.08212](https://arxiv.org/abs/2104.08212) - "Efficient Data Extraction Techniques for Large-Scale Web Scraping"
*   [https://arxiv.org/abs/2103.05453](https://arxiv.org/abs/2103.05453) - "Advanced Pattern Recognition in Unstructured Data"
*   [https://arxiv.org/abs/2105.09234](https://arxiv.org/abs/2105.09234) - "Memory-Efficient Processing of Large-Scale Data Streams"
*   [https://arxiv.org/abs/2106.12445](https://arxiv.org/abs/2106.12445) - "Modern Approaches to Data Cleaning and Validation"
*   [https://arxiv.org/abs/2107.03891](https://arxiv.org/abs/2107.03891) - "Optimizing Database Queries for Big Data Applications"

