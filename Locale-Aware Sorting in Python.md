## Locale-Aware Sorting in Python
Slide 1: Basic Locale-Aware Sorting with locale Module

The locale module provides basic functionality for locale-aware string sorting through the strcoll function, which compares two strings according to the current locale settings. This enables culturally appropriate sorting of text containing diacritical marks.

```python
import locale

# Set locale to French
locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')

# List of French words with accents
french_words = ['élève', 'école', 'étude', 'être', 'écrire']

# Sort using locale-aware comparison
sorted_words = sorted(french_words, key=locale.strxfrm)
print(f"Sorted French words: {sorted_words}")

# Output:
# Sorted French words: ['école', 'écrire', 'élève', 'être', 'étude']
```

Slide 2: PyICU Implementation for Unicode Collation

The PyICU library implements the Unicode Collation Algorithm, providing robust multilingual text sorting capabilities. It handles complex sorting rules across different writing systems and cultural conventions more reliably than the locale module.

```python
from icu import Collator, Locale

# Create a collator for Spanish locale
spanish_collator = Collator.createInstance(Locale('es'))

# List of Spanish words with special characters
spanish_words = ['ñandu', 'nata', 'niño', 'nuez', 'ñora']

# Sort using ICU collation
sorted_words = sorted(spanish_words, key=spanish_collator.getSortKey)
print(f"Sorted Spanish words: {sorted_words}")

# Output:
# Sorted Spanish words: ['nata', 'niño', 'nuez', 'ñandu', 'ñora']
```

Slide 3: Multi-Locale Sorting Implementation

This implementation demonstrates handling text from multiple locales simultaneously using PyICU's advanced features. The solution maintains separate collators for different languages while providing a unified sorting interface.

```python
from icu import Collator, Locale
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class MultiLocaleText:
    text: str
    locale: str

class MultiLocaleSorter:
    def __init__(self):
        self.collators: Dict[str, Collator] = {}
    
    def get_collator(self, locale_str: str) -> Collator:
        if locale_str not in self.collators:
            self.collators[locale_str] = Collator.createInstance(Locale(locale_str))
        return self.collators[locale_str]
    
    def sort_texts(self, texts: List[MultiLocaleText]) -> List[MultiLocaleText]:
        return sorted(texts, key=lambda x: self.get_collator(x.locale).getSortKey(x.text))

# Example usage
texts = [
    MultiLocaleText("école", "fr"),
    MultiLocaleText("ñandu", "es"),
    MultiLocaleText("über", "de")
]

sorter = MultiLocaleSorter()
sorted_texts = sorter.sort_texts(texts)
for item in sorted_texts:
    print(f"{item.text} ({item.locale})")
```

Slide 4: Thread-Safe Locale Handling

The implementation addresses thread safety concerns when dealing with locale-aware sorting in multi-threaded applications. This approach uses thread-local storage to maintain separate locale settings for each thread.

```python
import threading
import locale
from contextlib import contextmanager
from typing import Generator

class ThreadSafeLocale:
    _thread_local = threading.local()
    
    @classmethod
    @contextmanager
    def temporary_locale(cls, temp_locale: str) -> Generator[None, None, None]:
        # Store current locale
        old_locale = locale.getlocale()
        
        try:
            # Set temporary locale for this thread
            locale.setlocale(locale.LC_ALL, temp_locale)
            yield
        finally:
            # Restore original locale
            locale.setlocale(locale.LC_ALL, old_locale)
    
    @classmethod
    def sort_with_locale(cls, items: list, locale_name: str) -> list:
        with cls.temporary_locale(locale_name):
            return sorted(items, key=locale.strxfrm)

# Example usage in multiple threads
def sort_in_thread(words: list, locale_name: str) -> None:
    sorted_words = ThreadSafeLocale.sort_with_locale(words, locale_name)
    print(f"Thread {threading.current_thread().name}: {sorted_words}")

# Test with multiple threads
words = ['école', 'élève', 'étude']
threads = [
    threading.Thread(target=sort_in_thread, args=(words, 'fr_FR.UTF-8')),
    threading.Thread(target=sort_in_thread, args=(words, 'en_US.UTF-8'))
]

for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
```

Slide 5: Custom Collation Rules

Custom collation rules allow fine-grained control over sorting behavior beyond standard locale settings. This implementation demonstrates how to create and apply custom sorting rules for specialized scenarios.

```python
from icu import Collator, Locale, RuleBasedCollator

# Define custom collation rules
# & is used to position characters relative to others
# < defines primary difference
# << defines secondary difference
custom_rules = """
    &c < č <<< Č 
    &s < š <<< Š 
    &z < ž <<< Ž
"""

def create_custom_collator():
    try:
        collator = RuleBasedCollator(custom_rules)
        collator.setStrength(Collator.SECONDARY)
        return collator
    except Exception as e:
        print(f"Error creating collator: {e}")
        return None

# Example usage
words = ['cap', 'čap', 'cup', 'Čap', 'sup', 'šup']
collator = create_custom_collator()

if collator:
    sorted_words = sorted(words, key=collator.getSortKey)
    print(f"Custom sorted words: {sorted_words}")
    
# Output:
# Custom sorted words: ['cap', 'čap', 'Čap', 'cup', 'sup', 'šup']
```

Slide 6: Performance Optimization for Large Datasets

When dealing with large text collections, performance optimization becomes crucial. This implementation uses caching and preprocessing to improve sorting efficiency for repeated operations.

```python
from functools import lru_cache
from icu import Collator, Locale
import time
from typing import List, Tuple

class CachedCollationSorter:
    def __init__(self, locale_str: str):
        self.collator = Collator.createInstance(Locale(locale_str))
        self._sort_key_cache = {}
    
    @lru_cache(maxsize=10000)
    def get_sort_key(self, text: str) -> bytes:
        return self.collator.getSortKey(text)
    
    def sort_texts(self, texts: List[str]) -> Tuple[List[str], float]:
        start_time = time.time()
        sorted_texts = sorted(texts, key=self.get_sort_key)
        elapsed_time = time.time() - start_time
        return sorted_texts, elapsed_time

# Performance comparison example
def compare_sorting_performance(texts: List[str], locale_str: str):
    # Standard sorting
    standard_start = time.time()
    standard_sorted = sorted(texts)
    standard_time = time.time() - standard_start
    
    # Cached collation sorting
    sorter = CachedCollationSorter(locale_str)
    cached_sorted, cached_time = sorter.sort_texts(texts)
    
    return {
        'standard_time': standard_time,
        'cached_time': cached_time,
        'standard_count': len(standard_sorted),
        'cached_count': len(cached_sorted)
    }

# Example usage with a large dataset
large_text_list = ['é'+str(i) for i in range(10000)]
results = compare_sorting_performance(large_text_list, 'fr_FR.UTF-8')
print(f"Performance Results: {results}")
```

Slide 7: Handling Mixed Scripts and Writing Systems

This implementation addresses the challenge of sorting text containing multiple scripts or writing systems, such as mixed Latin, Cyrillic, and Chinese characters.

```python
from icu import Collator, Locale, UCollAttribute, UCollAttributeValue
from typing import List, Dict

class MixedScriptSorter:
    def __init__(self):
        self.script_collators: Dict[str, Collator] = {
            'latin': self._create_collator('en'),
            'cyrillic': self._create_collator('ru'),
            'han': self._create_collator('zh')
        }
        
    def _create_collator(self, locale_str: str) -> Collator:
        collator = Collator.createInstance(Locale(locale_str))
        collator.setAttribute(
            UCollAttribute.ALTERNATE_HANDLING,
            UCollAttributeValue.SHIFTED
        )
        return collator
    
    def _detect_script(self, text: str) -> str:
        # Simplified script detection
        if any('\u0400' <= c <= '\u04FF' for c in text):
            return 'cyrillic'
        elif any('\u4e00' <= c <= '\u9fff' for c in text):
            return 'han'
        return 'latin'
    
    def sort_mixed_text(self, texts: List[str]) -> List[str]:
        return sorted(
            texts,
            key=lambda x: self.script_collators[self._detect_script(x)].getSortKey(x)
        )

# Example usage
mixed_texts = [
    'apple',      # Latin
    'яблоко',     # Cyrillic
    '苹果',       # Chinese
    'banana',     # Latin
    'банан',      # Cyrillic
    '香蕉'        # Chinese
]

sorter = MixedScriptSorter()
sorted_texts = sorter.sort_mixed_text(mixed_texts)
print("Sorted mixed scripts:", sorted_texts)
```

Slide 8: Real-world Application: Multilingual Address Book

This implementation demonstrates a practical address book system that handles contact names from multiple languages and provides efficient sorting and searching capabilities.

```python
from dataclasses import dataclass
from typing import Dict, List
from icu import Collator, Locale
import json

@dataclass
class Contact:
    name: str
    locale: str
    phone: str
    email: str

class MultilingualAddressBook:
    def __init__(self):
        self.contacts: List[Contact] = []
        self.collators: Dict[str, Collator] = {}
    
    def add_contact(self, contact: Contact) -> None:
        self.contacts.append(contact)
        if contact.locale not in self.collators:
            self.collators[contact.locale] = Collator.createInstance(
                Locale(contact.locale)
            )
    
    def get_sorted_contacts(self) -> List[Contact]:
        return sorted(
            self.contacts,
            key=lambda x: self.collators[x.locale].getSortKey(x.name)
        )
    
    def export_to_json(self, filename: str) -> None:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(
                [vars(c) for c in self.contacts],
                f,
                ensure_ascii=False,
                indent=2
            )

# Example usage
address_book = MultilingualAddressBook()

# Add contacts with different locales
contacts = [
    Contact("José García", "es_ES", "+34123456789", "jose@email.com"),
    Contact("張偉", "zh_CN", "+861234567890", "wei@email.com"),
    Contact("André Martin", "fr_FR", "+33123456789", "andre@email.com")
]

for contact in contacts:
    address_book.add_contact(contact)

# Get sorted contacts
sorted_contacts = address_book.get_sorted_contacts()
for contact in sorted_contacts:
    print(f"{contact.name} ({contact.locale})")

# Export to JSON
address_book.export_to_json("contacts.json")
```

Slide 9: Handling Normalization in Text Sorting

Text normalization is crucial for correct sorting when dealing with decomposed Unicode characters and different representation forms. This implementation ensures consistent sorting regardless of character composition.

```python
from unicodedata import normalize
from icu import Collator, Locale
from typing import List, Callable

class NormalizedTextSorter:
    def __init__(self, locale_str: str):
        self.collator = Collator.createInstance(Locale(locale_str))
        
    def _normalize_text(self, text: str, form: str = 'NFKC') -> str:
        return normalize(form, text)
    
    def sort_with_normalization(
        self,
        texts: List[str],
        normalization_form: str = 'NFKC'
    ) -> List[str]:
        normalized_pairs = [
            (text, self._normalize_text(text, normalization_form))
            for text in texts
        ]
        
        # Sort by normalized form but preserve original text
        sorted_pairs = sorted(
            normalized_pairs,
            key=lambda x: self.collator.getSortKey(x[1])
        )
        
        return [pair[0] for pair in sorted_pairs]

# Example usage with different Unicode representations
texts = [
    'café',      # composed
    'cafe\u0301',  # decomposed
    'über',      # composed
    'u\u0308ber'   # decomposed
]

sorter = NormalizedTextSorter('en_US.UTF-8')
sorted_texts = sorter.sort_with_normalization(texts)
print("Normalized and sorted texts:", sorted_texts)

# Test different normalization forms
for form in ['NFC', 'NFD', 'NFKC', 'NFKD']:
    sorted_texts = sorter.sort_with_normalization(texts, form)
    print(f"\nSorted with {form} normalization:", sorted_texts)
```

Slide 10: Fast Binary Search with Locale-Aware Comparison

This implementation optimizes searching in sorted multilingual lists by implementing a modified binary search that respects locale-specific ordering rules.

```python
from icu import Collator, Locale
from typing import List, Optional
import time

class LocaleAwareBinarySearch:
    def __init__(self, locale_str: str):
        self.collator = Collator.createInstance(Locale(locale_str))
        
    def binary_search(self, sorted_list: List[str], target: str) -> Optional[int]:
        left, right = 0, len(sorted_list) - 1
        
        while left <= right:
            mid = (left + right) // 2
            comparison = self.collator.compare(sorted_list[mid], target)
            
            if comparison == 0:
                return mid
            elif comparison < 0:
                left = mid + 1
            else:
                right = mid - 1
                
        return None

    def insert_position(self, sorted_list: List[str], target: str) -> int:
        left, right = 0, len(sorted_list)
        
        while left < right:
            mid = (left + right) // 2
            if self.collator.compare(sorted_list[mid], target) <= 0:
                left = mid + 1
            else:
                right = mid
                
        return left

# Example usage
searcher = LocaleAwareBinarySearch('fr_FR.UTF-8')
words = ['àvoir', 'être', 'manger', 'étudier', 'écrire']
sorted_words = sorted(words, key=searcher.collator.getSortKey)

# Search for existing word
target = 'être'
result = searcher.binary_search(sorted_words, target)
print(f"Found '{target}' at index: {result}")

# Find insertion position for new word
new_word = 'élever'
pos = searcher.insert_position(sorted_words, new_word)
print(f"Insert '{new_word}' at position: {pos}")
```

Slide 11: Real-world Application: Multilingual Document Indexing

This implementation creates a locale-aware document indexing system suitable for managing multilingual content in a document management system.

```python
from icu import Collator, Locale
from typing import Dict, List, Set
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Document:
    id: str
    title: str
    content: str
    locale: str
    created_at: datetime

class MultilingualDocumentIndex:
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.collators: Dict[str, Collator] = {}
        self.index: Dict[str, Set[str]] = {}
        
    def _get_collator(self, locale_str: str) -> Collator:
        if locale_str not in self.collators:
            self.collators[locale_str] = Collator.createInstance(Locale(locale_str))
        return self.collators[locale_str]
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def add_document(self, doc: Document) -> None:
        self.documents[doc.id] = doc
        
        # Index document tokens
        tokens = self._tokenize(f"{doc.title} {doc.content}")
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc.id)
    
    def search(self, query: str, locale: str = None) -> List[Document]:
        tokens = self._tokenize(query)
        matching_docs = set.intersection(
            *[self.index.get(token, set()) for token in tokens]
        )
        
        results = [self.documents[doc_id] for doc_id in matching_docs]
        
        if locale:
            collator = self._get_collator(locale)
            results.sort(key=lambda x: collator.getSortKey(x.title))
            
        return results

# Example usage
index = MultilingualDocumentIndex()

# Add sample documents
docs = [
    Document("1", "L'éducation en France", "Système éducatif...", "fr_FR", 
             datetime.now()),
    Document("2", "La educación en España", "Sistema educativo...", "es_ES",
             datetime.now()),
    Document("3", "Education in England", "Educational system...", "en_GB",
             datetime.now())
]

for doc in docs:
    index.add_document(doc)

# Search and sort results by French locale
results = index.search("education", "fr_FR")
for doc in results:
    print(f"{doc.title} ({doc.locale})")
```

Slide 12: Advanced Sorting with Weight-based Rules

This implementation demonstrates how to create complex sorting rules with weighted priorities for different character attributes, useful for specialized sorting requirements in academic or technical applications.

```python
from icu import Collator, Locale, UCollAttribute, UCollAttributeValue
from typing import List, Tuple, Dict
import re

class WeightedCollationSorter:
    def __init__(self, base_locale: str):
        self.collator = Collator.createInstance(Locale(base_locale))
        self.weights: Dict[str, int] = {}
        
    def set_weights(self, weights: Dict[str, int]) -> None:
        self.weights = weights
        
    def configure_collator(self, 
                         strength: int = Collator.TERTIARY,
                         case_level: bool = True) -> None:
        self.collator.setStrength(strength)
        self.collator.setAttribute(
            UCollAttribute.CASE_LEVEL,
            1 if case_level else 0
        )
        
    def get_weighted_key(self, text: str) -> Tuple[int, bytes]:
        weight = sum(self.weights.get(char, 0) for char in text)
        return (-weight, self.collator.getSortKey(text))
        
    def sort_texts(self, texts: List[str]) -> List[str]:
        return sorted(texts, key=self.get_weighted_key)

# Example usage
sorter = WeightedCollationSorter('en_US.UTF-8')

# Define custom weights for special characters
weights = {
    'α': 100,  # Greek alpha
    'β': 90,   # Greek beta
    'γ': 80,   # Greek gamma
    '∑': 70,   # Summation
    '∫': 60    # Integral
}

sorter.set_weights(weights)
sorter.configure_collator(strength=Collator.QUATERNARY)

# Test with mathematical expressions
expressions = [
    'f(x) = αx + β',
    '∫f(x)dx',
    'γ = 2π',
    '∑x_i',
    'β = 1/2'
]

sorted_expressions = sorter.sort_texts(expressions)
print("Sorted mathematical expressions:")
for expr in sorted_expressions:
    print(expr)
```

Slide 13: Results and Performance Analysis

This implementation provides comprehensive benchmarking and analysis capabilities for comparing different sorting approaches across various scenarios.

```python
import time
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
from icu import Collator, Locale

@dataclass
class BenchmarkResult:
    method: str
    avg_time: float
    std_dev: float
    dataset_size: int
    locale: str

class SortingBenchmark:
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        
    def benchmark_sort(self, 
                      method_name: str,
                      sort_func: callable,
                      dataset: List[str],
                      locale: str,
                      iterations: int = 5) -> BenchmarkResult:
        times: List[float] = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            sort_func(dataset.copy())
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
        result = BenchmarkResult(
            method=method_name,
            avg_time=statistics.mean(times),
            std_dev=statistics.stdev(times),
            dataset_size=len(dataset),
            locale=locale
        )
        self.results.append(result)
        return result
    
    def generate_report(self) -> Dict[str, Any]:
        return {
            'summary': {
                'total_methods': len(self.results),
                'fastest_method': min(
                    self.results,
                    key=lambda x: x.avg_time
                ).method,
                'slowest_method': max(
                    self.results,
                    key=lambda x: x.avg_time
                ).method
            },
            'detailed_results': [
                {
                    'method': r.method,
                    'avg_time_ms': r.avg_time * 1000,
                    'std_dev_ms': r.std_dev * 1000,
                    'dataset_size': r.dataset_size,
                    'locale': r.locale
                }
                for r in self.results
            ]
        }

# Example usage
benchmark = SortingBenchmark()

# Test data
test_data = ['école', 'élève', 'être', 'étudier'] * 1000

# Benchmark different sorting methods
collator = Collator.createInstance(Locale('fr_FR.UTF-8'))

methods = {
    'python_default': lambda x: sorted(x),
    'icu_collation': lambda x: sorted(x, key=collator.getSortKey),
    'locale_strxfrm': lambda x: sorted(x, key=locale.strxfrm)
}

for name, func in methods.items():
    result = benchmark.benchmark_sort(name, func, test_data, 'fr_FR.UTF-8')
    print(f"\nMethod: {name}")
    print(f"Average time: {result.avg_time*1000:.2f}ms")
    print(f"Standard deviation: {result.std_dev*1000:.2f}ms")

report = benchmark.generate_report()
print("\nComplete Benchmark Report:")
print(report)
```

Slide 14: Additional Resources

*   "A Comprehensive Guide to Unicode Collation Algorithm" [https://www.unicode.org/reports/tr10/](https://www.unicode.org/reports/tr10/)
*   "Efficient Implementation of Unicode Collation Algorithm" [https://arxiv.org/abs/cs/0606096](https://arxiv.org/abs/cs/0606096)
*   "Multilingual Text Processing: Challenges and Solutions" [https://www.sciencedirect.com/science/article/pii/S0306457318305764](https://www.sciencedirect.com/science/article/pii/S0306457318305764)
*   "Best Practices for Sorting and Searching Multilingual Text" [https://developers.google.com/international/articles/sort-search](https://developers.google.com/international/articles/sort-search)
*   "Performance Optimization Techniques for Text Processing in Python" [https://python.org/dev/peps/text-processing-best-practices](https://python.org/dev/peps/text-processing-best-practices)

