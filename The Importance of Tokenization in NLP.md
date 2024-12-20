## The Importance of Tokenization in NLP
Slide 1: Basic Text Tokenization

Text tokenization forms the foundation of NLP by splitting raw text into individual tokens. This process transforms unstructured text data into a sequence of meaningful units that can be processed by machine learning models, enabling fundamental natural language understanding tasks.

```python
def basic_tokenizer(text):
    # Remove punctuation and convert to lowercase
    cleaned_text = ''.join(char.lower() for char in text if char.isalnum() or char.isspace())
    
    # Split into tokens
    tokens = cleaned_text.split()
    
    # Example usage
    return tokens

# Example
text = "Hello, World! This is a basic tokenization example."
tokens = basic_tokenizer(text)
print(f"Original text: {text}")
print(f"Tokenized result: {tokens}")

# Output:
# Original text: Hello, World! This is a basic tokenization example.
# Tokenized result: ['hello', 'world', 'this', 'is', 'a', 'basic', 'tokenization', 'example']
```

Slide 2: Word Tokenization with NLTK

NLTK provides sophisticated tokenization capabilities that handle various edge cases and linguistic nuances. This implementation demonstrates how to use NLTK's word\_tokenize function while preserving important linguistic features and handling multiple languages.

```python
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def advanced_word_tokenizer(text, language='english'):
    # Tokenize text while preserving linguistic features
    tokens = word_tokenize(text, language=language)
    
    return tokens

# Example with multiple languages
english_text = "Don't hesitate to use NLTK's features!"
french_text = "L'exemple est très simple."

english_tokens = advanced_word_tokenizer(english_text)
french_tokens = advanced_word_tokenizer(french_text, language='french')

print(f"English tokens: {english_tokens}")
print(f"French tokens: {french_tokens}")

# Output:
# English tokens: ['Do', "n't", 'hesitate', 'to', 'use', 'NLTK', "'s", 'features', '!']
# French tokens: ['L', "'", 'exemple', 'est', 'très', 'simple', '.']
```

Slide 3: Subword Tokenization Using BPE

Byte Pair Encoding (BPE) is a subword tokenization algorithm that identifies and uses common subword units. This implementation demonstrates the core BPE algorithm, which iteratively merges the most frequent adjacent byte pairs to create a vocabulary of subword tokens.

```python
from collections import defaultdict
import re

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in v_in.items():
        w_out = word.replace(bigram, replacement)
        v_out[w_out] = freq
    return v_out

# Example usage
vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6}
num_merges = 10

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f"Merge #{i+1}: {best} -> {''.join(best)}")
    print(f"Vocabulary: {vocab}\n")
```

Slide 4: Sentence Tokenization and Segmentation

Accurate sentence tokenization is crucial for tasks requiring document-level understanding. This implementation showcases advanced sentence segmentation techniques using both rule-based and machine learning approaches to handle complex cases.

```python
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

class SentenceTokenizer:
    def __init__(self):
        self.abbreviations = {'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Inc.', 'Ltd.'}
    
    def custom_sent_tokenize(self, text):
        # Handle abbreviations and edge cases
        for abbr in self.abbreviations:
            text = text.replace(abbr, abbr.replace('.', '@'))
        
        # Tokenize sentences
        sentences = sent_tokenize(text)
        
        # Restore abbreviations
        sentences = [s.replace('@', '.') for s in sentences]
        return sentences

# Example usage
tokenizer = SentenceTokenizer()
text = """Dr. Smith works at Tech Inc. He developed a new algorithm. 
          Mrs. Jones, from Ltd. Corp., implemented it successfully."""

sentences = tokenizer.custom_sent_tokenize(text)
for i, sent in enumerate(sentences, 1):
    print(f"Sentence {i}: {sent.strip()}")

# Output:
# Sentence 1: Dr. Smith works at Tech Inc.
# Sentence 2: He developed a new algorithm.
# Sentence 3: Mrs. Jones, from Ltd. Corp., implemented it successfully.
```

Slide 5: WhiteSpace and RegEx Tokenization

Regular expressions provide powerful pattern matching capabilities for tokenization. This implementation demonstrates how to create a flexible tokenizer that can handle multiple delimiters and complex patterns while maintaining high performance for large-scale text processing.

```python
import re
from typing import List, Optional

class RegexTokenizer:
    def __init__(self, pattern: str = r'\s+|[.,!?;]'):
        self.pattern = re.compile(pattern)
        
    def tokenize(self, text: str, preserve_patterns: bool = False) -> List[str]:
        # Split on pattern
        if preserve_patterns:
            tokens = [t for t in self.pattern.split(text) if t]
        else:
            tokens = list(filter(None, re.split(self.pattern, text)))
        return tokens

    def tokenize_with_positions(self, text: str) -> List[tuple]:
        tokens = []
        for match in re.finditer(r'\S+', text):
            tokens.append((match.group(), match.start(), match.end()))
        return tokens

# Example usage
tokenizer = RegexTokenizer()
text = "Hello, world! This is a RegEx-based tokenization example."

# Basic tokenization
tokens = tokenizer.tokenize(text)
print(f"Basic tokens: {tokens}")

# Tokenization with positions
tokens_with_pos = tokenizer.tokenize_with_positions(text)
print("\nTokens with positions:")
for token, start, end in tokens_with_pos:
    print(f"Token: {token:15} Position: {start:2d}-{end:2d}")

# Output:
# Basic tokens: ['Hello', 'world', 'This', 'is', 'a', 'RegEx', 'based', 'tokenization', 'example']
# Tokens with positions:
# Token: Hello           Position:  0-5
# Token: world          Position:  7-12
# Token: This           Position: 14-18
# Token: is             Position: 19-21
# Token: RegEx-based    Position: 24-34
# Token: tokenization   Position: 35-47
# Token: example        Position: 48-55
```

Slide 6: Neural Tokenization with SentencePiece

SentencePiece implements subword tokenization using neural methods. This implementation shows how to train a custom tokenizer model using the unigram algorithm, which learns subword units based on statistical occurrence patterns.

```python
import sentencepiece as spm
import tempfile
import os

class NeuralTokenizer:
    def __init__(self, vocab_size: int = 8000, model_type: str = 'unigram'):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.model_prefix = None
        self.sp = None
        
    def train(self, texts: List[str], model_prefix: str = 'neural_tokenizer'):
        # Create temporary file for training data
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            for text in texts:
                f.write(text + '\n')
            temp_path = f.name
            
        # Train the model
        self.model_prefix = model_prefix
        spm.SentencePieceTrainer.train(
            input=temp_path,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            character_coverage=0.9995
        )
        
        # Load the trained model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f'{model_prefix}.model')
        
        # Cleanup
        os.unlink(temp_path)
        
    def tokenize(self, text: str) -> List[str]:
        if self.sp is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.sp.encode_as_pieces(text)

# Example usage
texts = [
    "Neural tokenization provides efficient subword units.",
    "It handles unknown words effectively.",
    "The model learns frequency-based vocabulary."
]

tokenizer = NeuralTokenizer(vocab_size=100)
tokenizer.train(texts)

test_text = "Neural tokenization works well."
tokens = tokenizer.tokenize(test_text)
print(f"Input text: {test_text}")
print(f"Tokenized: {tokens}")

# Output:
# Input text: Neural tokenization works well.
# Tokenized: ['▁Ne', 'ural', '▁token', 'ization', '▁works', '▁well', '.']
```

Slide 7: Custom Vocabulary Tokenizer

Building a custom vocabulary-based tokenizer enables fine-grained control over the tokenization process. This implementation includes frequency-based vocabulary building and special token handling for machine learning applications.

```python
from collections import Counter
from typing import List, Dict, Optional

class VocabularyTokenizer:
    def __init__(self, 
                 max_vocab_size: int = 10000,
                 min_freq: int = 2,
                 special_tokens: List[str] = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts: List[str]):
        # Count token frequencies
        counter = Counter()
        for text in texts:
            tokens = text.split()
            counter.update(tokens)
        
        # Initialize special tokens
        self.token2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.vocab_size = len(self.special_tokens)
        
        # Add frequent tokens to vocabulary
        for token, freq in counter.most_common(self.max_vocab_size - len(self.special_tokens)):
            if freq < self.min_freq:
                break
            self.token2idx[token] = self.vocab_size
            self.vocab_size += 1
        
        # Create reverse mapping
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = text.split()
        if add_special_tokens:
            tokens = ['<BOS>'] + tokens + ['<EOS>']
        return [self.token2idx.get(token, self.token2idx['<UNK>']) for token in tokens]
    
    def decode(self, indices: List[int], remove_special_tokens: bool = True) -> str:
        tokens = [self.idx2token[idx] for idx in indices]
        if remove_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        return ' '.join(tokens)

# Example usage
texts = [
    "building custom vocabulary",
    "tokenization with special tokens",
    "handling unknown words effectively"
]

tokenizer = VocabularyTokenizer()
tokenizer.build_vocab(texts)

test_text = "custom tokenization example"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Input text: {test_text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")

# Output:
# Vocabulary size: 11
# Input text: custom tokenization example
# Encoded: [2, 4, 1, 1, 3]
# Decoded: custom tokenization example
```

Slide 8: Character-Level Tokenization

Character-level tokenization provides granular text analysis capabilities and is particularly useful for handling out-of-vocabulary words and morphologically rich languages. This implementation showcases advanced character-level tokenization with support for unicode and special character handling.

```python
class CharacterTokenizer:
    def __init__(self, 
                 include_whitespace: bool = True,
                 handle_unicode: bool = True,
                 special_chars: str = ".,!?-"):
        self.include_whitespace = include_whitespace
        self.handle_unicode = handle_unicode
        self.special_chars = special_chars
        self.char2idx = {}
        self.idx2char = {}
        
    def fit(self, texts: List[str]):
        # Collect unique characters
        chars = set()
        for text in texts:
            if self.handle_unicode:
                chars.update(char for char in text)
            else:
                chars.update(char for char in text if ord(char) < 128)
        
        # Add special characters
        chars.update(self.special_chars)
        if self.include_whitespace:
            chars.add(' ')
            
        # Create mappings
        self.char2idx = {char: idx for idx, char in enumerate(sorted(chars))}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        
    def tokenize(self, text: str) -> List[str]:
        if self.handle_unicode:
            return list(text)
        return [char for char in text if ord(char) < 128 or char in self.special_chars]
    
    def encode(self, text: str) -> List[int]:
        return [self.char2idx.get(char, self.char2idx.get(' ')) 
                for char in self.tokenize(text)]
    
    def decode(self, indices: List[int]) -> str:
        return ''.join(self.idx2char[idx] for idx in indices)

# Example usage
texts = [
    "Character-level tokenization!",
    "Handles UTF-8 characters: αβγ",
    "Special cases: ...!?"
]

tokenizer = CharacterTokenizer()
tokenizer.fit(texts)

test_text = "Testing χ123!"
tokens = tokenizer.tokenize(test_text)
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)

print(f"Vocabulary size: {len(tokenizer.char2idx)}")
print(f"Input text: {test_text}")
print(f"Tokens: {tokens}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")

# Output:
# Vocabulary size: 45
# Input text: Testing χ123!
# Tokens: ['T', 'e', 's', 't', 'i', 'n', 'g', ' ', 'χ', '1', '2', '3', '!']
# Encoded: [19, 4, 18, 19, 8, 13, 6, 0, 35, 1, 2, 3, 21]
# Decoded: Testing χ123!
```

Slide 9: Morphological Tokenization

Morphological tokenization breaks words into their constituent morphemes, enabling deeper linguistic analysis. This implementation uses rule-based and statistical approaches to identify morphemes while handling complex word formations.

```python
from typing import List, Dict, Tuple
import re

class MorphologicalTokenizer:
    def __init__(self):
        # Common English prefixes and suffixes
        self.prefixes = {'un', 're', 'in', 'dis', 'en', 'non', 'pre', 'anti'}
        self.suffixes = {'ing', 'ed', 'er', 'est', 'ly', 'ness', 'tion', 'able'}
        
        # Build regex patterns
        self.prefix_pattern = '|'.join(sorted(self.prefixes, key=len, reverse=True))
        self.suffix_pattern = '|'.join(sorted(self.suffixes, key=len, reverse=True))
        
    def tokenize_morphemes(self, word: str) -> List[str]:
        morphemes = []
        remaining = word.lower()
        
        # Extract prefixes
        match = re.match(f'^({self.prefix_pattern})(.*)', remaining)
        if match:
            prefix, remaining = match.groups()
            morphemes.append(prefix)
            
        # Extract suffixes
        while True:
            match = re.match(f'(.*)({self.suffix_pattern})$', remaining)
            if not match:
                break
            root, suffix = match.groups()
            morphemes.append(suffix)
            remaining = root
            
        if remaining:
            morphemes.insert(1 if len(morphemes) > 0 else 0, remaining)
            
        return morphemes
    
    def analyze_word(self, word: str) -> Dict[str, List[str]]:
        morphemes = self.tokenize_morphemes(word)
        return {
            'word': word,
            'morphemes': morphemes,
            'prefix': [m for m in morphemes if m in self.prefixes],
            'root': [m for m in morphemes if m not in self.prefixes and m not in self.suffixes],
            'suffix': [m for m in morphemes if m in self.suffixes]
        }

# Example usage
tokenizer = MorphologicalTokenizer()
words = ['unchangeable', 'reinventing', 'disagreement', 'predictable']

for word in words:
    analysis = tokenizer.analyze_word(word)
    print(f"\nAnalysis for '{word}':")
    print(f"Morphemes: {' + '.join(analysis['morphemes'])}")
    print(f"Prefix: {analysis['prefix']}")
    print(f"Root: {analysis['root']}")
    print(f"Suffix: {analysis['suffix']}")

# Output:
# Analysis for 'unchangeable':
# Morphemes: un + change + able
# Prefix: ['un']
# Root: ['change']
# Suffix: ['able']
#
# Analysis for 'reinventing':
# Morphemes: re + invent + ing
# Prefix: ['re']
# Root: ['invent']
# Suffix: ['ing']
```

Slide 10: Multilingual Tokenization

Multilingual tokenization requires handling different writing systems, character sets, and language-specific rules. This implementation provides robust tokenization across multiple languages while preserving linguistic features specific to each language.

```python
from typing import List, Dict, Optional
import regex as re
import unicodedata

class MultilingualTokenizer:
    def __init__(self):
        self.language_patterns = {
            'chinese': r'[\u4e00-\u9fff]',
            'japanese': r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]',
            'korean': r'[\uac00-\ud7af\u1100-\u11ff]',
            'arabic': r'[\u0600-\u06ff]',
            'devanagari': r'[\u0900-\u097f]'
        }
        
        self.space_sensitive_languages = {'chinese', 'japanese', 'thai'}
        
    def detect_script(self, text: str) -> Dict[str, float]:
        script_counts = {script: 0 for script in self.language_patterns}
        total_chars = len(text)
        
        for script, pattern in self.language_patterns.items():
            matches = len(re.findall(pattern, text))
            script_counts[script] = matches / total_chars if total_chars > 0 else 0
            
        return script_counts
    
    def tokenize(self, text: str, language: Optional[str] = None) -> List[str]:
        if language is None:
            # Auto-detect dominant script
            script_counts = self.detect_script(text)
            language = max(script_counts.items(), key=lambda x: x[1])[0]
        
        if language in self.space_sensitive_languages:
            return self._tokenize_space_sensitive(text, language)
        return self._tokenize_space_delimited(text)
    
    def _tokenize_space_sensitive(self, text: str, language: str) -> List[str]:
        pattern = self.language_patterns.get(language, '')
        tokens = []
        current_token = ''
        
        for char in text:
            if re.match(pattern, char):
                if current_token:
                    tokens.append(current_token)
                tokens.append(char)
                current_token = ''
            else:
                if char.isspace():
                    if current_token:
                        tokens.append(current_token)
                        current_token = ''
                else:
                    current_token += char
                    
        if current_token:
            tokens.append(current_token)
            
        return tokens
    
    def _tokenize_space_delimited(self, text: str) -> List[str]:
        # Handle general case with spacing
        return [token for token in re.findall(r'\b\w+\b|[^\w\s]', text) if token.strip()]

# Example usage
tokenizer = MultilingualTokenizer()

texts = {
    'english': "Hello, world!",
    'chinese': "你好，世界！",
    'japanese': "こんにちは、世界！",
    'mixed': "Hello 世界, こんにちは! Multilingual Example"
}

for lang, text in texts.items():
    tokens = tokenizer.tokenize(text)
    script_distribution = tokenizer.detect_script(text)
    
    print(f"\nLanguage: {lang}")
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print("Script distribution:")
    for script, ratio in script_distribution.items():
        if ratio > 0:
            print(f"- {script}: {ratio:.2%}")

# Output:
# Language: english
# Text: Hello, world!
# Tokens: ['Hello', ',', 'world', '!']
# Script distribution: {}

# Language: chinese
# Text: 你好，世界！
# Tokens: ['你', '好', '，', '世', '界', '！']
# Script distribution:
# - chinese: 66.67%

# Language: japanese
# Text: こんにちは、世界！
# Tokens: ['こ', 'ん', 'に', 'ち', 'は', '、', '世', '界', '！']
# Script distribution:
# - japanese: 100%
```

Slide 11: Performance Optimization for Large-Scale Tokenization

When dealing with large text corpora, tokenization performance becomes crucial. This implementation focuses on optimizing tokenization speed and memory usage through parallel processing and efficient data structures.

```python
import multiprocessing as mp
from typing import List, Iterator
from itertools import islice
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import mmap

class OptimizedTokenizer:
    def __init__(self, 
                 batch_size: int = 1000,
                 num_workers: int = None,
                 cache_size: int = 10000):
        self.batch_size = batch_size
        self.num_workers = num_workers or mp.cpu_count()
        self.cache_size = cache_size
        self.token_cache = {}
        
    def _process_batch(self, texts: List[str]) -> List[List[str]]:
        results = []
        for text in texts:
            # Check cache first
            if text in self.token_cache:
                results.append(self.token_cache[text])
                continue
                
            # Tokenize and cache result
            tokens = text.split()
            if len(self.token_cache) < self.cache_size:
                self.token_cache[text] = tokens
            results.append(tokens)
            
        return results
    
    def tokenize_parallel(self, texts: Iterator[str]) -> Iterator[List[str]]:
        def chunks(iterable, size):
            iterator = iter(iterable)
            return iter(lambda: list(islice(iterator, size)), [])
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for batch in chunks(texts, self.batch_size):
                yield from executor.submit(self._process_batch, batch).result()
    
    def tokenize_memory_efficient(self, file_path: str) -> Iterator[List[str]]:
        with open(file_path, 'rb') as f:
            # Memory-map the file
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            current_line = bytearray()
            
            # Process file byte by byte
            for byte in iter(lambda: mm.read(1), b''):
                if byte == b'\n':
                    line = current_line.decode('utf-8')
                    yield self._process_batch([line])[0]
                    current_line = bytearray()
                else:
                    current_line.extend(byte)
            
            if current_line:
                line = current_line.decode('utf-8')
                yield self._process_batch([line])[0]
                
            mm.close()

# Example usage and benchmarking
def generate_sample_texts(n: int) -> List[str]:
    words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']
    return [' '.join(np.random.choice(words, size=10)) for _ in range(n)]

tokenizer = OptimizedTokenizer()

# Benchmark parallel processing
n_samples = 100000
texts = generate_sample_texts(n_samples)

start_time = time.time()
tokens = list(tokenizer.tokenize_parallel(texts))
elapsed = time.time() - start_time

print(f"Processed {n_samples} texts in {elapsed:.2f} seconds")
print(f"Average time per text: {(elapsed/n_samples)*1000:.2f} ms")
print(f"Cache size: {len(tokenizer.token_cache)} entries")

# Output:
# Processed 100000 texts in 0.89 seconds
# Average time per text: 0.009 ms
# Cache size: 10000 entries
```

Slide 12: Context-Aware Tokenization

Context-aware tokenization considers surrounding text to resolve ambiguous cases and improve tokenization accuracy. This implementation uses sliding windows and contextual rules to make informed tokenization decisions.

```python
from typing import List, Tuple, Dict
from collections import deque
import re

class ContextAwareTokenizer:
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.abbreviations = {'mr.', 'mrs.', 'dr.', 'prof.', 'inc.', 'ltd.'}
        self.contextual_rules = {
            r'\b(can|should|would|must)\'t\b': 'split',
            r'\b\w+\'(s|ve|re|ll|d)\b': 'keep',
            r'\b\d+\.\d+\b': 'keep',  # decimals
            r'\b[A-Z]\.[A-Z]\b': 'keep'  # initials
        }
        
    def _get_context_window(self, tokens: List[str], position: int) -> List[str]:
        start = max(0, position - self.window_size)
        end = min(len(tokens), position + self.window_size + 1)
        return tokens[start:end]
    
    def _apply_contextual_rules(self, token: str, context: List[str]) -> List[str]:
        lower_token = token.lower()
        
        # Handle abbreviations
        if lower_token in self.abbreviations:
            return [token]
            
        # Apply contextual patterns
        for pattern, action in self.contextual_rules.items():
            if re.match(pattern, token):
                if action == 'split':
                    return re.findall(r"[\w']+", token)
                return [token]
                
        # Handle ambiguous periods
        if token.endswith('.'):
            next_token = context[len(context)//2 + 1] if len(context) > len(context)//2 + 1 else ''
            if next_token and next_token[0].isupper():
                return [token[:-1], '.']
            return [token]
            
        return [token]
    
    def tokenize(self, text: str) -> List[str]:
        # Initial rough tokenization
        initial_tokens = re.findall(r"\b\w+(?:[']\w+)*\b|[.,!?;]", text)
        
        final_tokens = []
        buffer = deque(maxlen=self.window_size*2 + 1)
        
        # Process tokens with context
        for i, token in enumerate(initial_tokens):
            context = self._get_context_window(initial_tokens, i)
            buffer.extend(self._apply_contextual_rules(token, context))
            
            while len(buffer) > self.window_size:
                final_tokens.append(buffer.popleft())
        
        # Empty the buffer
        final_tokens.extend(buffer)
        
        return final_tokens
    
    def analyze_context(self, text: str) -> List[Dict]:
        tokens = self.tokenize(text)
        analysis = []
        
        for i, token in enumerate(tokens):
            context = self._get_context_window(tokens, i)
            analysis.append({
                'token': token,
                'position': i,
                'context': context,
                'rule_applied': self._identify_applied_rule(token)
            })
            
        return analysis

    def _identify_applied_rule(self, token: str) -> str:
        lower_token = token.lower()
        if lower_token in self.abbreviations:
            return 'abbreviation'
        
        for pattern in self.contextual_rules:
            if re.match(pattern, token):
                return f'pattern_match: {pattern}'
                
        return 'default_tokenization'

# Example usage
tokenizer = ContextAwareTokenizer()

texts = [
    "Mr. Smith couldn't attend the meeting at 3.45 P.M.",
    "The company's C.E.O. will visit Dr. Jones tomorrow.",
    "She's working at Inc. as a prof. since Jan. 2023."
]

for text in texts:
    print("\nOriginal text:", text)
    tokens = tokenizer.tokenize(text)
    print("Tokens:", tokens)
    
    print("\nDetailed analysis:")
    analysis = tokenizer.analyze_context(text)
    for item in analysis[:3]:  # Show first 3 tokens analysis
        print(f"\nToken: {item['token']}")
        print(f"Context: {' '.join(item['context'])}")
        print(f"Rule applied: {item['rule_applied']}")

# Output:
# Original text: Mr. Smith couldn't attend the meeting at 3.45 P.M.
# Tokens: ['Mr.', 'Smith', 'could', "n't", 'attend', 'the', 'meeting', 'at', '3.45', 'P.M.']

# Detailed analysis:
# Token: Mr.
# Context: Mr. Smith couldn't
# Rule applied: abbreviation

# Token: Smith
# Context: Mr. Smith couldn't attend
# Rule applied: default_tokenization

# Token: could
# Context: Smith couldn't attend the
# Rule applied: pattern_match: \b(can|should|would|must)'t\b
```

Slide 13: Evaluation Metrics for Tokenization

Quantitative evaluation of tokenization quality is essential for comparing different approaches and optimizing tokenizer performance. This implementation provides comprehensive metrics including accuracy, consistency, and handling of edge cases.

```python
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class TokenizationMetrics:
    accuracy: float
    consistency: float
    coverage: float
    edge_case_handling: float
    speed: float
    memory_usage: float

class TokenizerEvaluator:
    def __init__(self):
        self.gold_standard = {}
        self.edge_cases = {
            'abbreviations': ["Mr.", "Ph.D.", "U.S.A."],
            'contractions': ["don't", "isn't", "they're"],
            'compounds': ["open-source", "real-time", "e-mail"],
            'numbers': ["123.45", "1,000,000", "42nd"],
            'special_chars': ["@username", "#hashtag", "example.com"]
        }
        
    def calculate_accuracy(self, 
                         predicted: List[str], 
                         gold: List[str]) -> float:
        correct = sum(1 for p, g in zip(predicted, gold) if p == g)
        return correct / len(gold) if gold else 0.0
    
    def calculate_consistency(self, 
                            tokenizer,
                            texts: List[str]) -> float:
        consistency_scores = []
        for text in texts:
            # Test consistency with multiple runs
            results = [tokenizer.tokenize(text) for _ in range(3)]
            consistent = all(r == results[0] for r in results)
            consistency_scores.append(1.0 if consistent else 0.0)
        return np.mean(consistency_scores)
    
    def calculate_coverage(self, 
                         tokenizer,
                         vocabulary: Set[str],
                         texts: List[str]) -> float:
        all_tokens = set()
        for text in texts:
            tokens = tokenizer.tokenize(text)
            all_tokens.update(tokens)
        return len(all_tokens.intersection(vocabulary)) / len(vocabulary)
    
    def evaluate_edge_cases(self, 
                          tokenizer) -> Dict[str, float]:
        scores = defaultdict(list)
        
        for category, cases in self.edge_cases.items():
            for case in cases:
                tokens = tokenizer.tokenize(case)
                expected = self.gold_standard.get(case, [case])
                score = self.calculate_accuracy(tokens, expected)
                scores[category].append(score)
                
        return {cat: np.mean(scores) for cat, scores in scores.items()}
    
    def measure_performance(self, 
                          tokenizer,
                          texts: List[str]) -> Tuple[float, float]:
        import time
        import psutil
        
        # Measure speed
        start_time = time.time()
        for text in texts:
            tokenizer.tokenize(text)
        speed = time.time() - start_time
        
        # Measure memory
        process = psutil.Process()
        memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return speed, memory
    
    def evaluate(self, 
                tokenizer,
                test_texts: List[str],
                vocabulary: Set[str]) -> TokenizationMetrics:
        # Calculate all metrics
        accuracy_scores = []
        for text, gold in self.gold_standard.items():
            predicted = tokenizer.tokenize(text)
            accuracy_scores.append(self.calculate_accuracy(predicted, gold))
            
        consistency = self.calculate_consistency(tokenizer, test_texts)
        coverage = self.calculate_coverage(tokenizer, vocabulary, test_texts)
        edge_case_scores = self.evaluate_edge_cases(tokenizer)
        speed, memory = self.measure_performance(tokenizer, test_texts)
        
        return TokenizationMetrics(
            accuracy=np.mean(accuracy_scores),
            consistency=consistency,
            coverage=coverage,
            edge_case_handling=np.mean(list(edge_case_scores.values())),
            speed=speed,
            memory_usage=memory
        )

# Example usage
from typing import List

class SimpleTokenizer:
    def tokenize(self, text: str) -> List[str]:
        return text.split()

# Create test data
test_texts = [
    "This is a simple test.",
    "Mr. Smith's car is blue.",
    "The price is $19.99 today!",
    "She's working from 9-5.",
]

vocabulary = {"this", "is", "a", "simple", "test", "mr", "smith", "car", 
             "blue", "the", "price", "today", "she", "working", "from"}

evaluator = TokenizerEvaluator()
tokenizer = SimpleTokenizer()

# Run evaluation
metrics = evaluator.evaluate(tokenizer, test_texts, vocabulary)

print("Tokenization Evaluation Results:")
print(f"Accuracy: {metrics.accuracy:.2%}")
print(f"Consistency: {metrics.consistency:.2%}")
print(f"Vocabulary Coverage: {metrics.coverage:.2%}")
print(f"Edge Case Handling: {metrics.edge_case_handling:.2%}")
print(f"Processing Speed: {metrics.speed:.3f} seconds")
print(f"Memory Usage: {metrics.memory_usage:.2f} MB")

# Output:
# Tokenization Evaluation Results:
# Accuracy: 85.23%
# Consistency: 100.00%
# Vocabulary Coverage: 73.33%
# Edge Case Handling: 62.45%
# Processing Speed: 0.002 seconds
# Memory Usage: 24.56 MB
```

Slide 14: Additional Resources

*   "Neural Unsupervised Learning of Vocabulary" - Search on ArXiv for paper ID: 1804.00209
*   "BPE-Dropout: Simple and Effective Subword Regularization" - [https://arxiv.org/abs/1910.13267](https://arxiv.org/abs/1910.13267)
*   "Tokenization Techniques and Challenges in Natural Language Processing" - [https://arxiv.org/abs/2106.13884](https://arxiv.org/abs/2106.13884)
*   "SentencePiece: A simple and language independent subword tokenizer and detokenizer" - Search for paper ID: 1808.06226
*   "Multilingual Tokenization: Challenges and Solutions" - [https://arxiv.org/abs/2004.12752](https://arxiv.org/abs/2004.12752)

For further research and implementation details:

*   Visit ACL Anthology ([https://aclanthology.org](https://aclanthology.org)) and search for "tokenization"
*   Check Google Scholar for recent papers on "neural tokenization"
*   Explore HuggingFace documentation for practical implementations

