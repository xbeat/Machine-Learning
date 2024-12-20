## Tokens and Tokenization in Large Language Models in Python
Slide 1: 

Introduction to Tokens and Tokenization in LLMs

Tokenization is a fundamental process in natural language processing, especially for Large Language Models (LLMs). It involves breaking down text into smaller units called tokens. These tokens serve as the basic building blocks for text processing and understanding. Let's explore how tokenization works and its importance in LLMs using Python.

```python
text = "Hello, world! How are you?"
tokens = text.split()
print(tokens)
```

Slide 2: 

What are Tokens?

Tokens are the smallest units of text that carry meaning. They can be words, subwords, or even individual characters, depending on the tokenization strategy. In LLMs, tokens are used to represent and process text efficiently. Understanding tokens is crucial for working with LLMs and estimating computational requirements.

```python
import re

def simple_tokenize(text):
    return re.findall(r'\b\w+\b|[^\w\s]', text)

sample_text = "Don't forget: AI is amazing!"
tokens = simple_tokenize(sample_text)
print(tokens)
```

Slide 3: 

Types of Tokenization

There are various tokenization approaches, each with its own strengths and use cases. Word-based tokenization splits text into words, while character-based tokenization treats each character as a token. Subword tokenization, such as Byte-Pair Encoding (BPE) or WordPiece, strikes a balance between the two, allowing for better handling of rare words and morphological variations.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "Tokenization is fascinating!"
tokens = tokenizer.tokenize(text)
print(tokens)
```
 
Slide 4: 

Byte-Pair Encoding (BPE)

BPE is a popular subword tokenization algorithm used in many LLMs. It starts with a character-level vocabulary and iteratively merges the most frequent pairs of adjacent tokens to create new tokens. This process continues until a desired vocabulary size is reached, resulting in a balance between vocabulary size and token granularity.

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

files = ["path/to/file1.txt", "path/to/file2.txt"]
tokenizer.train(files, trainer)

encoded = tokenizer.encode("Hello, how are you?")
print(encoded.tokens)
```

Slide 5: 

WordPiece Tokenization

WordPiece is another subword tokenization method, commonly used in models like BERT. It's similar to BPE but uses a different scoring mechanism for merging tokens. WordPiece tends to prefer longer subword units and is particularly effective for languages with rich morphology.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "Unleashing the power of natural language processing!"
tokens = tokenizer.tokenize(text)
print(tokens)
```

Slide 6: 

Sentence Piece Tokenization

SentencePiece is a language-independent subword tokenizer that can handle any language without pre-tokenization. It treats the input text as a sequence of Unicode characters and learns to tokenize based on the frequency of character sequences. This makes it particularly useful for multilingual models and languages without clear word boundaries.

```python
import sentencepiece as spm

# Train SentencePiece model
spm.SentencePieceTrainer.train('--input=input.txt --model_prefix=m --vocab_size=2000')

# Load the model
sp = spm.SentencePieceProcessor()
sp.load('m.model')

text = "SentencePiece works on any language!"
tokens = sp.encode_as_pieces(text)
print(tokens)
```

Slide 7: 

Tokenization for Different Languages

Tokenization strategies can vary depending on the language. While word-based tokenization works well for languages with clear word boundaries (like English), it may not be suitable for languages without explicit word separators (like Chinese or Japanese). Subword tokenization methods like BPE and SentencePiece are more versatile across different languages.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

english_text = "Hello, world!"
chinese_text = "你好，世界！"

print(tokenizer.tokenize(english_text))
print(tokenizer.tokenize(chinese_text))
```

Slide 8: 

Token-to-ID Mapping

In LLMs, tokens are typically converted to integer IDs for efficient processing. This mapping is done using a vocabulary that associates each token with a unique ID. Understanding this mapping is crucial for working with LLMs and interpreting their inputs and outputs.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "Tokens and IDs"

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

for token, id in zip(tokens, token_ids):
    print(f"Token: {token}, ID: {id}")
```

Slide 9: 

Handling Unknown Tokens

When tokenizing text, we may encounter words or characters that are not in the model's vocabulary. These are typically represented by a special "unknown" token (often denoted as "\[UNK\]" or "<unk>"). Properly handling unknown tokens is important for maintaining the integrity of the input and preventing information loss.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "I love to eat 寿司 (sushi)!"

tokens = tokenizer.tokenize(text)
print(tokens)

# Check for unknown tokens
unk_token = tokenizer.special_tokens_map['unk_token']
unk_count = tokens.count(unk_token)
print(f"Number of unknown tokens: {unk_count}")
```

Slide 10: 

Token Limits in LLMs

LLMs typically have a maximum number of tokens they can process in a single input. This limit affects how much text can be processed at once and is an important consideration when working with these models. Understanding and managing token limits is crucial for effective use of LLMs in various applications.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

max_length = model.config.max_position_embeddings
print(f"Maximum token limit: {max_length}")

text = "This is a long text " * 1000
tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
print(f"Number of tokens after truncation: {len(tokens)}")
```

Slide 11: 

Token-Aware Text Splitting

When dealing with long texts that exceed an LLM's token limit, it's important to split the text in a way that respects token boundaries. This ensures that we don't cut words or subwords in the middle, which could lead to incorrect processing or loss of meaning.

```python
def split_text(text, max_tokens, tokenizer):
    tokens = tokenizer.encode(text)
    chunks = []
    current_chunk = []

    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= max_tokens:
            chunks.append(tokenizer.decode(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(tokenizer.decode(current_chunk))

    return chunks

tokenizer = AutoTokenizer.from_pretrained("gpt2")
long_text = "This is a very long text " * 1000
chunks = split_text(long_text, max_tokens=1000, tokenizer=tokenizer)

print(f"Number of chunks: {len(chunks)}")
print(f"First chunk: {chunks[0][:50]}...")
```

Slide 12: 

Tokenization for Fine-tuning

When fine-tuning an LLM on a specific task or domain, it's sometimes beneficial to adapt the tokenizer to better represent the target vocabulary. This can involve adding new tokens or adjusting the existing vocabulary to improve the model's performance on the specific task.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Add new tokens
new_tokens = ["[DISEASE]", "[TREATMENT]", "[SYMPTOM]"]
num_added_tokens = tokenizer.add_tokens(new_tokens)

# Resize model embeddings to account for new tokens
model.resize_token_embeddings(len(tokenizer))

print(f"Added {num_added_tokens} new tokens")

# Example usage
text = "Patients with [DISEASE] often experience [SYMPTOM] and require [TREATMENT]."
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
print(decoded)
```

Slide 13: 

Analyzing Token Distribution

Understanding the distribution of tokens in your dataset can provide insights into the effectiveness of your tokenization strategy and help identify potential issues or biases. This analysis can be particularly useful when working with domain-specific texts or multilingual datasets.

```python
from collections import Counter
import matplotlib.pyplot as plt

def analyze_token_distribution(texts, tokenizer):
    all_tokens = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        all_tokens.extend(tokens)
    
    token_counts = Counter(all_tokens)
    top_tokens = token_counts.most_common(10)
    
    tokens, counts = zip(*top_tokens)
    plt.bar(tokens, counts)
    plt.title("Top 10 Most Common Tokens")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage
texts = ["This is a sample text.", "Another example sentence.", "More text for analysis."]
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
analyze_token_distribution(texts, tokenizer)
```

Slide 14: 

Tokenization in Real-world Applications

Tokenization plays a crucial role in various NLP applications. For instance, in sentiment analysis, proper tokenization can help capture nuanced expressions and emoticons. In machine translation, subword tokenization helps handle rare words and improve translation quality across languages with different morphological structures.

```python
from transformers import pipeline

# Sentiment analysis example
sentiment_analyzer = pipeline("sentiment-analysis")
text = "I absolutely loved 😍 the new movie! It was fantastic."
result = sentiment_analyzer(text)
print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.2f}")

# Machine translation example
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
english_text = "The quick brown fox jumps over the lazy dog."
translation = translator(english_text)
print(f"Translation: {translation[0]['translation_text']}")
```

Slide 15: 

Additional Resources

For those interested in diving deeper into tokenization and its applications in LLMs, here are some valuable resources:

1. "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016) - Introduces BPE for NMT: [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)
2. "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing" (Kudo and Richardson, 2018) - Describes the SentencePiece algorithm: [https://arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226)
3. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2019) - Discusses WordPiece tokenization in the context of BERT: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

These papers provide in-depth explanations of various tokenization techniques and their applications in modern NLP models.

