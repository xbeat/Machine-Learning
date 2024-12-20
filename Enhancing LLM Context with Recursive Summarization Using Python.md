## Enhancing LLM Context with Recursive Summarization Using Python
Slide 1: Introduction to LLM Context Enhancement

Large Language Models (LLMs) have limited context windows. Recursive summarization is a technique to extend this context by iteratively condensing information. This approach allows LLMs to process larger documents while retaining key information.

```python
import transformers

def load_llm():
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer

model, tokenizer = load_llm()
```

Slide 2: Understanding Context Windows

Context windows define the maximum amount of text an LLM can process at once. For example, GPT-3 has a context window of 4096 tokens. Recursive summarization helps overcome this limitation by condensing long texts into shorter, informative summaries.

```python
def get_context_window(model):
    return model.config.max_position_embeddings

context_window = get_context_window(model)
print(f"Model context window: {context_window} tokens")
```

Slide 3: Text Chunking

The first step in recursive summarization is dividing the input text into manageable chunks that fit within the LLM's context window. This ensures that each chunk can be processed independently.

```python
def chunk_text(text, max_chunk_size):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= max_chunk_size:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

Slide 4: Summarizing Individual Chunks

After chunking, each text segment is summarized independently. This reduces the content while preserving key information. The summarization process can be customized based on the specific requirements of your application.

```python
def summarize_chunk(chunk, model, tokenizer):
    inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
```

Slide 5: Recursive Summarization Process

Recursive summarization involves repeatedly summarizing the summaries until the entire text fits within the context window. This process creates a hierarchical structure of summaries, with each level containing more condensed information.

```python
def recursive_summarize(text, model, tokenizer, max_chunk_size, target_size):
    if len(text) <= target_size:
        return text

    chunks = chunk_text(text, max_chunk_size)
    summaries = [summarize_chunk(chunk, model, tokenizer) for chunk in chunks]
    combined_summary = " ".join(summaries)

    return recursive_summarize(combined_summary, model, tokenizer, max_chunk_size, target_size)
```

Slide 6: Handling Long Documents

For extremely long documents, the recursive summarization process may need to be applied multiple times. This ensures that the final summary fits within the LLM's context window while still capturing the essence of the entire document.

```python
def process_long_document(document, model, tokenizer, max_chunk_size, target_size):
    sections = document.split("\n\n")  # Assuming sections are separated by double newlines
    section_summaries = []

    for section in sections:
        summary = recursive_summarize(section, model, tokenizer, max_chunk_size, target_size // len(sections))
        section_summaries.append(summary)

    return " ".join(section_summaries)
```

Slide 7: Preserving Context Hierarchy

To maintain the document's structure, it's important to preserve the hierarchy of information during summarization. This can be achieved by summarizing at different levels (e.g., paragraphs, sections, chapters) and combining the results.

```python
def hierarchical_summarization(document, model, tokenizer, max_chunk_size, target_size):
    chapters = document.split("Chapter")
    chapter_summaries = []

    for chapter in chapters[1:]:  # Skip the first empty split
        sections = chapter.split("Section")
        section_summaries = []

        for section in sections[1:]:  # Skip the first empty split
            summary = recursive_summarize(section, model, tokenizer, max_chunk_size, target_size // (len(chapters) * len(sections)))
            section_summaries.append(summary)

        chapter_summary = " ".join(section_summaries)
        chapter_summaries.append(chapter_summary)

    return " ".join(chapter_summaries)
```

Slide 8: Balancing Compression and Information Retention

Finding the right balance between compression and information retention is crucial. Experiment with different summarization ratios and techniques to achieve optimal results for your specific use case.

```python
def adaptive_summarization(text, model, tokenizer, max_chunk_size, target_size, compression_ratio=0.5):
    if len(text) <= target_size:
        return text

    chunks = chunk_text(text, max_chunk_size)
    summaries = []

    for chunk in chunks:
        chunk_target_size = int(len(chunk) * compression_ratio)
        summary = summarize_chunk(chunk, model, tokenizer)
        
        if len(summary) > chunk_target_size:
            summary = summary[:chunk_target_size]
        
        summaries.append(summary)

    combined_summary = " ".join(summaries)

    if len(combined_summary) <= target_size:
        return combined_summary
    else:
        return adaptive_summarization(combined_summary, model, tokenizer, max_chunk_size, target_size, compression_ratio * 0.9)
```

Slide 9: Implementing Custom Tokenization

For more control over the summarization process, implement custom tokenization tailored to your specific domain or language. This can improve the quality of summaries for specialized texts.

```python
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_custom_tokenizer(texts):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer

custom_tokenizer = train_custom_tokenizer(your_text_corpus)
```

Slide 10: Enhancing Summaries with Key Information Extraction

Improve the quality of summaries by extracting and prioritizing key information such as named entities, dates, or domain-specific terms. This ensures that critical details are preserved in the final summary.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_key_info(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    key_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    return list(set(entities + key_phrases))

def enhanced_summarization(chunk, model, tokenizer, key_info):
    summary = summarize_chunk(chunk, model, tokenizer)
    key_info_text = ", ".join(key_info)
    enhanced_summary = f"{summary}\n\nKey information: {key_info_text}"
    return enhanced_summary
```

Slide 11: Handling Multi-modal Input

Extend the recursive summarization technique to handle multi-modal input, such as text with images or tables. This requires adapting the summarization process to incorporate information from different modalities.

```python
from PIL import Image
import pytesseract

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def summarize_multimodal_content(text, image_paths, model, tokenizer, max_chunk_size, target_size):
    image_texts = [extract_text_from_image(img_path) for img_path in image_paths]
    combined_text = text + "\n" + "\n".join(image_texts)
    return recursive_summarize(combined_text, model, tokenizer, max_chunk_size, target_size)
```

Slide 12: Evaluating Summary Quality

Assess the quality of generated summaries using metrics like ROUGE scores or semantic similarity. This helps in fine-tuning the summarization process and ensuring that the recursive approach maintains content accuracy.

```python
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util

def evaluate_summary(original_text, summary):
    rouge = Rouge()
    scores = rouge.get_scores(summary, original_text)
    
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    original_embedding = model.encode(original_text, convert_to_tensor=True)
    summary_embedding = model.encode(summary, convert_to_tensor=True)
    semantic_similarity = util.pytorch_cos_sim(original_embedding, summary_embedding).item()
    
    return {
        'rouge': scores[0],
        'semantic_similarity': semantic_similarity
    }
```

Slide 13: Optimizing for Real-time Applications

For real-time applications, optimize the recursive summarization process to reduce latency. Implement caching mechanisms and parallel processing to improve performance when dealing with large volumes of text.

```python
import concurrent.futures
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_summarize_chunk(chunk, model_name, tokenizer_name):
    model, tokenizer = load_llm(model_name, tokenizer_name)
    return summarize_chunk(chunk, model, tokenizer)

def parallel_summarize(chunks, model_name, tokenizer_name, max_workers=4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(cached_summarize_chunk, chunk, model_name, tokenizer_name) for chunk in chunks]
        summaries = [future.result() for future in concurrent.futures.as_completed(futures)]
    return summaries
```

Slide 14: Integrating with Document Retrieval Systems

Combine recursive summarization with document retrieval systems to enhance search capabilities. Use the generated summaries to create more informative search indices and improve query matching.

```python
from elasticsearch import Elasticsearch

def index_document_with_summary(es, doc_id, original_text, summary):
    es.index(index="documents", id=doc_id, body={
        "original_text": original_text,
        "summary": summary
    })

def search_documents(es, query, size=10):
    result = es.search(index="documents", body={
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["original_text", "summary^2"]
            }
        },
        "size": size
    })
    return result['hits']['hits']

es = Elasticsearch()
index_document_with_summary(es, "doc1", original_text, summary)
search_results = search_documents(es, "your search query")
```

Slide 15: Additional Resources

1. "Recursive Summarization for Long Document Understanding" by Balachandran et al. (2023) arXiv:2301.13703 \[cs.CL\] [https://arxiv.org/abs/2301.13703](https://arxiv.org/abs/2301.13703)
2. "Longformer: The Long-Document Transformer" by Beltagy et al. (2020) arXiv:2004.05150 \[cs.CL\] [https://arxiv.org/abs/2004.05150](https://arxiv.org/abs/2004.05150)
3. "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension" by Lewis et al. (2019) arXiv:1910.13461 \[cs.CL\] [https://arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461)

