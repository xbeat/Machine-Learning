## Building a Content Summarizer with OpenAI LangChain and Gradio
Slide 1: Environment Setup and Dependencies

The foundation of our YouTube and website summarizer requires specific Python packages. We'll use OpenAI for the language model, LangChain for chain operations, and Gradio for the interface. This setup ensures all required dependencies are properly installed and configured.

```python
# Install required packages
!pip install openai langchain gradio yt-dlp python-dotenv unstructured

# Import necessary libraries
import os
import openai
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import UnstructuredURLLoader
from dotenv import load_dotenv
import gradio as gr

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')
```

Slide 2: YouTube Content Extraction

This module handles the extraction of video metadata using yt-dlp. The implementation focuses on retrieving the video title, description, and other relevant metadata that will be used for summarization.

```python
import yt_dlp

def extract_video_info(url):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            video_info = ydl.extract_info(url, download=False)
            return {
                'title': video_info.get('title', ''),
                'description': video_info.get('description', ''),
                'duration': video_info.get('duration', 0)
            }
        except Exception as e:
            return {'error': str(e)}

# Example usage
video_url = "https://www.youtube.com/watch?v=example"
video_data = extract_video_info(video_url)
```

Slide 3: LangChain Summarization Setup

The summarization chain configuration is crucial for generating high-quality summaries. We implement a custom chain using LangChain's architecture to process both YouTube and website content effectively.

```python
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_summary_chain():
    llm = OpenAI(temperature=0.5)
    
    prompt_template = """
    Create a comprehensive summary in approximately 300 words:
    {text}
    SUMMARY:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    
    chain = load_summarize_chain(
        llm=llm,
        chain_type="stuff",
        prompt=PROMPT
    )
    
    return chain

# Initialize text splitter for long content
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)
```

Slide 4: Website Content Processing

The website content processor utilizes UnstructuredURLLoader to extract readable text from web pages. This implementation handles various HTML structures and returns clean, processable text content.

```python
from langchain.document_loaders import UnstructuredURLLoader
from typing import List, Dict

def process_website_content(url: str) -> List[Dict]:
    try:
        loader = UnstructuredURLLoader(urls=[url])
        documents = loader.load()
        
        # Process and clean the content
        processed_docs = []
        for doc in documents:
            cleaned_text = doc.page_content.replace('\n', ' ').strip()
            processed_docs.append({
                'content': cleaned_text,
                'metadata': doc.metadata
            })
        
        return processed_docs
    except Exception as e:
        return [{'error': f"Failed to process URL: {str(e)}"}]

# Example usage
website_url = "https://example.com"
website_content = process_website_content(website_url)
```

Slide 5: Content Summarization Engine

The summarization engine combines extracted content processing with LangChain's summarization capabilities. This implementation handles both YouTube and website content through a unified interface.

```python
from langchain.docstore.document import Document

def summarize_content(content: str, content_type: str) -> str:
    # Initialize summarization chain
    chain = create_summary_chain()
    
    # Prepare documents
    docs = [Document(page_content=content)]
    
    # Handle large content
    if len(content) > 2000:
        splits = text_splitter.split_documents(docs)
        summarized = chain.run(splits)
    else:
        summarized = chain.run(docs)
    
    return summarized

# Example implementation
def process_and_summarize(url: str, content_type: str = 'website') -> str:
    if content_type == 'youtube':
        content = extract_video_info(url)['description']
    else:
        content = process_website_content(url)[0]['content']
    
    return summarize_content(content, content_type)
```

Slide 6: Gradio Interface Implementation

The Gradio interface provides an intuitive way for users to interact with our summarizer. This implementation creates a clean, responsive interface with proper input validation and error handling.

```python
import gradio as gr

def create_interface():
    def process_url(url, content_type):
        try:
            summary = process_and_summarize(url, content_type)
            return summary
        except Exception as e:
            return f"Error processing content: {str(e)}"

    interface = gr.Interface(
        fn=process_url,
        inputs=[
            gr.Textbox(label="Enter URL", placeholder="https://..."),
            gr.Radio(["youtube", "website"], label="Content Type", value="website")
        ],
        outputs=gr.Textbox(label="Summary"),
        title="Content Summarizer",
        description="Enter a YouTube video URL or website URL to get a summary",
        examples=[
            ["https://www.youtube.com/watch?v=example", "youtube"],
            ["https://example.com/article", "website"]
        ]
    )
    return interface

# Launch the interface
if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)
```

Slide 7: Content Preprocessing Pipeline

A robust preprocessing pipeline ensures clean, standardized input for our summarization engine. This implementation handles text normalization, special character removal, and content structuring.

```python
import re
from typing import Dict, Any

class ContentPreprocessor:
    def __init__(self):
        self.cleaners = {
            'remove_special_chars': lambda x: re.sub(r'[^\w\s.,!?-]', '', x),
            'normalize_whitespace': lambda x: re.sub(r'\s+', ' ', x),
            'remove_urls': lambda x: re.sub(r'http\S+', '', x)
        }
    
    def preprocess(self, content: str) -> str:
        processed = content
        for cleaner in self.cleaners.values():
            processed = cleaner(processed)
        return processed.strip()
    
    def process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: self.preprocess(str(value))
            for key, value in metadata.items()
        }

# Example usage
preprocessor = ContentPreprocessor()
sample_text = """Check out this link: https://example.com
                This text has   multiple    spaces and special $#@ characters!"""
cleaned_text = preprocessor.preprocess(sample_text)
print(f"Cleaned text: {cleaned_text}")
```

Slide 8: Error Handling and Validation

Comprehensive error handling and input validation ensure robust operation of the summarizer. This implementation includes URL validation, content type checking, and appropriate error responses.

```python
from urllib.parse import urlparse
from typing import Tuple, Optional

class ValidationHandler:
    @staticmethod
    def validate_url(url: str) -> Tuple[bool, Optional[str]]:
        try:
            result = urlparse(url)
            is_valid = all([result.scheme, result.netloc])
            error_message = None if is_valid else "Invalid URL format"
            return is_valid, error_message
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def validate_content_type(content_type: str) -> Tuple[bool, Optional[str]]:
        valid_types = ['youtube', 'website']
        is_valid = content_type.lower() in valid_types
        error_message = None if is_valid else f"Content type must be one of: {valid_types}"
        return is_valid, error_message

def safe_process_url(url: str, content_type: str) -> Dict[str, Any]:
    validator = ValidationHandler()
    
    # Validate inputs
    url_valid, url_error = validator.validate_url(url)
    type_valid, type_error = validator.validate_content_type(content_type)
    
    if not url_valid:
        return {'error': url_error}
    if not type_valid:
        return {'error': type_error}
        
    try:
        if content_type == 'youtube':
            return extract_video_info(url)
        else:
            return {'content': process_website_content(url)}
    except Exception as e:
        return {'error': f"Processing error: {str(e)}"}
```

Slide 9: Rate Limiting and Caching

To optimize performance and manage API usage, we implement rate limiting and caching mechanisms. This ensures efficient resource utilization and improved response times.

```python
from functools import lru_cache
from time import time
from collections import deque

class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        
    def can_proceed(self) -> bool:
        current_time = time()
        
        # Remove expired timestamps
        while self.requests and current_time - self.requests[0] >= self.time_window:
            self.requests.popleft()
            
        if len(self.requests) < self.max_requests:
            self.requests.append(current_time)
            return True
        return False

# Caching implementation
@lru_cache(maxsize=100)
def cached_summarize(content: str, content_type: str) -> str:
    return summarize_content(content, content_type)

# Rate limiter implementation
rate_limiter = RateLimiter(max_requests=60, time_window=60)  # 60 requests per minute

def rate_limited_process(url: str, content_type: str) -> Dict[str, Any]:
    if not rate_limiter.can_proceed():
        return {'error': 'Rate limit exceeded. Please try again later.'}
    return safe_process_url(url, content_type)
```

Slide 10: Performance Monitoring System

The performance monitoring system tracks key metrics including processing time, API usage, and success rates. This implementation helps identify bottlenecks and optimize system performance.

```python
import time
from dataclasses import dataclass
from typing import Dict, List
import statistics

@dataclass
class PerformanceMetrics:
    processing_time: float
    api_calls: int
    success: bool
    error_message: str = None

class PerformanceMonitor:
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        
    def measure_performance(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            api_calls = 0
            
            try:
                result = func(*args, **kwargs)
                success = True
                error_message = None
            except Exception as e:
                success = False
                error_message = str(e)
                result = None
            
            processing_time = time.time() - start_time
            
            self.metrics.append(PerformanceMetrics(
                processing_time=processing_time,
                api_calls=api_calls,
                success=success,
                error_message=error_message
            ))
            
            return result
        return wrapper
    
    def get_statistics(self) -> Dict:
        if not self.metrics:
            return {"error": "No metrics available"}
            
        processing_times = [m.processing_time for m in self.metrics]
        success_rate = sum(m.success for m in self.metrics) / len(self.metrics)
        
        return {
            "avg_processing_time": statistics.mean(processing_times),
            "success_rate": success_rate,
            "total_requests": len(self.metrics),
            "total_api_calls": sum(m.api_calls for m in self.metrics)
        }

# Example usage
monitor = PerformanceMonitor()

@monitor.measure_performance
def monitored_summarize(url: str, content_type: str) -> str:
    return process_and_summarize(url, content_type)
```

Slide 11: Custom Exception Handling

A specialized exception handling system manages various error scenarios specific to content summarization. This implementation provides detailed error information and appropriate recovery strategies.

```python
class SummarizerException(Exception):
    """Base exception class for summarizer errors"""
    pass

class ContentExtractionError(SummarizerException):
    """Raised when content extraction fails"""
    pass

class SummarizationError(SummarizerException):
    """Raised when summarization process fails"""
    pass

class ContentTooLongError(SummarizerException):
    """Raised when content exceeds maximum length"""
    pass

def handle_summarizer_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ContentExtractionError as e:
            return {
                'error': f"Content extraction failed: {str(e)}",
                'error_type': 'extraction',
                'recoverable': True
            }
        except SummarizationError as e:
            return {
                'error': f"Summarization failed: {str(e)}",
                'error_type': 'summarization',
                'recoverable': True
            }
        except ContentTooLongError as e:
            return {
                'error': f"Content too long: {str(e)}",
                'error_type': 'content_length',
                'recoverable': False
            }
        except Exception as e:
            return {
                'error': f"Unexpected error: {str(e)}",
                'error_type': 'unknown',
                'recoverable': False
            }
    return wrapper

@handle_summarizer_errors
def safe_summarize(content: str, max_length: int = 10000) -> str:
    if len(content) > max_length:
        raise ContentTooLongError(f"Content length {len(content)} exceeds maximum {max_length}")
    
    try:
        return summarize_content(content, 'text')
    except Exception as e:
        raise SummarizationError(str(e))
```

Slide 12: Text Quality Assessment

The text quality assessment module evaluates the quality of generated summaries using various metrics. This implementation helps ensure consistent and high-quality output.

```python
from typing import Dict, List
import re
from collections import Counter

class QualityAssessor:
    def __init__(self):
        self.metrics = {
            'coherence': self._assess_coherence,
            'completeness': self._assess_completeness,
            'redundancy': self._assess_redundancy,
            'readability': self._assess_readability
        }
    
    def _assess_coherence(self, text: str) -> float:
        sentences = text.split('.')
        if len(sentences) <= 1:
            return 0.0
            
        # Simple coherence score based on sentence transitions
        transition_words = {'however', 'therefore', 'consequently', 'moreover'}
        transitions = sum(1 for s in sentences if any(w in s.lower() for w in transition_words))
        return transitions / (len(sentences) - 1)
    
    def _assess_completeness(self, text: str) -> float:
        # Check for key elements (subject, verb, object)
        sentences = text.split('.')
        complete_sentences = sum(1 for s in sentences if self._is_complete_sentence(s))
        return complete_sentences / len(sentences) if sentences else 0.0
    
    def _assess_redundancy(self, text: str) -> float:
        words = text.lower().split()
        word_freq = Counter(words)
        return 1 - (len(word_freq) / len(words)) if words else 0.0
    
    def _assess_readability(self, text: str) -> float:
        words = text.split()
        sentences = text.split('.')
        if not words or not sentences:
            return 0.0
        avg_word_length = sum(len(w) for w in words) / len(words)
        return 1.0 / (avg_word_length * (len(words) / len(sentences)))
    
    def _is_complete_sentence(self, sentence: str) -> bool:
        return bool(re.search(r'\b[A-Z][^.!?]*[.!?]', sentence.strip()))
    
    def assess_quality(self, text: str) -> Dict[str, float]:
        return {
            metric: func(text)
            for metric, func in self.metrics.items()
        }

# Example usage
assessor = QualityAssessor()
sample_summary = "The project implements an AI-powered summarizer. It uses advanced NLP techniques. The system processes both videos and articles effectively."
quality_scores = assessor.assess_quality(sample_summary)
print("Quality Assessment Results:", quality_scores)
```

Slide 13: Results Analysis and Visualization

The results analysis module provides comprehensive visualization and analysis of summarization results. This implementation helps users understand the quality and characteristics of generated summaries.

```python
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple

class ResultsAnalyzer:
    def __init__(self):
        self.quality_assessor = QualityAssessor()
        
    def analyze_summary(self, original: str, summary: str) -> Dict[str, Any]:
        # Calculate compression ratio
        compression_ratio = len(summary) / len(original)
        
        # Get quality metrics
        quality_metrics = self.quality_assessor.assess_quality(summary)
        
        # Calculate key phrases retention
        original_keywords = self._extract_key_phrases(original)
        summary_keywords = self._extract_key_phrases(summary)
        retention_rate = len(set(summary_keywords) & set(original_keywords)) / len(original_keywords)
        
        return {
            'compression_ratio': compression_ratio,
            'quality_metrics': quality_metrics,
            'retention_rate': retention_rate
        }
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        # Simple key phrase extraction (could be enhanced with NLP)
        sentences = text.lower().split('.')
        words = [s.split() for s in sentences]
        return [' '.join(w) for w in words if len(w) > 2]
    
    def visualize_results(self, results: Dict[str, Any]) -> None:
        plt.figure(figsize=(12, 6))
        
        # Plot metrics
        metrics = list(results['quality_metrics'].items())
        x = np.arange(len(metrics))
        values = [v for _, v in metrics]
        labels = [k for k, _ in metrics]
        
        plt.bar(x, values, align='center', alpha=0.8)
        plt.xticks(x, labels, rotation=45)
        plt.ylabel('Score')
        plt.title('Summary Quality Metrics')
        
        # Add compression ratio and retention rate
        plt.axhline(y=results['compression_ratio'], color='r', linestyle='--', 
                   label=f'Compression Ratio: {results["compression_ratio"]:.2f}')
        plt.axhline(y=results['retention_rate'], color='g', linestyle='--',
                   label=f'Retention Rate: {results["retention_rate"]:.2f}')
        
        plt.legend()
        plt.tight_layout()
        plt.show()

# Example usage
analyzer = ResultsAnalyzer()
original_text = """Long original text about artificial intelligence and its 
                  applications in modern technology..."""
summary_text = "AI has revolutionized modern technology applications."
results = analyzer.analyze_summary(original_text, summary_text)
analyzer.visualize_results(results)
```

Slide 14: Additional Resources

*   Research paper on Neural Text Summarization Techniques:
    *   [https://arxiv.org/abs/2111.09764](https://arxiv.org/abs/2111.09764)
*   Comprehensive Survey of Text Summarization Methods:
    *   [https://arxiv.org/abs/2008.11293](https://arxiv.org/abs/2008.11293)
*   Deep Learning for Automatic Text Summarization:
    *   [https://arxiv.org/abs/1912.08777](https://arxiv.org/abs/1912.08777)
*   Youtube Content Analysis and Summarization:
    *   [https://www.google.com/search?q=youtube+content+analysis+research+papers](https://www.google.com/search?q=youtube+content+analysis+research+papers)
*   Advances in Extractive and Abstractive Summarization:
    *   [https://www.google.com/search?q=extractive+abstractive+summarization+papers](https://www.google.com/search?q=extractive+abstractive+summarization+papers)

Note: For the most current research, please search on Google Scholar or arXiv using keywords related to text summarization, content analysis, and natural language processing.

