## Summarizing Long Documents with Python and LLMs
Slide 1: Introduction to Document Summarization with LLMs and LangChain

Document summarization is a crucial task in natural language processing that involves condensing large texts into concise, informative summaries. With the advent of Large Language Models (LLMs) and frameworks like LangChain, this process has become more efficient and accurate. In this presentation, we'll explore how to leverage these technologies using Python to create powerful summarization tools.

```python
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text:\n\n{text}\n\nSummary:"
)
chain = LLMChain(llm=llm, prompt=prompt)

text = "Your long document or text here..."
summary = chain.run(text)
print(summary)
```

Slide 2: Understanding LLMs (Large Language Models)

Large Language Models are advanced AI systems trained on vast amounts of text data. They can understand and generate human-like text, making them ideal for tasks like summarization. LLMs can comprehend context, identify key information, and generate coherent summaries. Popular LLMs include GPT-3, BERT, and T5.

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
Long article or document text goes here...
"""

summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
print(summary[0]['summary_text'])
```

Slide 3: Introduction to LangChain

LangChain is a framework designed to simplify the development of applications using LLMs. It provides a set of tools and components that make it easier to chain together multiple LLM calls, integrate with external data sources, and create more complex AI-powered applications. LangChain is particularly useful for tasks like document summarization.

```python
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter

llm = OpenAI(temperature=0)
text_splitter = CharacterTextSplitter()

def summarize_text(text):
    chunks = text_splitter.split_text(text)
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize this text:\n\n{text}\n\nSummary:"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    summaries = [chain.run(chunk) for chunk in chunks]
    return " ".join(summaries)

long_text = "Your long document goes here..."
final_summary = summarize_text(long_text)
print(final_summary)
```

Slide 4: Text Preprocessing

Before summarizing a large document, it's crucial to preprocess the text. This step involves cleaning the text, removing unnecessary elements, and splitting it into manageable chunks. Proper preprocessing ensures better quality summaries and helps manage token limits of LLMs.

```python
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def preprocess_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

long_text = "Your long document goes here..."
preprocessed_chunks = preprocess_text(long_text)
print(f"Number of chunks: {len(preprocessed_chunks)}")
print(f"First chunk: {preprocessed_chunks[0][:100]}...")
```

Slide 5: Extractive Summarization

Extractive summarization involves selecting the most important sentences or phrases from the original text to form a summary. This method preserves the original wording and is useful when you want to maintain the exact language of the source document.

```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def extractive_summarize(text, sentences_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)

text = """
Long article or document text goes here...
"""

summary = extractive_summarize(text)
print(summary)
```

Slide 6: Abstractive Summarization with LLMs

Abstractive summarization generates new text that captures the essence of the original document. LLMs excel at this task as they can understand context and generate human-like text. This method often produces more coherent and concise summaries compared to extractive methods.

```python
from transformers import pipeline

def abstractive_summarize(text, max_length=150, min_length=50):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

text = """
Long article or document text goes here...
"""

summary = abstractive_summarize(text)
print(summary)
```

Slide 7: Chaining LLM Calls with LangChain

LangChain allows us to chain multiple LLM calls together, which is particularly useful for summarizing long documents. We can split the text into chunks, summarize each chunk, and then summarize the summaries to get a final, concise summary.

```python
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter

llm = OpenAI(temperature=0)
text_splitter = CharacterTextSplitter()

def summarize_long_text(text):
    chunks = text_splitter.split_text(text)
    
    chunk_summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize this text:\n\n{text}\n\nSummary:"
    )
    chunk_summarize_chain = LLMChain(llm=llm, prompt=chunk_summary_prompt)
    
    chunk_summaries = [chunk_summarize_chain.run(chunk) for chunk in chunks]
    
    final_summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="Combine these summaries into a coherent summary:\n\n{text}\n\nFinal Summary:"
    )
    final_summarize_chain = LLMChain(llm=llm, prompt=final_summary_prompt)
    
    final_summary = final_summarize_chain.run(" ".join(chunk_summaries))
    return final_summary

long_text = "Your very long document goes here..."
final_summary = summarize_long_text(long_text)
print(final_summary)
```

Slide 8: Handling Different Document Formats

In real-world scenarios, documents come in various formats like PDF, DOCX, or HTML. LangChain provides tools to handle these formats, allowing us to create a versatile summarization pipeline.

```python
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain

def load_document(file_path):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    return loader.load()

def summarize_document(file_path):
    docs = load_document(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    
    summary = chain.run(texts)
    return summary

file_path = "path/to/your/document.pdf"  # or .docx, or .html
summary = summarize_document(file_path)
print(summary)
```

Slide 9: Evaluating Summary Quality

Assessing the quality of generated summaries is crucial. We can use metrics like ROUGE (Recall-Oriented Understudy for Gisting Evaluation) to compare our summaries with human-generated ones or to evaluate different summarization methods.

```python
from rouge import Rouge

def evaluate_summary(reference, generated):
    rouge = Rouge()
    scores = rouge.get_scores(generated, reference)
    return scores[0]

reference_summary = "Human-written summary goes here..."
generated_summary = "AI-generated summary goes here..."

scores = evaluate_summary(reference_summary, generated_summary)
print(f"ROUGE-1: {scores['rouge-1']}")
print(f"ROUGE-2: {scores['rouge-2']}")
print(f"ROUGE-L: {scores['rouge-l']}")
```

Slide 10: Customizing Summaries with Prompts

LangChain allows us to customize the summarization process using prompts. We can guide the LLM to focus on specific aspects of the text or to generate summaries in a particular style or format.

```python
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)

custom_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Summarize the following text, focusing on the main arguments and key points. 
    Provide a brief introduction, 3-4 main points, and a conclusion.
    
    Text: {text}
    
    Summary:
    """
)

chain = LLMChain(llm=llm, prompt=custom_prompt)

text = "Your long document or text here..."
custom_summary = chain.run(text)
print(custom_summary)
```

Slide 11: Real-Life Example: Summarizing Research Papers

Researchers often need to quickly grasp the key points of numerous papers. Let's create a tool that summarizes academic papers, focusing on the abstract, methodology, and conclusions.

```python
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader

def summarize_paper(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    llm = OpenAI(temperature=0.5)
    
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Summarize this academic paper, focusing on:
        1. The main research question or hypothesis
        2. Key methodology used
        3. Main findings or conclusions
        
        Text: {text}
        
        Summary:
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    full_text = " ".join([page.page_content for page in pages])
    summary = chain.run(full_text)
    
    return summary

paper_path = "path/to/research_paper.pdf"
paper_summary = summarize_paper(paper_path)
print(paper_summary)
```

Slide 12: Real-Life Example: Summarizing News Articles

News aggregators often need to provide quick summaries of articles from various sources. Here's a tool that can summarize news articles, focusing on the key events, people involved, and implications.

```python
import requests
from bs4 import BeautifulSoup
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

def fetch_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    return " ".join([p.text for p in paragraphs])

def summarize_news_article(url):
    article_text = fetch_article(url)
    
    llm = OpenAI(temperature=0.5)
    
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Summarize this news article, focusing on:
        1. The main event or topic
        2. Key people or organizations involved
        3. Potential implications or consequences
        
        Article: {text}
        
        Summary:
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(article_text)
    
    return summary

news_url = "https://example.com/news-article"
article_summary = summarize_news_article(news_url)
print(article_summary)
```

Slide 13: Challenges and Limitations

While LLMs and LangChain offer powerful summarization capabilities, it's important to be aware of their limitations:

1. Token limits: LLMs have maximum token limits, which can be challenging for very long documents.
2. Accuracy: Summaries may occasionally contain inaccuracies or hallucinations, especially for complex topics.
3. Bias: LLMs can reflect biases present in their training data.
4. Context understanding: LLMs might miss nuanced context or domain-specific information.

To address these challenges, consider implementing fact-checking mechanisms, using domain-specific models when available, and always reviewing the generated summaries for accuracy and relevance.

```python
def check_summary_length(summary, max_tokens=500):
    tokens = summary.split()
    if len(tokens) > max_tokens:
        print(f"Warning: Summary exceeds {max_tokens} tokens. Consider further summarization.")
    return " ".join(tokens[:max_tokens])

def add_disclaimer(summary):
    disclaimer = "Note: This summary was generated by an AI model and may contain inaccuracies. Please verify important information."
    return f"{disclaimer}\n\n{summary}"

# Usage
summary = "Your generated summary here..."
checked_summary = check_summary_length(summary)
final_summary = add_disclaimer(checked_summary)
print(final_summary)
```

Slide 14: Future Directions and Improvements

The field of document summarization using LLMs and LangChain is rapidly evolving. Some promising future directions include:

1. Multi-modal summarization: Incorporating images and videos alongside text.
2. Personalized summaries: Tailoring summaries based on user preferences or expertise levels.
3. Improved fact-checking: Integrating external knowledge bases for verification.
4. Domain-specific models: Developing LLMs trained on specific domains for more accurate summaries.

Researchers and developers are continuously working on these areas to enhance the capabilities of summarization systems.

```python
def generate_personalized_summary(text, user_preference):
    llm = OpenAI(temperature=0.7)
    
    prompt = PromptTemplate(
        input_variables=["text", "preference"],
        template="""
        Summarize the following text, focusing on aspects related to {preference}:
        
        {text}
        
        Personalized Summary:
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    personalized_summary = chain.run(text=text, preference=user_preference)
    return personalized_summary

# Usage example
long_text = "Your long document here..."
user_preference = "technical details"
summary = generate_personalized_summary(long_text, user_preference)
print(summary)
```

Slide 15: Additional Resources

For those interested in diving deeper into document summarization using LLMs and LangChain, here are some valuable resources:

1. LangChain Documentation: [https://python.langchain.com/](https://python.langchain.com/)
2. "Attention Is All You Need" paper (introduces the Transformer architecture): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
3. "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension" paper: [https://arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461)
4. "Leveraging Large Language Models for Text Summarization" tutorial: [https://arxiv.org/abs/2303.08119](https://arxiv.org/abs/2303.08119)

These resources provide in-depth information on the underlying technologies and advanced techniques in the field of document summarization.

