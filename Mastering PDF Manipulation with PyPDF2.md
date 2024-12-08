## Mastering PDF Manipulation with PyPDF2
Slide 1: Basic PDF File Manipulation with PyPDF2

PyPDF2 provides fundamental capabilities for PDF manipulation in Python, allowing developers to perform operations like merging, splitting, and basic information extraction. This library serves as the foundation for more complex PDF processing tasks.

```python
from PyPDF2 import PdfReader, PdfWriter

# Open and read PDF file
reader = PdfReader("input.pdf")
writer = PdfWriter()

# Extract basic information
print(f"Number of pages: {len(reader.pages)}")
print(f"Metadata: {reader.metadata}")

# Extract first page and save to new PDF
writer.add_page(reader.pages[0])
with open("output.pdf", "wb") as output_file:
    writer.write(output_file)
```

Slide 2: PDF Merging and Page Ordering

PDF merging is a common requirement in document processing workflows. PyPDF2 enables precise control over page selection and ordering when combining multiple PDF files into a single document.

```python
from PyPDF2 import PdfMerger

def merge_pdfs(pdf_list, output_filename):
    merger = PdfMerger()
    
    # Add each PDF file to the merger
    for pdf in pdf_list:
        merger.append(pdf)
    
    # Write the merged PDF to file
    with open(output_filename, "wb") as output_file:
        merger.write(output_file)
    
    merger.close()

# Example usage
pdf_files = ["file1.pdf", "file2.pdf", "file3.pdf"]
merge_pdfs(pdf_files, "merged_output.pdf")
```

Slide 3: Text Extraction and Processing

Text extraction from PDF documents requires careful handling of document structure and formatting. PyPDF2 provides methods to extract text while maintaining relative positioning and formatting information.

```python
from PyPDF2 import PdfReader

def extract_text_with_formatting(pdf_path):
    reader = PdfReader(pdf_path)
    text_content = []
    
    for page in reader.pages:
        # Extract text while preserving formatting
        text = page.extract_text()
        
        # Process and clean extracted text
        cleaned_text = " ".join(text.split())
        text_content.append(cleaned_text)
    
    return "\n\n".join(text_content)

# Example usage
text = extract_text_with_formatting("document.pdf")
print(f"Extracted Text:\n{text[:500]}...")  # Show first 500 characters
```

Slide 4: PDF Encryption and Security

PDF security is crucial for sensitive documents. PyPDF2 provides comprehensive encryption capabilities, allowing developers to implement password protection and permission controls for PDF files.

```python
from PyPDF2 import PdfReader, PdfWriter

def encrypt_pdf(input_path, output_path, user_pwd, owner_pwd):
    reader = PdfReader(input_path)
    writer = PdfWriter()

    # Copy all pages to the writer
    for page in reader.pages:
        writer.add_page(page)

    # Set up encryption with permissions
    writer.encrypt(user_pwd, owner_pwd,
                  use_128bit=True,
                  permissions_flag=permissions.PRINT | 
                                 permissions.MODIFY | 
                                 permissions.COPY)

    # Save the encrypted PDF
    with open(output_path, "wb") as output_file:
        writer.write(output_file)

# Example usage
encrypt_pdf("input.pdf", "encrypted.pdf", "user123", "owner456")
```

Slide 5: PDF Metadata Manipulation

PDF metadata contains essential document information like title, author, and creation date. PyPDF2 provides comprehensive methods to read, modify, and update these metadata fields programmatically.

```python
from PyPDF2 import PdfReader, PdfWriter
from datetime import datetime

def update_pdf_metadata(input_path, output_path, metadata_dict):
    reader = PdfReader(input_path)
    writer = PdfWriter()
    
    # Copy pages
    for page in reader.pages:
        writer.add_page(page)
    
    # Update metadata
    writer.add_metadata(metadata_dict)
    
    # Save updated PDF
    with open(output_path, "wb") as output_file:
        writer.write(output_file)

# Example usage
metadata = {
    "/Title": "Updated Document",
    "/Author": "John Doe",
    "/Subject": "PDF Processing",
    "/Producer": "Custom PDF Tool",
    "/CreationDate": datetime.now().strftime("D:%Y%m%d%H%M%S")
}
update_pdf_metadata("input.pdf", "updated_metadata.pdf", metadata)
```

Slide 6: Advanced Page Manipulation

The ability to manipulate individual pages within PDF documents is crucial for document processing workflows. This implementation demonstrates rotation, scaling, and page composition techniques.

```python
from PyPDF2 import PdfReader, PdfWriter
import math

def manipulate_pages(input_path, output_path):
    reader = PdfReader(input_path)
    writer = PdfWriter()
    
    for page in reader.pages:
        # Rotate page by 90 degrees
        page.rotate(90)
        
        # Scale page content
        page.scale(sx=1.5, sy=1.5)
        
        # Translate content position
        page.scale_by(1.0)
        page.transfer_rotation_to_content()
        
        writer.add_page(page)
    
    with open(output_path, "wb") as output_file:
        writer.write(output_file)

# Example usage
manipulate_pages("input.pdf", "manipulated.pdf")
```

Slide 7: PDF Form Field Extraction

PDF forms often contain interactive fields for data input. This implementation demonstrates how to extract and process form field data using PyPDF2's form field extraction capabilities.

```python
from PyPDF2 import PdfReader

def extract_form_fields(pdf_path):
    reader = PdfReader(pdf_path)
    form_fields = {}
    
    if reader.is_encrypted:
        reader.decrypt("")  # Handle encrypted PDFs
    
    for page in reader.pages:
        if '/Annots' in page:
            for annotation in page['/Annots']:
                if annotation.get('/FT') == '/Tx':  # Text field
                    field_name = annotation.get('/T')
                    field_value = annotation.get('/V')
                    form_fields[field_name] = field_value
    
    return form_fields

# Example usage
fields = extract_form_fields("form.pdf")
print("Extracted form fields:", fields)
```

Slide 8: PDF Watermarking Implementation

Watermarking is essential for document protection and branding. This implementation shows how to add both text and image watermarks to PDF documents programmatically.

```python
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

def add_watermark(input_path, output_path, watermark_text):
    # Create watermark
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)
    c.setFont("Helvetica", 50)
    c.setFillColorRGB(0.5, 0.5, 0.5, 0.3)  # Gray, 30% opacity
    c.saveState()
    c.translate(300, 400)
    c.rotate(45)
    c.drawString(0, 0, watermark_text)
    c.restoreState()
    c.save()
    
    # Apply watermark to PDF
    packet.seek(0)
    watermark = PdfReader(packet)
    reader = PdfReader(input_path)
    writer = PdfWriter()
    
    for page in reader.pages:
        page.merge_page(watermark.pages[0])
        writer.add_page(page)
    
    with open(output_path, "wb") as output_file:
        writer.write(output_file)

# Example usage
add_watermark("input.pdf", "watermarked.pdf", "CONFIDENTIAL")
```

Slide 9: PDF Image Extraction

PDF documents often contain embedded images that need to be extracted for various purposes. This implementation demonstrates robust image extraction while preserving image quality and metadata.

```python
from PyPDF2 import PdfReader
import io
from PIL import Image

def extract_images(pdf_path, output_dir):
    reader = PdfReader(pdf_path)
    image_count = 0
    
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        
        if '/Resources' in page and '/XObject' in page['/Resources']:
            xObject = page['/Resources']['/XObject']
            
            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':
                    image_count += 1
                    
                    if '/Filter' in xObject[obj]:
                        if xObject[obj]['/Filter'] == '/DCTDecode':
                            # JPEG image
                            img_data = xObject[obj]._data
                            img = Image.open(io.BytesIO(img_data))
                            img.save(f"{output_dir}/image_{image_count}.jpg")
                        elif xObject[obj]['/Filter'] == '/FlateDecode':
                            # PNG image
                            width = xObject[obj]['/Width']
                            height = xObject[obj]['/Height']
                            data = xObject[obj]._data
                            img = Image.frombytes('RGB', (width, height), data)
                            img.save(f"{output_dir}/image_{image_count}.png")
    
    return image_count

# Example usage
num_images = extract_images("document_with_images.pdf", "./extracted_images")
print(f"Extracted {num_images} images from PDF")
```

Slide 10: PDF Table Extraction and Processing

Extracting tabular data from PDFs requires sophisticated parsing techniques. This implementation provides a robust approach to identify and extract tables while maintaining their structure.

```python
import tabula
import pandas as pd
from PyPDF2 import PdfReader

def extract_tables(pdf_path):
    # Extract tables using tabula
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
    
    # Process and clean extracted tables
    processed_tables = []
    for idx, table in enumerate(tables):
        # Remove empty rows and columns
        cleaned_table = table.dropna(how='all').dropna(axis=1, how='all')
        
        # Convert to proper data types
        for column in cleaned_table.columns:
            try:
                cleaned_table[column] = pd.to_numeric(cleaned_table[column])
            except:
                pass  # Keep as string if conversion fails
        
        processed_tables.append(cleaned_table)
    
    return processed_tables

# Example usage
tables = extract_tables("document_with_tables.pdf")
for idx, table in enumerate(tables):
    print(f"\nTable {idx + 1}:")
    print(table.head())
```

Slide 11: PDF Page Layout Analysis

Understanding the layout of PDF pages is crucial for accurate content extraction. This implementation provides methods to analyze page structure and identify content regions.

```python
from PyPDF2 import PdfReader
import numpy as np

def analyze_page_layout(pdf_path, page_num=0):
    reader = PdfReader(pdf_path)
    page = reader.pages[page_num]
    
    # Extract page dimensions
    media_box = page.mediabox
    width = float(media_box.width)
    height = float(media_box.height)
    
    # Analyze content distribution
    def get_content_regions(page):
        text = page.extract_text()
        lines = text.split('\n')
        regions = []
        
        current_y = height
        for line in lines:
            if line.strip():
                # Estimate line position
                line_height = 12  # Approximate font size
                regions.append({
                    'type': 'text',
                    'content': line,
                    'bbox': (0, current_y - line_height, width, current_y)
                })
                current_y -= line_height
        
        return regions
    
    layout_info = {
        'dimensions': {'width': width, 'height': height},
        'regions': get_content_regions(page),
        'orientation': page.get('/Rotate', 0)
    }
    
    return layout_info

# Example usage
layout = analyze_page_layout("document.pdf")
print(f"Page dimensions: {layout['dimensions']}")
print(f"Number of content regions: {len(layout['regions'])}")
print(f"Page orientation: {layout['orientation']} degrees")
```

Slide 12: PDF Form Creation and Population

Creating dynamic PDF forms programmatically enables automated document generation. This implementation demonstrates how to create interactive forms and populate them with data.

```python
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

def create_fillable_form(output_path, form_data):
    # Create form template
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)
    
    # Add form fields
    for field_name, properties in form_data.items():
        c.drawString(properties['x'], properties['y'], f"{field_name}:")
        c.acroForm.textfield(
            name=field_name,
            x=properties['x'] + 100,
            y=properties['y'],
            width=200,
            height=20
        )
    
    c.save()
    packet.seek(0)
    
    # Create PDF with form fields
    writer = PdfWriter()
    writer.add_page(PdfReader(packet).pages[0])
    
    with open(output_path, "wb") as output_file:
        writer.write(output_file)

# Example usage
form_fields = {
    'Name': {'x': 50, 'y': 700},
    'Email': {'x': 50, 'y': 650},
    'Phone': {'x': 50, 'y': 600}
}
create_fillable_form("fillable_form.pdf", form_fields)
```

Slide 13: PDF Digital Signatures and Verification

Digital signatures provide document authenticity and integrity. This implementation shows how to digitally sign PDFs and verify existing signatures.

```python
from PyPDF2 import PdfReader, PdfWriter
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
import datetime

def sign_pdf(input_path, output_path, certificate_path, private_key_path):
    reader = PdfReader(input_path)
    writer = PdfWriter()
    
    # Load certificate and private key
    with open(certificate_path, 'rb') as cert_file:
        certificate = serialization.load_pem_x509_certificate(
            cert_file.read()
        )
    
    with open(private_key_path, 'rb') as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None
        )
    
    # Add pages to writer
    for page in reader.pages:
        writer.add_page(page)
    
    # Create signature dictionary
    signature = {
        '/Type': '/Sig',
        '/Filter': '/Adobe.PPKLite',
        '/SubFilter': '/adbe.pkcs7.detached',
        '/Name': 'Digital Signature',
        '/SigningTime': datetime.datetime.utcnow(),
        '/Location': 'PDF Signature',
        '/Reason': 'Document Authentication'
    }
    
    # Add signature to PDF
    writer.add_unsaved_signature(signature)
    
    with open(output_path, "wb") as output_file:
        writer.write(output_file)

# Example usage
sign_pdf("input.pdf", "signed.pdf", "certificate.pem", "private_key.pem")
```

Slide 14: Additional Resources

*   A Comprehensive Survey of PDF Document Processing on ArXiv:
    *   [https://arxiv.org/abs/2010.12628](https://arxiv.org/abs/2010.12628)
*   Deep Learning Approaches for PDF Document Understanding:
    *   [https://arxiv.org/abs/2005.14279](https://arxiv.org/abs/2005.14279)
*   PDF Form Processing and Information Extraction:
    *   [https://arxiv.org/abs/1907.05333](https://arxiv.org/abs/1907.05333)
*   General Resources for PDF Processing:
    *   PyPDF2 Documentation: [https://pypdf2.readthedocs.io/](https://pypdf2.readthedocs.io/)
    *   PDF Technical Documentation: [https://www.adobe.com/devnet/pdf/pdf\_reference.html](https://www.adobe.com/devnet/pdf/pdf_reference.html)
    *   StackOverflow PDF Processing Tag: [https://stackoverflow.com/questions/tagged/pdf-parsing](https://stackoverflow.com/questions/tagged/pdf-parsing)

