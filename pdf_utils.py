import os
import re
import requests
import tempfile
import fitz  # PyMuPDF

def download_pdf(url):
    """Downloads a PDF from a URL and saves it to a temporary file."""
    response = requests.get(url)
    response.raise_for_status()  # Throw error if bad response
    
    temp_dir = tempfile.gettempdir()
    filename = os.path.join(temp_dir, "downloaded_paper.pdf")
    
    with open(filename, 'wb') as f:
        f.write(response.content)
    
    return filename

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def clean_section_content(content):
    """Cleans up common artifacts in section content"""
    # Remove page numbers 
    content = re.sub(r'\n\s*\d+\s*\n', '\n', content)
    
    # Remove header/footer artifacts
    content = re.sub(r'\n[^a-zA-Z0-9\s\.,;:\(\)\[\]\{\}\-_=\+\*\/\\]{2,}\n', '\n', content)
    
    # Join hyphenated words at end of lines
    content = re.sub(r'(\w+)-\n(\w+)', r'\1\2', content)
    
    # Remove excessive whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content

def extract_chunks(text, chunk_size=1500, overlap=300):
    """
    Creates overlapping chunks of the text without trying to identify sections.
    
    Args:
        text (str): The full text of the paper
        chunk_size (int): Size of each chunk in characters
        overlap (int): Overlap between chunks in characters
        
    Returns:
        list: A list of dictionaries with chunks and their positions
    """
    chunks = []
    start = 0
    
    while start < len(text):
        # Determine end position
        end = min(start + chunk_size, len(text))
        
        # If not at the end of text, try to find a good break point
        if end < len(text):
            # Look for paragraph breaks
            paragraph_break = text.rfind('\n\n', start + chunk_size - overlap, end)
            if paragraph_break > start:
                end = paragraph_break + 2  # Include the newlines
            else:
                # Look for sentence breaks
                sentence_break = max(
                    text.rfind('. ', start + chunk_size - overlap, end),
                    text.rfind('? ', start + chunk_size - overlap, end),
                    text.rfind('! ', start + chunk_size - overlap, end)
                )
                if sentence_break > start:
                    end = sentence_break + 2  # Include the period and space
        
        # Extract chunk
        chunk_text = text[start:end].strip()
        
        # Get the first line as a "title" for reference
        first_line = chunk_text.split('\n', 1)[0] if '\n' in chunk_text else chunk_text[:50]
        
        chunks.append({
            'title': f"Chunk {len(chunks)+1}: {first_line}...",
            'content': chunk_text,
            'start_pos': start,
            'end_pos': end
        })
        
        # Move to next chunk with overlap
        start = end - overlap if end < len(text) else len(text)
    
    return chunks