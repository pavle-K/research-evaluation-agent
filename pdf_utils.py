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

def detect_sections(text: str):
    """
    Enhanced academic paper section detector that finds both main sections and subsections.
    
    This detector specifically addresses:
    - Numbered sections like "3 Modal Interface Automata"
    - Numbered subsections like "2.1 Input/Output Conformance"
    - Non-numbered section titles
    - Computer science specific terminology in section headings
    
    Args:
        text (str): The full text of the academic paper
        
    Returns:
        dict: A hierarchical structure with main sections and their subsections
    """
    lines = text.split('\n')
    all_sections = []
    
    # Common section names in academic papers
    common_sections = [
        "abstract", "introduction", "background", "related work",
        "methodology", "methods", "experiments", "results", 
        "discussion", "conclusion", "future work", "references",
        "preliminaries", "implementation", "evaluation", "analysis"
    ]
    
    # Computer science specific terms that often appear in section titles
    cs_specific_terms = [
        "interface", "automata", "conformance", "testing",
        "modal", "correctness", "proof", "theorem", "proposition"
    ]
    
    # Extract potential sections
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines or very long lines (likely paragraphs)
        if not line or len(line) > 70:
            continue
        
        is_section = False
        
        # 1. Check for section numbers with decimal points (e.g., "2.1 Methods")
        if re.match(r'^(\d+\.\d+)\s+[A-Z]', line):
            is_section = True
        
        # 2. Check for single-number section titles (e.g., "3 Modal Interface Automata")
        elif re.match(r'^(\d+)\s+[A-Z]', line):
            is_section = True
        
        # 3. Check for standalone section numbers (e.g., "3" on its own line)
        elif re.match(r'^\d+$', line) and i < len(lines)-1 and lines[i+1].strip() and lines[i+1].strip()[0].isupper():
            is_section = True
        
        # 4. Check for common section titles (exact match)
        elif any(section.lower() == line.lower() for section in common_sections):
            is_section = True
        
        # 5. Check for Computer Science specific section titles
        elif (any(term.lower() in line.lower() for term in cs_specific_terms) and 
              len(line.split()) <= 6 and line[0].isupper()):
            is_section = True
        
        # 6. Check for I/O-related sections (specific to this paper domain)
        elif ("I/O" in line or "Input/Output" in line) and len(line.split()) <= 6:
            is_section = True
            
        if is_section:
            all_sections.append((i, line))  # Store line number for ordering
    
    # Organize sections into a hierarchical structure
    hierarchy = {}
    current_main_section = None
    
    # Sort sections by line number to preserve original order
    all_sections.sort(key=lambda x: x[0])
    sections = [s[1] for s in all_sections]
    
    for section in sections:
        # Identify main sections vs subsections
        if (not re.match(r'^\d+\.\d+', section) and  # Not like "2.1"
            (not re.match(r'^\d+\s+', section) or     # Not like "3 Something" 
             section in ["Introduction", "Preliminaries", "References"])):  # Common main sections
            current_main_section = section
            hierarchy[current_main_section] = []
        elif re.match(r'^\d+\s+', section):  # Like "3 Modal Interface Automata"
            current_main_section = section
            hierarchy[current_main_section] = []
        elif current_main_section is not None:
            # This is a subsection of the current main section
            hierarchy[current_main_section].append(section)
    
    return hierarchy


def extract_sections_flat(text: str):
    """
    Returns a simple flat list of all detected sections and subsections.
    
    Args:
        text (str): The full text of the academic paper
        
    Returns:
        list: All detected sections and subsections in order of appearance
    """
    hierarchy = detect_sections(text)
    flat_list = []
    
    for main_section, subsections in hierarchy.items():
        flat_list.append(main_section)
        flat_list.extend(subsections)
    
    return flat_list