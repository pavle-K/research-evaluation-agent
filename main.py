# main.py

from pdf_utils import download_pdf, extract_text_from_pdf, detect_sections, extract_sections_flat

def main():
    # Example URL (replace with a real one)
    pdf_url = "https://arxiv.org/pdf/1504.03473"

    # Step 1: Download the PDF
    pdf_path = download_pdf(pdf_url)
    print(f"Downloaded PDF to: {pdf_path}")

    # Step 2: Extract text from the downloaded PDF
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters of text.")
    
    # Step 3: Chunk the extracted text (optional, based on your needs)
    text_sections = extract_sections_flat(text)
    print(f"Created {len(text_sections)} chunks.")

    # Print out the first chunk for verification (or print other relevant info)
    print("\nFirst chunk of text:")
    print(text_sections)

if __name__ == "__main__":
    main()
