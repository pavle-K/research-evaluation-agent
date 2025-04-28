# main.py

import json
from pdf_utils import download_pdf, extract_text_from_pdf, extract_chunks
from semantic_index import create_semantic_index
from query_paper import query_paper
from generate_prompt import generate_analysis_prompt
from call_llm import analyze_with_openai

question = "Evaluate the robustness of this paper?"

def main():
    # Example URL (replace with a real one)
    #pdf_url = "https://arxiv.org/pdf/2504.17131"#
    #pdf_url = "https://arxiv.org/pdf/1504.03473"
    pdf_url = "https://arxiv.org/pdf/2305.16291"

    # Step 1: Download the PDF
    pdf_path = download_pdf(pdf_url)
    print(f"Downloaded PDF to: {pdf_path}")

    # Step 2: Extract text from the downloaded PDF
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters of text.")
    #print(text)

    sections = extract_chunks(text)
    print(len(sections))

    semantic_index = create_semantic_index(sections)
    print(len(semantic_index))

    relevant_paragraphs = query_paper(semantic_index, question)
    print(relevant_paragraphs)

    prompt = generate_analysis_prompt(question, relevant_paragraphs)
    print(prompt)
    
    # Step 6: Analyze the paper with OpenAI
    analysis = analyze_with_openai(question, prompt)
    print("\nAnalysis:")
    print(analysis)

if __name__ == "__main__":
    main()
