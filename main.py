# main.py

import sys
import argparse
from pdf_utils import download_pdf, extract_text_from_pdf, extract_chunks
from paper_evaluation import PaperEvaluator

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a research paper.')
    parser.add_argument('--url', type=str, default="https://arxiv.org/pdf/2305.16291",
                        help='URL of the PDF to evaluate')
    parser.add_argument('--evaluation', type=str, default="robustness",
                        choices=['methodology', 'robustness', 'significance', 'comprehensive'],
                        help='Type of evaluation to perform')
    args = parser.parse_args()
    
    # Step 1: Download the PDF
    pdf_path = download_pdf(args.url)
    print(f"Downloaded PDF to: {pdf_path}")

    # Step 2: Extract text from the downloaded PDF
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters of text.")

    # Step 3: Extract chunks from the text
    chunks = extract_chunks(text)
    print(f"Created {len(chunks)} chunks from the text.")

    # Step 4: Create paper evaluator
    evaluator = PaperEvaluator(text, chunks)
    print("Created paper evaluator and extracted statistics.")
    
    # Step 5: Perform the requested evaluation
    print(f"\nPerforming {args.evaluation} evaluation...\n")
    
    if args.evaluation == 'methodology':
        evaluation = evaluator.evaluate_methodology()
    elif args.evaluation == 'robustness':
        evaluation = evaluator.evaluate_robustness()
    elif args.evaluation == 'significance':
        evaluation = evaluator.evaluate_significance()
    elif args.evaluation == 'comprehensive':
        evaluation = evaluator.evaluate_comprehensive()
    else:
        print(f"Unknown evaluation type: {args.evaluation}")
        sys.exit(1)
    
    # Step 6: Print the evaluation
    print("\nEvaluation:")
    print(evaluation)

if __name__ == "__main__":
    main()
