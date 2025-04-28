#!/usr/bin/env python3
"""
Command-line interface for evaluating research papers.
"""

import argparse
import sys
import time
from pdf_utils import download_pdf, extract_text_from_pdf, extract_chunks
from paper_evaluation import PaperEvaluator

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a research paper.')
    
    # Required arguments
    parser.add_argument('url', type=str, help='URL of the PDF to evaluate')
    
    # Optional arguments
    parser.add_argument('--evaluation', '-e', type=str, default="comprehensive",
                        choices=['methodology', 'robustness', 'significance', 'comprehensive'],
                        help='Type of evaluation to perform (default: comprehensive)')
    parser.add_argument('--output', '-o', type=str, help='Output file to save the evaluation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')
    
    args = parser.parse_args()
    
    # Print welcome message
    print("=" * 80)
    print(f"Research Paper Evaluator")
    print("=" * 80)
    print(f"Paper URL: {args.url}")
    print(f"Evaluation type: {args.evaluation}")
    if args.output:
        print(f"Output file: {args.output}")
    print("=" * 80)
    
    try:
        # Step 1: Download the PDF
        print("\n[1/5] Downloading PDF...")
        start_time = time.time()
        pdf_path = download_pdf(args.url)
        if args.verbose:
            print(f"Downloaded PDF to: {pdf_path}")
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # Step 2: Extract text from the downloaded PDF
        print("\n[2/5] Extracting text from PDF...")
        start_time = time.time()
        text = extract_text_from_pdf(pdf_path)
        if args.verbose:
            print(f"Extracted {len(text)} characters of text.")
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # Step 3: Extract chunks from the text
        print("\n[3/5] Chunking text...")
        start_time = time.time()
        chunks = extract_chunks(text)
        if args.verbose:
            print(f"Created {len(chunks)} chunks from the text.")
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # Step 4: Create paper evaluator
        print("\n[4/5] Creating paper evaluator...")
        start_time = time.time()
        evaluator = PaperEvaluator(text, chunks)
        if args.verbose:
            print("Created paper evaluator and extracted statistics:")
            print(f"- Word count: {evaluator.stats['word_count']}")
            print(f"- Sentence count: {evaluator.stats['sentence_count']}")
            print(f"- Paragraph count: {evaluator.stats['paragraph_count']}")
            print(f"- Citation count: {evaluator.stats['citation_count']}")
            print(f"- Figure count: {evaluator.stats['figure_count']}")
            print(f"- Table count: {evaluator.stats['table_count']}")
            print(f"- Equation count: {evaluator.stats['equation_count']}")
            print(f"- Top 10 terms: {', '.join([f'{term[0]} ({term[1]})' for term in evaluator.stats['common_words'][:10]])}")
        print(f"Done in {time.time() - start_time:.2f} seconds.")
        
        # Step 5: Perform the requested evaluation
        print(f"\n[5/5] Performing {args.evaluation} evaluation...")
        print("This may take a few minutes depending on the length of the paper...")
        start_time = time.time()
        
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
        
        print(f"Done in {time.time() - start_time:.2f} seconds.")
        
        # Step 6: Print and save the evaluation
        print("\n" + "=" * 80)
        print(f"EVALUATION RESULTS: {args.evaluation.upper()}")
        print("=" * 80 + "\n")
        print(evaluation)
        print("\n" + "=" * 80)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"# {args.evaluation.upper()} EVALUATION\n\n")
                f.write(f"Paper URL: {args.url}\n\n")
                f.write(evaluation)
            print(f"\nEvaluation saved to {args.output}")
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
