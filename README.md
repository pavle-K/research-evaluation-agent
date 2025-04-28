# Deep Research Quality Check

A specialized tool for evaluating academic research papers. This system performs in-depth analysis of research papers, providing detailed evaluations of methodology, robustness, significance, and more.

## Features

- **PDF Processing**: Automatically downloads and extracts text from research papers
- **Semantic Analysis**: Uses embeddings to understand the content and context of the paper
- **Specialized Evaluations**:
  - **Methodology Evaluation**: Assesses the appropriateness and rigor of research methods
  - **Robustness Evaluation**: Analyzes reliability, validity, and generalizability of findings
  - **Significance Evaluation**: Evaluates the importance, novelty, and potential impact
  - **Comprehensive Evaluation**: Combines all evaluations for a complete assessment
- **Statistical Analysis**: Extracts and analyzes key metrics from the paper
- **Command-line Interface**: Easy-to-use CLI for evaluating papers

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/deep_research_quality_check.git
   cd deep_research_quality_check
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```
   Or create a `.env` file with:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Command-line Interface

The easiest way to use the system is through the command-line interface:

```bash
python evaluate_paper.py https://arxiv.org/pdf/2305.16291 --evaluation comprehensive --output evaluation.md
```

Arguments:
- `url`: URL of the PDF to evaluate (required)
- `--evaluation`, `-e`: Type of evaluation to perform (choices: methodology, robustness, significance, comprehensive; default: comprehensive)
- `--output`, `-o`: Output file to save the evaluation
- `--verbose`, `-v`: Print verbose output

### Python API

You can also use the system programmatically:

```python
from pdf_utils import download_pdf, extract_text_from_pdf, extract_chunks
from paper_evaluation import PaperEvaluator

# Download and process the paper
pdf_path = download_pdf("https://arxiv.org/pdf/2305.16291")
text = extract_text_from_pdf(pdf_path)
chunks = extract_chunks(text)

# Create evaluator
evaluator = PaperEvaluator(text, chunks)

# Perform evaluations
methodology_eval = evaluator.evaluate_methodology()
robustness_eval = evaluator.evaluate_robustness()
significance_eval = evaluator.evaluate_significance()
comprehensive_eval = evaluator.evaluate_comprehensive()

# Print or save the evaluations
print(methodology_eval)
```

## System Components

- **pdf_utils.py**: Functions for downloading PDFs and extracting text
- **semantic_index.py**: Creates semantic embeddings for paper content
- **query_paper.py**: Retrieves relevant sections based on queries
- **generate_prompt.py**: Generates prompts for analysis
- **call_llm.py**: Interfaces with OpenAI API for analysis
- **paper_evaluation.py**: Main evaluation logic and specialized assessments
- **evaluate_paper.py**: Command-line interface
- **main.py**: Example script showing the evaluation process

## Evaluation Types

### Methodology Evaluation

Assesses the research methods used in the paper, including:
- Appropriateness of methods for the research question
- Experimental design quality
- Data collection and analysis techniques
- Methodological limitations and biases
- Overall assessment of methodological rigor

### Robustness Evaluation

Analyzes the reliability and validity of the research, including:
- Reliability and reproducibility of results
- Statistical significance and effect sizes
- Treatment of confounding variables and biases
- Generalizability of findings
- Overall assessment of research robustness

### Significance Evaluation

Evaluates the importance and impact of the research, including:
- Importance of the research question in the field
- Novelty of approach or findings
- Advancement of knowledge in the field
- Potential impact on theory or practice
- Overall assessment of research significance

### Comprehensive Evaluation

Combines all the above evaluations for a complete assessment of the paper, including:
- Summary of key strengths and weaknesses
- Overall assessment of paper quality and contribution
- Constructive suggestions for improvement
- Final verdict on the paper's merit

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

## License

MIT License
