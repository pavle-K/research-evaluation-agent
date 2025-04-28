# Research Evaluation Agent

A specialized tool for evaluating academic research papers. This system performs in-depth analysis of research papers, providing detailed evaluations of methodology, robustness, significance, and more.

## Directory Structure

```
src/
├── evaluate_paper.py       # Main entry point and CLI interface
├── evaluation/            # Core evaluation logic
│   ├── paper_evaluation.py     # Paper evaluation implementation
│   └── research_classifier.py  # Research type classification
└── utils/
    ├── pdf_utils/        # PDF processing utilities
    │   └── pdf_utils.py  # PDF handling functions
    └── llm_utils/        # LLM interaction utilities
        ├── call_llm.py         # OpenAI API interaction
        ├── generate_prompt.py  # Prompt generation
        ├── semantic_index.py   # Semantic search functionality
        └── query_paper.py      # Paper content querying
```

## How It Works

The system follows this pipeline when evaluating a paper:

1. **Paper Download & Processing** (`utils/pdf_utils/pdf_utils.py`):
   - Downloads PDF from provided URL
   - Extracts text content
   - Chunks text into manageable segments for analysis

2. **Research Classification** (`evaluation/research_classifier.py`):
   - Analyzes paper content to determine research type
   - Categories include: empirical_quantitative, empirical_qualitative, theoretical, review, etc.
   - Provides tailored evaluation criteria based on research type

3. **Semantic Indexing** (`utils/llm_utils/semantic_index.py`):
   - Creates embeddings of paper chunks using OpenAI's API
   - Enables semantic search through paper content
   - Helps find relevant sections for specific evaluation aspects

4. **Paper Querying** (`utils/llm_utils/query_paper.py`):
   - Uses semantic search to find relevant paper sections
   - Applies heuristics based on query type
   - Refines results for better relevance

5. **Evaluation Process** (`evaluation/paper_evaluation.py`):
   - Methodology Evaluation: Assesses research methods and approach
   - Robustness Evaluation: Analyzes reliability and validity
   - Significance Evaluation: Evaluates importance and impact
   - Comprehensive Evaluation: Combines all aspects

6. **LLM Integration**:
   - `utils/llm_utils/generate_prompt.py`: Creates structured prompts for analysis
   - `utils/llm_utils/call_llm.py`: Handles OpenAI API interaction with retry logic

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

### Basic Usage

The simplest way to evaluate a paper:

```bash
python src/evaluate_paper.py https://arxiv.org/pdf/2305.16291 --evaluation comprehensive
```

### Advanced Usage

For more detailed output and control:

```bash
python src/evaluate_paper.py https://arxiv.org/pdf/2305.16291 --evaluation robustness --verbose --output evaluation.md
```

Arguments:
- `url`: URL of the PDF to evaluate (required)
- `--evaluation`, `-e`: Evaluation type (choices below)
- `--output`, `-o`: Save evaluation to file
- `--verbose`, `-v`: Print detailed progress

### Evaluation Types

1. **Methodology** (`--evaluation methodology`):
   - Assesses research methods
   - Evaluates experimental design
   - Analyzes data collection techniques
   - Reviews methodological limitations

2. **Robustness** (`--evaluation robustness`):
   - Checks statistical validity
   - Examines reproducibility
   - Assesses generalizability
   - Reviews potential biases

3. **Significance** (`--evaluation significance`):
   - Evaluates research contribution
   - Assesses novelty
   - Reviews potential impact
   - Analyzes theoretical/practical implications

4. **Comprehensive** (`--evaluation comprehensive`):
   - Combines all above evaluations
   - Provides overall assessment
   - Includes improvement suggestions
   - Gives final quality verdict

## Technical Details

### Paper Processing Pipeline

1. **PDF Processing**:
   - Downloads PDF using requests
   - Extracts text using PyMuPDF
   - Chunks text with overlap for context preservation

2. **Semantic Analysis**:
   - Creates embeddings using OpenAI's API
   - Builds searchable index
   - Enables semantic similarity search

3. **Research Classification**:
   - Uses both rule-based and LLM-based classification
   - Identifies research type
   - Provides type-specific evaluation criteria

4. **Evaluation Process**:
   - Generates targeted queries
   - Retrieves relevant sections
   - Creates specialized prompts
   - Analyzes using OpenAI's GPT models

### Key Components

- **Semantic Search**: Uses OpenAI embeddings for finding relevant paper sections
- **Adaptive Evaluation**: Tailors criteria based on research type
- **Robust LLM Integration**: Includes retry logic and error handling
- **Modular Design**: Separates concerns for maintainability

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages:
  - openai
  - numpy
  - PyMuPDF
  - python-dotenv
  - requests

## License

MIT License
