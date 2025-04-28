"""
Module for classifying research papers by type and providing tailored evaluation criteria.
"""

import re
from collections import Counter
from utils.llm_utils.call_llm import analyze_with_openai

# Research paper types and their characteristics
RESEARCH_TYPES = {
    "empirical_quantitative": {
        "description": "Empirical research with quantitative methods, statistical analysis, and hypothesis testing",
        "keywords": ["experiment", "statistical", "hypothesis", "sample", "significant", "p-value", "correlation", "regression", "variance", "mean", "standard deviation", "control group", "treatment group", "random", "variable", "data", "analysis", "quantitative", "measure", "test"],
        "sections": ["methodology", "results", "discussion", "limitations", "future work"],
        "evaluation_focus": ["statistical validity", "sample size", "methodology rigor", "reproducibility", "generalizability", "effect size", "power analysis"]
    },
    "empirical_qualitative": {
        "description": "Empirical research with qualitative methods such as interviews, case studies, or ethnography",
        "keywords": ["interview", "participant", "theme", "qualitative", "case study", "ethnography", "observation", "narrative", "discourse", "content analysis", "grounded theory", "phenomenology", "coding", "transcript", "focus group"],
        "sections": ["methodology", "findings", "discussion", "limitations"],
        "evaluation_focus": ["methodological rigor", "trustworthiness", "credibility", "transferability", "dependability", "confirmability", "reflexivity", "thick description"]
    },
    "theoretical": {
        "description": "Theoretical research proposing new concepts, frameworks, or models without empirical testing",
        "keywords": ["theory", "framework", "model", "concept", "proposition", "axiom", "paradigm", "theoretical", "conceptual", "philosophy", "logic", "argument", "premise", "conclusion", "deductive", "inductive"],
        "sections": ["theoretical framework", "model development", "implications", "future research"],
        "evaluation_focus": ["logical consistency", "conceptual clarity", "theoretical contribution", "explanatory power", "parsimony", "scope", "utility"]
    },
    "review": {
        "description": "Literature review or meta-analysis synthesizing existing research",
        "keywords": ["review", "literature", "meta-analysis", "systematic", "synthesis", "summarize", "previous research", "state of the art", "survey", "overview"],
        "sections": ["search methodology", "inclusion criteria", "synthesis", "research gaps", "future directions"],
        "evaluation_focus": ["comprehensiveness", "systematic approach", "quality assessment", "synthesis methods", "research gap identification"]
    },
    "methodology": {
        "description": "Research proposing new research methods, tools, or techniques",
        "keywords": ["method", "technique", "tool", "approach", "procedure", "protocol", "algorithm", "measurement", "instrument", "assessment", "validation", "reliability", "accuracy", "precision"],
        "sections": ["method description", "validation", "comparison", "limitations"],
        "evaluation_focus": ["novelty", "validity", "reliability", "usability", "efficiency", "comparison with existing methods"]
    },
    "case_study": {
        "description": "In-depth analysis of a specific case, organization, or phenomenon",
        "keywords": ["case", "organization", "company", "industry", "specific", "particular", "instance", "example", "illustration", "in-depth", "detailed"],
        "sections": ["case description", "analysis", "findings", "implications"],
        "evaluation_focus": ["depth of analysis", "contextual understanding", "transferability of insights", "practical implications"]
    },
    "simulation": {
        "description": "Research using computational models or simulations",
        "keywords": ["simulation", "model", "computational", "parameter", "algorithm", "iteration", "convergence", "optimization", "agent-based", "monte carlo", "stochastic", "deterministic"],
        "sections": ["model description", "simulation setup", "results", "validation"],
        "evaluation_focus": ["model validity", "parameter justification", "sensitivity analysis", "comparison with real-world data", "computational efficiency"]
    },
    "design_science": {
        "description": "Research designing and evaluating artifacts, systems, or solutions",
        "keywords": ["design", "artifact", "system", "solution", "prototype", "implementation", "evaluation", "usability", "utility", "effectiveness", "efficiency", "satisfaction"],
        "sections": ["problem identification", "design", "implementation", "evaluation", "discussion"],
        "evaluation_focus": ["problem relevance", "design quality", "evaluation rigor", "utility", "novelty", "practical implications"]
    },
    "whitepaper": {
        "description": "Technical document describing a problem, solution, or technology, often with preliminary results or proof of concept",
        "keywords": ["whitepaper", "technical", "solution", "technology", "architecture", "implementation", "proof of concept", "preliminary", "proposal", "roadmap"],
        "sections": ["problem statement", "proposed solution", "architecture", "implementation", "preliminary results", "future work"],
        "evaluation_focus": ["problem definition clarity", "solution feasibility", "technical soundness", "preliminary validation", "limitations acknowledgment"]
    },
    "position_paper": {
        "description": "Paper presenting an opinion, viewpoint, or argument on a topic",
        "keywords": ["position", "viewpoint", "perspective", "opinion", "argument", "debate", "controversial", "propose", "advocate", "critique", "challenge"],
        "sections": ["position statement", "arguments", "counterarguments", "implications"],
        "evaluation_focus": ["argument strength", "evidence quality", "consideration of alternatives", "implications"]
    }
}

class ResearchClassifier:
    """
    Class for classifying research papers and providing tailored evaluation criteria.
    """
    
    def __init__(self, paper_text, chunks):
        """
        Initialize the classifier with the paper text and chunks.
        
        Args:
            paper_text (str): The full text of the paper
            chunks (list): List of chunk dictionaries with 'content' key
        """
        self.paper_text = paper_text
        self.chunks = chunks
        self.paper_stats = self._extract_basic_stats()
        self.abstract = self._extract_abstract()
    
    def _extract_abstract(self):
        """
        Extract the abstract from the paper text using common patterns.
        
        Returns:
            str: The extracted abstract or None if not found
        """
        # Common patterns for abstract sections
        abstract_patterns = [
            r'(?i)abstract\s*\n+([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\n\s*1\.|\n\s*Introduction)',
            r'(?i)ABSTRACT[:\s]*\n+([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\n\s*1\.|\n\s*Introduction)',
            r'(?i)Summary[:\s]*\n+([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\n\s*1\.|\n\s*Introduction)',
            r'(?i)Overview[:\s]*\n+([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\n\s*1\.|\n\s*Introduction)'
        ]
        
        # Try each pattern
        for pattern in abstract_patterns:
            match = re.search(pattern, self.paper_text)
            if match:
                abstract = match.group(1).strip()
                # Clean up the abstract
                abstract = re.sub(r'\s+', ' ', abstract)  # Normalize whitespace
                return abstract
        
        # If no abstract found, try to get the first substantial paragraph
        paragraphs = re.split(r'\n\s*\n', self.paper_text)
        for para in paragraphs[:3]:  # Check first 3 paragraphs
            if len(para.strip()) > 100 and not re.match(r'(?i)(keywords|index terms|table of contents)', para):
                return para.strip()
        
        return None
        
    def _extract_basic_stats(self):
        """
        Extract basic statistics from the paper text.
        
        Returns:
            dict: Dictionary of paper statistics
        """
        stats = {}
        
        # Count words
        words = re.findall(r'\b\w+\b', self.paper_text.lower())
        stats['word_count'] = len(words)
        
        # Count sentences
        sentences = re.split(r'[.!?]+', self.paper_text)
        stats['sentence_count'] = len([s for s in sentences if len(s.strip()) > 0])
        
        # Count citations
        citations = re.findall(r'\[\d+\]|\(\w+\s+et\s+al\.', self.paper_text)
        stats['citation_count'] = len(citations)
        
        # Count figures and tables
        figures = re.findall(r'[Ff]ig(?:ure)?\.?\s*\d+', self.paper_text)
        tables = re.findall(r'[Tt]able\.?\s*\d+', self.paper_text)
        stats['figure_count'] = len(figures)
        stats['table_count'] = len(tables)
        
        # Count equations
        equations = re.findall(r'[=><≥≤±×÷≈≠∝∞∫∑∏√]', self.paper_text)
        stats['equation_count'] = len(equations)
        
        # Count statistical terms
        statistical_terms = re.findall(r'\bp(?:\s*[<>=]|\s*value)\s*[<>=]?\s*0\.\d+|\bt\s*\(\s*\d+\s*\)\s*[<>=]\s*\d+\.\d+|chi[\s-]*square|anova|manova|regression|correlation|mean|median|standard\s+deviation|variance|significance|statistical|sample\s+size', self.paper_text.lower())
        stats['statistical_terms_count'] = len(statistical_terms)
        
        # Count methodology terms
        methodology_terms = re.findall(r'\bmethodology|method|approach|technique|procedure|experiment|study|analysis|design|protocol|framework|model|algorithm|implementation|evaluation|validation|testing|measurement|assessment|data\s+collection|sampling\b', self.paper_text.lower())
        stats['methodology_terms_count'] = len(methodology_terms)
        
        # Count sections
        section_headers = re.findall(r'\n\s*\d+\.?\s*[A-Z][A-Za-z\s]+\n|\n\s*[A-Z][A-Za-z\s]+\n', self.paper_text)
        stats['section_count'] = len(section_headers)
        
        # Most common words (excluding stopwords)
        stopwords = {'the', 'and', 'of', 'to', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'as', 'are', 'be', 'we', 'our', 'from', 'an', 'or', 'at', 'not', 'it', 'which', 'have', 'was', 'were', 'has', 'been', 'can', 'will', 'their', 'they', 'these', 'those', 'such', 'but', 'also', 'than', 'when', 'where', 'who', 'what', 'how', 'why', 'all', 'any', 'some', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'}
        filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
        word_freq = Counter(filtered_words)
        stats['common_words'] = word_freq.most_common(30)
        
        return stats
    
    def classify_research_type(self):
        """
        Classify the research paper type based primarily on the abstract.
        
        Returns:
            dict: Classification results with type, confidence, and rationale
        """
        # Prepare classification prompt focusing on abstract
        classification_prompt = f"""
        I need to classify a research paper into one of the following types:
        
        {', '.join([f"{t}: {RESEARCH_TYPES[t]['description']}" for t in RESEARCH_TYPES.keys()])}
        
        Here is the paper's abstract:
        
        {self.abstract if self.abstract else "Abstract not found. Using paper excerpts instead:"}
        
        """
        
        # If no abstract found, add excerpts from the introduction
        if not self.abstract:
            # Try to find introduction section
            intro_match = re.search(r'(?i)(?:1\.|I\.|)Introduction\s*\n+([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\n\s*2\.|\n\s*II\.)', self.paper_text)
            if intro_match:
                classification_prompt += f"\nFrom introduction:\n{intro_match.group(1)[:1000]}\n"
            else:
                # Fall back to first few chunks
                for i, chunk in enumerate(self.chunks[:2]):
                    classification_prompt += f"\nExcerpt {i+1}:\n{chunk['content'][:500]}\n"
        
        classification_prompt += """
        
        Please classify this paper into exactly ONE of the research types listed above. Focus on:
        1. The primary research methodology described
        2. The main contribution of the paper
        3. The type of results or findings presented
        4. The overall structure and approach
        
        Do NOT over-emphasize:
        - Presence of data or analysis (many papers use these)
        - Statistical terms (these are common across types)
        - Generic research vocabulary
        
        Instead, look for clear indicators of the research approach and primary contribution.
        
        Format your response as:
        RESEARCH_TYPE: [type]
        CONFIDENCE: [high/medium/low]
        RATIONALE: [detailed explanation]
        KEY_CHARACTERISTICS: [bullet points of key characteristics]
        """
        
        # Get classification from LLM
        classification_result = analyze_with_openai("Classify research paper", classification_prompt)
        
        # Parse the classification result
        research_type = None
        confidence = None
        rationale = None
        characteristics = None
        
        for line in classification_result.split('\n'):
            if line.startswith('RESEARCH_TYPE:'):
                research_type = line.replace('RESEARCH_TYPE:', '').strip().lower()
            elif line.startswith('CONFIDENCE:'):
                confidence = line.replace('CONFIDENCE:', '').strip().lower()
            elif line.startswith('RATIONALE:'):
                rationale = line.replace('RATIONALE:', '').strip()
            elif line.startswith('KEY_CHARACTERISTICS:'):
                characteristics = line.replace('KEY_CHARACTERISTICS:', '').strip()
        
        # If the research type is not in our predefined types, default to empirical_quantitative with low confidence
        if research_type not in RESEARCH_TYPES:
            research_type = "empirical_quantitative"
            confidence = "low"
            rationale = "Failed to determine research type from abstract/introduction. Defaulting to empirical_quantitative with low confidence."
        
        return {
            "research_type": research_type,
            "confidence": confidence,
            "rationale": rationale,
            "characteristics": characteristics,
            "type_info": RESEARCH_TYPES[research_type]
        }
    
    def get_tailored_evaluation_criteria(self, research_type=None):
        """
        Get tailored evaluation criteria based on the research type.
        
        Args:
            research_type (str, optional): Research type. If None, will classify automatically.
            
        Returns:
            dict: Tailored evaluation criteria
        """
        if research_type is None:
            classification = self.classify_research_type()
            research_type = classification["research_type"]
            type_info = classification["type_info"]
            classification_details = classification
        else:
            if research_type not in RESEARCH_TYPES:
                raise ValueError(f"Unknown research type: {research_type}")
            type_info = RESEARCH_TYPES[research_type]
            classification_details = None
        
        # Generate tailored evaluation criteria
        criteria_prompt = f"""
        I need to create tailored evaluation criteria for a research paper of type: {research_type} ({type_info['description']}).
        
        This type of research should typically focus on: {', '.join(type_info['evaluation_focus'])}.
        
        Paper statistics:
        - Word count: {self.paper_stats['word_count']}
        - Citation count: {self.paper_stats['citation_count']}
        - Figure count: {self.paper_stats['figure_count']}
        - Table count: {self.paper_stats['table_count']}
        - Equation count: {self.paper_stats['equation_count']}
        - Statistical terms count: {self.paper_stats['statistical_terms_count']}
        - Methodology terms count: {self.paper_stats['methodology_terms_count']}
        
        Please create detailed evaluation criteria for the following aspects:
        
        1. Methodology Evaluation:
           - What specific methodological elements should be evaluated for this type of research?
           - What constitutes methodological rigor in this context?
           - What are common methodological pitfalls or limitations to look for?
           - What standards or benchmarks should be applied?
        
        2. Robustness Evaluation:
           - How should reliability and validity be assessed for this research type?
           - What specific robustness checks are appropriate?
           - What level of evidence is required for claims made?
           - How should generalizability or transferability be evaluated?
        
        3. Significance Evaluation:
           - How should the contribution be assessed for this type of research?
           - What constitutes novelty or innovation in this context?
           - How should theoretical or practical impact be evaluated?
           - What standards exist for significance in this research domain?
        
        For each aspect, provide 5-7 specific criteria or questions that should be used in the evaluation, tailored specifically to this research type.
        
        Format your response as:
        
        METHODOLOGY_CRITERIA:
        1. [Specific criterion 1]
        2. [Specific criterion 2]
        ...
        
        ROBUSTNESS_CRITERIA:
        1. [Specific criterion 1]
        2. [Specific criterion 2]
        ...
        
        SIGNIFICANCE_CRITERIA:
        1. [Specific criterion 1]
        2. [Specific criterion 2]
        ...
        """
        
        # Get tailored criteria from LLM
        criteria_result = analyze_with_openai("Generate evaluation criteria", criteria_prompt)
        
        # Parse the criteria result
        methodology_criteria = []
        robustness_criteria = []
        significance_criteria = []
        
        current_section = None
        for line in criteria_result.split('\n'):
            line = line.strip()
            if line.startswith('METHODOLOGY_CRITERIA:'):
                current_section = 'methodology'
            elif line.startswith('ROBUSTNESS_CRITERIA:'):
                current_section = 'robustness'
            elif line.startswith('SIGNIFICANCE_CRITERIA:'):
                current_section = 'significance'
            elif line and line[0].isdigit() and '. ' in line:
                criterion = line.split('. ', 1)[1].strip()
                if current_section == 'methodology':
                    methodology_criteria.append(criterion)
                elif current_section == 'robustness':
                    robustness_criteria.append(criterion)
                elif current_section == 'significance':
                    significance_criteria.append(criterion)
        
        return {
            "research_type": research_type,
            "type_description": type_info['description'],
            "classification_details": classification_details,
            "methodology_criteria": methodology_criteria,
            "robustness_criteria": robustness_criteria,
            "significance_criteria": significance_criteria,
            "evaluation_focus": type_info['evaluation_focus']
        }
