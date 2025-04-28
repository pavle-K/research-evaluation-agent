import re
from collections import Counter
import numpy as np
from semantic_index import create_semantic_index
from query_paper import query_paper
from generate_prompt import generate_analysis_prompt
from call_llm import analyze_with_openai
from research_classifier import ResearchClassifier

class PaperEvaluator:
    """
    A class for specialized evaluation of research papers.
    """
    
    def __init__(self, paper_text, chunks):
        """
        Initialize the evaluator with the paper text and chunks.
        
        Args:
            paper_text (str): The full text of the paper
            chunks (list): List of chunk dictionaries with 'content' key
        """
        self.paper_text = paper_text
        self.chunks = chunks
        self.semantic_index = create_semantic_index(chunks)
        
        # Extract basic paper statistics
        self.stats = self._extract_paper_statistics()
        
        # Create research classifier
        self.classifier = ResearchClassifier(paper_text, chunks)
        
        # Classify research type and get tailored criteria
        self.research_classification = self.classifier.classify_research_type()
        self.evaluation_criteria = self.classifier.get_tailored_evaluation_criteria(
            self.research_classification["research_type"]
        )
        
    def _extract_paper_statistics(self):
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
        
        # Count paragraphs
        paragraphs = re.split(r'\n\s*\n', self.paper_text)
        stats['paragraph_count'] = len([p for p in paragraphs if len(p.strip()) > 0])
        
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
        
        # Most common words (excluding stopwords)
        stopwords = {'the', 'and', 'of', 'to', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'as', 'are', 'be', 'we', 'our', 'from', 'an', 'or', 'at', 'not', 'it', 'which', 'have', 'was', 'were', 'has', 'been', 'can', 'will', 'their', 'they', 'these', 'those', 'such', 'but', 'also', 'than', 'when', 'where', 'who', 'what', 'how', 'why', 'all', 'any', 'some', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'}
        filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
        word_freq = Counter(filtered_words)
        stats['common_words'] = word_freq.most_common(20)
        
        return stats
    
    def evaluate_methodology(self):
        """
        Evaluate the methodology of the paper.
        
        Returns:
            str: Detailed evaluation of the paper's methodology
        """
        # Get research type and tailored criteria
        research_type = self.research_classification["research_type"]
        type_description = self.evaluation_criteria["type_description"]
        methodology_criteria = self.evaluation_criteria["methodology_criteria"]
        
        # Generate custom queries based on research type
        methodology_queries = [
            "What methods does this paper use?",
            "How is the experimental design structured?",
            "What data collection techniques are used?",
            "How does the paper analyze data?",
            "What are the limitations of the methodology?"
        ]
        
        # Add research-type specific queries
        if research_type == "empirical_quantitative":
            methodology_queries.extend([
                "What statistical methods are used in this paper?",
                "How large is the sample size and how was it determined?",
                "What control measures or variables are used in the experiments?"
            ])
        elif research_type == "empirical_qualitative":
            methodology_queries.extend([
                "What qualitative methods are used in this paper?",
                "How were participants or cases selected?",
                "What data collection and analysis procedures were used?"
            ])
        elif research_type == "simulation":
            methodology_queries.extend([
                "How is the simulation model designed and implemented?",
                "What parameters and assumptions are used in the simulation?",
                "How is the simulation validated or verified?"
            ])
        elif research_type == "whitepaper":
            methodology_queries.extend([
                "What proof-of-concept or preliminary testing is described?",
                "How are the proposed solutions or technologies implemented?",
                "What evaluation methods are used to assess the proposed solution?"
            ])
        
        # Query for methodology-related content
        methodology_results = []
        for query in methodology_queries:
            relevant_paragraphs = query_paper(self.semantic_index, query)
            prompt = generate_analysis_prompt(query, relevant_paragraphs)
            analysis = analyze_with_openai(query, prompt)
            methodology_results.append({
                'query': query,
                'analysis': analysis
            })
        
        # Look for methodology-specific keywords
        methodology_keywords = ['method', 'approach', 'technique', 'procedure', 'experiment', 'study', 'analysis', 'design', 'protocol', 'framework', 'model', 'algorithm', 'implementation', 'evaluation', 'validation', 'testing', 'measurement', 'assessment', 'data collection', 'sampling']
        
        keyword_counts = {}
        for keyword in methodology_keywords:
            count = len(re.findall(r'\b' + keyword + r'\b', self.paper_text.lower()))
            keyword_counts[keyword] = count
        
        # Generate comprehensive methodology evaluation
        methodology_prompt = f"""
        Provide a comprehensive evaluation of the research methodology in this paper.
        
        IMPORTANT: This paper has been classified as a {research_type} paper ({type_description}).
        
        For this type of research, the methodology evaluation should focus on the following criteria:
        {', '.join([f"{i+1}. {criterion}" for i, criterion in enumerate(methodology_criteria)])}
        
        Paper statistics:
        - Word count: {self.stats['word_count']}
        - Citation count: {self.stats['citation_count']}
        - Figure count: {self.stats['figure_count']}
        - Table count: {self.stats['table_count']}
        - Equation count: {self.stats['equation_count']}
        
        Methodology keyword frequencies:
        {', '.join([f"{k}: {v}" for k, v in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)])}
        
        Methodology aspects:
        """
        
        for result in methodology_results:
            methodology_prompt += f"\n\n{result['query']}:\n{result['analysis']}"
        
        methodology_prompt += f"""
        
        Based on the above information, provide a detailed evaluation of the methodology for this {research_type} paper. Your evaluation must:
        
        1. Be extremely critical and rigorous
        2. Identify specific methodological weaknesses and limitations
        3. Assess whether the methodology is appropriate for the research goals
        4. Evaluate whether the evidence presented is sufficient to support the claims made
        5. Consider whether alternative methodologies would have been more appropriate
        
        For each of the following criteria, provide a detailed assessment:
        
        {chr(10).join([f"- {criterion}" for criterion in methodology_criteria])}
        
        Structure your evaluation with clear sections, highlighting both strengths and weaknesses. Be specific and precise in your critique, avoiding generic statements. If the paper makes claims without sufficient methodological support, explicitly identify these issues.
        """
        
        final_evaluation = analyze_with_openai("Evaluate methodology", methodology_prompt)
        return final_evaluation
    
    def evaluate_robustness(self):
        """
        Evaluate the robustness of the paper.
        
        Returns:
            str: Detailed evaluation of the paper's robustness
        """
        # Get research type and tailored criteria
        research_type = self.research_classification["research_type"]
        type_description = self.evaluation_criteria["type_description"]
        robustness_criteria = self.evaluation_criteria["robustness_criteria"]
        
        # Generate custom queries based on research type
        robustness_queries = [
            "How reliable are the results in this paper?",
            "What statistical methods are used to ensure validity?",
            "How does the paper address potential confounding variables?",
            "What limitations or threats to validity are discussed?",
            "How generalizable are the findings of this paper?"
        ]
        
        # Add research-type specific queries
        if research_type == "empirical_quantitative":
            robustness_queries.extend([
                "What statistical power analysis or sample size justification is provided?",
                "How are effect sizes reported and interpreted?",
                "What measures are taken to ensure reproducibility of the results?"
            ])
        elif research_type == "simulation":
            robustness_queries.extend([
                "What sensitivity analysis is performed on the simulation parameters?",
                "How is the simulation validated against real-world data?",
                "What measures are taken to ensure reproducibility of the simulation?"
            ])
        elif research_type == "whitepaper":
            robustness_queries.extend([
                "What evidence is provided to support the claims made?",
                "How thoroughly are the limitations of the proposed solution discussed?",
                "What potential challenges to implementation are addressed?"
            ])
        
        # Query for robustness-related content
        robustness_results = []
        for query in robustness_queries:
            relevant_paragraphs = query_paper(self.semantic_index, query)
            prompt = generate_analysis_prompt(query, relevant_paragraphs)
            analysis = analyze_with_openai(query, prompt)
            robustness_results.append({
                'query': query,
                'analysis': analysis
            })
        
        # Look for robustness-specific keywords
        robustness_keywords = ['robust', 'reliability', 'validity', 'reproducibility', 'replication', 'generalizability', 'significance', 'p-value', 'confidence interval', 'effect size', 'power', 'sample size', 'bias', 'confound', 'limitation', 'threat', 'error', 'uncertainty', 'variance', 'outlier']
        
        keyword_counts = {}
        for keyword in robustness_keywords:
            count = len(re.findall(r'\b' + keyword + r'\b', self.paper_text.lower()))
            keyword_counts[keyword] = count
        
        # Generate comprehensive robustness evaluation
        robustness_prompt = f"""
        Provide a comprehensive evaluation of the research robustness in this paper.
        
        IMPORTANT: This paper has been classified as a {research_type} paper ({type_description}).
        
        For this type of research, the robustness evaluation should focus on the following criteria:
        {', '.join([f"{i+1}. {criterion}" for i, criterion in enumerate(robustness_criteria)])}
        
        Paper statistics:
        - Word count: {self.stats['word_count']}
        - Citation count: {self.stats['citation_count']}
        - Figure count: {self.stats['figure_count']}
        - Table count: {self.stats['table_count']}
        - Equation count: {self.stats['equation_count']}
        
        Robustness keyword frequencies:
        {', '.join([f"{k}: {v}" for k, v in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)])}
        
        Robustness aspects:
        """
        
        for result in robustness_results:
            robustness_prompt += f"\n\n{result['query']}:\n{result['analysis']}"
        
        robustness_prompt += f"""
        
        Based on the above information, provide a detailed evaluation of the robustness for this {research_type} paper. Your evaluation must:
        
        1. Be extremely critical and rigorous
        2. Identify specific robustness weaknesses and limitations
        3. Assess whether the evidence presented is sufficient and reliable
        4. Evaluate whether appropriate measures were taken to ensure validity
        5. Consider whether the claims made are justified by the evidence presented
        
        For each of the following criteria, provide a detailed assessment:
        
        {chr(10).join([f"- {criterion}" for criterion in robustness_criteria])}
        
        Structure your evaluation with clear sections, highlighting both strengths and weaknesses. Be specific and precise in your critique, avoiding generic statements. If the paper makes claims without sufficient evidence or validation, explicitly identify these issues.
        
        IMPORTANT: Be particularly critical of sample sizes, number of trials/runs, statistical validity, and generalizability claims. If the paper only conducted a few trials or has a small sample size, explicitly state that this is insufficient for statistical validity.
        """
        
        final_evaluation = analyze_with_openai("Evaluate robustness", robustness_prompt)
        return final_evaluation
    
    def evaluate_significance(self):
        """
        Evaluate the significance and innovation of the paper.
        
        Returns:
            str: Detailed evaluation of the paper's significance
        """
        # Get research type and tailored criteria
        research_type = self.research_classification["research_type"]
        type_description = self.evaluation_criteria["type_description"]
        significance_criteria = self.evaluation_criteria["significance_criteria"]
        
        # Generate custom queries based on research type
        significance_queries = [
            "What is the main contribution of this paper?",
            "How does this paper advance the field?",
            "What novel ideas or approaches does this paper introduce?",
            "What is the potential impact of this research?",
            "How does this paper compare to related work?"
        ]
        
        # Add research-type specific queries
        if research_type == "theoretical":
            significance_queries.extend([
                "How does this theory extend or challenge existing theoretical frameworks?",
                "What new conceptual insights does this paper provide?",
                "How might this theoretical contribution influence future research?"
            ])
        elif research_type == "design_science":
            significance_queries.extend([
                "What practical problem does this research address?",
                "How does the proposed artifact or solution improve upon existing approaches?",
                "What evidence is provided for the utility of the proposed solution?"
            ])
        elif research_type == "whitepaper":
            significance_queries.extend([
                "What problem or opportunity does this whitepaper address?",
                "How does the proposed solution compare to existing alternatives?",
                "What potential applications or implications are discussed?"
            ])
        
        # Query for significance-related content
        significance_results = []
        for query in significance_queries:
            relevant_paragraphs = query_paper(self.semantic_index, query)
            prompt = generate_analysis_prompt(query, relevant_paragraphs)
            analysis = analyze_with_openai(query, prompt)
            significance_results.append({
                'query': query,
                'analysis': analysis
            })
        
        # Look for significance-specific keywords
        significance_keywords = ['contribution', 'novel', 'new', 'advance', 'improve', 'enhance', 'outperform', 'state-of-the-art', 'breakthrough', 'innovation', 'impact', 'important', 'significant', 'major', 'key', 'crucial', 'critical', 'essential', 'valuable', 'useful']
        
        keyword_counts = {}
        for keyword in significance_keywords:
            count = len(re.findall(r'\b' + keyword + r'\b', self.paper_text.lower()))
            keyword_counts[keyword] = count
        
        # Generate comprehensive significance evaluation
        significance_prompt = f"""
        Provide a comprehensive evaluation of the research significance and innovation in this paper.
        
        IMPORTANT: This paper has been classified as a {research_type} paper ({type_description}).
        
        For this type of research, the significance evaluation should focus on the following criteria:
        {', '.join([f"{i+1}. {criterion}" for i, criterion in enumerate(significance_criteria)])}
        
        Paper statistics:
        - Word count: {self.stats['word_count']}
        - Citation count: {self.stats['citation_count']}
        - Figure count: {self.stats['figure_count']}
        - Table count: {self.stats['table_count']}
        - Equation count: {self.stats['equation_count']}
        
        Significance keyword frequencies:
        {', '.join([f"{k}: {v}" for k, v in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)])}
        
        Significance aspects:
        """
        
        for result in significance_results:
            significance_prompt += f"\n\n{result['query']}:\n{result['analysis']}"
        
        significance_prompt += f"""
        
        Based on the above information, provide a detailed evaluation of the significance and innovation for this {research_type} paper. Your evaluation must:
        
        1. Be extremely critical and rigorous
        2. Identify the specific contributions and their importance
        3. Assess whether the claimed contributions are truly novel or incremental
        4. Evaluate the potential impact on theory and/or practice
        5. Consider whether the paper's significance claims are justified by the evidence presented
        
        For each of the following criteria, provide a detailed assessment:
        
        {chr(10).join([f"- {criterion}" for criterion in significance_criteria])}
        
        Structure your evaluation with clear sections, highlighting both strengths and weaknesses. Be specific and precise in your critique, avoiding generic statements. If the paper makes exaggerated claims about its significance or novelty, explicitly identify these issues.
        """
        
        final_evaluation = analyze_with_openai("Evaluate significance", significance_prompt)
        return final_evaluation
    
    def evaluate_comprehensive(self):
        """
        Perform a comprehensive evaluation of the paper covering all aspects.
        
        Returns:
            str: Comprehensive evaluation of the paper
        """
        # Get research type and classification details
        research_type = self.research_classification["research_type"]
        type_description = self.evaluation_criteria["type_description"]
        classification_rationale = self.research_classification["rationale"]
        
        # Get evaluations
        methodology_eval = self.evaluate_methodology()
        robustness_eval = self.evaluate_robustness()
        significance_eval = self.evaluate_significance()
        
        comprehensive_prompt = f"""
        Provide a comprehensive evaluation of this research paper based on the following detailed assessments:
        
        PAPER CLASSIFICATION:
        This paper has been classified as a {research_type} paper ({type_description}).
        Rationale for classification: {classification_rationale}
        
        METHODOLOGY EVALUATION:
        {methodology_eval}
        
        ROBUSTNESS EVALUATION:
        {robustness_eval}
        
        SIGNIFICANCE EVALUATION:
        {significance_eval}
        
        Paper statistics:
        - Word count: {self.stats['word_count']}
        - Citation count: {self.stats['citation_count']}
        - Figure count: {self.stats['figure_count']}
        - Table count: {self.stats['table_count']}
        - Equation count: {self.stats['equation_count']}
        - Most common terms: {', '.join([f"{term[0]} ({term[1]})" for term in self.stats['common_words'][:10]])}
        
        Based on all the above information, provide a final comprehensive evaluation of this {research_type} paper that:
        1. Summarizes the key strengths and weaknesses across all dimensions
        2. Provides an overall assessment of the paper's quality and contribution
        3. Offers constructive suggestions for improvement
        4. Concludes with a final verdict on the paper's merit
        
        Your evaluation must be extremely critical and rigorous, identifying specific issues and limitations. Be particularly critical of methodological flaws, insufficient evidence, exaggerated claims, and lack of statistical validity. Avoid generic statements and be specific in your critique.
        
        Structure your evaluation with clear sections and a final summary.
        """
        
        final_evaluation = analyze_with_openai("Comprehensive evaluation", comprehensive_prompt)
        return final_evaluation
