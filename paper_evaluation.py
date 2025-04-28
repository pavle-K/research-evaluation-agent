import re
from collections import Counter
import numpy as np
from semantic_index import create_semantic_index
from query_paper import query_paper
from generate_prompt import generate_analysis_prompt
from call_llm import analyze_with_openai

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
        # Query for methodology-related content
        methodology_queries = [
            "What methods does this paper use?",
            "How is the experimental design structured?",
            "What data collection techniques are used?",
            "How does the paper analyze data?",
            "What are the limitations of the methodology?"
        ]
        
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
        
        methodology_prompt += """
        
        Based on the above information, provide a detailed evaluation of the methodology covering:
        1. Appropriateness of methods for the research question
        2. Experimental design quality
        3. Data collection and analysis techniques
        4. Methodological limitations and biases
        5. Overall assessment of methodological rigor
        
        Structure your evaluation with clear sections, highlighting both strengths and weaknesses.
        """
        
        final_evaluation = analyze_with_openai("Evaluate methodology", methodology_prompt)
        return final_evaluation
    
    def evaluate_robustness(self):
        """
        Evaluate the robustness of the paper.
        
        Returns:
            str: Detailed evaluation of the paper's robustness
        """
        # Query for robustness-related content
        robustness_queries = [
            "How reliable are the results in this paper?",
            "What statistical methods are used to ensure validity?",
            "How does the paper address potential confounding variables?",
            "What limitations or threats to validity are discussed?",
            "How generalizable are the findings of this paper?"
        ]
        
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
        
        robustness_prompt += """
        
        Based on the above information, provide a detailed evaluation of the robustness covering:
        1. Reliability and reproducibility of results
        2. Statistical significance and effect sizes
        3. Treatment of confounding variables and biases
        4. Generalizability of findings
        5. Overall assessment of research robustness
        
        Structure your evaluation with clear sections, highlighting both strengths and weaknesses.
        """
        
        final_evaluation = analyze_with_openai("Evaluate robustness", robustness_prompt)
        return final_evaluation
    
    def evaluate_significance(self):
        """
        Evaluate the significance and innovation of the paper.
        
        Returns:
            str: Detailed evaluation of the paper's significance
        """
        # Query for significance-related content
        significance_queries = [
            "What is the main contribution of this paper?",
            "How does this paper advance the field?",
            "What novel ideas or approaches does this paper introduce?",
            "What is the potential impact of this research?",
            "How does this paper compare to related work?"
        ]
        
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
        
        significance_prompt += """
        
        Based on the above information, provide a detailed evaluation of the significance and innovation covering:
        1. Importance of the research question in the field
        2. Novelty of approach or findings
        3. Advancement of knowledge in the field
        4. Potential impact on theory or practice
        5. Overall assessment of research significance
        
        Structure your evaluation with clear sections, highlighting both strengths and weaknesses.
        """
        
        final_evaluation = analyze_with_openai("Evaluate significance", significance_prompt)
        return final_evaluation
    
    def evaluate_comprehensive(self):
        """
        Perform a comprehensive evaluation of the paper covering all aspects.
        
        Returns:
            str: Comprehensive evaluation of the paper
        """
        methodology_eval = self.evaluate_methodology()
        robustness_eval = self.evaluate_robustness()
        significance_eval = self.evaluate_significance()
        
        comprehensive_prompt = f"""
        Provide a comprehensive evaluation of this research paper based on the following detailed assessments:
        
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
        
        Based on all the above information, provide a final comprehensive evaluation of the paper that:
        1. Summarizes the key strengths and weaknesses across all dimensions
        2. Provides an overall assessment of the paper's quality and contribution
        3. Offers constructive suggestions for improvement
        4. Concludes with a final verdict on the paper's merit
        
        Structure your evaluation with clear sections and a final summary.
        """
        
        final_evaluation = analyze_with_openai("Comprehensive evaluation", comprehensive_prompt)
        return final_evaluation
