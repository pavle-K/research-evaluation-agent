import re
import numpy as np
import openai

def query_paper(index, question, top_k=5, embedding_model='text-embedding-ada-002'):
    """
    Given a question about the paper, returns the most relevant paragraphs.
    
    Args:
        index (dict): The searchable index created by create_semantic_index
        question (str): The question or query about the paper
        top_k (int): The number of paragraphs to return
        embedding_model (str): The OpenAI embedding model to use
        
    Returns:
        list: Relevant paragraphs with their metadata
    """
    
    # Create OpenAI client
    openai_client = openai.OpenAI()
    
    # Generate embedding for the question
    response = openai_client.embeddings.create(
        input=question,
        model=embedding_model
    )
    query_embedding = response.data[0].embedding
    
    # Calculate cosine similarities
    similarities = np.dot(index['embeddings'], query_embedding) / (
        np.linalg.norm(index['embeddings'], axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get the top_k most similar paragraphs
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Prepare the results
    results = []
    for i in top_indices:
        # Include similarity score with the result
        chunk = index['chunks'][i].copy()
        chunk['similarity'] = float(similarities[i])
        results.append(chunk)
    
    # Apply additional heuristics based on question type
    results = refine_results_by_question_type(question, results)
    
    return results

def refine_results_by_question_type(question, results):
    """
    Refines the search results based on the type of question asked.
    
    Args:
        question (str): The question asked
        results (list): The initial search results
        
    Returns:
        list: Refined search results
    """
    question_lower = question.lower()
    
    # For methodology questions, prioritize paragraphs that mention methods
    if any(term in question_lower for term in ['how did', 'method', 'approach', 'technique']):
        for result in results:
            if 'method' in result['content'].lower() or 'approach' in result['content'].lower():
                result['similarity'] *= 1.3  # Boost the score
    
    # For results questions, prioritize paragraphs with numerical content
    elif any(term in question_lower for term in ['result', 'finding', 'outcome', 'performance']):
        for result in results:
            # Check for numerical content
            if len(re.findall(r'\d+\.\d+|\d+%', result['content'])) > 2:
                result['similarity'] *= 1.3
    
    # For theoretical questions, prioritize paragraphs with equations
    elif any(term in question_lower for term in ['theory', 'concept', 'principle', 'mechanism']):
        for result in results:
            # Check for equation-like content
            if len(re.findall(r'[=><≥≤±×÷≈≠∝∞∫∑∏√]', result['content'])) > 2:
                result['similarity'] *= 1.3
    
    # Re-sort based on adjusted scores
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return results
