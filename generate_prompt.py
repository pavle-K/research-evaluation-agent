

def generate_analysis_prompt(question, relevant_paragraphs, paper_overview=None):
    """
    Creates a prompt for the LLM that includes the question and relevant context.
    
    Args:
        question (str): The question about the paper
        relevant_paragraphs (list): Relevant paragraphs from the paper
        paper_overview (str, optional): A brief overview of the paper
        
    Returns:
        str: A formatted prompt for the LLM
    """
    prompt = "You are analyzing a research paper. "
    
    # Add paper overview if available
    if paper_overview:
        prompt += f"The paper is about: {paper_overview}\n\n"
    
    prompt += "Based on the following excerpts from the paper:\n\n"
    
    # Add the relevant paragraphs with their title context
    for i, para in enumerate(relevant_paragraphs, 1):
        prompt += f"[Excerpt {i} from {para['title']}]\n{para['content']}\n\n"
    
    # Add the question
    prompt += f"Question: {question}\n\n"
    
    # Add instructions for answering
    prompt += "Please provide a detailed and accurate answer based strictly on the provided excerpts. "
    prompt += "If the information needed is not contained in the excerpts, acknowledge this limitation rather than speculating."
    
    return prompt
