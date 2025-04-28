import openai
import time

def get_system_prompt():
    """
    Returns the system prompt for the OpenAI model to analyze research papers.
    """
    system_prompt = """
    You are ResearchGPT, an AI specialized in analyzing and explaining research papers.
    
    Your task is to answer questions about the provided research paper excerpts accurately and precisely.
    
    Guidelines:
    - Base your answers strictly on the provided excerpts from the paper
    - If the excerpts don't contain the information needed to answer the question, acknowledge this limitation
    - Be precise and technical in your explanations
    - When discussing methodologies, be specific about the techniques used
    - When discussing results, include relevant numbers and metrics from the paper
    - When asked to evaluate or assess aspects of the paper, use standard academic evaluation criteria
    - Maintain a scholarly tone while remaining clear and accessible
    - Do not fabricate information that isn't in the provided excerpts
    - If mathematical equations or technical details are relevant, explain them clearly
    - Use appropriate academic terminology for the field of the paper
    
    The user's question and relevant excerpts from the paper will be provided. Focus on delivering a thorough analysis based solely on this information.
    """
    return system_prompt




def analyze_with_openai(question, 
                        context, 
                        model="gpt-4-turbo", 
                        max_retries=3):
    """
    Analyzes the paper context with OpenAI to answer the given question.
    
    Args:
        question (str): The question about the paper
        context (str): The context from the paper, including relevant excerpts
        api_key (str, optional): OpenAI API key. If not provided, uses environment variable
        model (str): The model to use for analysis
        max_retries (int): Maximum number of retries on rate limit or server errors
        
    Returns:
        str: The analysis result
    """
    system_prompt = get_system_prompt()
    
    # Create OpenAI client (uses OPENAI_API_KEY environment variable)
    client = openai.OpenAI()
    
    # Try multiple times in case of rate limiting or server errors
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=1500,  # Adjust based on expected answer length
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Extract the response
            analysis = response.choices[0].message.content.strip()
            return analysis
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"API error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                raise
    
    raise Exception("Failed to get response after multiple retries")
