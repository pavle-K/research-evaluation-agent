import re
import openai
import numpy as np
from dotenv import load_dotenv

load_dotenv()

openai_client = openai.OpenAI()

def create_semantic_index(chunks, embedding_model='text-embedding-ada-002'):
    """
    Creates a searchable index using OpenAI embeddings.
    
    Args:
        chunks (list): List of chunk dictionaries with 'content' key
        embedding_model (str): The OpenAI embedding model to use
        
    Returns:
        dict: An index object that can be used for searching
    """
    
    # Extract content from chunks
    contents = [chunk['content'] for chunk in chunks]
    
    # Generate embeddings for each chunk
    embeddings = []
    for content in contents:
        response = openai_client.embeddings.create(
            input=content,
            model=embedding_model
        )
        embeddings.append(response.data[0].embedding)
    
    # Convert to numpy array for efficient similarity calculations
    embedding_matrix = np.array(embeddings)
    
    return {
        'chunks': chunks,
        'embeddings': embedding_matrix
    }

def query_index_with_embeddings(index, question, embedding_model='text-embedding-ada-002', top_k=3):
    """
    Queries the semantic index using embeddings.
    
    Args:
        index (dict): The semantic index
        question (str): The question to ask
        embedding_model (str): The OpenAI embedding model to use
        top_k (int): Number of results to return
        
    Returns:
        list: The most relevant chunks
    """
    
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
    
    # Get top K results
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Return relevant chunks
    return [index['chunks'][i] for i in top_indices]

def extract_paragraph_metadata(paragraph):
    """
    Extracts useful metadata from a paragraph to aid in retrieval.
    
    Args:
        paragraph (str): The paragraph text
        
    Returns:
        dict: Metadata about the paragraph
    """
    # Look for citations
    citations = len(re.findall(r'\[\d+\]|\(\w+\s+et\s+al\.', paragraph))
    
    # Look for mathematical content
    equations = len(re.findall(r'[=><≥≤±×÷≈≠∝∞∫∑∏√]', paragraph))
    
    # Check for numerical content (likely results)
    numbers = len(re.findall(r'\d+\.\d+|\d+%', paragraph))
    
    # Check for references to figures or tables
    figures = len(re.findall(r'[Ff]ig(?:ure)?\.?\s*\d+|[Tt]able\.?\s*\d+', paragraph))
    
    # Detect paragraph type based on content patterns
    paragraph_type = "general"
    if equations > 5:
        paragraph_type = "technical"
    elif citations > 3:
        paragraph_type = "literature_review"
    elif "we propose" in paragraph.lower() or "our approach" in paragraph.lower():
        paragraph_type = "contribution"
    elif "experiment" in paragraph.lower() and numbers > 3:
        paragraph_type = "results"
    
    return {
        'citations': citations,
        'equations': equations,
        'numerical_content': numbers,
        'figures': figures,
        'type': paragraph_type,
        'length': len(paragraph)
    }
