from pdf_utils import extract_text_from_pdf
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import sys

def create_embeddings(text, model_name='all-MiniLM-L6-v2'):
    """
    Generate embeddings for text chunks extracted from a PDF file using a specified SentenceTransformer model.
    Args:
        pdf_path (str): The file path to the PDF document.
        model_name (str, optional): The name of the SentenceTransformer model to use for generating embeddings. 
                                    Defaults to 'all-MiniLM-L6-v2'.
    Returns:
        list: A list of embeddings, where each embedding corresponds to a text chunk from the PDF.
    """
    # Load the SentenceTransformer model
    model = SentenceTransformer(model_name)
    
    # # Generate embeddings for each chunk
    # embeddings = [model.encode(chunk) for chunk in text]

    if isinstance(text, str):
        text = [text]  # wrap single string

    embeddings = model.encode(text, convert_to_numpy=True)

    return embeddings