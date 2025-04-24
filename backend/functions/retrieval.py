from pdf_utils import chunk_pdf_to_text
from embeddings import create_embeddings
import faiss
import numpy as np

def get_top_k_chunks_from_pdf(pdf_path, user_query, k=1):
    # 1. Extract text from PDF and chunk it
    chunks = chunk_pdf_to_text(pdf_path)

    # 1. Fetch the embeddings
    chunk_embeddings = create_embeddings(chunks)

    # 2. Embed the query
    query_embedding = create_embeddings(user_query)

    # 3. Create in-memory FAISS index
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(chunk_embeddings)

    # 4. Search top-k relevant chunks
    distances, indices = index.search(query_embedding, k)
    top_chunks = [chunks[i] for i in indices[0]]

    return top_chunks

def cosine_similarity(a, b):
    """
    Computes the cosine similarity between the prompt and the knowledge base.

    Args:
        a (numpy.ndarray): First vector.
        b (numpy.ndarray): Second vector.
    
    Returns:
        float: Cosine similarity value.
    """

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

def get_top_k_chunks_from_pdf_with_similarity(pdf_path, user_query, k=1):

    """
    Retrieves the top-k chunks from a PDF based on cosine similarity with the user query.
    
    Args:
        pdf_path (str): The path to the PDF file.
        user_query (str): The user's query.
        k (int): The number of top chunks to retrieve.
        
    Returns:
    list: A list of the top-k chunks from the PDF.
    """

    # 1. Extract text from PDF and chunk it
    chunks = chunk_pdf_to_text(pdf_path)

    # 2. Fetch the embeddings
    chunk_embeddings = create_embeddings(chunks)

    # 3. Embed the query
    query_embedding = create_embeddings(user_query)

    # 4. Compute cosine similarity for each chunk
    similarities = [cosine_similarity(query_embedding, chunk) for chunk in chunk_embeddings]

    # 5. Get top-k chunks based on similarity
    top_k_indices = np.argsort(similarities)[-k:]
    top_chunks = [chunks[int(i)] for i in top_k_indices]

    print("Top-k chunks based on cosine similarity:")
    for i, chunk in enumerate(top_chunks):
        print(f"Chunk {i+1}: {chunk}")

    return top_chunks