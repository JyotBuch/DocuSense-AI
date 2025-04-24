import subprocess

def query_ollama(prompt, model='llama2'):

    """
    Queries the LLM using the Ollama CLI.
    Args:
        prompt (str): The prompt to send to the LLM.
        model (str): The model to use for querying. Default is 'llama2'.
    Returns:
        str: The response from the LLM.
    """

    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode()

def format_prompt(top_chunks, question):

    """
    Formats the prompt for the LLM by combining the top-k chunks and the user question.
    
    Args:
        top_chunks (list): List of top-k chunks from the PDF.
        question (str): User's question.
    
    Returns:
        str: Formatted prompt for the LLM.
    """
    
    context = "\n\n".join(top_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return prompt