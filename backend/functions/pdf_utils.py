from pymupdf import Document
import re
import tiktoken

def extract_text_from_pdf(pdf_path):
    """
    Extracts and returns text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    try:
        with Document(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()

                # Replace multiple tabs with a single space
                text = re.sub(r"\t+", " ", text)
                # Condense multiple spaces (including any new spaces created) to a single space
                text = re.sub(r"\s+", " ", text)   
            return text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")
    
def tokenize_text(text):
    """
    Tokenizes the input text into chunks of a specified size.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list: A list of tokenized chunks.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return tokens

def detokenize_text(tokens):
    """
    Detokenizes the input tokens back into text.

    Args:
        tokens (list): The list of tokenized chunks.

    Returns:
        str: The detokenized text.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return encoding.decode(tokens)

def chunk_tokens(tokens, chunk_size=500, overlap=0.2):
    """
    Splits the tokenized input into chunks of specified size with overlap.

    Args:
        tokens (list): The list of tokenized input.
        chunk_size (int): The size of each chunk.
        overlap (float): The fraction of overlap between chunks.

    Returns:
        list: A list of tokenized chunks with overlap.
    """
    step = int(chunk_size * (1 - overlap))
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), step) if i + chunk_size <= len(tokens)]

def chunk_pdf_to_text(pdf_path, chunk_size=1000):
    """
    Extracts text from a PDF and splits it into chunks of specified size.

    Args:
        pdf_path (str): The path to the PDF file.
        chunk_size (int): The size of each chunk.

    Returns:
        list: A list of text chunks.
    """
    text = extract_text_from_pdf(pdf_path)
    tokens = tokenize_text(text)
    chunks = chunk_tokens(tokens, chunk_size)
    return [detokenize_text(chunk) for chunk in chunks]

# def __main__():
#     # Example usage
#     pdf_path = "/Users/jyotbuch/Desktop/DocuSense-AI/backend/data/Offer Letter.pdf"
#     try:
#         text = extract_text_from_pdf(pdf_path)
#         print("Extracted Text:")
#         print(text)
#     except RuntimeError as e:
#         print(e)