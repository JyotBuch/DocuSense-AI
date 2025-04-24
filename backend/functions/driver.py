from retrieval import get_top_k_chunks_from_pdf, get_top_k_chunks_from_pdf_with_similarity
from prompt_utils import query_ollama, format_prompt

pdf_path = "/Users/jyotbuch/Desktop/DocuSense-AI/backend/data/Offer Letter.pdf"
question = "What is the location of the internship?"

# top_chunks = get_top_k_chunks_from_pdf(pdf_path, question)
top_chunks = get_top_k_chunks_from_pdf_with_similarity(pdf_path, question, k=1)
prompt = format_prompt(top_chunks, question)
print(prompt)
answer = query_ollama(prompt)

print("\nAnswer:\n", answer)