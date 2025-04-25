import os
import pickle
import faiss
from pdf_utils import chunk_pdf_to_text
from embeddings import create_embeddings

class DocumentMemoryManager:
    def __init__(self, storage_dir="doc_memory"):
        """
        Initializes the DocumentMemoryManager.
        Args:
            storage_dir (str): Directory to store document chunks and FAISS index.
        """
        os.makedirs(storage_dir, exist_ok=True)
        self.registry_path = os.path.join(storage_dir, "registry.pkl")
        # registry: doc_id → {"chunks_file", "index_file", "dim"}
        self.registry = self._load_registry()

    def _load_registry(self):
        """
        Loads the registry from the storage directory.
        Returns:
            dict: The loaded registry.
        """
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_registry(self):
        """
        Saves the registry to the storage directory.
        """
        with open(self.registry_path, "wb") as f:
            pickle.dump(self.registry, f)

    def add_document(self, doc_id, pdf_path, chunk_size=1000):
        """
        Adds a document to the memory manager.
        Args:
            doc_id (str): Unique identifier for the document.
            pdf_path (str): Path to the PDF file.
            chunk_size (int): Size of each text chunk.
        """
        # 1. Chunk text
        chunks = chunk_pdf_to_text(pdf_path, chunk_size=chunk_size)
        # 2. Embed
        embeddings = create_embeddings(chunks)  # shape (n_chunks, dim)
        dim = embeddings.shape[1]
        # 3. Build FAISS index
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        # 4. Persist chunks and index
        chunks_file = os.path.join("doc_memory", f"{doc_id}_chunks.pkl")
        index_file  = os.path.join("doc_memory", f"{doc_id}_faiss.index")
        with open(chunks_file, "wb") as f:
            pickle.dump(chunks, f)
        faiss.write_index(index, index_file)
        # 5. Update registry
        self.registry[doc_id] = {
            "chunks_file": chunks_file,
            "index_file": index_file,
            "dim": dim
        }
        self._save_registry()

    def query(self, doc_id, user_query, k=1):
        """
        Queries the document memory manager for the top k chunks similar to the user query.
        Args:
            doc_id (str): Unique identifier for the document.
            user_query (str): The query string.
            k (int): Number of top chunks to return.
        Returns:
            list: The top k chunks similar to the user query.
        """
        # 1. Look up files
        entry = self.registry.get(doc_id)
        if not entry:
            raise KeyError(f"No document with id {doc_id}")
        # 2. Load chunks
        with open(entry["chunks_file"], "rb") as f:
            chunks = pickle.load(f)
        # 3. Load FAISS index
        index = faiss.read_index(entry["index_file"])
        # 4. Embed query
        q_emb = create_embeddings([user_query])
        # 5. Search
        D, I = index.search(q_emb, k)
        return [chunks[i] for i in I[0]]

# Usage:

# mgr = DocumentMemoryManager()
# mgr.add_document(doc_id="user_guide_v1", pdf_path="/Users/jyotbuch/Desktop/DocuSense-AI/backend/data/Offer Letter.pdf")
# top_chunks = mgr.query("user_guide_v1", "What is the internship location?", k=2)
# print(top_chunks)