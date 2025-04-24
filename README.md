# 📄 DocuSense AI

DocuSense AI is a Flask-based application that allows users to ask natural language questions about documents (e.g., offer letters, leases, agreements). It uses Retrieval-Augmented Generation (RAG) with open-source models and runs locally using [Ollama](https://ollama.com) for LLM inference.

---

## 🚀 Features

- 📤 Upload a PDF
- 🔍 Ask questions about the document
- 🧠 Semantic chunk retrieval using FAISS and MiniLM embeddings
- 🤖 LLM-powered answer generation using Ollama + LLaMA 2
- ⚙️ Fully local and open-source (no API keys required)

---

## 🛠️ Tech Stack

- **Frontend**: HTML (Flask templating)
- **Backend**: Python (Flask)
- **LLM Inference**: Ollama + LLaMA 2
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector Search**: FAISS
- **PDF Parsing**: PyMuPDF
- **Tokenization**: tiktoken

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/docusense-ai.git
cd docusense-ai