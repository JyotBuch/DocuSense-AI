import os
import fitz  # PyMuPDF
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from memory import DocumentMemoryManager
from prompt_utils import format_prompt, query_ollama

# Paths
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
THUMB_FOLDER  = os.path.join(BASE_DIR, 'thumbnails')
TEMPLATES_FOLDER = os.path.join(BASE_DIR, 'templates')
MEMORY_FOLDER = os.path.join(BASE_DIR, 'doc_memory')

# Ensure directories exist
for d in (UPLOAD_FOLDER, THUMB_FOLDER, TEMPLATES_FOLDER):
    os.makedirs(d, exist_ok=True)

# Flask app
app = Flask(__name__, template_folder=TEMPLATES_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.environ.get('FLASK_SECRET', 'supersecret')

# Memory manager
doc_manager = DocumentMemoryManager(storage_dir=MEMORY_FOLDER)

@app.route('/')
def index():
    docs = list(doc_manager.registry.keys())
    return render_template('index.html', docs=docs)

@app.route('/upload', methods=['POST'])
def upload():
    pdf = request.files.get('pdf')
    if not pdf or not pdf.filename.lower().endswith('.pdf'):
        flash('Please upload a PDF document.', 'error')
        return redirect(url_for('index'))

    filename = pdf.filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf.save(save_path)

    # Generate thumbnail (first page)
    doc = fitz.open(save_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(0.2, 0.2))
    thumb_path = os.path.join(THUMB_FOLDER, f"{filename}.png")
    pix.save(thumb_path)
    doc.close()

    # Index document
    doc_manager.add_document(doc_id=filename, pdf_path=save_path)
    flash(f'Indexed "{filename}"', 'success')
    return redirect(url_for('index'))

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/thumbnails/<path:filename>')
def thumbnail_file(filename):
    return send_from_directory(THUMB_FOLDER, filename)

@app.route('/view/<doc_id>')
def view(doc_id):
    docs = list(doc_manager.registry.keys())
    file_url = url_for('uploaded_file', filename=doc_id)
    return render_template('view.html', docs=docs, selected=doc_id, file_url=file_url)

@app.route('/ask', methods=['POST'])
def ask():
    doc_id = request.form.get('doc_id')
    question = request.form.get('question')
    if not doc_id or not question:
        flash('Select a document and enter a question.', 'error')
        return redirect(url_for('index'))

    top_chunks = doc_manager.query(doc_id, question, k=1)
    prompt = format_prompt(top_chunks, question)
    answer = query_ollama(prompt)

    docs = list(doc_manager.registry.keys())
    file_url = url_for('uploaded_file', filename=doc_id)
    return render_template('view.html', docs=docs, selected=doc_id, file_url=file_url,
                           question=question, answer=answer)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)