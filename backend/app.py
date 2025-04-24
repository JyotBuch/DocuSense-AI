from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

from functions.retrieval import get_top_k_chunks
from functions.pdf_utils import extract_text_from_pdf
from functions.prompt_utils import format_prompt
from functions.embeddings import create_embeddings
from functions.prompt_utils import query_ollama

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------- Flask Routes ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    error = None
    file_path = None

    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
        else:
            file_path = request.form.get('pdf_path')

        question = request.form.get('question')

        if not file_path or not question:
            error = "Please upload a file and enter a question."
        else:
            try:
                full_text = extract_text_from_pdf(file_path)
                top_chunks = get_top_k_chunks(full_text, question)
                prompt = format_prompt(top_chunks, question)
                answer = query_ollama(prompt)
            except Exception as e:
                error = str(e)

    return render_template('index.html', answer=answer, error=error)

if __name__ == '__main__':
    app.run(debug=True)