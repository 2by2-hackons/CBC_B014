import os
from flask import Flask, request, jsonify, send_file
from ocr2 import compare_invoices_with_gstr2a
import pytesseract

# Importing from llm_utils
from llm_utils import UPLOAD_FOLDER, save_to_chroma, generate_audit_pdf

# Tesseract OCR Path (update as needed for deployment)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Flask App Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Invoice Comparison Route ===
@app.route('/upload_invoice', methods=['POST'])
def upload_invoices():
    if 'invoice_images' not in request.files or 'excel_file' not in request.files:
        return jsonify({"error": "Missing files"}), 400

    invoice_images = request.files.getlist('invoice_images')
    excel_file = request.files['excel_file']

    # Save uploaded files
    for file in invoice_images:
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    excel_path = os.path.join(UPLOAD_FOLDER, excel_file.filename)
    excel_file.save(excel_path)

    # Compare OCR invoice data with GSTR-2A Excel
    result = compare_invoices_with_gstr2a(
        invoice_folder=UPLOAD_FOLDER,
        gstr2a_excel_path=excel_path,
        endpoint_url=""
    )
    return jsonify(result)

# === Financial Audit Route ===
@app.route("/upload_audit", methods=["POST"])
def upload_audit():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    file_paths = []
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        file_paths.append(file_path)

    save_to_chroma(file_paths)
    pdf_path = generate_audit_pdf()
    return send_file(pdf_path, as_attachment=True)


# === Run App ===
if __name__ == "__main__":
    app.run(debug=True)
