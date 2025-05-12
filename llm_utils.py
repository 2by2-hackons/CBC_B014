import os
import uuid
import pdfplumber
import chromadb
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import pytesseract
from PIL import Image
import cv2
import re
import numpy as np

UPLOAD_FOLDER = "uploads"
AUDIT_PDF_PATH = "audit_report.pdf"
CHROMA_DIR = "vectorstore"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name="financial_audit")

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    groq_api_key="<YOUR_GROQ_API_KEY>"
)

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def save_to_chroma(file_paths):
    for path in file_paths:
        content = extract_text_from_pdf(path)
        collection.add(
            documents=[content],
            metadatas=[{"source": os.path.basename(path)}],
            ids=[str(uuid.uuid4())]
        )

def generate_audit_pdf():
    all_docs = collection.get()["documents"]
    combined = "\n\n".join(all_docs)

    prompt = PromptTemplate.from_template(
        """
        ### Financial Reports
        {content}

        ### Instruction:
        You are a financial auditor. Analyze the reports and return JSON with:
        - Q1 Summary
        - Q2 Summary
        - Discrepancies
        - Financial Health
        - Recommendations
        ## Valid JSON (no preamble)
        """
    )

    chain = prompt | llm
    response = chain.invoke({"content": combined})
    parser = JsonOutputParser()
    parsed = parser.parse(response.content)

    write_audit_to_pdf(parsed, AUDIT_PDF_PATH)
    return AUDIT_PDF_PATH

def write_audit_to_pdf(audit_data, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    y = height - 50

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Financial Audit Report")
    y -= 30

    for key, val in audit_data.items():
        c.setFont("Helvetica-Bold", 12)
        c.drawString(30, y, f"{key}:")
        y -= 20

        c.setFont("Helvetica", 10)
        for line in str(val).split("\n"):
            c.drawString(50, y, line)
            y -= 15
            if y < 60:
                c.showPage()
                y = height - 50
    c.save()

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)
    temp_path = "temp_preprocessed.png"
    cv2.imwrite(temp_path, denoised)
    return temp_path

def extract_text(image_path):
    preprocessed_path = preprocess_image(image_path)
    img = Image.open(preprocessed_path)
    custom_config = r'--oem 3 --psm 6 -l eng'
    text = pytesseract.image_to_string(img, config=custom_config)
    os.remove(preprocessed_path)
    return text

def post_process_field(value, field_type=None):
    if value:
        if field_type == 'gstin':
            if len(value) >= 2:
                value = value[:2].replace('O', '0') + value[2:]
        elif field_type == 'invoice_number':
            if re.match(r'G5T\d+', value):
                value = value.replace('5', 'S')
        elif field_type == 'invoice_date':
            value = value.replace('O', '0')
        else:
            value = value.replace('S', '5').replace('O', '0')
        value = value.strip()
        value = re.sub(r'(\d)\s+(\d)', r'\1\2', value)
    return value

def validate_gstin(gstin):
    pattern = r'^\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z0-9]$'
    return bool(re.match(pattern, gstin))

def extract_invoice_details(text):
    details = {}
    gstin_pattern = r"GSTIN\s*[:\-]?\s*([A-Z0-9]{15})"
    invoice_number_pattern = r"(?:Invoice\s*No\.?|Bill\s*No\.?)\s*[:\-]?\s*([A-Z0-9]+)"
    invoice_date_pattern = r"(?:Invoice\s*Date|Date|Bill\s*Date)\s*[:\-]?\s*([\d\-\sA-Za-z,]+)"
    amount_pattern = r"(?:Total\s*Amount|Amount\s*Due|Amount)\s*[:\-]?\s*([\d,\.]+)"

    gstins = re.findall(gstin_pattern, text, re.IGNORECASE)
    processed_gstins = [post_process_field(gstin, 'gstin') for gstin in gstins]
    valid_gstins = [gstin for gstin in processed_gstins if validate_gstin(gstin)]
    details['GSTIN'] = valid_gstins[0] if valid_gstins else (processed_gstins[0] if processed_gstins else None)

    invoice_number_match = re.search(invoice_number_pattern, text, re.IGNORECASE)
    details['Invoice Number'] = invoice_number_match.group(1) if invoice_number_match else None

    invoice_date_match = re.search(invoice_date_pattern, text, re.IGNORECASE)
    details['Invoice Date'] = invoice_date_match.group(1).strip() if invoice_date_match else None

    total_amount_match = re.search(amount_pattern, text, re.IGNORECASE)
    details['Total Amount'] = total_amount_match.group(1).replace(',', '').strip() if total_amount_match else None

    for key in details:
        field_type = key.lower().replace(' ', '_')
        if details[key]:
            details[key] = post_process_field(details[key], field_type)
        else:
            details[key] = None

    if not details['GSTIN'] and not details['Invoice Number']:
        return None

    return details

def process_invoice_image(image_path):
    text = extract_text(image_path)
    return extract_invoice_details(text)

def process_multiple_invoices(folder_path):
    invoice_list = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            invoice_details = process_invoice_image(image_path)
            if invoice_details:
                invoice_list.append(invoice_details)
    return invoice_list
