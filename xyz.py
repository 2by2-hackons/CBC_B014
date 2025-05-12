import os
import uuid
import pandas as pd
import streamlit as st
import pdfplumber
import chromadb
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
import pytesseract
from PIL import Image
import cv2
import re
import numpy as np
import platform
import tempfile
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from fpdf import FPDF
import fitz  # PyMuPDF
import time

# Configuration
UPLOAD_FOLDER = "uploads"
AUDIT_PDF_PATH = "audit_report.pdf"
CHROMA_DIR = "vectorstore"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    client.delete_collection(name="financial_audit")
except:
    pass
collection = client.get_or_create_collection(name="financial_audit")

# Initialize Groq LLM with environment variable for API key
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY", "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")
)

# Set Tesseract path
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "tesseract"

# === Helper Function for Safe File Deletion ===
def safe_remove(file_path, retries=5, delay=0.5):
    """Attempt to remove a file with retries to handle PermissionError."""
    for attempt in range(retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except PermissionError as e:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            st.warning(f"Failed to remove {file_path}: {str(e)}")
            return False
    return False

# === OCR and Invoice Processing Functions ===
def preprocess_image(image_path):
    """Preprocess image to improve OCR accuracy"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        # Resize for better OCR (scale up by 1.5x to avoid over-distortion)
        img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Use simple thresholding to binarize the image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Light denoising to preserve character integrity
        denoised = cv2.fastNlMeansDenoising(thresh, h=10)

        temp_path = os.path.join(tempfile.gettempdir(), "temp_preprocessed.png")
        cv2.imwrite(temp_path, denoised)
        return temp_path
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def extract_text(image_path):
    """Extract text from an image using OCR"""
    try:
        preprocessed_path = preprocess_image(image_path)
        if not preprocessed_path:
            return ""
        img = Image.open(preprocessed_path)
        custom_config = r'--oem 3 --psm 6 -l eng'
        text = pytesseract.image_to_string(img, config=custom_config)
        os.remove(preprocessed_path)
        return text
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""

def post_process_field(value, field_type=None):
    """Clean up extracted field values"""
    if value:
        # Fix common OCR mistakes based on field type
        if field_type == 'gstin':
            # GSTIN format: First 2 characters should be digits (state code)
            if len(value) >= 2:
                value = value[:2].replace('O', '0') + value[2:]
        elif field_type == 'invoice_number':
            # For invoice number, convert "5" to "S" only if it matches a GST pattern (e.g., G5T)
            if re.match(r'G5T\d+', value):
                value = value.replace('5', 'S')
        elif field_type == 'invoice_date':
            # For dates, fix "O" to "0"
            value = value.replace('O', '0')
        else:
            # For numeric fields (Taxable Amount, Total Tax, Total Amount)
            value = value.replace('S', '5').replace('O', '0')
        # Remove any remaining spaces and newlines
        value = value.strip()
        value = re.sub(r'(\d)\s+(\d)', r'\1\2', value)
    return value

def validate_gstin(gstin):
    """Validate GSTIN format: 2 digits, 5 letters, 4 digits, 1 letter, 1 digit, 1 digit/letter"""
    if len(gstin) != 15:
        return False
    # First 2 characters: digits (state code)
    if not re.match(r'^\d{2}', gstin):
        return False
    # Next 5 characters: letters (PAN)
    if not re.match(r'^\d{2}[A-Z]{5}', gstin):
        return False
    # Next 4 characters: digits (entity code)
    if not re.match(r'^\d{2}[A-Z]{5}\d{4}', gstin):
        return False
    # Next 1 character: letter (PAN holder type)
    if not re.match(r'^\d{2}[A-Z]{5}\d{4}[A-Z]', gstin):
        return False
    # Next 1 character: digit (state code check)
    if not re.match(r'^\d{2}[A-Z]{5}\d{4}[A-Z]\d', gstin):
        return False
    # Last 1 character: digit or letter (checksum)
    if not re.match(r'^\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z0-9]$', gstin):
        return False
    return True

def extract_invoice_details(text):
    details = {}

    # Regex patterns for extracting data
    gstin_pattern = r"GSTIN\s*[:\-]?\s*([A-Z0-9]{15})"
    invoice_number_pattern = r"(?:Invoice\s*No\.?|Bill\s*No\.?)(?!\s*to)\s*[:\-]?\s*([A-Z0-9]+)"
    generic_invoice_pattern = r"\b[A-Z0-9]{3,}\b"  # Fallback for standalone alphanumeric codes
    invoice_date_pattern = r"(?:Invoice\s*Date|Date|Bill\s*Date)\s*[:\-]?\s*([0\d]{1,2}-?[A-Za-z]{3}-?\d{4}|[A-Za-z]+\s*\d{1,2},\s*\d{4})"
    taxable_amount_pattern = r"(?:Taxable\s*(?:Amount|Value)|Total\s*(?:Amount\s*)?Before\s*Tax)\s*[:\-=]?\s*([\d,\.]+)(?=\s|$|[^\d,\.])"
    total_tax_pattern = r"(?:Total\s*Tax|IGST|TOTAL\s*TAX\s*AMOUNT)\s*[:\-]?\s*([\d,\.]+)"
    total_amount_pattern = r"(?:Total\s*Amount\s*After\s*Tax\s*[₹X]?|TOTAL\s*AMOUNT\s*[€¥]?|Total:)\s*([\d,\.]+)"
    table_row_pattern = r"^\d+\s*\|[^|]*?\b(\d+\.\d+)\s+\d+\.\d+[\]\)]?\s+\d+\.\d+\s+(\d+\.\d+)\s+\d+\.\d+"
    alt_table_row_pattern = r"^\d+\s+[A-Za-z\s\-\%]+(?:\d{3,4})\s+\d+\.\d+\s+[_\-\d\s]+\s+([\d,]+\.\d+)\s+([\d,]+\.\d+)\s+\d+\.\d+\s+[\d,]+\.\d+"

    # Search for table totals once, to be used for fallbacks
    total_line = re.search(r"Total\s+\d+\.\d+\s+([\d,\.]+)\s+[\d,\.]+", text, re.IGNORECASE)

    # Searching for patterns in the text
    gstins = re.findall(gstin_pattern, text, re.IGNORECASE)
    # Post-process GSTINs before validation
    processed_gstins = [post_process_field(gstin, 'gstin') for gstin in gstins]
    # Validate and select the first valid GSTIN (likely the issuer's)
    valid_gstins = [gstin for gstin in processed_gstins if validate_gstin(gstin)]
    details['GSTIN'] = valid_gstins[0] if valid_gstins else (processed_gstins[0] if processed_gstins else None)

    # Extract Invoice Number
    invoice_number_match = re.search(invoice_number_pattern, text, re.IGNORECASE)
    if invoice_number_match:
        details['Invoice Number'] = invoice_number_match.group(1)
    else:
        # Fallback: Look for a standalone alphanumeric code in the first 5 lines
        lines = text.split('\n')[:5]  # Limit to first 5 lines
        early_text = '\n'.join(lines)
        generic_match = re.search(generic_invoice_pattern, early_text)
        if generic_match:
            # Ensure it's not a known field we already tried
            candidate = generic_match.group(0)
            if candidate not in ['02', '234', 'B256', '05']:  # Exclude previously rejected values
                details['Invoice Number'] = candidate
            else:
                details['Invoice Number'] = None
        else:
            details['Invoice Number'] = None

    # Extract Invoice Date
    invoice_date_match = re.search(invoice_date_pattern, text, re.IGNORECASE)
    details['Invoice Date'] = invoice_date_match.group(1) if invoice_date_match else None

    # Find all taxable amounts, total taxes, and total amounts
    taxable_amounts = re.findall(taxable_amount_pattern, text, re.IGNORECASE)
    total_taxes = re.findall(total_tax_pattern, text, re.IGNORECASE)
    total_amounts = re.findall(total_amount_pattern, text, re.IGNORECASE)

    # Extract Taxable Amount and Total Tax from table rows (as a fallback)
    table_rows = re.findall(table_row_pattern, text, re.MULTILINE)
    if not table_rows:
        # Split the text into lines for better table row extraction
        lines = text.split('\n')
        table_rows = []
        for line in lines:
            match = re.match(alt_table_row_pattern, line)
            if match:
                table_rows.append((match.group(1), match.group(2)))

    # Prioritize summary for Taxable Amount if available
    if taxable_amounts:
        taxable_amount = taxable_amounts[-1]
        try:
            taxable_amount = float(taxable_amount.replace(',', '').strip())
            details['Taxable Amount'] = str(taxable_amount)
        except (ValueError, TypeError) as e:
            details['Taxable Amount'] = None
    elif table_rows:
        # Sum the Taxable Value column from table rows
        try:
            taxable_total = sum(float(row[0].replace(',', '').strip()) for row in table_rows)
            details['Taxable Amount'] = str(taxable_total)
        except (ValueError, TypeError) as e:
            details['Taxable Amount'] = None
    elif total_line:
        try:
            taxable_amount = float(total_line.group(1).replace(',', '').strip())
            details['Taxable Amount'] = str(taxable_amount)
        except (ValueError, TypeError) as e:
            details['Taxable Amount'] = None
    else:
        details['Taxable Amount'] = None

    # Extract Total Tax from table rows or summary
    if total_taxes:
        total_tax = total_taxes[-1]
        try:
            total_tax = float(total_tax.replace(',', '').strip())
            details['Total Tax'] = str(total_tax)
        except (ValueError, TypeError) as e:
            details['Total Tax'] = None
    elif table_rows:
        # Sum the IGST column from table rows
        try:
            tax_total = sum(float(row[1].replace(',', '').strip()) for row in table_rows)
            details['Total Tax'] = str(tax_total)
        except (ValueError, TypeError) as e:
            details['Total Tax'] = None
    elif total_line:
        try:
            total_tax = float(total_line.group(2).replace(',', '').strip())
            details['Total Tax'] = str(total_tax)
        except (ValueError, TypeError) as e:
            details['Total Tax'] = None
    else:
        details['Total Tax'] = None

    # Extract Total Amount if present
    if total_amounts:
        total_amount = total_amounts[-1]
        try:
            total_amount = float(total_amount.replace(',', '').strip())
            details['Total Amount'] = str(total_amount)
        except (ValueError, TypeError) as e:
            details['Total Amount'] = None
    else:
        details['Total Amount'] = None

    # If Total Amount is not found or is an empty string, compute it as Taxable Amount + Total Tax
    if not details.get('Total Amount') or details['Total Amount'].strip() == "":
        if details.get('Taxable Amount') and details.get('Total Tax'):
            try:
                taxable = float(details['Taxable Amount'].replace(',', ''))
                tax = float(details['Total Tax'].replace(',', ''))
                details['Total Amount'] = str(taxable + tax)
            except (ValueError, TypeError) as e:
                details['Total Amount'] = None
        else:
            details['Total Amount'] = None

    # Clean up the results with correct field types
    for key in details:
        field_type = key.lower().replace(' ', '_')
        if details[key]:
            value = details[key] if isinstance(details[key], str) else details[key].group(1)
            details[key] = post_process_field(value, field_type)
        else:
            details[key] = None

    # Check if critical fields are missing
    if not details['GSTIN'] and not details['Invoice Number']:
        return None

    return details

def process_invoice_image(image_path):
    """Process a single invoice image"""
    try:
        text = extract_text(image_path)
        details = extract_invoice_details(text)
        if details:
            details['filename'] = os.path.basename(image_path)
        return details
    except Exception as e:
        st.error(f"Error processing image {image_path}: {str(e)}")
        return None

def process_multiple_invoices(folder_path):
    """Process all images in a folder and return a list of dictionaries"""
    invoice_list = []
    try:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image_path = os.path.join(folder_path, filename)
                invoice_details = process_invoice_image(image_path)
                if invoice_details:
                    invoice_list.append(invoice_details)
        return invoice_list
    except Exception as e:
        st.error(f"Error processing invoices in folder: {str(e)}")
        return []

def compare_invoices_with_gstr2a(invoice_folder, gstr2a_excel_path):
    """Compare OCR-extracted invoice data with GSTR-2A Excel"""
    try:
        # Read Excel file
        df = pd.read_excel(gstr2a_excel_path)

        # Convert column names to snake_case
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace('[^0-9a-zA-Z]+', '_', regex=True)
            .str.strip('_')
        )

        # Define required columns and their possible aliases
        column_mappings = {
            'gstin': ['gstin', 'gst_no', 'gst_number', 'gst_id'],
            'invoice_number': ['invoice_number', 'invoice_no', 'bill_number', 'bill_no'],
            'taxable_value': ['taxable_value', 'taxable_amount', 'value_before_tax', 'base_amount'],
            'total_tax': ['total_tax', 'tax_amount', 'igst', 'cgst_sgst', 'tax'],
            'total_amount': ['total_amount', 'amount_after_tax', 'grand_total', 'total']
        }

        # Find matching columns
        found_columns = {}
        missing_columns = []
        for required_col, aliases in column_mappings.items():
            for alias in aliases:
                if alias in df.columns:
                    found_columns[required_col] = alias
                    break
            else:
                missing_columns.append(required_col)

        if missing_columns:
            error_msg = (
                f"Excel file is missing required columns: {missing_columns}. "
                "Please ensure the Excel file includes columns for GSTIN, Invoice Number, Taxable Value, Total Tax, and Total Amount. "
                "Possible column names include:\n" +
                "\n".join(f"- {col}: {', '.join(aliases)}" for col, aliases in column_mappings.items())
            )
            raise ValueError(error_msg)

        # Process invoices
        invoice_details = process_multiple_invoices(invoice_folder)
        if not invoice_details:
            return [{"error": "No valid invoice data extracted"}]

        mismatches = []
        for invoice in invoice_details:
            gstin = invoice.get('GSTIN')
            invoice_number = invoice.get('Invoice Number')
            taxable_amount = invoice.get('Taxable Amount')  # Matches OCR field
            total_tax = invoice.get('Total Tax')  # Matches OCR field
            total_amount = invoice.get('Total Amount')  # Matches OCR field
            filename = invoice.get('filename', 'unknown')

            if not gstin or not invoice_number:
                mismatches.append({
                    'filename': filename,
                    'error': f'Missing GSTIN or Invoice Number in OCR data {filename}'
                })
                continue

            # Find match on GSTIN and Invoice Number
            match = df[
                (df[found_columns['gstin']].str.upper() == str(gstin).upper()) &
                (df[found_columns['invoice_number']].astype(str).str.strip().str.upper() == str(invoice_number).strip().upper())
            ]

            if match.empty:
                mismatches.append({
                    'filename': filename,
                    'gstin': gstin,
                    'invoice_number': invoice_number,
                    'error': 'No matching record found in Excel'
                })
            else:
                row = match.iloc[0]
                differences = {}
                # Compare fields, converting to strings for consistency
                if str(row[found_columns['taxable_value']]) != str(taxable_amount):
                    differences['taxable_value'] = {'excel': str(row[found_columns['taxable_value']]), 'ocr': str(taxable_amount)}
                if str(row[found_columns['total_tax']]) != str(total_tax):
                    differences['total_tax'] = {'excel': str(row[found_columns['total_tax']]), 'ocr': str(total_tax)}
                if str(row[found_columns['total_amount']]) != str(total_amount):
                    differences['total_amount'] = {'excel': str(row[found_columns['total_amount']]), 'ocr': str(total_amount)}

                if differences:
                    mismatches.append({
                        'filename': filename,
                        'gstin': gstin,
                        'invoice_number': invoice_number,
                        'differences': differences
                    })

        # Clean up files
        for invoice in invoice_details:
            file_path = os.path.join(invoice_folder, invoice.get('filename', ''))
            safe_remove(file_path)
        safe_remove(gstr2a_excel_path)

        return mismatches
    except Exception as e:
        st.error(f"Error comparing invoices: {str(e)}")
        return [{"error": str(e)}]

# === Audit Generation Functions ===
def extract_text_from_pdf(file_path):
    """Extract text from a PDF file, returning page-wise content"""
    try:
        with pdfplumber.open(file_path) as pdf:
            pages = []
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page_num": page_num, "text": text})
            return pages
    except Exception as e:
        st.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        return []

def save_to_chroma(file_paths):
    """Save PDF content to ChromaDB, storing each page separately"""
    try:
        for path in file_paths:
            pages = extract_text_from_pdf(path)
            filename = os.path.basename(path)
            for page in pages:
                if page["text"]:
                    collection.add(
                        documents=[page["text"]],
                        metadatas=[{
                            "source": filename,
                            "page_num": page["page_num"]
                        }],
                        ids=[f"{filename}_page_{page['page_num']}_{uuid.uuid4()}"]
                    )
    except Exception as e:
        st.error(f"Error saving to Chroma: {str(e)}")

def generate_audit_pdf():
    """Generate audit report PDF using LLM with targeted queries"""
    try:
        # Define key metrics to query
        metrics = [
            "revenue", "net profit", "operating profit", "basic eps",
            "return on equity", "cash and cash equivalents", "total assets",
            "total liabilities", "gross profit", "tax rate"
        ]
        query_terms = metrics + ["Q1 2024", "Q2 2024", "June 30, 2024", "September 30, 2024"]

        # Query ChromaDB for relevant sections
        relevant_docs = []
        for term in query_terms:
            results = collection.query(
                query_texts=[term],
                n_results=5,
                include=["documents", "metadatas"]
            )
            for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                relevant_docs.append({
                    "text": doc,
                    "source": metadata["source"],
                    "page_num": metadata["page_num"]
                })

        # Deduplicate documents based on text content
        seen_texts = set()
        unique_docs = []
        for doc in relevant_docs:
            if doc["text"] not in seen_texts:
                seen_texts.add(doc["text"])
                unique_docs.append(doc)

        # Combine relevant documents, prioritizing Q1 and Q2
        combined_content = ""
        for doc in unique_docs:
            header = f"--- {doc['source']} (Page {doc['page_num']}) ---\n"
            combined_content += header + doc["text"] + "\n\n"

        if not combined_content.strip():
            st.error("No relevant financial data found in ChromaDB.")
            return None

        # Define a detailed prompt with escaped curly braces and comprehensive summary
        prompt = PromptTemplate.from_template(
            """
            ### Financial Reports (Q1 and Q2 2024)
            {content}

            ### Instruction:
            You are a financial auditor analyzing Q1 (June 30, 2024) and Q2 (September 30, 2024) financial reports. Generate a JSON audit report with the following sections:
            - **Q1_Summary**: Summarize key financial metrics for Q1, including revenue, net profit, operating profit, basic EPS, return on equity (ROE), cash and cash equivalents, total assets, total liabilities, gross profit, and effective tax rate. Provide values in INR (crores) and USD (millions) where available, as nested objects (e.g., {{ "INR": 39315, "USD": 4714 }}). Use numbers for all values.
            - **Q2_Summary**: Summarize the same metrics for Q2, following the same format.
            - **Comparison**: Compare Q1 and Q2, providing percentage growth or change for revenue, net profit, operating profit, basic EPS, ROE, and cash and cash equivalents. Use nested objects for monetary changes (e.g., {{ "INR": 4.3, "USD": 4.1 }}) and numbers for percentages.
            - **Discrepancies**: List any anomalies or inconsistencies (e.g., unexpected changes in tax rates, liabilities). Each discrepancy should be a dictionary with a descriptive key (e.g., "Tax Rate Increase") and a value as a number, string, or object (e.g., {{ "Q1": 29.3, "Q2": 29.6 }}). Do not include uncomputed expressions or comments.
            - **Financial_Health**: Assess the company's financial stability based on cash position, ROE, debt levels, and profitability. Use strings or objects to describe each aspect.
            - **Recommendations**: Provide a list of actionable suggestions as strings to improve financial performance or address discrepancies.
            - **Comprehensive_Summary**: Provide a concise narrative (string) summarizing the financial performance of Q1 and Q2, highlighting key trends, discrepancies, and financial health. This summary should be suitable for a Chartered Accountant to use in preparing a basic audit report. Do not repeat detailed metrics from other sections; focus on high-level insights.

            ### Requirements:
            - Use data from the provided reports only.
            - Include numerical values and percentages where applicable (e.g., compute changes like liabilities Q2 - Q1 and provide the result as a number).
            - Ensure the output is valid JSON with no preamble, comments, or uncomputed expressions (e.g., do not include "60601 - 60601" or "// comments").
            - If a metric or value is missing, use "Not available" for strings or null for numbers/objects.
            - Validate the JSON structure to ensure it is parsable.
            """
        )

        chain = prompt | llm
        response = chain.invoke({"content": combined_content})
        parser = JsonOutputParser()

        # Try parsing the JSON, with error handling
        try:
            parsed = parser.parse(response.content)
        except Exception as parse_error:
            st.error(f"Failed to parse LLM output as JSON: {str(parse_error)}")
            st.write("Raw LLM output for debugging:", response.content)
            return None

        # Generate PDF with improved multi-page handling
        c = canvas.Canvas(AUDIT_PDF_PATH, pagesize=letter)
        width, height = letter
        margin = 50
        max_width = width - 2 * margin  # Available width for text
        y = height - margin
        line_height = 15
        bottom_margin = 60

        def new_page():
            c.showPage()
            c.setFont("Helvetica", 10)
            return height - margin

        # Title
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, "Financial Audit Report")
        y -= 30

        c.setFont("Helvetica", 10)

        for key, val in parsed.items():
            # Section header
            c.setFont("Helvetica-Bold", 12)
            if y < bottom_margin:
                y = new_page()
            c.drawString(margin, y, f"{key.replace('_', ' ')}:")
            y -= line_height
            c.setFont("Helvetica", 10)

            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    text = f"{sub_key}: {sub_val}"
                    # Split text into lines that fit within max_width
                    lines = simpleSplit(text, "Helvetica", 10, max_width)
                    for line in lines:
                        if y < bottom_margin:
                            y = new_page()
                        c.drawString(margin + 20, y, line)
                        y -= line_height
            elif isinstance(val, list):
                for item in val:
                    text = str(item)
                    lines = simpleSplit(text, "Helvetica", 10, max_width)
                    for line in lines:
                        if y < bottom_margin:
                            y = new_page()
                        c.drawString(margin + 20, y, line)
                        y -= line_height
            else:
                # Handle strings (e.g., Comprehensive_Summary)
                text = str(val)
                lines = simpleSplit(text, "Helvetica", 10, max_width)
                for line in lines:
                    if y < bottom_margin:
                        y = new_page()
                    c.drawString(margin + 20, y, line)
                    y -= line_height

            # Add extra spacing after each section
            y -= line_height // 2

        c.save()
        return AUDIT_PDF_PATH
    except Exception as e:
        st.error(f"Error generating audit PDF: {str(e)}")
        return None

# === Insights Generation Functions ===
def generate_model_insights(excel_path):
    """Generate SHAP and LIME model insights from an Excel file and save as PDF."""
    try:
        # Load Excel file with context manager
        with pd.ExcelFile(excel_path) as excel_file:
            if "Profit & Loss" not in excel_file.sheet_names:
                st.error("Excel file must contain a 'Profit & Loss' sheet.")
                return None
            df = excel_file.parse("Profit & Loss")
        
        df = df.dropna()
        if not all(col in df.columns for col in ["year", "Price"]):
            st.error("Excel file must contain 'year' and 'Price' columns.")
            return None

        # Define features and target
        features = df.drop(columns=["year", "Price"])
        target = df["Price"]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Create plots directory
        plots_dir = os.path.join(UPLOAD_FOLDER, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # SHAP analysis
        explainer_shap = shap.TreeExplainer(model)
        shap_values = explainer_shap.shap_values(X_test)

        # SHAP Summary Plot
        shap_summary_path = os.path.join(plots_dir, "shap_summary.png")
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(shap_summary_path, bbox_inches='tight')
        plt.close()

        # LIME explainer
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            mode='regression'
        )
        lime_exp = explainer_lime.explain_instance(
            X_test.iloc[0].values,
            lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
        )

        # LIME Plot
        lime_plot_path = os.path.join(plots_dir, "lime_explanation.png")
        fig = lime_exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(lime_plot_path, bbox_inches='tight')
        plt.close()

        # Create PDF Report
        pdf_output_path = os.path.join(UPLOAD_FOLDER, f"model_insights_{uuid.uuid4()}.pdf")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Model Interpretation Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt="SHAP Summary Plot", ln=True)
        pdf.image(shap_summary_path, x=10, y=None, w=180)
        pdf.ln(10)
        pdf.cell(200, 10, txt="LIME Explanation (1st Test Instance)", ln=True)
        pdf.image(lime_plot_path, x=10, y=None, w=180)
        pdf.output(pdf_output_path)

        return pdf_output_path
    except Exception as e:
        st.error(f"Error generating model insights: {str(e)}")
        return None

def generate_insights_summary(input_pdf_path):
    """Generate AI summary of SHAP and LIME plots from a PDF."""
    try:
        # Setup for image extraction
        images_dir = os.path.join(UPLOAD_FOLDER, f"extracted_graphs_llm_{uuid.uuid4()}")
        os.makedirs(images_dir, exist_ok=True)
        image_paths = []

        # Extract images from PDF
        doc = fitz.open(input_pdf_path)
        for page_num in range(len(doc)):
            for img_index, img in enumerate(doc.get_page_images(page_num)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = os.path.join(images_dir, f"page{page_num+1}_img{img_index+1}.{image_ext}")
                with open(image_filename, "wb") as f:
                    f.write(image_bytes)
                image_paths.append(image_filename)
        doc.close()

        # Generate AI summaries
        output_pdf_path = os.path.join(UPLOAD_FOLDER, f"explanation_summary_llm_{uuid.uuid4()}.pdf")
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="AI-Generated Model Interpretation Summary", ln=True, align='C')
        pdf.ln(10)

        for image_path in image_paths:
            img = Image.open(image_path)
            ocr_text = pytesseract.image_to_string(img)
            image_name = os.path.basename(image_path)

            # AI summary using Groq
            prompt = PromptTemplate.from_template(
                """
                You are a Chartered Accountant reviewing a machine learning model interpretation plot (like SHAP or LIME) related to bill auditing.

                The plot text was extracted from an image titled: '{image_name}'.

                Instructions:
                1. Identify the type of interpretation plot (e.g., SHAP or LIME).
                2. Describe which features are having the highest positive or negative influence on the prediction.
                3. Highlight any unusual behavior or unexpected feature importance.
                4. Based on these insights, generate *audit-relevant questions* you would ask the client — e.g., "Why did the 'Quantity' suddenly increase this month?" or "What explains the large negative impact of 'Vendor Type'?"

                OCR Extract:
                {ocr_text}

                Respond with a paragraph summary, followed by a list of audit-relevant questions.
                """
            )
            chain = prompt | llm
            response = chain.invoke({"image_name": image_name, "ocr_text": ocr_text})
            summary = response.content.strip()

            # Add to PDF
            pdf.image(image_path, x=10, w=180)
            pdf.ln(5)
            pdf.multi_cell(0, 10, summary)
            pdf.ln(10)

        pdf.output(output_pdf_path)
        return output_pdf_path
    except Exception as e:
        st.error(f"Error generating insights summary: {str(e)}")
        return None

# === Streamlit App ===
st.title("Audit Automation App")

tab1, tab2, tab3 = st.tabs(["Invoice Comparison", "Financial Audit", "Insights"])

with tab1:
    st.header("Invoice Comparison")
    st.write("Upload an Excel file and invoice images to compare GSTIN, Invoice Number, Taxable Value, Total Tax, and Total Amount.")
    
    excel_file = st.file_uploader("Upload Excel File", type=["xls", "xlsx"], key="invoice_excel")
    invoice_images = st.file_uploader("Upload Invoice Images", type=["png", "jpg", "jpeg", "tiff", "bmp"], accept_multiple_files=True, key="invoice_images")
    
    if st.button("Compare Invoices"):
        if not excel_file or not invoice_images:
            st.error("Please upload both an Excel file and at least one invoice image.")
        else:
            with st.spinner("Processing..."):
                # Save files temporarily with unique names
                excel_path = os.path.join(UPLOAD_FOLDER, f"invoice_excel_{uuid.uuid4()}.xlsx")
                with open(excel_path, "wb") as f:
                    f.write(excel_file.getbuffer())
                
                image_paths = []
                for img in invoice_images:
                    img_path = os.path.join(UPLOAD_FOLDER, f"invoice_img_{uuid.uuid4()}{os.path.splitext(img.name)[1]}")
                    with open(img_path, "wb") as f:
                        f.write(img.getbuffer())
                    image_paths.append(img_path)
                
                # Compare invoices
                results = compare_invoices_with_gstr2a(UPLOAD_FOLDER, excel_path)
                
                # Display results
                if results and "error" in results[0]:
                    st.error(results[0]["error"])
                elif not results:
                    st.success("No mismatches found. All invoices matched the Excel data.")
                else:
                    st.warning("Mismatches found:")
                    for mismatch in results:
                        st.write(f"**File**: {mismatch.get('filename', 'Unknown')}")
                        if 'error' in mismatch:
                            st.write(f"**Error**: {mismatch['error']}")
                        else:
                            st.write(f"**GSTIN**: {mismatch.get('gstin', 'None')}")
                            st.write(f"**Invoice Number**: {mismatch.get('invoice_number', 'None')}")
                            st.write("**Differences**:")
                            for field, values in mismatch.get('differences', {}).items():
                                st.write(f"- {field}: Excel = {values['excel']}, OCR = {values['ocr']}")
                        st.markdown("---")

with tab2:
    st.header("Financial Audit")
    st.write("Upload financial reports (PDFs) to generate an audit report.")
    
    audit_files = st.file_uploader("Upload Financial Reports (PDFs)", type=["pdf"], accept_multiple_files=True, key="audit_pdfs")
    
    if st.button("Generate Audit Report"):
        if not audit_files:
            st.error("Please upload at least one PDF file.")
        else:
            with st.spinner("Generating audit report..."):
                # Save files temporarily with unique names
                file_paths = []
                for file in audit_files:
                    file_path = os.path.join(UPLOAD_FOLDER, f"audit_pdf_{uuid.uuid4()}{os.path.splitext(file.name)[1]}")
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    file_paths.append(file_path)
                
                # Generate audit
                save_to_chroma(file_paths)
                pdf_path = generate_audit_pdf()
                
                # Clean up uploaded files
                for file_path in file_paths:
                    safe_remove(file_path)
                
                # Provide download link
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Download Audit Report",
                            data=f,
                            file_name="audit_report.pdf",
                            mime="application/pdf"
                        )
                    safe_remove(pdf_path)
                else:
                    st.error("Failed to generate audit report.")

with tab3:
    st.header("Insights")
    st.write("Upload an Excel file with financial data (including a 'Profit & Loss' sheet with 'year' and 'Price' columns) to generate SHAP and LIME model insights and an AI-generated summary.")
    
    insights_excel = st.file_uploader("Upload Excel File", type=["xls", "xlsx"], key="insights_excel")
    
    if st.button("Generate Insights"):
        if not insights_excel:
            st.error("Please upload an Excel file.")
        else:
            with st.spinner("Generating insights..."):
                # Save Excel file with unique name
                excel_path = os.path.join(UPLOAD_FOLDER, f"insights_excel_{uuid.uuid4()}.xlsx")
                with open(excel_path, "wb") as f:
                    f.write(insights_excel.getbuffer())
                
                # Generate model insights PDF
                model_insights_pdf = generate_model_insights(excel_path)
                
                if model_insights_pdf:
                    # Generate AI summary PDF
                    summary_pdf = generate_insights_summary(model_insights_pdf)
                    
                    # Clean up temporary files
                    safe_remove(excel_path)
                    plots_dir = os.path.join(UPLOAD_FOLDER, "plots")
                    if os.path.exists(plots_dir):
                        for plot_file in os.listdir(plots_dir):
                            safe_remove(os.path.join(plots_dir, plot_file))
                        os.rmdir(plots_dir)
                    images_dir = os.path.join(UPLOAD_FOLDER, f"extracted_graphs_llm_{os.path.basename(model_insights_pdf).split('_')[2].split('.')[0]}")
                    if os.path.exists(images_dir):
                        for img_file in os.listdir(images_dir):
                            safe_remove(os.path.join(images_dir, img_file))
                        os.rmdir(images_dir)
                    safe_remove(model_insights_pdf)
                
                    # Provide download link for summary PDF
                    if summary_pdf and os.path.exists(summary_pdf):
                        with open(summary_pdf, "rb") as f:
                            st.download_button(
                                label="Download Insights Summary",
                                data=f,
                                file_name="explanation_summary_llm.pdf",
                                mime="application/pdf"
                            )
                        safe_remove(summary_pdf)
                    else:
                        st.error("Failed to generate insights summary.")
                else:
                    st.error("Failed to generate model insights.")