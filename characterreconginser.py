import pytesseract
from PIL import Image
import cv2
import re
import numpy as np
import os

def preprocess_image(image_path):
    """Preprocess image to improve OCR accuracy"""
    img = cv2.imread(image_path)
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
    
    temp_path = "temp_preprocessed.png"
    cv2.imwrite(temp_path, denoised)
    return temp_path

def post_process_text(text):
    """Fix specific OCR errors"""
    # Fix "S" to "5" in numbers
    text = re.sub(r'\bS(?=\d)', '5', text)
    # Fix "O" to "0" in numbers and dates
    text = re.sub(r'\bO(?=\d)', '0', text)
    return text

def post_process_field(value, field_type=None):
    """Clean up extracted field values"""
    if value:
        # Fix common OCR mistakes
        if field_type == 'gstin':
            # Ensure GSTIN format: 2 digits, 5 letters, 5 digits/letters, 1 digit, 1 letter
            value = value.replace('S', '5').replace('0', 'O').replace('Z', '7')
        elif field_type == 'invoice_number':
            value = value.replace('5', 'S')
        else:
            value = value.replace('S', '5').replace('O', '0')
        # Remove any remaining spaces and newlines
        value = value.strip()
        value = re.sub(r'(\d)\s+(\d)', r'\1\2', value)
    return value

# Function to extract text from image
def extract_text(image_path):
    # Preprocess the image
    preprocessed_path = preprocess_image(image_path)
    
    # Open image using PIL
    img = Image.open(preprocessed_path)
    
    # Run OCR using Tesseract with a custom configuration
    custom_config = r'--oem 3 --psm 6 -l eng'
    text = pytesseract.image_to_string(img, config=custom_config)
    
    # Clean up temporary file
    os.remove(preprocessed_path)
    
    return post_process_text(text)

# Function to extract relevant details from OCR text
def extract_invoice_details(text):
    details = {}
    
    # Regex patterns for extracting data
    gstin_pattern = r"GSTIN\s*[:\-]?\s*([A-Z0-9]{15})"
    invoice_number_pattern = r"Invoice\s*No\.?\s*[:\-]?\s*([A-Z0-9]+)"
    invoice_date_pattern = r"Invoice\s*Date\s*[:\-]?\s*([O0\d]{2}-[A-Za-z]{3}-\d{4})"
    taxable_amount_pattern = r"Taxable\s*(?:Amount|Value)\s*[:\-]?\s*[\d,\.\sO]+"
    total_tax_pattern = r"(?:Total\s*Tax|IGST)\s*[:\-]?\s*[\d,\.\sO]+"
    total_amount_pattern = r"Total\s*Amount\s*After\s*Tax\s*[₹X]?\s*[\d,\.\sO]+"
    table_row_pattern = r"^\d+\s*\|.*?(\d+\.\d+)\s+\d+\.\d+\]\s+\d+\.\d+\s+(\d+\.\d+)\s+\d+\.\d+"
    
    # Search for table totals once, to be used for fallbacks
    total_line = re.search(r"Total\s+\d+\.\d+\s+([\d,\.\sO]+)\s+[\d,\.\sO]+\s+([\d,\.\sO]+)", text)
    
    # Searching for patterns in the text
    details['GSTIN'] = re.search(gstin_pattern, text)
    details['Invoice Number'] = re.search(invoice_number_pattern, text)
    details['Invoice Date'] = re.search(invoice_date_pattern, text)
    
    # Find all taxable amounts, total tax, and total amounts
    taxable_amounts = re.findall(taxable_amount_pattern, text)
    total_taxes = re.findall(total_tax_pattern, text)
    total_amounts = re.findall(total_amount_pattern, text)
    
    # Extract Taxable Amount and Total Tax from table rows
    table_rows = re.findall(table_row_pattern, text, re.MULTILINE)
    if table_rows:
        # Sum the Taxable Value and IGST columns
        taxable_total = sum(float(row[0].replace(',', '')) for row in table_rows)
        tax_total = sum(float(row[1].replace(',', '')) for row in table_rows)
        details['Taxable Amount'] = str(taxable_total)
        details['Total Tax'] = str(tax_total)
    else:
        # Fallback: Use summary lines if available
        if taxable_amounts:
            taxable_amount = re.search(r'[\d,\.\sO]+$', taxable_amounts[-1])
            details['Taxable Amount'] = taxable_amount.group(0) if taxable_amount else None
        elif total_line:
            details['Taxable Amount'] = total_line.group(1)
        else:
            details['Taxable Amount'] = None
        
        if total_taxes:
            total_tax = re.search(r'[\d,\.\sO]+$', total_taxes[-1])
            details['Total Tax'] = total_tax.group(0) if total_tax else None
        elif total_line:
            details['Total Tax'] = total_line.group(2)
        else:
            details['Total Tax'] = None
    
    # Extract the last total amount
    if total_amounts:
        total_amount = re.search(r'[\d,\.\sO]+$', total_amounts[-1])
        details['Total Amount'] = total_amount.group(0) if total_amount else None
    else:
        # Fallback: Calculate from Taxable Amount + Total Tax if both are available
        if details.get('Taxable Amount') and details.get('Total Tax'):
            try:
                taxable = float(details['Taxable Amount'].replace(',', ''))
                tax = float(details['Total Tax'].replace(',', ''))
                details['Total Amount'] = str(taxable + tax)
            except (ValueError, TypeError):
                details['Total Amount'] = None
        else:
            details['Total Amount'] = None
    
    # Clean up the results
    for key in details:
        if details[key]:
            value = details[key].group(1) if hasattr(details[key], 'group') and key in ['GSTIN', 'Invoice Number', 'Invoice Date'] else details[key]
            details[key] = post_process_field(value, key)
        else:
            details[key] = None
    
    return details

# Function to process the image and extract relevant details
def process_invoice_image(image_path):
    # Extract text from image
    text = extract_text(image_path)
    print("Extracted text:\n", text)
    
    # Extract details from the text
    invoice_details = extract_invoice_details(text)
    
    print("\nExtracted Invoice Details:")
    for key, value in invoice_details.items():
        print(f"{key}: {value}")
    return invoice_details

# Example Usage
image_path = "/content/files/Template1-1.jpg"  # Replace with your image path
process_invoice_image(image_path)