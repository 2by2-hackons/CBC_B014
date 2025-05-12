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
            print(f"Post-processed Invoice Number: {value}")  # Debug print
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

    return text

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
    print(f"GSTINs matched: {gstins}")  # Debug print
    # Post-process GSTINs before validation
    processed_gstins = [post_process_field(gstin, 'gstin') for gstin in gstins]
    print(f"Processed GSTINs: {processed_gstins}")  # Debug print
    # Validate and select the first valid GSTIN (likely the issuer's)
    valid_gstins = [gstin for gstin in processed_gstins if validate_gstin(gstin)]
    print(f"Valid GSTINs: {valid_gstins}")  # Debug print
    details['GSTIN'] = valid_gstins[0] if valid_gstins else (processed_gstins[0] if processed_gstins else None)

    # Extract Invoice Number
    invoice_number_match = re.search(invoice_number_pattern, text, re.IGNORECASE)
    print(f"Invoice Number match (primary pattern): {invoice_number_match}")  # Debug print
    if invoice_number_match:
        details['Invoice Number'] = invoice_number_match.group(1)
    else:
        # Fallback: Look for a standalone alphanumeric code in the first 5 lines
        lines = text.split('\n')[:5]  # Limit to first 5 lines
        early_text = '\n'.join(lines)
        generic_match = re.search(generic_invoice_pattern, early_text)
        print(f"Invoice Number match (generic pattern in first 5 lines): {generic_match}")  # Debug print
        if generic_match:
            # Ensure it's not a known field we already tried
            candidate = generic_match.group(0)
            if candidate not in ['02', '234', 'B256', '05']:  # Exclude previously rejected values
                details['Invoice Number'] = candidate
            else:
                details['Invoice Number'] = None
        else:
            details['Invoice Number'] = None

    details['Invoice Date'] = re.search(invoice_date_pattern, text, re.IGNORECASE)

    # Find all taxable amounts, total taxes, and total amounts
    taxable_amounts = re.findall(taxable_amount_pattern, text, re.IGNORECASE)
    print(f"Taxable amounts matched: {taxable_amounts}")  # Debug print

    total_taxes = re.findall(total_tax_pattern, text, re.IGNORECASE)
    print(f"Total taxes matched: {total_taxes}")  # Debug print

    total_amounts = re.findall(total_amount_pattern, text, re.IGNORECASE)
    print(f"Total amounts matched: {total_amounts}")  # Debug print

    # Extract Taxable Amount and Total Tax from table rows (as a fallback)
    table_rows = re.findall(table_row_pattern, text, re.MULTILINE)
    print(f"Table rows matched: {table_rows}")  # Debug print
    if not table_rows:
        # Split the text into lines for better table row extraction
        lines = text.split('\n')
        table_rows = []
        for line in lines:
            match = re.match(alt_table_row_pattern, line)
            if match:
                table_rows.append((match.group(1), match.group(2)))
        print(f"Table rows matched (alt pattern): {table_rows}")  # Debug print

    # Prioritize summary for Taxable Amount if available
    if taxable_amounts:
        taxable_amount = taxable_amounts[-1]
        try:
            taxable_amount = float(taxable_amount.replace(',', '').strip())
            details['Taxable Amount'] = str(taxable_amount)
        except (ValueError, TypeError) as e:
            print(f"Error parsing Taxable Amount from summary: {e}")
            details['Taxable Amount'] = None
    elif table_rows:
        # Sum the Taxable Value column from table rows
        try:
            taxable_total = sum(float(row[0].replace(',', '').strip()) for row in table_rows)
            details['Taxable Amount'] = str(taxable_total)
        except (ValueError, TypeError) as e:
            print(f"Error summing Taxable Amount from table rows: {e}")
            details['Taxable Amount'] = None
    elif total_line:
        try:
            taxable_amount = float(total_line.group(1).replace(',', '').strip())
            details['Taxable Amount'] = str(taxable_amount)
        except (ValueError, TypeError) as e:
            print(f"Error parsing Taxable Amount from total line: {e}")
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
            print(f"Error parsing Total Tax from summary: {e}")
            details['Total Tax'] = None
    elif table_rows:
        # Sum the IGST column from table rows
        try:
            tax_total = sum(float(row[1].replace(',', '').strip()) for row in table_rows)
            details['Total Tax'] = str(tax_total)
        except (ValueError, TypeError) as e:
            print(f"Error summing Total Tax from table rows: {e}")
            details['Total Tax'] = None
    elif total_line:
        try:
            total_tax = float(total_line.group(2).replace(',', '').strip())
            details['Total Tax'] = str(total_tax)
        except (ValueError, TypeError) as e:
            print(f"Error parsing Total Tax from total line: {e}")
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
            print(f"Error parsing Total Amount: {e}")
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
                print(f"Computed Total Amount: {details['Total Amount']}")
            except (ValueError, TypeError) as e:
                print(f"Error computing Total Amount: {e}")
                details['Total Amount'] = None
        else:
            details['Total Amount'] = None
            print("Could not compute Total Amount: Taxable Amount or Total Tax is missing")

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
        print("Warning: Critical fields (GSTIN and Invoice Number) are missing. Skipping invoice.")
        return None

    return details

# Function to process all images in a folder and return a list of dictionaries
def process_multiple_invoices(folder_path):
    invoice_list = []

    # Get all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            image_path = os.path.join(folder_path, filename)

            print(f"Processing {filename}...")
            invoice_details = process_invoice_image(image_path)
            if invoice_details:  # Only append if invoice details are valid
                invoice_list.append(invoice_details)

    return invoice_list

# Function to process a single invoice image
def process_invoice_image(image_path):
    # Extract text from image
    text = extract_text(image_path)
    print("Extracted text:\n", text)

    # Extract details from the text
    invoice_details = extract_invoice_details(text)

    if invoice_details:
        print("\nExtracted Invoice Details:")
        for key, value in invoice_details.items():
            print(f"{key}: {value}")
    return invoice_details

