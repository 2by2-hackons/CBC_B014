import cv2
import pytesseract
import re
import os
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
import requests

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image

def extract_text(image_path):
    image = preprocess_image(image_path)
    return pytesseract.image_to_string(image)

def extract_invoice_details(text):
    invoice_data = {
        "GSTIN": "",
        "Invoice Number": "",
        "Invoice Date": "",
        "Total Amount": 0.0,
        "Taxable Value": 0.0
    }

    gstin_match = re.search(r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b', text)
    if gstin_match:
        invoice_data["GSTIN"] = gstin_match.group(0)

    invoice_number_match = re.search(r'Invoice\s*Number[:\s]*([\w\-\/]+)', text, re.IGNORECASE)
    if invoice_number_match:
        invoice_data["Invoice Number"] = invoice_number_match.group(1)

    date_match = re.search(r'(\d{2}[/-]\d{2}[/-]\d{4})', text)
    if date_match:
        try:
            invoice_data["Invoice Date"] = datetime.strptime(date_match.group(1).replace('-', '/'), '%d/%m/%Y').strftime('%Y-%m-%d')
        except ValueError:
            pass

    amounts = re.findall(r'[\₹]?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text)
    cleaned_amounts = [float(a.replace(',', '')) for a in amounts if a]

    if cleaned_amounts:
        invoice_data["Total Amount"] = max(cleaned_amounts)
        if len(cleaned_amounts) > 1:
            invoice_data["Taxable Value"] = sorted(cleaned_amounts)[-2]

    return invoice_data

def compare_invoices_with_gstr2a(invoice_folder, gstr2a_excel_path, endpoint_url):
    gstr2a_df = pd.read_excel(gstr2a_excel_path)
    gstr2a_df.columns = gstr2a_df.columns.str.strip()

    extracted_data = []

    for filename in os.listdir(invoice_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(invoice_folder, filename)
            invoice_details = process_invoice_image(image_path)

            match = gstr2a_df[gstr2a_df['Invoice Number'].astype(str).str.strip() == invoice_details['Invoice Number']]

            comparison_result = {
                "File": filename,
                "OCR Extracted Data": invoice_details,
                "Matches GSTR2A": False,
                "Mismatch Reason": []
            }

            if not match.empty:
                match_row = match.iloc[0]

                def normalize_gstin(gstin): return gstin.strip().upper()
                def normalize_amount(val): return round(float(str(val).replace(',', '')), 2)

                if normalize_gstin(invoice_details["GSTIN"]) != normalize_gstin(str(match_row["GSTIN"])):
                    comparison_result["Mismatch Reason"].append("GSTIN mismatch")

                try:
                    ocr_date = datetime.strptime(invoice_details["Invoice Date"], '%Y-%m-%d').date()
                    gstr_date = pd.to_datetime(match_row["Invoice Date"]).date()
                    if ocr_date != gstr_date:
                        comparison_result["Mismatch Reason"].append("Invoice Date mismatch")
                except Exception as e:
                    comparison_result["Mismatch Reason"].append("Date parsing error")

                if abs(normalize_amount(invoice_details["Total Amount"]) - normalize_amount(match_row["Total Amount"])) > 1:
                    comparison_result["Mismatch Reason"].append("Total Amount mismatch")

                if abs(normalize_amount(invoice_details["Taxable Value"]) - normalize_amount(match_row["Taxable Value"])) > 1:
                    comparison_result["Mismatch Reason"].append("Taxable Value mismatch")

                if not comparison_result["Mismatch Reason"]:
                    comparison_result["Matches GSTR2A"] = True
            else:
                comparison_result["Mismatch Reason"].append("Invoice Number not found in GSTR2A")

            extracted_data.append(comparison_result)

    try:
        response = requests.post(endpoint_url, json=extracted_data)
        print("Data sent to Flask app, response:", response.text)
    except Exception as e:
        print("Error sending data to Flask app:", str(e))

    return extracted_data

# ✅ ADDED MISSING FUNCTION HERE
def process_invoice_image(image_path):
    text = extract_text(image_path)
    details = extract_invoice_details(text)
    return details
