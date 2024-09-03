import streamlit as st
from PIL import Image
import pytesseract
import numpy as np
import pandas as pd
from io import BytesIO
from docx import Document
import pyperclip
from googletrans import Translator
import cv2

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust the path if necessary

def extract_table(text):
    # Simple heuristic to convert text into a DataFrame
    rows = text.split('\n')
    table = [row.split('\t') for row in rows if row.strip() != '']
    return pd.DataFrame(table)

def detect_table_cells(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Define kernel sizes for detecting lines
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    
    # Detect horizontal and vertical lines
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_horizontal)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_vertical)
    
    # Find contours for lines
    contours_horizontal, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_vertical, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract table cells based on detected lines
    cells = extract_table_cells_from_contours(contours_horizontal, contours_vertical)
    
    return cells

def extract_table_cells_from_contours(horizontal_contours, vertical_contours, img_shape):
    cells = []
    for h in range(len(horizontal_contours) - 1):
        for v in range(len(vertical_contours) - 1):
            x1, y1, w1, h1 = cv2.boundingRect(vertical_contours[v])
            x2, y2, w2, h2 = cv2.boundingRect(horizontal_contours[h])
            x3, y3, w3, h3 = cv2.boundingRect(vertical_contours[v + 1])
            x4, y4, w4, h4 = cv2.boundingRect(horizontal_contours[h + 1])
            
            # Ensure the coordinates are within the image bounds
            x1, y1 = max(x1, 0), max(y1, 0)
            x3, y3 = min(x3, img_shape[1]), min(y3, img_shape[0])
            x4, y4 = min(x4, img_shape[1]), min(y4, img_shape[0])
            
            cell = {
                'x': x1,
                'y': y2,
                'width': x3 - x1,
                'height': y4 - y2
            }
            cells.append(cell)
    return cells


def extract_text_from_cells(image, cells):
    texts = []
    img_height, img_width = image.shape[:2]
    for cell in cells:
        x, y, w, h = cell['x'], cell['y'], cell['width'], cell['height']
        # Ensure cropping is within image bounds
        x, y = max(x, 0), max(y, 0)
        w, h = min(w, img_width - x), min(h, img_height - y)
        
        cell_image = image[y:y+h, x:x+w]
        if cell_image.size == 0:
            texts.append("")
            continue
        
        text = pytesseract.image_to_string(cell_image, config='--psm 6')
        texts.append(text.strip())
    return texts


def main():
    st.title('OCR')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        
        # Detect table cells
        cells = detect_table_cells(img_array)
        
        # Extract text from detected cells
        cell_texts = extract_text_from_cells(img_array, cells)
        
        ocr_languages = {
            "English": "eng",
            "Spanish": "spa",
            "French": "fra",
            "German": "deu",
            "Italian": "ita",
            "Chinese (Simplified)": "chi_sim",
            "Chinese (Traditional)": "chi_tra",
            "Japanese": "jpn",
            "Korean": "kor",
            "Russian": "rus",
            "Portuguese": "por",
            "Arabic": "ara"
        }
        ocr_language = st.selectbox("Select OCR Language", options=list(ocr_languages.keys()), index=0)
        ocr_lang_code = ocr_languages[ocr_language]
        
        text = pytesseract.image_to_string(img_array, lang=ocr_lang_code)
        
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        st.subheader('Extracted Text')
        container = st.container(border=True)
        container.write(f"{text}")
        
        st.subheader('Extracted Table')
        cells = detect_table_cells(img_array)
        if cells:
            cell_texts = extract_text_from_cells(img_array, cells)
            table_df = pd.DataFrame(cell_texts)
            st.write(table_df)
        else:
            st.write("No table detected.")
        
        translate_languages = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Chinese (Simplified)": "zh-cn",
            "Chinese (Traditional)": "zh-tw",
            "Japanese": "ja",
            "Korean": "ko",
            "Russian": "ru",
            "Portuguese": "pt",
            "Arabic": "ar"
        }
        translate_language = st.selectbox("Select Translation Language", options=list(translate_languages.keys()), index=0)
        translate_lang_code = translate_languages[translate_language]

        # Translate text
        if translate_lang_code != "en":  # No need to translate if it's already in English
            translator = Translator()
            translated_text = translator.translate(text, dest=translate_lang_code).text
        else:
            translated_text = text
        
        st.subheader('Translated Text')
        container = st.container(border=True)
        container.write(f"{translated_text}")
        
        output_format = st.selectbox("Select Output Format", options=["Plain Text", "MS Word", "Excel"], index=0)
        
        # Create file for download
        if output_format == "Plain Text":
            st.download_button(
                label="Download as Text File",
                data=translated_text,
                file_name="translated_text.txt",
                mime="text/plain"
            )
        
        elif output_format == "MS Word":
            doc = Document()
            doc.add_paragraph(translated_text)
            doc_io = BytesIO()
            doc.save(doc_io)
            doc_io.seek(0)
            st.download_button(
                label="Download as Word Document",
                data=doc_io,
                file_name="translated_text.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        
        elif output_format == "Excel":
            df = pd.DataFrame([translated_text], columns=["Translated Text"])
            excel_io = BytesIO()
            df.to_excel(excel_io, index=False)
            excel_io.seek(0)
            st.download_button(
                label="Download as Excel File",
                data=excel_io,
                file_name="translated_text.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        st.text_area("Extracted Text", translated_text, height=200)
        if st.button("Copy Text to Clipboard"):
            pyperclip.copy(translated_text)
            st.success("Text copied to clipboard!")

if __name__ == "__main__":
    main()
