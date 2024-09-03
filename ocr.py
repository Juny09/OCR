import streamlit as st
from PIL import Image
import pytesseract
import numpy as np
import pandas as pd
from io import BytesIO
from docx import Document
import pyperclip
from googletrans import Translator
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

def extract_text(img_array, langs):
    # Extract text from the image with specified languages
    return pytesseract.image_to_string(img_array, lang='+'.join(langs))

def main():
    st.title('OCR')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_array = np.array(img)

        # Define OCR languages
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
        
        # Allow users to select multiple OCR languages
        selected_languages = st.multiselect("Select OCR Language(s)", options=list(ocr_languages.keys()), default=["English"])
        selected_lang_codes = [ocr_languages[lang] for lang in selected_languages]

        # Extract text from the image
        try:
            text = extract_text(img_array, selected_lang_codes)
        except pytesseract.TesseractError as e:
            st.error(f"An error occurred: {e}")
            return

        st.image(img, caption='Uploaded Image', use_column_width=True)

        st.subheader('Extracted Text')
        container = st.container()
        container.write(text)

        # Translation options
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
        container = st.container()
        container.write(translated_text)
        
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
