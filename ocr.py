

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from io import BytesIO
from docx import Document
import pyperclip
from googletrans import Translator
import easyocr
import platform

def extract_text(img_array, langs):
    # Extract text from the image using EasyOCR
    reader = easyocr.Reader(langs, gpu=False)
    results = reader.readtext(img_array, detail=0)  # detail=0 gives plain text
    return '\n'.join(results)

def main():
    st.title('OCR')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_array = np.array(img)

        # Define OCR languages for EasyOCR
        ocr_languages = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Chinese (Simplified)": "ch_sim",
            "Chinese (Traditional)": "ch_tra",
            "Japanese": "ja",
            "Korean": "ko",
            "Russian": "ru",
            "Portuguese": "pt",
            "Arabic": "ar"
        }

        # Allow users to select multiple OCR languages
        selected_languages = st.multiselect("Select OCR Language(s)", options=list(ocr_languages.keys()), default=["English"])
        selected_lang_codes = [ocr_languages[lang] for lang in selected_languages]

        # Extract text from the image
        try:
            text = extract_text(img_array, selected_lang_codes)
        except Exception as e:
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
        # Only enable clipboard if running locally (optional)
        if platform.system() != "Linux":  # Or use a better detection method for Streamlit Cloud
            if st.button("Copy Text to Clipboard"):
                try:
                    pyperclip.copy(translated_text)
                    st.success("Text copied to clipboard!")
                except pyperclip.PyperclipException as e:
                    st.warning(f"Clipboard not supported: {e}")
        else:
            st.info("Clipboard copy is not supported in this environment.")

if __name__ == "__main__":
    main()
