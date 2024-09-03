import streamlit as st
from PIL import Image
import pytesseract
import numpy as np
import pandas as pd
from io import BytesIO
from docx import Document
from transformers import pipeline
import requests

# Initialize Hugging Face text extraction pipeline
def initialize_pipeline():
    return pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def main():
    st.title('Text Extraction with Hugging Face')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the image
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        
        # Perform OCR using Tesseract
        text = pytesseract.image_to_string(img_array)

        # Perform text analysis using Hugging Face
        nlp_pipeline = initialize_pipeline()
        results = nlp_pipeline(text)
        
        # Display the image
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        # Display the extracted text
        st.subheader('Extracted Text')
        st.write(text)
        
        # Display NER results
        st.subheader('Named Entity Recognition Results')
        st.write(results)
        
        # Output format selection
        output_format = st.selectbox("Select Output Format", options=["Plain Text", "MS Word", "Excel"], index=0)
        
        # Create file for download
        if output_format == "Plain Text":
            st.download_button(
                label="Download as Text File",
                data=text,
                file_name="extracted_text.txt",
                mime="text/plain"
            )
        
        elif output_format == "MS Word":
            doc = Document()
            doc.add_paragraph(text)
            doc_io = BytesIO()
            doc.save(doc_io)
            doc_io.seek(0)
            st.download_button(
                label="Download as Word Document",
                data=doc_io,
                file_name="extracted_text.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        
        elif output_format == "Excel":
            df = pd.DataFrame([text], columns=["Extracted Text"])
            excel_io = BytesIO()
            df.to_excel(excel_io, index=False)
            excel_io.seek(0)
            st.download_button(
                label="Download as Excel File",
                data=excel_io,
                file_name="extracted_text.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Option to copy text to clipboard
        st.text_area("Extracted Text", text, height=200)
        st.button("Copy Text to Clipboard", on_click=lambda: st.text_area("Extracted Text", text).clipboard.copy())

if __name__ == "__main__":
    main()
