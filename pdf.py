import PyPDF2
import streamlit as st

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        # Extract text from each page
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def split_text_into_chunks(text, chunk_size=500):
    """Split text into smaller chunks for better retrieval"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks
