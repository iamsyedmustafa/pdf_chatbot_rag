import streamlit as st
from pdf import extract_text_from_pdf, split_text_into_chunks
from rag import SimpleRAG

# Set page title
st.title("PDF Chatbot with RAG")
st.write("Upload a PDF and ask questions about it!")

# Initialize RAG system
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = SimpleRAG()
    st.session_state.pdf_processed = False
    st.session_state.chat_history = []

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if not st.session_state.pdf_processed:
        with st.spinner("Processing PDF..."):
            # Extract text from PDF
            text = extract_text_from_pdf(uploaded_file)
            
            if text:
                # Split into chunks
                chunks = split_text_into_chunks(text, chunk_size=300)
                
                # Store in RAG system
                st.session_state.rag_system.store_chunks(chunks)
                st.session_state.pdf_processed = True
                
                st.success(f"✅ PDF processed! Found {len(chunks)} text chunks.")
                st.info(f"📄 File: {uploaded_file.name}")
            else:
                st.error("Could not extract text from PDF")

# Chat interface
if st.session_state.pdf_processed:
    st.write("### Ask questions about your PDF:")
    
    # User input
    user_question = st.text_input("Your question:", placeholder="e.g., What is this document about?")
    
    if user_question:
        with st.spinner("Finding answer..."):
            answer = st.session_state.rag_system.get_answer(user_question)
            
            # Add to chat history (bonus feature)
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": answer
            })
            
            st.write("**Answer:**")
            st.write(answer)
    
    # Show chat history (bonus feature)
    if st.session_state.chat_history:
        st.write("### Previous Questions:")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):  # Show last 3
            with st.expander(f"Q: {chat['question'][:50]}..."):
                st.write(f"**Question:** {chat['question']}")
                st.write(f"**Answer:** {chat['answer']}")

# Reset button
if st.button("Upload New PDF"):
    st.session_state.pdf_processed = False
    st.session_state.rag_system = SimpleRAG()
    st.session_state.chat_history = []
    st.rerun()
