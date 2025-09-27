import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv

# ✅ Load local environment variables (for local testing)
load_dotenv()

# ✅ Read Groq API key — works both locally and on Streamlit Cloud
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ Groq API endpoint and model (use active model)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"   # ✅ current recommended lightweight LLaMA model

# ----------------------------------------------------------- #
# 🧩 Simple RAG (Retrieval-Augmented Generation) Implementation
# ----------------------------------------------------------- #
class SimpleRAG:
    def __init__(self):
        self.chunks = []
        self.headers = {"Authorization": f"Bearer {GROQ_API_KEY}"} if GROQ_API_KEY else {}

    # ----------------------------------------------------------- #
    def store_chunks(self, text_chunks):
        """Store extracted PDF text chunks"""
        self.chunks = text_chunks

    # ----------------------------------------------------------- #
    def find_relevant_text(self, query, top_k=2):
        """Find top chunks related to the query"""
        query_words = query.lower().split()
        scored = []
        for chunk in self.chunks:
            count = sum(word in chunk.lower() for word in query_words)
            if count > 0:
                scored.append((chunk, count))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]] or self.chunks[:top_k]

    # ----------------------------------------------------------- #
    def llama_generate(self, prompt):
        """Generate response using Groq’s LLaMA API"""
        if not GROQ_API_KEY:
            return None

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant that answers based on the provided PDF context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 400
        }

        try:
            resp = requests.post(GROQ_API_URL, headers=self.headers, json=payload, timeout=30)
            resp.raise_for_status()
            output = resp.json()
            return output["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("Groq API error:", e)
            return None

    # ----------------------------------------------------------- #
    def manual_fallback(self, query, chunks):
        """Fallback: simple keyword-based extraction"""
        text = " ".join(chunks)
        sentences = re.split(r'(?<=[.!?]) +', text)
        relevant = [
            s for s in sentences
            if any(word in s.lower() for word in query.lower().split())
        ]

        if not relevant:
            relevant = sentences[:3]

        reply = "Here’s what I found:\n"
        for sent in relevant[:3]:
            reply += f"- {sent.strip()}\n"
        reply += "\nLet me know if you want a deeper explanation."
        return reply

    # ----------------------------------------------------------- #
    def get_answer(self, query):
        """Main RAG pipeline — retrieve + generate"""
        if not self.chunks:
            return "Please upload a PDF first."

        chunks = self.find_relevant_text(query)
        context = "\n\n".join(chunks)

        prompt = (
            "Use only the following PDF excerpts to answer the user's question clearly, briefly, "
            "and conversationally.\n\n"
            f"{context}\n\nQuestion: {query}\nAnswer:"
        )

        # 1️⃣ Try Groq LLaMA API
        llama_answer = self.llama_generate(prompt)
        if llama_answer:
            return llama_answer

        # 2️⃣ Fallback to manual extraction
        return self.manual_fallback(query, chunks)

