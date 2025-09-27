from groq import Groq
import os
import streamlit as st
import re
from dotenv import load_dotenv

load_dotenv()

# Load API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class SimpleRAG:
    def __init__(self):
        self.chunks = []
        self.client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

    def store_chunks(self, text_chunks):
        """Store PDF text chunks."""
        self.chunks = text_chunks

    def find_relevant_text(self, query, top_k=3):
        """Find most relevant text chunks for the given query."""
        query_words = query.lower().split()
        scored = []
        for chunk in self.chunks:
            count = sum(word in chunk.lower() for word in query_words)
            if count > 0:
                scored.append((chunk, count))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]] or self.chunks[:top_k]

    def get_answer(self, query):
        """Generate an answer using LLaMA (via Groq API)."""
        chunks = self.find_relevant_text(query)
        context = "\n\n".join(chunks)

        prompt = (
            "You are an AI assistant that helps users understand PDF documents.\n"
            "Use the context below to answer the question clearly and in a conversational way.\n"
            "If the answer isn't found, say 'The document doesn't contain that information.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        # Use Groq (LLaMA)
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print("Groq API error:", e)

        # Manual fallback if Groq fails
        return self.manual_fallback(query, chunks)

    def manual_fallback(self, query, chunks):
        """Fallback if API fails."""
        text = " ".join(chunks)
        sentences = re.split(r'(?<=[.!?]) +', text)
        relevant = [
            s for s in sentences
            if any(word in s.lower() for word in query.lower().split())
        ]
        reply = "Here’s what I found:\n"
        for sent in relevant[:3]:
            reply += f"- {sent.strip()}\n"
        reply += "\nLet me know if you want a deeper explanation."
        return reply


