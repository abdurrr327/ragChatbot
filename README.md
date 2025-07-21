# RAG Chatbot

## Objective

Build a conversational chatbot using LangChain and RAG to retrieve answers from a custom corpus with context memory, deployed via Streamlit in Google Colab.

Methodology

Created a sample corpus with AI/ML-related documents.
Used sentence-transformers/all-MiniLM-L6-v2 for embeddings and FAISS for vector search.
Implemented RAG with GPT-2 and ConversationBufferMemory in LangChain.
Deployed using Streamlit with ngrok in Google Colab.
Key Results

Retrieves relevant document content for queries (e.g., "What are Transformers?").
Maintains conversational context for follow-ups.
Streamlit app enables interactive queries via public URL.
