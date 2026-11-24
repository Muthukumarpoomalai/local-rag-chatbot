# 🤖 RAG Chatbot – Local Embeddings

## Overview
A Retrieval-Augmented Generation (RAG) chatbot built using Streamlit, LangChain, HuggingFace embeddings, FAISS and a local language model.

The application allows users to upload a `.txt` document or paste text directly into the application and ask questions based on the provided content.

## Features
- Upload text documents (.txt)
- Paste custom text
- Generate embeddings using Sentence Transformers
- Store and retrieve information using FAISS
- Retrieve top 3 relevant results
- Generate responses using a locally loaded DistilGPT-2 model
- Interactive Streamlit UI
- No external LLM API key required

## Tech Stack
- Python
- Streamlit
- LangChain
- HuggingFace Transformers
- Sentence Transformers
- FAISS

## Architecture
1. Upload or paste text.
2. Load the text into LangChain documents.
3. Generate embeddings using `all-MiniLM-L6-v2`.
4. Store embeddings in a FAISS vector database.
5. Retrieve the top 3 relevant results based on the user's query.
6. Pass the retrieved context to a local DistilGPT-2 model.
7. Display the generated response.

## Installation

```bash
git clone YOUR_REPOSITORY_URL
cd local-rag-chatbot
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Models Used
- Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- Language Model: `distilgpt2`

## Project Purpose
This project demonstrates the implementation of a basic Retrieval-Augmented Generation pipeline using local embedding generation, similarity search and local language model inference.

## Future Improvements
- PDF document support
- Text chunking
- Conversation memory
- Improved answer generation
- Source document display
- Better local LLM integration

---
Developed as a hands-on AI/ML project exploring Retrieval-Augmented Generation.
