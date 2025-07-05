# AI-eBay-User-Agreement-Chatbot
AI eBay User Agreement Chatbot: Your go-to RAG bot for instant, accurate answers on eBay policies. Leveraging TinyLlama, FAISS, and Streamlit, it makes complex info clear and accessible.
This project implements a Retrieval Augmented Generation (RAG) chatbot designed to provide instant and accurate answers based on the official eBay User Agreement document. Leveraging a combination of cutting-edge open-source tools, this chatbot aims to make complex policy information easily accessible and understandable.

✨ Features
Accurate Responses: Provides answers directly grounded in the eBay User Agreement, minimizing factual errors and hallucinations.

User-Friendly Interface: Built with Streamlit for an intuitive and interactive web application.

Efficient Retrieval: Utilizes a FAISS vector database for fast and relevant information retrieval.

Lightweight LLM: Integrates TinyLlama for efficient text generation, suitable for various environments.

⚙️ How it Works (Retrieval Augmented Generation - RAG)
The chatbot operates on the RAG principle, combining the strengths of a large language model (LLM) with a robust information retrieval system:

Data Ingestion & Chunking: The eBay User Agreement document is processed, split into smaller, manageable chunks of text.

Embedding Generation: Each text chunk is converted into a numerical vector (embedding) using a SentenceTransformer model.

Vector Database Indexing: These embeddings are stored and indexed in a FAISS vector database.

User Query: When a user asks a question, their query is also converted into an embedding.

Context Retrieval: The query embedding is used to search the FAISS index for the most semantically similar text chunks from the eBay User Agreement.

Augmented Generation: The retrieved relevant text chunks are then provided as context to the TinyLlama language model, along with the user's original question.

Answer Generation: The LLM generates a concise and factual answer based only on the provided context, ensuring relevance and accuracy derived directly from the document.

🛠️ Technologies Used
Language Model (LLM): TinyLlama/TinyLlama-1.1B-Chat-v1.0

Embedding Model: SentenceTransformers (all-MiniLM-L6-v2)

Vector Database: FAISS

Web Framework: Streamlit

Language: Python

Libraries: transformers, torch, numpy, json
