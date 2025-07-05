import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter  # NEW IMPORT

print("--- prepare_data.py script started ---")

def prepare_data():
    print("Starting data preparation for RAG chatbot...")

    data_dir = "data"
    raw_data_path = os.path.join(data_dir, "ebay_user_agreement.txt")
    vectordb_dir = "vectordb"
    chunks_dir = "chunks"

    os.makedirs(vectordb_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)

    print(f"Looking for raw data at: {raw_data_path}")

    # Read the entire text file
    with open(raw_data_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # --- UPDATED CHUNKING STRATEGY ---
    # Initialize RecursiveCharacterTextSplitter
    # This splits by characters, trying different separators to keep chunks meaningful,
    # and includes an overlap to maintain context between chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Each chunk will aim for 500 characters
        chunk_overlap=50, # 50 characters overlap between chunks
        length_function=len,
        is_separator_regex=False,
    )
    processed_chunks = text_splitter.split_text(full_text)
    # Remove any empty chunks that might result from splitting
    processed_chunks = [chunk.strip() for chunk in processed_chunks if chunk.strip()]
    # ---------------------------------

    print(f"Loaded {len(processed_chunks)} raw chunks from the knowledge base.")

    chunks_json_path = os.path.join(chunks_dir, "ebay_chunks.json")
    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(processed_chunks, f, ensure_ascii=False, indent=4)
    print(f"Saving chunks to {chunks_json_path}")

    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Encoding {len(processed_chunks)} chunks into embeddings...")
    embeddings = embedding_model.encode(processed_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    faiss_index_path = os.path.join(vectordb_dir, "faiss.index")
    embeddings_npy_path = os.path.join(vectordb_dir, "embeddings.npy")

    print(f"Creating FAISS index with dimension: {embeddings.shape[1]}")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    print(f"Saving FAISS index to {faiss_index_path}")
    faiss.write_index(index, faiss_index_path)

    print(f"Saving embeddings to {embeddings_npy_path}")
    np.save(embeddings_npy_path, embeddings)

    print(f"Successfully processed {len(processed_chunks)} chunks and stored data.")
    print("Data preparation complete. You can now run 'streamlit run app.py'.")

if __name__ == "__main__":
    try:
        prepare_data()
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        import traceback
        traceback.print_exc()
