import sentence_transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import faiss
import numpy as np
import torch

class RAGChatbot:
    def __init__(self):
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        print(f"Loading LLM model '{self.model_name}'...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto"
        )

        print(f"DEBUG: Tokenizer model_max_length: {self.tokenizer.model_max_length}")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print("Initializing HuggingFace pipeline for direct generation...")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=250,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            truncation=True,
            max_length=self.tokenizer.model_max_length
        )

        self.embedding_model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

        # --- PROMPT TEMPLATE CHANGE START ---
        # Simplified prompt for better generation from smaller models
        # The model should generate the answer directly after "Answer:"
        self.prompt_template_str = (
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        # --- PROMPT TEMPLATE CHANGE END ---

        self.index_path = "vectordb/faiss.index"
        self.embeddings_path = "vectordb/embeddings.npy"
        self.chunks_path = "chunks/ebay_chunks.json"

        print(f"Loading chunks from {self.chunks_path}...")
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        print(f"Successfully loaded {len(self.chunks)} chunks from JSON.")

        print(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(self.index_path)
        print(f"Successfully loaded FAISS index with dimension {self.index.d}.")


    def get_response(self, user_input):
        print(f"DEBUG: User input received: {user_input}")
        user_input = user_input.strip()

        user_embedding = self.embedding_model.encode([user_input]).astype("float32")
        print(f"DEBUG: User embedding created with shape {user_embedding.shape}")

        top_k = 7
        distances, indices = self.index.search(user_embedding, top_k)
        retrieved_chunks = [self.chunks[idx] for idx in indices[0]]

        context = "\n\n".join(retrieved_chunks)
        print(f"DEBUG: Retrieved Context (Top {top_k} chunks):\n---START CONTEXT---\n{context}\n---END CONTEXT---")

        full_prompt_text = self.prompt_template_str.format(question=user_input, context=context)

        input_ids = self.tokenizer.encode(
            full_prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length
        )
        print(f"DEBUG: Input token length after tokenizer.encode truncation: {input_ids.shape[1]}")

        if input_ids.shape[1] > self.tokenizer.model_max_length:
            print(f"WARNING: Input still too long ({input_ids.shape[1]} > {self.tokenizer.model_max_length}). This should ideally not happen after explicit truncation.")

        print(f"DEBUG: Full Prompt sent to LLM:\n---START PROMPT---\n{full_prompt_text}\n---END PROMPT---")

        try:
            generated_output = self.pipe(
                full_prompt_text,
                max_new_tokens=250,
                num_beams=1,
                do_sample=False,
            )
            print(f"DEBUG: Raw generated_output from pipeline: {generated_output}")

            response = generated_output[0]['generated_text']
            print(f"DEBUG: Response before prompt removal: {response}")

            if response.startswith(full_prompt_text):
                response = response[len(full_prompt_text):].strip()
                print(f"DEBUG: Response after prompt removal: {response}")

            # Removed the specific 'Answer:' cleanup as the prompt itself no longer instructs it to start with 'Answer: <instruction>'
            # If the model still generates 'Answer:', it will be part of the generated text after prompt removal.
            # You can add more sophisticated parsing if needed for models that spontaneously add prefixes.

            response = response.strip()
            print(f"DEBUG: Final response after all cleanups: '{response}'")

        except Exception as e:
            print(f"ERROR: Error during pipeline generation: {e}")
            response = "An error occurred while generating the response."

        if not response or response.isspace():
            response = "I don't have enough information from the provided context to answer that question clearly."
            print(f"DEBUG: Response was empty/whitespace, setting default fallback: {response}")

        print(f"DEBUG: Final LLM Raw Response (after cleanup and fallback check):\n{response}")

        return response