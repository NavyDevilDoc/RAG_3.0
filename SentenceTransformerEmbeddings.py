# SentenceTransformerEmbeddings provides document and query embedding functionality using Sentence Transformers.
# Converts text into vector representations for semantic search and similarity comparisons.

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Loaded SentenceTransformer model '{model_name}' successfully.")
        except Exception as e:
            raise ValueError(f"Error loading SentenceTransformer model '{model_name}': {e}")
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("Input 'texts' must contain at least one document.")
        
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> list[float]:
        if not text:
            raise ValueError("Input 'text' must not be empty.")
        
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
