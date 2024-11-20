# SentenceTransformerEmbeddings provides document and query embedding functionality using Sentence Transformers.
# Converts text into vector representations for semantic search and similarity comparisons.

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        """Initialize SentenceTransformerEmbeddings with a specified model.
        
        Args:
            model_name (str): Name of the pre-trained model (e.g., 'all-mpnet-base-v2').
        
        Raises:
            ValueError: If the model name is invalid or the model fails to load.
        """
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Loaded SentenceTransformer model '{model_name}' successfully.")
        except Exception as e:
            raise ValueError(f"Error loading SentenceTransformer model '{model_name}': {e}")
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of texts into vectors.
        
        Args:
            texts (list[str]): Collection of documents to be embedded.
            
        Returns:
            list[list[float]]: List of embedding vectors, where each vector is represented
                               as a list of floating-point numbers.
        
        Raises:
            ValueError: If input `texts` is empty.
        
        Note:
            Uses SentenceTransformer to batch encode texts and convert the resulting tensor
            to a nested list format.
        """
        if not texts:
            raise ValueError("Input 'texts' must contain at least one document.")
        
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query text into a vector.
        
        Args:
            text (str): Single query text to be embedded.
            
        Returns:
            list[float]: Embedding vector as a list of floating-point numbers.
        
        Raises:
            ValueError: If the input `text` is empty.
        
        Note:
            Processes the query as a batch of one for compatibility with model encoding.
        """
        if not text:
            raise ValueError("Input 'text' must not be empty.")
        
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
