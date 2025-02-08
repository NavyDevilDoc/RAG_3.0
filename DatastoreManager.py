"""
DatastoreManager.py

A module for managing vector database operations with Pinecone integration.

Features:
- Pinecone initialization and configuration
- Storage mode management
- Document and embedding handling
- Index setup and maintenance
"""

from pinecone import Pinecone, ServerlessSpec
from PineconeManager import PineconeManager
from storage_constants import StorageType

class DatastoreManager:
    """Manages vector database operations and storage configurations."""
    
    def __init__(self, api_key: str, embedding_model: str, dimensions: int):
        """Initialize datastore manager with API key and model settings."""
        self.pc = Pinecone(api_key=api_key)
        self.manager = PineconeManager(self.pc, api_key, dimensions, embedding_model)
        self.embedding_model = embedding_model  

    def configure_datastore(self, doc_name: str, data_storage_mode: str, documents=None, embeddings=None):
        """Configure and initialize vector storage with specified settings."""
        try:
            # Convert string to StorageType enum
            storage_type = StorageType.from_string(data_storage_mode)
            
            # Construct the index name
            index_name = f"{doc_name}-{self.embedding_model}".lower()

            # Handle NEW storage mode by creating the index
            if storage_type == StorageType.PINECONE_NEW:
                spec = ServerlessSpec(cloud="aws", region="us-east-1")
                self.manager.setup_index(index_name, spec)

            # Delegate to PineconeManager's setup_datastore
            return self.manager.setup_datastore(storage_type, documents, embeddings, index_name)
            
        except ValueError as e:
            raise ValueError(f"Storage configuration error: {str(e)}")