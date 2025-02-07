"""
DatastoreInitializer.py

A module for initializing and managing vector storage backends.

Features:
- Pinecone vector database integration
- Local storage support
- Dynamic index management
- Document embedding storage
"""

from enum import Enum
from typing import Optional, Any
from pinecone import Pinecone, ServerlessSpec
from PineconeManager import PineconeManager
from storage_constants import StorageType


class DatastoreInitializer:
    """Manages datastore setup and configuration for vector storage."""
    def __init__(self, 
                 doc_name: str,
                 pinecone_api_key: str,
                 dimensions: int,
                 embedding_model: Any):
        """Initialize datastore manager with configuration."""
        self.doc_name = doc_name
        self.pinecone_api_key = pinecone_api_key
        self.dimensions = dimensions
        self.embedding_model = embedding_model
        self.pinecone_client = None
        self.manager = None
        

    def initialize_pinecone(self):
        """Initialize Pinecone client and manager."""
        try:
            self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
            self.manager = PineconeManager(
                self.pinecone_client,
                self.pinecone_api_key,
                self.dimensions,
                self.embedding_model
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone: {e}")


    def _validate_pinecone_name(self, name: str) -> str:
        """Validate and clean name for Pinecone compatibility."""
        # Remove invalid characters
        clean_name = ''.join(c for c in name if c.isalnum() or c in '-_')
        # Ensure name starts with letter/number
        if not clean_name[0].isalnum():
            clean_name = 'idx-' + clean_name
        # Truncate if too long (Pinecone limit is 45 chars)
        return clean_name[:45].lower()


    def _get_index_name(self) -> str:
        """Generate unique index name."""
        return self.doc_name


    def setup_datastore(self, 
                       storage_type: StorageType,
                       documents: Optional[list] = None,
                       embeddings: Optional[Any] = None) -> Any:
        """Set up and configure vector storage backend."""
        try:
            # Initialize Pinecone if not already done
            if not self.pinecone_client:
                self.initialize_pinecone()
                
            index_name = self._get_index_name()
            print(f"Active Index: {index_name}")

            # Handle new Pinecone index creation
            if storage_type == StorageType.PINECONE_NEW:
                spec = ServerlessSpec(cloud='aws', region='us-east-1')
                self.manager.setup_index(index_name, spec)

            # Set up datastore using manager
            datastore = self.manager.setup_datastore(
                storage_type.value,
                documents,
                embeddings,
                index_name
            )
            return datastore
        except Exception as e:
            raise RuntimeError(f"Failed to setup datastore: {e}")
        

    def initialize(self, storage_type: StorageType) -> Any:
        """Initialize and configure datastore with specified storage type."""
        try:
            # Setup Pinecone client and manager
            self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
            self.manager = PineconeManager(
                self.pinecone_client,
                self.pinecone_api_key,
                self.dimensions,
                self.embedding_model
            )
            # Configure index
            index_name = self._get_index_name()
            print(f"Active Index: {index_name}")

            # Create new index if needed
            if storage_type == StorageType.PINECONE_NEW:
                spec = ServerlessSpec(cloud='aws', region='us-east-1')
                self.manager.setup_index(index_name, spec)

            # Initialize documents for existing index
            documents = None if storage_type == StorageType.PINECONE_EXISTING else []

            # Setup and return datastore
            return self.manager.setup_datastore(
                storage_type.value,
                documents,
                self.embedding_model,
                index_name
            )

        except Exception as e:
            print(f"Datastore initialization failed: {e}")
            raise