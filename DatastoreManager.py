# DatastoreManager.py

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
from PineconeManager import PineconeManager, PINECONE_NEW, PINECONE_ADD, PINECONE_EXISTING, LOCAL_STORAGE

class DatastoreManager:
    """
    Manages vector database operations and storage configurations.

    Features:
    1. Pinecone client setup
    2. Storage mode handling
    3. Index management
    4. Document processing

    Attributes:
        pc (Pinecone): Pinecone client instance
        manager (PineconeManager): Manager for Pinecone operations
        embedding_model (str): Name of embedding model

    Example:
        >>> manager = DatastoreManager(
        ...     api_key="your-key",
        ...     embedding_model="text-embedding-ada-002",
        ...     dimensions=1536
        ... )
        >>> store = manager.configure_datastore(
        ...     doc_name="my_docs",
        ...     data_storage_mode="NEW"
        ... )
    """
    def __init__(self, api_key: str, embedding_model: str, dimensions: int):
        """
        Initialize datastore manager with API key and model settings.

        Args:
            api_key (str): Pinecone API key
            embedding_model (str): Name of embedding model
            dimensions (int): Vector dimensions

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If Pinecone initialization fails
        """
        self.pc = Pinecone(api_key=api_key)
        self.manager = PineconeManager(self.pc, api_key, dimensions, embedding_model)
        self.embedding_model = embedding_model  

    def configure_datastore(self, doc_name: str, data_storage_mode: str, documents=None, embeddings=None):
        """
        Configure and initialize vector storage with specified settings.

        Args:
            doc_name (str): Name for document collection
            data_storage_mode (str): Storage mode ("NEW", "ADD", "EXISTING", "LOCAL")
            documents (Optional[List[Any]]): Documents to store
            embeddings (Optional[Any]): Pre-computed embeddings

        Returns:
            Any: Configured datastore instance

        Raises:
            ValueError: If storage mode is invalid
            RuntimeError: If configuration fails

        Example:
            >>> store = manager.configure_datastore(
            ...     "my_docs",
            ...     "NEW",
            ...     documents=[doc1, doc2]
            ... )
        """
        # Map string-based modes to integer constants
        storage_mode_mapping = {
            "NEW": PINECONE_NEW,
            "ADD": PINECONE_ADD,
            "EXISTING": PINECONE_EXISTING,
            "LOCAL": LOCAL_STORAGE
        }

        # Validate the storage mode
        if data_storage_mode not in storage_mode_mapping:
            raise ValueError(f"Invalid data_storage_mode: {data_storage_mode}")

        # Convert to the corresponding constant
        data_storage = storage_mode_mapping[data_storage_mode]

        # Construct the index name
        index_name = f"{doc_name}-{self.embedding_model}".lower()

        # Handle NEW storage mode by creating the index
        if data_storage == PINECONE_NEW:
            spec = ServerlessSpec(cloud="aws", region="us-east-1")
            self.manager.setup_index(index_name, spec)

        # Delegate to PineconeManager's setup_datastore
        return self.manager.setup_datastore(data_storage, documents, embeddings, index_name)
