# DatastoreInitializer.py

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

class StorageType(Enum):
    """
    Enumeration of supported vector storage types.

    Values:
        PINECONE_NEW: Create new Pinecone index
        PINECONE_ADD: Add to existing Pinecone index
        PINECONE_EXISTING: Use existing Pinecone index
        LOCAL_STORAGE: Use local vector storage
    """
    PINECONE_NEW = 0
    PINECONE_ADD = 1
    PINECONE_EXISTING = 2
    LOCAL_STORAGE = 3

class DatastoreInitializer:
    """
    Manages datastore setup and configuration for vector storage.

    Features:
    1. Pinecone client initialization
    2. Index management
    3. Document storage
    4. Error handling

    Attributes:
        doc_name (str): Document collection name
        pinecone_api_key (str): Pinecone API key
        dimensions (int): Vector dimensions
        embedding_model (Any): Model for embeddings
        pinecone_client (Optional[Pinecone]): Pinecone client
        manager (Optional[PineconeManager]): Pinecone manager

    Example:
        >>> initializer = DatastoreInitializer(
        ...     doc_name="my_docs",
        ...     pinecone_api_key="api_key",
        ...     dimensions=1536,
        ...     embedding_model=embeddings
        ... )
        >>> index = initializer.setup_datastore(StorageType.PINECONE_NEW)
    """
    
    def __init__(self, 
                 doc_name: str,
                 pinecone_api_key: str,
                 dimensions: int,
                 embedding_model: Any):
        """
        Initialize datastore manager with configuration.

        Args:
            doc_name (str): Name for document collection
            pinecone_api_key (str): Pinecone API key
            dimensions (int): Embedding dimensions
            embedding_model (Any): Model for generating embeddings

        Raises:
            ValueError: If parameters are invalid
        """
        self.doc_name = doc_name
        self.pinecone_api_key = pinecone_api_key
        self.dimensions = dimensions
        self.embedding_model = embedding_model
        self.pinecone_client = None
        self.manager = None
        
    def initialize_pinecone(self):
        """
        Initialize Pinecone client and manager.

        Raises:
            RuntimeError: If Pinecone initialization fails

        Example:
            >>> initializer.initialize_pinecone()
        """
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

    def _get_index_name(self) -> str:
        """
        Generate unique index name.

        Returns:
            str: Formatted index name

        Example:
            >>> name = initializer._get_index_name()
            >>> print(name)  # "my_docs-ada002"
        """
        return f"{self.doc_name}-{self.embedding_model}".lower()

    def setup_datastore(self, 
                       storage_type: StorageType,
                       documents: Optional[list] = None,
                       embeddings: Optional[Any] = None) -> Any:
        """
        Set up and configure vector storage backend.

        Args:
            storage_type (StorageType): Type of storage to initialize
            documents (Optional[list]): Documents to store
            embeddings (Optional[Any]): Pre-computed embeddings

        Returns:
            Any: Configured datastore instance

        Raises:
            RuntimeError: If datastore setup fails

        Example:
            >>> docs = [Document(text="content")]
            >>> store = initializer.setup_datastore(
            ...     StorageType.PINECONE_NEW,
            ...     documents=docs
            ... )
        """
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
        """
        Initialize and configure datastore with specified storage type.

        Performs:
        1. Pinecone client setup
        2. Manager initialization
        3. Index configuration
        4. Storage backend setup

        Args:
            storage_type (StorageType): Type of storage to initialize

        Returns:
            Any: Configured datastore instance

        Raises:
            RuntimeError: If initialization fails
            ValueError: If storage type is invalid

        Example:
            >>> store = initializer.initialize(StorageType.PINECONE_NEW)
            >>> print(f"Initialized {store.index_name}")
        """
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
            if storage_type == StorageType.NEW:
                spec = ServerlessSpec(cloud='aws', region='us-east-1')
                self.manager.setup_index(index_name, spec)

            # Initialize documents for existing index
            documents = None if storage_type == StorageType.EXISTING else []

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