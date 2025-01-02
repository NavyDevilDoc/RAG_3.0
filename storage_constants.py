from enum import Enum

class StorageType(Enum):
    """Unified storage type definitions for vector databases."""
    PINECONE_NEW = 0
    PINECONE_ADD = 1
    PINECONE_EXISTING = 2
    LOCAL_STORAGE = 3
    
    @classmethod
    def from_string(cls, storage_type: str) -> 'StorageType':
        """Convert string to StorageType enum."""
        try:
            return cls[storage_type.upper()]
        except KeyError:
            raise ValueError(f"Invalid storage type: {storage_type}")