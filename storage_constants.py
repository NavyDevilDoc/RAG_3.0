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
            valid_types = [t.name for t in cls]
            raise ValueError(f"Invalid storage type: {storage_type}. Valid types are: {valid_types}")

    def __str__(self):
        return self.name
    
class LLMType(Enum):
    """Available LLM types."""
    GPT = 'gpt'
    OLLAMA = 'ollama'

class EmbeddingType(Enum):
    GPT = "gpt" 
    SENTENCE_TRANSFORMER = "sentence_transformer"
    OLLAMA = "ollama"

class ChunkingMethod(Enum):
    """Supported document chunking methods."""
    SEMANTIC = "semantic"
    PAGE = "page"
    HIERARCHICAL = "hierarchical" 