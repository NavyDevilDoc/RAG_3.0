# ChunkingMethod defines available document processing strategies, either by semantic similarity or page boundaries.

from enum import Enum

class ChunkingMethod(Enum):
    """Supported document chunking methods.
    
    Attributes:
        SEMANTIC: Chunk by semantic similarity and size.
        PAGE: Chunk by physical page boundaries.
    """
    SEMANTIC = "semantic"
    PAGE = "page"
