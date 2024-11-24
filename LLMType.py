# LLMType.py

from enum import Enum

class LLMType(Enum):
    """Available LLM types."""
    GPT = 'gpt'
    OLLAMA = 'ollama'
    
    @classmethod
    def from_string(cls, value: str) -> 'LLMType':
        """Convert string to enum, case-insensitive."""
        try:
            # Handle any case
            return next(
                t for t in cls 
                if t.value.lower() == value.lower()
            )
        except StopIteration:
            raise ValueError(
                f"Invalid LLM type: {value}. "
                f"Available types: {[t.value for t in cls]}"
            )