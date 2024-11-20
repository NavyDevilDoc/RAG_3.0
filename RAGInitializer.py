# RAGInitializer.py (Test)
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Any
from ModelManager import ModelManager
from ComputeResourceManager import ComputeResourceManager

class LLMType(Enum):
    GPT = "gpt"
    OLLAMA = "ollama"

class EmbeddingType(Enum):
    GPT = "gpt" 
    OLLAMA = "ollama"
    SENTENCE_TRANSFORMER = "sentence_transformer"

@dataclass
class RAGConfig:
    env_path: str
    llm_type: LLMType
    embedding_type: EmbeddingType
    llm_index: int
    embedding_index: int

    def to_dict(self) -> Dict[str, str]:
        """Convert config to format expected by ModelManager"""
        return {
            "selected_llm_type": self.llm_type.value,
            "selected_embedding_scheme": self.embedding_type.value
        }

def initialize_rag_components(config: RAGConfig) -> Tuple[Any, Any, int]:
    """Initialize RAG components using ModelManager's existing methods"""
    try:
        model_manager = ModelManager(config.env_path)
        resource_manager = ComputeResourceManager().get_compute_settings()
        
        # Use existing validate_and_load_models method
        model, embeddings, dimensions, selected_llm, selected_embedding_model = model_manager.validate_and_load_models(
            config=config.to_dict(),
            select_llm=config.llm_index,
            select_embed=config.embedding_index,
            resource_manager=resource_manager
        )
        
        return model, embeddings, dimensions, selected_llm, selected_embedding_model, model_manager
        
    except Exception as e:
        raise Exception(f"Failed to initialize RAG components: {str(e)}")