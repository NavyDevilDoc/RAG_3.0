# LLMQueryManager.py
from typing import Optional, Any
from ModelManager import ModelManager
from langchain_core.output_parsers import StrOutputParser

class LLMQueryManager:
    """Direct LLM query interface without RAG."""
    
    def __init__(
        self,
        env_path: str,
        llm_type: str,
        embedding_type: str,
        llm_model: str,
        embedding_model: str,
        debug_mode: bool = False
    ):
        """Initialize LLM interface."""
        self.model_manager = ModelManager(env_path)
        self.parser = StrOutputParser()
        self.llm_type = llm_type
        self.embedding_type = embedding_type
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.debug_mode = debug_mode
        self.model = self._initialize_model()
        
    def _initialize_model(self) -> Any:
        """Initialize the language model."""
        config = {
            "selected_llm_type": self.llm_type,
            "selected_embedding_scheme": self.embedding_type  
        }
        
        model, _, _, self.selected_llm, self.selected_embedding_model = self.model_manager.validate_and_load_models(
            config=config,
            select_llm=self.llm_model,
            select_embed=self.embedding_model,
            resource_manager={}
        )
        return model
        
    def ask(self, question: str) -> str:
        """Ask a direct question to the LLM."""
        try:
            # Create chain with parser
            chain = self.model | self.parser
            
            # Get response
            response = chain.invoke(question)
            
            if self.debug_mode:
                print(f"\nModel: {self.selected_llm}")
                print(f"Question: {question}")
                
            # Return response string directly - no .content needed
            return response
            
        except Exception as e:
            print(f"Error in LLM query: {e}")  # Debug info
            return f"Error processing question: {e}"