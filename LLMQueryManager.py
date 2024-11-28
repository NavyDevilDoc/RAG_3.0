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
        debug_mode: bool = False,
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
        self.conversation_history = []
        
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
    
    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
    def get_conversation_context(self) -> str:
        return "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history
        ])

    def ask(self, question: str, use_history: bool = True) -> str:
        try:
            self.add_to_history("user", question)
            
            # Include history in prompt if enabled
            prompt = (
                self.get_conversation_context() if use_history and self.conversation_history 
                else question
            )
            
            chain = self.model | self.parser
            response = chain.invoke(prompt)
            
            self.add_to_history("assistant", response)
            
            if self.debug_mode:
                print(f"\nModel: {self.selected_llm}")
                print(f"History length: {len(self.conversation_history)}")
                
            return response
            
        except Exception as e:
            print(f"Error in LLM query: {e}")
            return f"Error processing question: {e}"