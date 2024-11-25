# Driver.py - Driver class for RAG model execution

from RAGInitializer import LLMType, EmbeddingType, RAGConfig, initialize_rag_components
from CombinedProcessor import CombinedProcessor, ChunkingMethod
from DatastoreInitializer import StorageType
from QuestionInitializer import QuestionInitializer
from TextPreprocessor import TextPreprocessor
from ResponseFormatter import QAResponse
from LLMQueryManager import LLMQueryManager
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Driver:
    def __init__(
        self,
        env_path: str,
        doc_input: List[str] = None,
        llm_type: str = 'OLLAMA',
        embedding_type: str = 'SENTENCE_TRANSFORMER',
        llm_index: int = 1,
        embedding_index: int = 2,
        doc_name: str = 'test-index',
        chunking_method: str = 'HIERARCHICAL',
        storage_type: str = 'PINECONE_EXISTING',
        template_name: str = 'default',
        use_ground_truth: bool = False,
        debug_mode: bool = False,
        mode: str = 'rag'
    ):
        """Initialize RAG driver based on mode."""
        self.mode = mode.lower()
        
        # For all modes
        self.env_path = env_path
        self.llm_type = llm_type.lower() if mode == 'llm' else llm_type.upper()
        self.llm_index = llm_index
        self.debug_mode = debug_mode
        
        # Initialize LLM query manager for direct queries
        if self.mode == 'llm':
            self.llm_query = LLMQueryManager(
                env_path=env_path,
                llm_type=llm_type,
                llm_index=llm_index,
                debug_mode=debug_mode
            )
            return  # Early exit - don't initialize RAG components
            
        # Only initialize RAG components if in RAG mode
        self.doc_input = doc_input
        self.embedding_type = embedding_type
        self.embedding_index = embedding_index
        self.doc_name = doc_name
        self.chunking_method = chunking_method
        self.storage_type = storage_type
        self.template_name = template_name
        self.use_ground_truth = use_ground_truth
        self.mode = mode
        
        # Initialize instance variables
        self.model = None
        self.embeddings = None
        self.dimensions = None
        self.selected_llm = None
        self.selected_embedding_model = None
        self.model_manager = None
        self.datastore = None
        
        # Initial setup for RAG mode
        self.setup()
        
    @staticmethod
    def _convert_llm_type(llm_type_str: str) -> LLMType:
        return getattr(LLMType, llm_type_str)
    
    @staticmethod
    def _convert_embedding_type(embedding_type_str: str) -> EmbeddingType:
        return getattr(EmbeddingType, embedding_type_str)
    
    @staticmethod
    def _convert_chunking_method(chunking_method_str: str) -> ChunkingMethod:
        return getattr(ChunkingMethod, chunking_method_str)
    
    @staticmethod
    def _convert_storage_type(storage_type_str: str) -> StorageType:
        return getattr(StorageType, storage_type_str)
    

    def setup(self):
        """Initialize RAG components and process documents."""
        # Initialize RAG components
        config = RAGConfig(
            env_path=self.env_path,
            llm_type=self._convert_llm_type(self.llm_type),
            embedding_type=self._convert_embedding_type(self.embedding_type),
            llm_index=self.llm_index,
            embedding_index=self.embedding_index,
            mode = self.mode
        )
        
        components = initialize_rag_components(config)
        if len(components) == 6:  # Full RAG mode
            (self.model, self.embeddings, self.dimensions, 
            self.selected_llm, self.selected_embedding_model, 
            self.model_manager) = components
        elif len(components) == 4:  # LLM mode
            (self.model, self.embeddings, self.selected_llm, 
            self.selected_embedding_model) = components
            self.dimensions = None  # Set default
            self.model_manager = None  # Set default
        else:
            raise ValueError(f"Unexpected number of components: {len(components)}")
        
        if self.model and self.embeddings and self.dimensions:
            print("\nRAG components successfully initialized.")
            print(f"Model: {self.selected_llm}")
            print(f"Embeddings: {self.selected_embedding_model}")
            
        # Process documents
        processor = CombinedProcessor(
            doc_name=self.doc_name,
            model_manager=self.model_manager,
            embedding_model=self.selected_embedding_model,
            embeddings=self.embeddings,
            dimensions=self.dimensions,
            chunking_method=self._convert_chunking_method(self.chunking_method),
            storage_type=self._convert_storage_type(self.storage_type),
            model_name=self.selected_llm
        )
        
        self.datastore = processor.process_and_store(self.doc_input)
        
    def process_questions(self, questions: List[str]) -> List[QAResponse]:
        """Process questions and return QAResponse objects."""
        if not self.datastore:
            raise RuntimeError("Setup must be called before processing questions")
        
        if not questions:
            raise ValueError("Questions list cannot be empty")
                
        processor = QuestionInitializer(
            datastore=self.datastore,
            model=self.model,
            embedding_model=self.selected_embedding_model
        )
        
        try:
            # Get responses and handle tuple return
            result = processor.process_questions(
                questions=questions,
                template_name=self.template_name,     
                use_ground_truth=self.use_ground_truth[0] if isinstance(self.use_ground_truth, tuple) else self.use_ground_truth
            )
            
            # Unpack tuple if needed
            if isinstance(result, tuple):
                results = result[0]  # Get just the results, ignore processing time
            else:
                results = result
                
            # Ensure results is a list
            if not isinstance(results, list):
                print(f"Warning: Invalid results format after unpacking: {type(results)}")
                return []
                
            responses = []
            text_processor = TextPreprocessor()
            # Map each result to its corresponding question
            for question, result in zip(questions, results):
                if not isinstance(result, dict):
                    print(f"Warning: Invalid result format: {type(result)}")
                    continue
                    
                try:
                    response = QAResponse(
                        question=question,
                        answer=result['answer'],
                        confidence=result.get('confidence', 0.0),
                        references=result.get('references', []),
                        metadata={
                            'model': self.selected_llm,
                            'embeddings': self.selected_embedding_model,
                            'template': self.template_name,
                            'chunking_method': self.chunking_method,
                            'quality_scores': result.get('quality_scores', {})
                        }
                    )
                    # Add token count to the answer
                    response.answer += f"\n\nToken Count: {text_processor.count_tokens(response.answer)}"
                    responses.append(response)
                except KeyError as e:
                    print(f"Warning: Missing required field in result: {e}")
                    continue
                    
            return responses
            
        except Exception as e:
            raise RuntimeError(f"Error processing questions: {e}")


    def process_query(self, query: str) -> str:
        """Process query based on mode and format response."""
        text_processor = TextPreprocessor()
        
        if self.mode == 'llm':
            if not self.llm_query:
                raise RuntimeError("LLM mode not initialized")
            response = self.llm_query.ask(query)
            formatted_response = text_processor.format_text(response)
            token_count = text_processor.count_tokens(formatted_response)
            return f"{formatted_response}\n\nToken Count: {token_count}"
        else:
            qa_response = self.process_questions([query])[0]
            formatted_response = text_processor.format_text(qa_response.answer)
            token_count = text_processor.count_tokens(formatted_response)
            return f"{formatted_response}\n\nToken Count: {token_count}"


    def run(self,  questions: List[str]) -> List[str]:
        """Run complete RAG pipeline."""
        return self.process_questions(questions)