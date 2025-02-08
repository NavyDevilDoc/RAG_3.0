# Driver.py - Driver class for RAG model execution

from RAGInitializer import RAGConfig, initialize_rag_components
from CombinedProcessor import CombinedProcessor
from QuestionInitializer import QuestionInitializer
from TextPreprocessor import TextPreprocessor
from ResponseFormatter import QAResponse
from FileUtils import TypeConverter
from typing import List
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Driver:
    def __init__(
        self,
        env_path: str,
        json_path: str,
        doc_input: List[str] = None,
        llm_type: str = 'OLLAMA',
        embedding_type: str = 'SENTENCE_TRANSFORMER',
        llm_model: str = 'llama3.2:latest', 
        embedding_model: str = 'all-mpnet-base-v2', 
        doc_name: str = 'test-index',
        chunking_method: str = 'HIERARCHICAL',
        storage_type: str = 'PINECONE_EXISTING',
        template_name: str = 'default',
        use_ground_truth: bool = False,
        process_questions: bool = True
    ):
        """Initialize RAG driver based on mode."""
        self.should_process_questions = process_questions
        # For all modes
        self.env_path = env_path
        self.json_path = json_path
        self.llm_type = llm_type
        self.embedding_type = embedding_type
        self.llm_model = llm_model
        self.embedding_model = embedding_model
            
        self.doc_input = doc_input
        self.embedding_type = embedding_type
        self.embedding_index = embedding_model
        self.doc_name = doc_name
        self.chunking_method = chunking_method
        self.storage_type = storage_type
        self.template_name = template_name
        self.use_ground_truth = use_ground_truth
            
        # Initialize instance variables
        self.model = None
        self.embeddings = None
        self.dimensions = None
        self.selected_llm = None
        self.selected_embedding_model = None
        self.model_manager = None
        self.datastore = None


    def set_doc_input(self, doc_input: List[str]):
        """Set document input paths after initialization."""
        self.doc_input = doc_input

    def setup(self):
        """Initialize RAG components and process documents."""
        # Initialize RAG components
        config = RAGConfig(
            env_path=self.env_path,
            llm_type=TypeConverter.convert_llm_type(self.llm_type),
            embedding_type=TypeConverter.convert_embedding_type(self.embedding_type),
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
        )
        components = initialize_rag_components(config)
        (self.model, self.embeddings, self.dimensions, 
        self.selected_llm, self.selected_embedding_model, 
        self.model_manager) = components

        
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
            chunking_method=TypeConverter.convert_chunking_method(self.chunking_method),
            storage_type=TypeConverter.convert_storage_type(self.storage_type),
            model_name=self.selected_llm,
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
            embedding_model=self.selected_embedding_model,
            embedding_type=self.embedding_type,
            num_responses=7
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
                results = result[0]
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

    def run(self, questions: List[str]) -> List[str]:
        """Run complete RAG pipeline."""
        if not self.should_process_questions:
            print("Question processing disabled - datastore operations completed")
            return []
        return self.process_questions(questions)