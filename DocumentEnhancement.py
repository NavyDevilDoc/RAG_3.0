from DocumentProcessor import DocumentProcessor
from langchain_core.documents import Document
from OCREnhancedPDFLoader import OCREnhancedPDFLoader
from TextPreprocessor import TextPreprocessor
from ChunkingMethod import ChunkingMethod
from typing import List, Optional, Any


class DocumentEnhancement:
    """Orchestrates document loading, preprocessing, and chunking."""

    def __init__(self, embedding_model: Any):
        """Initialize DocumentEnhancement with an embedding model."""
        if embedding_model is None:
            raise ValueError("Embedding model must be provided.")
        self.embedding_model = embedding_model
        self.processor = DocumentProcessor(embedding_model=embedding_model)

    def load_and_preprocess_documents(
        self, source_path: str, enable_preprocessing: bool = False
    ) -> List[Document]:
        """Load and optionally preprocess documents from a given source."""
        # Load document using OCR loader
        print("Loading documents with OCR enhancement...")
        ocr_loader = OCREnhancedPDFLoader(source_path)
        documents = ocr_loader.load()
        print(f"Loaded {len(documents)} pages.")

        # Apply preprocessing if enabled
        if enable_preprocessing:
            print("Preprocessing documents...")
            preprocess_text = TextPreprocessor().preprocess
            documents = [
                Document(
                    page_content=preprocess_text(doc.page_content),
                    metadata=doc.metadata,
                )
                for doc in documents
            ]
        else:
            print("Skipping preprocessing.")

        return documents

    def process_documents(
        self,
        source_path: str,
        chunking_method: str,
        enable_preprocessing: bool = False,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.85,
        model_name: Optional[str] = None,
    ) -> List[Document]:
        """Process documents with the specified chunking method."""
        # Load and preprocess documents
        documents = self.load_and_preprocess_documents(
            source_path=source_path, enable_preprocessing=enable_preprocessing
        )

        # Delegate chunking to DocumentProcessor
        if chunking_method.upper() == "SEMANTIC":
            return self.processor.process_document(
                source_path,
                ChunkingMethod.SEMANTIC,
                enable_preprocessing,
                chunk_size,
                chunk_overlap,
                similarity_threshold,
            )
        elif chunking_method.upper() == "PAGE":
            return self.processor.process_document(
                source_path,
                ChunkingMethod.PAGE,
                enable_preprocessing,
                model_name=model_name,
            )
        else:
            raise ValueError(f"Unsupported chunking method: {chunking_method}")
