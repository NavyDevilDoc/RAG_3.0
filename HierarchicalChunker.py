# HierarchicalChunker.py
from typing import Dict, List, Optional, Any
from langchain_core.documents import Document
from PageChunker import PageChunker
import spacy

class HierarchicalChunker(PageChunker):
    def __init__(
        self,
        model_name: Optional[str] = None,
        embedding_model: Optional[Any] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.85
    ):
        super().__init__(model_name, embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Installing spacy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], 
                         capture_output=True)
            self.nlp = spacy.load("en_core_web_sm")

    def _analyze_chunk(self, content: str) -> dict:
        """Analyze chunk content using parent class methods."""
        return self._analyze_page(content)  # Reuse parent analysis

    def _create_semantic_chunks(self, content: str, page_num: int) -> List[Document]:
        """Create semantic chunks with detailed metadata."""
        if not content.strip():
            return []

        sentences = list(self.nlp(content).sents)
        chunks = []
        current_chunk = []
        current_length = 0

        for sent in sentences:
            sent_text = sent.text.strip()
            sent_length = len(sent_text)

            if current_length + sent_length > self.chunk_size:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    stats = self._analyze_chunk(chunk_text)
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata={
                            "level": "chunk",
                            "page_num": page_num,
                            "chunk_num": len(chunks),
                            "parent_page": page_num,
                            "char_count": stats["char_count"],
                            "token_count": stats["token_count"],
                            "sentence_count": stats["sentence_count"],
                            "word_count": stats["word_count"],
                            "has_ocr": str(stats["has_ocr"])
                        }
                    ))
                current_chunk = [sent_text]
                current_length = sent_length
            else:
                current_chunk.append(sent_text)
                current_length += sent_length

        # Handle final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            stats = self._analyze_chunk(chunk_text)
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    "level": "chunk",
                    "page_num": page_num,
                    "chunk_num": len(chunks),
                    "parent_page": page_num,
                    "char_count": stats["char_count"],
                    "token_count": stats["token_count"],
                    "sentence_count": stats["sentence_count"],
                    "word_count": stats["word_count"],
                    "has_ocr": str(stats["has_ocr"])
                }
            ))

        self.page_stats.append(f"Created {len(chunks)} chunks for page {page_num}")
        return chunks

    def process_document(self, file_path: str, preprocess: bool = False) -> Dict[str, List[Document]]:
        """Process document with enhanced stats tracking."""
        self.page_stats = []  # Reset stats
        page_docs = super().process_document(file_path, preprocess)
        chunk_docs = []
        total_chunks = 0

        for page_doc in page_docs:
            page_num = page_doc.metadata["page"]
            page_doc.metadata["level"] = "page"
            page_chunks = self._create_semantic_chunks(
                page_doc.page_content, 
                page_num
            )
            chunk_docs.extend(page_chunks)
            total_chunks += len(page_chunks)

        print(f"\nHierarchical Processing Summary:")
        print(f"Total Pages: {len(page_docs)}")
        print(f"Total Chunks: {total_chunks}")
        print("\n".join(self.page_stats))

        return {
            "pages": page_docs,
            "chunks": chunk_docs
        }