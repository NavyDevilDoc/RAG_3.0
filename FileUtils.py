import os
from typing import List
from storage_constants import EmbeddingType, LLMType
from ChunkingManager import ChunkingMethod
from storage_constants import StorageType



class TypeConverter:
    """Utility class for converting string inputs to their corresponding enum types"""
    @staticmethod
    def convert_llm_type(llm_type_str: str) -> LLMType:
        """Convert string to LLMType enum case-insensitively"""
        try:
            return getattr(LLMType, llm_type_str.upper())
        except AttributeError:
            raise ValueError(f"Invalid LLM type: {llm_type_str}. Must be one of: {[t.name for t in LLMType]}")

    @staticmethod
    def convert_embedding_type(embedding_type_str: str) -> EmbeddingType:
        """Convert string to EmbeddingType enum case-insensitively"""
        try:
            return getattr(EmbeddingType, embedding_type_str.upper())
        except AttributeError:
            raise ValueError(f"Invalid embedding type: {embedding_type_str}. Must be one of: {[t.name for t in EmbeddingType]}")

    @staticmethod
    def convert_chunking_method(chunking_method_str: str) -> ChunkingMethod:
        """Convert string to ChunkingMethod enum"""
        return getattr(ChunkingMethod, chunking_method_str)

    @staticmethod
    def convert_storage_type(storage_type_str: str) -> StorageType:
        """Convert string to StorageType enum"""
        return StorageType.from_string(storage_type_str)


class FileUtils:
    """Utility class for file operations and type conversions"""
    @staticmethod
    def find_pdfs_in_folder(folder_path: str) -> List[str]:
        """Find all PDF files in the specified folder and return their file paths."""
        pdf_files = []
        for root, dirs, files in os.walk(folder_path):
            print(f"Found {len(files)} files in {root}")
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files

    @staticmethod
    def find_txts_in_folder(folder_path: str) -> List[str]:
        """Find all TXT files in the specified folder and return their file paths."""
        txt_files = []
        for root, dirs, files in os.walk(folder_path):
            print(f"Found {len(files)} files in {root}")
            for file in files:
                if file.lower().endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        return txt_files