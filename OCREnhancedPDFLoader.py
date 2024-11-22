# OCREnhancedPDFLoader.py

"""
OCREnhancedPDFLoader.py

A module for loading PDFs with OCR (Optical Character Recognition) support.

Features:
- PDF loading with OCR fallback
- Image text extraction
- Multiple page handling
- Text quality optimization
"""

import os
import pytesseract
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from pdf2image import convert_from_path

class OCREnhancedPDFLoader:
    """
    Loads PDFs with OCR support for text extraction.

    Features:
    1. Native PDF text extraction
    2. OCR fallback for scanned documents
    3. Multi-page handling
    4. Image preprocessing

    Attributes:
        file_path (str): Path to PDF file
        ocr_languages (str): Languages for OCR
        dpi (int): DPI for image conversion
        use_ocr_always (bool): Force OCR usage

    Example:
        >>> loader = OCREnhancedPDFLoader("doc.pdf")
        >>> documents = loader.load()
    """
    BLANK_THRESHOLD = 10  # Minimum character count to consider a page non-blank

    def __init__(self, file_path: str, tesseract_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
        """Initialize the OCR-enhanced PDF loader.
        
        Args:
            file_path (str): Path to the PDF file to be processed.
            tesseract_path (str): Path to the Tesseract OCR executable.
        
        Raises:
            FileNotFoundError: If the specified PDF file does not exist.
            ValueError: If the Tesseract path is invalid.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"PDF file not found at path: {file_path}")
        
        self.file_path = file_path
        self.skipped_pages = []
        
        # Set Tesseract path, validating existence
        if not os.path.isfile(tesseract_path):
            raise ValueError(f"Tesseract executable not found at path: {tesseract_path}")
        pytesseract.pytesseract.tesseract_cmd = tesseract_path


    def _is_blank_page(self, text: str) -> bool:
        """Check if page is blank or contains only whitespace/special characters.
        
        Args:
            text (str): Page content to analyze.
            
        Returns:
            bool: True if page is considered blank, False otherwise.
        """
        if not text or not text.strip():
            return True
        cleaned_text = text.strip().replace('\n', '').replace('\r', '').replace('\t', '')
        return len(cleaned_text) < self.BLANK_THRESHOLD


    def _process_page(self, doc, img, page_number: int):
        """Process a single page by combining standard extracted text with OCR.
        
        Args:
            doc (Document): Document object containing standard extracted text.
            img: PIL Image of the page for OCR processing.
            page_number (int): Page number for metadata.
        
        Returns:
            Document: Document object with combined content and metadata.
        """
        existing_text = doc.page_content
        
        # Apply OCR to page image and handle potential errors
        try:
            ocr_text = pytesseract.image_to_string(img)
        except Exception as e:
            print(f"Error applying OCR to page {page_number}: {e}")
            ocr_text = ""
        
        combined_text = f"{existing_text}\n{ocr_text}".strip()
        
        # Check if the page is blank after combining
        if self._is_blank_page(combined_text):
            self.skipped_pages.append(page_number)
            return None
        
        # Return the Document with enhanced content and metadata
        return Document(
            page_content=combined_text,
            metadata={
                **doc.metadata,
                "source": "combined_text_and_ocr",
                "page": page_number,
                "is_blank": "false",
                "has_ocr": "true"
            }
        )


    def load(self):
        """Load and process PDF file with OCR enhancement.
        
        Returns:
            List[Document]: List of processed documents with combined text and OCR content.
        
        Implementation:
            1. Extract text using PyMuPDF
            2. Convert pages to images
            3. Apply OCR to each page
            4. Combine extracted text with OCR results
            5. Skip blank pages
            6. Preserve metadata and page structure
        
        Raises:
            Exception: If any error occurs during loading or processing.
        """
        try:
            # Standard PDF text extraction using PyMuPDF
            loader = PyMuPDFLoader(self.file_path)
            text_documents = loader.load()
            
            # Convert PDF pages to high-resolution images for OCR
            images = convert_from_path(self.file_path, dpi=300)
            
            # Process each page and combine standard extraction with OCR
            enhanced_documents = []
            for idx, (doc, img) in enumerate(zip(text_documents, images)):
                page_number = idx + 1
                enhanced_doc = self._process_page(doc, img, page_number)
                
                # Append only non-blank pages
                if enhanced_doc:
                    enhanced_documents.append(enhanced_doc)
            
            # Report skipped blank pages, if any
            if self.skipped_pages:
                print(f"Skipped {len(self.skipped_pages)} blank pages: {self.skipped_pages}")
            
            return enhanced_documents
            
        except Exception as e:
            print(f"Error in OCR-enhanced loading: {e}")
            raise
