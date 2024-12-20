# ResponseFormatter.py
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from TextPreprocessor import TextPreprocessor

@dataclass
class QAResponse:
    """Structure for holding question-answer pairs with metadata."""
    question: str
    answer: str
    confidence: float
    references: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    timestamp: datetime = datetime.now()

class ResponseFormatter:
    """Formats RAG responses for user-friendly output."""
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.text_processor = TextPreprocessor()
        

    def format_response(self, qa_response: QAResponse) -> str:
        """Format single question-answer pair based on mode."""
        return self._format_debug(qa_response) if self.debug_mode else self._format_user(qa_response)


    def _format_user(self, qa_response: QAResponse) -> str:
        """Format response for end users."""
        output = []
        output.append("\n" + "="*50)
        output.append(f"\nQuestion: {qa_response.question}")
        # Format the answer text
        formatted_answer = self.text_processor.format_text(qa_response.answer, line_length=100)
        output.append(f"\nAnswer: {formatted_answer}")
        return "\n".join(output)


    def _format_debug(self, qa_response: QAResponse) -> str:
        """Format response with debug information."""
        output = []
        output.append("\n" + "="*50)
        output.append(f"\nTimestamp: {qa_response.timestamp}")
        output.append(f"\nQuestion: {qa_response.question}")
        # Format the answer text
        formatted_answer = self.text_processor.format_text(qa_response.answer, line_length=100)
        output.append(f"\nAnswer: {formatted_answer}")
        output.append(f"\nConfidence Score: {qa_response.confidence:.2f}")
        
        if qa_response.references:
            output.append("\nReferences:")
            for ref in qa_response.references:
                output.append(f"- {ref}")
                
        if qa_response.metadata:
            output.append("\nMetadata:")
            for key, value in qa_response.metadata.items():
                output.append(f"- {key}: {value}")
                
        return "\n".join(output)
    

    def format_batch_responses(self, responses: List[QAResponse]) -> str:
        """Format multiple question-answer pairs."""
        return "\n".join(
            self.format_response(response) for response in responses
        )


    def save_to_file(self, 
                        responses: List[QAResponse], 
                        filename: str,
                        output_dir: Optional[str] = None) -> str:
            """Save responses to file with timestamp in specified directory."""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use current directory if none specified
            if output_dir is None:
                output_dir = os.getcwd()
            # Create directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            # Construct full file path
            full_path = os.path.join(
                output_dir, 
                f"{filename}_{timestamp}.txt"
            )
            with open(full_path, "w", encoding='utf-8') as f:
                f.write(self.format_batch_responses(responses))
            print(f"Responses saved to: {full_path}")
            return full_path