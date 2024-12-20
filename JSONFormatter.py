import json
import os
from TextPreprocessor import TextPreprocessor

class JSONFormatter:
    """A class to format JSON output for user-friendly display."""
    
    def __init__(self):
        """Initialize the JSONFormatter with a TextPreprocessor instance."""
        self.text_preprocessor = TextPreprocessor()

    def format_json_output(self, input_file: str, output_file: str):
        """Format the JSON output and save it to a new text file."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            formatted_output = ""
            for entry in data:
                if entry['role'] == 'user':
                    formatted_output += f"User: {entry['content']}\n\n"
                elif entry['role'] == 'assistant':
                    formatted_response = self.text_preprocessor.format_text(entry['content'])
                    formatted_output += f"Assistant: {formatted_response}\n\n"
                formatted_output += "\n\n\n"  # Add three lines between each set of pairs
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            
            print(f"Formatted output saved to {output_file}")
        except Exception as e:
            print(f"Error formatting JSON output: {e}")