# Setup.py
import os
import tkinter as tk
from tkinter import filedialog
from Driver import Driver
from FileUtils import FileUtils
from TextPreprocessor import TextPreprocessor
from ResponseFormatter import ResponseFormatter

text_preprocessor = TextPreprocessor()
response_formatter = ResponseFormatter()

def load_environment_variables(file_path: str):
    from dotenv import load_dotenv
    load_dotenv(file_path)


def get_document_input():
    tk_root = tk.Tk()
    tk_root.withdraw()
    selection_type = input("Select 'files', 'folder', or 'exit': ").strip().lower()
    if selection_type == 'files':
        selected_files = filedialog.askopenfilenames(
            title="Select PDF(s)",
            filetypes=[("PDF files", "*.pdf")]
        )
        return list(selected_files)
    elif selection_type == 'folder':
        selected_folder = filedialog.askdirectory(title="Select a folder with PDFs")
        if selected_folder:
            return FileUtils.find_pdfs_in_folder(selected_folder)
    elif selection_type == 'exit':
        return []
    return []


def initiate_rag(driver: Driver):
    try:
        # Check if using existing Pinecone index
        if driver.storage_type.lower() != 'pinecone_existing':
            # Get document input only if not using existing index
            doc_input = get_document_input()
            if not doc_input:
                print("No documents selected. Exiting...")
                return
            driver.set_doc_input(doc_input)
        else:
            print("Using existing Pinecone index - skipping document selection...")
            driver.set_doc_input([])  # Set empty list since no new documents needed

        # Initialize RAG components
        driver.setup()
        
        if driver.should_process_questions:
            output_dir = os.getenv("OUTPUT_DIR")
            responses_list = []
            while True:
                user_input = input("Enter your questions separated by a comma (or type 'exit/quit' to quit): ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                questions = [q.strip() for q in user_input.split(',')]
                responses = driver.run(questions)
                responses_list.extend(responses)
                response_formatter.save_to_file(responses=responses, filename='rag_responses', output_dir=output_dir)
                formatted_responses = response_formatter.format_batch_responses(responses)
                print(formatted_responses)
    except Exception as e:
        print(f"Error in RAG mode: {str(e)}")
        return