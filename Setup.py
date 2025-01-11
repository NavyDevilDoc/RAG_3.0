# Setup.py
import os
import tkinter as tk
from tkinter import filedialog
from Driver import Driver
from TextPreprocessor import TextPreprocessor
from ResponseFormatter import ResponseFormatter
from JSONFormatter import JSONFormatter

text_preprocessor = TextPreprocessor()
response_formatter = ResponseFormatter()

def load_environment_variables(file_path: str):
    from dotenv import load_dotenv
    load_dotenv(file_path)


def get_document_input():
    tk_root = tk.Tk()
    tk_root.withdraw()
    selection_type = input("Select 'files' or 'folder': ").strip().lower()
    if selection_type == 'files':
        selected_files = filedialog.askopenfilenames(
            title="Select PDF(s)",
            filetypes=[("PDF files", "*.pdf")]
        )
        return list(selected_files)
    elif selection_type == 'folder':
        selected_folder = filedialog.askdirectory(title="Select a folder with PDFs")
        if selected_folder:
            return Driver.find_pdfs_in_folder(selected_folder)
    return []


def handle_rag_mode(driver: Driver):
    # Get document input first
    doc_input = get_document_input()
    if not doc_input:
        print("No documents selected. Exiting...")
        return
    try:
        # Initialize RAG components
        driver.set_doc_input(doc_input)
        driver.setup()
        
        if driver.should_process_questions:
            output_dir = os.getenv("OUTPUT_DIR")
            responses_list = []
            while True:
                user_input = input("Enter your questions separated by a comma (or type 'exit' to quit): ")
                if user_input.lower() == 'exit':
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


def handle_llm_mode(driver: Driver):
    action = input("Do you want to (1) load an existing conversation history file or (2) start a new chat history? Enter 1 or 2: ")
    user_named_file = None
    if action == '1':
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        existing_file = filedialog.askopenfilename(
            title="Select conversation history file",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if existing_file:
            driver.load_existing_conversation(existing_file)
            driver.llm_query.set_history_file(existing_file)  # Update history file path
    elif action == '2':
        user_named_file = driver.clear_conversation_history()
        print("\nConversation history cleared.")
    else:
        user_named_file = driver.clear_conversation_history()
        print("Invalid input. Starting a new chat history by default.")
        print("\nConversation history cleared.")
    
    while True:
        user_input = input("Enter your query (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        response = driver.process_query(user_input)
        print(response)
    
    # Ask the user if they want to format the LLM output
    format_output = input("Do you want to format the LLM output? (yes/no): ")
    if format_output.lower() == 'yes':
        json_formatter = JSONFormatter()
        input_file = driver.llm_query.history_file
        
        # Ask the user for the new name for the conversation history file
        new_name = input("Enter the new name for the conversation history file (without extension): ")
        new_file_path = os.path.join(driver.llm_query.json_path, f"{new_name}.json")
        
        # Rename the conversation history file
        os.rename(input_file, new_file_path)
        
        # Update the history file path in the driver
        driver.llm_query.set_history_file(new_file_path)
        
        # Create the formatted output file
        output_file = new_file_path.replace(".json", "_formatted.txt")
        json_formatter.format_json_output(new_file_path, output_file)