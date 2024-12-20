# Setup.py
from Driver import Driver
from TextPreprocessor import TextPreprocessor
from ResponseFormatter import ResponseFormatter


text_preprocessor = TextPreprocessor()
response_formatter = ResponseFormatter()

def load_environment_variables(file_path: str):
    from dotenv import load_dotenv
    load_dotenv(file_path)

def get_document_input(use_manual_input: bool, pdf_folder_path: str = None):
    if use_manual_input:
        return [r""]
    else:
        return Driver.find_pdfs_in_folder(pdf_folder_path)

def handle_rag_mode(driver: Driver):
    if driver.mode == 'rag' and driver.should_process_questions: 
        while True:
            user_input = input("Enter your questions separated by a comma (or type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                print("Exiting the program.")
                break
            questions = [q.strip() for q in user_input.split(',')]
            responses = driver.run(questions)
            formatted_responses = response_formatter.format_batch_responses(responses)
            print(formatted_responses)
        print("Question answering component is disabled.")

def handle_llm_mode(driver: Driver):
    clear_history = True
    if clear_history:
        driver.clear_conversation_history()
        print("\nConversation history cleared.")
    while True:
        user_input = input("Enter your query (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        response = driver.process_query(user_input)
        print(response)
        