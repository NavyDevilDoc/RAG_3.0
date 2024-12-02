# main.py
import os
from Driver import Driver
from ResponseFormatter import ResponseFormatter
from TextPreprocessor import TextPreprocessor
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# File path for environment variables and JSON file
env_path = r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\env_variables.env"
json_path = r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\RAG_Model\RAG_Outputs\JSON_Output"

# Flag to choose between manual input and automatic folder search. Can be True for manual upload or False for folder upload
use_manual_input = True
if use_manual_input:
    # List of document paths; useful for a small number of documents
    doc_input = [r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\Source_Documents\MCDP_1_Warfighting.pdf"]
else:
    # Specify the folder containing the PDF documents
    pdf_folder_path = r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\EDQP_RAG_Model\Local_RAG_Model\RAG_Model\Partial_EDQP"
    # Use the helper function to find all PDF files in the specified folder
    doc_input = Driver.find_pdfs_in_folder(pdf_folder_path)
# Output directory for RAG responses
output_dir = r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\RAG_Model\RAG_Outputs"


# Initialize driver with all required parameters
driver = Driver(
## Set the mode to 'rag' for RAG mode or 'llm' for LLM mode ##
    mode='rag',
## Set for both modes ##
    env_path=env_path, 
    json_path=json_path,
    llm_type='gpt', 
    embedding_type='sentence_transformer', 
    llm_model='gpt-4o', 
    embedding_model='all-mpnet-base-v2', 
    debug_mode=False, 
## Set for RAG mode ##
    doc_input=doc_input,
    doc_name='arwg-index', 
    chunking_method='SEMANTIC', 
    storage_type='PINECONE_EXISTING',
    template_name='short', 
    use_ground_truth=False,
    process_questions=True,
)


## Run the driver in the selected mode ##
# LLM mode #
if driver.mode == 'llm': 
    # Initialize the text preprocessor module
    text_preprocessor = TextPreprocessor()

    # Flag to clear conversation history
    clear_history = True  # Set to True to clear history
    if clear_history:
        driver.clear_conversation_history()
        print("\nConversation history cleared.")


    while True:
        # Enter the input query
        user_input = input("Enter your query (or type 'exit' to quit): ")

        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break

        # Process the input query and display the response
        response = driver.process_query(user_input)

        # Get updated history after processing
        current_history = driver.get_conversation_history()
        print(f"\nQ: {user_input}")
        print(f"A: {response}")
        print(f"\nConversation length: {len(current_history)} messages")

# RAG mode #
elif driver.mode == 'rag': 
    # Define question(s)
    '''
    questions = ["What are the three medical service groups for Class 1 aviation personnel?"]
    
    # Run the model and and display the responses
    qa_responses = driver.run(questions) 
    # Save the question/answer pairs to a text file
    formatter = ResponseFormatter(debug_mode=False)
    saved_path = formatter.save_to_file(qa_responses,
                                        "rag_responses",
                                        output_dir=output_dir)
    for response in qa_responses:
        print(formatter.format_response(response))
    '''
    while True:
        # Prompt the user for input
        user_input = input("Enter your questions separated by a comma (or type 'exit' to quit): ")
        
        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        
        # Split the input into multiple questions
        questions = [q.strip() for q in user_input.split(',')]
        
        # Process the questions and display the responses
        responses = driver.run(questions)
        
        for response in responses:
            print(f"\nQ: {response.question}")
            print(f"A: {response.answer}")
            print(f"Confidence: {response.confidence:.2f}")
            if response.references:
                print("References:")
                for ref in response.references:
                    print(f"- {ref}")
            print("="*50)