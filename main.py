# main.py
import os
from Driver import Driver, find_pdfs_in_folder
from ResponseFormatter import ResponseFormatter
from TextPreprocessor import TextPreprocessor
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# File path for environment variables
env_path = r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\EDQP_RAG_Model\env_variables.env"
# Flag to choose between manual input and automatic folder search. Can be True for manual upload or False for folder upload
use_manual_input = True
if use_manual_input:
    # List of document paths; useful for a small number of documents
    doc_input = [r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\Source_Documents\ARWG COMPLETE_210811.pdf"]
else:
    # Specify the folder containing the PDF documents
    pdf_folder_path = r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\EDQP_RAG_Model\Local_RAG_Model\RAG_Model\Partial_EDQP"
    # Use the helper function to find all PDF files in the specified folder
    doc_input = find_pdfs_in_folder(pdf_folder_path)
# Output directory for RAG responses
output_dir = r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\EDQP_RAG_Model\Local_RAG_Model\RAG_Model\RAG_Outputs"

# Initialize driver with all required parameters
driver = Driver(
    mode='rag',
## Set for both modes ##
    env_path=env_path, 
    llm_type='OLLAMA', 
    embedding_type='GPT', 
    llm_model='llama3.2:latest', 
    embedding_model='text-embedding-3-small', 
    debug_mode=False, 
## Set for RAG mode ##
    doc_input=doc_input,
    doc_name='arwg-index', 
    chunking_method='HIERARCHICAL', 
    storage_type='PINECONE_NEW',
    template_name='default', 
    use_ground_truth=False,
    process_questions=False
)

## Run the driver in the selected mode ##
# LLM mode
if driver.mode == 'llm': 
    # Initialize the text preprocessor module
    text_preprocessor = TextPreprocessor()
    # Enter the input query
    input = "How would I use Python to code Gauss-Jordan elimination?"
    # Process the input query and display the response
    response = driver.process_query(input)
    print(f"\nQ: {input}")
    print(f"A: {response}")

# RAG mode
elif driver.mode == 'rag': 
    # Define question(s)
    questions = []
    
    # Run the model and and display the responses
    qa_responses = driver.run(questions) 
    # Save the question/answer pairs to a text file
    formatter = ResponseFormatter(debug_mode=False)
    saved_path = formatter.save_to_file(qa_responses,
                                        "rag_responses",
                                        output_dir=output_dir)
    for response in qa_responses:
        print(formatter.format_response(response))