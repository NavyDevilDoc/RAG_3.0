# main.py
import os
from Setup import load_environment_variables, get_document_input, handle_rag_mode, handle_llm_mode
from Driver import Driver
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# File path for environment variables and JSON file
file_path = r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\RAG_Model\file_variables.env"
load_environment_variables(file_path)

# Flag to choose between manual input and automatic folder search. Can be True for manual upload or False for folder upload
use_manual_input = True
pdf_folder_path = os.getenv("PDF_FOLDER_PATH")
doc_input = get_document_input(use_manual_input, pdf_folder_path)


"""
Available Ollama LLMs:
    llama3.2:latest
    llama3.2-vision:latest
    llama3.3:70b-instruct-q2_K
    codestral:latest
    codellama:13b
    granite3.1-dense:8b (use with granite-embedding:latest as the embedding model)
    granite-code:20b
    granite-code:8b
"""


# Initialize driver with all required parameters
driver = Driver(
    mode='llm',
    env_path=os.getenv("ENV_PATH"), 
    json_path=os.getenv("JSON_PATH"),
    llm_type='ollama',
    embedding_type='ollama',
    llm_model='granite3.1-dense:8b',
    embedding_model='granite-embedding:latest',
    debug_mode=False, 
    doc_input=doc_input,
    doc_name='test-index', 
    chunking_method='HIERARCHICAL', 
    storage_type='LOCAL_STORAGE',
    template_name='assistant', 
    use_ground_truth=False,
    process_questions=True,
)

# Handle modes
if driver.mode == 'rag':
    handle_rag_mode(driver)
elif driver.mode == 'llm':
    handle_llm_mode(driver)