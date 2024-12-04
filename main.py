# main.py
import os
from Setup import load_environment_variables, get_document_input, handle_rag_mode, handle_llm_mode
from Driver import Driver
from ResponseFormatter import ResponseFormatter
from TextPreprocessor import TextPreprocessor
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

# Initialize driver with all required parameters
driver = Driver(
    mode='llm',
    env_path=os.getenv("ENV_PATH"), 
    json_path=os.getenv("JSON_PATH"),
    llm_type='ollama', 
    embedding_type='sentence_transformer', 
    llm_model='llama3.2:latest', 
    embedding_model='all-mpnet-base-v2', 
    debug_mode=False, 
    doc_input=doc_input,
    doc_name='arwg-index', 
    chunking_method='SEMANTIC', 
    storage_type='PINECONE_EXISTING',
    template_name='detailed', 
    use_ground_truth=False,
    process_questions=True,
)

# Handle modes
if driver.mode == 'rag':
    handle_rag_mode(driver)
elif driver.mode == 'llm':
    handle_llm_mode(driver)