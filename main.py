# main.py
import os
from Setup import load_environment_variables, handle_rag_mode, handle_llm_mode
from Driver import Driver
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# File path for environment variables and JSON file
file_path = r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\RAG_Model\file_variables.env"
load_environment_variables(file_path)

"""
Available Ollama LLMs:
    llama3.2:latest
    llama3.2-vision:latest
    llama3.3:70b-instruct-q2_K
    codestral:latest
    codellama:13b
    granite3.1-dense:8b
        embedding_type = sentence_transformer
        embedding_model = ibm-granite/granite-embedding-30(or 125)m-english
    granite-code:20b
    granite-code:8b
"""

# Initialize driver with all required parameters
driver = Driver(
    # Mode Selection
    mode='rag',
    env_path=os.getenv("ENV_PATH"), 
    json_path=os.getenv("JSON_PATH"),
    # Language and Embedding Model Selection
    llm_type='gpt',
    embedding_type='gpt',
    llm_model='gpt-4o',
    embedding_model='text-embedding-3-small',
    # RAG-specific Parameters
    debug_mode=False, 
    doc_name='test-index', 
    chunking_method='PAGE', 
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