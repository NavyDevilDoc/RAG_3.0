# main.py
import os
from Setup import load_environment_variables, initiate_rag
from Driver import Driver
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# File path for environment variables and JSON file
file_path = r"C:\Users\docsp\Desktop\Test_RAG\file_variables.env"
load_environment_variables(file_path)

# Initialize driver with all required parameters
driver = Driver(
    # Mode Selection
    env_path=os.getenv("ENV_PATH"), 
    json_path=os.getenv("JSON_PATH"),
    llm_type='ollama',
    embedding_type='sentence_transformer',
    llm_model='granite3.1-dense:8b',
    embedding_model='ibm-granite/granite-embedding-30m-english',
    doc_name='edqp-datastore-granite-30m-pg-test', 
    chunking_method='PAGE', 
    storage_type='local_storage',
    template_name='assistant', 
    use_ground_truth=False,
    process_questions=True,
)

initiate_rag(driver)