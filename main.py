# main.py
from Driver import Driver
from ResponseFormatter import ResponseFormatter
from TextPreprocessor import TextPreprocessor
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuration
env_path = r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\EDQP_RAG_Model\env_variables.env"

doc_input = [r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\Source_Documents\ED_Basic_Course_Guide\2.1.1_Command_Structures.pdf",
             r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\Source_Documents\ED_Basic_Course_Guide\2.1.2_NAVSEA_Organization.pdf",
             r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\Source_Documents\ED_Basic_Course_Guide\2.1.3_NAVWAR_Enterprise.pdf"]

output_dir = r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\EDQP_RAG_Model\Local_RAG_Model\RAG_Model\RAG_Outputs"

'''
Available choices for language models (LLMs)
GPT: 
    0 - gpt-4o
Ollama:
    0 - llama3.1:8b-instruct-q5_K_M
    1 - llama3.2:latest
    2 - mistral-nemo:12b-instruct-2407-q5_K_M

Available choices for embedding models
GPT:
    0 - text-embedding-3-small
    1 - text-embedding-3-large
Ollama:
    0 - nomic-embed-text
    1 - mxbai-embed-large
    2 - all-minilm
    3 - snowflake-arctic-embed
Sentence Transformer:
    0 - all-MiniLM-L6-v2
    1 - all-MiniLM-L12-v2
    2 - all-mpnet-base-v2
    3 - all-distilbert-base-v2
    4 - multi-qa-mpnet-base-dot-v1
'''

mode = 'rag' # 'llm' or 'rag'

# Initialize driver with all required parameters
driver = Driver(
    env_path=env_path,
    doc_input=doc_input,
    llm_type='ollama',
    embedding_type='SENTENCE_TRANSFORMER',
    llm_index=1,
    embedding_index=2,
    doc_name='test-index',
    chunking_method='HIERARCHICAL',
    storage_type='PINECONE_EXISTING',
    template_name='default',
    use_ground_truth=False,
    debug_mode=False,
    mode=mode
)

# Run the driver in the selected mode
if mode == 'llm': # LLM mode
    # Direct LLM query
    text_preprocessor = TextPreprocessor()
    question = "Who is the very model of a modern major general?"
    response = driver.process_query(question)
    print(f"\nQ: {question}")
    print(f"A: {response}")


elif mode == 'rag': # RAG mode
    # Define questions
    questions = [
        "What is the Goldwater-Nichols Reorganization Act?",
        "What is the vision of NAVSEA?",
        "What is the mission of NAVWAR?",
        "What year was the National Security Act passed?",
        "What year did the National Military Establishment become the Department of Defense?"
        ]
    # Run and get answers
    qa_responses = driver.run(questions) 
    # Save File
    formatter = ResponseFormatter(debug_mode=False)
    saved_path = formatter.save_to_file(qa_responses,
                                        "rag_responses",
                                        output_dir=output_dir)

    # Print formatted responses to console
    for response in qa_responses:
        print(formatter.format_response(response))