# main.py
from Driver import Driver
from ResponseFormatter import ResponseFormatter
from TextPreprocessor import TextPreprocessor
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# File path for environment variables
env_path = r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\EDQP_RAG_Model\env_variables.env"

# List of document paths
doc_input = [r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\Source_Documents\ED_Basic_Course_Guide\2.1.1_Command_Structures.pdf",
             r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\Source_Documents\ED_Basic_Course_Guide\2.1.2_NAVSEA_Organization.pdf",
             r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\Source_Documents\ED_Basic_Course_Guide\2.1.3_NAVWAR_Enterprise.pdf"]

# Output directory for RAG responses
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

# Select the mode to run the driver in. Can be:
#    'llm' - Send individual queries to the selected large language model
#    'rag' - Run the RAG model on a list of questions
mode = 'rag'


# Initialize driver with all required parameters
driver = Driver(
    # Path to environment variables file
    env_path=env_path, 

    # List of document paths
    doc_input=doc_input, 

    # Select OpenAI model ('gpt') or Ollama open-source model ('ollama') and is case-insensitive
    llm_type='ollama', 

    # Select the embedding model. Can be 'gpt', 'ollama', or 'sentence_transformer' and is case-insensitive
    embedding_type='SENTENCE_TRANSFORMER', 

    # See chart above for available choices
    llm_index=1, 

    # See chart above for available choices
    embedding_index=2, 

    # Set the Pinecone index name
    doc_name='test-index', 

    # Select the document chunking method to use. Can be:
    #    'PAGE'         - chunks each page separately
    #    'SEMANTIC'     - chunks based on semantic similarity
    #    'HIERARCHICAL' - chunks based on hierarchical structure
    chunking_method='HIERARCHICAL', 

    # Select the database action. Can be: 
    #    '_NEW'           - build a new datastore; using this will erase a datastore with the same index name/embedding model combo
    #    '_ADD'           - add to an existing datastore; will not erase any data
    #    '_EXISTING'      - use an existing datastore; will not erase any data
    #    '_LOCAL_STORAGE' - use local storage for the datastore
    storage_type='PINECONE_EXISTING',

    # Select a pre-defined template for the response. Can be:
    #    'default'  - standard response template
    #    'short'    - short response template
    #    'detailed' - detailed response template
    template_name='default', 

    # Load the ground truth file. Can be True or False; keep set to false unless the ground truth file has been updated
    use_ground_truth=False, 

    # Switch to debugging mode. Can be True or False; keep set to false for standard use
    debug_mode=False, 

    mode=mode
)

# Run the driver in the selected mode

# LLM mode
if mode == 'llm': 
    # Initialize the text preprocessor module
    text_preprocessor = TextPreprocessor()

    # Enter the input query
    input = "Generate a recipe for homemade apple pie. Include ingredients and detailed instructions."

    # Process the input query and display the response
    response = driver.process_query(input)
    print(f"\nQ: {input}")
    print(f"A: {response}")

# RAG mode
elif mode == 'rag': 
    # Define question(s)
    questions = [
        "What is the Goldwater-Nichols Reorganization Act?",
        "What is the vision of NAVSEA?",
        "What is the mission of NAVWAR?",
        "What year was the National Security Act passed?",
        "What year did the National Military Establishment become the Department of Defense?"
        ]
    
    # Run the model and and display the responses
    qa_responses = driver.run(questions) 
    
    # Save the question/answer pairs to a text file
    formatter = ResponseFormatter(debug_mode=False)
    saved_path = formatter.save_to_file(qa_responses,
                                        "rag_responses",
                                        output_dir=output_dir)
    for response in qa_responses:
        print(formatter.format_response(response))