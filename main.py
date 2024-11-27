# main.py
import os
from Driver import Driver
from ResponseFormatter import ResponseFormatter
from TextPreprocessor import TextPreprocessor
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def find_pdfs_in_folder(folder_path: str) -> list:
    """Find all PDF files in the specified folder and return their file paths."""
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        print(f"Found {len(files)} files in {root}")
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

# File path for environment variables
env_path = r""


# Flag to choose between manual input and automatic folder search. Can be True for manual input or False for automatic search
use_manual_input = False

if use_manual_input:
    # List of document paths; useful for a small number of documents
    doc_input = []
else:
    # Specify the folder containing the PDF documents
    pdf_folder_path = r""
    
    # Use the helper function to find all PDF files in the specified folder
    doc_input = find_pdfs_in_folder(pdf_folder_path)


# Output directory for RAG responses
output_dir = r""

'''
## NOTE: Remember that you need your own OpenAI API key to use a GPT model. You might also have different Ollama models loaded on your computer. Adjust as needed. ##

Available choices for large language models 
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
Sentence Transformer:
    0 - all-MiniLM-L6-v2
    1 - all-MiniLM-L12-v2
    2 - all-mpnet-base-v2
    3 - all-distilbert-base-v2
    4 - multi-qa-mpnet-base-dot-v1
'''


# Initialize driver with all required parameters
driver = Driver(

    # Select the mode to run the driver in. Can be:
    #   'llm' - Send individual queries to the selected large language model
    #   'rag' - Run the RAG model on a list of questions
    mode='llm',


                                    ## Set for both modes ##

    # Path to environment variables file
    env_path=env_path, 
    # Select OpenAI model ('GPT') or Ollama open-source model ('OLLAMA') and is case-insensitive
    llm_type='ollama', 
    # Select the embedding model. Can be 'GPT'or 'SENTENCE_TRANSFORMER' and is case-insensitive
    embedding_type='sentence_transformer', 
    # Enter the number corresponding to the desired models; see chart above for available choices
    llm_model='llama3.2:latest', 
    embedding_model='all-mpnet-base-v2', 
    # Switch to debugging mode. Can be True or False; keep set to false for standard use
    debug_mode=False, 


                                    ## Set for RAG mode ##

    # List of document paths
    doc_input=doc_input,
    # Set the Pinecone index name
    doc_name='edqp-index', 
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
)


## Run the driver in the selected mode ##
# LLM mode
if driver.mode == 'llm': 
    # Initialize the text preprocessor module
    text_preprocessor = TextPreprocessor()

    # Enter the input query
    input = "Generate a recipe for homemade apple pie. Include ingredients and detailed instructions."

    # Process the input query and display the response
    response = driver.process_query(input)
    print(f"\nQ: {input}")
    print(f"A: {response}")

# RAG mode
elif driver.mode == 'rag': 
    # Define question(s)
    questions = [
        #"What is the Goldwater-Nichols Reorganization Act?",
        #"What is the vision of NAVSEA?",
        #"What is the mission of NAVWAR?",
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
