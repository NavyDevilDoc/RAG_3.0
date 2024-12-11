Program Notes

1. Development and Testing Environment:
    a. OS: Windows
    b. Hardware: 
        1. GPU: NVIDIA RTX 4060i (20 GB RAM)
        2. CPU: Intel i9 2 TB SSD w/ 40 GB RAM
    c. Average OCR/Chunking Times:
        1. PAGE: 2 min 15 sec
        2. SEMANTIC: 3 min
    d. AVerage Model Execution Times:
        1. GPT: 1 min 5 sec
        2. OLLAMA: 5 min 30 sec


2. Necessary downloads:
    a. Ollama (https://ollama.com/download)
        1. llama3.1:8b-instruct-q5_K_M
        2. llama3.2:latest
        3. mistral-nemo:12b-instruct-2407-q5_K_M
        4. nomic-embed-text
        5. mxbai-embed-large
        6. all-minilm
        7. snowflake-arctic-embed
    b. PyTesseract (https://tesseract-ocr.github.io/tessdoc/Installation.html)
    c. Poppler (must be a Path variable and is used with pdf2image package, https://github.com/oschwartz10612/poppler-windows/releases/tag/v24.08.0-0)
    d. Run this line in the terminal: spacy download en_core_web_sm
    e. Run this line in the notebook or Python file once: nltk.download()
    f. Run this line in the notebook or Python file once: nltk.download('stopwords')
    g. Recommend downloading Notepad++ because it's better than Mk 1 Mod 0 Notepad. Not essential, but you'll thank me later.


3. Necessary API Keys:
    a. OpenAI 
    b. Pinecone
    c. Best Practice: Create an environment variable file. This is a text file with your API keys stored inside. When saving the
    file, something like "env_variables.env" works just fine. The "dotenv" Python package is used to locate the environment
    variable file and automatically load them for later use. In the file extension drop-down menu on the file saving screen, 
    scroll to the bottom and select "No Extension". If you need to modify the file, go to the folder where it sits, right
    click, and select "Edit with Notepad".


4. Sentence Transformers: 
    a. Embedding models can be found here: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#original-models
    b. The models are plug-and-play; lower values in the "Speed" column translate into longer loading times when running


5. JSON Files:
    a. Templates - store all callable templates for the LLM
    b. Ground Truth - answers to the questions; used for BLEU and ROUGE scoring


6. Recommend creating a virtual environment for the necessary packages. From the terminal:
    a. py -m venv venv
    b. .\venv\Scripts\activate
    c. pip install -r requirements.txt


7. How to use:
    a. The refactored code has two operating modes:
        1. RAG - Process, store, and retrieve documents for additional LLM context
        2. LLM - Standard query to the selected large language model

    b. Inputs:
        1. env_path: path to your environment variable file

        2. use_manual_input: 
            a. True: paste individual file paths for loading and processing into the vector database
            b. False: search through a folder and load all PDFs found

        3. doc_input: path to individual files (first instance, separated by commas) or folder (second instance)

        4. save_file: name of the text file with saved and formatted RAG responses

        5. output_dir: path to save the RAG output, which consists of question/answer pairs

        6. mode: selects the RAG model or LLM interface (case-sensitive)
            a. 'rag'
            b. 'llm'

        7. llm_type: selects the type of LLM to use (case-insensitive)
            a. 'gpt'    - uses the OpenAI API key to access an OpenAI LLM
            b. 'ollama' - accesses one of the downloaded open-source LLMs

        8. embedding_type: selects the type of embedding model to use (case-insensitive)
            a. 'gpt'                  - uses the OpenAI API key to access an OpenAI embedding model
            b. 'ollama'               - accesses one of the downloaded open-source embedding models
            c. 'sentence_transformer' - access a pre-trained embedding model through the Hugging Face library

        9. llm_index: selects the model to use

        10. embedding_index: selects the embedding model to use

        11. doc_name: index name sent to Pinecone that, along with the embedding model, defines the vector database's name

        12. chunking_method: document splitting scheme
            a. PAGE         -  splits documents at the page boundary
            b. SEMANTIC     -  splits documents based on semantic meaning of chunks
            c. HIERARCHICAL -  hybrid method that splits pages by semantic meaning

        13. storage_type: defines how to interface with Pinecone
            a. PINECONE_NEW      - sets up a new datastore and overwrites an existing datastore with the same name
            b. PINECONE_ADD      - adds documents to an existing datastore without overwriting existing documents
            c. PINECONE_EXISTING - loads an existing datastore without overwriting or adding new documents
            d. LOCAL_STORAGE     - experimental version of storing data locally in a modified SQLite database

        14. template_name: selects the RAG model's question answering template
            a. 'default'  - loads the default template
            b. 'short'    - loads the short response template
            c. 'detailed' - loads the detailed response template

        15. use_ground_truth: 
            a. True: load the ground truth file to calculate advanced metrics; only use if the file has been updated with current question/answer pairs
            b. False: skip ground truth calculations

        16. debug_mode:
            a. True: enter debugging mode
            b. False: normal operations