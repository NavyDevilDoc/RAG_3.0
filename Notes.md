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
    e. Run this line in the notebook or Python file once: nltk.download('stopwords')
    f. Recommend downloading Notepad++ because it's better than Mk 1 Mod 0 Notepad. Not essential, but you'll thank me later.


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


7. How to use the retrieval-augmented generation model:
    a. Driver.ipynb is meant to be the "one ring to rule them all". 

    b. In the first cell, there is a list of large language models (LLMs) and embedding models and it has all models currently on my computer. Given the Ollama LLM and embedding model downloads, you can customize it how you want. Note that using GPT-4o as your LLM requires your own OpenAI API key.
        1. Settable Parameters:
            a. env_path = r"" -> file path to your environment variable file
            b. llm_type=LLMType.OLLAMA -> can be OLLAMA or GPT
            c. embedding_type=EmbeddingType.SENTENCE_TRANSFORMER -> can be OLLAMA, GPT, or SENTENCE_TRANSFORMER
            d. llm_index=1 -> Corresponds to the LLM choices 
            e. embedding_index=2 -> Corresponds to the embedding model choices

    c. The second cell configures the document chunking scheme and loads the desired file.
        1. Settable Parameters:
            a. source_path = r"" -> file path to your source document; this must be a PDF
            b. chunking_method=ChunkingMethod.PAGE -> can be PAGE or SEMANTIC
                1. Note: PAGE is good for discrete pages of information such as a PowerPoint presentation that has been converted
                into a PDF. SEMANTIC is good for continuous pages of information such as a publication or doctrine document
            c. enable_preprocessing=False -> Can be False or True
                1. Note: In general, keep it set at False. This controls textual preprocessing separate from optical character
                recognition (OCR) and can have unintended side effects if set to True without a complete grasp of the document
                or code.

    d. The third cell configures the Pinecone database. Remember: You need your own API key for this. Their "free" tier is not
    at all expensive. Future versions will include ChromaDB since it's free. I have used Pinecone for most of the year and have
    been quite pleased with it. 
        1. Settable Parameters:
            a. doc_name='test-datastore' -> This is part of the Pinecone index's name. The code automatically configures the database (a.k.a."Index") as doc_name-embedding_model to help keep the databases as separate as possible. Note that you MUST only use the '-' character to separate words...no underscores. You are limited to 45 alpha-numeric characters for
            your index name.
        b. storage_type=StorageType.PINECONE_EXISTING -> Can be _NEW, _ADD, _EXISTING, or LOCAL_STORAGE
            1. _NEW: write your chunked documents to a new index. BE CAREFUL: THIS WILL OVERWRITE AN EXISTING INDEX OF THE SAME     NAME!
            2. _ADD: write new documents to an existing index. THIS WILL NOT OVERWRITE AN EXISTING INDEX
            3. _EXISTING: load an existing index into the "datastore" variable so the LLM knows where to look
            4. LOCAL_STORAGE: configures a basic vector datastore on your local machine instead of Pinecone

    e. The final cell sends questions to the pipeline.
        1. Settable Parameters:
            a. template_name = "detailed" -> Can be default, short, or detailed (more templates will be built over time)
            b. questions = [] -> Questions to be answered by the model. Ensure questions are in quotation marks and separated
            by a comma.
            c. use_ground_truth = False -> Can be True or False. Only set to True if answers to the questions currently being asked
            have been loaded into the ground_truth.json file. This is sent to the scoring class and will output BLEU and ROUGE scores to augment the confidence score.
