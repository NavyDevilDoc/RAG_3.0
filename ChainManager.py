# ChainManager.py

"""
ChainManager.py
A module for managing LangChain question-answering chains with vector datastores.

This module provides functionality to set up and manage retrieval-augmented 
generation (RAG) chains using LangChain components.
"""

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

class ChainManager:
    def __init__(self, datastore, model, template):

        """
        Initialize the ChainManager with required components.

        Args:
            datastore (Any): Vector datastore for document retrieval
            model (Any): Language model for text generation
            template (str): Prompt template for structuring queries

        Raises:
            ValueError: If any required component is invalid
        """
        self.datastore = datastore
        self.model = model
        self.template = template


    def setup_chain(self):
        """
        Set up and return the question-answering chain.

        Returns:
            Any: Configured LangChain chain ready for question answering

        Raises:
            RuntimeError: If chain setup fails
        """
        try:
            parser = StrOutputParser()
            prompt = PromptTemplate.from_template(self.template)
            retriever = self.datastore.as_retriever()
            chain = (
                {
                    "context": itemgetter("question") | retriever,
                    "question": itemgetter("question"),
                }
                | prompt
                | self.model
                | parser
            )
            return chain
        except Exception as e:
            raise RuntimeError(f"Error setting up chain: {e}")