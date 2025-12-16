from llama_index.core import SimpleDirectoryReader
import sys
from exception import customexception
from logger import logging

def load_data(data):
    """
    
    Load PDF document from a specified directoty

    Parameters:
    - data (str): The path to the directory conatiningn PDF files.

    Returns:
    - A list of loaded PDF documents. The specific types of documents may vary.

    """

    try:
        logging.info("Data Loading Started.....")
        loader = SimpleDirectoryReader("Data")
        document = loader.load_data()
        logging.info("Data Loading Completed.....")
        return document
    
    except Exception as e:
        logging.info("Exception in loading data.....")
        raise customexception(e, sys)
    
