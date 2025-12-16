import sys
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.gemini import GeminiEmbedding

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model
from exception import customexception
from logger import logging


def download_gemini_embedding(model, documents):
    """
    Initialize Gemini Embedding model and create vector index.

    Returns:
        QueryEngine
    """

    try:
        logging.info("Initializing Gemini embedding model")

        # Embedding model
        embed_model = GeminiEmbedding(
            model_name="models/gemini-embedding-001"
        )

        # Global settings (NEW API)
        Settings.llm = model
        Settings.embed_model = embed_model
        Settings.chunk_size = 800
        Settings.chunk_overlap = 20

        logging.info("Creating VectorStoreIndex")

        index = VectorStoreIndex.from_documents(documents)

        # Persist index
        index.storage_context.persist()

        logging.info("Index created and persisted successfully")

        return index.as_query_engine()

    except Exception as e:
        logging.error("Error in download_gemini_embedding", exc_info=True)
        raise customexception(e, sys)
