"""
Phase 3B — Vector Store Setup (ChromaDB)

This module handles vector storage, indexing, and retrieval
using ChromaDB.
"""

import os
import logging
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.ingestion.vectorstore.embeddings import get_embedding_model


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
# Resolved base directory to avoid relative path issues in Docker/production
_current_dir = os.path.dirname(os.path.abspath(__file__))
# Moves up from src/ingestion/vectorstore/ to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, "../../../"))

COLLECTION_NAME = "atliq_docs"
PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

# Logger
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# ChromaDB Client
# ──────────────────────────────────────────────
def get_chroma_client(mode: str = "persistent") -> chromadb.ClientAPI:
    """
    Get a ChromaDB client in either persistent or in-memory mode.

    Priority: function parameter > CHROMA_MODE env var > default ("persistent").

    Args:
        mode (str): "persistent" or "memory".

    Returns:
        chromadb.ClientAPI: ChromaDB client instance.

    Raises:
        ConnectionError: If client initialization fails.
        ValueError: If an invalid mode is provided.
    """
    # Resolve mode: function param takes priority, then env var
    env_mode = os.environ.get("CHROMA_MODE", "").lower()
    resolved_mode = mode if mode != "persistent" or not env_mode else env_mode

    # Override: if the caller explicitly passed something other than default,
    # always honor the function parameter
    if mode != "persistent":
        resolved_mode = mode

    try:
        if resolved_mode == "persistent":
            logger.info(f"Connecting to ChromaDB in PERSISTENT mode at: {PERSIST_DIR}")
            client = chromadb.PersistentClient(path=PERSIST_DIR)

        elif resolved_mode == "memory":
            logger.info("Connecting to ChromaDB in IN-MEMORY (ephemeral) mode.")
            client = chromadb.EphemeralClient()

        else:
            raise ValueError(
                f"Invalid ChromaDB mode: '{resolved_mode}'. "
                "Use 'persistent' or 'memory'."
            )

        logger.info("ChromaDB client initialized successfully.")
        return client

    except ValueError:
        raise  # re-raise ValueError as-is

    except Exception as e:
        logger.exception("Failed to connect to ChromaDB.")
        raise ConnectionError(
            f"Error connecting to ChromaDB in '{resolved_mode}' mode: {str(e)}"
        ) from e


# ──────────────────────────────────────────────
# Build Vector Store (Full Indexing)
# ──────────────────────────────────────────────
def build_vectorstore(
    documents: list[Document],
    mode: str = "persistent",
) -> Chroma:
    """
    Build a ChromaDB vector store from a list of LangChain Documents.

    This function:
      1. Initializes the embedding model.
      2. Connects to ChromaDB.
      3. Deletes the existing collection (if any) for a clean re-index.
      4. Creates a new collection and indexes all documents.

    Args:
        documents (list[Document]): LangChain Document objects to index.
        mode (str): ChromaDB mode — "persistent" or "memory".

    Returns:
        Chroma: The built LangChain Chroma vectorstore.

    Raises:
        RuntimeError: If the vectorstore build process fails.
    """
    if not documents:
        raise ValueError("No documents provided. Cannot build vectorstore.")

    try:
        # Step 1: Get embedding model
        logger.info("Loading embedding model...")
        embedding_model = get_embedding_model()

        # Step 2: Get ChromaDB client
        client = get_chroma_client(mode)

        # Step 3: Delete existing collection if it exists
        existing_collections = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in existing_collections:
            logger.warning(
                f"Collection '{COLLECTION_NAME}' already exists. Deleting for clean re-index..."
            )
            client.delete_collection(name=COLLECTION_NAME)
            logger.info(f"Deleted existing collection: '{COLLECTION_NAME}'")

        # Step 4: Create new vectorstore from documents
        logger.info(f"Indexing {len(documents)} documents into collection '{COLLECTION_NAME}'...")

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=COLLECTION_NAME,
            client=client,
        )

        # Verify
        collection = client.get_collection(name=COLLECTION_NAME)
        doc_count = collection.count()
        logger.info(f"Vector store built successfully. Documents indexed: {doc_count}")

        return vectorstore

    except ValueError:
        raise  # re-raise validation errors

    except Exception as e:
        logger.exception("Failed to build vector store.")
        raise RuntimeError(
            f"Error building vector store: {str(e)}"
        ) from e


# ──────────────────────────────────────────────
# Load Existing Vector Store (No Re-indexing)
# ──────────────────────────────────────────────
def load_vectorstore(mode: str = "persistent") -> Chroma:
    """
    Load an existing ChromaDB vector store (without re-indexing).

    Use this during chatbot runtime when the vectorstore is already built.

    Args:
        mode (str): ChromaDB mode — "persistent" or "memory".

    Returns:
        Chroma: The existing LangChain Chroma vectorstore.

    Raises:
        RuntimeError: If the collection doesn't exist or loading fails.
    """
    try:
        client = get_chroma_client(mode)

        # Verify the collection exists
        existing_collections = [c.name for c in client.list_collections()]
        if COLLECTION_NAME not in existing_collections:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found. "
                "Run build_vectorstore() first."
            )

        embedding_model = get_embedding_model()

        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=PERSIST_DIR,
        )

        # Basic verification: check if collection exists and has documents
        collection = vectorstore._collection
        doc_count = collection.count()

        if doc_count == 0:
             logger.warning("Vector store loaded but collection is empty. Forcing re-index.")
             raise RuntimeError("Empty vector store.")

        # --- VALIDATION SEARCH ---
        # Some ChromaDB errors (like KeyError: '_type') only appear during search.
        # We perform a tiny search to ensure the vector store is actually healthy.
        logger.info("Validating vector store health with a test search...")
        try:
            vectorstore.similarity_search("health check", k=1)
            logger.info("Vector store health check passed.")
        except Exception as search_error:
            logger.error(f"Vector store health check failed: {search_error}")
            raise search_error

        logger.info(f"Loaded existing vector store. Documents in collection: {doc_count}")
        return vectorstore

    except Exception as e:
        logger.warning(f"Failed to load vector store: {e}. Attempting automatic re-index...")
        
        try:
            # Step 1: Clear the corrupted persist directory
            if os.path.exists(PERSIST_DIR):
                import shutil
                shutil.rmtree(PERSIST_DIR)
                logger.info(f"Cleared corrupted persist directory: {PERSIST_DIR}")

            # Step 2: Trigger fresh ingestion
            from src.ingestion.loader import load_and_split
            documents = load_and_split()
            
            if not documents:
                raise ValueError("No documents found in Data/ folder to re-index.")

            # Step 3: Rebuild
            vectorstore = build_vectorstore(documents, mode=mode)
            logger.info("Vector store rebuilt successfully after failure.")
            return vectorstore

        except Exception as rebuild_error:
            logger.exception("Automatic re-index failed.")
            raise RuntimeError(
                f"Critical Error: Vector store is corrupted and re-indexing failed. "
                f"Original error: {str(e)}. Re-index error: {str(rebuild_error)}"
            ) from rebuild_error


# ──────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────
def get_retriever(
    department: Optional[str] = None,
    mode: str = "persistent",
):
    """
    Get a LangChain retriever from the existing ChromaDB vector store.

    Args:
        department (str, optional): If specified, filter results to only
            this department. e.g. "HR", "Finance".
        mode (str): ChromaDB mode — "persistent" or "memory".

    Returns:
        LangChain Retriever: A retriever that can be used in a RAG chain.

    Raises:
        RuntimeError: If the vector store cannot be loaded.
    """
    try:
        vectorstore = load_vectorstore(mode)

        search_kwargs = {"k": 5}

        if department:
            search_kwargs["filter"] = {"department": department}
            logger.info(f"Creating retriever with department filter: '{department}'")
        else:
            logger.info("Creating retriever without department filter (full collection).")

        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

        logger.info("Retriever created successfully.")
        return retriever

    except Exception as e:
        logger.exception("Failed to create retriever.")
        raise RuntimeError(
            f"Error creating retriever: {str(e)}"
        ) from e


# ──────────────────────────────────────────────
# Main — Full Pipeline Test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    )

    print("=" * 60)
    print("Phase 3B — Vector Store Setup (ChromaDB)")
    print("=" * 60)

    try:
        # Step 1: Ingest documents
        from src.ingestion.loader import load_and_split

        print("\n[1/4] Running ingestion pipeline...")
        chunks = load_and_split()
        print(f"       Chunks from ingestion: {len(chunks)}")

        # Step 2: Build vector store
        print("\n[2/4] Building vector store...")
        vectorstore = build_vectorstore(chunks, mode="persistent")

        # Print stats
        client = get_chroma_client("persistent")
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"       Documents indexed: {collection.count()}")

        # Step 3: Test retriever WITHOUT department filter
        print("\n[3/4] Testing retriever (no filter)...")
        retriever = get_retriever(department=None, mode="persistent")
        results = retriever.invoke("What is the leave policy?")

        print(f"       Retrieved {len(results)} chunks:")
        for i, doc in enumerate(results[:3], 1):
            print(f"\n       --- Chunk {i} ---")
            print(f"       Department: {doc.metadata.get('department', 'N/A')}")
            print(f"       Source:     {doc.metadata.get('source', 'N/A')}")
            print(f"       Content:    {doc.page_content[:200]}...")

        # Step 4: Test retriever WITH department filter
        print("\n[4/4] Testing retriever (department='HR')...")
        try:
            retriever_hr = get_retriever(department="hr", mode="persistent")
            results_hr = retriever_hr.invoke("What is the leave policy?")

            print(f"       Retrieved {len(results_hr)} chunks:")
            for i, doc in enumerate(results_hr[:3], 1):
                print(f"\n       --- Chunk {i} ---")
                print(f"       Department: {doc.metadata.get('department', 'N/A')}")
                print(f"       Source:     {doc.metadata.get('source', 'N/A')}")
                print(f"       Content:    {doc.page_content[:200]}...")
        except Exception as e:
            print(f"       Filtered retriever test skipped: {e}")
            print("       (This is OK if no 'HR' department exists in your data.)")

        print("\n" + "=" * 60)
        print("Vector store built and tested successfully.")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\nERROR: {e}")
