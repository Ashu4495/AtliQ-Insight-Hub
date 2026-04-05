"""
Phase 2 — Data Ingestion Pipeline
File: src/ingestion/loader.py

This module:
- Walks through Data/ directory
- Loads CSV and Markdown files
- Adds metadata (department, source)
- Splits documents into chunks
- Returns chunked documents
"""

import os
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from langchain_community.document_loaders import (
    CSVLoader,
    UnstructuredMarkdownLoader,
    TextLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("Data")


def load_documents() -> List[Document]:
    """
    Walk through Data directory and load documents
    """
    documents = []

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    for department_folder in DATA_DIR.iterdir():
        if not department_folder.is_dir():
            continue

        department = department_folder.name
        logger.info(f"Loading department: {department}")

        for file_path in department_folder.rglob("*"):
            if not file_path.is_file():
                continue

            try:
                # CSV Files
                if file_path.suffix.lower() == ".csv":
                    loader = CSVLoader(
                        file_path=str(file_path),
                        encoding="utf-8"
                    )
                    docs = loader.load()

                # Markdown Files
                elif file_path.suffix.lower() == ".md":
                    try:
                        loader = UnstructuredMarkdownLoader(
                            str(file_path),
                            mode="elements"
                        )
                        docs = loader.load()
                    except Exception as e:
                        logger.warning(
                            f"Markdown parsing failed ({file_path}): {e} — using TextLoader fallback"
                        )
                        loader = TextLoader(
                            str(file_path),
                            encoding="utf-8"
                        )
                        docs = loader.load()

                else:
                    continue

                # Add metadata
                for doc in docs:
                    doc.metadata["department"] = department
                    doc.metadata["source"] = str(file_path)

                documents.extend(docs)

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks
    """

    logger.info("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    logger.info(f"Created {len(chunks)} chunks")

    return chunks


def load_and_split() -> List[Document]:
    """
    Full ingestion pipeline
    """

    logger.info("Starting data ingestion...")

    docs = load_documents()
    logger.info(f"Loaded {len(docs)} documents")

    chunks = split_documents(docs)

    return chunks


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 2 — Data Ingestion Pipeline")
    logger.info("=" * 60)

    chunks = load_and_split()

    if chunks:
        sample = chunks[0]

        logger.info("Sample Chunk:")
        logger.info("-" * 60)

        logger.info("Content Preview:")
        logger.info(sample.page_content[:500])

        logger.info("Metadata:")
        logger.info(sample.metadata)

    logger.info(f"Total Chunks: {len(chunks)}")
    logger.info("=" * 60)


