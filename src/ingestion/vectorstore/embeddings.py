"""
Phase 3A — Embedding Setup

This module initializes and provides the embedding model
used across the project for generating vector representations.
"""

import logging
from typing import List, Optional

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    raise ImportError(
        "Required package 'langchain-huggingface' is not installed. "
        "Install it using: pip install langchain-huggingface"
    ) from e


# Logger setup
logger = logging.getLogger(__name__)

# Cached model and its name
_embedding_model: Optional[HuggingFaceEmbeddings] = None
_current_model_name: Optional[str] = None


def get_embedding_model(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> HuggingFaceEmbeddings:
    """
    Initialize and return a cached HuggingFace embedding model.

    Args:
        model_name (str): Name of the HuggingFace model to load.

    Returns:
        HuggingFaceEmbeddings: Initialized embedding model.

    Raises:
        RuntimeError: If model initialization fails.
    """
    global _embedding_model, _current_model_name

    if _embedding_model is not None and _current_model_name == model_name:
        logger.debug("Returning cached embedding model.")
        return _embedding_model

    try:
        logger.info(f"Initializing embedding model: {model_name}")

        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # Safe fallback
            encode_kwargs={"normalize_embeddings": True},
        )

        _current_model_name = model_name

        logger.info("Embedding model initialized successfully.")
        return _embedding_model

    except Exception as e:
        logger.exception("Failed to initialize embedding model.")
        raise RuntimeError(
            f"Error initializing embedding model '{model_name}': {str(e)}"
        ) from e


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts (List[str]): List of input text strings.

    Returns:
        List[List[float]]: List of embedding vectors.

    Raises:
        RuntimeError: If embedding generation fails.
    """
    try:
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of strings.")

        # Clean input texts
        cleaned_texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]

        if not cleaned_texts:
            logger.warning("No valid texts provided for embedding generation.")
            return []

        model = get_embedding_model()
        embeddings = model.embed_documents(cleaned_texts)

        logger.debug(f"Generated embeddings for {len(cleaned_texts)} texts.")
        return embeddings

    except Exception as e:
        logger.exception("Failed to generate embeddings.")
        raise RuntimeError(
            f"Error generating embeddings: {str(e)}"
        ) from e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        # Initialize model
        model = get_embedding_model()

        # Sample texts
        sample_texts = [
            "What is the leave policy?",
            "How do I reset my password?",
            "Tell me about onboarding process.",
        ]

        # Generate embeddings
        embeddings = generate_embeddings(sample_texts)

        if embeddings:
            dimension = len(embeddings[0])
            print(f"\nEmbedding dimension: {dimension}\n")

            for i, emb in enumerate(embeddings):
                print(f"Text {i+1} first 5 values: {emb[:5]}")

        print("\nEmbedding model loaded successfully.")

    except Exception as e:
        logger.error(f"Error in standalone execution: {e}")