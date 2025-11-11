"""
Word2Vec CBOW Embedding Training Pipeline

This module provides a refactored, Pythonic approach to training Word2Vec CBOW embeddings
using tokenized Polish language corpora.
"""

import json
import logging
from dataclasses import dataclass

from typing import Tuple, List, Optional, Dict

import numpy as np
from gensim.models import Word2Vec
from tokenizers import Tokenizer

from corpora import CORPORA_FILES  # type: ignore

# Configure logging for gensim
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Special tokens to filter during word vector computation
SPECIAL_TOKENS = {'[CLS]', '<s>', '[SEP]', '</s>', 'Ġ'}


@dataclass
class EmbeddingConfig:
    """Configuration for Word2Vec CBOW embedding training."""

    # File paths
    tokenizer_file: str = "../tokenizer/tokenizers/tokenizer-all-corpora-50k.json"
    output_tensor_file: str = "embedding_tensor_cbow.npy"
    output_map_file: str = "embedding_token_to_index_map.json"
    output_model_file: str = "embedding_word2vec_cbow_model.model"
    corpus_type: str = "ALL"  # Options: "WOLNELEKTURY", "PAN_TADEUSZ", "ALL"

    # Model hyperparameters
    vector_size: int = 100
    window: int = 3
    min_count: int = 5
    workers: int = 4
    epochs: int = 20
    sample: float = 1e-2
    sg: int = 0  # 0 for CBOW, 1 for Skip-gram

    @property
    def corpus_files(self) -> List[str]:
        """Get corpus files based on corpus type."""
        return CORPORA_FILES[self.corpus_type]


def load_tokenizer(tokenizer_file: str) -> Tokenizer:
    """
    Load a BPE tokenizer from file.

    Args:
        tokenizer_file: Path to the tokenizer JSON file

    Returns:
        Loaded Tokenizer instance

    Raises:
        FileNotFoundError: If tokenizer file does not exist
    """
    logger.info(f"Loading tokenizer from: {tokenizer_file}")
    try:
        tokenizer = Tokenizer.from_file(tokenizer_file)
        logger.info("Tokenizer loaded successfully")
        return tokenizer
    except FileNotFoundError as e:
        logger.error(f"Tokenizer file not found: {tokenizer_file}")
        raise


def aggregate_raw_sentences(files: List[str]) -> List[str]:
    """
    Load and aggregate raw sentences from corpus files.

    Args:
        files: List of file paths to load

    Returns:
        List of non-empty sentences from all files

    Raises:
        ValueError: If no sentences could be loaded from any file
    """
    raw_sentences = []
    logger.info(f"Loading text from {len(files)} files...")

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                raw_sentences.extend(lines)
                logger.debug(f"Loaded {len(lines)} lines from {file_path}")
        except FileNotFoundError:
            logger.warning(f"File not found, skipping: {file_path}")
            continue

    if not raw_sentences:
        raise ValueError("No sentences could be loaded from input files")

    logger.info(f"Total sentences loaded: {len(raw_sentences)}")
    return raw_sentences


def tokenize_sentences(
    raw_sentences: List[str],
    tokenizer: Tokenizer
) -> List[List[str]]:
    """
    Tokenize sentences using BPE tokenizer.

    Args:
        raw_sentences: List of raw sentence strings
        tokenizer: Tokenizer instance to use

    Returns:
        List of tokenized sentences (each sentence is a list of tokens)
    """
    logger.info(f"Tokenizing {len(raw_sentences)} sentences...")
    encodings = tokenizer.encode_batch(raw_sentences)

    # Convert Encoding objects to list of token lists
    tokenized_sentences = [encoding.tokens for encoding in encodings]

    logger.info(f"Prepared {len(tokenized_sentences)} sequences for training")
    return tokenized_sentences


# ============================================================================
# STAGE 1: Data Loading and Preparation
# ============================================================================

def load_and_prepare_data(
    config: EmbeddingConfig,
    tokenizer: Tokenizer
) -> List[List[str]]:
    """
    ETAP 1: Load corpus files and prepare tokenized sentences.

    Args:
        config: EmbeddingConfig instance with file paths and parameters
        tokenizer: Pre-loaded Tokenizer instance

    Returns:
        List of tokenized sentences ready for model training
    """
    logger.info("=" * 70)
    logger.info("STAGE 1: Loading and Preparing Data")
    logger.info("=" * 70)

    raw_sentences = aggregate_raw_sentences(config.corpus_files)
    tokenized_sentences = tokenize_sentences(raw_sentences, tokenizer)

    return tokenized_sentences


# ============================================================================
# STAGE 2: Model Training
# ============================================================================

def train_word2vec_model(
    tokenized_sentences: List[List[str]],
    config: EmbeddingConfig
) -> Word2Vec:
    """
    ETAP 2: Train Word2Vec CBOW model.

    Args:
        tokenized_sentences: List of tokenized sentences
        config: EmbeddingConfig instance with model hyperparameters

    Returns:
        Trained Word2Vec model
    """
    logger.info("=" * 70)
    logger.info("STAGE 2: Training Word2Vec CBOW Model")
    logger.info("=" * 70)
    logger.info(
        f"Training with: vector_size={config.vector_size}, window={config.window}, "
        f"min_count={config.min_count}, epochs={config.epochs}"
    )

    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=config.vector_size,
        window=config.window,
        min_count=config.min_count,
        workers=config.workers,
        sg=config.sg,  # 0: CBOW, 1: Skip-gram
        epochs=config.epochs,
        sample=config.sample,
    )

    logger.info("Training completed successfully")
    return model


# ============================================================================
# STAGE 3: Export and Save Results
# ============================================================================

def save_model_artifacts(
    model: Word2Vec,
    config: EmbeddingConfig
) -> Dict[str, str]:
    """
    ETAP 3: Export and save model artifacts.

    Saves:
    - Embedding matrix as NumPy array (.npy)
    - Token-to-index mapping as JSON
    - Full Word2Vec model

    Args:
        model: Trained Word2Vec model
        config: EmbeddingConfig instance with output file paths

    Returns:
        Dictionary with keys: 'tensor', 'map', 'model' and their file paths
    """
    logger.info("=" * 70)
    logger.info("STAGE 3: Exporting and Saving Results")
    logger.info("=" * 70)

    # 1. Save embedding tensor
    embedding_matrix = np.array(model.wv.vectors, dtype=np.float32)
    np.save(config.output_tensor_file, embedding_matrix)
    logger.info(
        f"Embedding matrix saved: {config.output_tensor_file} "
        f"(shape: {embedding_matrix.shape})"
    )

    # 2. Save token-to-index mapping
    token_to_index = {
        token: model.wv.get_index(token)
        for token in model.wv.index_to_key
    }
    with open(config.output_map_file, "w", encoding="utf-8") as f:
        json.dump(token_to_index, f, ensure_ascii=False, indent=4)
    logger.info(f"Token-to-index mapping saved: {config.output_map_file}")

    # 3. Save full model
    model.save(config.output_model_file)
    logger.info(f"Full Word2Vec model saved: {config.output_model_file}")

    return {
        'tensor': config.output_tensor_file,
        'map': config.output_map_file,
        'model': config.output_model_file,
    }


# ============================================================================
# Utility Functions for Word Vector Analysis
# ============================================================================

def get_word_vector_and_similar(
    word: str,
    tokenizer: Tokenizer,
    model: Word2Vec,
    topn: int = 20
) -> Tuple[Optional[np.ndarray], Optional[List[Tuple[str, float]]]]:
    """
    Get averaged word vector and find most similar tokens.

    Tokenizes the word using BPE, retrieves individual token vectors,
    and returns their average along with most similar tokens.

    Args:
        word: Word to compute vector for
        tokenizer: BPE tokenizer instance
        model: Trained Word2Vec model
        topn: Number of similar words to return

    Returns:
        Tuple of (word_vector, similar_tokens) or (None, None) if word
        cannot be represented
    """
    # Tokenize word with spaces to preserve BPE context
    encoding = tokenizer.encode(" " + word + " ")
    word_tokens = [t.strip() for t in encoding.tokens if t.strip()]

    # Remove special tokens from start and end
    if word_tokens and word_tokens[0] in SPECIAL_TOKENS:
        word_tokens = word_tokens[1:]
    if word_tokens and word_tokens[-1] in SPECIAL_TOKENS:
        word_tokens = word_tokens[:-1]

    valid_vectors = []
    missing_tokens = []

    # Collect vectors for each token
    for token in word_tokens:
        if token in model.wv:
            valid_vectors.append(model.wv[token])
        else:
            missing_tokens.append(token)

    if not valid_vectors:
        logger.warning(
            f"No valid vectors found for word '{word}' "
            f"(tokens: {word_tokens}, min_count={model.min_count})"
        )
        return None, None

    # Average vectors to get word representation
    word_vector = np.mean(valid_vectors, axis=0)

    # Find most similar tokens
    similar_tokens = model.wv.most_similar(positive=[word_vector], topn=topn)

    return word_vector, similar_tokens


def get_analogy_results(
    tokens: List[str],
    model: Word2Vec,
    topn: int = 10
) -> Optional[List[Tuple[str, float]]]:
    """
    Find most similar tokens to a combination of tokens.

    Args:
        tokens: List of tokens to combine
        model: Trained Word2Vec model
        topn: Number of similar tokens to return

    Returns:
        List of (token, similarity) tuples or None if tokens not in vocabulary
    """
    # Check if all tokens exist in model
    missing = [t for t in tokens if t not in model.wv]
    if missing:
        logger.warning(f"Tokens not in vocabulary: {missing}")
        return None

    similar = model.wv.most_similar(positive=tokens, topn=topn)
    return similar


# ============================================================================
# Verification and Testing
# ============================================================================

def verify_embeddings(
    tokenizer: Tokenizer,
    model: Word2Vec
) -> None:
    """
    Verify embeddings by testing word similarity queries.

    Args:
        tokenizer: BPE tokenizer instance
        model: Trained Word2Vec model
    """
    logger.info("=" * 70)
    logger.info("VERIFICATION: Word Similarity Queries")
    logger.info("=" * 70)

    words_to_test = ['wojsko', 'szlachta', 'choroba', 'król']

    for word in words_to_test:
        word_vector, similar_tokens = get_word_vector_and_similar(
            word, tokenizer, model, topn=10
        )

        if word_vector is not None:
            tokens = tokenizer.encode(word).tokens
            logger.info(
                f"\n10 most similar tokens to '{word}' "
                f"(tokens: {tokens}):"
            )
            logger.info(f"  > Word vector (first 5): {word_vector[:5]}...")
            for token, similarity in similar_tokens:
                logger.info(f"  - {token}: {similarity:.4f}")
        else:
            logger.warning(f"Could not compute vector for '{word}'")

    # Test token analogy
    logger.info("=" * 70)
    logger.info("VERIFICATION: Token Analogy")
    logger.info("=" * 70)

    tokens_analogy = ['dziecko', 'kobieta']
    similar = get_analogy_results(tokens_analogy, model, topn=10)

    if similar:
        logger.info(f"\n10 most similar tokens to: {tokens_analogy}")
        for token, similarity in similar:
            logger.info(f"  - {token}: {similarity:.4f}")
    else:
        logger.warning(f"Could not compute analogy for: {tokens_analogy}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main() -> None:
    """Execute the complete Word2Vec CBOW training pipeline."""
    logger.info("\n" + "=" * 70)
    logger.info("Word2Vec CBOW Embedding Training Pipeline")
    logger.info("=" * 70 + "\n")

    # Initialize configuration
    config = EmbeddingConfig()

    # Stage 1: Load tokenizer and prepare data
    tokenizer = load_tokenizer(config.tokenizer_file)
    tokenized_sentences = load_and_prepare_data(config, tokenizer)

    # Stage 2: Train model
    model = train_word2vec_model(tokenized_sentences, config)

    # Stage 3: Save artifacts
    saved_files = save_model_artifacts(model, config)

    # Verification
    verify_embeddings(tokenizer, model)

    logger.info("\n" + "=" * 70)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 70)
    logger.info(f"Output files:")
    for key, filepath in saved_files.items():
        logger.info(f"  - {key}: {filepath}")


if __name__ == "__main__":
    main()
