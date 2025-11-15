# pip install sentence-transformers numpy scikit-learn

import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import os
import time
from corpora import CORPORA_FILES

# Configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- CONFIGURATION ---
MODEL_NAME = 'sdadas/mmlw-roberta-large'
OUTPUT_EMBEDDINGS_FILE = "sdadas_sentence_embeddings.npy"


def load_raw_sentences(file_list):
    """Loads raw sentences from a list of files."""
    raw_sentences = []
    print(f"Loading text from {len(file_list)} files...")
    for file in file_list:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                # Load lines, strip whitespace and skip empty lines
                lines = [line.strip() for line in f if line.strip()]
                raw_sentences.extend(lines)
        except FileNotFoundError:
            # Warning if file is not found
            print(f"WARNING: File '{file}' not found. Skipping.")
        except Exception as e:
            print(f"ERROR while processing file '{file}': {e}")

    if not raw_sentences:
        raise ValueError("Corpus is empty or was not loaded.")

    return raw_sentences


def load_or_generate_embeddings(files):
    """
    Loads or generates embeddings for the corpus.

    Returns:
        tuple: (raw_sentences, sentence_embeddings, model_sbert)
    """
    # Load raw sentences
    try:
        raw_sentences = load_raw_sentences(files)
        print(f"Loaded {len(raw_sentences)} sentences for processing.")
    except ValueError as e:
        print(f"ERROR: {e}")
        raise

    # Check if corpus vectors already exist on disk
    if os.path.exists(OUTPUT_EMBEDDINGS_FILE):
        # Variant 1: Load from file (.npy)
        print(f"\n--- Variant 1: Loading vectors from file '{OUTPUT_EMBEDDINGS_FILE}' ---")
        try:
            start_time = time.time()
            sentence_embeddings = np.load(OUTPUT_EMBEDDINGS_FILE)
            end_time = time.time()
            print(f"Vectors loaded successfully in {end_time - start_time:.2f} seconds. Encoding skipped.")
            needs_generation = False
        except Exception as e:
            # If loading fails (e.g., corrupted file), generate from scratch
            print(f"ERROR while loading .npy file: {e}. Processing corpus from scratch.")
            needs_generation = True
    else:
        needs_generation = True

    if needs_generation:
        # Variant 2: Load Model and Generate
        print(f"\n--- Variant 2: Loading Model and Generating Vectors ---")
        print(f"Loading Sentence-Transformer: {MODEL_NAME}...")
        try:
            # Load model from Hugging Face
            model_sbert = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            print(f"FATAL ERROR while loading model {MODEL_NAME}: {e}")
            raise

        print(f"Generating vectors for {len(raw_sentences)} sentences...")
        start_time = time.time()
        # The .encode() method automatically tokenizes and generates vectors
        sentence_embeddings = model_sbert.encode(
            raw_sentences,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        end_time = time.time()
        print(f"Generation completed in {end_time - start_time:.2f} seconds.")

        # Save newly created vectors to file
        np.save(OUTPUT_EMBEDDINGS_FILE, sentence_embeddings)
        print(f"Sentence vectors saved as: '{OUTPUT_EMBEDDINGS_FILE}'.")
    else:
        # Load model even if embeddings were loaded from cache
        print(f"\nLoading Sentence-Transformer: {MODEL_NAME}...")
        try:
            model_sbert = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            print(f"ERROR while loading model: {e}")
            raise

    print(f"\nEmbedding matrix shape: {sentence_embeddings.shape}")
    print(f"Sentence vector dimension: {sentence_embeddings.shape[1]}")

    return raw_sentences, sentence_embeddings, model_sbert


def initialize_model_and_data():
    """
    Initialize and load model with data.

    Returns:
        tuple: (raw_sentences, sentence_embeddings, model_sbert)
    """
    files = CORPORA_FILES["ALL"]
    raw_sentences, sentence_embeddings, model_sbert = load_or_generate_embeddings(files)
    return raw_sentences, sentence_embeddings, model_sbert