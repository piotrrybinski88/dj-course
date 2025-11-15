# pip install sentence-transformers numpy scikit-learn

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sbert_model import initialize_model_and_data


def find_similar_sentences(query_sentence, sentence_embeddings, raw_sentences, model_sbert, top_k=5):
    """
    Find the most similar sentences in the corpus for a given query.

    Args:
        query_sentence (str): The query sentence
        sentence_embeddings (np.ndarray): Embeddings matrix of corpus sentences
        raw_sentences (list): List of raw sentences from corpus
        model_sbert (SentenceTransformer): The SBERT model instance
        top_k (int): Number of top similar sentences to return

    Returns:
        list: List of tuples (similarity_score, sentence_text, sentence_index)
    """
    print(f"\n--- Searching for similarity to: '{query_sentence}' ---")

    # Generate vector for the query
    query_embedding = model_sbert.encode(
        [query_sentence],
        convert_to_numpy=True
    )

    # Calculate cosine similarity between query and all sentences
    # Cosine similarity is a standard measure of vector similarity
    similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]

    # Find top k most similar
    top_k_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    print(f"\n{top_k} most similar sentences from the corpus:")
    for idx, i in enumerate(top_k_indices):
        similarity_score = similarities[i]
        sentence_text = raw_sentences[i]
        results.append((similarity_score, sentence_text, i))
        print(f"  {idx + 1}. Sim: {similarity_score:.4f} | Sentence: {sentence_text}")

    return results


def test_queries(raw_sentences, sentence_embeddings, model_sbert):
    """
    Test multiple queries to demonstrate model usage.

    Args:
        raw_sentences (list): List of raw sentences from corpus
        sentence_embeddings (np.ndarray): Embeddings matrix of corpus sentences
        model_sbert (SentenceTransformer): The SBERT model instance
    """
    test_queries_list = [
        # "Jestem głodny.",
        "Wojsko wejdzie do miast i skończą się bunty",
        "Leczenie tego schorzenia jest bardzo ważne i wymaga interwencji lekarza.",
        "Litwo ojczyzno moja!"
    ]

    for query_sentence in test_queries_list:
        find_similar_sentences(
            query_sentence,
            sentence_embeddings,
            raw_sentences,
            model_sbert,
            top_k=5
        )
        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Initialize model and data
    print("Initializing model and loading data...")
    raw_sentences, sentence_embeddings, model_sbert = initialize_model_and_data()

    # Run test queries
    test_queries(raw_sentences, sentence_embeddings, model_sbert)