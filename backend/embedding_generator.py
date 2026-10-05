"""
embedding_generator.py
----------------------
Responsibility: Generate vector embeddings for car text chunks and
                for user query strings using the OpenAI Embeddings API.

This module is the only place in the project that calls
client.embeddings.create(). Keeping it isolated makes it easy to swap
the embedding model without touching anything else.

Used by: vectorstore_loader.py (batch embeddings), retriever.py (query embedding)
Depends on: openai (external)
"""

import numpy as np
from openai import OpenAI


# OpenAI model used for all embeddings — change here to update everywhere
EMBEDDING_MODEL = "text-embedding-3-small"


def generate_chunk_embeddings(chunks: list[dict], client: OpenAI) -> np.ndarray:
    """
    Generate embeddings for a list of car text chunks in batches.

    Input:  chunks (list[dict]) — output of chunk_builder.build_chunks_from_csv()
            client (OpenAI)     — authenticated OpenAI client
    Output: np.ndarray shape (N, D) float32 — one row per chunk

    Processes in batches of 20 to stay within API rate limits.
    Saves the result to disk via the caller (vectorstore_loader).
    """
    texts = [chunk['text'] for chunk in chunks]
    embeddings = []
    batch_size = 20

    print(f"Generating embeddings for {len(texts)} chunks using {EMBEDDING_MODEL}...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) - 1) // batch_size + 1
        print(f"  Processing batch {batch_num}/{total_batches}")

        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for item in response.data:
            embeddings.append(item.embedding)

    array = np.array(embeddings, dtype='float32')
    print(f"✅ Embeddings shape: {array.shape}")
    return array


def embed_query(query_text: str, client: OpenAI) -> np.ndarray:
    """
    Generate a single embedding vector for a user query string.

    Input:  query_text (str) — the search query (e.g. "petrol automatic under 15L")
            client (OpenAI)  — authenticated OpenAI client
    Output: np.ndarray shape (1, D) float32 — ready for FAISS search

    The vector is reshaped to (1, D) so it can be passed directly to
    faiss_index.search() without extra reshaping at the call site.
    """
    print(f"[EMBED] Embedding query: '{query_text}'")

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=query_text)
    vector = np.array(response.data[0].embedding, dtype='float32')

    print(f"[EMBED] ✓ Vector dimension: {vector.shape[0]}")
    return vector.reshape(1, -1)