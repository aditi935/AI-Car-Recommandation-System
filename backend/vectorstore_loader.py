"""
vectorstore_loader.py
---------------------
Responsibility: Create, save, and load the FAISS vector index and its
                associated metadata/chunks from disk.

On first run it builds everything from the CSV.
On subsequent runs it loads the pre-built files so startup is instant.

This module is the "database layer" of the RAG pipeline. It knows about
file paths and FAISS operations, but nothing about business logic.

Used by: app.py (called once at startup to get index + metadata + chunks)
Depends on: chunk_builder.py, embedding_generator.py
"""

import os
import pickle
import numpy as np
import faiss
from openai import OpenAI

from chunk_builder import (
    build_chunks_from_csv,
    save_chunks,
    load_chunks,
)
from embedding_generator import generate_chunk_embeddings


def _create_and_save_faiss_index(
    embeddings: np.ndarray,
    metadata: list[dict],
    faiss_path: str,
    metadata_path: str,
) -> faiss.Index:
    """
    Build a flat L2 FAISS index from embeddings and persist both index
    and metadata to disk.

    Input:  embeddings    (np.ndarray)  — shape (N, D), float32
            metadata      (list[dict])  — one dict per car
            faiss_path    (str)         — where to save the FAISS index
            metadata_path (str)         — where to save the metadata pickle
    Output: faiss.Index — the in-memory index (also written to disk)
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))

    faiss.write_index(index, faiss_path)

    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"✅ FAISS index saved  → {faiss_path}  ({index.ntotal} vectors)")
    print(f"✅ Metadata saved     → {metadata_path} ({len(metadata)} records)")
    return index


def load_or_create_vectorstore(
    csv_path: str,
    chunks_path: str,
    faiss_path: str,
    metadata_path: str,
    embeddings_path: str,
    openai_client: OpenAI,
) -> tuple[faiss.Index, list[dict], list[dict]]:
    """
    Top-level loader: returns (faiss_index, metadata, chunks).

    • If all three artifact files already exist → load them and return.
    • Otherwise → read CSV, generate embeddings, build FAISS, save everything.

    Input:  csv_path        (str)    — raw data source
            chunks_path     (str)    — JSON file for chunk cache
            faiss_path      (str)    — .index file for FAISS
            metadata_path   (str)    — .pkl file for car metadata
            embeddings_path (str)    — .npy file for raw embeddings
            openai_client   (OpenAI) — authenticated client (only needed on first run)
    Output: tuple(faiss.Index, list[dict] metadata, list[dict] chunks)
    """

    # --- Cache hit: all files present → just load ---
    if os.path.exists(faiss_path) and os.path.exists(metadata_path):
        print("Loading existing vector store from disk...")

        index = faiss.read_index(faiss_path)

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        chunks = load_chunks(chunks_path) if os.path.exists(chunks_path) else None

        print(f"✅ FAISS index loaded  — {index.ntotal} vectors")
        print(f"✅ Metadata loaded     — {len(metadata)} records")
        return index, metadata, chunks

    # --- Cache miss: build from scratch ---
    print("No cached vector store found. Building from CSV...")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    chunks    = build_chunks_from_csv(csv_path)
    save_chunks(chunks, chunks_path)

    embeddings = generate_chunk_embeddings(chunks, openai_client)
    np.save(embeddings_path, embeddings)

    metadata = [chunk['metadata'] for chunk in chunks]
    index    = _create_and_save_faiss_index(embeddings, metadata, faiss_path, metadata_path)

    return index, metadata, chunks