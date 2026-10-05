"""
chunk_builder.py
----------------
Responsibility: Transform raw CSV car data into structured text chunks
                for embedding and vector search.

This version focuses purely on chunk creation (no price parsing logic).

Used by: vectorstore_loader.py
Depends on: None (pure data transformation)
"""

import json
import pandas as pd


def build_chunks_from_csv(csv_path: str) -> list[dict]:
    """
    Function: build_chunks_from_csv

    Description:
        Reads a CSV file and converts each row into a structured chunk.
        Each chunk contains:
          - 'text': human-readable description (used for embeddings)
          - 'metadata': structured fields (used for filtering & retrieval)

        This function focuses only on chunk generation and avoids any
        complex parsing logic to keep it clean and maintainable.

    Usage:
        Called during data preprocessing before creating embeddings.
        Output is passed to vector store (FAISS, etc.).

    LLM Interaction:
        None — this is a pure data preparation step.

    Args:
        csv_path (str): path to dataset CSV file

    Returns:
        list[dict]: list of chunk objects
    """

    df = pd.read_csv(csv_path)
    chunks = []

    for idx, row in df.iterrows():

        # -------- Handle optional fields safely --------
        usage    = str(row.get('usage', '')) if not pd.isna(row.get('usage', '')) else "daily use"
        features = str(row.get('additional_features', '')) if not pd.isna(row.get('additional_features', '')) else "Standard features"
        mileage  = str(row.get('mileage', '')) if not pd.isna(row.get('mileage', '')) else "Good mileage"
        call     = str(row.get('call', '')) if not pd.isna(row.get('call', '')) else ""
        link     = str(row.get('link', '')) if not pd.isna(row.get('link', '')) else ""

        # -------- Build text chunk --------
        text = f"""Car Name: {row.get('name', '')}
Brand: {row.get('brand', '')}
Body Type: {row.get('body_type', '')}
Fuel Type: {row.get('fuel_type', '')}
Transmission: {row.get('transmission', '')}
Seating Capacity: {row.get('seating_capacity', '')}
Mileage: {mileage}
Airbags: {row.get('airbags', '')}
Features: {features}
Usage: {usage}
Price: {row.get('price_lakhs', '')}
Phone: {call}
Link: {link}"""

        # -------- Metadata --------
        metadata = {
            "name":          str(row.get('name', '')),
            "brand":         str(row.get('brand', '')),
            "body_type":     str(row.get('body_type', '')),
            "fuel_type":     str(row.get('fuel_type', '')).lower().strip(),
            "transmission":  str(row.get('transmission', '')).lower().strip(),
            "price":         str(row.get('price_lakhs', '')),  # kept raw
            "seats":         str(row.get('seating_capacity', '')),
            "mileage":       mileage,
            "airbags":       str(row.get('airbags', '')),
            "features":      features,
            "usage":         usage,
            "call":          call,
            "link":          link,
        }

        chunks.append({
            "id": idx,
            "text": text,
            "metadata": metadata
        })

    print(f"Built {len(chunks)} car chunks from CSV")
    return chunks


def save_chunks(chunks: list[dict], chunks_path: str) -> None:
    """
    Function: save_chunks

    Description:
        Saves generated chunks into a JSON file for reuse.

    Usage:
        Called after chunk generation to persist data and avoid rebuilding.

    LLM Interaction:
        None — simple file operation.

    Args:
        chunks (list[dict]): chunk data
        chunks_path (str): file path to save JSON

    Returns:
        None
    """

    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Chunks saved to {chunks_path}")


def load_chunks(chunks_path: str) -> list[dict]:
    """
    Function: load_chunks

    Description:
        Loads previously saved chunks from a JSON file.

    Usage:
        Used during application startup to load preprocessed data.

    LLM Interaction:
        None — simple file read.

    Args:
        chunks_path (str): path to JSON file

    Returns:
        list[dict]: chunk data
    """

    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    return chunks