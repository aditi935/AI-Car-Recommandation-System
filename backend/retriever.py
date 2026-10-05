import numpy as np
import faiss
from openai import OpenAI

from price_parser import format_price_for_display

client = OpenAI()


def compute_dynamic_k(attrs: dict, index_size: int, max_k: int = 20) -> int:
    """K=20 so strict fuel+transmission filter has enough pool to work with."""

    prompt = f"""
You are a retrieval optimization assistant.

Task: Decide how many results (K) to retrieve from a car database.
Input attributes: {attrs}

Rules:
- All 3 known (fuel + transmission + budget) → return 15
- 2 known → return 18
- 1 or 0 known → return {max_k}
- Output ONLY an integer. Max: {max_k}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        k = int(response.choices[0].message.content.strip())
    except Exception:
        k = max_k

    k = min(max(1, k), index_size, max_k)
    print(f"[RETRIEVER] Dynamic K = {k} (LLM-driven)")
    return k


def retrieve_top_k(
    query_vector: np.ndarray,
    k: int,
    faiss_index: faiss.Index,
    car_metadata: list[dict],
    car_chunks: list[dict] | None,
) -> list[dict]:

    distances, indices = faiss_index.search(query_vector, k)

    retrieved = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1 or idx >= len(car_metadata):
            continue

        meta = car_metadata[idx].copy()
        meta.setdefault('price_display', format_price_for_display(meta.get('price', 0)))
        meta.setdefault('price',        0)
        meta.setdefault('fuel_type',    'unknown')
        meta.setdefault('transmission', 'unknown')
        meta.setdefault('name',         meta.get('car_name', 'Unknown'))
        meta.setdefault('brand',        'Unknown')
        meta.setdefault('usage',        'daily use')
        meta.setdefault('seats',        '5 Seater')
        meta.setdefault('mileage',      'Good')

        chunk_text = car_chunks[idx]['text'] if car_chunks else str(meta)

        retrieved.append({
            "metadata":   meta,
            "chunk":      chunk_text,
            "distance":   float(dist),
            "similarity": 1 / (1 + float(dist)),
        })

    print("-" * 80)
    print(f"[STEP 4 - FAISS RETRIEVAL] Retrieved {len(retrieved)} cars (K={k})\n")
    for i, car in enumerate(retrieved):
        m = car['metadata']
        print(
            f"  CAR {i+1}: {m['name']} ({m['brand']}) | "
            f"{m['fuel_type']}/{m['transmission']} | "
            f"Price: {m['price_display']} | Score: {car['similarity']:.3f}"
        )
    print("\n" + "-" * 80)
    return retrieved