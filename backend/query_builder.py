"""
query_builder.py
----------------
Responsibility: 
  1. Build a natural-language search query string for FAISS (used as fallback).
  2. Provide hard metadata filtering directly on car_metadata list —
     this is the PRIMARY retrieval path for exact fuel+transmission matches.
"""

import re
from price_parser import extract_numeric_price


def build_retrieval_query(attrs: dict) -> str:
    """
    Build FAISS query string from attributes.
    Used only as fallback when metadata filter returns 0 results.
    """
    fuel         = attrs.get("fuel_type",    "")
    transmission = attrs.get("transmission", "")
    budget       = attrs.get("budget",       "")
    usage        = attrs.get("usage",        "")

    query = f"{fuel} {transmission} car".strip()

    if budget:
        budget_num = extract_numeric_price(budget)
        if budget_num > 0:
            query += f" under {budget_num:.1f} lakhs"

    if usage:
        query += f" suitable for {usage}"

    print(f"[QUERY BUILDER] Built query: '{query}'")
    return query.strip()


def extract_price_from_meta(meta: dict) -> float:
    """
    Safely extract numeric price from metadata.
    Handles formats: 22, "22", "22L", "22 lakh", "22.0L"
    """
    raw = meta.get("price", 0)
    if isinstance(raw, (int, float)):
        return float(raw)
    cleaned = re.sub(r"[^\d.]", "", str(raw))
    return float(cleaned) if cleaned else 0.0


# def filter_by_metadata(
#     car_metadata: list[dict],
#     fuel_type: str,
#     transmission: str,
#     budget: str,
# ) -> list[dict]:
#     """
#     PRIMARY retrieval function — filters car_metadata directly.
#     No FAISS involved. Guarantees exact fuel+transmission matches.

#     Returns cars sorted by:
#       1. Exact budget match first (price <= budget, sorted by proximity)
#       2. Over-budget cars after (sorted by how close they are to budget)
#       3. If still empty, relaxes transmission, then fuel

#     Each returned item is wrapped as:
#       {
#         "metadata":        <meta dict>,
#         "chunk":           <str>,        ← empty string (no chunk needed)
#         "distance":        0.0,
#         "similarity":      1.0,
#         "price_diff":      <float>,
#         "price_proximity": <float>,
#         "match_type":      "strict" | "partial_transmission" | "partial_fuel"
#       }
#     """
#     ft = fuel_type.strip().lower()
#     tr = transmission.strip().lower()
#     budget_num = extract_numeric_price(budget)

#     def normalize(val):
#         return str(val).strip().lower()

#     def score(meta, match_type):
#         price = extract_price_from_meta(meta)
#         price_diff = abs(price - budget_num)
#         price_proximity = 1 / (1 + price_diff)
#         return {
#             "metadata":        meta,
#             "chunk":           "",
#             "distance":        0.0,
#             "similarity":      1.0,
#             "price_diff":      price_diff,
#             "price_proximity": price_proximity,
#             "match_type":      match_type,
#             "_price":          price,
#         }

#     # ── Pass 1: exact fuel + transmission, within budget ──────────────────
#     pass1 = [
#         score(m, "strict")
#         for m in car_metadata
#         if normalize(m.get("fuel_type", "")) == ft
#         and normalize(m.get("transmission", "")) == tr
#         and extract_price_from_meta(m) <= budget_num
#     ]
#     pass1.sort(key=lambda x: x["price_diff"])  # closest to budget first

#     # ── Pass 2: exact fuel + transmission, over budget ─────────────────────
#     pass2 = [
#         score(m, "strict")
#         for m in car_metadata
#         if normalize(m.get("fuel_type", "")) == ft
#         and normalize(m.get("transmission", "")) == tr
#         and extract_price_from_meta(m) > budget_num
#     ]
#     pass2.sort(key=lambda x: x["price_diff"])  # least over-budget first

#     strict_results = pass1 + pass2

#     if strict_results:
#         print(f"[META FILTER] ✓ {len(pass1)} within budget + {len(pass2)} over budget "
#               f"(fuel={ft}, trans={tr})")
#         return strict_results

#     # ── Pass 3: fuel only (relax transmission) ─────────────────────────────
#     pass3 = [
#         score(m, "partial_transmission")
#         for m in car_metadata
#         if normalize(m.get("fuel_type", "")) == ft
#     ]
#     pass3.sort(key=lambda x: x["price_diff"])

#     if pass3:
#         print(f"[META FILTER] ⚠ No {ft}/{tr} found — relaxing transmission. "
#               f"Found {len(pass3)} {ft} cars.")
#         return pass3

#     # ── Pass 4: transmission only (relax fuel) ─────────────────────────────
#     pass4 = [
#         score(m, "partial_fuel")
#         for m in car_metadata
#         if normalize(m.get("transmission", "")) == tr
#     ]
#     pass4.sort(key=lambda x: x["price_diff"])

#     if pass4:
#         print(f"[META FILTER] ⚠ No {ft} cars found — relaxing fuel. "
#               f"Found {len(pass4)} {tr} cars.")
#         return pass4

#     print(f"[META FILTER] ✗ No cars found even after relaxing all filters.")
#     return []