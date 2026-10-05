"""
app.py — orchestrator

Retrieval flow:
  FAISS (K=20) → strict fuel+transmission filter → LLM ranks by budget → top 4
"""

import os
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

from vectorstore_loader  import load_or_create_vectorstore
from query_builder       import build_retrieval_query
from embedding_generator import embed_query
from retriever           import compute_dynamic_k, retrieve_top_k
from car_filter          import filter_cars_by_requirements
from response_formatter  import build_reply_text, build_recommendations_payload
from llm_handler         import extract_attributes_via_llm
from price_parser        import extract_numeric_price

load_dotenv()

app = Flask(__name__)
CORS(app)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
VECTOR_DIR  = os.path.join(BASE_DIR, "vector_store")
CHUNKS_DIR  = os.path.join(BASE_DIR, "chunks")

os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

CSV_PATH        = os.path.join(BASE_DIR, "car_dataset_30.csv")
CHUNKS_PATH     = os.path.join(CHUNKS_DIR, "car_chunks.json")
FAISS_PATH      = os.path.join(VECTOR_DIR, "cars.index")
METADATA_PATH   = os.path.join(VECTOR_DIR, "cars_metadata.pkl")
EMBEDDINGS_PATH = os.path.join(VECTOR_DIR, "embeddings.npy")

_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


print("\n" + "=" * 80)
print("INITIALIZING CAR RECOMMENDATION SYSTEM")
print("=" * 80)

try:
    faiss_index, car_metadata, car_chunks = load_or_create_vectorstore(
        csv_path        = CSV_PATH,
        chunks_path     = CHUNKS_PATH,
        faiss_path      = FAISS_PATH,
        metadata_path   = METADATA_PATH,
        embeddings_path = EMBEDDINGS_PATH,
        openai_client   = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "")),
    )
    print("[OK] System initialized successfully!")
except Exception as e:
    print(f"[ERROR] Error: {e}")
    exit(1)

print("=" * 80 + "\n")

chat_history       = {}
session_attributes = {}


@app.route("/")
def index():
    return jsonify({"status": "ok", "cars_available": faiss_index.ntotal})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "faiss_vectors": faiss_index.ntotal,
        "cars_loaded": len(car_metadata),
    })


@app.route("/chat", methods=["POST"])
def chat():
    data         = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    session_id   = data.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "message required"}), 400

    client = get_openai_client()
    if not client:
        return jsonify({"error": "OPENAI_API_KEY not set"}), 503

    if session_id not in chat_history:
        chat_history[session_id] = []
    if session_id not in session_attributes:
        session_attributes[session_id] = {
            "budget": "", "fuel_type": "", "transmission": "", "usage": ""
        }

    history       = chat_history[session_id]
    current_attrs = session_attributes[session_id]

    print()
    print("-" * 80)
    print(f"[REQUEST] Session: {session_id} | User: {user_message}")
    print("-" * 80)

    history.append({"role": "user", "content": user_message})

    try:
        # ── STEP 1: Extract + merge attributes ───────────────────────────────
        parsed = extract_attributes_via_llm(
            user_message           = user_message,
            conversation_history   = history,
            client                 = client,
            accumulated_attributes = current_attrs,
        )

        attrs = parsed.get("attributes", current_attrs)
        session_attributes[session_id] = attrs
        ready = parsed.get("ready_for_retrieval", False)

        if not ready:
            reply = parsed.get("message", "Could you share more details?")
            history.append({"role": "assistant", "content": reply})
            return jsonify({
                "reply": reply,
                "show_fuel_buttons":         parsed.get("show_fuel_buttons",         False),
                "show_transmission_buttons": parsed.get("show_transmission_buttons", False),
                "show_budget_buttons":       parsed.get("show_budget_buttons",       False),
            })

        # ── STEP 2: Clean values ──────────────────────────────────────────────
        fuel         = attrs.get("fuel_type",    "").strip()
        transmission = attrs.get("transmission", "").strip()
        budget       = attrs.get("budget",       "").strip()
        budget_num   = extract_numeric_price(budget)

        print(f"[PIPELINE] fuel={fuel} | trans={transmission} | budget={budget}")

        # ── STEP 3: FAISS retrieval (K=20) ───────────────────────────────────
        query        = build_retrieval_query(attrs)
        query_vector = embed_query(query, client)
        k            = compute_dynamic_k(attrs, index_size=faiss_index.ntotal)
        retrieved_cars = retrieve_top_k(
            query_vector, k, faiss_index, car_metadata, car_chunks
        )

        if not retrieved_cars:
            reply = f"Sorry, I couldn't retrieve any cars. Please try again."
            history.append({"role": "assistant", "content": reply})
            return jsonify({"reply": reply})

        # ── STEP 4: Strict filter + LLM budget ranking ────────────────────────
        filtered_cars = filter_cars_by_requirements(
            retrieved_cars, fuel, transmission, budget
        )

        if not filtered_cars:
            # Strict filter found 0 petrol/automatic in FAISS top-K
            # This means the dataset may have very few such cars
            reply = (
                f"I couldn't find any {fuel} cars with {transmission} transmission "
                f"in our current listings. Would you like to try a different fuel type "
                f"or transmission?"
            )
            history.append({"role": "assistant", "content": reply})
            return jsonify({"reply": reply})

        # ── STEP 5: Top 4 — within budget first, over budget after ───────────
        recommendations = filtered_cars[:4]

        within_budget = [
            c for c in recommendations
            if _parse_price(c["metadata"]) <= budget_num
        ]
        over_budget = [
            c for c in recommendations
            if _parse_price(c["metadata"]) > budget_num
        ]

        print("-" * 80)
        print(f"[RESULTS] {len(within_budget)} within budget | {len(over_budget)} over budget\n")
        for i, car in enumerate(recommendations):
            m = car["metadata"]
            print(
                f"  REC {i+1}: {m.get('name', m.get('car_name','?'))} ({m.get('brand','?')}) | "
                f"{m.get('fuel_type','?')}/{m.get('transmission','?')} | "
                f"Price: {m.get('price_display', m.get('price','?'))} | "
                f"diff={car.get('price_diff', 0):.1f}L"
            )
        print("-" * 80)

        # ── STEP 6: Format reply ──────────────────────────────────────────────
        reply   = build_reply_text(recommendations, fuel, transmission, budget)
        payload = build_recommendations_payload(recommendations, budget)

        # Honest budget note
        if within_budget and over_budget:
            reply += (
                f"\n\n💡 Note: Some suggestions are slightly above your {budget} budget "
                f"but are the closest {fuel} {transmission} cars available."
            )
        elif not within_budget and over_budget:
            reply = (
                f"I couldn't find any {fuel} {transmission} cars within your {budget} budget, "
                f"but here are the closest available options:\n\n" + reply
            )

        history.append({"role": "assistant", "content": reply})

        return jsonify({
            "reply":                    reply,
            "recommendations":          payload,
            "cars_analyzed":            len(retrieved_cars),
            "matches_found":            len(recommendations),
            "show_fuel_buttons":         False,
            "show_transmission_buttons": False,
            "show_budget_buttons":       False,
        })

    except Exception as e:
        if history and history[-1]["role"] == "user":
            history.pop()
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _parse_price(meta: dict) -> float:
    import re
    raw = meta.get("price", 0)
    if isinstance(raw, (int, float)):
        return float(raw)
    cleaned = re.sub(r"[^\d.]", "", str(raw))
    return float(cleaned) if cleaned else 0.0


if __name__ == "__main__":
    print(f"\nCars: {faiss_index.ntotal} | http://0.0.0.0:5001\n")
    app.run(host="0.0.0.0", debug=True, port=5001)