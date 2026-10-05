import json
import re
from openai import OpenAI
from price_parser import extract_numeric_price

client = OpenAI()


def _parse_price(meta: dict) -> float:
    raw = meta.get("price", 0)
    if isinstance(raw, (int, float)):
        return float(raw)
    cleaned = re.sub(r"[^\d.]", "", str(raw))
    return float(cleaned) if cleaned else 0.0


def _parse_llm_json(raw: str):
    """Strip markdown fences and parse JSON safely."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    if not cleaned or cleaned == "[]":
        return []
    if cleaned.startswith("{"):
        cleaned = "[" + cleaned + "]"
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[LLM FILTER] JSON parse error: {e}")
        print(f"[LLM FILTER] Raw: {cleaned[:300]}")
        return None


def filter_cars_by_requirements(
    retrieved_cars: list[dict],
    fuel_type: str,
    transmission: str,
    budget_str: str,
) -> list[dict]:

    budget_num = extract_numeric_price(budget_str)

    # ── Build compact payload for LLM ────────────────────────────────────────
    cars_data = []
    for idx, car in enumerate(retrieved_cars):
        meta = car["metadata"]
        cars_data.append({
            "id":           idx,
            "name":         meta.get("name") or meta.get("car_name") or "Unknown",
            "price":        _parse_price(meta),
            "fuel_type":    str(meta.get("fuel_type", "")).strip().lower(),
            "transmission": str(meta.get("transmission", "")).strip().lower(),
        })

    print(f"[LLM FILTER] Sending {len(cars_data)} cars to LLM "
          f"(fuel={fuel_type}, trans={transmission}, budget={budget_num}L)")

    # ── LLM does ALL filtering + ranking ─────────────────────────────────────
    prompt = f"""
You are a car recommendation assistant. Filter and rank the car list below.

USER REQUIREMENTS:
- Fuel type:     {fuel_type.strip().lower()}
- Transmission:  {transmission.strip().lower()}
- Budget:        {budget_num} Lakhs

RULES (apply in order):
1. FILTER OUT any car whose fuel_type does not exactly match "{fuel_type.strip().lower()}".
2. FILTER OUT any car whose transmission does not exactly match "{transmission.strip().lower()}".
3. From the remaining cars, RANK them:
   a. Cars within budget (price <= {budget_num}) → rank first, closest to budget first.
   b. Cars over budget → rank after, sorted by least over-budget first.
4. Include ALL cars that pass the filter — never drop any that match.

OUTPUT: ONLY a valid JSON array. No markdown, no code fences, no explanation.

FORMAT:
[
  {{
    "id":             <int, original id from input>,
    "matched":        <bool, true if passes fuel + transmission filter>,
    "price_diff":     <float, abs(price - {budget_num})>,
    "within_budget":  <bool, price <= {budget_num}>
  }}
]

CAR LIST:
{json.dumps(cars_data, indent=2)}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict JSON generator. "
                    "Output ONLY a valid JSON array. "
                    "No markdown, no code fences, no explanation."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content
    parsed = _parse_llm_json(content)

    # ── Fallback: Python filter + rank if LLM fails ───────────────────────────
    if not parsed:
        print("[LLM FILTER] LLM failed — falling back to Python filter + rank")
        ft = fuel_type.strip().lower()
        tr = transmission.strip().lower()
        fallback = []
        for car in retrieved_cars:
            meta = car["metadata"]
            if (str(meta.get("fuel_type", "")).strip().lower() == ft and
                    str(meta.get("transmission", "")).strip().lower() == tr):
                price = _parse_price(meta)
                diff = abs(price - budget_num)
                fallback.append({
                    **car,
                    "price_diff":      diff,
                    "price_proximity": 1 / (1 + diff),
                    "within_budget":   price <= budget_num,
                    "match_type":      "fallback",
                })
        fallback.sort(key=lambda x: (not x["within_budget"], x["price_diff"]))
        return fallback

    # ── Map LLM result back to full car objects ───────────────────────────────
    matched = []
    for item in parsed:
        if not item.get("matched", False):
            continue
        car_id = item.get("id")
        if car_id is None or car_id >= len(retrieved_cars):
            continue
        car   = retrieved_cars[car_id]
        price = _parse_price(car["metadata"])
        matched.append({
            **car,
            "price_diff":      item.get("price_diff", abs(price - budget_num)),
            "price_proximity": 1 / (1 + item.get("price_diff", abs(price - budget_num))),
            "within_budget":   item.get("within_budget", price <= budget_num),
            "match_type":      "llm",
        })

    within = [c for c in matched if c["within_budget"]]
    over   = [c for c in matched if not c["within_budget"]]
    within.sort(key=lambda x: x["price_diff"])
    over.sort(key=lambda x:   x["price_diff"])

    final = within + over
    print(f"[LLM FILTER] {len(within)} within budget | {len(over)} over budget")
    return final