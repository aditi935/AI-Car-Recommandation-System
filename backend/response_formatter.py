"""
response_formatter.py
---------------------
Responsibility: Format filtered car recommendations using LLM prompting.

This version replaces hardcoded formatting logic with LLM-driven response
generation while preserving the same functionality and output structure.

Used by: app.py
"""

from openai import OpenAI
from price_parser import extract_numeric_price
import json

client = OpenAI()


def build_reply_text(
    recommendations: list[dict],
    fuel: str,
    transmission: str,
    budget: str,
) -> str:

    from price_parser import extract_numeric_price
    import json

    budget_num = extract_numeric_price(budget)

    # ✅ STEP 1: Prepare clean structured data
    formatted_data = []

    for car in recommendations:
        m = car["metadata"]
        
        formatted_data.append({
            "name": m.get("name"),
            "brand": m.get("brand"),
            "price": m.get("price"),
            "fuel": m.get("fuel_type"),
            "transmission": m.get("transmission"),
            "seats": m.get("seats"),
            "mileage": m.get("mileage"),
            "airbags": m.get("airbags"),
            "features": m.get("features"),
            "usage": m.get("usage"),
            "call": m.get("call"),
            "link": m.get("link")
        })

    # ✅ STEP 2: LLM Prompt (strict formatting)
    prompt = f"""
            You are a car recommendation assistant.

            Generate a structured response.

            STRICT RULES:
            - DO NOT write in paragraph
            - Each car MUST be multi-line
            - DO NOT compress into one line
            - DO NOT skip fields
            - DO NOT invent data
                Instructions:
                

                1. If the recommendations perfectly match the user's preferences:
                - Start with: "Here are the best cars based on your preferences 👇"

                2. If the recommendations are only partially matching (not exact):
                - Start with:
                    "We couldn’t find an exact match for your preferences.
                    However, here are some closely related options you might like 👇"

                3. Always sound natural and helpful.
                4. Show cars in clean bullet or numbered format.
                5. For each car, include:
                    - Name (Brand)
                    - Price
                    - Budget Fit (Within / Slightly Above)
                    - Fuel
                    - Transmission
                    - Seats
                    - Mileage
                    - Airbags
                    - Features (max 5)
                    - Best For (usage)
                    - Contact
                    - Link
            FORMAT:

            1. Car Name (Brand)
            Price: ₹X
            Budget Fit: Within / Slightly Above
            Fuel: <fuel>
            Transmission: <type>
            Seats: <number>
            Mileage: <value>
            Airbags: <number>
            Features: <max 5 features>
            Best For: <usage>
            Contact: <call>
            Link: <link>

            User Preferences:
            Fuel: {fuel}
            Transmission: {transmission}
            Budget: {budget}

            Budget Numeric: {budget_num}

            Car Data:
            {json.dumps(formatted_data, indent=2)}

            
            """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        reply = response.choices[0].message.content.strip()

        # ✅ SAFETY CHECK (if LLM still messes up)
        if "\n" not in reply or "Price:" not in reply:
            raise ValueError("Bad formatting from LLM")

        return reply

    except Exception as e:
        print("⚠️ LLM formatting failed, using fallback:", e)

        # ✅ STEP 3: FALLBACK (100% guaranteed correct format)
        lines = [f"Here are the best {fuel} {transmission} cars under {budget}:\n"]

        for i, car in enumerate(recommendations, 1):
            m = car["metadata"]

            features = m.get("features", "")
            if isinstance(features, list):
                features = ", ".join(features[:5])

            lines.append(
                f"{i}. {m.get('name')} ({m.get('brand')})\n"
                f"   Price: ₹{m.get('price')}\n"
                f"   Budget Fit: Within\n"
                f"   Fuel: {m.get('fuel_type')}\n"
                f"   Transmission: {m.get('transmission')}\n"
                f"   Seats: {m.get('seats')}\n"
                f"   Mileage: {m.get('mileage')}\n"
                f"   Airbags: {m.get('airbags')}\n"
                f"   Features: {features}\n"
                f"   Best For: {m.get('usage')}\n"
                f"   Contact: {m.get('call')}\n"
                f"   Link: {m.get('link')}\n"
            )

        lines.append("Would you like automatic or SUV options as well?")

        return "\n".join(lines)


def build_recommendations_payload(
    recommendations: list[dict],
    budget: str,
) -> list[dict]:
    """
    Function: build_recommendations_payload

    Description:
        Uses LLM prompting to transform recommendation data into a structured
        JSON payload suitable for frontend consumption.

        The LLM:
          - Extracts required fields
          - Determines if car is within budget
          - Ensures consistent JSON formatting

    Usage:
        Called before sending API JSON response.

    LLM Interaction:
        YES — structured JSON generated via prompt.

    Args:
        recommendations (list[dict])
        budget (str)

    Returns:
        list[dict]
    """

    budget_num = extract_numeric_price(budget)

    prompt = f"""
    You are a data formatting assistant.

    Task:
    Convert the given car recommendation data into structured JSON.

    Budget (lakhs): {budget_num}

    Rules:
    - For each car include:
        car_name, brand, price, fuel, transmission,
        seats, mileage, usage, call, link
    - Add "is_within_budget": true/false
    - If price <= budget → true else false
    - Output MUST be a valid JSON list

    Input:
    {recommendations}

    Output ONLY JSON.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return json.loads(response.choices[0].message.content.strip())

    except Exception:
        # -------- Minimal safe fallback --------
        payload = []
        for car in recommendations:
            meta = car['metadata']
            payload.append({
                "car_name": meta.get('name', ''),
                "brand": meta.get('brand', ''),
                "price": meta.get('price', ''),
                "fuel": meta.get('fuel_type', ''),
                "transmission": meta.get('transmission', ''),
                "seats": meta.get('seats', ''),
                "mileage": meta.get('mileage', ''),
                "usage": meta.get('usage', ''),
                "call": meta.get('call', ''),
                "link": meta.get('link', ''),
                "is_within_budget": True
            })
        return payload