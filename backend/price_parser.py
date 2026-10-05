"""
price_parser.py
---------------
Responsibility: Parse and format car price strings using LLM prompting.

This module replaces traditional regex-based parsing with LLM-driven
interpretation for flexible and scalable price understanding.

Used by: chunk_builder.py, car_filter.py, response_formatter.py
"""

from openai import OpenAI
import pandas as pd

client = OpenAI()


def extract_numeric_price(price_str) -> float:
    """
    Function: extract_numeric_price  
    Description: Uses LLM prompting to extract and convert a price string 
                 (e.g., '10L', '1.5Cr', '15 lakhs') into a numeric value in lakhs.  
    Usage: Called when raw price data needs to be normalized for filtering, 
           comparison, or storage.  

    Input:  price_str (str or NaN)  
    Output: float (price in lakhs, 0.0 if invalid)
    """

    if not price_str or price_str == 'nan':
        return 0.0

    try:
        if pd.isna(price_str):
            return 0.0
    except Exception:
        pass

    # -------- Prompt-based price extraction --------
    prompt = f"""
    You are a price normalization assistant.

    Task:
    Convert the given car price into a numeric value in LAKHS.

    Rules:
    - "1 Cr" = 100 lakhs
    - "1.5 Cr" = 150 lakhs
    - "10L" or "10 lakhs" = 10
    - If invalid or unclear, return 0

    Output ONLY a number. No text.

    Input: {price_str}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        value = response.choices[0].message.content.strip()
        return float(value)

    except Exception:
        return 0.0


def format_price_for_display(price: float) -> str:
    """
    Function: format_price_for_display  
    Description: Uses LLM prompting to convert numeric price (in lakhs) 
                 into a human-readable format (e.g., "16.0L", "1.5Cr").  
    Usage: Called when displaying price to users in UI or chatbot responses.  

    Input:  price (float)  
    Output: str (formatted price string)
    """

    if price == 0:
        return "Price not available"

    # -------- Prompt-based formatting --------
    prompt = f"""
    You are a price formatting assistant.

    Task:
    Convert a numeric price (in lakhs) into a readable string.

    Rules:
    - If >= 100 → convert to Cr (e.g., 150 → 1.5Cr)
    - If < 100 → keep in L (e.g., 16 → 16.0L)
    - Keep 1 decimal place

    Output ONLY the formatted value.

    Input: {price}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return "Price not available"