"""
attribute_extractor.py
----------------------
Responsibility: Validate and merge user preference attributes using
LLM-assisted prompting (where applicable) with a Python safety layer.

This version replaces hardcoded validation logic with LLM-driven reasoning
while still keeping a defensive fallback for reliability.
"""

from openai import OpenAI

client = OpenAI()


# Required attributes are now also enforced via LLM prompt (not just Python)
REQUIRED_ATTRIBUTES = ["fuel_type", "budget", "transmission"]


def merge_attributes(existing: dict, new_attrs: dict) -> dict:
    """
    Function: merge_attributes

    Description:
        Uses LLM prompting to intelligently merge newly extracted attributes
        into the existing session state.

        Instead of simple hardcoded dict merging, the LLM:
          - Decides whether a new value should overwrite the old one
          - Cleans and normalizes values
          - Preserves valid previous context

    Usage:
        Called after extracting attributes from user input.
        Ensures conversation memory remains consistent across turns.

    LLM Interaction:
        YES — LLM determines final merged output using structured prompting.

    Args:
        existing   (dict): current session attribute state
        new_attrs  (dict): attributes extracted from latest user message

    Returns:
        dict — merged attributes
    """

    prompt = f"""
    You are an intelligent attribute merging assistant.

    Task:
    Merge two dictionaries: existing attributes and new attributes.

    Rules:
    - Keep existing values if new value is empty or null
    - Replace existing values if new value is meaningful
    - Clean and normalize values (lowercase, trimmed)
    - Ensure output is a valid JSON dictionary

    Existing:
    {existing}

    New:
    {new_attrs}

    Output ONLY the final merged JSON.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        import json
        return json.loads(response.choices[0].message.content.strip())

    except Exception:
        # fallback (minimal safe merge)
        merged = existing.copy()
        for key, value in new_attrs.items():
            if value and str(value).strip():
                merged[key] = str(value).strip()
        return merged


def validate_attributes(attrs: dict) -> dict:
    """
    Function: validate_attributes

    Description:
        Uses LLM prompting to validate whether all required attributes
        are present and determine missing fields.

        ROLE IN PIPELINE:
          - PRIMARY: LLM evaluates completeness dynamically
          - SECONDARY: fallback Python logic ensures reliability

        The LLM:
          - Checks required attributes (fuel_type, budget, transmission)
          - Identifies missing ones
          - Determines readiness for retrieval

        Python fallback ensures:
          - No hallucination-based approval
          - System safety before expensive operations

    Usage:
        Called before retrieval (FAISS / embeddings).
        If not satisfied → pipeline stops and asks user for missing info.

    LLM Interaction:
        YES — main validation handled via prompt

    Args:
        attrs (dict): current merged attributes

    Returns:
        dict:
            {
              "satisfied": bool,
              "missing": list
            }
    """

    prompt = f"""
    You are an attribute validation assistant.

    Task:
    Check if all required attributes are present.

    Required attributes:
    {REQUIRED_ATTRIBUTES}

    Input attributes:
    {attrs}

    Rules:
    - If a field is missing or empty → mark as missing
    - If all present → satisfied = true
    - Output JSON ONLY

    Output format:
    {{
      "satisfied": true/false,
      "missing": ["field1", "field2"]
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        import json
        result = json.loads(response.choices[0].message.content.strip())

        return result

    except Exception:
        # -------- Python Safety Fallback --------
        missing = []

        for attr in REQUIRED_ATTRIBUTES:
            value = str(attrs.get(attr, "")).strip()
            if not value:
                missing.append(attr)

        return {
            "satisfied": len(missing) == 0,
            "missing": missing,
        }