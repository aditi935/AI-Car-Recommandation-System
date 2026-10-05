"""
llm_handler.py
--------------
Responsibility: Send messages to the OpenAI Chat Completions API and
                return the parsed response.

KEY FIX 1: Accumulates attributes across conversation turns so previously
           collected values (fuel_type, transmission) are never lost when
           the user provides budget in a later message.

KEY FIX 2: Button visibility is driven by current_question from the LLM
           (which attribute the message is actively asking about), NOT just
           by which attributes happen to be empty. This prevents fuel buttons
           from showing during a budget-redirect message (e.g. joke budgets).

KEY FIX 3: Python-side budget recovery — if the LLM drops a valid budget
           (e.g. misclassifies "28 lakh" as a joke), we re-extract it
           directly from the raw user message as a safety net.
"""

import json
import re
from openai import OpenAI

from prompt_builder import build_attribute_extraction_prompt

CHAT_MODEL = "gpt-4o-mini"

# Canonical allowed values — used for validation after LLM response
ALLOWED_FUEL  = {"petrol", "diesel", "cng", "electric", "hybrid"}
ALLOWED_TRANS = {"manual", "automatic"}
ALLOWED_USAGE = {"city", "highway", "off-road", "family"}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_budget(raw: str) -> str:
    """
    Convert any budget string → canonical "X lakh" form.
    Examples:
        "25 lakhs"  → "25 lakh"
        "25L"       → "25 lakh"
        "around 25" → "25 lakh"
        "23"        → "23 lakh"
        "1 crore"   → "100 lakh"
        ""          → ""
    """
    if not raw:
        return ""
    raw_lower = str(raw).strip().lower()

    # Handle crore → convert to lakh
    crore_match = re.search(r"(\d+(?:\.\d+)?)\s*crore", raw_lower)
    if crore_match:
        amount = float(crore_match.group(1)) * 100
        val = str(int(amount)) if amount == int(amount) else str(amount)
        return f"{val} lakh"

    # Extract first number (int or float)
    match = re.search(r"(\d+(?:\.\d+)?)", raw_lower)
    if not match:
        return ""

    amount = match.group(1)
    # Remove trailing .0
    if amount.endswith(".0"):
        amount = amount[:-2]

    return f"{amount} lakh"


def _is_joke_budget(budget_str: str) -> bool:
    """
    Return True ONLY if the extracted budget is below ₹50,000 (0.5 lakh),
    meaning it is clearly a joke or unrealistic input (e.g. "1 rupee").

    Any value >= 0.5 lakh is treated as a valid budget and returned as-is.
    This means "5 lakh", "28 lakh", "50 lakh" are all valid — never rejected.
    """
    if not budget_str:
        return False
    match = re.search(r"(\d+(?:\.\d+)?)", budget_str)
    if not match:
        return False
    amount = float(match.group(1))
    return amount < 0.5  # only true for sub-₹50,000 amounts


def _extract_budget_from_text(text: str) -> str:
    """
    Fallback: extract a valid budget directly from the raw user message.

    This is a Python-side safety net for cases where the LLM incorrectly
    drops a valid budget (e.g. misclassifying "28 lakh" as a joke).

    Handles formats like:
        "28 lakh", "28 lakhs", "28L", "28lac", "around 28 lakh",
        "28", "1 crore", "budget of 15 lakh"

    Returns canonical "X lakh" string, or "" if nothing valid found.
    """
    text_lower = text.strip().lower()

    # Handle crore first
    crore_match = re.search(r"(\d+(?:\.\d+)?)\s*crore", text_lower)
    if crore_match:
        amount = float(crore_match.group(1)) * 100
        val = str(int(amount)) if amount == int(amount) else str(amount)
        return f"{val} lakh"

    # Match number followed optionally by lakh/lac/l/lakhs
    # Covers: "28 lakh", "28L", "28lac", "28", "around 28", "28 lakhs"
    pattern = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|l\b)?",
        text_lower
    )
    if pattern:
        amount = float(pattern.group(1))
        # Only accept as a budget if >= 0.5 lakh (not a joke amount)
        if amount >= 0.5:
            val = str(int(amount)) if amount == int(amount) else str(amount)
            return f"{val} lakh"

    return ""


def _merge_attributes(existing: dict, new_attrs: dict) -> dict:
    """
    Merge newly extracted attributes into the existing accumulated dict.
    Rule: NEVER overwrite a valid value with an empty string.
    This ensures attributes collected in earlier turns are preserved.
    """
    merged = dict(existing)
    for key, new_val in new_attrs.items():
        new_val = str(new_val).strip()
        if new_val:  # only update if LLM actually returned something
            merged[key] = new_val
    return merged


def _validate_attributes(attrs: dict) -> dict:
    """
    Validate and clean extracted attributes against allowed value sets.
    Invalid values are set back to "" so the bot re-asks for them.
    Also guards against joke budgets by clearing them here.
    """
    fuel   = str(attrs.get("fuel_type",    "")).strip().lower()
    trans  = str(attrs.get("transmission", "")).strip().lower()
    usage  = str(attrs.get("usage",        "")).strip().lower()
    budget = _normalize_budget(attrs.get("budget", ""))

    # Reject only true joke / unrealistic budgets (< 0.5 lakh)
    if _is_joke_budget(budget):
        budget = ""

    return {
        "fuel_type":    fuel   if fuel   in ALLOWED_FUEL  else "",
        "transmission": trans  if trans  in ALLOWED_TRANS else "",
        "budget":       budget,
        "usage":        usage  if usage  in ALLOWED_USAGE else "",
    }


def _is_ready(attrs: dict) -> bool:
    """All three required attributes must be non-empty."""
    return bool(
        attrs.get("fuel_type")
        and attrs.get("transmission")
        and attrs.get("budget")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_attributes_via_llm(
    user_message: str,
    conversation_history: list[dict],
    client: OpenAI,
    accumulated_attributes: dict | None = None,
) -> dict:
    """
    Send the full conversation history to the LLM and extract car preference
    attributes, merging them with any previously collected attributes.

    Args:
        user_message            — latest message from the user
        conversation_history    — full [{role, content}] history including new msg
        client                  — authenticated OpenAI client
        accumulated_attributes  — attributes collected in previous turns (or None)

    Returns:
        dict with keys:
            'message'                   (str)  — conversational reply for the user
            'attributes'                (dict) — MERGED attributes (all turns combined)
            'ready_for_retrieval'       (bool) — True when fuel+transmission+budget set
            'show_fuel_buttons'         (bool) — show fuel selector buttons
            'show_transmission_buttons' (bool) — show transmission selector buttons
            'show_budget_buttons'       (bool) — show budget input prompt
    """
    if accumulated_attributes is None:
        accumulated_attributes = {
            "fuel_type": "", "transmission": "", "budget": "", "usage": ""
        }

    system_prompt = build_attribute_extraction_prompt()

    # Inject current known attributes into the system prompt so the LLM
    # doesn't re-ask for things already collected
    known_summary = "\n".join(
        f"  - {k}: {v}" for k, v in accumulated_attributes.items() if v
    )
    if known_summary:
        system_prompt += f"""

══════════════════════════════════════════════════════
ALREADY COLLECTED (do NOT ask for these again):
══════════════════════════════════════════════════════
{known_summary}

Only ask for attributes that are still missing from the above list.
"""

    messages = [{"role": "system", "content": system_prompt}] + conversation_history

    # ── LLM call ────────────────────────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content
        llm_result  = json.loads(raw_content)

    except json.JSONDecodeError as e:
        print(f"[LLM HANDLER] JSON parse error: {e}")
        return {
            "message":                   "Sorry, I didn't catch that. Could you repeat your preference?",
            "attributes":                accumulated_attributes,
            "ready_for_retrieval":       False,
            "show_fuel_buttons":         False,
            "show_transmission_buttons": False,
            "show_budget_buttons":       False,
        }
    except Exception as e:
        print(f"[LLM HANDLER] API error: {e}")
        return {
            "message":                   "Something went wrong. Please try again.",
            "attributes":                accumulated_attributes,
            "ready_for_retrieval":       False,
            "show_fuel_buttons":         False,
            "show_transmission_buttons": False,
            "show_budget_buttons":       False,
        }

    # ── Extract & validate new attributes from LLM ──────────────────────────
    raw_new_attrs = llm_result.get("attributes", {})
    validated_new = _validate_attributes(raw_new_attrs)

    # ── KEY FIX 3: Python-side budget recovery ───────────────────────────────
    # If the LLM dropped a valid budget (e.g. misclassified "28 lakh" as a
    # joke), re-extract it directly from the raw user message as a safety net.
    if not validated_new.get("budget"):
        fallback_budget = _extract_budget_from_text(user_message)
        if fallback_budget:
            validated_new["budget"] = fallback_budget
            print(
                f"[LLM HANDLER] ⚠️  LLM dropped valid budget — "
                f"recovered from raw message: '{fallback_budget}'"
            )

    # ── Merge with accumulated state (never lose old values) ─────────────────
    merged = _merge_attributes(accumulated_attributes, validated_new)

    # ── Re-check readiness based on merged state (not LLM's claim) ──────────
    ready = _is_ready(merged)

    if ready != llm_result.get("ready_for_retrieval", False):
        print(
            f"[LLM HANDLER] Overriding LLM ready_for_retrieval: "
            f"LLM={llm_result.get('ready_for_retrieval')} → Actual={ready}"
        )

    # If ready, message must be empty (retrieval handles the response)
    message = "" if ready else llm_result.get("message", "")

    # ── Button logic — driven by current_question, NOT just empty attrs ──────
    #
    # current_question tells us what the LLM message is ACTIVELY asking about
    # this turn. This prevents fuel buttons from showing during a budget-
    # redirect message (e.g. when user says "my budget is 1 rupee").
    # ────────────────────────────────────────────────────────────────────────

    fuel_collected         = bool(merged.get("fuel_type"))
    transmission_collected = bool(merged.get("transmission"))
    budget_collected       = bool(merged.get("budget"))

    current_question = llm_result.get("current_question", "").strip().lower()

    # If budget was recovered by Python fallback, override current_question
    # so the UI doesn't show budget buttons when we're already ready
    if ready:
        current_question = "none"

    show_fuel_buttons         = False
    show_transmission_buttons = False
    show_budget_buttons       = False

    if not ready:
        if current_question in ("fuel", "transmission", "budget"):
            # Trust LLM's declared intent — only show buttons matching question
            if current_question == "fuel" and not fuel_collected:
                show_fuel_buttons = True
            elif current_question == "transmission" and not transmission_collected:
                show_transmission_buttons = True
            elif current_question == "budget" and not budget_collected:
                show_budget_buttons = True
        else:
            # Fallback: LLM didn't set current_question — infer sequentially
            print("[LLM HANDLER] current_question missing or 'none' — using sequential fallback")
            if not fuel_collected:
                show_fuel_buttons = True
            elif not transmission_collected:
                show_transmission_buttons = True
            elif not budget_collected:
                show_budget_buttons = True

    # ── Debug logging ────────────────────────────────────────────────────────
    print(f"[ATTR EXTRACT] Raw LLM attrs    : {raw_new_attrs}")
    print(f"[ATTR EXTRACT] Validated        : {validated_new}")
    print(f"[ATTR EXTRACT] Merged state     : {merged}")
    print(f"[ATTR EXTRACT] Ready            : {ready}")
    print(f"[ATTR EXTRACT] current_question : {current_question}")
    print(
        f"[ATTR EXTRACT] show_fuel={show_fuel_buttons}  "
        f"show_trans={show_transmission_buttons}  "
        f"show_budget={show_budget_buttons}"
    )

    return {
        "message":                   message,
        "attributes":                merged,
        "ready_for_retrieval":       ready,
        "show_fuel_buttons":         show_fuel_buttons,
        "show_transmission_buttons": show_transmission_buttons,
        "show_budget_buttons":       show_budget_buttons,
    }