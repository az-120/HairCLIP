import re
from typing import Dict, List

# Hairstyle list
VALID_HAIRSTYLES = {
    "buzzcut", "fade", "taper", "crew cut", "undercut", "mullet",
    "pompadour", "quiff", "curtain bangs", "wolf cut", "bob", "pixie",
    "layered cut", "shag", "afro", "braids", "cornrows", "twists", "locs", 
    "dreadlocks", "balayage", "cornrows",  "warrior cut", "butterfly cut", "flow", "part", "hair"
}

# Non-hair edits to block
NON_HAIR_KEYWORDS = {
    "face", "nose", "eyes", "lips", "jaw", "chin", "cheek", "skin",
    "color", "race", "ethnicity", "younger", "older", "age", "gender",
    "make me look", "change my", "fix my", "reshape", "body", "muscle", "height"
}

def validate_prompt(text: str) -> Dict[str, object]:
    """
    Combined validation:
    1. General input quality
    2. Hairstyle presence
    3. Blocks non-hair edits
    4. Provides recommendations/warnings
    """
    errors: List[str] = []
    warnings: List[str] = []
    matched_style = None

    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    text = text.strip()

    if len(text) == 0:
        raise ValueError("Input text cannot be empty.")

    if len(text) < 5:
        raise ValueError("Input text is too short. Please provide at least 20 characters.")

    if len(text) > 20000:
        raise ValueError("Input text is too long. Maximum allowed characters: 20,000.")

    if len(set(text)) < 5:
        raise ValueError("Input text appears invalid or low-quality.")

    
    p = text.lower()

    
    for word in NON_HAIR_KEYWORDS:
        if word in p:
            errors.append(
                f"Prompt includes non-hair changes ('{word}'). Only hairstyle edits are allowed."
            )
            return {"valid": False, "errors": errors, "warnings": warnings, "style": None}

  
    for style in VALID_HAIRSTYLES:
        if style in p:
            matched_style = style
            break

    if matched_style is None:
        errors.append("No recognizable hairstyle found in the prompt. Please specify a hairstyle.")
        return {"valid": False, "errors": errors, "warnings": warnings, "style": None}

   -
    recommended_pattern = r"give me (a |an )?" + re.escape(matched_style)
    if not re.search(recommended_pattern, p):
        warnings.append(
            "Consider using the format 'Give me a <hairstyle> hairstyle' to make prompts clearer."
        )

   
    if p.count("and") + p.count(",") > 1:
        warnings.append("Prompt may contain multiple topics; consider keeping it focused on one hairstyle.")

    return {
        "valid": True,
        "errors": errors,
        "warnings": warnings,
        "style": matched_style
    }


if __name__ == "__main__":
    test_prompts = [
        "Give me a fade hairstyle",
        "I want a wolf cut",
        "Make my jawline sharper and give me a bob",
        "Give me blue eyes and a pixie cut",
        "Try a braid and a bun",
        "I'd like a mullet, please",
        "aaaaaaaaaaaaaaaaaaaa",  
        "Short"  
    ]

    for prompt in test_prompts:
        try:
            result = validate_prompt(prompt)
            print(f"Prompt: '{prompt}'\nResult: {result}\n")
        except ValueError as e:
            print(f"Prompt: '{prompt}'\nValidation Error: {e}\n")