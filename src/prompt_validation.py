import os
from typing import Dict, List
import torch
import clip
from pathlib import Path
import re

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Hairstyle prompts for similarity check
VALID_HAIRSTYLES = {
    "buzz cut", "fade", "taper", "crew cut", "undercut", "mullet",
    "pompadour", "quiff", "curtain bangs", "wolf cut", "bob", "pixie",
    "layered cut", "shag", "afro", "braids", "cornrows", "twists", "locs", 
    "dreadlocks", "balayage", "cornrows",  "warrior cut", "butterfly cut", "flow", "part", "hair"
}

SIM_THRESHOLD = 0.3  # minimum cosine similarity to accept prompt

# Non-hair edits to block
NON_HAIR_KEYWORDS = {
    "face", "nose", "eyes", "lips", "jaw", "chin", "cheek", "skin",
    "color", "race", "ethnicity", "younger", "older", "age", "gender",
    "make me look", "change my", "fix my", "reshape", "body", "muscle", "height"
}


print("Loading CLIP model...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

with torch.no_grad():
    canonical_tokens = clip.tokenize(VALID_HAIRSTYLES).to(DEVICE)
    canonical_embeddings = clip_model.encode_text(canonical_tokens)
    canonical_embeddings /= canonical_embeddings.norm(dim=-1, keepdim=True)


def clip_validate_prompt(prompt: str) -> str:
    """
    Return the prompt only if it passes validation.
    """

    if not isinstance(prompt, str) or len(prompt.strip()) == 0:
        raise ValueError("Prompt must be a non-empty string.")

    p = prompt.strip().lower()

    if len(p) > 20000:
        raise ValueError("Input text is too long. Maximum allowed characters: 20,000.")

    for word in NON_HAIR_KEYWORDS:
        if word in p:
            raise ValueError(f"Prompt includes non-hair edits ('{word}').")

    with torch.no_grad():
        tokenized = clip.tokenize([prompt]).to(DEVICE)
        prompt_embedding = clip_model.encode_text(tokenized)
        prompt_embedding /= prompt_embedding.norm(dim=-1, keepdim=True)
        sims = (prompt_embedding @ canonical_embeddings.T).squeeze(0)
        max_score = sims.max().item()

    if max_score < SIM_THRESHOLD:
        raise ValueError(f"Prompt too dissimilar from recognized hairstyle prompts (max similarity {max_score:.2f}).")

    return prompt


if __name__ == "__main__":
    test_prompts = [
        "Give me a fade hairstyle",
        "I want a wolf cut",
        "Make my jawline sharper and give me a bob",
        "Give me blue eyes and a pixie cut",
        "Try a braid and a bun",
        "I'd like a mullet, please",
        "Short hair",
        "aaaaaaaaaaaaaaaaaaaa"
    ]

    for prompt in test_prompts:
        result = clip_validate_prompt(prompt)
        print(f"Prompt: '{prompt}'\nResult: {result}\n")