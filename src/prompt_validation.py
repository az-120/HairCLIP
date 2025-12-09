import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


VALID_HAIRSTYLES = sorted({
    "buzz cut", "fade", "taper", "crew cut", "undercut", "mullet",
    "pompadour", "quiff", "curtain bangs", "wolf cut", "bob", "pixie",
    "layered cut", "shag", "afro", "braids", "cornrows", "twists", "locs",
    "dreadlocks", "balayage", "warrior cut", "butterfly cut", "flow", "part",
    "hair", "side part", "middle part", "comb over", "mohawk", "faux hawk",
    "top knot", "man bun", "bob cut", "lob", "fringe", "bangs",
    "short hair", "medium length hair", "long hair", "curly hair",
    "wavy hair", "straight hair"
})


# Sentence Embedding Model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


def embed_texts(texts):
    with torch.no_grad():
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(device)

        model_out = model(**encoded)

        token_embeddings = model_out.last_hidden_state
        attention_mask = encoded.attention_mask.unsqueeze(-1)

        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        embeddings = summed / counts

        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


CANONICAL_PHRASES = [f"a {name} hairstyle" for name in VALID_HAIRSTYLES]
CANONICAL_EMBEDS = embed_texts(CANONICAL_PHRASES)


def best_hairstyle_match(prompt: str):
    if prompt is None:
        return None, 0.0

    cleaned = prompt.strip()
    if not cleaned:
        return None, 0.0

    emb = embed_texts([cleaned])
    sims = torch.matmul(CANONICAL_EMBEDS, emb[0])
    best_idx = int(torch.argmax(sims).item())
    best_score = float(sims[best_idx].item())
    best_name = VALID_HAIRSTYLES[best_idx]

    return best_name, best_score


def is_valid_hairstyle_prompt(prompt: str, threshold: float = 0.333):
    best_name, score = best_hairstyle_match(prompt)
    return score >= threshold, best_name, score


if __name__ == "__main__":
    tests = [
        # clearly valid
        "buzz cut",
        "give me a buzzcut",
        "long wavy hair",
        "I want an afro hairstyle",
        "wolfcut",
        "make my hair into curtain bangs",
        "short taper fade",
        "curly bob",
        "mullet please",

        # ambiguous / borderline
        "make me look older",
        "cleaner haircut",
        "better hair",
        "more volume",
        "change my style",

        # invalid (should fail)
        "turn me into an oompa loompa",
        "make me look like Shrek",
        "give me a lightsaber",
        "put me in a tuxedo",
        "blue background",
        "add a moustache",
        "banana",
    ]

    print("\n=== PROMPT VALIDATION TESTS ===\n")
    for t in tests:
        is_valid, best, score = is_valid_hairstyle_prompt(t, threshold=0.30)
        print(f"Prompt: {t!r}")
        print(f"  → is_valid: {is_valid}")
        print(f"  → best match: {best}  (score={score:.3f})")
        print()
