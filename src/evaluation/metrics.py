import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# Identity Score
import insightface
arcface_model = insightface.app.FaceAnalysis(name="buffalo_s")
arcface_model.prepare(ctx_id=0)

def identity_score(orig, edited):
    f1 = arcface_model.get(orig)[0].embedding
    f2 = arcface_model.get(edited)[0].embedding
    f1 = f1 / np.linalg.norm(f1)
    f2 = f2 / np.linalg.norm(f2)
    return float(np.dot(f1, f2))


# Prompt Score
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def prompt_score(image, prompt):
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    img_emb = outputs.image_embeds[0]
    txt_emb = outputs.text_embeds[0]
    img_emb = img_emb / img_emb.norm()
    txt_emb = txt_emb / txt_emb.norm()
    return float((img_emb @ txt_emb.T))


# Locality Score
def locality_score(orig, edited, mask):
    inv = 1 - mask
    diff = ((orig - edited) * inv[..., None]) ** 2
    return float(1.0 / (1.0 + diff.mean()))