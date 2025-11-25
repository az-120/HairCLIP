import torch
from PIL import Image
import numpy as np
import cv2
from diffusers import AutoPipelineForInpainting
import matplotlib.pyplot as plt
from hair_segmentation import get_editable_mask

pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# pipe.enable_xformers_memory_efficient_attention() TODO: properly use this

bgr = cv2.imread("data/test/IMG_2817.png")

if bgr is None:
    raise ValueError("Image not found at data/test/IMG_2817.png")

# Convert to RGB for SDXL
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

mask_np_full = get_editable_mask(bgr)

# Ensure mask is single-channel
if len(mask_np_full.shape) == 3:
    mask_np_full = mask_np_full[:, :, 0]

# Convert to uint8 0–255 if needed
if mask_np_full.max() <= 1:
    mask_np_full = (mask_np_full * 255).astype(np.uint8)
else:
    mask_np_full = mask_np_full.astype(np.uint8)


target_size = (1024, 1024)

rgb_small = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
mask_small = cv2.resize(mask_np_full, target_size, interpolation=cv2.INTER_NEAREST)

cv2.imwrite("data/test/debug_mask_small.png", mask_small)

# Convert to PIL for diffusers
image_pil = Image.fromarray(rgb_small)
mask_pil = Image.fromarray(mask_small).convert("L")

prompt = "asian man with buzz cut"

with torch.no_grad():
    out = pipe(
        prompt=prompt,
        image=image_pil,
        mask_image=mask_pil,
        guidance_scale=7.5,
        num_inference_steps=20,
        strength=0.9
    )

result_pil = out.images[0]

result_np = np.array(result_pil)[:, :, ::-1]
cv2.imwrite("data/test/diffusion.png", result_np)
