import torch
from PIL import Image
import numpy as np
import cv2
from diffusers import AutoPipelineForInpainting

pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

bgr = cv2.imread("data/test/IMG_2817.png")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

rgb = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA)

image_pil = Image.fromarray(rgb)


from hair_segmentation import get_editable_mask

mask_np = get_editable_mask(rgb)

mask_pil = Image.fromarray(mask_np).convert("L")

prompt = "asian man with buzz cut"

with torch.no_grad():
    out = pipe(
        prompt=prompt,
        image=image_pil,
        mask_image=mask_pil,
        guidance_scale=7.5,
        num_inference_steps=20,
        strength=0.9,
    )

result = np.array(out.images[0])[:, :, ::-1]  # RGB → BGR for cv2
# cv2.imwrite("data/test/diffusion.png", result)