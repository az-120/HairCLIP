import torch
from PIL import Image
import numpy as np
import cv2

from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")


from hair_segmentation import get_editable_mask
image = cv2.imread("data/test/IMG_2817.png")
mask = get_editable_mask(image)

# .resize((1024, 1024))


prompt = "asian man with buzz cut"
generator = torch.Generator(device="mps").manual_seed(0)

image = pipe(
  prompt=prompt,
  image=image,
  mask_image=mask,
  guidance_scale=8.0,
  num_inference_steps=20,  # steps between 15 and 30 work well for us
  strength=0.99,  # make sure to use `strength` below 1.0
  generator=generator,
).images[0]

cv2.imwrite("data/test/diffusion.png")

# class SDXLInpaint:
#     def __init__(self, model_id="stabilityai/sdxl-inpainting-0.1", device=None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
#         self.pipe = AutoPipelineForInpainting.from_pretrained(
#             model_id,
#             torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#             variant="fp16" if self.device == "cuda" else None,
#         ).to(self.device)

#     def _bgr_to_pil(self, img_bgr):
#         """Convert BGR (OpenCV) to RGB PIL."""
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         return Image.fromarray(img_rgb)

#     def _mask_to_pil(self, mask):
#         """
#         mask: np.ndarray (H,W) with values {0,1}
#         Diffusers requires RGB mask where:
#         - white (255) = editable
#         - black (0)   = preserved
#         """
#         mask_uint8 = (mask * 255).astype(np.uint8)
#         return Image.fromarray(mask_uint8, mode="L")  # grayscale

#     def generate(self, img_bgr, mask, prompt, num_inference_steps=50, guidance_scale=7.5):
#         """
#         img_bgr: input image in BGR (OpenCV)
#         mask: binary mask (H,W) with 1=editable, 0=protected
#         prompt: text prompt (e.g., "give me a buzz cut")
#         """

#         img_pil = self._bgr_to_pil(img_bgr)
#         mask_pil = self._mask_to_pil(mask)

#         # Optional: upscale mask & image to 1024x1024 (SDXL works best at 1024)
#         img_pil = img_pil.resize((1024, 1024), Image.LANCZOS)
#         mask_pil = mask_pil.resize((1024, 1024), Image.NEAREST)

#         result = self.pipe(
#             prompt=prompt,
#             image=img_pil,
#             mask_image=mask_pil,
#             guidance_scale=guidance_scale,
#             num_inference_steps=num_inference_steps,
#         )

#         return result.images[0]
    

# from hair_segmentation import get_editable_mask
# import cv2

# img = cv2.imread("data/test/IMG_2817.png")

# mask = get_editable_mask(img)

# pipe = SDXLInpaint()

# output = pipe.generate(
#     img_bgr=img,
#     mask=mask,
#     prompt="give me a buzz cut"
# )

# output.save("result.png")