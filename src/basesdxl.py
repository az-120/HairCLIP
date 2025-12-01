from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image
from PIL import Image
import cv2
import torch

# 1. Load SDXL Base
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

# 2. Load the IP-Adapter for SDXL (Face or Image version)
ip_adapter = AutoPipelineForText2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  torch_dtype=torch.float16
).to("cuda")
ip_adapter.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  weight_name="ip-adapter_sdxl.bin"
)
ip_adapter.set_ip_adapter_scale(0.8)

# 3. Load your original image
init_image = Image.open("data/test/IMG_2817.PNG").convert("RGB")
init_image = init_image.resize((1024, 1024))

# 4. Prompt for drastic edit
prompt = "a portrait photo of the same person with short blonde hair, cinematic lighting"

# 5. Generate pre-edit (big global changes)
out = ip_adapter(
    prompt=prompt,
    image=init_image,          # ← use the original image as conditioning
    num_inference_steps=30,
    guidance_scale=5.0,
    strength=0.7,              # controls how far from original image to move
)

res = out.images[0].save("pre_edit_output.png")
cv2.imwrite("data/test/nomaskres.png", res)