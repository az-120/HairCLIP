import torch
from PIL import Image
import numpy as np
import cv2
from diffusers import AutoPipelineForInpainting


def load_model(device="cuda"):
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)

    print("Trying to load lora weights...")
    pipe.load_lora_weights("models", weight_name="pytorch_lora_weights.safetensors")
    print("Loaded!")

    pipe.enable_xformers_memory_efficient_attention()
    return pipe


def ensure_mask_uint8(mask):
    # If 3-channel, collapse to 1-channel
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    mask = mask.astype(np.float32)

    # Convert 0–1 masks to 0–255
    if mask.max() <= 1.0:
        mask = mask * 255.0

    return mask.astype(np.uint8)

def resize_to_1024(image_rgb, mask):
    target_size = (1024, 1024)

    img_small = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    return img_small, mask_small

def to_pil(img_rgb, mask):
    image_pil = Image.fromarray(img_rgb)
    mask_pil = Image.fromarray(mask).convert("L")
    return image_pil, mask_pil


def run_diffusion(
    orig_bgr,
    mask,
    prompt,
    guidance_scale=12,
    strength=0.99999,
    num_inference_steps=20
):
    """
    Runs SDXL inpainting.
    
    Args:
        orig_bgr: input image (NumPy BGR)
        mask: mask (0 or 255)
        prompt: text prompt
        guidance_scale: classifier-free guidance
        strength: inpainting strength
        num_inference_steps: diffusion steps
        
    Returns:
        output_bgr: edited image in BGR format
    """

    pipe = load_model()

    rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    mask_uint8 = ensure_mask_uint8(mask)

    rgb_small, mask_small = resize_to_1024(rgb, mask_uint8)

    image_pil, mask_pil = to_pil(rgb_small, mask_small)

    with torch.no_grad():
        out = pipe(
            prompt=prompt,
            image=image_pil,
            mask_image=mask_pil,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength
        )

    result_pil = out.images[0]
    result_np = np.array(result_pil)
    result_bgr = result_np[:, :, ::-1]

    return result_bgr


# Test
if __name__ == "__main__":
    from masking import get_editable_mask

    bgr = cv2.imread("data/test/headshotclip.jpg")
    mask = get_editable_mask(bgr)

    cv2.imwrite("data/test/mask_debug.png", mask * 255)

    prompts = [
        "wolf cut",
        "person with a wolf cut hairstyle",
        "edgar cut",
        "mod cut",
        "warrior cut",
        "person with a warrior cut hairstyle",
        "person with twists hairstyle",
        "waves hairstyle",
        "modern mullet hairstyle"
    ]

    for i, pr in enumerate(prompts, 1):
        out_bgr = run_diffusion(
            orig_bgr=bgr,
            mask=mask,
            prompt=pr,
            guidance_scale=10,
            strength=0.99999,
            num_inference_steps=30
        )
        cv2.imwrite(f"data/test/output_{i}.png", out_bgr)