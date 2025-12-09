import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import AutoPipelineForInpainting


def load_model(device="cuda"):
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)

    pipe.load_lora_weights("models", weight_name="pytorch_lora_weights.safetensors")

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

def resize_and_pad_to_1024(image_rgb, mask):
    target = 1024
    orig_h, orig_w = image_rgb.shape[:2]

    scale = min(target / orig_w, target / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    if new_w == 0 or new_h == 0:
        raise ValueError("Invalid resize dimensions computed.")

    interp_img = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    img_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=interp_img)
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    pad_top = (target - new_h) // 2
    pad_bottom = target - new_h - pad_top
    pad_left = (target - new_w) // 2
    pad_right = target - new_w - pad_left

    img_padded = cv2.copyMakeBorder(
        img_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
    )
    mask_padded = cv2.copyMakeBorder(
        mask_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
    )

    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "new_h": new_h,
        "new_w": new_w,
        "pad_top": pad_top,
        "pad_left": pad_left,
    }
    return img_padded, mask_padded, meta

def unpad_and_resize(result_rgb, meta):
    new_h = meta["new_h"]
    new_w = meta["new_w"]
    pad_top = meta["pad_top"]
    pad_left = meta["pad_left"]
    orig_h = meta["orig_h"]
    orig_w = meta["orig_w"]

    cropped = result_rgb[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    if (new_w, new_h) == (orig_w, orig_h):
        return cropped

    resized = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    return resized

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

    rgb_padded, mask_padded, meta = resize_and_pad_to_1024(rgb, mask_uint8)

    image_pil, mask_pil = to_pil(rgb_padded, mask_padded)

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
    result_rgb = unpad_and_resize(result_np, meta)
    result_bgr = result_rgb[:, :, ::-1]

    return result_bgr


# Test
if __name__ == "__main__":
    from masking import get_editable_mask

    bgr = cv2.imread("data/test/headshotclip.jpg")
    mask = get_editable_mask(bgr)

    cv2.imwrite("data/test/Tanium_Badge.jpeg", mask * 255)

    prompts = [
        "a picture of a person with a mullet hairstyle",
        "a picture of a person with a wolfcut hairstyle",
        "a picture of a person with a low taper fade hairstyle",
        "a picture of a person with a buzzcut hairstyle"
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
