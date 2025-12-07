import math, os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.optimization import get_cosine_schedule_with_warmup


# LoRA Hairstyle Dataset (quickly custom built, could expand)
class HairstylesDataset(Dataset):
    def __init__(self, root_dir, image_size=1024):
        self.samples = []
        root = Path(root_dir)

        for style_dir in root.iterdir():
            if not style_dir.is_dir():
                continue
            style_name = style_dir.name.replace("_", " ")
            prompt = f"a photo of a person with a {style_name} hairstyle"

            for img_path in style_dir.glob("*"):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                    continue
                self.samples.append((img_path, prompt))

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, prompt = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return {"pixel_values": img, "prompt": prompt}


# LoRA Setup
def add_lora_to_unet(unet, rank=8):
    lora_attn_procs = {}

    for name, module in unet.named_modules():
        if hasattr(module, "set_attn_processor") and hasattr(module, "processor"):
            cfg = module.processor.config

            hidden_size = cfg.hidden_size
            cross_dim = cfg.cross_attention_dim

            lora_attn_procs[name] = LoRAAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_dim,
                rank=rank,
            )

    unet.set_attn_processor(lora_attn_procs)

    lora_params = []
    for proc in unet.attn_processors.values():
        lora_params.extend(list(proc.parameters()))

    return lora_params


# Main Training Function
def train_lora_sdxl(
    train_data_dir,
    output_dir="data/lora-res-1",
    pretrained="stabilityai/stable-diffusion-xl-base-1.0",
    lr=1e-4,
    rank=8,
    image_size=1024,
    batch_size=2,
    epochs=2
):
    device = "cuda"
    os.makedirs(output_dir, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        pretrained,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to(device)

    # No grad
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # Add LoRA to UNet
    lora_params = add_lora_to_unet(pipe.unet, rank=rank).to(device)
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    dataset = HairstylesDataset(train_data_dir, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    steps_per_epoch = math.ceil(len(dataset)/batch_size)
    total_steps = steps_per_epoch * epochs

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    noise_scheduler = pipe.scheduler
    step = 0

    pipe.unet.train()

    for ep in range(epochs):
        for batch in dataloader:

            with torch.no_grad():
                pixels = batch["pixel_values"].to(device, dtype=torch.float16)
                latents = pipe.vae.encode(pixels).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                          (latents.shape[0],), device=device)

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                prompt_embeds, pooled, add_ids = pipe.encode_prompt(
                    batch["prompt"], device=device
                )

            noise_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": add_ids},
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % 50 == 0:
                print(f"Epoch {ep}, Step {step}/{total_steps}, Loss={loss.item():.4f}")

            if step >= total_steps:
                break
        if step >= total_steps:
            break

    pipe.save_attn_procs(output_dir)
    print(f"Saved LoRA weights to: {output_dir}")


# Evaluation loop
def evaluate_lora(
    image_path,
    prompt,
    lora_dir,
    output_path="data/lora/eval.png",
    pretrained="stabilityai/stable-diffusion-xl-base-1.0"
):
    from PIL import Image
    import numpy as np
    import cv2

    device = "cuda"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        pretrained,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to(device)

    orig = Image.open(image_path).convert("RGB")
    orig_np = np.array(orig)

    before = pipe(prompt=prompt).images[0]

    pipe.load_attn_procs(lora_dir)

    after = pipe(prompt=prompt).images[0]

    before_np = np.array(before)
    after_np = np.array(after)

    combined = np.hstack([orig_np, before_np, after_np])
    Image.fromarray(combined).save(output_path)

    print(f"Saved evaluation comparison to {output_path}")


if __name__ == "__main__":
    train_lora_sdxl(
        train_data_dir="data/lora",
        output_dir="data/lora-res-1",
        lr=1e-4,
        rank=8,
        epochs=3,
    )

    evaluate_lora(
        image_path="data/test/headshotclip.jpg",
        prompt="a person with a wolfcut hairstyle",
        lora_dir="data/lora-res-1",
        output_path="data/lora/eval.png"
    )