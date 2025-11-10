# HairCLIP

HairCLIP lets users upload a selfie and preview various hairstyles on themselves using text prompts. The pipeline fine-tunes HairCLIP-style latent editing on top of StyleGAN/CLIP to produce photorealistic previews in seconds.

## What it Does

Given a single selfie, HairCLIP extracts facial embeddings, aligns the image, and performs latent edits guided by either text prompts ("short wavy bob") or sample hairstyle references. The system then renders multiple candidates, tracks edit history so users can compare looks, and exports the chosen hairstyle image at high resolution for salons or personal planning.

## Quick Start

1. Follow `SETUP.md` to create a Python virtual environment and install `requirements.txt`.
2. Place raw selfies or sample inputs inside `data/inputs/` and pretrained weights inside `models/` as described in `SETUP.md`.
3. Run the CLI demo:
   ```bash
   python src/app.py --input data/inputs/sample.jpg --prompt "short wavy bob"
   ```
4. (Optional) Launch the Gradio UI for interactive editing:
   ```bash
   python src/ui/gradio_app.py
   ```
5. Generated previews are written to `data/outputs/` for later evaluation.

Sample Output (replace with your own once generated):
![Sample hairstyle preview](docs/sample_output.png)

## Video Links

- Demo video: ❗ add public link (e.g., Google Drive or YouTube) showcasing end-to-end usage.
- Technical walkthrough: ❗ add link explaining architecture, training strategy, and evaluation.

## Evaluation

- Quantitative: ❗ report metrics such as CLIP text-image similarity, LPIPS diversity, or user study satisfaction scores.
- Qualitative: include before/after grids or GIFs (place assets inside `docs/` or embed via markdown once available).
- Reproduction: document the exact commands or notebooks used to regenerate metrics (see `notebooks/` for experiments).

## Individual Contributions

| Team Member       | Contributions |
| ----------------- | ------------- |
| ❗ Arthur Zhao    | filler        |
| ❗ Leon Eberhardt | filler        |
