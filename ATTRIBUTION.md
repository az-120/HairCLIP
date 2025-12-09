# Attribution

This document provides credit for all resources used in HairCLIP.

## AI Assistance

- Generating template files and initial structure of the project.
- Updating the .gitignore file.
- Adding docstrings to files.
- Creating the csv output file in `src/hyperparameter_selection.py`.
- Creating the prompt engineering evaluation markdown in `data/prompt_engineering/prompts.md`.

## Third-Party Libraries

- PIL, cv2, numpy: used for image processing.
- HuggingFace Diffusers: for stable diffusion pipelining and LoRA fine-tuning (https://github.com/huggingface/diffusers/tree/main/examples/text_to_image).
- insightface: identity similarity metric.

## Datasets

- FFHQ (Flickr-Faces-HQ) Dataset - contains 70,000 high quality images of various faces without image captions
- Self curated dataset for finetuning hairstyles the model was less confident on (afro, balayage, bob cut, butterfly cut, cornrows, fade, mullet, pixie cut, warrior cut, waves, wolf cut)

## Pretrained Models

- MediaPipe Image Segmenter model: https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter
- Stable Diffusion XL Inpainting: https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1
