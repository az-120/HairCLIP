# HairCLIP

[Deployed Web App](https://huggingface.co/spaces/arthurzhao120/HairCLIP)

HairCLIP lets users upload or take an image and preview various hairstyles using text prompts.


## What it Does

The inputted image is first run through a Mediapipe segmentation model. The hair is initially masked out, and then this mask is expanded outward and downward to allow for hair growth in these areas. The face segment is then subtracted out, leaving a mask where the white area is where stable diffusion will inpaint on. See data/test/multimask.png for an example mask on the headshotclip.jpg input. This mask, the original image, and the validated text prompt are then passed into the stable diffusion pipeline. Here, the image is normalized and processed into the proper format and size for the model. We use the Stable Diffusion XL Inpainting model from HuggingFace to produce a generated image, which is then resized back to its original dimensions before being outputted.


## Quick Start
### User Interface
1. Head to https://huggingface.co/spaces/arthurzhao120/HairCLIP, upload/take/paste an image, enter a hairstyle in the textbox, and click generate! (Note to grader: if the HuggingFace Space is down, email adz6@duke.edu and I'll try to get it up as quick as possible).

### Manually on Local (harder to replicate here, not sure if necessary since we have the UI)
1. Follow `SETUP.md` to create a Python virtual environment and install `requirements.txt`. You must have the repo cloned and setup in your own environment.
2. Place raw image inside `data/test`.
3. In `src/run_inpainting.py`, in the __main__ block, make sure the proper image is being read, and type in your hairstyles in the prompts array.
4. Commit and push these changes to your own Git repo.
5. Open the `notebooks/hairclip_diffusion.ipynb` in Google Colab (if you don't have cuda support).
6. Follow the instructions in this notebook, making sure to update the repo name and branch that you'll be using.
7. Running the first 9 code cells will transform your inputted image based on the prompts. In the 9th cell, make sure to update the number of prompts the outputter iterates through.

Sample Output (replace with your own once generated):
![Sample hairstyle preview](docs/sample_output.png)

## Video Links

- Demo video: ❗ add public link (e.g., Google Drive or YouTube) showcasing end-to-end usage.
- Technical walkthrough: ❗ add link explaining architecture, training strategy, and evaluation.

## Evaluation

- Quantitative: our metrics are in `src/metrics.py`: identity, prompt, and locality. These measure how much the generated image preserves facial identity, adherence to the text prompt, and overall photorealism, respectively. Results vary from image to image, but scores are calcuated in the `data/hyperparameters` folder in the csvs, where the 3 scores are calculated for different images, prompts, and model parameters.
- Qualitative: the same 3 general metrics are more effective when conducted qualitatively by the human eye, and we did not rely solely on quantitative metrics. It's easier to tell how well the model performs by simply observing and understanding from the human eye. 

## Individual Contributions

| Team Member       | Contributions |
| ----------------- | ------------- |
| Arthur Zhao    | Implemented and refined hair/face segmentation, developed stable diffusion inpainting pipeline, created metrics and ran them over various hyperparaters and prompting styles, added lora finetuning pipeline, deployed to web via Gradio UI.        |
| Leon Eberhardt | filler        |
