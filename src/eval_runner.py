import csv
import cv2
from diffusion import run_diffusion
from masking import get_editable_mask
from metrics import identity_score, prompt_score, locality_score

# Grid
# guidance_grid = [5, 8.5, 12]
# strength_grid  = [0.5, 0.7, 0.9, 0.99999]
# steps_grid     = [20, 35, 50]

# dataset = [
#     ("data/test/headshotclip.jpg", "photo of a person with a buzzcut hairstyle"),
#     # ("data/test/headshotclip.jpg", "medium wavy hair"),
#     # ("data/test/headshotclip.jpg", "long flowy hair"),
#     # ("data/test/matthew.png", "buzzcut hairstyle"),
#     # ("data/test/matthew.png", "long curly hair"),
#     # ("data/test/matthew.png", "wavy hair"),
#     # ("data/test/stockportrait.png", "very short hair"),
#     # ("data/test/stockportrait.png", "red hair"),
#     # ("data/test/stockportrait.png", "long straigh hair"),
# ]
#
# For Prompt Evaluations:
guidance_grid = [10]
strength_grid  = [0.99999]
steps_grid     = [30]

dataset = [
    ("data/test/headshotclip.jpg", "buzzcut"),
    ("data/test/headshotclip.jpg", "buzzcut hairstyle"),
    ("data/test/headshotclip.jpg", "person with a buzzcut"),
    ("data/test/headshotclip.jpg", "photo of a person with a buzzcut hairstyle"),
    ("data/test/headshotclip.jpg", "portrait of a person with a buzzcut hairstyle"),
    ("data/test/headshotclip.jpg", "realistic person with a clean buzzcut hairstyle"),
    ("data/test/headshotclip.jpg", "close-up portrait of a person with a buzzcut haircut"),
    ("data/test/headshotclip.jpg", "high-detail portrait of a person with a buzzcut hairstyle, realistic skin texture"),
    ("data/test/headshotclip.jpg", "studio photo of a person with a sharp buzzcut haircut, dramatic lighting"),
    ("data/test/headshotclip.jpg", "ultra-realistic photo of a person with a very short buzzcut"),
    ("data/test/headshotclip.jpg", "cinematic portrait of a person with a buzzcut, shallow depth of field"),
    ("data/test/headshotclip.jpg", "person with freshly-trimmed buzzcut hair in natural lighting"),
    ("data/test/headshotclip.jpg", "The person in the image should now have a short buzzcut hairstyle."),
    ("data/test/headshotclip.jpg", "Transform the subject’s hair into a clean buzzcut while keeping facial features unchanged."),
    ("data/test/headshotclip.jpg", "Render the person with a natural-looking buzzcut haircut in a realistic photographic style."),
    ("data/test/headshotclip.jpg", "Change the subject's hairstyle to a buzzcut, ensuring the new hair matches lighting, background, and perspective."),
]

with open("results.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image",
        "prompt",
        "guidance",
        "strength",
        "steps",
        "identity",
        "prompt_sim",
        "locality"
    ])

    for img_path, prompt in dataset:
        orig_bgr = cv2.imread(img_path)
        mask = get_editable_mask(orig_bgr)

        for g in guidance_grid:
            for s in strength_grid:
                for steps in steps_grid:

                    edited = run_diffusion(
                        orig_bgr,
                        mask,
                        prompt,
                        guidance_scale=g,
                        strength=s,
                        num_inference_steps=steps
                    )

                    id_score = identity_score(orig_bgr, edited)
                    pr_score = prompt_score(edited, prompt)
                    loc_score = locality_score(orig_bgr, edited, mask)

                    writer.writerow([
                        img_path,
                        prompt,
                        g, s, steps,
                        id_score,
                        pr_score,
                        loc_score,
                    ])