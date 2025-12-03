import csv
import cv2
from diffusion import run_diffusion
from masking import get_editable_mask
from metrics import identity_score, prompt_score, locality_score

# Grid
guidance_grid = [5, 7.5, 10, 12, 15]
strength_grid  = [0.3, 0.5, 0.7]
steps_grid     = [20, 30]

dataset = [
    ("data/test/IMG_2817.png", "buzzcut hairstyle"),
    ("data/test/IMG_2817.png", "medium wavy hair"),
    ("data/test/IMG_2817.png", "long flowy hair"),
    ("data/test/matthew.png", "buzzcut hairstyle"),
    ("data/test/matthew.png", "long curly hair"),
    ("data/test/matthew.png", "wavy hair"),
    ("data/test/stockportrait.png", "very short hair"),
    ("data/test/stockportrait.png", "red hair"),
    ("data/test/stockportrait.png", "long straigh hair"),
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