import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hair_segmenter.tflite"

BaseOptions = python.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.IMAGE,
    output_category_mask=True,
)

segmenter = vision.ImageSegmenter.create_from_options(options)

def expand_mask(mask, amount=40):
    kernel = np.ones((amount, amount), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)

def get_hair_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Given a BGR image, return a (H, W) binary mask where
    1 = hair, 0 = non-hair.
    """

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    segmented_image = segmenter.segment(mp_image)
    mask = segmented_image.category_mask.numpy_view()

    mask = (mask > 0.5).astype(np.uint8)

    mask = expand_mask(mask)

    return mask


if __name__ == "__main__":
    image = cv2.imread("data/test/IMG_2817.png")
    mask = get_hair_mask(image)

    cv2.imwrite("data/test/2817_mask.png", mask * 255)
