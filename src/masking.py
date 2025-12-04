import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "selfie_multiclass.tflite"

BaseOptions = python.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.IMAGE,
    output_category_mask=True,
)

segmenter = vision.ImageSegmenter.create_from_options(options)


def get_multiclass_mask(image_bgr):
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
    )
    result = segmenter.segment(mp_image)
    return result.category_mask.numpy_view()  # shape (H, W) integers 0–5

def expand_downward(mask, pixels):
    h, w = mask.shape
    expanded = mask.copy()

    # Expand mask downward by ORing pixel below
    for i in range(1, pixels):
        shifted = np.roll(mask, shift=i, axis=0)
        shifted[:i, :] = 0
        expanded = np.logical_or(expanded, shifted)

    return expanded.astype(np.uint8)

def get_editable_mask(image_bgr, expand_px=200):
    cls_mask = get_multiclass_mask(image_bgr)

    hair = (cls_mask == 1).astype(np.uint8)
    face = (cls_mask == 3).astype(np.uint8)
    body = (cls_mask == 2).astype(np.uint8)

    # Expand the hair mask for haircuts that grow hair
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px, expand_px))
    dilated = cv2.dilate(hair, kernel, iterations=1)

    # Expand downward because above don't really cover longer longer hair
    downward = expand_downward(hair, 1500)

    expanded = np.logical_or(dilated, downward).astype(np.uint8)

    # Slightly erode face mask to account for hair falling on face
    face_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
    shrunken_face = cv2.erode(face, face_kernel, iterations=1)

    # Subtract out the face & body
    protected = (shrunken_face | body).astype(np.uint8)
    expanded[protected == 1] = 0

    return expanded.astype(np.uint8)


# Test
if __name__ == "__main__":
    img = cv2.imread("data/test/headshotclip.jpg")
    mask = get_editable_mask(img)

    cv2.imwrite("data/test/multimask.png", mask * 255)
