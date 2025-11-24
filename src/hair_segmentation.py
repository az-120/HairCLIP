"""Hair segmentation utilities built around a pretrained U2Net model."""

import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

MODEL_PATH = "models/u2netp.pth"

def _load_u2net_model():
    """Load the lightweight U2NetP checkpoint for hair segmentation.

    Returns:
        torch.nn.Module: Initialized U2NetP model in eval mode on CPU.
    """
    
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(ROOT)

    from models.u2net_model import U2NETP

    model = U2NETP(3, 1)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model


u2net = _load_u2net_model()


_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def get_hair_mask(image_bgr):
    """Predict a binary hair mask from an aligned BGR image.

    Args:
        image_bgr (np.ndarray): OpenCV image in BGR format with shape (H, W, 3).

    Returns:
        np.ndarray: Binary hair mask of shape (H, W) with values 0 or 1.
    """

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)

    inp = _transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        pred = u2net(inp)[0][0][0]

    pred = pred.squeeze().cpu().numpy()  
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    H, W = image_bgr.shape[:2]
    mask = cv2.resize(pred, (W, H))

    mask = (mask > 0.5).astype(np.uint8)

    return mask

if __name__ == "__main__":
    image = cv2.imread("data/test/2817_aligned.png")
    mask = get_hair_mask(image)

    cv2.imwrite("data/test/2817_mask.png", mask * 255)
