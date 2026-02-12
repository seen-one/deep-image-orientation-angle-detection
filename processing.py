from PIL import Image
import cv2
import numpy as np
from transformers import ViTFeatureExtractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
from utils import rotate_preserve_size
import os
from config import SAVE_IMAGE_DIR
from loguru import logger
import datetime
import re

def preprocess(model_name, image_input):
    """
    image_input: can be a string (path), a numpy array, or a list/tuple of them.
    """
    if model_name in ["vit", "tag-cnn"]:
        image_size = 224
    else:
        image_size = 299

    images = []
    if not isinstance(image_input, (list, tuple)):
        image_input = [image_input]

    for item in image_input:
        if isinstance(item, str):
            img = Image.open(item)
        elif isinstance(item, np.ndarray):
            # Expecting BGR from OpenCV, convert to RGB for PIL
            img = Image.fromarray(cv2.cvtColor(item, cv2.COLOR_BGR2RGB))
        else:
            img = item # Assume it's already a PIL image

        img = img.resize((image_size, image_size))
        images.append(np.array(img))
    
    if model_name == "vit":
        X_vit = feature_extractor(images=images, return_tensors="tf")["pixel_values"]
        X_vit = np.array(X_vit)
        X = X_vit

    elif model_name == "tag-cnn":
        X_vit = feature_extractor(images=images, return_tensors="pt")["pixel_values"]
        X_vit = np.array(X_vit)
        
        # Original code used np.expand_dims(img, axis=0) for single image.
        # For batched tag-cnn, we'd need a stack of images.
        stacked_imgs = np.stack(images, axis=0)
        X = [X_vit, stacked_imgs]

    elif model_name in ["efficientnetv2b2", "en"]:
        X = np.stack(images, axis=0)

    return X


def sanitize_filename(name):
    """Remove invalid characters for Windows filenames."""
    # Replace colon, space, and backslash with underscore
    return re.sub(r'[:\\/*?"<>| ]', '_', name)


def postprocess(img_path, angle, image_size):
    # Rotate image
    img = rotate_preserve_size(img_path, angle, (image_size, image_size), False)

    # Use only the basename and add "pred_" prefix
    base_name = os.path.basename(img_path)
    safe_name = "pred_" + sanitize_filename(base_name)

    output_path = os.path.join(SAVE_IMAGE_DIR, safe_name)

    # If file save fails (very unlikely), add timestamp
    try:
        img.save(output_path)
    except Exception:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = f"{timestamp}_pred_" + sanitize_filename(base_name)
        output_path = os.path.join(SAVE_IMAGE_DIR, safe_name)
        img.save(output_path)

    logger.info(f"Image after orientation angle correction has been saved here: {output_path}")
