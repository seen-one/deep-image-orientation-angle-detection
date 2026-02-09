import os
import cv2
import numpy as np
import py360convert
from PIL import Image
from infer import Inference
import tempfile
import argparse
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Correct 360 photo orientation using cubemap faces.")
    parser.add_argument("image_path", type=str, help="Path to the equirectangular image.")
    parser.add_argument("--model", type=str, default="vit", help="Model name to use for prediction.")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        logger.error(f"Image not found: {args.image_path}")
        return

    # Load equirectangular image
    logger.info(f"Loading image: {args.image_path}")
    img = cv2.imread(args.image_path)
    if img is None:
        logger.error("Failed to load image.")
        return

    # Convert to cubemap faces
    logger.info("Converting to cubemap faces...")
    # Mode 'bilinear' for better quality/smoothness in detection
    # cube_format='dict' returns faces with keys 'f', 'r', 'b', 'l', 'u', 'd'
    cube_dict = py360convert.e2c(img, face_w=512, mode='bilinear', cube_format='dict')

    # Initialize Inference
    model = Inference()

    angles = {}
    faces_to_predict = ['F', 'R', 'B', 'L']
    
    # Map faces to names for clarity
    face_names = {
        'F': 'Front',
        'R': 'Right',
        'B': 'Back',
        'L': 'Left'
    }

    temp_dir = tempfile.gettempdir()

    for face_key in faces_to_predict:
        face_img = cube_dict[face_key]
        
        # Save face to temp file because Inference.predict takes a path
        face_path = os.path.join(temp_dir, f"temp_face_{face_key}.jpg")
        cv2.imwrite(face_path, face_img)
        
        # Predict angle
        # Note: model.predict returns the angle 'y' which is the predicted tilt.
        # It logs "Predicted angle is: {y} degree"
        angle = model.predict(args.model, face_path, postprocess_and_save=False)
        angles[face_key] = float(angle)
        
        logger.info(f"Face {face_names[face_key]} predicted angle: {angle:.2f} degrees")
        
        # Clean up
        try:
            os.remove(face_path)
        except:
            pass

    # Calculation based on camera tilt:
    # 1. Roll: Rotation around the Z-axis (Forward/Backward axis)
    #    When the camera rolls, Front and Back faces tilt in opposite directions in terms of image rotation.
    #    If camera rolls Right:
    #      Front tilts Right-side-down (Positive angle if clockwise is positive)
    #      Back tilts Left-side-down (viewing the back, the rotation appears inverted relative to front)
    #    So Roll = (Angle_Front - Angle_Back) / 2
    
    # 2. Pitch: Rotation around the X-axis (Left/Right axis)
    #    When the camera pitches (tilts up/down), the side faces (Left/Right) rotate.
    #    If camera tilts Up:
    #      Left face rotates clockwise.
    #      Right face rotates counter-clockwise.
    #    So Pitch = (Angle_Left - Angle_Right) / 2

    angle_f = angles['F']
    angle_r = angles['R']
    angle_b = angles['B']
    angle_l = angles['L']

    roll = (angle_f - angle_b) / 2.0
    pitch = (angle_l - angle_r) / 2.0

    print("\n" + "="*40)
    print("360 ORIENTATION PREDICTION RESULTS")
    print("="*40)
    print(f"Front Face Angle: {angle_f:8.2f}°")
    print(f"Back Face Angle:  {angle_b:8.2f}°")
    print(f"Left Face Angle:  {angle_l:8.2f}°")
    print(f"Right Face Angle: {angle_r:8.2f}°")
    print("-" * 40)
    print(f"CALCULATED ROLL:  {roll:8.2f}°")
    print(f"CALCULATED PITCH: {pitch:8.2f}°")
    print("="*40)
    print("\nApply these values as-is (or inverted depending on your viewer's convention)")
    print("to correct the horizon of your 360 photo.")

if __name__ == "__main__":
    main()
