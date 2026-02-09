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

    # Extract 4 patches (Front, Back, Left, Right) with 120 FOV
    # e2p(e_img, fov_h, fov_v, u_deg, v_deg, out_hw, mode='bilinear')
    face_configs = {
        'F': (0, 0),    # Front
        'R': (90, 0),   # Right
        'B': (180, 0),  # Back
        'L': (-90, 0)   # Left
    }
    
    faces_to_predict = ['F', 'R', 'B', 'L']
    
    # Map faces to names for clarity
    face_names = {
        'F': 'Front',
        'R': 'Right',
        'B': 'Back',
        'L': 'Left'
    }

    # Initialize Inference
    model = Inference()
    angles = {}
    # --- PASS 1: Crude Estimation ---
    logger.info("Pass 1: Crude estimation...")
    angles_p1 = {}
    for face_key in faces_to_predict:
        u_deg, v_deg = face_configs[face_key]
        # Standard extraction (no rotation/tilt)
        face_img = py360convert.e2p(img, fov_deg=90, u_deg=u_deg, v_deg=v_deg, out_hw=(400, 400), mode='bilinear')
        
        face_path = os.path.join(temp_dir, f"temp_face_{face_key}_p1.jpg")
        cv2.imwrite(face_path, face_img)
        
        try:
            angle = model.predict(args.model, face_path, postprocess_and_save=False)
            angles_p1[face_key] = float(angle)
        finally:
            if os.path.exists(face_path): os.remove(face_path)

    roll_p1 = (angles_p1['F'] - angles_p1['B']) / 2.0
    pitch_p1 = (angles_p1['L'] - angles_p1['R']) / 2.0

    logger.info(f"Pass 1 result - Roll: {roll_p1:.2f}°, Pitch: {pitch_p1:.2f}°")

    # --- PASS 2: Refined Estimation ---
    logger.info("Pass 2: Refined estimation...")
    refinement_configs = {
        'F': (0, pitch_p1, -roll_p1),
        'B': (180, -pitch_p1, roll_p1),
        'L': (-90, -roll_p1, -pitch_p1),
        'R': (90, roll_p1, pitch_p1)
    }

    angles_p2 = {}
    for face_key in faces_to_predict:
        u_deg, v_deg, in_rot = refinement_configs[face_key]
        face_img = py360convert.e2p(img, fov_deg=90, u_deg=u_deg, v_deg=v_deg, 
                                    in_rot_deg=in_rot, out_hw=(400, 400), mode='bilinear')
        
        face_path = os.path.join(temp_dir, f"temp_face_{face_key}_p2.jpg")
        cv2.imwrite(face_path, face_img)
        
        try:
            angle = model.predict(args.model, face_path, postprocess_and_save=False)
            angles_p2[face_key] = float(angle)
            logger.info(f"Face {face_names[face_key]} residual angle: {angle:.2f}°")
        finally:
            if os.path.exists(face_path): os.remove(face_path)

    # Residual correction from Pass 2
    roll_p2 = (angles_p2['F'] - angles_p2['B']) / 2.0
    pitch_p2 = (angles_p2['L'] - angles_p2['R']) / 2.0

    # Final Combined Result
    final_roll = roll_p1 + roll_p2
    final_pitch = pitch_p1 + pitch_p2

    print("\n" + "="*40)
    print("360 ORIENTATION PREDICTION RESULTS (2-PASS)")
    print("="*40)
    print(f"Final Calculated Roll:  {final_roll:8.2f}°")
    print(f"Final Calculated Pitch: {final_pitch:8.2f}°")
    print("-" * 40)
    print(f"Pass 1 Roll:  {roll_p1:8.2f}° | Pass 2 Residual: {roll_p2:8.2f}°")
    print(f"Pass 1 Pitch: {pitch_p1:8.2f}° | Pass 2 Residual: {pitch_p2:8.2f}°")
    print("="*40)
    print("\nApply these values as-is (or inverted depending on your viewer's convention)")
    print("to correct the horizon of your 360 photo.")

if __name__ == "__main__":
    main()
