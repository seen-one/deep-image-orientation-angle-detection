import os
from flask import Flask, session, request, render_template, flash, redirect, url_for, send_from_directory, jsonify
from infer import Inference
from config import SAVE_IMAGE_DIR, ROOT_DIR
from PIL import Image
from loguru import logger
import datetime
import tempfile
import cv2
import numpy as np
import py360convert
import time
from werkzeug.utils import secure_filename
from urllib.parse import urlparse

app = Flask(__name__)
model = Inference()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_360")
def predict_360_page():
    return render_template("predict_360.html")

@app.route("/viewer_360")
def viewer_360():
    return render_template("viewer_360.html")



@app.route("/login", methods=["GET", "POST"])
def login():
    try:
        session.pop("username")
    except:
        pass

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        if (username == "admin") and (password == "1234"):
            session["username"] = username
            flash("Logged in successfully")
            return render_template("index.html")
            
        else:
            flash("Either username or password is incorrect")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("username")
    flash("Successfully logged out")
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        logger.info("Reading Image")
        file = request.files["file"]

        # Create a safe filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}.jpg"

        # Use OS temp directory instead of /tmp
        input_path = os.path.join(tempfile.gettempdir(), filename)

        logger.info(f"Saving Image to {input_path}")
        file.save(input_path)
        logger.info("Saved Image successfully")
        
        logger.info("Resizing Image")
        img = Image.open(input_path)
        img = img.resize((400, 400))
        img.save(input_path)

        logger.info("Correcting orientation")
        model_name = request.form["model"].lower()
        model.predict(model_name, input_path)

        return render_template("results.html", filename=filename)

    return render_template("predict.html")


@app.route('/input/<path:filename>')
def get_input_images(filename):
    return send_from_directory(SAVE_IMAGE_DIR,
                               filename, as_attachment=True, cache_timeout=0)


@app.route('/pred/<path:filename>')
def get_pred_images(filename):
    return send_from_directory(SAVE_IMAGE_DIR,
                               "pred_"+filename, as_attachment=True, cache_timeout=0)



@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part", "status": "error"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file", "status": "error"}), 400

    # Create a safe filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}.jpg"

    # Use OS temp directory
    input_path = os.path.join(tempfile.gettempdir(), filename)
    file.save(input_path)
    
    # Resize Image - consistent with existing predict logic
    img = Image.open(input_path)
    img = img.resize((400, 400))
    img.save(input_path)

    # Predict angle
    # The existing views code uses 'vit' by default if not specified elsewhere, 
    # but the predict route gets it from form. For API, we'll default to 'vit'.
    model_name = request.form.get("model", "vit").lower()
    
    # Predict without postprocessing (saving corrected image) for API by default
    # but the infer.py predict method handles saving if postprocess_and_save is True.
    # We'll set it to False to just get the angle first.
    angle = model.predict(model_name, input_path, postprocess_and_save=False)

    # Clean up temp file
    try:
        os.remove(input_path)
    except:
        pass

    return jsonify({
        "angle": float(angle),
        "status": "success"
    })

@app.route("/api/predict_360", methods=["POST"])
def api_predict_360():
    if 'file' not in request.files and 'url' not in request.form:
        return jsonify({"error": "No file or URL part", "status": "error"}), 400
    
    # Ensure SAVE_IMAGE_DIR exists
    if not os.path.exists(SAVE_IMAGE_DIR):
        os.makedirs(SAVE_IMAGE_DIR)

    timestamp = int(time.time())
    filename = None
    
    file = request.files.get('file')
    url = request.form.get('url')
    
    if file and file.filename != '':
        filename = f"upload_{timestamp}_{secure_filename(file.filename)}"
        file_path = os.path.join(SAVE_IMAGE_DIR, filename)
        file.save(file_path)
    elif url:
        try:
            import requests
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Extract filename from URL or use a default
            url_path = urlparse(url).path
            base_name = os.path.basename(url_path)
            if not base_name or '.' not in base_name:
                base_name = "url_image.jpg"
            
            filename = f"url_{timestamp}_{secure_filename(base_name)}"
            file_path = os.path.join(SAVE_IMAGE_DIR, filename)
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            logger.error(f"Error downloading image from URL: {str(e)}")
            return jsonify({"error": f"Failed to download image: {str(e)}", "status": "error"}), 400
    
    if not filename:
        return jsonify({"error": "No selected file or valid URL", "status": "error"}), 400

    input_path = os.path.join(SAVE_IMAGE_DIR, filename)
    
    # Rest of the 360 processing logic...
    try:
        # Load image and extract cubemap faces
        img_bgr = cv2.imread(input_path)
        if img_bgr is None:
            return jsonify({"error": "Failed to load image", "status": "error"}), 400
            
        cube_dict = py360convert.e2c(img_bgr, face_w=400, mode='bilinear', cube_format='dict')
        
        # Faces to predict: Front, Back, Left, Right
        faces_to_predict = ['F', 'B', 'L', 'R']
        angles = {}
        model_name = request.form.get("model", "vit").lower()
        
        face_filenames = {}
        for face_key in faces_to_predict:
            face_img = cube_dict[face_key]
            
            # Save face to persistent storage for frontend display
            face_filename = f"face_{face_key}_{timestamp}.jpg"
            face_path = os.path.join(SAVE_IMAGE_DIR, face_filename)
            cv2.imwrite(face_path, face_img)
            
            face_filenames[face_key] = face_filename
            
            # Predict angle
            angle = model.predict(model_name, face_path, postprocess_and_save=False)
            angles[face_key] = float(angle)

        # Calculate Roll and Pitch
        roll = (angles['F'] - angles['B']) / 2.0
        pitch = (angles['L'] - angles['R']) / 2.0
        
        return jsonify({
            "roll": roll,
            "pitch": pitch,
            "face_angles": {
                "front": angles['F'],
                "back": angles['B'],
                "left": angles['L'],
                "right": angles['R']
            },
            "face_filenames": {
                "front": face_filenames['F'],
                "back": face_filenames['B'],
                "left": face_filenames['L'],
                "right": face_filenames['R']
            },
            "filename": filename,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error processing 360 image: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "status": "error"}), 500
    finally:
        # We NO LONGER clean up the original file here because the new tab preview needs it!
        # It will be served via /input/<filename>
        pass

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

