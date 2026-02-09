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
import io
import uuid
from collections import OrderedDict
from flask import send_file
from werkzeug.utils import secure_filename
from urllib.parse import urlparse

app = Flask(__name__)
model = Inference()

@app.route("/")
@app.route("/predict_360")
def predict_360_page():
    return render_template("predict_360.html")

@app.route("/viewer_360")
def viewer_360():
    return render_template("viewer_360.html")



# In-memory image cache
class ImageCache:
    def __init__(self, max_size=50):
        self.cache = OrderedDict()
        self.max_size = max_size

    def set(self, key, data, content_type="image/jpeg"):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = {"data": data, "content_type": content_type}
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

image_cache = ImageCache()

@app.route("/mem_image/<image_id>")
def get_mem_image(image_id):
    item = image_cache.get(image_id)
    if item:
        return send_file(
            io.BytesIO(item["data"]),
            mimetype=item["content_type"]
        )
    return "Image not found", 404

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
    
    image_bytes = None
    filename = "image.jpg"
    
    file = request.files.get('file')
    url = request.form.get('url')
    
    if file and file.filename != '':
        image_bytes = file.read()
        filename = secure_filename(file.filename)
    elif url:
        try:
            import requests
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            image_bytes = response.content
            url_path = urlparse(url).path
            filename = os.path.basename(url_path) or "url_image.jpg"
        except Exception as e:
            logger.error(f"Error downloading image from URL: {str(e)}")
            return jsonify({"error": f"Failed to download image: {str(e)}", "status": "error"}), 400
    
    if not image_bytes:
        return jsonify({"error": "No selected file or valid URL", "status": "error"}), 400

    # Store original image in memory
    main_image_id = str(uuid.uuid4())
    image_cache.set(main_image_id, image_bytes)

    try:
        # Load image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return jsonify({"error": "Failed to decode image", "status": "error"}), 400
            
        # Extract 4 patches (Front, Back, Left, Right) with 120 FOV
        # e2p(e_img, fov_h, fov_v, u_deg, v_deg, out_hw, mode='bilinear')
        # u_deg: longitude (-180 to 180), v_deg: latitude (-90 to 90)
        face_configs = {
            'F': (0, 0),    # Front
            'R': (90, 0),   # Right
            'B': (180, 0),  # Back
            'L': (-90, 0)   # Left
        }
        
        faces_to_predict = ['F', 'B', 'L', 'R']
        model_name = request.form.get("model", "vit").lower()
        
        # --- PASS 1: Crude Estimation ---
        angles_p1 = {}
        face_ids_p1 = {}
        for face_key in faces_to_predict:
            u_deg, v_deg = face_configs[face_key]
            face_img = py360convert.e2p(img_bgr, fov_deg=90, u_deg=u_deg, v_deg=v_deg, out_hw=(400, 400), mode='bilinear')
            
            # Save Pass 1 faces to cache for comparison
            _, buffer = cv2.imencode(".jpg", face_img)
            face_id = str(uuid.uuid4())
            image_cache.set(face_id, buffer.tobytes())
            face_ids_p1[face_key] = face_id
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(buffer.tobytes())
                tmp_path = tmp.name
            
            try:
                angle = model.predict(model_name, tmp_path, postprocess_and_save=False)
                angles_p1[face_key] = float(angle)
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)

        roll_p1 = (angles_p1['F'] - angles_p1['B']) / 2.0
        pitch_p1 = (angles_p1['L'] - angles_p1['R']) / 2.0

        # --- PASS 2: Refined Estimation ---
        refinement_configs = {
            'F': (0, pitch_p1, -roll_p1),
            'B': (180, -pitch_p1, roll_p1),
            'L': (-90, -roll_p1, -pitch_p1),
            'R': (90, roll_p1, pitch_p1)
        }

        angles_p2 = {}
        face_ids_p2 = {}
        for face_key in faces_to_predict:
            u_deg, v_deg, in_rot = refinement_configs[face_key]
            face_img = py360convert.e2p(img_bgr, fov_deg=60, u_deg=u_deg, v_deg=v_deg, 
                                        in_rot_deg=in_rot, out_hw=(400, 400), mode='bilinear')
            
            _, buffer = cv2.imencode(".jpg", face_img)
            face_id = str(uuid.uuid4())
            image_cache.set(face_id, buffer.tobytes())
            face_ids_p2[face_key] = face_id
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(buffer.tobytes())
                tmp_path = tmp.name
            
            try:
                angle = model.predict(model_name, tmp_path, postprocess_and_save=False)
                angles_p2[face_key] = float(angle)
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)

        roll_p2 = (angles_p2['F'] - angles_p2['B']) / 2.0
        pitch_p2 = (angles_p2['L'] - angles_p2['R']) / 2.0

        final_roll = roll_p1 + roll_p2
        final_pitch = pitch_p1 + pitch_p2
        
        return jsonify({
            "roll": final_roll,
            "pitch": final_pitch,
            "pass1": {
                "roll": roll_p1,
                "pitch": pitch_p1,
                "face_angles": angles_p1,
                "face_ids": face_ids_p1
            },
            "pass2_residual": {
                "roll": roll_p2,
                "pitch": pitch_p2,
                "face_angles": angles_p2,
                "face_ids": face_ids_p2
            },
            "main_id": main_image_id,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error processing 360 image: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "status": "error"}), 500
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

