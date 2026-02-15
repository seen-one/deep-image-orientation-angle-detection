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
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
from urllib.parse import urlparse
import random
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Tell Flask it is behind a proxy
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1
)

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

image_cache = ImageCache(max_size=300)

# Cache for Panoramax IDs
panoramax_ids = []
def load_panoramax_ids():
    global panoramax_ids
    if not panoramax_ids:
        ids_path = os.path.join(ROOT_DIR, "random_panoramax_bike_ids.txt")
        if os.path.exists(ids_path):
            with open(ids_path, 'r') as f:
                panoramax_ids = [line.strip() for line in f if line.strip()]
    return panoramax_ids

@app.route("/api/random_id")
def get_random_id():
    ids = load_panoramax_ids()
    if ids:
        return jsonify({"id": random.choice(ids)})
    return jsonify({"error": "No IDs found"}), 404

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

        model_name = request.form.get("model", "vit").lower()
        fov_deg = float(request.form.get("fov", 60))
        inlier_threshold_deg = float(request.form.get("inlier_threshold_deg", 2.0))
        sample_count = int(request.form.get("sample_count", 36))

        if sample_count < 4:
            return jsonify({"error": "sample_count should be >= 4", "status": "error"}), 400

        sample_yaws = np.linspace(0, 360, sample_count, endpoint=False, dtype=np.float32)

        def extract_sample(idx, yaw_deg):
            sample_img = py360convert.e2p(
                img_bgr,
                fov_deg=fov_deg,
                u_deg=float(yaw_deg),
                v_deg=0.0,
                in_rot_deg=0.0,
                out_hw=(400, 400),
                mode='bilinear'
            )
            return idx, sample_img

        max_workers = min(8, sample_count)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(extract_sample, i, y) for i, y in enumerate(sample_yaws)]
            sample_images = [None] * sample_count
            for future in futures:
                idx, sample_img = future.result()
                sample_images[idx] = sample_img

        predicted_rolls = np.array(model.predict_batch(model_name, sample_images), dtype=np.float32)
        yaw_rads = np.deg2rad(sample_yaws)
        roll_rads = np.deg2rad(predicted_rolls)

        def normalize_rad(rad):
            return (rad + np.pi) % (2 * np.pi) - np.pi

        def angular_diff_deg(a_deg, b_deg):
            return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)

        def model_roll_deg(alpha_rad, beta_rad, yaw_rad_values):
            # f(alpha_hat; alpha, beta) = atan(tan(beta) * cos(alpha_hat - alpha))
            return np.rad2deg(np.arctan(np.tan(beta_rad) * np.cos(yaw_rad_values - alpha_rad)))

        def evaluate_hypothesis(alpha_rad, beta_rad):
            modeled = model_roll_deg(alpha_rad, beta_rad, yaw_rads)
            errors = np.array(
                [angular_diff_deg(float(p), float(m)) for p, m in zip(predicted_rolls, modeled)],
                dtype=np.float32
            )
            inliers = errors <= inlier_threshold_deg
            inlier_count = int(np.sum(inliers))
            mae = float(np.mean(errors[inliers])) if inlier_count > 0 else float("inf")
            rmse = float(np.sqrt(np.mean(np.square(errors[inliers])))) if inlier_count > 0 else float("inf")
            return {
                "alpha_rad": float(normalize_rad(alpha_rad)),
                "beta_rad": float(beta_rad),
                "modeled_rolls": modeled,
                "errors": errors,
                "inliers": inliers,
                "inlier_count": inlier_count,
                "mae": mae,
                "rmse": rmse,
            }

        # Exhaustive hypothesis generation from all point pairs.
        best = None
        tiny = 1e-6
        tan_rolls = np.tan(roll_rads)
        for i in range(sample_count):
            for j in range(i + 1, sample_count):
                ai = tan_rolls[i]
                aj = tan_rolls[j]
                theta_i = yaw_rads[i]
                theta_j = yaw_rads[j]

                p = ai * np.cos(theta_j) - aj * np.cos(theta_i)
                q = ai * np.sin(theta_j) - aj * np.sin(theta_i)
                if abs(p) < tiny and abs(q) < tiny:
                    continue

                alpha_candidate_1 = np.arctan2(-p, q)
                for alpha_candidate in (alpha_candidate_1, alpha_candidate_1 + np.pi):
                    cos_term = np.cos(theta_i - alpha_candidate)
                    if abs(cos_term) < tiny:
                        continue
                    tan_beta = ai / cos_term
                    beta_candidate = np.arctan(tan_beta)

                    hypothesis = evaluate_hypothesis(alpha_candidate, beta_candidate)
                    if best is None:
                        best = hypothesis
                        continue
                    if hypothesis["inlier_count"] > best["inlier_count"]:
                        best = hypothesis
                    elif hypothesis["inlier_count"] == best["inlier_count"] and hypothesis["mae"] < best["mae"]:
                        best = hypothesis

        if best is None:
            return jsonify({"error": "Unable to estimate a stable hypothesis", "status": "error"}), 500

        # Least-squares-like local refinement around best hypothesis on inlier set.
        inlier_indices = np.where(best["inliers"])[0]
        if len(inlier_indices) >= 2:
            def objective(alpha_rad, beta_rad):
                modeled = model_roll_deg(alpha_rad, beta_rad, yaw_rads[inlier_indices])
                errors = np.array(
                    [angular_diff_deg(float(predicted_rolls[idx]), float(modeled[k])) for k, idx in enumerate(inlier_indices)],
                    dtype=np.float32
                )
                return float(np.mean(np.square(errors)))

            alpha_center = best["alpha_rad"]
            beta_center = best["beta_rad"]

            alpha_grid_1 = np.deg2rad(np.arange(-6.0, 6.0 + 0.001, 0.5))
            beta_grid_1 = np.deg2rad(np.arange(-6.0, 6.0 + 0.001, 0.5))
            best_obj = objective(alpha_center, beta_center)

            for da in alpha_grid_1:
                for db in beta_grid_1:
                    alpha_try = normalize_rad(alpha_center + da)
                    beta_try = np.clip(beta_center + db, np.deg2rad(-89.0), np.deg2rad(89.0))
                    obj = objective(alpha_try, beta_try)
                    if obj < best_obj:
                        best_obj = obj
                        alpha_center = alpha_try
                        beta_center = beta_try

            alpha_grid_2 = np.deg2rad(np.arange(-1.0, 1.0 + 0.001, 0.1))
            beta_grid_2 = np.deg2rad(np.arange(-1.0, 1.0 + 0.001, 0.1))
            for da in alpha_grid_2:
                for db in beta_grid_2:
                    alpha_try = normalize_rad(alpha_center + da)
                    beta_try = np.clip(beta_center + db, np.deg2rad(-89.0), np.deg2rad(89.0))
                    obj = objective(alpha_try, beta_try)
                    if obj < best_obj:
                        best_obj = obj
                        alpha_center = alpha_try
                        beta_center = beta_try

            best = evaluate_hypothesis(alpha_center, beta_center)

        alpha_deg = float(np.rad2deg(best["alpha_rad"]))
        beta_deg = float(np.rad2deg(best["beta_rad"]))

        # Convert (alpha, beta) to viewer-compatible pseudo roll/pitch.
        tan_beta = np.tan(best["beta_rad"])
        roll_deg = float(np.rad2deg(np.arctan(tan_beta * np.cos(best["alpha_rad"]))))
        pitch_deg = float(np.rad2deg(-np.arctan(tan_beta * np.sin(best["alpha_rad"]))))

        sample_entries = []
        modeled_rolls = best["modeled_rolls"]
        for idx in range(sample_count):
            _, buffer = cv2.imencode(".jpg", sample_images[idx])
            sample_image_id = str(uuid.uuid4())
            image_cache.set(sample_image_id, buffer.tobytes())
            sample_entries.append({
                "index": idx,
                "yaw_deg": float(sample_yaws[idx]),
                "predicted_roll_deg": float(predicted_rolls[idx]),
                "modeled_roll_deg": float(modeled_rolls[idx]),
                "error_deg": float(best["errors"][idx]),
                "inlier": bool(best["inliers"][idx]),
                "image_id": sample_image_id,
            })

        return jsonify({
            "status": "success",
            "main_id": main_image_id,
            "sample_count": sample_count,
            "fov_deg": fov_deg,
            "inlier_threshold_deg": inlier_threshold_deg,
            "alpha_deg": alpha_deg,
            "beta_deg": beta_deg,
            "roll": roll_deg,
            "pitch": pitch_deg,
            "inlier_count": int(best["inlier_count"]),
            "inlier_ratio": float(best["inlier_count"] / sample_count),
            "mae_inlier_deg": float(best["mae"]),
            "rmse_inlier_deg": float(best["rmse"]),
            "samples": sample_entries
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

