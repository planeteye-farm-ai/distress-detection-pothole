from flask import Flask, request, jsonify, render_template, send_file, abort
from flask_socketio import SocketIO
from flask_cors import CORS
from PIL import Image
import io, os, sqlite3, logging, requests
from datetime import datetime
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import folium
from fpdf import FPDF
import eventlet
eventlet.monkey_patch()

# ------------------------
# Logging configuration
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------
# Flask setup
# ------------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATABASE'] = 'potholes.db'
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------------
# Model setup
# ------------------------
HF_MODEL_URL = "https://huggingface.co/AkhileshYR/sam-vit-b-model/resolve/main/sam_vit_b_01ec64.pth"
MODEL_DIR = os.environ.get("MODEL_DIR", ".")
MODEL_NAME = "sam_vit_b_01ec64.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

predictor = None
sam_loaded = False


def download_model():
    """Download SAM model from Hugging Face if missing"""
    if not os.path.exists(MODEL_PATH):
        logger.info(f"Downloading SAM model to {MODEL_PATH}...")
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            with requests.get(HF_MODEL_URL, stream=True, timeout=120) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                block_size = 8192
                with open(MODEL_PATH, "wb") as f:
                    for data in r.iter_content(block_size):
                        f.write(data)
            logger.info("✅ SAM model downloaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to download SAM model: {e}")


def init_sam():
    """Initialize SAM model from local disk only."""
    global predictor, sam_loaded
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        if not os.path.exists(MODEL_PATH):
            logger.warning(f"SAM model not found at {MODEL_PATH}. Waiting for manual upload.")
            return False

        sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)
        sam.to(device)
        predictor = SamPredictor(sam)
        sam_loaded = True
        logger.info("✅ SAM model loaded successfully from disk!")
        return True
    except Exception as e:
        logger.error(f"SAM initialization error: {str(e)}")
        return False


# ------------------------
# Database setup
# ------------------------
def init_db():
    """Initialize pothole database if missing"""
    db_path = app.config['DATABASE']
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS potholes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL,
            longitude REAL,
            severity TEXT,
            area REAL,
            depth_meters REAL,
            image_path TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'reported'
        )
    ''')
    conn.commit()
    conn.close()
    logger.info(f"✅ SQLite database initialized successfully at {db_path}")


# ------------------------
# Utility functions
# ------------------------
def estimate_area(area_pixels):
    pixels_per_meter = 100
    return area_pixels / (pixels_per_meter ** 2)


def estimate_depth(area_m2):
    return 0.05 + min(area_m2 * 0.5, 0.5)


def determine_severity(area_m2):
    if area_m2 < 0.1:
        return 'low'
    elif area_m2 < 0.3:
        return 'medium'
    else:
        return 'high'


def overlay_image(image_np, mask):
    overlay = image_np.copy()
    overlay[mask > 0] = [255, 0, 0]
    return overlay


def safe_float(value):
    try:
        if value in (None, '', 'null', 'undefined'):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ------------------------
# Routes
# ------------------------
@app.route('/')
def index():
    return render_template('index1.html', sam_loaded=sam_loaded)


@app.route('/health')
def health():
    return jsonify({"status": "ok"}), 200


@app.route('/detect', methods=['POST'])
def detect_pothole():
    if not sam_loaded:
        return jsonify({'error': 'SAM model is not available or failed to load.'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    image_file = request.files.get('image')
    if not image_file or image_file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400

    try:
        latitude = safe_float(request.form.get('latitude', 0.0))
        longitude = safe_float(request.form.get('longitude', 0.0))

        image = Image.open(image_file.stream).convert('RGB')
        image_np = np.array(image)

        # SAM prediction
        predictor.set_image(image_np)
        h, w, _ = image_np.shape
        input_point = np.array([[w / 2, h / 2]])
        input_label = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        if not masks.any():
            return jsonify({'success': False, 'error': 'No defects were found in the image.'})

        # Process results
        mask = masks[0]
        confidence = float(scores[0])
        area_pixels = np.sum(mask)
        area_m2 = estimate_area(area_pixels)
        severity = determine_severity(area_m2)
        depth_meters = estimate_depth(area_m2)

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pothole_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        overlay = overlay_image(image_np, mask)
        Image.fromarray(overlay).save(filepath)

        # Save to DB
        with sqlite3.connect(app.config['DATABASE']) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO potholes (latitude, longitude, severity, area, depth_meters, image_path, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (latitude, longitude, severity, area_m2, depth_meters, filepath, confidence))
            pothole_id = c.lastrowid
            conn.commit()

        # Notify clients
        socketio.emit('new_pothole', {
            'id': pothole_id,
            'latitude': latitude,
            'longitude': longitude,
            'severity': severity,
            'area': area_m2,
            'depth_meters': depth_meters,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })

        return jsonify({
            'success': True,
            'pothole_id': pothole_id,
            'severity': severity,
            'area_m2': area_m2,
            'depth_meters': depth_meters,
            'confidence': confidence,
            'image_url': f'/image/{filename}'
        })
    except Exception as e:
        logger.error(f"An error occurred during pothole detection: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500


@app.route('/potholes')
def get_potholes():
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute('SELECT * FROM potholes ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            'id': r[0],
            'latitude': r[1],
            'longitude': r[2],
            'severity': r[3],
            'area': r[4],
            'depth_meters': r[5],
            'image_path': r[6],
            'confidence': r[7],
            'timestamp': r[8],
            'status': r[9]
        })
    return jsonify(result)


@app.route('/image/<filename>')
def get_image(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path)
    return abort(404)


@app.route('/upload_model', methods=['POST'])
def upload_model():
    """Upload SAM checkpoint to /data/models and reload it immediately."""
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        model_dir = os.environ.get("MODEL_DIR", "/data/models")
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, "sam_vit_b_01ec64.pth")

        file.save(save_path)
        size = os.path.getsize(save_path)

        logger.info(f"✅ Model uploaded to {save_path} ({size/1e6:.2f} MB)")

        # Reload SAM immediately
        global predictor
        init_sam()
        logger.info("✅ SAM reloaded successfully after upload.")

        return jsonify({
            'success': True,
            'path': save_path,
            'size_bytes': size,
            'message': 'Model uploaded successfully and reload attempted.'
        }), 200

    except Exception as e:
        logger.error(f"❌ Model upload failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ------------------------
# Initialization
# ------------------------
def initialize_app():
    """Initialize database and SAM model."""
    logger.info("🚀 Starting application initialization...")
    init_db()
    init_sam()
    if sam_loaded:
        logger.info("✅ SAM model is ready to use.")
    else:
        logger.warning("⚠️ SAM model not found. Please upload via /upload_model.")
    logger.info("✅ Application fully initialized and ready.")


# Initialize for Gunicorn
initialize_app()

if __name__ == "__main__":
    logger.info("Running Flask app in debug mode for local development.")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
