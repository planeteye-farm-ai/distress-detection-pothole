from flask import Flask, request, jsonify, render_template, send_file, abort
from flask_socketio import SocketIO
from flask_cors import CORS
from PIL import Image
import io, os, sqlite3, logging
from datetime import datetime
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import folium
from fpdf import FPDF

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
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024  # 16 MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------------
# SAM Model
# ------------------------
predictor = None
sam_loaded = False

def init_sam():
    global predictor, sam_loaded
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        checkpoint_name = "sam_vit_b_01ec64.pth"
        checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_name)

        # If model is not in the repo, download it as a fallback.
        if not os.path.exists(checkpoint_path):
            logger.warning(f"SAM checkpoint '{checkpoint_name}' not found. Downloading as a fallback...")
            model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            torch.hub.download_url_to_file(model_url, checkpoint_path)
            logger.info("SAM checkpoint downloaded successfully.")

        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam.to(device)
        predictor = SamPredictor(sam)
        sam_loaded = True
        logger.info("SAM loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"SAM init error: {str(e)}")
        return False

# ------------------------
# Database
# ------------------------
def init_db():
    conn = sqlite3.connect(app.config['DATABASE'])
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
    logger.info("Database initialized")

# ------------------------
# Utility
# ------------------------
def estimate_area(area_pixels):
    pixels_per_meter = 100  # adjust for real calibration
    return area_pixels / (pixels_per_meter**2)

def estimate_depth(area_m2):
    # Rough depth estimation: small area -> shallow, large area -> deeper
    # Example scaling: 0.05 m minimum, +0.2 m for large potholes
    return 0.05 + min(area_m2 * 0.5, 0.5)

def determine_severity(area_m2):
    if area_m2 < 0.1: return 'low'
    if area_m2 < 0.3: return 'medium'
    return 'high'

def overlay_image(image_np, mask):
    overlay = image_np.copy()
    overlay[mask>0] = [255,0,0]
    return overlay

# ------------------------
# Routes
# ------------------------
@app.route('/')
def index():
    return render_template('index1.html', sam_loaded=sam_loaded)

@app.route('/health')
def health_check():
    """Provides a health check endpoint for the frontend to poll."""
    return jsonify({
        'status': 'ok',
        'sam_loaded': sam_loaded
    })

@app.route('/detect', methods=['POST'])
def detect_pothole():
    if not sam_loaded:
        return jsonify({'error': 'SAM not loaded'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    latitude = float(request.form.get('latitude', 0.0))
    longitude = float(request.form.get('longitude', 0.0))

    image = Image.open(image_file.stream).convert('RGB')
    image_np = np.array(image)

    predictor.set_image(image_np)
    h,w = image_np.shape[:2]
    input_point = np.array([[w//2,h//2]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    if len(masks)==0 or masks[0].size==0:
        return jsonify({'success': False})

    mask = masks[0]
    confidence = float(scores[0])
    area_pixels = np.sum(mask)
    area_m2 = estimate_area(area_pixels)
    severity = determine_severity(area_m2)
    depth_meters = estimate_depth(area_m2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pothole_{timestamp}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    overlay = overlay_image(image_np, mask)
    Image.fromarray(overlay).save(filepath)

    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute('''
        INSERT INTO potholes (latitude, longitude, severity, area, depth_meters, image_path, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (latitude, longitude, severity, area_m2, depth_meters, filepath, confidence))
    pothole_id = c.lastrowid
    conn.commit()
    conn.close()

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

@app.route('/export/<int:pothole_id>')
def export_pdf(pothole_id):
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute('SELECT * FROM potholes WHERE id=?', (pothole_id,))
    row = c.fetchone()
    conn.close()
    if not row: return abort(404)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Pothole Report #{row[0]}", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0, 8, f"Latitude: {row[1]}", ln=True)
    pdf.cell(0, 8, f"Longitude: {row[2]}", ln=True)
    pdf.cell(0, 8, f"Severity: {row[3]}", ln=True)
    pdf.cell(0, 8, f"Area: {row[4]:.2f} m²", ln=True)
    pdf.cell(0, 8, f"Depth: {row[5]:.2f} m", ln=True)
    pdf.cell(0, 8, f"Confidence: {row[7]*100:.1f}%", ln=True)
    pdf.cell(0, 8, f"Timestamp: {row[8]}", ln=True)
    pdf.ln(5)
    if os.path.exists(row[6]):
        pdf.image(row[6], w=150)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"pothole_report_{row[0]}.pdf")
    pdf.output(pdf_path)
    return send_file(pdf_path)

@app.route('/map')
def show_map():
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute('SELECT latitude, longitude, severity, id FROM potholes')
    rows = c.fetchall()
    conn.close()
    center = (rows[0][0], rows[0][1]) if rows else (40.7128, -74.0060)
    m = folium.Map(
        location=center, zoom_start=13,
        tiles='http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='© Google'
    )
    for lat, lon, severity, pid in rows:
        color = 'red' if severity=='high' else 'orange' if severity=='medium' else 'green'
        folium.Marker([lat, lon], popup=f"Pothole #{pid}\nSeverity: {severity}", icon=folium.Icon(color=color)).add_to(m)
    return m._repr_html_()

# ------------------------
# Main
# ------------------------
def initialize_app():
    init_sam()
    init_db()
    logger.info("App initialized")

if __name__ == "__main__":
    initialize_app()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
