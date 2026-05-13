# ============================================================
#   AERIAL VEHICLE DETECTION USING SAHI + EfficientDet ONNX
#   Full Source Code — All Modules Combined
# ============================================================

# ────────────────────────────────────────────────────────────
# FILE: src/utils.py
# Helper functions — annotation parsing, drawing, filtering
# ────────────────────────────────────────────────────────────

import cv2
import numpy as np
import os

# Vehicle classes from COCO (used for SAHI filtering)
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']

# Vehicle classes from VisDrone annotations
VISDRONE_VEHICLE_CLASSES = {4: 'car', 5: 'van', 6: 'truck', 9: 'bus'}
VISDRONE_COLORS = {4: (0,255,0), 5: (0,165,255), 6: (0,0,255), 9: (255,0,0)}

def load_visdrone_annotation(annotation_path):
    """Load VisDrone annotation file and return vehicle bounding boxes."""
    boxes = []
    with open(annotation_path, 'r') as f:
        for line in f.readlines():
            values = line.strip().split(',')
            if len(values) < 6:
                continue
            x, y, w, h = int(values[0]), int(values[1]), int(values[2]), int(values[3])
            score = int(values[4])
            category = int(values[5])
            if score == 0 or category not in VISDRONE_VEHICLE_CLASSES:
                continue
            boxes.append((x, y, w, h, category))
    return boxes

def draw_visdrone_boxes(image, boxes):
    """Draw ground truth VisDrone bounding boxes on image."""
    for (x, y, w, h, category) in boxes:
        label = VISDRONE_VEHICLE_CLASSES.get(category, 'unknown')
        color = VISDRONE_COLORS.get(category, (0, 255, 0))
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, label, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def filter_vehicles(detections):
    """Filter SAHI detections to keep only vehicle classes."""
    return [d for d in detections if d.category.name in VEHICLE_CLASSES]

def draw_vehicle_boxes(image, detections):
    """Draw SAHI vehicle detection boxes on image."""
    for d in detections:
        box = d.bbox
        x1, y1, x2, y2 = int(box.minx), int(box.miny), int(box.maxx), int(box.maxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(image, d.category.name, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    count_text = f'Vehicles: {len(detections)}'
    cv2.putText(image, count_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
    return image

def side_by_side(img1, img2, label1, label2):
    """Combine two images side by side with labels."""
    h = max(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
    img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))
    cv2.putText(img1, label1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(img2, label2, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
    return np.hstack((img1, img2))


# ────────────────────────────────────────────────────────────
# FILE: src/visualize.py
# Load VisDrone image + annotation and save ground truth image
# ────────────────────────────────────────────────────────────

def run_visualize():
    image_folder = 'data/images'
    annotation_folder = 'data/annotations'

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    image_file = image_files[0]
    image_path = os.path.join(image_folder, image_file)
    annotation_path = os.path.join(annotation_folder, image_file.replace('.jpg', '.txt'))

    if not os.path.exists(annotation_path):
        print('Annotation file not found!')
        return

    image = cv2.imread(image_path)
    if image is None:
        print('Failed to load image')
        return

    boxes = load_visdrone_annotation(annotation_path)
    image = draw_visdrone_boxes(image, boxes)

    output_path = 'outputs/annotated.jpg'
    cv2.imwrite(output_path, image)
    print(f'Saved to {output_path}')
    print(f'Total boxes drawn: {len(boxes)}')


# ────────────────────────────────────────────────────────────
# FILE: src/inference.py
# Baseline single-pass ONNX detection (no SAHI)
# ────────────────────────────────────────────────────────────

import onnxruntime as ort

def preprocess(image, size=640):
    """Preprocess image for ONNX model input."""
    img = cv2.resize(image, (size, size))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def postprocess(outputs, orig_shape, threshold=0.3):
    """Parse ONNX model output and extract bounding boxes."""
    boxes = []
    h, w = orig_shape[:2]
    for pred in outputs[0][0].T:
        x, y, bw, bh = pred[:4]
        score = float(np.max(pred[4:]))
        if score < threshold:
            continue
        x1 = int((x - bw/2) * w / 640)
        y1 = int((y - bh/2) * h / 640)
        x2 = int((x + bw/2) * w / 640)
        y2 = int((y + bh/2) * h / 640)
        boxes.append((x1, y1, x2, y2, score))
    return boxes

def run_baseline(image_path, model_path='models/efficientdet.onnx'):
    """Run single-pass baseline ONNX inference on one image."""
    session = ort.InferenceSession(model_path)
    image = cv2.imread(image_path)
    outputs = session.run(None, {'images': preprocess(image)})
    boxes = postprocess(outputs, image.shape)
    for (x1, y1, x2, y2, score) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite('outputs/baseline.jpg', image)
    print(f'Baseline detections: {len(boxes)}')
    print('Saved to outputs/baseline.jpg')
    return image, boxes


# ────────────────────────────────────────────────────────────
# FILE: src/sahi_inference.py
# SAHI sliced inference for small object detection
# ────────────────────────────────────────────────────────────

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def run_sahi(image_path, model_path='models/efficientdet.onnx'):
    """Run SAHI sliced inference on one image."""
    model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.5,
        device='cpu'
    )

    result = get_sliced_prediction(
        image_path, model,
        slice_height=512, slice_width=512,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        postprocess_type='NMS',
        postprocess_match_metric='IOU',
        postprocess_match_threshold=0.5
    )

    all_det = result.object_prediction_list
    vehicles = filter_vehicles(all_det)

    print(f'Total detections: {len(all_det)}')
    print(f'Vehicle detections: {len(vehicles)}')

    image = cv2.imread(image_path)
    image = draw_vehicle_boxes(image, vehicles)
    cv2.imwrite('outputs/final_vehicle_detection.jpg', image)
    print('Saved to outputs/final_vehicle_detection.jpg')

    baseline = cv2.imread('outputs/baseline.jpg')
    if baseline is not None:
        comparison = side_by_side(baseline, image.copy(),
                                  'Baseline Detection', 'SAHI Detection')
        cv2.imwrite('outputs/final_comparison.jpg', comparison)
        print('Saved to outputs/final_comparison.jpg')

    return image, vehicles


# ────────────────────────────────────────────────────────────
# FILE: src/batch_inference.py
# Run SAHI detection on all images in data/images/
# ────────────────────────────────────────────────────────────

def run_batch(model_path='models/efficientdet.onnx'):
    """Run SAHI detection on all images in data/images/."""
    model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.5,
        device='cpu'
    )

    image_folder = 'data/images'
    output_folder = 'outputs/batch'
    os.makedirs(output_folder, exist_ok=True)

    images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    total_vehicles = 0

    for i, image_file in enumerate(images):
        image_path = os.path.join(image_folder, image_file)
        result = get_sliced_prediction(
            image_path, model,
            slice_height=512, slice_width=512,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
            postprocess_type='NMS',
            postprocess_match_metric='IOU',
            postprocess_match_threshold=0.5
        )
        vehicles = filter_vehicles(result.object_prediction_list)
        total_vehicles += len(vehicles)
        image = cv2.imread(image_path)
        image = draw_vehicle_boxes(image, vehicles)
        cv2.imwrite(os.path.join(output_folder, image_file), image)
        print(f'[{i+1}/{len(images)}] {image_file} - Vehicles: {len(vehicles)}')

    print(f'\nDone! Total vehicles detected: {total_vehicles}')
    print(f'Average per image: {total_vehicles / len(images):.1f}')


# ────────────────────────────────────────────────────────────
# FILE: app.py
# Streamlit web application for interactive detection
# ────────────────────────────────────────────────────────────

STREAMLIT_APP = '''
import streamlit as st
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import tempfile
import os
import sys
sys.path.append('src')
from utils import filter_vehicles, draw_vehicle_boxes

st.title("Aerial Vehicle Detection")
st.write("Upload a drone image to detect vehicles using SAHI + ONNX")

@st.cache_resource
def load_model():
    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path="models/efficientdet.onnx",
        confidence_threshold=0.5,
        device="cpu"
    )

model = load_model()
uploaded_file = st.file_uploader("Upload a drone image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.image(uploaded_file, caption="Input Image", use_column_width=True)

    with st.spinner("Running SAHI vehicle detection..."):
        result = get_sliced_prediction(
            tmp_path, model,
            slice_height=512, slice_width=512,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.5
        )
        all_detections = result.object_prediction_list
        vehicle_detections = filter_vehicles(all_detections)
        image = cv2.imread(tmp_path)
        image = draw_vehicle_boxes(image, vehicle_detections)
        output_path = "outputs/streamlit_result.jpg"
        cv2.imwrite(output_path, image)

    st.image(output_path,
             caption=f"Vehicles Detected: {len(vehicle_detections)}",
             use_column_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Total Detections", len(all_detections))
    col2.metric("Vehicle Detections", len(vehicle_detections))
    os.unlink(tmp_path)
'''

# ────────────────────────────────────────────────────────────
# MAIN — Run all steps in sequence
# ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("  AERIAL VEHICLE DETECTION — SAHI + EfficientDet ONNX")
    print("=" * 60)

    args = sys.argv[1:] if len(sys.argv) > 1 else ['all']
    mode = args[0]

    image_folder = 'data/images'
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    if not image_files:
        print("No images found in data/images/")
        sys.exit(1)

    image_path = os.path.join(image_folder, image_files[0])

    if mode in ('visualize', 'all'):
        print("\n[1] Drawing ground truth annotations...")
        run_visualize()

    if mode in ('baseline', 'all'):
        print("\n[2] Running baseline inference...")
        run_baseline(image_path)

    if mode in ('sahi', 'all'):
        print("\n[3] Running SAHI inference...")
        run_sahi(image_path)

    if mode in ('batch', 'all'):
        print("\n[4] Running batch inference on all images...")
        run_batch()

    print("\nAll done! Check outputs/ folder for results.")
