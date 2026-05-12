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

st.title('Aerial Vehicle Detection')
st.write('Upload a drone image to detect vehicles using SAHI + ONNX')

@st.cache_resource
def load_model():
    return AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='models/efficientdet.onnx',
        confidence_threshold=0.5,
        device='cpu'
    )

model = load_model()

uploaded_file = st.file_uploader('Upload a drone image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.image(uploaded_file, caption='Input Image', use_column_width=True)

    with st.spinner('Running SAHI vehicle detection...'):
        result = get_sliced_prediction(
            tmp_path, model,
            slice_height=512, slice_width=512,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
            postprocess_type='NMS',
            postprocess_match_metric='IOU',
            postprocess_match_threshold=0.5
        )

        all_detections = result.object_prediction_list
        vehicle_detections = filter_vehicles(all_detections)

        image = cv2.imread(tmp_path)
        image = draw_vehicle_boxes(image, vehicle_detections)
        output_path = 'outputs/streamlit_result.jpg'
        cv2.imwrite(output_path, image)

    st.image(output_path, caption=f'Vehicles Detected: {len(vehicle_detections)}', use_column_width=True)

    col1, col2 = st.columns(2)
    col1.metric('Total Detections', len(all_detections))
    col2.metric('Vehicle Detections', len(vehicle_detections))

    os.unlink(tmp_path)
