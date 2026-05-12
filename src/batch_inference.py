import cv2
import os
import sys
sys.path.append('src')
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from utils import filter_vehicles, draw_vehicle_boxes

model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='models/efficientdet.onnx',
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

print(f'Done! Total vehicles detected across all images: {total_vehicles}')
