from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2, os, sys
sys.path.append('src')
from utils import filter_vehicles, draw_vehicle_boxes

def run_sahi(image_path, model_path='models/efficientdet.onnx'):
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
    print(f'Total: {len(all_det)} | Vehicles: {len(vehicles)}')
    image = cv2.imread(image_path)
    image = draw_vehicle_boxes(image, vehicles)
    cv2.imwrite('outputs/final_vehicle_detection.jpg', image)
    print('Saved to outputs/final_vehicle_detection.jpg')
    return image, vehicles

if __name__ == '__main__':
    image_file = [f for f in os.listdir('data/images') if f.endswith('.jpg')][0]
    run_sahi('data/images/' + image_file)
