import cv2
import numpy as np

VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']

def draw_vehicle_boxes(image, detections):
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

def filter_vehicles(detections):
    return [d for d in detections if d.category.name in VEHICLE_CLASSES]

def side_by_side(img1, img2, label1, label2):
    h = max(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
    img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))
    cv2.putText(img1, label1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(img2, label2, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
    return np.hstack((img1, img2))
