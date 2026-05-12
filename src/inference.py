import cv2
import numpy as np
import onnxruntime as ort
import os

def preprocess(image, size=640):
    img = cv2.resize(image, (size, size))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def postprocess(outputs, orig_shape, threshold=0.3):
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
    session = ort.InferenceSession(model_path)
    image = cv2.imread(image_path)
    outputs = session.run(None, {'images': preprocess(image)})
    boxes = postprocess(outputs, image.shape)
    for (x1, y1, x2, y2, score) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite('outputs/baseline.jpg', image)
    print(f'Baseline detections: {len(boxes)}')
    return image, boxes

if __name__ == '__main__':
    image_file = [f for f in os.listdir('data/images') if f.endswith('.jpg')][0]
    run_baseline('data/images/' + image_file)
