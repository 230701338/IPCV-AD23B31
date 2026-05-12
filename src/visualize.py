import cv2
import os
import sys

sys.path.append('src')
from utils import load_visdrone_annotation, draw_boxes

image_folder = 'data/images'
annotation_folder = 'data/annotations'

# Get valid image files
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
image_file = image_files[0]

image_path = os.path.join(image_folder, image_file)
annotation_path = os.path.join(annotation_folder, image_file.replace('.jpg', '.txt'))

# Check annotation exists
if not os.path.exists(annotation_path):
    print("Annotation file not found!")
    exit()

# Load image
image = cv2.imread(image_path)
if image is None:
    print("Failed to load image")
    exit()

# Load boxes
boxes = load_visdrone_annotation(annotation_path)

# Draw boxes
image = draw_boxes(image, boxes)

# Save output
output_path = 'outputs/annotated.jpg'
cv2.imwrite(output_path, image)

print('Saved to', output_path)
print('Total boxes drawn:', len(boxes))