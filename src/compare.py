import cv2
import numpy as np

baseline = cv2.imread('outputs/baseline.jpg')
sahi = cv2.imread('outputs/sahi.png')

h = max(baseline.shape[0], sahi.shape[0])
baseline = cv2.resize(baseline, (int(baseline.shape[1] * h / baseline.shape[0]), h))
sahi = cv2.resize(sahi, (int(sahi.shape[1] * h / sahi.shape[0]), h))

cv2.putText(baseline, 'BASELINE', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
cv2.putText(sahi, 'SAHI', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,165,255), 3)

comparison = np.hstack((baseline, sahi))
cv2.imwrite('outputs/comparison.jpg', comparison)
print('Saved to outputs/comparison.jpg')
