import cv2
import pathlib


image = cv2.imread("images/lena.png", cv2.IMREAD_GRAYSCALE)
feature_file = pathlib.Path("features.csv")

features = []

lines = []
with open(feature_file, "r") as f:
    lines = f.readlines()

for line in lines:
    x, y, score = line.split(',')
    x = float(x)
    y = float(y)
    score = float(score)

    features.append(cv2.KeyPoint(x, y, score))

new_image = cv2.drawKeypoints(image, features, None, color=(255, 0, 0))

fast = cv2.FastFeatureDetector_create()
print(f"Threshold: {fast.getThreshold()}")
print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
print("neighborhood: {}".format(fast.getType()))

fast.setThreshold(50)

kpts = fast.detect(image, None)

new_image2 = cv2.drawKeypoints(image, kpts, None, color=(255, 0, 0))

cv2.imshow("image", image)
cv2.imshow("mine", new_image)
cv2.imshow("opencv", new_image2)
cv2.waitKey(0)
