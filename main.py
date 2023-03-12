import math
import time
from tkinter.filedialog import askopenfilename
from collections import Counter

import cv2

import matcher
from dataset import image_size


def sigmoid(x, b=(0.3*512**2)/(image_size**2), o=(40*image_size**2)/(512**2)):
    return 1. / (1 + math.exp(-b * (x - o)))


filename = askopenfilename(filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg")])
image = cv2.imread(filename)
image_resized = cv2.resize(image, (image_size, image_size))
sift_detector = cv2.SIFT_create()
_, descriptors = sift_detector.detectAndCompute(image_resized, None)


if __name__ == '__main__':
    start_time = time.time()
    matches = matcher.flann.knnMatch(descriptors, k=2)
    print(f"Matched input image in {time.time() - start_time} seconds.")
    good = [m.imgIdx for m, n in matches if m.distance / n.distance < 0.75]
    tally = Counter(good)
    values = [(key, pct) for key, value in tally.items() if (pct := sigmoid(value)) > 0.9]
    values_str = "Matched " + ', '.join("image {} with {:.2%} certainty".format(key + 1, pct) for key, pct in values)
    if len(values) > 0:
        print(values_str)
    else:
        print("No matches were found.")
