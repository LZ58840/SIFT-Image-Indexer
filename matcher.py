import pickle
import time

import cv2

from dataset import database_ctx, mysql_auth

FLANN_INDEX_KDTREE = 0
flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(flann_params, search_params)

start_time = time.time()

with database_ctx(mysql_auth) as db:
    db.execute('SELECT id, sift FROM images')
    images = db.fetchall()

for image in images:
    descriptors = pickle.loads(image["sift"])
    flann.add([descriptors])

flann.train()

print(f"Loaded {len(images)} images in {time.time() - start_time} seconds.")
