import os.path
import time
from contextlib import contextmanager

import cv2
import numpy as np
import pymysql
import pymysql.cursors

"""
export MYSQL_ROOT_PASS=...

docker run --name mysql -d \
    -p 3306:3306 \
    -e MYSQL_ROOT_PASSWORD=$MYSQL_ROOT_PASS \
    --restart unless-stopped \
    mysql:8

MYSQL_PWD=$MYSQL_ROOT_PASS mysql -h"127.0.0.1" --port "3306" -u "root" \
    -e "SET GLOBAL sql_mode = 'NO_ENGINE_SUBSTITUTION';"

MYSQL_PWD=$MYSQL_ROOT_PASS mysql -h"127.0.0.1" --port "3306" -u "root" < "schemas.sql"
"""
mysql_auth = {
    "host": "localhost",
    "user": "root",
    "password": "...",
    "db": "awb"
}

images_dir = "images"

image_size = 128


@contextmanager
def database_ctx(auth):
    con = pymysql.connect(**auth)
    cur = con.cursor(pymysql.cursors.DictCursor)
    try:
        yield cur
    finally:
        con.commit()
        cur.close()
        con.close()


if __name__ == "__main__":
    sift_detector = cv2.SIFT_create()
    images_path = os.path.join(os.getcwd(), images_dir)
    image_values = []
    start_time = time.time()
    for filename in os.listdir(images_path):
        file_path = os.path.join(images_path, filename)
        current_image = cv2.imread(file_path)
        height, width = current_image.shape[:2]
        resized_image = cv2.resize(current_image, (image_size, image_size))
        _, descriptors = sift_detector.detectAndCompute(resized_image, None)
        descriptors_blob = np.ndarray.dumps(descriptors)
        image_values.append((filename, width, height, descriptors_blob))
    with database_ctx(mysql_auth) as db:
        db.executemany('INSERT IGNORE INTO images(filename, width, height, sift) VALUES (%s, %s, %s, %s)', image_values)
    print(f"Processed {len(image_values)} images in {time.time() - start_time} seconds.")
