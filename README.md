# SIFT Image Indexer

The purpose of this program is to provide a miniaturized SIFT-based Image Retrieval system for the purposes of image similarity metrics given a dataset of existing images and a query image.


## Quick Start
1. Prepare a folder `images` in the repo directory with all the images you would like to train on.
2. Ensure MySQL server is up and running. Execute the file `schemas.sql`.
3. Initiate a Python environment and install `requirements.txt`.
4. Modify the file `dataset.py` with your MySQL login information and run it.
5. Run `main.py` and choose the query image from the window that appears. The image ID in the output corresponds to the image ID in the DB.


## Motivation
Image similarity needs to be performed frequently on both an expanding and contracting image dataset. It is desirable to have the convenience of organizing, adding, and deleting image entries in a relational database while being able to quickly perform one of the most accurate ways to measure image similarity.


## How It Works

### Building the dataset
1. An existing dataset of images is loaded into memory. Each image is resized into a square thumbnail.
2. Obtain descriptors for each image using the SIFT function.
3. Store the descriptor as pickled data in a MySQL blob.

### Building the Indexer
1. Upon initialization of the program, an empty FLANN Indexer was instantiated.
2. All images from the DB are downloaded onto memory, un-pickled, and fed into the indexer.
3. The Indexer is trained.

### Querying the Indexer
1. Load a query image into memory and resize it to the specified thumbnail size.
2. Obtain descriptors for the query image using the SIFT function.
3. Use KNN matching the query image with `k=2` on the Indexer.
4. Perform the ratio test on the resulting feature matches.
5. Each prominent feature match has an associated image index from the DB. Determine the image(s) with the most amount of features matched by tallying the results up.
6. Use proportioned sigmoid function to convert number of matched features into similarity percentage. 


## Limitations
Testing, investigation, and debugging was done on a laptop using a 1500-image dataset. 

With a thumbnail size of 128x128:
- Building the DB took five minutes,
- Building the indexer took one second, and
- Matching a query image took two seconds.

With a thumbnail size of 512x512:
- Building the DB took five minutes,
- Building the indexer took eight seconds, and
- Matching a query image took two minutes.

Results will vary. If indexer building and/or matching takes too long, consider decreasing the thumbnail size.


## References
- [OpenCV docs](https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html)
- [tutorialspoint: How to implement FLANN based feature matching in OpenCV Python?](https://www.tutorialspoint.com/how-to-implement-flann-based-feature-matching-in-opencv-python)
- [Reddit: Use of a FLANN index to match a picture with a database](https://redd.it/jwkjnd)
- [StackOverflow: Fast way to match a picture with a database](https://stackoverflow.com/q/29563429)