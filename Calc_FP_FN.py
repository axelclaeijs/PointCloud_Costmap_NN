# BAP 2019
# Author: Axel Claeijs
# This script is used to calculate the number of false positives and false negatives

#-----------------------------------------------------------
# IMPORTS
#-----------------------------------------------------------
import sys
import cv2
import math
import operator

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from functools import reduce
from PIL import ImageChops
from scipy.ndimage.interpolation import rotate
from proto import pointcloud_pb2

#-----------------------------------------------------------
# VARIABLES
#-----------------------------------------------------------
VERSION = 7
MODEL_PATH = 'results/v' + str(VERSION) + '/trained_model' + str(VERSION) + '.h5'
ANGLE = np.random.randint(0,35)
ROUNDS = 1000
EXTEND_DATA = 0

#-----------------------------------------------------------
# CALC DIFFERENCE BETWEEN LABEL AND PREDICTION
#-----------------------------------------------------------
def compare_image(img1,img2):
    diff = img1 - img2
    temp = diff.ravel()
    n_z = np.linalg.norm(temp, ord=0)
    # Calc the norm of the array with only values diff from 0
    return n_z

def rmsdiff(im1, im2):
    "Calculate the root-mean-square difference between two images"

    h = ImageChops.difference(im1, im2).histogram()

    # calculate rms
    return math.sqrt(reduce(operator.add,
        map(lambda h, i: h*(i**2), h, range(256))
    ) / (float(im1.size[0]) * im1.size[1]))

#-----------------------------------------------------------
# PRINT DIFFERENT HISTOGRAMS
#-----------------------------------------------------------
def show_histogram(image, SAMPLE):
    bins = [0, 0.5, 1]
    plt.figure()
    plt.ylabel('Counts')
    plt.xlabel('bins')
    plt.title('Histogram - sample ' + str(SAMPLE))
    plt.hist(image,bins=bins)
    plt.show()

def show_histogram_2(image, SAMPLE):
    bins = [0, 0.5, 1]
    hist, bin_edges = np.histogram(image, bins)  # make the histogram

    fig, ax = plt.subplots()

    # Plot the histogram heights against integers on the x axis
    ax.bar(range(len(hist)), hist, width=0.5)

    # Set the ticks to the middle of the bars
    ax.set_xticks([0.5 + i for i, j in enumerate(hist)])

    # Set the xticklabels to a string that tells us what the bin edges were
    ax.set_xticklabels(['{} - {}'.format(bins[i], bins[i + 1]) for i, j in enumerate(hist)])

    plt.ylabel('Count')
    plt.xlabel('Bins')
    plt.title('Histogram - sample ' + str(SAMPLE))
    plt.show()

def show_histogram_3(image, SAMPLE):
    plt.figure()
    plt.ylabel('Counts')
    plt.xlabel('bins')
    plt.title('Histogram - sample ' + str(SAMPLE))
    plt.hist(image.ravel(),64,[0,1])
    plt.show()

#-----------------------------------------------------------
# EXTEND DATASET
#-----------------------------------------------------------
def rotateLabels(labels, degree):
    temp_label = labels.reshape((40, 40))
    temp_label = rotate(temp_label, -degree, mode='constant', reshape=False).reshape(1600)
    return np.array(temp_label)

def rotateData(data, degree):
    radian = math.radians(degree)

    rotMatrix = np.array([[np.cos(radian), -np.sin(radian), 0],
                          [np.sin(radian), np.cos(radian), 0],
                          [0, 0, 1]])
    new_pc = []
    for i in range(len(data)):
        new_pc.append(np.dot(data[i], rotMatrix))

    return np.array(new_pc)

#-----------------------------------------------------------
# CALC FP
#-----------------------------------------------------------
def false_positive(image_p, image_r):
    count = 0
    image_p_vec = image_p.reshape(1600)
    image_r_vec = image_r.reshape(1600)

    for i in range(1600):

        # pixel is predicted as object while it is free space
        if ((image_p_vec[i] >= 0.5) and (image_r_vec[i] < 0.5)):
            count += 1

    # false positives / number of negatives
    return count/len(sum(np.where(image_r_vec < 0.5)))

#-----------------------------------------------------------
# CALC FN
#-----------------------------------------------------------
def false_negative(image_p, image_r):
    count = 0
    image_p_vec = image_p.reshape(1600)
    image_r_vec = image_r.reshape(1600)

    for i in range(1600):

        # pixel is predicted as free space while it is an object
        if ((image_p_vec[i] < 0.5) and (image_r_vec[i] >= 0.5)):
            count += 1

    # false negatives / number of positives
    if count == 0:
        return 0
    else:
        return count /len(sum(np.where(image_r_vec >= 0.5)))

#-----------------------------------------------------------
# FETCHING DATA
#-----------------------------------------------------------

if len(sys.argv) != 2:
    print("Usage:", sys.argv[0], "COLLECTION_FILE")
    sys.exit(-1)

collection = pointcloud_pb2.PairCollection()

# Open input and output files
inputFile = open(sys.argv[1], "rb")

# Read the existing collection.
collection.ParseFromString(inputFile.read())
inputFile.close()

data = []
label = []
label_new = []

# Append each PC and CM to the train and label set
valid = 0
for pair in collection.pair:

    pcPoints = []
    for pt in pair.pointcloud.points:
        pcPoints.append([pt.x, pt.y, pt.z])
        # np.append(pcPoints, [pt.x, pt.y, pt.i])

    # Only add scans of 300 points
    if len(pcPoints) == 300:
        data.append(np.array(pcPoints))
        label.append(pair.costmap.data)

# Train set with shape: (#pc, 300, 2)
data = np.array(data)

# Label set with shape: (#cm, 14400)
label = np.array(label)

# Normalize the labels
label = label/100

for i in range(len(label)):
    temp = label[i].reshape((120, 120))
    label_new.append(cv2.resize(temp, dsize=(40, 40), interpolation=cv2.INTER_CUBIC).reshape(1600))

# Label set with shape: (#cm, 1600)
label = np.array(label_new)

#-----------------------------------------------------------
# MAIN
#-----------------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)

all_fn = []
all_fp = []

for i in range(ROUNDS):

    SAMPLE = np.random.randint(0, len(data))
    data_sample = data[SAMPLE]
    label_sample = label[SAMPLE]

    if EXTEND_DATA:
        data_sample = rotateData(data_sample, ANGLE)
        label_sample = rotateLabels(label_sample, ANGLE)

    #predict the result
    result = model.predict(np.array([data_sample,]))

    image_p = result.reshape((40, 40))
    image_r = label_sample.reshape(40, 40)

    all_fp.append(false_positive(image_p, image_r) * 100)
    all_fn.append(false_negative(image_p, image_r) * 100)

print("False positives: " + str(np.mean(all_fp)) +
      "% (percentage of free space incorrectly identified as obstacles)")
print("False negatives: " + str(np.mean(all_fn)) +
      "% (percentage of objects incorrectly identified as free space)")

show_histogram(image_p, SAMPLE)
show_histogram(image_r, SAMPLE)

show_histogram_2(image_p, SAMPLE)
show_histogram_2(image_r, SAMPLE)

show_histogram_3(image_p, SAMPLE)
show_histogram_3(image_r, SAMPLE)
