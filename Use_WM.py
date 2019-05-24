# BAP 2019
# Author: Axel Claeijs
# This script uses the trained network to make predictions

#-----------------------------------------------------------
# IMPORTS
#-----------------------------------------------------------
import sys
import cv2
import math
import pptk

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.ndimage.interpolation import rotate
from proto import pointcloud_pb2

#-----------------------------------------------------------
# VARIABLES
#-----------------------------------------------------------
VERSION = 7
MODEL_PATH = 'results/v' + str(VERSION) + '/trained_model' + str(VERSION) + '.h5'
ANGLE = np.random.randint(0,35)
EXTEND_DATA = 0     # Wheter or not to extend dataset with rotated variants

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
# CALC fn
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

SAMPLE = np.random.randint(0, len(data))
data_sample = data[SAMPLE]
label_sample = label[SAMPLE]

if EXTEND_DATA:
    data_sample = rotateData(data_sample, ANGLE)
    label_sample = rotateLabels(label_sample, ANGLE)

#print((np.array([data_sample,])).shape)

#predict the result
result = model.predict(np.array([data_sample,]))

image_p = result.reshape((40, 40))
plt.figure()
# plt.subplot(2, 1, 1)
plt.imshow(image_p)
plt.colorbar(label='Intensity')
plt.title('Predicted Costmap - ' + str(SAMPLE))
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

image_r = label_sample.reshape(40, 40)
plt.figure()
# plt.subplot(2, 1, 2)
plt.imshow(image_r)
plt.colorbar(label='Intensity')
plt.title('Real Costmap - ' + str(SAMPLE))
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.tight_layout()

# temp = 'predictCostmap_sample' + str(SAMPLE) + '.png'
# plt.savefig(temp)
plt.show()

v = pptk.viewer(data_sample.tolist())
v.set(floor_level=0)
v.set(point_size=0.05)
v.set(r=30)
v.set(theta=180)
v.set(phi=270)
v.set(bg_color=(0, 0, 0, 1))

# temp = 'predictPointcloud_sample' + str(SAMPLE) + '.png'
# v.capture(temp)
