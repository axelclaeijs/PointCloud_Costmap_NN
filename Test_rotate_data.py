# BAP 2019
# Author: Axel Claeijs
# This script tests the rotation of data and labels

#-----------------------------------------------------------
# IMPORTS
#-----------------------------------------------------------
import sys
import cv2
import pptk
import math
import random

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import rotate
from proto import pointcloud_pb2

#-----------------------------------------------------------
# VARIABLES
#-----------------------------------------------------------
VERSION = 8
MODEL_PATH = 'results/v' + str(VERSION) + '/trained_model' + str(VERSION) + '.h5'
SAMPLE = 1587
DEGREE = 90
RADIAN = math.radians(DEGREE)

rotMatrix = np.array([[np.cos(RADIAN), -np.sin(RADIAN), 0],
                      [np.sin(RADIAN),  np.cos(RADIAN), 0],
                      [0,  0, 1]])
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

print('Angle: ' + str(random.randrange(0, 360, 10)))

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
# CM ROTATION
#-----------------------------------------------------------

label1 = label[SAMPLE].reshape((40,40))
label2 = rotate(label1, -DEGREE, mode='nearest')

#-----------------------------------------------------------
# PC ROTATION
#-----------------------------------------------------------

def rotatePC(*pc):
    newPc = []
    for i in range(len(pc)):
        newPc.append(np.dot(pc[i], rotMatrix))
    return newPc

data1 = data[SAMPLE]
data2 = []
for i in range(len(data1)):
    data2.append(rotatePC(data1[i]))

#-----------------------------------------------------------
# DISPLAY
#-----------------------------------------------------------
plt.subplot(2, 1, 1)
plt.imshow(label1)
plt.colorbar(label='Intensity')
plt.title('Original CM - ' + str(SAMPLE))
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(2, 1, 2)
plt.imshow(label2)
plt.colorbar(label='Intensity')
plt.title('Rotated CM - ' + str(SAMPLE))
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

v = pptk.viewer(data1)
v.set(floor_level=0)
v.set(point_size=0.05)
v.set(r=30)
v.set(phi = 0)
v.set(theta=270)
v.set(bg_color=(0, 0, 0, 1))

v = pptk.viewer(data2)
v.set(floor_level=0)
v.set(point_size=0.05)
v.set(r=30)
v.set(phi = 0)
v.set(theta=270)
v.set(bg_color=(0, 0, 0, 1))