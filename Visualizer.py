# BAP 2019
# Author: Axel Claeijs
# This script shows datasamples

#-----------------------------------------------------------
# IMPORTS
#-----------------------------------------------------------
import sys
import numpy as np
from proto import pointcloud_pb2
import cv2
import pptk
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# PARAMETERS
#-----------------------------------------------------------
OFFSET = 1  # where to start in collection
AMOUNT = 1  # how many samples from offset

#-----------------------------------------------------------
# SHOW SAMPLES
#-----------------------------------------------------------
def show_samples(data, label, offset, amount):

    # Show first 4 datapoints
    for i in range(amount):

        image = label[offset + i].reshape((40, 40))
        plt.figure()
        plt.imshow(image)
        plt.colorbar(label='Intensity')
        plt.xlabel('X')
        plt.ylabel('Y')

        v = pptk.viewer(data[offset + i].tolist())
        v.set(floor_level=0)
        v.set(point_size=0.05)
        v.set(r=30)
        v.set(theta=180)
        v.set(phi=270)
        v.set(bg_color=(0,0,0,1))

        # v.capture('pointcloud.png')
        # plt.savefig('costmap.png')

#-----------------------------------------------------------
# FETCHING DATA
#-----------------------------------------------------------
if len(sys.argv) != 2:
    print("Usage:", sys.argv[0], "COLLECTION_FILE")
    sys.exit(-1)

collection = pointcloud_pb2.PairCollection()

# Read the existing collection.
f = open(sys.argv[1], "rb")
collection.ParseFromString(f.read())
f.close()

train = []
label = []
label_new = []

# Append each PC and CM to the train and label set
valid = 0
for pair in collection.pair:

    pcPoints = []
    for pt in pair.pointcloud.points:
        pcPoints.append([pt.x, pt.y, pt.z])

    # Only add scans of 300 points
    if len(pcPoints) == 300:
        train.append(np.array(pcPoints))
        label.append(pair.costmap.data)

# Train set with shape: (#pc, 300, 2)
train = np.array(train)

# Label set with shape: (#cm, 14400)
label = np.array(label)

# Normalize the labels
label = label/100

for i in range(len(label)):
    temp = label[i].reshape((120, 120))
    label_new.append(cv2.resize(temp, dsize=(40, 40), interpolation=cv2.INTER_CUBIC).reshape(1600))

# Label set with shape: (#cm, 1600)
label_new = np.array(label_new)

show_samples(train, label_new, 1, 1)
