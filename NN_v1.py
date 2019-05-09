# We use tfKeras to build and train models in TensorFlow

import sys

# Helper libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pptk


# External scripts
from proto import pointcloud_pb2

#-----------------------------------------------------------
# FETCHING DATA
#-----------------------------------------------------------
# Import datasets

if len(sys.argv) != 2:
    print "Usage:", sys.argv[0], "COLLECTION_FILE"
    sys.exit(-1)

collection = pointcloud_pb2.PairCollection()

# Read the existing collection.
f = open(sys.argv[1], "rb")
collection.ParseFromString(f.read())
f.close()

train = []
label = []

# Append each PC and CM to the train and label set
for pair in collection.pair:

    pcPoints = []
    for pt in pair.pointcloud.points:
        pcPoints.append([pt.x, pt.y, pt.z])
        # np.append(pcPoints, [pt.x, pt.y, pt.i])
    train.append(np.array(pcPoints))

    label.append(pair.costmap.data)



# Train set with shape: (#pc, 300, 3)
train = np.array(train)
print "Shape of train PC: ", train[0].shape
# Label set with shape: (#cm, 14400)
label = np.array(label)
print "Shape of label image: ", label[0].shape

# label dalen door 100 voor bereik tussen 0-1

print "Size of dataset: ", len(train)

#-----------------------------------------------------------
# PREPROCESS DATA
#-----------------------------------------------------------

data_train, data_test, label_train, label_test = train_test_split(train, label, test_size=0.25)

print "Length train data: ", len(data_train)
print "Train data shape: ", data_train[0].shape
print "Length test data: ", len(data_test)
print "Test label shape: ", label_test[0].shape
#-----------------------------------------------------------
# VISUALIZE DATA
#-----------------------------------------------------------
startSample = 1

# v = pptk.viewer(data_train[sample])
#
# image = label_train[sample].reshape((120, 120))
# plt.imshow(image)
# plt.colorbar()
# plt.show()

for i in range(4):
    v = pptk.viewer(data_train[startSample + i].tolist())
    image = label_train[startSample + i].reshape((120, 120))

    plt.subplot(2, 2, i + 1)
    plt.imshow(image)
    plt.colorbar()

plt.show()

#-----------------------------------------------------------
# DESIGN MODEL
#-----------------------------------------------------------
# ConvNets 2D for input images
# Activation fct: hidden layers ReLu, top layer: sigmoid
#
#   OPTIMIZATIONS
#   1) reduce output layer to eg 3x3
#   2) Different depth
#   3) Different width
#   4) Loss function for images
#   5) Different optimizer
#
model = keras.Sequential([
    # First layer, Dense or flatten
    keras.layers.Flatten(input_shape=(300, 3)),
    # How many layers?
    keras.layers.Dense(1200, activation=tf.nn.relu),
    keras.layers.Dense(2400, activation=tf.nn.relu),
    keras.layers.Dense(4800, activation=tf.nn.relu),
    keras.layers.Dense(9600, activation=tf.nn.relu),
    keras.layers.Dense(14400, activation=tf.nn.sigmoid)
])

#-----------------------------------------------------------
# MODEL ATTRIBUTES
#-----------------------------------------------------------

# Few more SETTINGS for training
# Loss fct: minimize this fct to steer the model in certain direction
# Optimizer: how model is optimized based on the data and the lossfct
# Metrcis: for monitor the training and testing steps
#
sgd = keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)

model.compile(loss='mse',
              optimizer=sgd,
              metrics=['accuracy']
              )

#-----------------------------------------------------------
# TRAIN MODEL
#-----------------------------------------------------------

# TRAIN
# 1. Feed with trainingdata (labels and images)
# 2. It trains itself
# 3. Evaluate with the test images. Compare result with test_labels
# Start training with 'fit' command
#
print "Train data shape: ", data_train[0].shape
print "Train label shape: ", label_train[0].shape

for i in range (20):
    model.fit(data_train, label_train, epochs=2, batch_size=2)


#-----------------------------------------------------------
# EVALUATE MODEL
#-----------------------------------------------------------

# A little bit of overfitting -> The testdata has worse results than the traindata
#
    test_loss, test_acc = model.evaluate(data_test, label_test, batch_size=1)
    print('test accuracy: ', test_acc)

#-----------------------------------------------------------
# PREDICT MODEL
#-----------------------------------------------------------

# e.g.
    predictions = model.predict(data_test)
    print('Shape predictions: ', predictions.shape)

    for i in range(4):
        v = pptk.viewer(data_test[startSample + i].tolist())
        image = predictions[startSample + i].reshape((120, 120))

        plt.subplot(2, 2, i + 1)
        plt.imshow(image)
        plt.colorbar()

    plt.show()