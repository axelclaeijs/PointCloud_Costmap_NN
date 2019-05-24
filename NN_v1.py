# BAP 2019
# Author: Axel Claeijs
# This basic feed forward network will take a pointcloud as input and produces a costmap on its output. It uses the
# collected dataset in the included protobuf.
# Under HYPERPARAMS the networks parameters can be tweaked
# Under DESIGN MODEL the networks size can be changed
# Under MODEL ATTRIBUTES the networks optimizer, lossfunction and metrics can be changed

#-----------------------------------------------------------
# IMPORTS
#-----------------------------------------------------------
import errno
import sys
import cv2
import csv
import math
import pptk     # fix pptk viewer on ubuntu 18.06 -> https://github.com/heremaps/pptk/issues/3
import os
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from scipy.ndimage.interpolation import rotate
from contextlib import redirect_stdout
from proto import pointcloud_pb2

#-----------------------------------------------------------
# HYPERPARAMETERS
#-----------------------------------------------------------
VERSION = 23            # To save network in new folder
EPOCHS = 20
BATCH_SIZE = 18
ROUNDS = 5              # Each round will be a new training on the previous one (to publish intermediate results)
LEARNING_RATE = 0.01
EXTEND_DATASET = 0      # Rotate every datapoint with DEGREE
DEGREE = 10

#-----------------------------------------------------------
# FILEPATHS
#-----------------------------------------------------------
DIR = 'results/'
MODEL_IMG = DIR + 'v' + str(VERSION) + '/model' + str(VERSION) + '.png'
MODEL_SUM = DIR + 'v' + str(VERSION) + '/model' + str(VERSION) + '.txt'
GRAPH_MAE = DIR + 'v' + str(VERSION) + '/mae' + str(VERSION)
GRAPH_MSE = DIR + 'v' + str(VERSION) + '/mse' + str(VERSION)
GRAPH_LOSS = DIR + 'v' + str(VERSION) + '/loss' + str(VERSION)
GRAPH_LOSS2 = DIR + 'v' + str(VERSION) + '/evaluate_loss' + str(VERSION) + '.png'
MODEL_FILE = DIR + 'v' + str(VERSION) + '/trained_model' + str(VERSION) + '.h5'
PPTK_IMG = DIR + 'v' + str(VERSION) + '/PC_v' + str(VERSION) + '_'
PRED_COSTMAPS = DIR + 'v' + str(VERSION) + '/Predicted_CM' + str(VERSION) + '_'
REAL_COSTMAPS = DIR + 'v' + str(VERSION) + '/Real_CM' + str(VERSION) + '_'
GRAPH_PRED_MSE = DIR + 'v' + str(VERSION) + '/prediction_mse' + str(VERSION) + '.png'
GRAPH_EVAL_MSE = DIR + 'v' + str(VERSION) + '/evaluate_mse' + str(VERSION) + '.png'

#-----------------------------------------------------------
# HISTORY PRINT
#-----------------------------------------------------------
def plot_history(history):

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    # summarize history for mean absolute error
    plt.figure()
    plt.title('model mae')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Test Error')
    plt.legend()
    plt.savefig(GRAPH_MAE)

    # summarize history for mean squared error
    plt.figure()
    plt.title('model mse')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Test Error')
    plt.legend()
    plt.savefig(GRAPH_MSE)

    # summarize history for losses
    plt.figure()
    plt.title('model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train Loss')
    plt.plot(hist['epoch'], hist['val_loss'], label='Test Loss')
    plt.legend()
    plt.savefig(GRAPH_LOSS)

    plt.show()

#-----------------------------------------------------------
# HISTORY PRINT EVERY ROUND
#-----------------------------------------------------------
def plot_history_round(history, round):

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    # summarize history for mean absolute error
    plt.figure()
    plt.title('model mae')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Test Error')
    plt.legend()
    temp = GRAPH_MAE + '_r' + str(round) + ".png"
    plt.savefig(temp)

    # summarize history for mean squared error
    plt.figure()
    plt.title('model mse')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Test Error')
    plt.legend()
    temp = GRAPH_MSE + '_r' + str(round) + ".png"
    plt.savefig(temp)

    # summarize history for losses
    plt.figure()
    plt.title('model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train Loss')
    plt.plot(hist['epoch'], hist['val_loss'], label='Test Loss')
    plt.legend()
    temp = GRAPH_LOSS + '_r' + str(round) + ".png"
    plt.savefig(temp)


#-----------------------------------------------------------
# SHOW SAMPLES
#-----------------------------------------------------------
def show_result(data, predict_label, real_label, offset, amount, save, round):

    # Show and save the predicted costmaps in each round
    for i in range(amount):
        image = predict_label[offset + i].reshape((40, 40))
        plt.subplot(2, 2, i + 1)
        plt.imshow(image)
        plt.colorbar(label='Intensity')
        plt.title('Predicted Costmap - ' + str(offset + i))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()

    if save:
        temp = PRED_COSTMAPS + '_r' + str(round) + '.png'
        plt.savefig(temp)

    plt.close()

    # Show and save the real costmap (and pointcloud) only in the fist round (no need to do this every round)
    if round == 0:
        for j in range(amount):
            image = real_label[offset + j].reshape((40, 40))
            plt.subplot(2, 2, j + 1)
            plt.imshow(image)
            plt.colorbar(label='Intensity')
            plt.title('Real Costmap - ' + str(offset + j))
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.tight_layout()

        if save:
            temp = REAL_COSTMAPS + '_r' + str(round) + '.png'
            plt.savefig(temp)

        plt.close()

        for i in range(amount):
            v = pptk.viewer(data[offset + i].tolist())
            v.set(floor_level=0)
            v.set(point_size=0.05)
            v.set(r=30)
            v.set(theta=180)
            v.set(phi=270)
            v.set(bg_color=(0,0,0,1))

            if save:
                temp = PPTK_IMG + 'r' + str(round) + '_' + str(offset+i) + '.png'
                v.capture(temp)

#-----------------------------------------------------------
# CALCULATE MSE BETWEEN TWO COSTMAPS
#-----------------------------------------------------------
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(len(imageA))

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err
#-----------------------------------------------------------
# ROTATE LABELS
#-----------------------------------------------------------
def rotateLabels(labels, degree):
    new_labels = []
    for i in range(len(labels)):
        temp_label = labels[i].reshape((40, 40))
        new_labels.append(rotate(temp_label, -degree, mode='constant', reshape=False).reshape(1600))
    return np.array(new_labels)
#-----------------------------------------------------------
# ROTATE DATA
#-----------------------------------------------------------
def rotateData(data, degree):
    radian = math.radians(degree)

    rotMatrix = np.array([[np.cos(radian), -np.sin(radian), 0],
                          [np.sin(radian), np.cos(radian), 0],
                          [0, 0, 1]])
    new_data = []
    for i in range(len(data)):
        new_pc = []
        for j in range(len(data[i])):
            new_pc.append(np.dot(data[i][j], rotMatrix))
        new_data.append(new_pc)
    return np.array(new_data)

#---------------------------------------------- MAIN -------------------------------------------------------------------

#-----------------------------------------------------------
# FETCHING DATA
#-----------------------------------------------------------
if len(sys.argv) != 3:
    print("Usage:", sys.argv[0], "COLLECTION_FILE _ OUTPUT CSV")
    sys.exit(-1)

collection = pointcloud_pb2.PairCollection()

# Open input and output files
inputFile = open(sys.argv[1], "rb")
outputFileResults = open(sys.argv[2], "a")

resultWriter = csv.writer(outputFileResults, dialect='excel')

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

    # Only add scans of 300 points
    if len(pcPoints) == 300:
        data.append(np.array(pcPoints))
        label.append(pair.costmap.data)

# Train set with shape: (#pc, 300, 2)
data = np.array(data)

# Label set with shape: (#cm, 14400)
label = np.array(label)

# Normalize the labels [0 - 100] -> [0 - 1]
label = label/100

for i in range(len(label)):
    temp = label[i].reshape((120, 120))
    label_new.append(cv2.resize(temp, dsize=(40, 40), interpolation=cv2.INTER_CUBIC).reshape(1600))

# Label set with shape: (#cm, 1600)
label_new = np.array(label_new)

#-----------------------------------------------------------
# EXTEND DATASET
#-----------------------------------------------------------
if EXTEND_DATASET:
    data_to_rotate = data
    label_to_rotate = label_new

    for i in range(35):
        data = np.concatenate((data, rotateData(data_to_rotate, (i+1)*DEGREE)), axis=0)
        label_new = np.concatenate((label_new, rotateLabels(label_to_rotate, (i+1)*DEGREE)), axis=0)

    random.shuffle(data)
    random.shuffle(label_new)

#-----------------------------------------------------------
# PREPROCESS DATA
#-----------------------------------------------------------
data_train, data_test, label_train, label_test = train_test_split(data, label_new, test_size=0.25)

#-----------------------------------------------------------
# VISUALIZE DATA
#-----------------------------------------------------------
print("--------------------------------------------")
print("--------DATASET INFORMATION-----------------")
print("--------------------------------------------")
print("Size of total dataset: ", len(data))
print("Length train data: ", len(data_train))
print("Length test data: ", len(data_test))
print("Shape of inputset: ", data_train.shape)
print("Shape of labelset: ", label_train.shape)
print("Shape of input (pointcloud): ", data_train[0].shape)
print("Shape of label (costmap): ", label_train[0].shape)
print("--------------------------------------------")

#-----------------------------------------------------------
# DESIGN MODEL
#-----------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(300, 3,)),
    tf.keras.layers.Dense(1200, activation=tf.nn.relu, kernel_initializer='random_uniform', bias_initializer='zeros'),
    tf.keras.layers.Dense(1600, activation=tf.nn.sigmoid, kernel_initializer='random_uniform', bias_initializer='zeros')
])

#-----------------------------------------------------------
# MODEL DETAILS
#-----------------------------------------------------------
# Print model summery
model.summary()
with open(MODEL_SUM, 'w+') as f:
    with redirect_stdout(f):
        model.summary()

# If version-folder not exists, make new folder
if not os.path.exists(os.path.dirname(MODEL_IMG)):
    try:
        os.makedirs(os.path.dirname(MODEL_IMG))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

# Save schematic networkconfiguration
tf.keras.utils.plot_model(model, show_shapes='true', to_file=MODEL_IMG)

#-----------------------------------------------------------
# MODEL ATTRIBUTES
#-----------------------------------------------------------
# Loss fct: minimize this fct to steer the model in certain direction
# Optimizer: how model is optimized based on the data and the lossfct
# Metrics: for monitor the training and testing steps
#
sgd = tf.keras.optimizers.SGD(lr=LEARNING_RATE, momentum=0.0, decay=0.0, nesterov=False)

model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['mean_absolute_error', 'mean_squared_error']
              )

# Initialize all nodes in the network before training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

#-----------------------------------------------------------
# TRAIN MODEL
#-----------------------------------------------------------
# For every round
# 1. Train with trainingdata (labels and images)
# 2. Plot statistics
# 3. Evaluate with the test images. Compare result with test_labels and save mse
# 4. Evaluate with the train images. Compare result with train_labels and save mse
# 5. Make prediction and save mse
# 6. Write results to excel file
# Plot saves mse-values
#

test_loss = []
test_mae = []
test_mse = []
test_acc = []
train_loss = []
train_mae = []
train_mse = []
train_acc = []
predict_mse = []
all_mse_histories = []

for i in range(ROUNDS):
    print('------------- ROUND ' + str(i) + ' --------------')
    hist = model.fit(data_train, label_train, validation_data=(data_test, label_test), epochs=EPOCHS, verbose=2,
                     batch_size=BATCH_SIZE)
    plot_history(hist)
    all_mse_histories.append(hist.history['val_mean_squared_error'])

    #-----------------------------------------------------------
    # EVALUATE MODEL
    #-----------------------------------------------------------
    # A little bit of overfitting -> The testdata has worse results than the traindata
    # This tells us how well we can expect the model to predict when we use it in the real world

    test_loss_temp, test_mae_temp, test_mse_temp = model.evaluate(data_test, label_test,
                                                                                 batch_size=BATCH_SIZE)
    print("Testing set Mean Abs Error: {:5.2f} MPG".format(test_mae_temp))
    print("Testing set Mean Sqr Error: {:5.2f} MPG".format(test_mse_temp))
    print("Testing set Loss: ", test_loss_temp)

    test_loss.append(test_loss_temp)
    test_mae.append(test_mae_temp)
    test_mse.append(test_mse_temp)

    train_loss_temp, train_mae_temp, train_mse_temp = model.evaluate(data_train[:len(data_test)],
                                                                                     label_train[:len(label_test)],
                                                                                     batch_size=BATCH_SIZE)
    print("Training set Mean Abs Error: {:5.2f} MPG".format(train_mae_temp))
    print("Training set Mean Sqr Error: {:5.2f} MPG".format(train_mse_temp))
    print("Training set Loss: ", train_loss_temp)

    train_loss.append(train_loss_temp)
    train_mae.append(train_mae_temp)
    train_mse.append(train_mse_temp)

    #-----------------------------------------------------------
    # PREDICT MODEL
    #-----------------------------------------------------------
    # Predict MPG values using data in the testing set

    predictions = model.predict(data_test)

    show_result(data_test, predictions, label_test, 1, 4, 1, i)

    # Calc the mse from predicted images
    err = 0
    for j in range(len(data_test)):
        err += mse(predictions[j], label_test[j])
    err /= len(data_test)

    predict_mse.append(err)

    #-----------------------------------------------------------
    # WRITE RESULTS
    #-----------------------------------------------------------
    row = [VERSION, len(data), len(data_train), len(data_test), i, train_loss_temp, train_mae_temp, train_mse_temp,
           test_loss_temp, test_mae_temp, test_mse_temp]
    resultWriter.writerow(row)

outputFileResults.close()
model.save(MODEL_FILE)

#-----------------------------------------------------------
# PLOT STATISTICS
#-----------------------------------------------------------
average_mse_history = [
    np.mean([x[i] for x in all_mse_histories]) for i in range(EPOCHS)
]

# The average of the per-epoch MSE scores
plt.figure()
plt.title('Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error [$MPG^2$]]')
plt.plot(range(1, len(average_mse_history)+1), average_mse_history)
plt.savefig(GRAPH_EVAL_MSE)

# Mean Squared Error of predictions over different rounds
plt.figure()
plt.title('Prediction mse')
plt.xlabel('Round')
plt.ylabel('Mean Square Error [$MPG^2$]')
plt.plot(predict_mse)
plt.savefig(GRAPH_PRED_MSE)

# Loss of evaluation function over different rounds
plt.figure()
plt.title('Evaluation model mse')
plt.xlabel('Epoch')
plt.ylabel('mse')
plt.plot(train_mse, label='Train mse')
plt.plot(test_mse, label='Test mse')
plt.legend()
plt.savefig(GRAPH_LOSS2)
