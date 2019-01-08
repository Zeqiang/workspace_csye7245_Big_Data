#!/bin/python
# -*- coding: utf8 -*-

"""
CNN Transfer Learning Using VGG19

This module demonstrates the entire training part using VGG19 structure, 
and the pretrained weights as a strat point for new training.

There are three stage for training processing, at first stage, train only
the last fc layer (this layer's weight is initialized), after get a good start 
point, the second stage train the entire sturcture with learning rate of 1e-5, 
third stage will also train the entire sturcture with less learning rate of 1e-6.

Example:
    To run the train script in background, use command:

        $ nohup python vgg19_3stage.py &

Output:
    System out put is saved in '/nohup.out'
    Performance Report is saved in '/report/'
    Well trained model is saved in '/saved_model/model_name/'
    TensorBoard (if have) is saved in '/tf_board/'
"""


# ===============================================================================================================
# Import packages
# ===============================================================================================================
import pandas as pd
import numpy as np
import cv2
import random, os, sys, math, json
from datetime import datetime, timedelta
import time
from sklearn import metrics
# transfer learning using tf.slim, easy to load the original model structure and checkpoint
# https://github.com/tensorflow/models/tree/master/research/slim
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets


# ===============================================================================================================
# Set up some paramaters
# ===============================================================================================================
notebookname = 'VGG19_with_pertrained_weight_3stages'

# Generate 7 random char as instance running id
run_id = random.sample('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', 7)
run_id = "".join(x for x in run_id)
# attampt_name is the name of the model
attampt_name = 'VGG19_Dec3_' + run_id

# Path and Dir configration
rootDir=''
csvDataPath = os.path.join(rootDir, 'data_csv/')
imageDataPath = os.path.join(rootDir,'data_images/')
checkpoint_file = os.path.join(rootDir,'vgg_19.ckpt')
tfBoardDir = os.path.join(rootDir,'tf_board', attampt_name)

# Paramaters for model
BATCH_SIZE = 20
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CLASS = 7
lesion_type_list = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']


# ===============================================================================================================
# Load csv data
# ===============================================================================================================
df_train_ori = pd.read_csv(os.path.join(csvDataPath, 'train.csv'))
df_train_all = pd.read_csv(os.path.join(csvDataPath, 'train_aug.csv'))
df_test_all = pd.read_csv(os.path.join(csvDataPath, 'test.csv'))

# The number of last 3 types of images is still less (even after data augmentation), 
# replicate last 3 types of images to make training dataset more balanced
df_lastTwo = df_train_ori.loc[df_train_ori['dx'].isin(['df', 'vasc'])].reset_index(drop=True)
df_lastThree = df_train_ori.loc[df_train_ori['dx'].isin(['akiec'])].reset_index(drop=True)
df_train_all = df_train_all.append([df_lastTwo]*16, ignore_index=True)
df_train_all = df_train_all.append([df_lastThree]*4, ignore_index=True)


# ===============================================================================================================
# Sample the dataset, This is for script TESTING PURPOSE
# ===============================================================================================================
# df_train_all = df_train_all.sample(frac=1).reset_index(drop=True).loc[:30]
# df_test_all = df_test_all.sample(frac=1).reset_index(drop=True).loc[:10]


# ===============================================================================================================
# Helper functions for input data
# ===============================================================================================================
def get_Labels(labels):
    """
    ARG:
        labels: list of actual label for a batch input
    RETURN:
        labelList: list of one-hot encoding labels
    """
    labelList = np.zeros((len(labels),7))
    for i in range(len(labels)):
        labIdx = lesion_type_list.index(labels[i])
        labelList[i][labIdx] = 1
    return labelList

def get_augFlag(ifTrain, csv_data):
    """
    ARG:
        ifTrain: True or False for if doing training process
        csv_data: the dataframe for the batch input
    RETURN:
        augFlag_list: list of int number in (0,5)
    DESCRIPTION:
        For train process make image augmentation for original image, 
        for test process, return 0 for no image change
    """
    csv_data = csv_data.reset_index(drop=True)
    augFlag_list = []
    if ifTrain:
        for i in range(len(csv_data)):
            if csv_data.loc[i].image_id[-2] == '_' or csv_data.loc[i].image_id[-3] == '_':
                augFlag_list.append(0)
            else:
                augFlag_list.append(random.randint(a=0, b=5))
        augFlag_list = np.array(augFlag_list)
    else: 
        augFlag_list = np.zeros(len(csv_data))
    return augFlag_list

# Get Next Batch
def next_batch(indx1, indx2, csv_data, ifTrain):
    """
    ARG:
        indx1: start index for batch
        indx2: end index for batch
        csv_data: the dataframe of the all image information
        ifTrain: True or False for if doing training process
    RETURN:
        imgPaths: list of path for batch images
        imgLabels: list of labels for batch images
        augFlags: list of augFlag for batch images
    DESCRIPTION:
        Get all information for one batch
    """
    imgPaths = np.array(csv_data.path[indx1:indx2])
    imgLabels = get_Labels(list(csv_data.dx[indx1:indx2]))
    augFlags = get_augFlag(ifTrain, csv_data[indx1:indx2])
    return imgPaths, imgLabels, augFlags


# ===============================================================================================================
# Model Input Creation
# ===============================================================================================================
tf.reset_default_graph()

x_in = tf.placeholder(tf.string, shape=(None,), name='img_paths')
Aug_flag = tf.placeholder(tf.int32, shape=(None,), name='aug_flags')
isTrain = tf.placeholder(tf.bool, name='is_training')
# actual placeholder
y_label = tf.placeholder(tf.int32, shape=(None, NUM_CLASS), name='labels')


def f0(image_in): return image_in
def f1(image_in): 
    # random flip left right
    image_result = tf.image.random_flip_left_right(image_in)
    if random.randint(0,1) == 1:
        image_result = tf.image.transpose_image(image_result)
    return image_result
def f2(image_in): 
    # random flip up down
    image_result = tf.image.random_flip_up_down(image_in)
    if random.randint(a=0, b=1) == 1:
        image_result = tf.image.transpose_image(image_result)
    return image_result
def f3(image_in): 
    # random rotate within angle [-180, 180]
    angles = np.random.uniform(-180, 180, 1)
    image_result = tf.contrib.image.rotate(image_in,angles=angles,interpolation='NEAREST')
    return image_result
def f4(image_in): 
    # scale and random crop
    temp_img = tf.image.resize_images(image_in, [256, 256])
    image_result = tf.random_crop(temp_img, [IMG_HEIGHT, IMG_WIDTH, 3])
#     image_result = tf.image.central_crop(temp_img, (224/280))
    return image_result
def f5(image_in): 
    # random shear within range [-0.2,0.2]
    shear_angle = np.deg2rad(np.random.uniform(-0.2, 0.2))
    shear_matrix = np.array([1, -np.sin(shear_angle), 0, 0, np.cos(shear_angle), 0, 0, 0])
    image_result = tf.contrib.image.transform(image_in,shear_matrix,interpolation='NEAREST')
    return image_result
# def f6(image_in): 
# #     deltaScore = np.random.uniform(-200, 200)
# #     image_resized = tf.image.adjust_brightness(image_in, delta=0.9)
#     image_result = tf.image.adjust_contrast(image_in,contrast_factor=0.5)
#     return image_result

def load_image(input_elems):
    """
    ARG:
        input_elems: one image path and its augFlag
    RETURN:
        image_result: the real image that is going to be passed into madel
    DESCRIPTION:
        f0: make no change
        f1: random flip left right and random transpose
        f2: random flip up down and random transpose
        f3: random rotate within angle [-180, 180]
        f4: scale and random crop
        f5: random shear within range [-0.2,0.2]
    """
    
    image_file = input_elems[0]
    augFlag = input_elems[1]
    
    image = tf.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    input_image = tf.cast(image, tf.float32)
    # resizing to 224 x 224 x 3
    image_resized = tf.image.resize_images(input_image, [IMG_HEIGHT, IMG_WIDTH], align_corners=True, 
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Data Augmentation
    image_result = tf.case({tf.equal(augFlag, 0): lambda: f0(image_resized), 
                            tf.equal(augFlag, 1): lambda: f1(image_resized), 
                            tf.equal(augFlag, 2): lambda: f2(image_resized), 
                            tf.equal(augFlag, 3): lambda: f3(image_resized), 
                            tf.equal(augFlag, 4): lambda: f4(image_resized), 
                            tf.equal(augFlag, 5): lambda: f5(image_resized)},
                           default=lambda: f0(image_resized), 
                           exclusive=True)
    return image_result

elems = (x_in, Aug_flag)
train_dataset = tf.map_fn(load_image, elems, dtype=(tf.float32))
image_inputs = tf.identity(train_dataset, name='new_inputs')


# ===============================================================================================================
# Load vgg19 model using tf.slim
# ===============================================================================================================
# load model
vgg_model = nets.vgg
with slim.arg_scope(vgg_model.vgg_arg_scope()):
    old_pred,_ = vgg_model.vgg_19(inputs=image_inputs, num_classes=7, is_training=isTrain, spatial_squeeze=True)

# restore some weights for stage one training
exclude_variables = ['vgg_19/fc8']
variables_to_restore = slim.get_variables_to_restore(exclude=exclude_variables)
saver_ckpt = tf.train.Saver(variables_to_restore)


# ===============================================================================================================
# Output layers
# ===============================================================================================================
# output layer
vgg_19_fc8_last = tf.get_default_graph().get_tensor_by_name('vgg_19/fc8/squeezed:0')
with tf.variable_scope('prediction'):
    y_pred_softmax = tf.nn.softmax(vgg_19_fc8_last, name='pred_softmax')
    y_pred_cls = tf.argmax(y_pred_softmax, axis=1, name='pred_class')

# Performance Measures 
y_label_cls = tf.cast(tf.argmax(y_label, axis=1), tf.int32, name='y_label_cls')
correct_prediction = tf.equal(tf.cast(y_pred_cls, tf.int32), y_label_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='batch_accuracy')


# ===============================================================================================================
# Variables to train in different stages (change when fine tuning)
# ===============================================================================================================
variables_to_first_train = list(set(slim.get_variables_to_restore()) - set(variables_to_restore))
variables_new_layer = [var for var in tf.global_variables() if 'prediction' in var.name]
variables_to_train_all = list(variables_to_restore + variables_to_first_train + variables_new_layer)


# ===============================================================================================================
# TF Board
# ===============================================================================================================
with tf.variable_scope('TensorBoard'):
    graph_writer = tf.summary.FileWriter(tfBoardDir)

    tfb_loss = tf.placeholder(tf.float32, shape=(), name='tfb_loss')
    tfb_accuracy = tf.placeholder(tf.float32, shape=(), name='tfb_accuracy')

    tf.summary.scalar('tfb_loss', tfb_loss)
    tf.summary.scalar('tfb_accuracy', tfb_accuracy)
    merged = tf.summary.merge_all()
    
    
# ===============================================================================================================
# Loss Function and Training Optimization
# ===============================================================================================================
# L2 regularizer
weights_list_all = [var for var in variables_to_train_all if 'weights' in var.name]
# weights_list_all = [var for var in tf.global_variables() if 'weights' in var.name]
l2_loss_list_all = [tf.nn.l2_loss(var) for var in weights_list_all]
regularizers = sum(l2_loss_list_all)

# Cost-function
#tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=y_actual)
#cross_entropy = tf.losses.get_total_loss()
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=vgg_19_fc8_last, labels=y_label)
loss_1 = tf.reduce_mean(cross_entropy, name='loss_1')
loss_2 = tf.reduce_mean(cross_entropy + 0.00001 * regularizers, name='loss_2')

# Optimization method
myOptimizer_1 = tf.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.9)
myOptimizer_2 = tf.train.MomentumOptimizer(learning_rate=1e-5, momentum=0.9)
myOptimizer_3 = tf.train.MomentumOptimizer(learning_rate=1e-6, momentum=0.9)
train_op_1 = myOptimizer_1.minimize(loss_1, var_list=variables_to_first_train, name='train_op_1')
train_op_2 = myOptimizer_2.minimize(loss_2, var_list=variables_to_train_all, name='train_op_2')
train_op_3 = myOptimizer_3.minimize(loss_2, var_list=variables_to_train_all, name='train_op_3')


# ===============================================================================================================
# Start a Session for training
# ===============================================================================================================
# Session creation and variables initialization
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
logits_initializers = [var.initializer for var in tf.global_variables() if 'vgg_19/fc8' in var.name]
sess.run(logits_initializers)
momentum_initializers = [var.initializer for var in tf.global_variables() if 'Momentum' in var.name]
sess.run(momentum_initializers)
saver_ckpt.restore(sess, checkpoint_file)


# ===============================================================================================================
# Helper functions for model performance on train dataset
# ===============================================================================================================
# Get training data accuracy and loss
def train_result():
    """
    RETURN:
        loss_over_all: the loss_1 for entire train data in model prediction
        acc: the accuracy for entire train data in model prediction
        cm: the confusion matrix for entire train data in model prediction
    """
    num_train = len(df_train_all)
    train_label_cls = np.zeros(shape=num_train, dtype=np.int)
    train_pred_cls = np.zeros(shape=num_train, dtype=np.int)
    train_pred_prob = np.zeros(shape=[num_train,7])
    
    idx = 0
    num_of_steps = 0
    loss_value_all = 0.00
    train_accuracy_all = 0.00
    
    while idx < num_train:
        kidx = min(idx + BATCH_SIZE, num_train)
        x_batch, y_true_batch, augFlags = next_batch(idx, kidx, df_train_all, False)
        
        loss_batch = sess.run(loss_1, feed_dict = {x_in:x_batch, y_label:y_true_batch, isTrain:False, Aug_flag:augFlags})
        acc_batch = sess.run(accuracy, feed_dict = {x_in:x_batch, y_label:y_true_batch, isTrain:False, Aug_flag:augFlags})
        
        train_label_cls[idx:kidx] = sess.run(y_label_cls, feed_dict={y_label:y_true_batch})
        train_pred_cls[idx:kidx] = sess.run(y_pred_cls, feed_dict={x_in:x_batch, isTrain:False, Aug_flag:augFlags})
        train_pred_prob[idx:kidx] = sess.run(y_pred_softmax, feed_dict={x_in:x_batch, isTrain:False, Aug_flag:augFlags})
        
        loss_value_all += loss_batch
        train_accuracy_all += acc_batch
        idx = kidx
        num_of_steps += 1
        
    loss_over_all = loss_value_all/num_of_steps
    acc = train_accuracy_all/num_of_steps
    cm = metrics.confusion_matrix(y_true=train_label_cls, y_pred=train_pred_cls)
    
    return loss_over_all, acc, cm


# ===============================================================================================================
# Helper functions for model performance on test dataset
# ===============================================================================================================
# Test the model, Showing the performance
def print_test_result():
    """
    RETURN:
        loss_over_all: the loss_1 for entire test data in model prediction
        acc: the accuracy for entire test data in model prediction
        cm: the confusion matrix for entire test data in model prediction
    """
    num_test = len(df_test_all)
    test_label_cls = np.zeros(shape=num_test, dtype=np.int)
    test_pred_cls = np.zeros(shape=num_test, dtype=np.int)
    test_pred_prob = np.zeros(shape=[num_test,7])
    
    idx = 0
    num_of_steps = 0
    loss_value_all = 0.00
    
    while idx < num_test:
        
        kidx = min(idx + BATCH_SIZE, num_test)
        x_batch, y_true_batch, augFlags = next_batch(idx, kidx, df_test_all, False)
        
        loss_batch = sess.run(loss_1, feed_dict = {x_in:x_batch, y_label:y_true_batch, isTrain:False, Aug_flag:augFlags})
        acc_batch = sess.run(accuracy, feed_dict = {x_in:x_batch, y_label:y_true_batch, isTrain:False, Aug_flag:augFlags})
        
        test_label_cls[idx:kidx] = sess.run(y_label_cls, feed_dict={y_label:y_true_batch})
        test_pred_cls[idx:kidx] = sess.run(y_pred_cls, feed_dict={x_in:x_batch, isTrain:False, Aug_flag:augFlags})
        test_pred_prob[idx:kidx] = sess.run(y_pred_softmax, feed_dict={x_in:x_batch, isTrain:False, Aug_flag:augFlags})
        
        loss_value_all += loss_batch
        idx = kidx
        num_of_steps += 1
        
    loss_over_all = loss_value_all/num_of_steps
    correct = (test_label_cls == np.array(test_pred_cls))
    correct_sum = np.array(correct).sum()
    acc = float(correct_sum) / num_test

    cm = metrics.confusion_matrix(y_true=test_label_cls, y_pred=test_pred_cls)
    
    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2}),  loss: {3:>5.5}"
    msg_format = msg.format(acc, correct_sum, num_test, loss_over_all)
    message_all = msg_format + '\n' + str(cm)
    
    print_to_write(attampt_name, message_all, 'a')
    print(message_all)
    print('-------------------------------------------------------------------------')


# ===============================================================================================================
# Helper functions for saving model
# ===============================================================================================================
# model saver  
def save_model(version):
    """
    ARG:
        version: the executed epoch for the training process, also the version of the model
    DESCRIPTION:
        Save model into 'saved_model/version/' folder
    """
    if not os.path.exists('saved_model/' + attampt_name):
        os.mkdir('saved_model/' + attampt_name)
    all_saver = tf.train.Saver()
    all_saver.save(sess, 'saved_model/' + attampt_name + '/' + str(version) + '/' + attampt_name)


# ===============================================================================================================
# Helper functions for writing result into report file
# ===============================================================================================================
# write to file python3
def print_to_write(file_name, msg_content, do):
    """
    ARG:
        file_name: name of report file
        msg_content: message that is about to write into report file
        do: 'a' or 'w+', appending or writing
    DESCRIPTION:
        Write message into report file
    """
    logFile = open('report/' + file_name + '.log', do)
    print(msg_content, file=logFile)
    print('-------------------------------------------------------------------------', file=logFile)
    logFile.close()


# ===============================================================================================================
# Helper functions entire training and testing optimization
# ===============================================================================================================
def myOptimizing():
    """
    DESCRIPTION:
        Entire training and testing optimization, print and save the result including model and performance
    """
    start_time = time.time()
    # write to tf board
    graph_writer.add_graph(sess.graph)
    ############################################## For the First stage ##############################################
    for e in range(1,4):
        start_time_e = time.time()

        #shuffle order of data
        train_csv_shuffle = df_train_all.sample(frac=1).reset_index(drop=True)
        num_train = len(train_csv_shuffle)
        idx = 0
        
        while idx < num_train:
            kidx = min(idx + BATCH_SIZE, num_train)
            x_batch, y_true_batch, augFlags = next_batch(idx, kidx, train_csv_shuffle, True)
            sess.run(train_op_1, feed_dict = {x_in:x_batch, y_label:y_true_batch, isTrain:True, Aug_flag:augFlags})
            idx = kidx
        
        loss_over_all, acc, cm = train_result()
        msg = 'Epochs:{0:>3},  Training Accuracy: {1:>6.1%},  loss: {2:>5.5}'
        msg_format = msg.format(e, acc, loss_over_all)
        message_all = msg_format + '\n' + str(cm)

        print_to_write(attampt_name, message_all, 'a')
        print(message_all)
        print('-------------------------------------------------------------------------')
        print_test_result()
        
        # save to tensor board
        summary = sess.run(merged, feed_dict={tfb_loss: loss_over_all, tfb_accuracy: acc})
        graph_writer.add_summary(summary, e)
        
        end_time_e = time.time()
        time_dif_e = end_time_e - start_time_e
        e_time = "Time usage: " + str(timedelta(seconds=int(round(time_dif_e))))
        print_to_write(attampt_name, e_time, 'a')
        print(e_time)
        print('-------------------------------------------------------------------------')
    save_model(3)

    ############################################## For the Second stage #############################################
    for e in range(4,7):
        start_time_e = time.time()

        #shuffle order of data
        train_csv_shuffle = df_train_all.sample(frac=1).reset_index(drop=True)
        num_train = len(train_csv_shuffle)
        idx = 0
        
        while idx < num_train:
            kidx = min(idx + BATCH_SIZE, num_train)
            x_batch, y_true_batch, augFlags = next_batch(idx, kidx, train_csv_shuffle, True)
            sess.run(train_op_2, feed_dict = {x_in:x_batch, y_label:y_true_batch, isTrain:True, Aug_flag:augFlags})
            idx = kidx
        
        loss_over_all, acc, cm = train_result()
        msg = 'Epochs:{0:>3},  Training Accuracy: {1:>6.1%},  loss: {2:>5.5}'
        msg_format = msg.format(e, acc, loss_over_all)
        message_all = msg_format + '\n' + str(cm)
        
        print_to_write(attampt_name, message_all, 'a')
        print(message_all)
        print('-------------------------------------------------------------------------')
        print_test_result()
        
        # save to tensor board
        summary = sess.run(merged, feed_dict={tfb_loss: loss_over_all, tfb_accuracy: acc})
        graph_writer.add_summary(summary, e)
        
        end_time_e = time.time()
        time_dif_e = end_time_e - start_time_e
        e_time = "Time usage: " + str(timedelta(seconds=int(round(time_dif_e))))
        print_to_write(attampt_name, e_time, 'a')
        print(e_time)
        print('-------------------------------------------------------------------------')
    save_model(6)
    
    ############################################## For the Third stage ##############################################
    for e in range(7,11):
        start_time_e = time.time()

        #shuffle order of data
        train_csv_shuffle = df_train_all.sample(frac=1).reset_index(drop=True)
        num_train = len(train_csv_shuffle)
        idx = 0
        
        while idx < num_train:
            kidx = min(idx + BATCH_SIZE, num_train)
            x_batch, y_true_batch, augFlags = next_batch(idx, kidx, train_csv_shuffle, True)
            sess.run(train_op_3, feed_dict = {x_in:x_batch, y_label:y_true_batch, isTrain:True, Aug_flag:augFlags})
            idx = kidx
        
        loss_over_all, acc, cm = train_result()
        msg = 'Epochs:{0:>3},  Training Accuracy: {1:>6.1%},  loss: {2:>5.5}'
        msg_format = msg.format(e, acc, loss_over_all)
        message_all = msg_format + '\n' + str(cm)

        print_to_write(attampt_name, message_all, 'a')
        print(message_all)
        print('-------------------------------------------------------------------------')
        print_test_result()
        
        # save to tensor board
        summary = sess.run(merged, feed_dict={tfb_loss: loss_over_all, tfb_accuracy: acc})
        graph_writer.add_summary(summary, e)
        
        end_time_e = time.time()
        time_dif_e = end_time_e - start_time_e
        e_time = "Time usage: " + str(timedelta(seconds=int(round(time_dif_e))))
        print_to_write(attampt_name, e_time, 'a')
        print(e_time)
        print('-------------------------------------------------------------------------')
    save_model(10)
    
    # Total Time Usage
    end_time = time.time()
    time_dif = end_time - start_time
    total_time = "Total Trainning Time usage: " + str(timedelta(seconds=int(round(time_dif))))
    
    print_to_write(attampt_name, total_time, 'a')
    print(total_time)


# ===============================================================================================================
# Main function to start model training and generate performance report
# ===============================================================================================================
if __name__ == "__main__":
    
    print_to_write(attampt_name, attampt_name, 'w+')
    print_to_write(attampt_name, '--------------------------- Starting Training ---------------------------', 'a')
    print(attampt_name)
    print('--------------------------- Starting Training ---------------------------')
#     save_model(0)
    print_test_result()
    myOptimizing()
    print('------------------------------- Complete --------------------------------')
    sess.close()

    








