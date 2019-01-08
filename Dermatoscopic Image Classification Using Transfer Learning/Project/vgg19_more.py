#!/bin/python
# -*- coding: utf8 -*-

# CNN Transfer Learning Using VGG19


#############################################################
# Import packages
#############################################################
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


#############################################################
# Set up some paramaters
#############################################################
notebookname = 'More Epoch Training for VGG19'
pre_trained_model = 'VGG19_Dec3_W5yNfZu'
pre_trained_version = '10'

run_id = random.sample('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', 7)
run_id = "".join(x for x in run_id)
attampt_name = 'VGG19_More_Dec4_' + run_id

# rootDir=os.path.abspath(os.curdir)
rootDir=''
csvDataPath = os.path.join(rootDir, 'data_csv/')
imageDataPath = os.path.join(rootDir,'data_images/')
preModelDir = os.path.join(rootDir,'saved_model', pre_trained_model, pre_trained_version)
tfBoardDir = os.path.join(rootDir,'tf_board', attampt_name)

BATCH_SIZE = 20
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CLASS = 7

lesion_type_list = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']


#############################################################
# Load csv data
#############################################################
df_train_ori = pd.read_csv(os.path.join(csvDataPath, 'train.csv'))
df_train_all = pd.read_csv(os.path.join(csvDataPath, 'train_aug.csv'))
df_test_all = pd.read_csv(os.path.join(csvDataPath, 'test.csv'))

# Replicate last 3 types of images (inbalanced)
df_lastTwo = df_train_ori.loc[df_train_ori['dx'].isin(['df', 'vasc'])].reset_index(drop=True)
df_lastThree = df_train_ori.loc[df_train_ori['dx'].isin(['akiec'])].reset_index(drop=True)
df_train_all = df_train_all.append([df_lastTwo]*16, ignore_index=True)
df_train_all = df_train_all.append([df_lastThree]*4, ignore_index=True)


#############################################################
# df_train_all = df_train_all.sample(frac=1).reset_index(drop=True).loc[:100]
# df_test_all = df_test_all.sample(frac=1).reset_index(drop=True).loc[:50]
#############################################################


#############################################################
# Helper functions for input data
#############################################################
def get_Labels(labels):
    labelList = np.zeros((len(labels),7))
    for i in range(len(labels)):
        labIdx = lesion_type_list.index(labels[i])
        labelList[i][labIdx] = 1
    return labelList

def get_augFlag(ifTrain, csv_data):
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
    imgPaths = np.array(csv_data.path[indx1:indx2])
    imgLabels = get_Labels(list(csv_data.dx[indx1:indx2]))
    augFlags = get_augFlag(ifTrain, csv_data[indx1:indx2])
    return imgPaths, imgLabels, augFlags


#############################################################
# Pretrained Model Loading
#############################################################
tf.reset_default_graph()

# import meta graph
saver_import = tf.train.import_meta_graph(preModelDir + '/' + pre_trained_model + '.meta')
graph = tf.get_default_graph()


#############################################################
# Get placeholder tensor by name
#############################################################
# input
x_in = graph.get_tensor_by_name('img_paths:0')
Aug_flag = graph.get_tensor_by_name('aug_flags:0')
isTrain = graph.get_tensor_by_name('is_training:0')

# actual placeholder
y_label = graph.get_tensor_by_name('labels:0')


#############################################################
# Get useful tensor by name
#############################################################
# output tensor
vgg_19_fc8_last = graph.get_tensor_by_name('vgg_19/fc8/squeezed:0')
y_pred_softmax = graph.get_tensor_by_name('prediction/pred_softmax:0')
y_pred_cls = graph.get_tensor_by_name('prediction/pred_class:0')

# performance measures tensor
y_label_cls = graph.get_tensor_by_name('y_label_cls:0')
accuracy = graph.get_tensor_by_name('batch_accuracy:0')

# loss tensor
loss_1 = graph.get_tensor_by_name('loss_1:0')
loss_2 = graph.get_tensor_by_name('loss_2:0')

# performance measures tensor
train_op_2 = graph.get_operation_by_name('train_op_2')
train_op_3 = graph.get_operation_by_name('train_op_3')


#############################################################
# TF Board
#############################################################
# with tf.variable_scope('TensorBoard'):
#     graph_writer = tf.summary.FileWriter(tfBoardDir)

#     tfb_loss = tf.placeholder(tf.float32, shape=(), name='tfb_loss')
#     tfb_accuracy = tf.placeholder(tf.float32, shape=(), name='tfb_accuracy')

#     tf.summary.scalar('tfb_loss', tfb_loss)
#     tf.summary.scalar('tfb_accuracy', tfb_accuracy)
#     merged = tf.summary.merge_all()


#############################################################
# Start a Session for more training
#############################################################
# Session creation and variables loading
sess = tf.Session()

# load weights and bias
saver_import.restore(sess, preModelDir + '/' + pre_trained_model)


#############################################################
# Helper functions for model performance on train dataset
#############################################################
# Get training data accuracy and loss
def train_result():
    
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


#############################################################
# Helper functions for model performance on test dataset
#############################################################
# Test the model, Showing the performance
def print_test_result():
    
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


#############################################################
# Helper functions for saving model
#############################################################
# model saver  
def save_model(version):
    if not os.path.exists('saved_model/' + attampt_name):
        os.mkdir('saved_model/' + attampt_name)
    all_saver = tf.train.Saver()
    all_saver.save(sess, 'saved_model/' + attampt_name + '/' + str(version) + '/' + attampt_name)


#############################################################
# Helper functions for writing result into report file
#############################################################
# write to file python3
def print_to_write(file_name, msg_content, do):
    logFile = open('report/' + file_name + '.log', do)
    print(msg_content, file=logFile)
    print('-------------------------------------------------------------------------', file=logFile)
    logFile.close()


#############################################################
# Helper functions entire training and testing optimization
#############################################################
def myOptimizing():
    start_time = time.time()
    # write to tf board
#     graph_writer.add_graph(sess.graph)
    #########################################################################################################
    # More for the Second stage, 
    for e in range(11,15):
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
#         summary = sess.run(merged, feed_dict={tfb_loss: loss_over_all, tfb_accuracy: acc})
#         graph_writer.add_summary(summary, e)
        
        end_time_e = time.time()
        time_dif_e = end_time_e - start_time_e
        e_time = "Time usage: " + str(timedelta(seconds=int(round(time_dif_e))))
        print_to_write(attampt_name, e_time, 'a')
        print(e_time)
        print('-------------------------------------------------------------------------')
    save_model(14)
    
    ######################################################################################################### 
    # For the third stage, 
    for e in range(15,21):
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
#         summary = sess.run(merged, feed_dict={tfb_loss: loss_over_all, tfb_accuracy: acc})
#         graph_writer.add_summary(summary, e)
        
        end_time_e = time.time()
        time_dif_e = end_time_e - start_time_e
        e_time = "Time usage: " + str(timedelta(seconds=int(round(time_dif_e))))
        print_to_write(attampt_name, e_time, 'a')
        print(e_time)
        print('-------------------------------------------------------------------------')
    save_model(20)
    
    #########################################################################################################    
    # Print the time-usage.
    end_time = time.time()
    time_dif = end_time - start_time
    total_time = "Total Trainning Time usage: " + str(timedelta(seconds=int(round(time_dif))))
    
    print_to_write(attampt_name, total_time, 'a')
    print(total_time)


#############################################################
# Start model training and generate performance report
#############################################################
if __name__ == "__main__":
    
    print_to_write(attampt_name, attampt_name, 'w+')
    print_to_write(attampt_name, '--------------------------- Starting Training ---------------------------', 'a')
    print(attampt_name)
    print('--------------------------- Starting Training ---------------------------')
    print_test_result()
    myOptimizing()
    print('------------------------------- Complete --------------------------------')
    sess.close()

    








