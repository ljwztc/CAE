# -*- coding:utf-8 -*-
#  created by JAY
#  the main part of the convolutional autoencoder

import tensorflow as tf
import os
import glob
import random
import numpy as np
import cv2
import time

###############################################
#  some parameter
epochs = 1000
batch_size = 20
display_number = 2
store_number = 5
train0_file_path = "./data/0_for_train/"
train1_file_path = "./data/1_for_train/"
test0_file_path = "./data/0_for_test/"
test1_file_path = "./data/1_for_test/"


################################################
#  train_data
def data_parameter():
    data_path0 = os.path.join(train0_file_path, '*.jpg')
    data0 = glob.glob(data_path0)
    data_path1 = os.path.join(train1_file_path, '*.jpg')
    data1 = glob.glob(data_path1)
    data = data0 + data1
    np.random.shuffle(data)
    return data, len(data)


def data_getbatch(path_array, now_index):
    batch = []
    for ii in range(batch_size):
        image = cv2.imread(path_array[ii + now_index * batch_size])
        float_image = image.astype(np.float32)
        # print float_image.dtype
        batch.append(float_image)
    return batch

################################################
#  network frame

#  the parameter of image
height = 160
width = 240

input = tf.placeholder(tf.float32, (None, height, width, 3))

#  Encoder
conv1 = tf.layers.conv2d(inputs=input,
                         filters=32,
                         kernel_size=(3, 3),
                         padding='same',
                         activation=tf.nn.relu)
#  240*160*32

maxpool1 = tf.layers.max_pooling2d(inputs=conv1,
                                   pool_size=(5, 5),
                                   strides=(5, 5),
                                   padding='same')
#  48*32*32

conv2 = tf.layers.conv2d(inputs=maxpool1,
                         filters=64,
                         kernel_size=(3, 3),
                         padding='same',
                         activation=tf.nn.relu)
#  48*32*64

maxpool2 = tf.layers.max_pooling2d(inputs=conv2,
                                   pool_size=(4, 4),
                                   strides=(4, 4),
                                   padding='same')
#  12*8*64

conv3 = tf.layers.conv2d(inputs=maxpool2,
                         filters=16,
                         kernel_size=(3, 3),
                         padding='same',
                         activation=tf.nn.relu)
#  12*8*16

encoded = tf.layers.dense(inputs=conv3,
                          units=256,
                          activation=tf.nn.relu)
#  256

# Decoder
decoded = tf.layers.dense(inputs=encoded,
                          units=12*8*16,
                          activation=tf.nn.relu)
#  12*8*16 series

shaped = tf.reshape(decoded,
                    shape=[-1, 8, 12, 16])
#  12*8*16

deconv3 = tf.layers.conv2d(inputs=shaped,
                           filters=64,
                           kernel_size=(3, 3),
                           padding='same',
                           activation=tf.nn.relu)
#  12*8*64

upsample2 = tf.image.resize_images(deconv3,
                                   size=(32, 48),
                                   method=tf.image.ResizeMethod.BILINEAR)
#  48*32*64

deconv2 = tf.layers.conv2d(inputs=upsample2,
                           filters=32,
                           kernel_size=(3, 3),
                           padding='same',
                           activation=tf.nn.relu)
#  48*32*32

upsample3 = tf.image.resize_images(deconv2,
                                   size=(160, 240),
                                   method=tf.image.ResizeMethod.BILINEAR)
#  240*160*32

reconstruction = tf.layers.conv2d(inputs=upsample3,
                                  filters=3,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  activation=tf.nn.sigmoid)
#  240*160*3

loss = tf.nn.l2_loss(reconstruction - input)

learning_rate = 0.001
cost = tf.reduce_mean(loss)
opt = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
# or AdamOptimizer? have a try later

#  training parameter
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print "variable initialize success!"
    saver = tf.train.Saver()
    filepath = ".\CAE.ckpt"
    if os.path.isfile(filepath + ".meta"):
        saver.restore(sess, filepath)
        print "restore success!"
    else:
        print "train from the beginning!"
    for i in range(epochs):
        image_path, datanumber = data_parameter()           # each time we generate the batch is random
        print "read data success!"
        start_time = time.time()
        for ii in range(datanumber / batch_size):
            batch = data_getbatch(image_path, ii)
            batch_cost, _ = sess.run((cost, opt), feed_dict={input: batch})
            print "batch {} trained success!".format(ii)
        one_batch_time = time.time() - start_time
        print "the cost of this epochs: {}".format(one_batch_time)
        if i % display_number == 0:
            print "Epoch: {} of {}".format(i, epochs) + '\n' + "Training loss: {:.5f}".format(batch_cost)
        if i % store_number == 0:
            save_path = saver.save(sess=sess, save_path=filepath)
            print "Model saced in file: %s" % filepath



