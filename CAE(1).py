# -*- coding: utf-8 -*-
"""
Created in ubuntu. 
Copyright @ jay. All rights reserved.
"""

import os
import re
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob



checkpoint_dir = "./ckpt/"
model_name = "ConvAutoEnc.model"
logs_dir = "./logs/run1/"

# Fetch input data (faces/trees/imgs)
data_dir0 = "./0_new/"
data_dir1 = "./1_new/"
data_path0 = os.path.join(data_dir0, '*.jpg')
data0 = glob(data_path0)
data_path1 = os.path.join(data_dir1, '*.jpg')
data1 = glob(data_path1)
data = data1 + data0

# Some important consts
num_examples = len(data)
batch_size = 1024
n_epochs = 3000
save_steps = 10  # Number of training batches between checkpoint saves
total_cost = []

'''
Some util functions from https://github.com/carpedm20/DCGAN-tensorflow
'''

def path_to_img(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=32, resize_width=48, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def autoresize(image_path, input_height, input_width,
              resize_height=32, resize_width=48,
              crop=True, grayscale=False):
  image = path_to_img(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

np.random.shuffle(data)


'''
tf Graph Input
'''
x = tf.placeholder(tf.float32, [None, 32, 48, 3], name='InputData')

# strides = [Batch, Height, Width, Channels]  in default NHWC data_format. Batch and Channels
# must always be set to 1. If channels is set to 3, then we would increment the index for the
# color channel by 3 everytime we convolve the filter. So this means we would only use one of
# the channels and skip the other two. If we change the Batch number then it means some images
# in the batch are skipped.
#
# To calculate the size of the output of CONV layer:
# OutWidth = (InWidth - FilterWidth + 2*Padding)/Stride + 1
def conv2d(input, name, kshape, strides=[1, 1, 1, 1]):
    with tf.variable_scope(name):
        W = tf.get_variable(name='w_' + name,
                            shape=kshape,
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b_' + name,
                            shape=[kshape[3]],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        out = tf.nn.conv2d(input,W,strides=strides, padding='SAME')
        out = tf.nn.bias_add(out, b)
        out = tf.nn.leaky_relu(out)
        return out


# tf.contrib.layers.conv2d_transpose, do not get confused with 
# tf.layers.conv2d_transpose
def deconv2d(input, name, kshape, n_outputs, strides=[1, 1]):
    with tf.variable_scope(name):
        out = tf.contrib.layers.conv2d_transpose(input,
                 num_outputs= n_outputs,
                 kernel_size=kshape,
                 stride=strides,
                 padding='SAME',
                 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                 biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                 activation_fn=tf.nn.leaky_relu)
        return out


# Input to maxpool: [BatchSize, Width1, Height1, Channels]
# Output of maxpool: [BatchSize, Width2, Height2, Channels]
#
# To calculate the size of the output of maxpool layer:
# OutWidth = (InWidth - FilterWidth)/Stride + 1
#
# The kernel kshape will typically be [1,2,2,1] for a general 
# RGB image input of [batch_size,48,48,3]
# kshape is 1 for batch and channels because we don't want to take
# the maximum over multiple examples of channels.
def maxpool2d(x,name,kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    with tf.variable_scope(name):
        out = tf.nn.max_pool(x,
                 ksize=kshape, #size of window
                 strides=strides,
                 padding='SAME')
        return out


def upsample(input, name, factor=[2,2]):
    size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
    with tf.variable_scope(name):
        out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
        return out


def fullyConnected(input, name, output_size):
    with tf.variable_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size)) # get total num of cells in one input image
        W = tf.get_variable(name='w_'+name,
                shape=[input_size, output_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b_'+name,
                shape=[output_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        input = tf.reshape(input, [-1, input_size])
        out = tf.nn.leaky_relu(tf.add(tf.matmul(input, W), b))
        return out


def dropout(input, name, keep_rate):
    with tf.variable_scope(name):
        out = tf.nn.dropout(input, keep_rate)
        return out


def ConvAutoEncoder(x, name, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        input = tf.reshape(x, shape=[-1, 32, 48, 3])

        # kshape = [k_h, k_w, in_channels, out_chnnels]
        c1 = conv2d(input, name='c1', kshape=[3, 3, 3, 15])
        p1 = maxpool2d(c1, name='p1')
        do1 = dropout(p1, name='do1', keep_rate=0.75)
        c2 = conv2d(do1, name='c2', kshape=[3, 3, 15, 32])
        p2 = maxpool2d(c2, name='p2')
        do2 = dropout(p2, name='do2', keep_rate=0.75)
        c3 = conv2d(do2, name='c3', kshape=[3, 3, 32, 64])
        p3 = maxpool2d(c3, name='p3')
        p4 = tf.reshape(p3, shape=[-1, 4*6*64])
        fc3 = fullyConnected(p4, name='fc3', output_size=128)
        fc4 = fullyConnected(fc3, name='fc4', output_size=4*6*64)
        do3 = dropout(fc4, name='do3', keep_rate=0.75)
        do4 = tf.reshape(do3, shape=[-1, 4, 6, 64])
        dc1 = deconv2d(do4, name='dc1', kshape=[3, 3],n_outputs=32)
        up1 = upsample(dc1, name='up1', factor=[2, 2])
        dc2 = deconv2d(up1, name='dc2', kshape=[3, 3],n_outputs=15)
        up2 = upsample(dc2, name='up2', factor=[2, 2])
        dc3 = deconv2d(up2, name='dc3', kshape=[3, 3],n_outputs=3)
        up3 = upsample(dc3, name='up3', factor=[2, 2])
        output = fullyConnected(up3, name='output', output_size=32*48*3)

        with tf.variable_scope('cost'):
            # N.B. reduce_mean is a batch operation! finds the mean across the batch
            cost = tf.reduce_mean(tf.square(tf.subtract(output, tf.reshape(x,shape=[-1,32*48*3]))))
        return x, tf.reshape(output,shape=[-1,32,48,3]), cost # returning, input, output and cost


# Create checkpoint
def save(saver, step, session):
    print(">>> Saving to checkpoint, step:" + str(step))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(session,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)


# Restore from checkpoint
def restore(saver, session):
    print(">>> Restoring from checkpoints...")
    checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint_state and checkpoint_state.model_checkpoint_path:
      checkpoint_name = os.path.basename(checkpoint_state.model_checkpoint_path)
      saver.restore(session, os.path.join(checkpoint_dir, checkpoint_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",checkpoint_name)).group(0))
      print(">>> Found restore checkpoint {}".format(checkpoint_name))
      return True, counter
    else:
      return False, 0



with tf.Session() as sess:
    print ">>>>>1"
    _, _, cost = ConvAutoEncoder(x, 'ConvAutoEnc')
    print ">>>>>2"
    with tf.variable_scope('opt'):
        optimizer = tf.train.AdamOptimizer().minimize(cost)
    print ">>>>>3"
    # Create a summary to monitor cost tensor
    tf.summary.scalar("cost", cost)
    print ">>>>>4"
    tf.summary.image("face_input", ConvAutoEncoder(x, 'ConvAutoEnc', reuse=True)[0], max_outputs=4)
    tf.summary.image("face_output", ConvAutoEncoder(x, 'ConvAutoEnc', reuse=True)[1], max_outputs=4)
    merged_summary_op = tf.summary.merge_all()  # Merge all summaries into a single op
    print ">>>>>5"
    sess.run(tf.global_variables_initializer())  # memory allocation exceeded 10% issue
    print ">>>>>6"
    # Model saver
    saver = tf.train.Saver()
    print ">>>>>7"
    counter = 0  # Used for checkpointing
    success, restored_counter = restore(saver, sess)
    if success:
        counter = restored_counter
        print(">>> Restore successful")
    else:
        print(">>> No restore checkpoints detected")        
    print ">>>>>8"
    # create log writer object
    writer = tf.summary.FileWriter(logs_dir, graph=tf.get_default_graph())
    print ">>>>>9"
    for epoch in range(n_epochs):
        avg_cost = 0
        n_batches = int(num_examples / batch_size)
        # Loop over all batches
        for i in range(n_batches):
            counter += 1
            print("epoch " + str(epoch) + " batch " + str(i))

            batch_files = data[i*batch_size:(i+1)*batch_size]  # get the current batch of files

            batch = [autoresize(batch_file,
                                    input_height=32,
                                    input_width=48,
                                    resize_height=32,
                                    resize_width=48,
                                    crop=True,
                                    grayscale=False) for batch_file in batch_files]

            batch_images = np.array(batch).astype(np.float32)

            # Get cost function from running optimizer
            print ">>> {x: batch_images}"
            print {x: batch_images.shape}
            _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={x: batch_images})

            # Compute average loss
            avg_cost += c / n_batches

            writer.add_summary(summary, epoch * n_batches + i)

            if counter % save_steps == 0:
                save(saver, counter, sess)

        # Display logs per epoch step
        print('Epoch', epoch + 1, '/' ,n_epochs, 'cost:', avg_cost)
        total_cost.append(avg_cost)
    file = open('loss.txt','w')
    for i in range(len(total_cost)):
        file.write(str(total_cost[i])+'\n')
    file.close()
    print('>>> Optimization Finished')
