# -*- coding: utf-8 -*-

"""
import cv2

imagepath = "/Users/mac/Desktop/data/1_for_test/2.jpg"
imagename = "02.jpg"
image = cv2.imread(imagepath)

image = image[66:376, 88:568]

cv2.imwrite(imagepath, image)



import os
import cv2

dir = "/Users/mac/Desktop/data/0_for_test"  # chage this directory

for filename in os.listdir(dir):
    imagepath = "/Users/mac/Desktop/data/1_for_test/" + filename  # there is also a directory
    image = cv2.imread(imagepath)
    image = image[66:376, 88:568]
    cv2.imwrite(imagepath, image)



import tensorflow as tf

x1 = tf.constant(1.0, shape=[1, 3, 3, 1])

x2 = tf.constant(1.0, shape=[1, 6, 6, 3])

x3 = tf.constant(1.0, shape=[1, 5, 5, 3])

kernel = tf.constant(1.0, shape=[3, 3, 3, 1])

y1 = tf.nn.conv2d_transpose(x1, kernel, output_shape=[1, 6, 6, 3],
                            strides=[1, 2, 2, 1], padding="SAME")

y2 = tf.nn.conv2d(x3, kernel, strides=[1, 2, 2, 1], padding="SAME")

y3 = tf.nn.conv2d_transpose(y2, kernel, output_shape=[1, 5, 5, 3],
                            strides=[1, 2, 2, 1], padding="SAME")

y4 = tf.nn.conv2d(x2, kernel, strides=[1, 2, 2, 1], padding="SAME")

'''
Wrong!!This is impossible
y5 = tf.nn.conv2d_transpose(x1,kernel,output_shape=[1,10,10,3],strides=[1,2,2,1],padding="SAME")
'''
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
x1_decov, x3_cov, y2_decov, x2_cov, x3out = sess.run([y1, y2, y3, y4, x3])
print(x1_decov.shape)
print(x3_cov.shape)
print(y2_decov.shape)
print(x2_cov.shape)
print y2_decov
print x3out

"""

import os
from glob import glob
import random

batch_size = 100

train0_file_path = "./data/0_for_train/"
train1_file_path = "./data/1_for_train/"
test0_file_path = "./data/0_for_test/"
test1_file_path = "./data/1_for_test/"

data_path0 = os.path.join(train0_file_path, '*.jpg')
data0 = glob(data_path0)
data_path1 = os.path.join(train1_file_path, '*.jpg')
data1 = glob(data_path1)
data = data0 + data1

data_number = len(data)
rand_array = random.sample(range(0, data_number + 1), data_number)
batch_path = []
for i in range(data_number/batch_size):
    batch_path.append(data[i * batch_size: (i + 1) * batch_size])

print [data_number, len(batch_path), len(batch_path[1])]
print batch_path[0]
print batch_path[1]
print data[199]
