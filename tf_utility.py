# Copyright (c) 2017 Thomson Licensing. All rights reserved.
# This document contains Thomson Licensing proprietary and confidential information
# and trade secret. Passing on and copying of this document, use, extraction and 
# communication of its contents, is permitted under the license agreement enclosed 
# in this program. Thomson Licensing is a company of the group TECHNICOLOR.

import tensorflow as tf
import numpy as np
import os

def training(loss, learning_rate, solver='adam', var_list=[]):
  if(solver=='adam'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif (solver=='sgd'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    print("Error in training(): Solver not found")
    return
  global_step = tf.Variable(0, name='global_step', trainable=False)
  if(len(var_list)==0):
    train_op = optimizer.minimize(loss, global_step=global_step)
  else:
    train_op = optimizer.minimize(loss, global_step=global_step, var_list=var_list)
  return train_op

def entropy(ivf_vecs):
  zero = tf.constant(1e-30, dtype=tf.float32)
  ivf_vecs+=zero;
  entropy = -tf.reduce_sum(tf.multiply(ivf_vecs, tf.log(ivf_vecs))) / (tf.cast(tf.shape(ivf_vecs)[0], tf.float32) * tf.log(tf.cast(2,tf.float32)))
  return entropy

def unique_count(a):
  return tf.shape(tf.unique(a).y)[0]

def ext_features(net, param_file, data, N, bsize=100, seq=False, label_dim=1, gpu=0, frac=0.5):
  os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
  images_pl = tf.placeholder(tf.float32, shape=(bsize,)+data.imgshape)
  feat = net.inference(images_pl)
  dataX = np.zeros((N, net.output_size))
  if(label_dim==1):
    labels = np.zeros((N,))
  else:
    labels = np.zeros((N, label_dim))
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=frac)
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  sess.run(tf.global_variables_initializer())
  if(param_file.endswith('.npy')):
    net_params = np.load(param_file).item()
    net.set_params(net_params, net.nlayers, sess)
  else:
    saver = tf.train.Saver()
    saver.restore(sess, param_file)
  print("Weights loaded.")
  nbatch = int(N/bsize);
  for i in range(nbatch):
    if(seq):
      images, labels[i*bsize:(i+1)*bsize] = data.next_batch_seq(bsize)
    else:
      images, labels[i*bsize:(i+1)*bsize] = data.next_batch(bsize)
    feed = {images_pl: images}
    dataX[i*bsize:(i+1)*bsize] = sess.run(feat, feed_dict = feed)
  sess.close()
  return dataX, labels

