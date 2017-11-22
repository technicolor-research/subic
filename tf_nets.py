# Copyright (c) 2017 Thomson Licensing. All rights reserved.
# This document contains Thomson Licensing proprietary and confidential information
# and trade secret. Passing on and copying of this document, use, extraction and 
# communication of its contents, is permitted under the license agreement enclosed 
# in this program. Thomson Licensing is a company of the group TECHNICOLOR.

import numpy as np
import tensorflow as tf

class NET():
 def __init__():
   self.nlayer=0; self.l_names=[];
 def set_params(self, net_params, nlayers, sess, scope='', id_w='weights', id_b='biases'):
   with tf.variable_scope(scope, reuse=True):
     for i in range(self.nlayers):
       if(i<nlayers):
          sess.run(tf.get_variable(self.l_names[i]+'w').assign(net_params[self.l_names[i]][id_w]))
          sess.run(tf.get_variable(self.l_names[i]+'b').assign(net_params[self.l_names[i]][id_b]))
       print('layer %d read' % (i+1))


class VGG_M_128_dummy(NET):
  def __init__(self):
    self.IMAGE_SIZE = 224; self.IMAGE_DEPTH = 3
    self.IMAGE_SHAPE = (self.IMAGE_SIZE, self.IMAGE_SIZE, self.IMAGE_DEPTH)
    self.nlayers = 7
    self.output_size = 128

class VGG_M_128(NET):
  def __init__(self):
    self.IMAGE_SIZE = 224; self.IMAGE_DEPTH = 3
    self.IMAGE_SHAPE = (self.IMAGE_SIZE, self.IMAGE_SIZE, self.IMAGE_DEPTH)
    self.nlayers = 7
    self.output_size = 128
    self.l_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
    self.conv_ksize = [[7, 7, 3, 96], [5, 5, 96, 256], [3, 3, 256, 512], [3, 3, 512, 512], [3, 3, 512, 512]]
    self.weights = {}
    self.bias = {}
    for i in range(len(self.conv_ksize)):
      self.weights[self.l_names[i]] = tf.get_variable(self.l_names[i]+'w', self.conv_ksize[i], initializer= tf.random_normal_initializer(stddev=1e-2))
      self.bias[self.l_names[i]] = tf.get_variable(self.l_names[i]+'b', [self.conv_ksize[i][3]], initializer=tf.constant_initializer(0.0))

    self.weights[self.l_names[5]] = tf.get_variable(self.l_names[5]+'w',  [18432, 4096], initializer= tf.random_normal_initializer(stddev=0.1))
    self.bias[self.l_names[5]] = tf.get_variable(self.l_names[5]+'b', [4096], initializer=tf.constant_initializer(0.0))
    self.weights[self.l_names[6]] = tf.get_variable(self.l_names[6]+'w', [4096, 128], initializer= tf.random_normal_initializer(stddev=0.1))
    self.bias[self.l_names[6]] = tf.get_variable(self.l_names[6]+'b', [128], initializer=tf.constant_initializer(0.0))
  def inference (self, images):
    pad='SAME'
    self.images = images
  # layer 1
    self.relu1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(images, self.weights[self.l_names[0]], strides=[1,2,2,1], padding='VALID'), self.bias[self.l_names[0]])) 
    radius = 2; alpha = 1e-04; beta = 0.75; bias = 1.0
    self.lrn1 = tf.nn.local_response_normalization(self.relu1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
    self.pool1 = tf.nn.max_pool(self.lrn1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool1')
  # layer 2
    self.relu2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.pool1, self.weights[self.l_names[1]], strides=[1,2,2,1], padding=pad), self.bias[self.l_names[1]]))
    radius = 2; alpha = 1e-04; beta = 0.75; bias = 1.0
    self.lrn2 = tf.nn.local_response_normalization(self.relu2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
    self.pool2 = tf.nn.max_pool(self.lrn2, ksize=[1,3,3,1], strides=[1,2,2,1], padding=pad, name='pool2')
  # layer 3-4-5
    self.relu3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.pool2, self.weights[self.l_names[2]], strides=[1,1,1,1], padding=pad), self.bias[self.l_names[2]]))
    self.relu4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.relu3, self.weights[self.l_names[3]], strides=[1,1,1,1], padding=pad), self.bias[self.l_names[3]]))
    self.relu5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.relu4, self.weights[self.l_names[4]], strides=[1,1,1,1], padding=pad), self.bias[self.l_names[4]]))
    self.pool5 = tf.nn.max_pool(self.relu5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool5')
  # fully connected
    pool5_flat = tf.reshape(self.pool5, [-1, np.int(np.prod(self.pool5.get_shape()[1:]))])
    self.fc6 = tf.nn.relu(tf.matmul(pool5_flat, self.weights[self.l_names[5]]) + self.bias[self.l_names[5]])
    self.fc7 = tf.nn.relu(tf.matmul(self.fc6, self.weights[self.l_names[6]]) + self.bias[self.l_names[6]])
    return self.fc7
  def set_params(self, net_params, nlayers, sess, scope='', id_w='weights', id_b='biases'):
    with tf.variable_scope(scope, reuse=True):
      for i in range(self.nlayers):
        if(i<nlayers):
          sess.run(tf.get_variable(self.l_names[i]+'w').assign(net_params[self.l_names[i]][id_w]))
          sess.run(tf.get_variable(self.l_names[i]+'b').assign(net_params[self.l_names[i]][id_b]))
        print('layer %d read' % (i+1))


class SUBIC(NET):
  def __init__(self, net, m, k, no_softmax=False):
    self.m = m
    self.k = k
    self.no_softmax=no_softmax
    self.NET = net
    self.output_size = m*k
    self.nlayers=self.NET.nlayers+1
    self.l_names = ['ip']
    self.weights = { 'ipw': tf.get_variable('ipw',  [self.NET.output_size, self.output_size], initializer= tf.random_normal_initializer(stddev=0.1))}
    self.bias = {'ipb': tf.get_variable('ipb',  [self.output_size], initializer=tf.constant_initializer(0.0))}
  def inference(self, input_data, no_softmax=False):
    if(len(input_data.get_shape().as_list())==4):
      self.features = self.NET.inference(input_data)
    else:
      self.features = input_data
    sb_features = tf.nn.relu(tf.matmul(self.features, self.weights['ipw']) + self.bias['ipb'])
    if(self.no_softmax or no_softmax):
      return sb_features
    if(self.m>1):
      feat = tf.split(sb_features, num_or_size_splits=self.m, axis=1)
      out = [None]*self.m
      for i in range(self.m):
        out[i] = tf.nn.softmax(feat[i])
      self.sb_features = tf.concat(out, axis=1)
    else:
      self.sb_features = tf.nn.softmax(sb_features)
    return self.sb_features
  def set_params(self, net_params, nlayers, sess, scope='', id_w='weights', id_b='biases'):
    print('%d out of %d layers' %(nlayers, self.nlayers))
    self.NET.set_params(net_params, nlayers, sess, scope, id_w, id_b)
    with tf.variable_scope(scope, reuse=True):
      if(nlayers>=self.nlayers):
        sess.run(tf.get_variable('ipw').assign(net_params[self.l_names[0]]['weights']))
        sess.run(tf.get_variable('ipb').assign(net_params[self.l_names[0]]['biases']))
        print('layer %d read' % self.nlayers)
    

class CLASSIFY_SUBIC(NET):
  def __init__(self, net, nclass, m, k):
    self.NET = net
    self.Nclass = nclass
    self.m = m
    self.k = k
    self.nlayers=self.NET.nlayers+2
    self.output_size = m*k
    self.l_names = ['ip', 'cip']
    self.weights = {}
    self.bias = {}
    self.weights['ipw'] = tf.get_variable('ipw',  [self.NET.output_size, self.output_size], initializer= tf.random_normal_initializer(stddev=0.1))
    self.bias['ipb'] = tf.get_variable('ipb',  [self.output_size], initializer=tf.constant_initializer(0.0))
    self.weights['cipw'] = tf.get_variable('cipw',  [self.output_size, self.Nclass], initializer= tf.random_normal_initializer(stddev=0.1))
    self.bias['cipb'] = tf.get_variable('cipb',  [self.Nclass], initializer=tf.constant_initializer(0.0))
  def inference(self, input_data):
    if(len(input_data.get_shape().as_list())==4):
      self.features = self.NET.inference(input_data)
    else:
      self.features = input_data
    sb_features = tf.nn.relu(tf.matmul(self.features, self.weights['ipw']) + self.bias['ipb'])
    feat = tf.split(sb_features, num_or_size_splits=self.m, axis=1)
    out = [None]*self.m
    for i in range(self.m):
      out[i] = tf.nn.softmax(feat[i])
    self.sb_features = tf.concat(out, axis=1)
    self.scores = tf.matmul(self.sb_features, self.weights['cipw']) + self.bias['cipb']
    return self.scores
  def set_params(self, net_params, nlayers, sess, scope='', id_w='weights', id_b='biases'):
    print('%d out of %d layers' %(nlayers, self.nlayers))
    self.NET.set_params(net_params, nlayers, sess, scope, id_w, id_b)

