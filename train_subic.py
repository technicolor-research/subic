# Copyright (c) 2017 Thomson Licensing. All rights reserved.
# This document contains Thomson Licensing proprietary and confidential information
# and trade secret. Passing on and copying of this document, use, extraction and 
# communication of its contents, is permitted under the license agreement enclosed 
# in this program. Thomson Licensing is a company of the group TECHNICOLOR.

import argparse
import os
import time
import tensorflow as tf
from ReadDataset import Dataset
from tf_nets import *
from tf_utility import *


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--max_steps', type=int, default=5000, help='Number of iterations')
parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
parser.add_argument('--img_path', default='', help='File with image paths and labels')
parser.add_argument('--nclass', type=int, help='Number of classes')
parser.add_argument('--m', type=int, default=8, help='Number of blocks')
parser.add_argument('--k', type=int, default=256, help='block dimension')
parser.add_argument('--be_wt', type=float, default=1.0, help='Weight for batch entropy loss.')
parser.add_argument('--e_wt', type=float, default=1.0, help='Weight for mean entropy loss.')
parser.add_argument('--log_dir', default='log_subic/', help='Directory to put the trained model and training logs.')
parser.add_argument('--gpu', type=int, default=0, help='GPU id.')
parser.add_argument('--pretrained', default='no', help='Pretrained')
parser.add_argument('--skip_last', type=int, default=2, help='number of layer to finetune')
parser.add_argument('--finetune', type=int, default=0, help='number of layer to finetune')
parser.add_argument('--name', default='', help='A prefix to the model name')

args = parser.parse_args()
be_wt = args.be_wt
e_wt = args.e_wt
LOG_DIR = args.log_dir
k = args.k
m = args.m
name = args.name

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

bsize = args.batch_size
max_entropy = m*np.log2(k)


with tf.Graph().as_default():
# Prepare data and model
  data = Dataset(args.img_path)
  net = VGG_M_128()
  INPUT_SHAPE = (bsize,) + net.IMAGE_SHAPE
  input_pl = tf.placeholder(tf.float32, shape= INPUT_SHAPE)
  labels_pl = tf.placeholder(tf.int32, (bsize))

  nclass = args.nclass 
  c_subic = CLASSIFY_SUBIC(net, nclass,  m, k)

  logits = c_subic.inference(input_pl)
  sb_features = c_subic.sb_features

# Entropy losses
  sb_codes = tf.split(sb_features, num_or_size_splits=m, axis=1)
  sb_entropy = [None]*m
  sb_b = [None]*m
  sb_batch_ent = [None]*m
  batch_entropy = 0
  ind_entropy = 0

  for i in range(m):
    sb_entropy[i] = entropy(sb_codes[i])
    sb_b[i] = (tf.reduce_sum(sb_codes[i], 0)/tf.cast(tf.shape(sb_codes[i])[0], tf.float32)) + tf.constant(1e-30, dtype=tf.float32)
    sb_batch_ent[i] = -tf.reduce_sum(tf.multiply(sb_b[i], tf.log(sb_b[i])))/tf.log(tf.cast(2, tf.float32))
    batch_entropy += sb_batch_ent[i]
    ind_entropy += sb_entropy[i]

  batch_entropy = batch_entropy/max_entropy
  ind_entropy = ind_entropy/max_entropy

# Loss
  loss_cl = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_pl))/ np.log(nclass)
  acc1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels_pl, 1), tf.float32))
  acc5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels_pl, 5), tf.float32))

  loss =  loss_cl + e_wt*ind_entropy - be_wt*batch_entropy

# Plots for Tensorboard
  tf.summary.scalar('Loss_Entropy_batch', -batch_entropy)
  tf.summary.scalar('Loss_Entropy_mean', ind_entropy)
  tf.summary.scalar('Loss_classification', loss_cl)
  tf.summary.scalar('Loss_total', loss)
  tf.summary.scalar('accuracy_1', acc1)
  tf.summary.scalar('accuracy_5', acc5)
  for i in range(m):
    tf.summary.scalar('Entropy_sb'+str(i+1), sb_entropy[i])
    tf.summary.scalar('BatchEntropy_sb'+str(i+1), sb_batch_ent[i])
    tf.summary.scalar('uniqueBins_sb'+str(i+1), unique_count(tf.argmax(sb_codes[i], 1)))

  model_name = name +'subic_m'+str(m) + '_k'+ str(k) + '_be' + str(be_wt) + '_me' + str(e_wt)+ '_lr'+str(args.learning_rate)

# Set trainable variables and initialze nework
  if(args.finetune>0):
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[-2*args.finetune:]
    print ("Learning only:")
    for i in var_list:
       print (i.name)
    train_op = training(loss, args.learning_rate, solver='adam', var_list=var_list)
    model_name = model_name + '_lyr' + str(args.finetune)
  else:
    train_op = training(loss, args.learning_rate, solver='adam')

  summary = tf.summary.merge_all()
  init = tf.global_variables_initializer()
  saver = tf.train.Saver(max_to_keep=0)
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  sess.run(init)
  if(args.pretrained!='no'):
    print ("loading pretrained net")
    if(args.pretrained.endswith('.npy')):
      c_subic.set_params(np.load(args.pretrained).item(), c_subic.nlayers-args.skip_last, sess, "")
    else: ### Full model is restored, skip_last is not used here.
      saver.restore(sess, args.pretrained)

  summary_writer = tf.summary.FileWriter(LOG_DIR+model_name, sess.graph)

# Training
  start_time = time.time()
  for step in xrange(args.max_steps):
    inp, labels = data.next_batch(bsize)
    feed_dict = {input_pl: inp, labels_pl: labels}
    if(step%100==0):
      duration = time.time() - start_time
      _, loss_value, lcl, batch_ent, i_ent, a1, a5, summary_str= sess.run([train_op, loss, loss_cl, batch_entropy, ind_entropy, acc1, acc5, summary], feed_dict = feed_dict)
      print('Step %d: loss = %.2f, loss_cl=%f, b_ent=%f, i_ent=%f, acc1 = %2.2f, acc5 = %2.2f, (%.3f sec)' % (step, loss_value, lcl, batch_ent, i_ent, 100*a1, 100*a5, duration))
      start_time = time.time()
      summary_writer.add_summary(summary_str, step)
      summary_writer.flush()
    else:
     _ = sess.run([train_op], feed_dict = feed_dict)

    if ((step+1) % 5000 ==0):
      print ('Saving model '+LOG_DIR+model_name+'_iter'+str(step))
      saver.save(sess, LOG_DIR+model_name+'_iter'+str(step))

