# Copyright (c) 2017 Thomson Licensing. All rights reserved.
# This document contains Thomson Licensing proprietary and confidential information
# and trade secret. Passing on and copying of this document, use, extraction and 
# communication of its contents, is permitted under the license agreement enclosed 
# in this program. Thomson Licensing is a company of the group TECHNICOLOR.

import numpy as np
from utility import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('param_file', default='subic_imagenet_m8_k256_be1_me1_lr1e-4_5k.npy', help='param file')
parser.add_argument('net', default='VGG_M_128', help='Base CNN')
parser.add_argument('--wnet', default='SUBIC', help='wrapper net')
parser.add_argument('--m', type=int, default=8, help='m')
parser.add_argument('--k', type=int, default=256, help='k')
parser.add_argument('--gpu', type=int, default=0, help='gpu device')
parser.add_argument('--testset', default='pascalvoc', help='Target dataset')
parser.add_argument('--nclass', type=int, default=8, help='Number of classes')
parser.add_argument('--split', type=int, default=1000, help='Number of queries')
parser.add_argument('--N', type=int, help='dataset size')
parser.add_argument('--img_path', default='', help='File with image paths and labels')
parser.add_argument('--frac', type=float, default=0.5, help='gpu fraction')

args =  parser.parse_args()
m = args.m
k = args.k
param_file = args.param_file
net = args.net
wnet = args.wnet
gpu = args.gpu
dataset = args.testset
gpu_frac = args.frac

def split_dataset(testx, testy, split, nclass=10):
  ty = testy.astype(int)
  frac = split/nclass
  split = frac*nclass
  Dx = []; Dy=[]
  Qx = []; Qy=[]
  count = np.zeros((nclass,), dtype=int)
  for i in range(ty.shape[0]):
    if(count[ty[i]]<frac):
      Qx.append(testx[i])
      Qy.append(testy[i])
      count[ty[i]] += 1
    else:
      Dx.append(testx[i])
      Dy.append(testy[i])
  return np.concatenate((Qx, Dx), axis=0), np.concatenate((Qy, Dy), axis=0), split

split=0
mlabel = False; ldim=1
shape = (224,224,3)
if(dataset.startswith('pascalvoc')):
  nclass=20
  split=1000
  N = 9963
  mlabel = True; ldim=20
  data_file = ''
  if(dataset.endswith('features')):
    data_file = ['VOC_9963_vgg128_X.npy', 'VOC_9963_vgg128_Y_01.npy']
elif(dataset.startswith('caltech101')):
  nclass = 100 ## The two face classes are merged and background class is rejected. 
  split = 1000
  N = 8677
  data_file = 'caltech_images'
  if(dataset.endswith('features')):
    data_file = ['caltech_features_path', 'caltech_labels_path']
elif(dataset.startswith('imagenetval')):
  nclass = 1000
  split = 2000
  N=50000
else:
  nclass = args.nclass
  split = args.split
  N = args.N
  data_file = args.img_path
  
testx, testy = get_subic(dataset, data_file, wnet, net, m, k, param_file, N, shape, gpu=gpu, gpu_frac=gpu_frac, ldim=ldim)
if(not mlabel): # reordering testx and testy as [query_set, database] with equal number of queries per class. 
  testx, testy, split = split_dataset(testx, testy, split, nclass)
x = testx[split:]
q = testx[:split]
res, dis  = retrieve(m, k, x, q, testy[:split], testy[split:], multilabel=mlabel)


