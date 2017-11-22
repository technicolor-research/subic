# Copyright (c) 2017 Thomson Licensing. All rights reserved.
# This document contains Thomson Licensing proprietary and confidential information
# and trade secret. Passing on and copying of this document, use, extraction and 
# communication of its contents, is permitted under the license agreement enclosed 
# in this program. Thomson Licensing is a company of the group TECHNICOLOR.

import os
import numpy as np
import cv2
import threading

def job_lists(njobs, nt):
  start_id = np.zeros((nt+1,))
  x = njobs%nt
  f = njobs//nt
  for i in range(nt):
    start_id[i+1]=start_id[i] + f + (np.sum(x>0))
    x-=np.sum(x>0)
  return start_id

def load_images(paths, shape, mean=[103.939, 116.779, 123.68], crop=1, mirroring=False, nt=8):
  '''
  crop=[<0, 0, >0]: [random crop, no cropping, center crop]
  nt: number of threads
  '''
  n = len(paths)
  img = [None]*n
  images = np.zeros(((n,)+shape))
  mirror = np.zeros((n,), dtype=int)
  if(mirroring):
    mirror = np.random.randint(2, size=n)
  def preprocess(idx):
    ids = np.arange(start_id[idx], start_id[idx+1], dtype=int)
    if(crop==0):
      for i in ids:
        images[i] = cv2.resize(cv2.imread(paths[i]).astype(np.float32), shape[0:2]) - mean
        if(mirror[i]==1):
          images[i] = cv2.flip(images[i], 1)
    else:
      for i in ids:
        img[i] = cv2.imread(paths[i]).astype(np.float32)
        short_edge = np.min(img[i].shape[:2])
        resize_sp = ( (256*img[i].shape[1])//short_edge, (256*img[i].shape[0])//short_edge )
        resize_img = cv2.resize(img[i], resize_sp)
        if(crop<0): ### Random crop
          yy = np.random.randint(256-shape[1])
          xx = np.random.randint(256-shape[0])
        else: # center crop
          yy = int((resize_img.shape[0] - 256) / 2)+ int((256 - shape[0])/2)
          xx = int((resize_img.shape[1] - 256) / 2)+ int((256 - shape[1])/2)
        images[i] = resize_img[yy: yy + shape[1], xx: xx + shape[0]] - mean
        if(mirror[i]==1):
          images[i] = cv2.flip(images[i], 1)
  threads = []
  start_id = job_lists(n, nt)
  for i in range(nt):
    p_thread = threading.Thread(target=preprocess, args=([i]))
    threads.append(p_thread)
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  return images


class Dataset_features():
  def __init__(self, filex, filey, counter=0):
    self.images = np.load(filex)
    self.nimages = self.images.shape[0]
    if(filex.find('paris')==-1 or filex.find('oxford')==-1):
      self.labels = np.load(filey)
    else: ## no labels for paris and oxford
      self.labels = np.zeros((self.nimages,), dtype=int)
    self.nimages = self.images.shape[0]
    self.imgshape = (self.images.shape[1],)
    self.counter = counter
  def next_batch(self, bsize):
    index = np.random.randint(self.nimages, size=bsize)
    return self.images[index], self.labels[index]
  def next_batch_seq(self, bsize):
    bsize = int(bsize)
    if (self.counter>self.nimages):
      return None, None
    st = self.counter
    self.counter += bsize
    if(self.counter>self.nimages):
      print("Warning: Dataset has only %d images remaining, this batch of %d is padded with %d repeated images" %(self.nimages-st, bsize, self.counter-self.nimages))
      index = np.concatenate([np.arange(st, self.nimages), np.zeros((self.counter-self.nimages,))]).astype('int')
    else:
      index = np.arange(st, self.counter)
    return self.images[index], self.labels[index]
     
class Dataset_images():
  def __init__(self, data_file, shape=(224, 224, 3), mean=[103.939, 116.779, 123.68], counter=0):
    self.mean_img = mean
    self.counter = counter
    self.datapaths = open(data_file).readlines()
    self.nimages = len(self.datapaths)
    img_paths = [None]*self.nimages
    self.labels = np.zeros((self.nimages,))
    for i in range(self.nimages):
      img_paths[i], self.labels[i] = self.datapaths[i].split()
    self.img_paths = np.array(img_paths)
    self.imgshape = shape
  def next_batch(self, bsize, nt=4):
    index = np.random.randint(self.nimages, size=int(bsize))
    return load_images(self.img_paths[index], self.imgshape, mean=self.mean_img, crop=1, mirroring=False, nt=nt), self.labels[index]
  def next_batch_seq(self, bsize, nt=4):
    bsize = int(bsize)
    if (self.counter>self.nimages):
      return None, None
    st = self.counter
    self.counter += bsize
    if(self.counter>self.nimages):
      print("Warning: Dataset has only %d images remaining, this batch of %d is padded with %d repeated images" %(self.nimages-st, bsize, self.counter-self.nimages))
      index = np.concatenate([np.arange(st, self.nimages), np.zeros((self.counter-self.nimages,))]).astype('int')
    else:
      index = np.arange(st, self.counter)
    return load_images(self.img_paths[index], self.imgshape, mean=self.mean_img, crop=1, nt=nt), self.labels[index]
 
def Dataset(data_file, dataset_type='images', shape=(224,224,3), counter=0, mean=[103.939, 116.779, 123.68]):
  if(dataset_type.endswith('features')):
    return Dataset_features(data_file[0], data_file[1], counter)
  else:
    return Dataset_images(data_file, shape=shape, mean=mean, counter=counter) 
   
