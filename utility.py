# Copyright (c) 2017 Thomson Licensing. All rights reserved.
# This document contains Thomson Licensing proprietary and confidential information
# and trade secret. Passing on and copying of this document, use, extraction and 
# communication of its contents, is permitted under the license agreement enclosed 
# in this program. Thomson Licensing is a company of the group TECHNICOLOR.

from tf_nets import *
from tf_utility import ext_features
from ReadDataset import Dataset
import numpy as np


def get_net(net):
  if(net=='VGG_M_128'):
    return VGG_M_128()
  elif(net=='VGG_M_128_dummy'):
    return VGG_M_128_dummy()
  else:
    return None

def wrapper_net(wnet, net, m=8, k=256):
  if(wnet=='SUBIC'):
    return SUBIC(get_net(net), m, k, no_softmax=True)
  else:
    return get_net(net)

def get_subic(dataset, data_file, wnet, net, m, k, param_file, nsize, shape, gpu, gpu_frac, ldim=1):
  bsize=100
  TX, TY = ext_features(wrapper_net(wnet, net, m, k), param_file, Dataset(data_file, dataset), bsize*np.int(np.ceil(float(nsize)/bsize)), bsize=bsize, seq=True, label_dim=ldim, gpu=gpu, frac=gpu_frac)
  return TX[:nsize].astype('float32'), TY[:nsize]  ## Only first nsize images are used.

def retrieve(m, k, x, q, labelq, labelx, multilabel=False):
  code = np.zeros((m, x.shape[0]), dtype=int)
  approx_dis =  np.zeros((q.shape[0], code.shape[1]))
  for i in range(m):
    code[i,:] = np.argmax(x[:,i*k:(i+1)*k], axis=1)
    for j in range(q.shape[0]):
      approx_dis[j] += q[j, i*k+code[i]]
  qres = np.argsort(-approx_dis, axis=1)
  print ("mAP=%f" % get_results(qres, labelq, labelx, multilabel))
  return qres, approx_dis

def get_results(res, labelq, labelx, multilabel=True, n=21):
  if(multilabel):
    pr, rec = prec_recall_multiclass(res, labelq, labelx)
  else:
    pr, rec = prec_recall(res, labelq, labelx)
  ap = 0
  for i in range(labelq.shape[0]):
    ap += compute_ap(pr[i], rec[i], n)
  return ap/labelq.shape[0]

def compute_ap(prec, rec, n):
  ap=0;
  for t in np.linspace(0,1,n):
    all_p=prec[rec>=t];
    if all_p.size==0: p=0;
    else: p=all_p.max()
    ap=1.0*ap+1.0*p/n;
  return ap

def prec_recall(res, labelq, labelx):
  prec = np.zeros_like(res, dtype=float)
  recall = np.zeros_like(res, dtype=float)
  n = np.arange(res.shape[1])+1.0
  for i in range(res.shape[0]):
    x = np.cumsum((labelx[res[i]]==labelq[i]).astype(float))
    prec[i] = x/n
    recall[i] = x/x[-1]
  return prec, recall

def prec_recall_multiclass(res, labelq, labelx):
  prec = np.zeros_like(res, dtype=float)
  recall = np.zeros_like(res, dtype=float)
  n = np.arange(res.shape[1])+1.0
  for i in range(res.shape[0]):
    x = np.cumsum((labelx[res[i]].dot(labelq[i])>0).astype(float))
    prec[i] = x/n
    recall[i] = x/x[-1]
  return prec, recall


