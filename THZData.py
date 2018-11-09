# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 09:28:40 2018

@author: admin
"""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import os
import numpy as np
from PIL import Image
import extract_labels
from object_detection.utils import dataset_util
#from object_detection.utils import label_map_util
#from lxml import etree
from decode import decode
anchor_num = 1
# 根目录
#Root_path = 'THZImage'
class Dataset(object):
    def __init__(self,
               images,
               labels,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
     
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = images.shape[0]
     
    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
          perm0 = np.arange(self._num_examples)
          np.random.shuffle(perm0)
          self._images = self.images[perm0]
          self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Get the rest examples in this epoch
          rest_num_examples = self._num_examples - start
          images_rest_part = self._images[start:self._num_examples]
          labels_rest_part = self._labels[start:self._num_examples]
        # Shuffle the data
          if shuffle:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self.images[perm]
            self._labels = self.labels[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size - rest_num_examples
          end = self._index_in_epoch
          images_new_part = self._images[start:end]
          labels_new_part = self._labels[start:end]
          return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
          self._index_in_epoch += batch_size
          end = self._index_in_epoch
          return self._images[start:end], self._labels[start:end]
 
def read_all_data(Root_path = './THZDataset', year = "VOC2007",dtype=dtypes.float32):
    #root_path = Root_path + '/VOC2007/'
    #root_test_path = Root_path + '/VOC2007'
    test_images,test_labels = read_data(Root_path,year,"test")
    train_images,train_labels = read_data(Root_path,year,"train")
    
    train = Dataset(train_images,train_labels, dtype=dtype)
    test = Dataset(test_images,test_labels, dtype=dtype)
    return train,test
                     
def read_data(filedir,year,setclass = "train"):
    data_dir = filedir
    labels_filenames = []
    images_filenames = []
    
    examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main'#Layout,Main
                                  , 'gun_' + setclass + '.txt')
    annotations_dir = os.path.join(data_dir, year, "Annotations")    #FLAGS.annotations_dir
    JEPG_dir = os.path.join(data_dir, year, "JPEGImages")
    examples_list = dataset_util.read_examples_list(examples_path)
    for idx, example in enumerate(examples_list):
        label_path = os.path.join(annotations_dir, example + '.xml')
        image_path = os.path.join(JEPG_dir, example + '.jpg')
        labels_filenames.append(label_path)
        images_filenames.append(image_path)


    labels = extract_labels.labels_normalizer( labels_filenames, anchor_num )
    images = images_normalizer( images_filenames)
          
    return np.array(images), np.array(labels)


def images_normalizer(images_filenames,target_width = 416, target_height=416):
    images = []
    for imagename in images_filenames:
        img = Image.open(imagename)
        img = img.resize((target_width,target_height))
        image = np.array(img)               
        image = image/255 
        images.append(image)
    return images

#traindata,testdata= read_all_data()
#x,y = traindata.next_batch(1)
#a = y[0]




