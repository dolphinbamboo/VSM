# -*- coding: utf-8 -*-
from tools.utils import *

import random
import h5py
import cv2

"""
Created on Mon May 21 18:41:43 2018

@author: Meng Yuan

VSM code with TensorFlow-v1.4

> prepare data
"""

def h5ReaderStr(h5file, key):
    field = h5file[key][:]
    result = field.astype(np.uint8)
    result = result.tostring().decode('ascii')
    return result

def h5ReaderNum(h5file, key, type):
    field = h5file[key][:]
    result = field.astype(type)
    return result

# 将文件中的数据加载为numpy数组形式
def loadData2NP(conf, filename, database):
    '''
    @1: store image id into a 2d matrix
    @2: store the path of one user-image pair into lmdb database
    @3: store the box of one user-image pair into lmdb database
    @4: the number of user-image pair
    data:    user1:  1, path11, box11
             user1:  2, path12, box12
             user1:  3, path13, box13
             user2:  4, path24, box24
             user2:  5, path25, box25

    return: @1  [ [0, 0, 0], # favorite images of user0
                  [1, 2, 3], # favorite images of  user1
                  [4, 5, 0], # favorite images of  user2
                ]
            @2  every line is a array of storing image path
                ('user1image1', path11)
                ('user1image2', path12)
                ('user1image3', path13)
                ('user2image4', path24)
                ('user2image5', path25)
            @3  every line is a array of storing image box
                ('user1image1', box11)
                ('user1image2', box12)
                ('user1image3', box13)
                ('user2image4', box24)
                ('user2image5', box25)
    '''
    user2image = np.array(np.zeros(shape=[conf.user_number + 1, 20]), dtype=np.int32)#(101,20）zero matrix
    record_num = 0#record the lines of training_file
    with open(filename)as data:
        for line in data:
            record_num = record_num + 1
            line = line.split()
            cur_user = int(line.pop(0))#first colom
            cur_img  = int(line.pop(0))#second colom
            #print('Processing record ', record_num, ' with user ', cur_user, ' and image ', cur_img)
            user2image[cur_user][countNonZero1d_np(user2image[cur_user])] = cur_img
    return user2image, record_num

def getColunmofFile(filename, column, unique=True, dtype=np.int32):
    '''
    Extract the specific column data in file
    eg:  data of file:  1 2 slzhai 123 341
                        1 3 mengyuan 432 12
                        2 1 maolin 34 12
                        2 3 mengzhe 34 64
    return: if column = 1, and unique = False -->  [1, 1, 2, 2];
            if column = 1, and unique = True, -->  [1, 2]

            if column = 2, and unique = False -->  [2, 3, 1, 3];
            if column = 2, and unique = True, -->  [2, 3, 1]
    '''
    result = []
    with open(filename, 'r') as data:
        for line in data:
            line = line.split()
            cur_column = int(line[column-1])
            result.append(cur_column)
    result = np.asarray(result, dtype=dtype)
    if unique:
        result = np.unique(result)
    return np.asarray(result)

# Batch training pair with (u, p, n)
def getBPRTrainData(conf, train_matrix):
    user_batch = []  # Store batch of users
    pimage_batch = []# Store batch of positive images
    nimage_batch = []# Store batch of negative images
    index_num = 0    # Indicate whether the data is enough or not
    while True:
        rand_user = random.randint(1, conf.user_number)  # Random sample a user
        fav_images = padRemoverInt_np(train_matrix[rand_user]) # Obtain his favorite images in train file
        if fav_images.size == 0: # No favorite images
            continue
        positive_img = random.sample(list(fav_images), 1)     # Random sample a positive image
        negative_img = random.randint(1, conf.image_number) # Random sample a negative image
        while negative_img in fav_images:
            negative_img = random.randint(1, conf.image_number) # Negative image must not in train favorite images
        # process these data
        user_batch.append(rand_user)      # Prepare user batch
        pimage_batch.append(positive_img) # Prepare positive image batch
        nimage_batch.append(negative_img) # Prepare negative image batch
        index_num = index_num + 1
        if index_num == conf.batch_size: # Data is enough
            break
    return np.asarray(user_batch), np.reshape(np.asarray(pimage_batch), newshape=[-1]), np.asarray(nimage_batch)

def getVSMTrainData(conf, train_matrix):
    image_names = h5py.File(conf.names_h5, 'r')
    image_boxes = h5py.File(conf.boxes_h5, 'r')

    batch_user, batch_image, batch_Nimage = getBPRTrainData(conf, train_matrix)
    boxes = []
    boxes_group = []
    boxes_category = []
    batch_user_ = []
    for index in range(conf.batch_size):
        cur_user = batch_user[index]
        cur_image = batch_image[index]
        #print('current user: ', cur_user, '  current image: ', cur_image)
        cur_image_name = h5ReaderStr(image_names, 'user' + str(cur_user) + 'img' + str(cur_image))
        cur_boxes = image_boxes['user' + str(cur_user) + 'img' + str(cur_image)][:]
        cur_boxes = [float(x) for x in cur_boxes]
        cur_boxes = np.reshape(np.asarray(cur_boxes, dtype=np.float32), newshape=[-1, 5])  # [label, xmin, xmax, ymin, ymax]
        cur_path = conf.image_path + str(cur_image_name) + '.jpg'
        image = cv2.imread(cur_path)
        image_shape = image.shape
        image_std = per_image_standardization(image)
        # boxes.append(cv2.resize(image_std, (conf.box_width, conf.box_height)))
        boxes_group.append(index)
        boxes_category.append(0)
        batch_user_.append(cur_user)
        for j in range(6):#一个图片的box数量
            cur_offset_h, cur_offset_w, cur_h, cur_w, box_label = computeOffsetAndTarget(image_shape, cur_boxes[j])
            # if (cur_h <= 0 or cur_w <= 0 or cur_h < conf.box_limit or cur_w < conf.box_limit):
            #     continue  # If box do not meet our demand
            box_pixel = cv2.resize(image_std[cur_offset_h:cur_offset_h + cur_h, cur_offset_w:cur_offset_w + cur_w], (conf.box_width, conf.box_height))
            boxes.append(box_pixel)
            #boxes.append(box_label)
            boxes_group.append(index)
            boxes_category.append(int(cur_boxes[j][0]))
            batch_user_.append(cur_user)
    image_names.close()
    image_boxes.close()
    return np.asarray(batch_user), np.asarray(batch_image), np.asarray(batch_Nimage),\
               np.asarray(boxes), np.asarray(boxes_group), np.asarray(batch_user_), np.asarray(boxes_category)
