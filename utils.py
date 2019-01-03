# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

"""
Created on Mon May 21 18:41:43 2018

@author: Meng Yuan

VSM code with TensorFlow-v1.4
"""

def findTopK(scores, trainids, k):
    recommend_list = tf.nn.top_k(input=scores, k=tf.add(tf.size(trainids), k+1), sorted=True)
    recommend_ids = recommend_list.indices
    result, _ = tf.setdiff1d(recommend_ids, trainids)
    return result[0:k]

def calPrecision(prediction, groundtruth):
    '''
    Caculate precision based on the prdiction
    :param prediction: A tensor that store the top-K predicted items
    :param groundtruth: A tensor indicates the positive items
    :return: Precision that is same meaning with the prediction (top-K)
    '''
    nonhit = tf.setdiff1d(groundtruth, prediction).out
    hit_num = tf.size(groundtruth) - tf.size(nonhit)
    result = tf.divide(tf.cast(hit_num, dtype=tf.float32), tf.cast(tf.size(prediction), dtype=tf.float32))
    return result
def calRecall(prediction, groundtruth):
    nonhit = tf.setdiff1d(groundtruth, prediction).out
    hit_num = tf.size(groundtruth) - tf.size(nonhit)
    result = tf.divide(tf.cast(hit_num, dtype=tf.float32), tf.cast(tf.size(groundtruth), dtype=tf.float32))
    return result

def calNDCG(prediction, groundtruth):
    # Compute IDCG
    range = tf.cast(tf.add(tf.range(start=0, limit=tf.size(groundtruth)), 2), dtype=tf.float32)
    range_ = tf.cast(tf.tile(input=[2], multiples=[tf.size(groundtruth)]), dtype=tf.float32)
    idcg = tf.reduce_sum(tf.divide(tf.log(range_), tf.log(range)))
    # Compute DCG part
    non_hits = tf.setdiff1d(prediction,groundtruth).out
    hits = tf.setdiff1d(prediction, non_hits).idx
    ranks = tf.cast(tf.add(hits, 2), dtype=tf.float32)
    ranks_ = tf.cast(tf.tile(input=[2], multiples=[tf.size(ranks)]), dtype=tf.float32)
    dcg = tf.reduce_sum(tf.divide(tf.log(ranks_), tf.log(ranks)))
    return tf.divide(dcg, idcg)

def calAUC(scores_with_train, groundtruth, trainids):
    image_id_list = tf.nn.top_k(input=tf.multiply(scores_with_train,-1), k=tf.size(scores_with_train)).indices
    no_train_ids_sorted, _ = tf.setdiff1d(image_id_list, trainids)
    residual_part = tf.setdiff1d(no_train_ids_sorted, groundtruth).out
    rank_pos = tf.setdiff1d(no_train_ids_sorted, residual_part).idx
    pos_sum = tf.cast(tf.reduce_sum(tf.add(rank_pos, 1)), dtype=tf.float32)
    M = tf.cast(tf.size(groundtruth), dtype=tf.float32)
    N = tf.cast(tf.size(residual_part), dtype=tf.float32)
    return tf.divide(pos_sum-(M+1)*M/2.0, M*N)


def weightedMultiply(tensors, weights):
    '''
    Summarize these tensors by its weights

    :param tensors: [tensors with type of tf.float32] A matrix that should be weighted sum according to the weights
    :param weights: [tensors with type of tf.float32] Denotes the weight of row-vector in the tensors
    :return: [tensors with type of tf.float32y] A vector that has the same column size with weights

    Criterion:
    1. Generate a dialog matrix according to vector weights. eg. [1,2,3] -> [ [1,0,0], [0,2,0], [0,0,3]]
    2. Multiply the dialog matrix by the tensors
    3. Reduce summary
    '''
    # Generate a dialog matrix according to the weights vector
    dia_weights = tf.matrix_diag(weights)
    dia_weights = tf.cast(dia_weights, dtype=tf.float32)
    tensors = tf.cast(tensors, tf.float32)
    # Multiply these tensor by the dialog weight matrix
    weighted_tensors = tf.matmul(dia_weights, tensors)
    return weighted_tensors

def computeOffsetAndTarget(shape, box):
    image_height = shape[0]
    image_width = shape[1]
    offset_height = int(box[3] * image_height)
    offset_width = int(box[1] * image_width)
    target_height = int((box[4] - box[3]) * image_height)
    target_width = int((box[2] - box[1]) * image_width)
    return offset_height, offset_width, target_height, target_width#, box[0]

def per_image_standardization(img):
    img = Image.fromarray(img)
    prob = random.randint(0 ,1)
    if prob == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if img.mode == 'RGB':
        channel = 3
    num_compare = img.size[0] * img.size[1] * channel
    img_arr = np.array(img)
    img_t = (img_arr - np.mean(img_arr)) / max(np.std(img_arr), 1 / num_compare)
    img = np.array(img_t)
    return img

def padRemoverInt_np(vector):
    result = []
    vector = np.asarray(vector)
    for i in range(vector.size):
        if vector[i] != 0:
            result.append(int(vector[i]))
    return np.asarray(result)

def padRemoverInt(tensor):
    '''
    Remove padding from a tensor before sending to network
    Ex: input  = [[1, 2], [0, 0],  [0, 0], [3, 4], [5, 6]]
        output = [[1, 2], [3, 4], [5, 6]]
    :param tensor: Tensor needed to remove zero sclice
    :return: Removed padding tensor
    '''
    nonpad_ids = tf.to_int32(tf.where(tensor > 0))
    tensor_shape = tensor.get_shape().as_list()
    result = tf.gather_nd(tensor, indices=nonpad_ids)
    result = tf.reshape(result, shape=[-1] + tensor_shape[1:])
    return result

def countNonZero1d_np(arr):
    result = 0
    for i in range(np.array(arr).size):
        if arr[i] != 0:
            result = result + 1
    return result

def findOptimal(result, all_loss, conf):
    result = np.asarray(result, dtype=np.float32)
    best_pre5 = 0
    epoch_best_pre5 = 0
    best_pre10 = 0
    epoch_best_pre10 = 0
    best_rec5 = 0
    epoch_best_rec5 = 0
    best_rec10 = 0
    epoch_best_rec10 = 0
    for i in range(result.shape[0]):
        cur_result = result[i]
        if cur_result[0] > best_pre5:
            best_pre5 = cur_result[0]
            epoch_best_pre5 = i + 1
        if cur_result[1] > best_pre10:
            best_pre10 = cur_result[1]
            epoch_best_pre10 = i + 1
        if cur_result[2] > best_rec5:
            best_rec5 = cur_result[2]
            epoch_best_rec5 = i + 1
        if cur_result[3] > best_rec10:
            best_rec10 = cur_result[3]
            epoch_best_rec10 = i + 1
    print('\n\n')
    print('Optimal precision5 and epoch: ', result[epoch_best_pre5 - 1], '  ', epoch_best_pre5, '(',
          int(conf.epochs / conf.eval_epoch), ')')
    print('Optimal precision10 and epoch: ', result[epoch_best_pre10 - 1], '  ', epoch_best_pre10, '(',
          int(conf.epochs / conf.eval_epoch), ')')
    print('Optimal recall5 and epoch: ', result[epoch_best_rec5 - 1], '  ', epoch_best_rec5, '(',
          int(conf.epochs / conf.eval_epoch), ')')
    print('Optimal recall10 and epoch: ', result[epoch_best_rec10 - 1], '  ', epoch_best_rec10, '(',
          int(conf.epochs / conf.eval_epoch), ')')

    all_loss = np.asarray(all_loss, dtype=np.float32)
    drawFigure(all_loss, 'Loss tendency.', 'VSM')
    drawFigure(result[:, 0], 'Precision@5', 'VSM')
    drawFigure(result[:, 1], 'Precision@10', 'VSM')
    drawFigure(result[:, 2], 'Recall@5', 'VSM')
    drawFigure(result[:, 3], 'Recall@10', 'VSM')

def drawFigure(losses, title, model):
    y = np.asarray(losses)
    x = np.arange(1, y.size+1)
    plt.xlim(1, y.size)
    plt.title(model)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.grid(True)
    plt.plot(x, y)
    plt.show()
