# -*- coding: utf-8 -*-
from tools.utils import *

import tensorflow as tf
import numpy as np

"""
Created on Mon May 21 18:41:43 2018

@author: Meng Yuan

VSM code with TensorFlow-v1.4

> VSM model
"""

class VSM(object):
    def __init__(self, conf):
        tf.reset_default_graph()
        # Our configuration
        self.Config = conf
        # Training data (General BPR model data)
        self.batch_user = tf.placeholder(shape=[None], dtype=np.int32)
        self.batch_image = tf.placeholder(shape=[None], dtype=np.int32)
        self.batch_Nimage = tf.placeholder(shape=[None], dtype=np.int32)
        # Training data (Extract image-level and box-level feature)
        self.batch_user_ = tf.placeholder(shape=[None], dtype=np.int32)
        self.batch_box = tf.placeholder(shape=[None, self.Config.box_height, self.Config.box_width, 3], dtype=np.uint8)
        self.box_group = tf.placeholder(shape=[None], dtype=np.int32)
        self.box_category = tf.placeholder(shape=[None], dtype=np.int32)
        # For evaluation
        self.test_users = tf.placeholder(shape=[None], dtype=np.int32)
        self.train_user_image = tf.placeholder(shape=[None, None], dtype=np.int32)
        self.test_user_image = tf.placeholder(shape=[None, None], dtype=np.int32)

        # Extract box feature
        box_features = tf.reshape(self.AlexNet(self.batch_box), shape=[-1, self.Config.feature_dimension])
        # Compute box attention weights
        box_weights = self.softmax(tf.reshape(self.towNetBox(box_features), shape=[-1]), level='box')
        # Obtain image features
        image_features = self.weightedSum(box_features, box_weights, level='box')
        # Compute image attention weights
        image_weights = self.softmax(tf.reshape(self.towNetImage(image_features), shape=[-1]), level='image')
        # Obtain user auxiliary vector
        auxiliary_vectors = self.weightedSum(image_features, image_weights, level='image')

        # Compute loss
        latent_users = self.getEmbeddings(self.batch_user, 'user')
        latent_images = self.getEmbeddings(self.batch_image, 'image')
        latent_Nimages = self.getEmbeddings(self.batch_Nimage, 'image')
        diff_scores = tf.reduce_sum(tf.multiply(tf.add(latent_users, auxiliary_vectors), tf.subtract(latent_images, latent_Nimages)), reduction_indices=1)
        self.loss = tf.reduce_sum(tf.multiply(tf.log(tf.sigmoid(diff_scores)), -1))
        l2_norm = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        self.loss_reg = l2_norm * self.Config.regularization
        loss_total = self.loss + self.loss_reg
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.Config.learning_rate).minimize(loss_total)
        # Build evaluation network
        self.evaluationVSM()

    def evaluationVSM(self):
        all_images = tf.range(start=0, limit=self.Config.image_number + 1)
        with tf.variable_scope('temp_vars', reuse=tf.AUTO_REUSE):
            temp_auxi_feature = tf.get_variable(shape=[self.Config.user_number + 1, self.Config.latent_dimension],
                                                dtype=np.float32, initializer=tf.constant_initializer(0.),
                                                name='tmp_user_feature', trainable=False)
        def cond(iter, size, results):
            return iter < size
        def body(iter, size, results):
            cur_user = self.test_users[iter]
            cur_train_images = self.train_user_image[cur_user]
            cur_test_images = padRemoverInt(self.test_user_image[cur_user])
            cur_auxi = tf.nn.embedding_lookup(temp_auxi_feature, cur_user)
            all_scores_ = tf.multiply(tf.add(self.getEmbeddings(cur_user, 'user'), cur_auxi), self.getEmbeddings(all_images, 'image'))
            all_scores = tf.reduce_sum(all_scores_, axis=1)
            auc = calAUC(scores_with_train=all_scores, groundtruth=cur_test_images, trainids=cur_train_images)
            top_5 = findTopK(scores=all_scores, trainids=cur_train_images, k=5)
            top_10 =findTopK(scores=all_scores, trainids=cur_train_images, k=10)
            pre5 = calPrecision(prediction=top_5, groundtruth=cur_test_images)
            pre10 = calPrecision(prediction=top_10, groundtruth=cur_test_images)
            rec5 = calRecall(prediction=top_5, groundtruth=cur_test_images)
            rec10 = calRecall(prediction=top_10, groundtruth=cur_test_images)
            ndcg5 = calNDCG(prediction=top_5, groundtruth=cur_test_images)
            ndcg10 = calNDCG(prediction=top_10, groundtruth=cur_test_images)
            one_user_result = tf.reshape(tf.convert_to_tensor([pre5, pre10, rec5, rec10, ndcg5, ndcg10, auc]), shape=[7])
            result_ = tf.cond(tf.equal(tf.reduce_sum(results), 0), lambda: [one_user_result],
                             lambda: tf.concat(values=[results[:], [one_user_result]], axis=0))
            return iter + 1, size, tf.reshape(result_, shape=[-1, 7])

        start = tf.constant(0)
        size = tf.size(self.test_users)
        iter, size, results = tf.while_loop(cond, body, loop_vars=[start, size, tf.zeros(shape=[1, 7], dtype=tf.float32)],
                                            shape_invariants=[start.get_shape(), size.get_shape(),
                                                              tf.TensorShape([None, 7])])
        self.results = results
        self.mean_results = tf.reduce_mean(results, axis=0)

    def getEmbeddings(self, data, name, init=1):
        if init == 1:
            initializer = tf.keras.initializers.he_normal()
        elif init == 2:
            initializer = tf.keras.initializers.he_uniform()
        elif init == 3:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.random_uniform_initializer(-1,1)
        # Parameters that should be learned from the model
        with tf.variable_scope('embedding_matrixs', reuse=tf.AUTO_REUSE):
            user_embeddings = tf.get_variable(shape=[self.Config.user_number + 1, self.Config.latent_dimension],
                                              dtype=np.float32, initializer=initializer,
                                              name='user_embedding', trainable=False)
            image_embeddings = tf.get_variable(shape=[self.Config.image_number + 1, self.Config.latent_dimension],
                                               dtype=np.float32, initializer=initializer,
                                               name='image_embedding', trainable=False)
        emb = None
        if name == 'user':
            emb = tf.nn.embedding_lookup(user_embeddings, data)
        elif name == 'image':
            emb = tf.nn.embedding_lookup(image_embeddings, data)
        else: exit('No this embedding!')
        return emb

    def towNetImage(self, image_content, hidden_num=2048, init=1):
        '''
        It is a full-connected layer that can weighted sum
        the input, which is the attention weight
        :param conf:
        :return:
        '''
        if init == 1:
            initializer = tf.keras.initializers.he_normal()
        elif init == 2:
            initializer = tf.keras.initializers.he_uniform()
        elif init == 3:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.random_uniform_initializer(-1,1)
        with tf.variable_scope('twonet_image', reuse=tf.AUTO_REUSE) as scope:
            # Parameters of first layer
            img_attention_uw1 = tf.get_variable("iattention_weight_user", shape=[self.Config.latent_dimension, hidden_num],
                                                initializer=initializer, trainable=True)
            img_attention_iw1 = tf.get_variable("iattention_weight_image", shape=[self.Config.latent_dimension, hidden_num],
                                                initializer=initializer, trainable=True)
            img_attention_cw1 = tf.get_variable("iattention_weight_cont", shape=[self.Config.feature_dimension, hidden_num],
                                                initializer=initializer, trainable=True)
            img_attention_b1 = tf.get_variable("iattention_biase1", [hidden_num], trainable=True)
            # Parameters of second layer
            img_attention_w2 = tf.get_variable("iattention_weight2", shape=[hidden_num, 1],
                                               initializer=initializer, trainable=True)
            img_attention_b2 = tf.get_variable("iattention_biase2", [1],
                                               initializer=initializer, trainable=True)
        user_emb = self.getEmbeddings(self.batch_user, 'user')
        image_emb = self.getEmbeddings(self.batch_image, 'image')
        # Matrix multiply weights and inputs and add bias
        layer1 = tf.add(tf.reduce_sum([
            tf.matmul(user_emb, img_attention_uw1),
            tf.matmul(image_emb, img_attention_iw1),
            tf.matmul(image_content, img_attention_cw1)
        ], reduction_indices=0), img_attention_b1)
        layer2 = tf.nn.xw_plus_b(tf.nn.elu(layer1), img_attention_w2, img_attention_b2)
        return layer2

    def towNetBox(self, box, hidden_num=1024, init=1):
        '''
        It is a full-connected layer
        :param conf:
        :return:
        '''
        if init == 1:
            initializer = tf.keras.initializers.he_normal()
        elif init == 2:
            initializer = tf.keras.initializers.he_uniform()
        elif init == 3:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.random_uniform_initializer(-1,1)
        # Parameters of first layer
        with tf.variable_scope('twonet_box', reuse=tf.AUTO_REUSE) as scope:
            box_attention_uw1 = tf.get_variable("battention_weight_user", shape=[self.Config.latent_dimension, hidden_num],
                                                initializer=initializer, trainable=True)
            box_attention_bw1 = tf.get_variable("battention_weight_image", shape=[self.Config.feature_dimension, hidden_num],
                                                initializer=initializer, trainable=True)
            box_attention_b1 = tf.get_variable("battention_biase1", [hidden_num],
                                               initializer=initializer, trainable=True)
            # Parameters of second layer
            box_attention_w2 = tf.get_variable("battention_weight2", shape=[hidden_num, 1],
                                               initializer=initializer, trainable=True)
            box_attention_b2 = tf.get_variable("battention_biase2", [1],
                                               initializer=initializer, trainable=True)
        user_emb = self.getEmbeddings(self.batch_user_, 'user')
        # Matrix multiply weights and inputs and add bias
        layer1 = tf.add(tf.reduce_sum([
            tf.matmul(user_emb, box_attention_uw1),
            tf.matmul(box, box_attention_bw1)
        ], reduction_indices=0), box_attention_b1)
        layer2 = tf.nn.xw_plus_b(tf.nn.elu(layer1), box_attention_w2, box_attention_b2)
        return layer2

    def AlexNet(self, image):
        '''
        :param image: objects in images
        :return: feature_dimension-Vector of objects
        '''
        def batch_norm(x):
            '''Batch normlization(I didn't include the offset and scale)'''
            epsilon = 1e-3
            batch_mean, batch_var = tf.nn.moments(x, [0])
            x = tf.nn.batch_normalization(x, mean=batch_mean, variance=batch_var, offset=None, scale=None,
                                          variance_epsilon=epsilon)
            return x
        image = tf.cast(image, dtype=tf.float32)
        with tf.variable_scope("conv1"):
            conv1 = tf.contrib.layers.conv2d(image, 96, [11, 11], stride=4, activation_fn=tf.nn.relu, padding='SAME',
                                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            pool1 = tf.contrib.layers.max_pool2d(conv1, [3, 3], padding='VALID')
        with tf.variable_scope("conv2"):
            conv2 = tf.contrib.layers.conv2d(pool1, 256, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            pool2 = tf.contrib.layers.max_pool2d(conv2, [3, 3], padding='VALID')
        with tf.variable_scope("conv3"):
            conv3 = tf.contrib.layers.conv2d(pool2, 384, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            pool3 = tf.contrib.layers.max_pool2d(conv3, [2, 2], padding='VALID')
        with tf.variable_scope("conv4"):
            conv4 = tf.contrib.layers.conv2d(pool3, 384, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        with tf.variable_scope("conv5"):
            conv5 = tf.contrib.layers.conv2d(conv4, 256, [3, 3], activation_fn=None, padding='SAME',
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv5 = tf.contrib.layers.flatten(conv5)
        with tf.variable_scope('fc6'):
            fc6 = tf.contrib.layers.fully_connected(conv5, 4096, activation_fn=tf.nn.relu)
            fc6 = batch_norm(fc6)#tf.nn.dropout(x=fc6, keep_prob=self.Config.keep_probability)
        with tf.variable_scope('fc7'):
           fc7 = tf.contrib.layers.fully_connected(fc6, 1024, activation_fn=tf.nn.relu)
           fc7 = batch_norm(fc7)#tf.nn.dropout(x=fc7, keep_prob=self.Config.keep_probability)
        with tf.variable_scope('fc8'):
            fc8 = tf.contrib.layers.fully_connected(fc7, self.Config.feature_dimension, activation_fn=tf.nn.relu)
        result = tf.reshape(fc8, shape=[-1, self.Config.feature_dimension])
        return result

    def softmax(self, weights, level):
        # Temporary variables for getting the image-feature and image weights
        with tf.variable_scope('temp_vars', reuse=tf.AUTO_REUSE):
            temp_image_weights = tf.get_variable(shape=[self.Config.batch_size], dtype=np.float32,
                                                 initializer=tf.constant_initializer(0.),
                                                 name='tmp_img_weights', trainable=False)
            temp_user_weights = tf.get_variable(shape=[self.Config.user_number + 1], dtype=np.float32,
                                                initializer=tf.constant_initializer(0.),
                                                name='tmp_user_weights', trainable=False)
        weights = tf.exp(weights)
        summation = None
        if level == 'box':
            temp_image_weights = tf.assign(temp_image_weights, tf.zeros(shape=[self.Config.batch_size], dtype=np.float32))
            temp_image_weights = tf.scatter_add(temp_image_weights, indices=self.box_group, updates=weights)
            summation = tf.nn.embedding_lookup(temp_image_weights, self.box_group)
        elif level == 'image':
            temp_user_weights = tf.assign(temp_user_weights,tf.zeros(shape=[self.Config.user_number + 1], dtype=np.float32))
            temp_user_weights = tf.scatter_add(temp_user_weights, indices=self.batch_user, updates=weights)
            summation = tf.nn.embedding_lookup(temp_user_weights, self.batch_user)
        else:
            exit('Error!')
        return tf.divide(weights, summation)

    def weightedSum(self, features, weights, level):
        with tf.variable_scope('temp_vars', reuse=tf.AUTO_REUSE):
            temp_image_feature = tf.get_variable(shape=[self.Config.batch_size, self.Config.feature_dimension],
                                                 dtype=np.float32, initializer=tf.constant_initializer(0.),
                                                 name='tmp_img_feature', trainable=False)
            temp_auxi_feature = tf.get_variable(shape=[self.Config.user_number + 1, self.Config.latent_dimension],
                                                dtype=np.float32, initializer=tf.constant_initializer(0.),
                                                name='tmp_user_feature', trainable=False)
        features = weightedMultiply(features, weights)
        result = None
        if level == 'box':
            temp_image_feature = tf.assign(temp_image_feature, tf.zeros(shape=[self.Config.batch_size, self.Config.feature_dimension], dtype=np.float32))
            temp_image_feature = tf.scatter_add(temp_image_feature, indices=self.box_group, updates=features)
            result = tf.nn.embedding_lookup(temp_image_feature, self.batch_image)
        elif level == 'image':
            temp_auxi_feature = tf.scatter_add(temp_auxi_feature, indices=self.batch_user, updates=features)
            result = tf.nn.embedding_lookup(temp_auxi_feature, self.batch_user)
        else:
            exit('Error!')
        return result
