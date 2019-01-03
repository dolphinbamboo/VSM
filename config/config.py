# -*- coding: utf-8 -*-
import tensorflow as tf

"""
Created on Mon May 21 18:41:43 2018

@author: Meng Yuan

VSM code with TensorFlow-v1.4

> configuration file
"""

flags = tf.app.flags

flags.DEFINE_integer("batch_size", 128, "Batch size of data in training")
flags.DEFINE_integer("epochs", 110, "Epochs of training")
flags.DEFINE_integer("eval_epoch", 1, "Epoch when evaluate")
flags.DEFINE_string("train_data", "[path]", "Path to store train txt file")
flags.DEFINE_string("test_data", "[path]", "Path to store test txt file")
flags.DEFINE_string("image_path", "[path]", "Path to store images")
flags.DEFINE_string("names_h5", "[path]", "Path to store image names")
flags.DEFINE_string("boxes_h5", "[path]", "Path to store box names")

flags.DEFINE_integer("user_number", 2000, "Number of users in dataset")
flags.DEFINE_integer("image_number", 16350, "Number of images in dataset")
flags.DEFINE_integer("box_height", 64, "Height of the box")
flags.DEFINE_integer("box_width", 64, "Width of the box")
flags.DEFINE_integer("box_limit", 0, "Minimal box size")

flags.DEFINE_integer("latent_dimension", 128, "Dimension of user and image latent matrix")
flags.DEFINE_integer("feature_dimension", 128, "Dimension of image feature")

flags.DEFINE_float("regularization", 0.001, "Regularization term")
flags.DEFINE_float("learning_rate", 0.002, "Learning rate in our model")
flags.DEFINE_float("grad_clip", 1, "Clip for the learned gradient")
flags.DEFINE_float("momentum", 0.9, "Momentum for training the model")
flags.DEFINE_float("keep_probability",0.8, "Config in dropout")

# Prepare configuration
conf = flags.FLAGS
