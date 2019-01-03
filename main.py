# -*- coding: utf-8 -*-
from conf.config import conf
from tools.data import *
from tools.model import *

import tensorflow as tf
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
Created on Mon May 21 18:41:43 2018

@author: Meng Yuan

VSM code with TensorFlow-v1.4

> main routine
"""

runModel = VSM(conf)

def main(unuse_args):
    train_matrix, num_pair = loadData2NP(conf, conf.train_data,False)  # 2-d matrix of [user, favorite-image] in train file
    test_matrix, _ = loadData2NP(conf, conf.test_data, False)  # 2-d matrix of [user, favorite-image] in test file
    test_data = getColunmofFile(conf.test_data, 1)  # test users
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        writer = tf.summary.FileWriter('logs/', sess.graph)
        if 'session' in locals() and sess is not None:
            print('Close interactive session')
            sess.close()
        print('Start training VSM model...')
        with tf.device("/gpu:0"):
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            # Training bpr model
            last_loss = 0  # Store loss of last epoch for computing delta_loss
            result = []  # Store precision and recall of every evaluation
            all_loss = []  # Store loss while training
            for i in range(conf.epochs):
                loss = 0
                train_start = time.time()
                for j in range(int(num_pair / conf.batch_size)):
                    batch_u, batch_i, batch_ni, boxes, box_groups, batch_u_, box_category = getVSMTrainData(conf, train_matrix)

                    loss_, loss_reg, _ = sess.run([runModel.loss, runModel.loss_reg, runModel.optimizer],
                                                  feed_dict={runModel.batch_user: batch_u,
                                                             runModel.batch_image: batch_i,
                                                             runModel.batch_Nimage: batch_ni,
                                                             runModel.batch_box: boxes,
                                                             runModel.box_group: box_groups,
                                                             runModel.batch_user_: batch_u_,
                                                           })
                    loss = loss + loss_ + loss_reg
                    print('Have finished batch learning with loss : ', loss_ + loss_reg)
                train_end = time.time()
                all_loss.append(loss)
                print('Epoch ', i + 1, ' : ', loss, '\tDelta loss: ', loss - last_loss, '\tElapsed time(s):  ',
                      round(train_end - train_start))
                last_loss = loss
                # Evaluate model
                if i % conf.eval_epoch == 0:
                    rall, rmean = sess.run([runModel.results, runModel.mean_results],
                                           feed_dict={runModel.train_user_image: train_matrix,
                                                      runModel.test_user_image: test_matrix,
                                                      runModel.test_users: test_data})
                    print(
                        'Evaluation result of this epoch\n\t-->  precision@5, precision@10, recall@5, recall@10, NDCG@5, NDCG@10, AUC:\n\t--> ',
                        rmean, '\n')
                    result.append(rmean)
            result = np.asarray(result, dtype=np.float32)
            # Find epoch that reached the optimal results
            findOptimal(result, all_loss, conf)
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
