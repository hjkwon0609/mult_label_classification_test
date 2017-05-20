import time
import argparse
import math
import random
import os
import numpy as np
import tensorflow as tf
import hickle as hkl

import pdb
from time import gmtime, strftime

from config import Config

from model import ClassificationModel
from threshold_train_model import ThresholdTrainModel

TESTING_MODE = True

if __name__ == "__main__":

    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())

    INPUT_DIR = 'data/preprocessed/'
    SAVED_MODEL_DIR = 'model/'

    filename = 'data.npz'

    # train_features_batch_name = 'train_features_batch.npy'
    # train_label_batch_name = 'train_label_batch.npy'
    # dev_features_batch_name = 'dev_features_batch.npy'
    # dev_label_batch_name = 'dev_label_batch.npy'

    if TESTING_MODE:
        filename = 'smaller_' + filename

    train_labels = None
    train_features = None
    dev_labels = None
    dev_features = None

    with np.load(INPUT_DIR + filename) as data:
        train_features = data['train_features_padded_batch']
        train_labels = data['train_label_batch']
        dev_features = data['dev_features_padded_batch']
        dev_labels = data['dev_label_batch']

    # train_labels = np.load(INPUT_DIR + train_label_batch_name)
    # train_features = np.load(INPUT_DIR + train_features_batch_name)

    # dev_labels = np.load(INPUT_DIR + dev_label_batch_name)
    # dev_features = np.load(INPUT_DIR + dev_features_batch_name)

    num_data = np.sum(len(batch) for batch in train_labels)
    num_batches_per_epoch = int(math.ceil(num_data / Config.batch_size))
    num_dev_data = np.sum(len(batch) for batch in dev_labels)
    num_dev_batches_per_epoch = int(math.ceil(num_dev_data / Config.batch_size))

    print('Finished reading in data...')

    print(num_data)

    with tf.Graph().as_default():
        model = ClassificationModel()
        threshold_train_model = ThresholdTrainModel()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as session:
            # Initializate the weights and biases
            session.run(init)
            # if args.load_from_file is not None:
            #     new_saver = tf.train.import_meta_graph('%s.meta' % args.load_from_file, clear_devices=True)
            #     new_saver.restore(session, args.load_from_file)

            train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)

            global_start = time.time()

            step_ii = 0

            for curr_epoch in range(Config.num_epochs):
                total_train_cost = total_train_wer = 0
                start = time.time()

                for batch in random.sample(range(num_batches_per_epoch), num_batches_per_epoch):
                    cur_batch_size = len(train_labels[batch])

                    batch_cost, logits, summary = model.train_on_batch(session, 
                                                            train_features[batch],
                                                            train_labels[batch], 
                                                            train=True)
                    # print('logits: %s' % (logits))
                    total_train_cost += batch_cost * cur_batch_size
                    train_writer.add_summary(summary, step_ii)
                    threshold_train_model.add_data(logits, train_labels[batch])

                    step_ii += 1

                train_cost = total_train_cost / num_data

                threshold_train_model.determine_threshold()
                threshold_train_model.train_regression()

                num_dev_batches = len(dev_labels)
                dev_logits = None
                total_dev_labels = None
                total_batch_cost = 0
                total_batch_examples = 0

                for batch in random.sample(range(num_dev_batches_per_epoch), num_dev_batches_per_epoch):
                    cur_batch_size = len(dev_labels[batch])
                    total_batch_examples += cur_batch_size

                    _val_batch_cost, _dev_logits, _ = model.train_on_batch(session, dev_features[batch], dev_labels[batch], train=False)
                    
                    if dev_logits is None:
                        dev_logits = _dev_logits
                        total_dev_labels = dev_labels[batch]
                    else:
                        dev_logits = np.concatenate((dev_logits, _dev_logits), axis=0)
                        total_dev_labels = np.concatenate((total_dev_labels, dev_labels[batch]), axis=0)
                    # print('total_dev_labels: %s' % (total_dev_labels))

                    total_batch_cost += cur_batch_size * _val_batch_cost

                batch_cost = None
                try:
                    batch_cost = total_batch_cost / total_batch_examples
                except ZeroDivisionError:
                    batch_cost = 0
                
                thresholds = threshold_train_model.calculate_thresholds(dev_logits)
                predicted_labels = threshold_train_model.predict_labels(dev_logits, thresholds)

                f_point_five, precision, recall = threshold_train_model.calculate_F_score(predicted_labels, total_dev_labels)

                log = "Epoch {}/{}, thresholds: {}, train_cost = {:.3f}, val_cost = {:.3f}, precision = {:.3f}, recall = {:.3f}, f0.5 = {:.3f}, time = {:.3f}"
                print(
                log.format(curr_epoch + 1, Config.num_epochs, thresholds, train_cost, batch_cost, precision, recall, f_point_five, time.time() - start))

                if (curr_epoch + 1) % 10 == 0:
                    saver.save(session, '%ssaved_model_epoch%d' % (SAVED_MODEL_DIR, curr_epoch + 1), global_step=curr_epoch + 1)
