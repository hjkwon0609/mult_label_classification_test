import time
import argparse
import math
import random
import os
import numpy as np
from six.moves import xrange as range

# from utils import *
import pdb
from time import gmtime, strftime

from config import Config

def split_train_dev(filename):
    with open(filename, 'r') as f:
        _, labels, features = zip(*(line.strip().split('\t') for line in f))
        
        num_data = len(labels)
        print('Num data read: %d' % (num_data))

        features = [[f.split('/') for f in sample.split()] for sample in features]

        dev_ix = set(random.sample(xrange(num_data), num_data / 5))

        train_labels = [l for i, l in enumerate(labels) if i not in dev_ix]
        dev_labels = [l for i, l in enumerate(labels) if i in dev_ix]

        train_features = [l for i, l in enumerate(features) if i not in dev_ix]
        dev_features = [l for i, l in enumerate(features) if i in dev_ix]

        train_seq_len = [len(sample) for sample in train_features]
        dev_seq_len = [len(sample) for sample in dev_features]


        print(train_features[:10])

        return train_labels, train_features, dev_labels, dev_features



if __name__ == "__main__":

    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())

    train_data = 'data/AClassification.train.txt'
    train_labels, train_features, dev_labels, dev_features = split_train_dev(train_data)




    # train_feature_minibatches, train_labels_minibatches, train_seqlens_minibatches = make_batches(train_dataset,
    #                                                                                               batch_size=Config.batch_size)
    # val_feature_minibatches, val_labels_minibatches, val_seqlens_minibatches = make_batches(train_dataset,
    #                                                                                         batch_size=len(
    #                                                                                             val_dataset[0]))

    # def pad_all_batches(batch_feature_array):
    #     for batch_num in range(len(batch_feature_array)):
    #         batch_feature_array[batch_num] = pad_sequences(batch_feature_array[batch_num])[0]
    #     return batch_feature_array

    # train_feature_minibatches = pad_all_batches(train_feature_minibatches)
    # val_feature_minibatches = pad_all_batches(val_feature_minibatches)

    # num_examples = np.sum([batch.shape[0] for batch in train_feature_minibatches])
    # num_batches_per_epoch = int(math.ceil(num_examples / Config.batch_size))

    # with tf.Graph().as_default():
    #     model = SeparationModel()
    #     init = tf.global_variables_initializer()

    #     saver = tf.train.Saver(tf.trainable_variables())

    #     with tf.Session() as session:
    #         # Initializate the weights and biases
    #         session.run(init)
    #         if args.load_from_file is not None:
    #             new_saver = tf.train.import_meta_graph('%s.meta' % args.load_from_file, clear_devices=True)
    #             new_saver.restore(session, args.load_from_file)

    #         train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)

    #         global_start = time.time()

    #         step_ii = 0

    #         for curr_epoch in range(Config.num_epochs):
    #             total_train_cost = total_train_wer = 0
    #             start = time.time()

    #             for batch in random.sample(range(num_batches_per_epoch), num_batches_per_epoch):
    #                 cur_batch_size = len(train_seqlens_minibatches[batch])

    #                 batch_cost, batch_ler, summary = model.train_on_batch(session, train_feature_minibatches[batch],
    #                                                                       train_labels_minibatches[batch],
    #                                                                       train_seqlens_minibatches[batch], train=True)
    #                 total_train_cost += batch_cost * cur_batch_size
    #                 total_train_wer += batch_ler * cur_batch_size

    #                 train_writer.add_summary(summary, step_ii)
    #                 step_ii += 1

    #             train_cost = total_train_cost / num_examples
    #             train_wer = total_train_wer / num_examples

    #             val_batch_cost, val_batch_ler, _ = model.train_on_batch(session, val_feature_minibatches[0],
    #                                                                     val_labels_minibatches[0],
    #                                                                     val_seqlens_minibatches[0], train=False)

    #             log = "Epoch {}/{}, train_cost = {:.3f}, train_ed = {:.3f}, val_cost = {:.3f}, val_ed = {:.3f}, time = {:.3f}"
    #             print(
    #             log.format(curr_epoch + 1, Config.num_epochs, train_cost, train_wer, val_batch_cost, val_batch_ler,
    #                        time.time() - start))

    #             if args.print_every is not None and (curr_epoch + 1) % args.print_every == 0:
    #                 batch_ii = 0
    #                 model.print_results(train_feature_minibatches[batch_ii], train_labels_minibatches[batch_ii],
    #                                     train_seqlens_minibatches[batch_ii])

    #             if args.save_every is not None and args.save_to_file is not None and (
    #                 curr_epoch + 1) % args.save_every == 0:
    #                 saver.save(session, args.save_to_file, global_step=curr_epoch + 1)