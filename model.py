#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import math
import random
import os
# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import xrange as range

import pdb
from time import gmtime, strftime

from config import Config

class ClassificationModel():
    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None, None, Config.num_final_features), name='inputs')
        self.targets_placeholder = tf.placeholder(tf.float32, shape=(None, Config.num_classes), name='targets')


    def create_feed_dict(self, inputs_batch, targets_batch): 
        return {
            self.inputs_placeholder: inputs_batch,
            self.targets_placeholder: targets_batch,
        }


    def add_prediction_op(self):
        xavier = tf.contrib.layers.xavier_initializer()

        cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(Config.num_hidden, 
            input_size=Config.num_final_features) for _ in range(Config.num_layers)], state_is_tuple=False)

        output, state = tf.nn.dynamic_rnn(cell, self.inputs_placeholder, dtype=tf.float32)
        
        self.W = tf.get_variable('W', shape=[Config.num_hidden, Config.num_classes], initializer=xavier)
        self.b = tf.get_variable('b', shape=[Config.num_classes])

        # batch_size = tf.shape(output)[0]
        output_seq_length = tf.shape(output)[1]
        # batch_range = tf.range(batch_size)
        # indices = tf.stack([batch_range, output_seq_length - 1], axis=1)
        last_output = output[:,output_seq_length - 1,:]
        self.logits = tf.matmul(last_output, self.W) + self.b
        # self.threshold_train_model.add_data(self.logits, self.targets_placeholder)


    def add_loss_op(self):
        l2_cost = 0.0

        for var in tf.trainable_variables():
            if len(var.get_shape().as_list()) != 1:
                l2_cost += tf.nn.l2_loss(var)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets_placeholder, name='cost'))
        self.logits = tf.nn.softmax(self.logits)

        self.loss = Config.l2_lambda * l2_cost + cost


    def add_training_op(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=Config.lr).minimize(self.loss)


    def add_summary_op(self):
        tf.summary.scalar("loss", self.loss)

        self.merged_summary_op = tf.summary.merge_all()


    # This actually builds the computational graph 
    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_training_op()     
        self.add_summary_op()
        

    def train_on_batch(self, session, train_inputs_batch, train_targets_batch, train=True):
        feed = self.create_feed_dict(train_inputs_batch, train_targets_batch)
        batch_cost, logits, summary = session.run([self.loss, self.logits, self.merged_summary_op], feed)

        if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
            return 0  # should cause error
        if train:
            _ = session.run([self.loss, self.optimizer], feed)

        return batch_cost, logits, summary


    def predict_on_batch(self, session, test_inputs_batch, test_targets_batch):
        logistic, thresholds = session([self.logistic, self.thresholds], feed)
        classified_labels = tf.squeeze(tf.where(logistic > thresholds, tf.ones(self.thresholds.shape), tf.zeros(self.thresholds.shape)))
        logistic_shape = tf.shape(self.logistic)

        target_shaped_ones = tf.ones(tf.shape(test_targets_batch))
        target_shaped_zeros = tf.zeros(tf.shape(test_targets_batch))
        
        true_positive_matrix = tf.where(tf.logical_and(\
            tf.equal(test_targets_batch, tf.ones([1])), tf.equal(classified_labels, tf.ones([1]))), target_shaped_ones, target_shaped_zeros)
        true_positives = tf.reduce_sum(true_positive_matrix)

        true_negative_matrix = tf.where(tf.logical_and(\
            tf.equal(test_targets_batch, tf.zeros([1])), tf.equal(classified_labels, tf.zeros([1]))), target_shaped_ones, target_shaped_zeros)
        true_negatives = tf.reduce_sum(true_negative_matrix)

        false_positive_matrix = tf.where(tf.logical_and(\
            tf.equal(test_targets_batch, tf.zeros([1])), tf.equal(classified_labels, tf.ones([1]))), target_shaped_ones, target_shaped_zeros)
        false_positives = tf.reduce_sum(false_positive_matrix)

        false_negative_matrix = tf.where(tf.logical_and(\
            tf.equal(test_targets_batch, tf.ones([1])), tf.equal(classified_labels, tf.zeros([1]))), target_shaped_ones, target_shaped_zeros)
        false_negatives = tf.reduce_sum(false_negative_matrix)
        
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        f_point_five = 1.25 * precision * recall / (0.25 * precision + recall)

        return f_point_five

    def __init__(self):
        self.build()
