#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cPickle as pickle
import os
import functools
import tensorflow as tf
import sys

np.set_printoptions(threshold=np.nan)

script_path = os.path.dirname(os.path.realpath(__file__))

data_path = ''.join([script_path, "/mit_bih_delight_subsampled.pkl"])

if len(sys.argv) > 1:
	_save_path = sys.argv[1]

else:
	_save_path = ''.join([script_path, '/logs/vars/tmp_delight/model.ckpt'])

data_file = open(data_path, 'r')

true_data = pickle.load(data_file)

data_file.close()


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

# Class definition modified from Danijar Hafner's example at https://gist.github.com/danijar/3f3b547ff68effb03e20c470af22c696

class VariableSequenceClassification:

    def __init__(self, data, target, num_hidden=150, num_layers=2, num_fc=2, fc_len=20):
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._num_fc = num_fc
        self._fc_len = fc_len
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        subcells = []
        for i in range(self._num_layers):
                if i == 0 and self._num_layers > 1:
		    subcells.append(tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self._num_hidden, initializer=tf.contrib.layers.xavier_initializer()), input_keep_prob = 0.8))
                else:
                    subcells.append(tf.nn.rnn_cell.LSTMCell(self._num_hidden, initializer = tf.contrib.layers.xavier_initializer()))
        main_cell = tf.nn.rnn_cell.MultiRNNCell(subcells, state_is_tuple=True)
        # Recurrent network.
        output, _ = tf.nn.dynamic_rnn(
            main_cell,
            self.data,
            dtype=tf.float32,
            sequence_length=self.length,
        )
        last = self._last_relevant(output, self.length)
        if self._num_fc == 0:
            last_before_softmax = last
            out_num = self._num_hidden
        else:
            fc_layers = []
            if self._num_fc == 1:
            	fc_layers.append(tf.contrib.layers.fully_connected(last, self._fc_len))
            else:
                fc_layers.append(tf.contrib.layers.fully_connected(last, self._fc_len, activation_fn=tf.nn.sigmoid))
                for l in range(1, self._num_fc-1):
                    fc_layers.append(tf.contrib.layers.fully_connected(fc_layers[l-1], self._fc_len, activation_fn=tf.nn.sigmoid))
                fc_layers.append(tf.contrib.layers.fully_connected(fc_layers[self._num_fc-2], self._fc_len))
            last_before_softmax = fc_layers[self._num_fc-1]
            out_num = self._fc_len

        # Softmax layer.
        weight, bias = self._weight_and_bias(
            out_num, int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last_before_softmax, weight) + bias)
        prediction = tf.clip_by_value(prediction, 1e-3, 1.0-1e-3)
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.005
	momentum = 0.0
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.get_variable("weight", shape=[in_size,out_size], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.constant(0.1, shape=[out_size], dtype=tf.float32)
        return weight, tf.Variable(bias)

    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant




if __name__ == '__main__':
    all_data = true_data
    train = all_data[0]
    test = all_data[1]
    conf_weights =  np.sum(test[1], axis=0)
    batch_size = 500
    no_examples, rows, row_size = test[0].shape
    num_classes = len(test[1][0])
    no_batches = no_examples/batch_size
    data = tf.placeholder(tf.float32, [None, rows, row_size])
    target = tf.placeholder(tf.float32, [None, num_classes])
    model = VariableSequenceClassification(data, target)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(test[0].shape,test[1].shape)
    saver = tf.train.Saver()
    sess.run(model.error, feed_dict={data:test[0][:batch_size], target:test[1][:batch_size]})
    saver.restore(sess, _save_path)
    test_pred_targets = np.zeros((batch_size*no_batches), dtype=np.int32)
    test_pred_int = np.zeros((batch_size*no_batches), dtype=np.int32)
    error_sum = 0.0
    for i in range(no_batches):
        batch_data = test[0][i*batch_size:(i+1)*batch_size]
        batch_target = test[1][i*batch_size:(i+1)*batch_size]
        error = sess.run(model.error, feed_dict={data:batch_data, target:batch_target})
        error_sum+=error*100.0
        pred = sess.run(model.prediction, feed_dict={data:batch_data, target:batch_target})
        for j in range(i*batch_size,(i+1)*batch_size):
            test_pred_targets[j] = np.argmax(test[1][j])
            test_pred_int[j] = np.argmax(pred[j-i*batch_size])
        print('Test batch {:2d} error {:3.1f}%'.format(i+1, 100*error))

    print('Total error: {:3.1f}%'.format(error_sum/no_batches))

    conf_mat = tf.confusion_matrix(
        labels=tf.convert_to_tensor(test_pred_targets,dtype=tf.int32),
        predictions=tf.convert_to_tensor(test_pred_int,dtype=tf.int32),
        num_classes = num_classes)
    print('Confusion matrix:')
    print(sess.run(conf_mat, feed_dict=None))

    batch_size = 100
    no_examples, rows, row_size = train[0].shape
    num_classes = len(train[1][0])
    no_batches = no_examples/batch_size
		
    if False:
        for i in range(no_batches):
            batch_data = train[0][i*batch_size:(i+1)*batch_size]
            batch_target = train[1][i*batch_size:(i+1)*batch_size]
            error = sess.run(model.error, feed_dict={data:batch_data, target:batch_target})
            print('Train batch {:2d} error {:3.1f}%'.format(i+1, 100*error))
