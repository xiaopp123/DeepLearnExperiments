# -*- coding: utf-8 -*-

import tensorflow as tf
import collections
import random
import numpy as np
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import static_rnn


#parameters
data_path = './data.txt'
n_input = 3
hidden_num = 500
learning_rate = 0.0001
training_iters = 50000
display_step = 100

log_path = './log'
writer = tf.summary.FileWriter(log_path)


def get_words(data_path):
    with open(data_path) as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    data = [word for i in range(len(data)) for word in data[i].split()]

    return data


def build_dictionay(words):
    dictionary = {}
    data_cnt = collections.Counter(words)
    words = data_cnt.keys()
    print(len(data_cnt))
    for i in range(len(words)):
        dictionary[words[i]] = len(dictionary)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return dictionary, reversed_dictionary


words = get_words(data_path)
#print(words)
dictionary, reversed_dictionary = build_dictionay(words)
#print(dictionary)
#print(reversed_dictionary)
vocab_size = len(dictionary)

weights = {
    'out': tf.Variable(tf.random_normal([hidden_num, vocab_size]))
}
bias = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])


def rnn(x, weights, bias):
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_input, 1)

    rnn_cell = LSTMCell(hidden_num)
    outputs, states = static_rnn(rnn_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + bias['out']


pre = rnn(x, weights, bias)


#loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pre))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

#model evaluation
correct_pre = tf.equal(tf.argmax(pre, 1), tf.argmax(y, 1))
accurary = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

#initialize
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #writer.add_graph(sess.graph)
    offset = random.randint(0, n_input + 1)
    end_offset = offset + n_input
    acc_total = 0
    loss_total = 0

    step = 0

    while step < training_iters:
        if offset > len(words) - end_offset:
            offset = random.randint(0, n_input + 1)

        symbols_in_keys = [[dictionary[words[i]]] for i in range(offset, offset + n_input)]
        #print(symbols_in_keys)
        #symbols_in_keys = tf.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
        symbols_in_keys = np.reshape(symbols_in_keys, [-1, n_input, 1])

        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[words[offset + n_input]]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

        _, acc, loss, onehot_pre = sess.run([optimizer, accurary, cost, pre],\
                                            feed_dict={x: symbols_in_keys, y: symbols_out_onehot})

        loss_total += loss
        acc_total += acc
        if step % display_step == 0:
            print("Iter= " + str(step) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " +\
                  "{:.2f}%".format(100*acc_total/display_step))

            loss_total = 0
            acc_total = 0

            symbols_in = [words[i] for i in range(offset, offset + n_input)]
            symbol_out = words[offset + n_input]
            pred_out = reversed_dictionary[int(tf.argmax(onehot_pre, 1).eval())]
            print('%s - [%s] vs [%s]' % (symbols_in, symbol_out, pred_out))

        step += 1
        offset += n_input + 1

    print("Optimization Finished!")
