import os
import sys
import json
import time
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
from color import colors

n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

def rnn_model(x, weights, biases):
    """RNN (LSTM or GRU) model for image"""
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, axis=0, num_or_size_splits=n_steps)

    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights) + biases

def train():
    """Train an image classifier"""
    print(colors.info('Step 0: load image data and training parameters'))
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    if len(sys.argv) > 1:
        parameter_file = sys.argv[1]
    else:
        parameter_file = 'parameters.json'
    params = json.loads(open(parameter_file).read())

    print(colors.info('Step 1: build a rnn model for image'))
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])

    weights = tf.Variable(tf.random_normal([n_hidden, n_classes]), name='weights')
    biases = tf.Variable(tf.random_normal([n_classes]), name='biases')
    pred = rnn_model(x, weights, biases)

    # optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(cost)
    # accuracy
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print(colors.info('Step 2: train the image classification model'))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch = 1

        print(colors.info('Step 2.0: create a directory for saving model files'))
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model.ckpt")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

        print(colors.info('Step 2.1: train the image classifier batch by batch'))
        while epoch < params['training_iters']:
            batch_x, batch_y = mnist.train.next_batch(params['batch_size'])
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((params['batch_size'], n_steps, n_input))
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            # Step 2.2: save the model
            if epoch % params['display_step'] == 0:
                saver.save(sess, checkpoint_prefix, global_step=epoch)
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print(colors.header('Iter: {}, Loss: {:.6f}, Accuracy: {:.6f}'
                                    .format(epoch * params['batch_size'], loss, acc)))
            epoch += 1
        print(colors.success('The training is done'))

        # Step 3: test the model
        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        test_label = mnist.test.labels[:test_len]
        print(colors.header("Testing Accuracy: {}"
                            .format(sess.run(accuracy, feed_dict={x: test_data, y: test_label}))))

if __name__ == '__main__':
    train()
