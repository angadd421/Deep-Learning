from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import numpy as np
import random
import matplotlib.pyplot as mat
import librosa
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
epsilon = 1e-3


def add_layer(name1,input_boolean,inputs, in_size, out_size, activation_function=None,):
    stddev_calculated=math.sqrt(2/in_size)
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=stddev_calculated))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        #outputs = BN2
        outputs=Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs


xs = tf.placeholder(tf.float32, shape=(None, 513))
ys = tf.placeholder(tf.float32, shape=(None, 513))


input_layer = add_layer("input",True,xs, 513, 1024, activation_function=tf.nn.relu)
l1 = add_layer("l1",False,input_layer, 1024, 1024, activation_function=tf.nn.relu)
l2 = add_layer("l2",False,l1, 1024, 1024, activation_function=tf.nn.relu)
prediction = add_layer("prediction",True,l2, 1024, 513, activation_function=tf.nn.relu)
loss=tf.losses.mean_squared_error(labels=ys,predictions=prediction)

train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

sess = tf.Session()


init = tf.global_variables_initializer()
saver=tf.train.Saver()

sess.run(init)
def train():
    s, sr = librosa.load('train_clean_male.wav', sr=None)
    S = librosa.stft(s, n_fft=1024, hop_length=512)
    sn, sr = librosa.load('train_dirty_male.wav', sr=None)
    X = librosa.stft(sn, n_fft=1024, hop_length=512)

    for i in range(150):
        batch_xs, batch_ys = np.transpose(np.abs(X)), np.transpose(np.abs(S))
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        print(i)

        if i % 100 == 0:
            saver.save(sess, "model_3.ckpt/")


train()


"""
Testing occurs below 
"""
saver.restore(sess, "model_3.ckpt/")
sn, sr = librosa.load('test_x_01.wav', sr=None)
X = librosa.stft(sn, n_fft=1024, hop_length=512)
T = X

batch_xs = np.transpose(np.abs(X))

saving_output = sess.run(prediction, feed_dict={xs: batch_xs})
saving_output_transpose = np.transpose(saving_output)

multiplying_factor = np.divide(T, np.abs(T))
final_answer = np.multiply(multiplying_factor, saving_output_transpose)
final_answer1 = librosa.istft(final_answer, win_length=1024, hop_length=512)
librosa.output.write_wav('test_s_01_recons.wav', final_answer1, sr)


sn, sr = librosa.load('test_x_02.wav', sr=None)
X = librosa.stft(sn, n_fft=1024, hop_length=512)
T = X

batch_xs = np.transpose(np.abs(X))

saving_output = sess.run(prediction, feed_dict={xs: batch_xs})
saving_output_transpose = np.transpose(saving_output)

multiplying_factor = np.divide(T, np.abs(T))
final_answer = np.multiply(multiplying_factor, saving_output_transpose)
final_answer1 = librosa.istft(final_answer, win_length=1024, hop_length=512)
librosa.output.write_wav('test_s_02_recons.wav', final_answer1, sr)


sess.close()
