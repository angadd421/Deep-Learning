from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import numpy as np
import random
import matplotlib.pyplot as mat
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
epsilon = 1e-3
def add_layer(inputs, in_size, out_size, activation_function=None,):
    stddev_calculated=math.sqrt(2/in_size)
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size],stddev=stddev_calculated))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    W_plus_b = tf.matmul(inputs, Weights) + biases
    batch_mean2, batch_var2 = tf.nn.moments(W_plus_b, [0])
    scale = tf.Variable(tf.ones([out_size]))
    beta2 = tf.Variable(tf.zeros([out_size]))
    normalized_input = tf.nn.batch_normalization(W_plus_b, batch_mean2, batch_var2, beta2, scale,epsilon)
    normalized_input = tf.nn.dropout(normalized_input, 0.95)
    if activation_function is None:
        outputs = normalized_input
    else:
        outputs = activation_function(normalized_input,)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

"""
Architecture of the DNN
"""
input_layer = add_layer(xs, 784, 1024, activation_function=tf.nn.relu)
l1 = add_layer(input_layer, 1024, 1024, activation_function=tf.nn.relu)
l2 = add_layer(l1, 1024, 1024, activation_function=tf.nn.relu)
l3 = add_layer(l2, 1024, 1024, activation_function=tf.nn.relu)
l4 = add_layer(l3, 1024, 1024, activation_function=tf.nn.relu)
l5 = add_layer(l4, 1024, 1024, activation_function=tf.nn.relu)
prediction = add_layer(l5, 1024, 10,  activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))*100


#Using exponential Decay
step = tf.Variable(0, trainable=False)
rate = tf.train.exponential_decay(0.001, step, 1, 0.9999)
train_step = tf.train.AdamOptimizer(learning_rate=rate).minimize(cross_entropy)

sess = tf.Session()


init = tf.global_variables_initializer()
sess.run(init)
saver=tf.train.Saver()

def train():
    for i in range(5000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs:batch_xs, ys: batch_ys})
        if i % 100 == 0:
            epoch=i/100
            print("Current epoc:",epoch," out of 50 epochs and accuracy is:",100*compute_accuracy(
                mnist.test.images, mnist.test.labels))
    saver.save(sess,"model.ckpt/")
#train()



"""
Following is the implementation of testing code.

"""

saver.restore(sess,"model.ckpt/")
sample=random.sample(range(0,len(mnist.test.images)),1000)

saving_output=sess.run(prediction, feed_dict={xs:mnist.test.images[sample]})
saving_output1=sess.run(l1, feed_dict={xs:mnist.test.images[sample]})
saving_output2=sess.run(l2, feed_dict={xs:mnist.test.images[sample]})
saving_output3=sess.run(l3, feed_dict={xs:mnist.test.images[sample]})
saving_output4=sess.run(l4, feed_dict={xs:mnist.test.images[sample]})
saving_output5=sess.run(l5, feed_dict={xs:mnist.test.images[sample]})


"""
Following Function plots 10 images of each class corresponding
to the class label. This can be done for output of the nural net 
or for hidden layers.

"""
def class_vs_images(hiddenlayer,temp1,name):
    fig,axis=mat.subplots(10,10)

    dic1 = {}
    for i in range(0,len(temp1)):
        if temp1[i] not in dic1:
            dic1[temp1[i]]=[sample[i]]
        else:
            dic1[temp1[i]].append(sample[i])
    if(hiddenlayer==False):
        for i in dic1.keys():
            for j in range(10):
                axis[i,j].imshow(mnist.test.images[dic1[i][j],:].reshape(28,28))
    else:
        dic2 = {}
        count = 0
        for i in dic1.keys():
            if i not in dic2.keys():
                dic2[i] = count
                count += 1

        for i in dic1.keys():
            if dic2[i]>=10:
                break
            for j in range(len(dic1[i])):
               if j<10:
                axis[dic2[i], j].imshow(mnist.test.images[dic1[i][j], :].reshape(28, 28))




class_vs_images(True,sess.run(tf.argmax(saving_output1, 1)),"Layer 1")
class_vs_images(True,sess.run(tf.argmax(saving_output2, 1)),"Layer 2")
class_vs_images(True,sess.run(tf.argmax(saving_output3, 1)),"Layer 3")
class_vs_images(True,sess.run(tf.argmax(saving_output4, 1)),"Layer 4")
class_vs_images(True,sess.run(tf.argmax(saving_output5, 1)),"Layer 5")
class_vs_images(False,sess.run(tf.argmax(saving_output, 1)),"Final Output")



"""
The method below takes testing data, labels , to be reduced dimensions and title as input.
"""
def PCA_tSNE_plot(data_given, label_given, dimension,name):

    tsne = TSNE(n_components=dimension)
    data_fit_tsne = tsne.fit_transform(data_given)

    pca = PCA(n_components=dimension)
    data_fit_pca = pca.fit_transform(data_given)

    final_tsne = []
    final_pca=[]
    data_stack_tsne = np.hstack((data_fit_tsne, label_given.reshape(1000, 1)))
    data_stack_pca = np.hstack((data_fit_pca, label_given.reshape(1000, 1)))

    for x in sorted(np.unique(data_stack_tsne[..., 2])):
        final_tsne.append([np.average(data_stack_tsne[np.where(data_stack_tsne[..., 2] == x)][..., 0]),
                        np.average(data_stack_tsne[np.where(data_stack_tsne[..., 2] == x)][..., 1]), x])

    for x in sorted(np.unique(data_stack_pca[..., 2])):
        final_pca.append([np.average(data_stack_pca[np.where(data_stack_pca[..., 2] == x)][..., 0]),
                        np.average(data_stack_pca[np.where(data_stack_pca[..., 2] == x)][..., 1]), x])

    final_tsne = np.array(final_tsne)
    final_pca = np.array(final_pca)

    fig, axis = mat.subplots(1, 2)
    axis[0].scatter(data_fit_tsne[:, 0], data_fit_tsne[:, 1], c=label_given, alpha=0.2)
    tSNE_title = "tSNE for " + name
    axis[0].set_title(tSNE_title)

    axis[0].scatter(data_fit_tsne[:, 0], data_fit_tsne[:, 1], c=label_given, alpha=0.2)
    for i in range(final_tsne.shape[0]):
      axis[0].annotate(i, xy=(final_tsne[i, 0], final_tsne[i, 1]), size=20, color='red')

    axis[0].plot()
    axis[0].set_title(tSNE_title)

    axis[1].scatter(data_fit_pca[:, 0], data_fit_pca[:, 1], c=label_given, alpha=0.2)
    pca_title = "PCA for " + name
    axis[1].set_title(pca_title)

    axis[1].scatter(data_fit_pca[:, 0], data_fit_pca[:, 1], c=label_given, alpha=0.2)
    for i in range(final_pca.shape[0]):
        axis[1].annotate(i, xy=(final_pca[i, 0], final_pca[i, 1]), size=20, color='red')
    axis[1].plot()
    axis[1].set_title(pca_title)



PCA_tSNE_plot(mnist.test.images[sample],np.argmax(mnist.test.labels[sample], axis = 1),2,"MNIST Data")
PCA_tSNE_plot(saving_output1,np.argmax(mnist.test.labels[sample], axis = 1),2,"Layer1")
PCA_tSNE_plot(saving_output2,np.argmax(mnist.test.labels[sample], axis = 1),2,"Layer2")
PCA_tSNE_plot(saving_output3,np.argmax(mnist.test.labels[sample], axis = 1),2,"Layer3")
PCA_tSNE_plot(saving_output4,np.argmax(mnist.test.labels[sample], axis = 1),2,"Layer4")
PCA_tSNE_plot(saving_output5,np.argmax(mnist.test.labels[sample], axis = 1),2,"Layer5")
PCA_tSNE_plot(saving_output,np.argmax(mnist.test.labels[sample], axis = 1),2," Output")
mat.show()

sess.close()
