from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import matplotlib.pyplot as mat
import os

os.environ["CUDA_VISIBLE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

gpu_options = tf.GPUOptions(allow_growth = True)
config=tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.per_process_gpu_memory_fraction = 0.33


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
epsilon = 1e-3


xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])


output1 = tf.layers.dense(xs, 1024, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(), name = 'output1')
output2 = tf.layers.dense(output1, 1024, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(), name = 'output2')
output3 = tf.layers.dense(output2, 1024, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(), name = 'output3')
output4 = tf.layers.dense(output3, 1024, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(), name = 'output4')
output5 = tf.layers.dense(output4, 1024, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(), name = 'output5')
prediction1 = tf.layers.dense(output5, 10, kernel_initializer= tf.contrib.layers.xavier_initializer(), name = 'prediction1')
prediction = tf.nn.softmax(logits = prediction1)


def compute_accuracy(prediction,v_xs, v_ys):

    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))*100
train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)

sess = tf.InteractiveSession(config=config)
init = tf.global_variables_initializer()
sess.run(init)
saver=tf.train.Saver()


def train():
    for i in range(3000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs:batch_xs, ys: batch_ys})
        if i % 100 == 0:
            epoch=i/100
            print("Current epoc:",epoch," out of 30 epochs and accuracy is:",100*compute_accuracy(prediction,
                mnist.test.images, mnist.test.labels))
    saver.save(sess,"model.ckpt/")
train()


saver.restore(sess,"model.ckpt/")

Weights1 = tf.get_default_graph().get_tensor_by_name(os.path.split(output1.name)[0] + '/kernel:0')
biases1 = tf.get_default_graph().get_tensor_by_name(os.path.split(output1.name)[0] + '/bias:0')

Weights2 = tf.get_default_graph().get_tensor_by_name(os.path.split(output2.name)[0] + '/kernel:0')
biases2 = tf.get_default_graph().get_tensor_by_name(os.path.split(output2.name)[0] + '/bias:0')

Weights3 = tf.get_default_graph().get_tensor_by_name(os.path.split(output3.name)[0] + '/kernel:0')
biases3 = tf.get_default_graph().get_tensor_by_name(os.path.split(output3.name)[0] + '/bias:0')

Weights4 = tf.get_default_graph().get_tensor_by_name(os.path.split(output4.name)[0] + '/kernel:0')
biases4 = tf.get_default_graph().get_tensor_by_name(os.path.split(output4.name)[0] + '/bias:0')

Weights5 = tf.get_default_graph().get_tensor_by_name(os.path.split(output5.name)[0] + '/kernel:0')
biases5 = tf.get_default_graph().get_tensor_by_name(os.path.split(output5.name)[0] + '/bias:0')

Weights6 = tf.get_default_graph().get_tensor_by_name(os.path.split(prediction1.name)[0] + '/kernel:0')
biases6 = tf.get_default_graph().get_tensor_by_name(os.path.split(prediction1.name)[0] + '/bias:0')


def sliced_svd(weights,d):

    s,u,v=tf.svd(weights,compute_uv=True)
    s=tf.diag(s)
    s=s[:d,:d]
    v=tf.transpose(v)
    v=v[:d,:]
    u=u[:,:d]

    v_temp = tf.matmul(s, v)
    return u,v_temp


def layer_svd(input,w,bias):

    output=tf.matmul(input, w)
    output=output+bias
    output=tf.nn.relu(output)
    #print(output.shape)
    return output


Slices=[10,20,50,100,200,1024]
accuracy=[]
for i in range(len(Slices)):
    u1,v1 = sliced_svd(Weights1,Slices[i] )
    u2,v2= sliced_svd(Weights2, Slices[i])
    u3,v3 = sliced_svd(Weights3, Slices[i])
    u4,v4= sliced_svd(Weights4, Slices[i])
    u5,v5= sliced_svd(Weights5, Slices[i])
    l1_svd=tf.matmul(tf.matmul(xs,u1),v1)
    l1_svd=l1_svd+biases1
    l1_svd = tf.nn.relu(l1_svd)

    l2_svd = tf.matmul(tf.matmul(l1_svd, u2), v2)
    l2_svd = l2_svd + biases2
    l2_svd=tf.nn.relu(l2_svd)

    l3_svd = tf.matmul(tf.matmul(l2_svd, u3), v3)
    l3_svd = l3_svd + biases3
    l3_svd = tf.nn.relu(l3_svd)

    l4_svd = tf.matmul(tf.matmul(l3_svd, u4), v4)
    l4_svd = l4_svd + biases4
    l4_svd = tf.nn.relu(l4_svd)

    l5_svd = tf.matmul(tf.matmul(l4_svd, u5), v5)
    l5_svd = l5_svd + biases5
    l5_svd = tf.nn.relu(l5_svd)

    output_svd=tf.matmul(l5_svd,Weights6)
    output_svd=output_svd+biases6
    output_svd=tf.nn.softmax(output_svd)
    accuracy.append(100*compute_accuracy(output_svd,mnist.test.images, mnist.test.labels))
    print("For slicing :",Slices[i],"Accuracy is: ", accuracy[i])


mat.plot(Slices, accuracy, '.b-')
mat.xlabel("Slicing")
mat.ylabel("Accuracy")
mat.show()
