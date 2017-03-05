from tensorflow.examples.tutorials.mnist import input_data

"""
Import the MINST data set.
"""
mnist = input_data.read_data_sets("/training_data/MINST_data/", one_hot=True)

import tensorflow as tf

"""
None means that the dimension of vectors can be
of any length.
The images are 28x28, for processing 784 tensors
are needed.
"""
x = tf.placeholder(tf.float32, [None, 784])

"""
A Variable is a modifiable tensor within the TensorFlow
framework. The Variable can be used, and alo be modified
by the computation.

W = Weight
B = Bias

The number 784 represents the pixels in the 28x28 image.
The 10 is for the total number of classes (digits 0-9).

W
The 784 is dimensional image vector. The 10 is to create
10-dimensional vectors for the evidence for the classes.

b has a 10 so it can be added to the output of all of
the 10 class streams.

The dimensions are filled with zeros for they will be
trained later on.
"""
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

"""
Implementing the model.
y = softmax(Wx + b)

x is multiplied by W with tf.matmul. After that b is
added, and apply the tf.nn.softmax function.
"""
y = tf.nn.softmax(tf.matmul(x, W) + b)

"""
Implementing the cross-entropy.

y_ is the placeholder for the final correct answer.

1. First the logarithm of each element of y is computed.
2. Then each element of y_ is multiplied by the
corresponding element of tf.log(y).
3. Then tf.reduce_sum add the elements to the second
dimension because of the param reduction_indices=[1].

This setup can be numerically unstable. Advised is to
use tf.nn.softmax_cross_entropy_with_logits.
"""
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

"""
While training, TensorFlow will descent the gradient
value with 0.5 to reduce the cost.

Backpropagation is added automatically by TensorFlow.
"""
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

"""
Launch the model
"""
sess = tf.InteractiveSession()
"""
This function is used to initialize the variables that
were created within this script.
"""
tf.global_variables_initializer().run()

"""
Train with running the training step 1000 times.

With each bash, a random 100 data points are provided
from the training set. With running the train_step the
data i nthe placeholder is replaced.
"""
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

"""
The tf.argmax function can be to retrieve the highest
entry in a tensor along some axis.

tf.argmax(y, 1) = label the model thinks is most
likely for each input.
tf.argmax(y_, 1) = the correct label.

tf.equal is used to check of the labels match.

In this construct, correct_prediction will be filled
with a list of booleans. To determine the fraction
which is correct, the list is cast to floating points
and after that the mean is taken.
[True, False, True, True] = [1, 0, 1, 1] = 0.75

- tf.cast is used to cast the list.
- tf.float32 is the type the list will be cast to
- tf.reduce_mean calculates the mean of the list
"""
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
Shows the accuracy on our test data.
"""
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
