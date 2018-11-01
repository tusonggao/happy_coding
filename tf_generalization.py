import numpy as np
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b

TRAINING_NUM = 10000
TEST_NUM = 2000
DIMENSION_NUM = 5

learning_rate = 0.5

X_training = np.random.randn(TRAINING_NUM, DIMENSION_NUM) #[0, 1)
X_test = 1 + np.random.randn(TEST_NUM, DIMENSION_NUM)     #[1, 2)

y_training = np.sin(np.sum(np.power(X_training, 3), axis=1))
y_training = np.where(y_training>=0, 1., -1.)
y_test = np.sin(np.sum(np.power(X_test, 3), axis=1))
y_test = np.where(y_test>=0, 1., -1.)

print('y_training.shape is ', y_test.shape)
print('y_test.shape is ', y_test.shape)

print(y_test)


x_tf = tf.placeholder(tf.float32, shape=[None, DIMENSION_NUM])
y_true_tf = tf.placeholder(tf.float32,shape=None)

y_conv_1 = tf.sigmoid(full_layer(x_tf, 12))
y_conv_2 = tf.sigmoid(full_layer(y_conv_1, 25))
y_conv_3 = tf.sigmoid(full_layer(y_conv_2, 15))
y_conv_4 = full_layer(y_conv_3, 1)

loss = tf.reduce_mean(tf.square(y_true_tf - y_conv_4))
accuracy = tf.reduce_mean(tf.cast(loss, tf.float32))

with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

STEP_NUM = 500

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(STEP_NUM):
        sess.run(train, feed_dict={x_tf: X_training, y_true_tf: y_training})

        if i%50==0:
            train_loss_f = sess.run(loss, feed_dict={x_tf: X_training,
                                                    y_true_tf: y_training})
            test_loss_f = sess.run(loss, feed_dict={x_tf: X_test,
                                             y_true_tf: y_test})
            print('i is', i, 'loss is', test_loss_f, train_loss_f)

print('get last')

# w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
# b = tf.Variable(0,dtype=tf.float32,name='bias')