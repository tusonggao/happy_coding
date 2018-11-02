import numpy as np
import tensorflow as tf

import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")

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

def one_hot(vec, vals=2):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

TRAINING_NUM = 10000
TEST_NUM = 2000
DIMENSION_NUM = 5

learning_rate = 0.5

X_train = np.random.randn(TRAINING_NUM, DIMENSION_NUM) #[0, 1)
X_test = 1 + np.random.randn(TEST_NUM, DIMENSION_NUM)     #[1, 2)

y_train = np.sin(np.sum(np.power(X_train, 3), axis=1))
y_train = np.where(y_train>=0, 0, 1)
y_train_onehoted = one_hot(y_train, vals=2)

print('y_train_onehoted is', y_train_onehoted)

y_test = np.sin(np.sum(np.power(X_test, 3), axis=1))
y_test = np.where(y_test>=0, 0, 1)
y_test_onehoted = one_hot(y_test, vals=2)

print('y_test_onehoted is', y_test_onehoted)

print('y_train.shape is ', y_train.shape)
print('y_test.shape is ', y_test.shape)
print(y_test)

x_tf = tf.placeholder(tf.float32, shape=[None, DIMENSION_NUM])
y_true = tf.placeholder(tf.float32, shape=[None, 2])

y_l_1 = tf.nn.sigmoid(full_layer(x_tf, 12))
# y_l_2 = tf.nn.relu(full_layer(y_l_1, 25))
y_l_2 = tf.nn.relu(full_layer(y_l_1, 25))
# y1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w1, tf.transpose(inputs)), b1)), keep_prob=0.5)
y_l_3 = tf.nn.relu(full_layer(y_l_2, 15))
y_l_4 = tf.nn.sigmoid(full_layer(y_l_3, 2))
y_l_5 = tf.nn.softmax(y_l_4)

print('y_true shape ', y_true.get_shape(),
      'y_l_4 shape ', y_l_5.get_shape())

# loss = tf.reduce_mean(tf.square(y_true_tf - y_conv_4))
# accuracy = tf.reduce_mean(tf.cast(loss, tf.float32))

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_l_5, labels=y_true))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_l_5, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# with tf.name_scope('train') as scope:
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     train = optimizer.minimize(loss)

STEP_NUM = 101

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(STEP_NUM):
        sess.run(train_step, feed_dict={x_tf: X_train, y_true: y_train_onehoted})

        # print('y_true shape ', y_true.get_shape(),
        #       'y_l_4 shape ', y_l_4.get_shape())

        if i%50==0:
            train_acc = sess.run(accuracy,
                                 feed_dict={x_tf: X_train, y_true: y_train_onehoted})
            test_acc = sess.run(accuracy,
                                feed_dict={x_tf: X_test, y_true: y_test_onehoted})
            print('i: {0:5} train_acc: {1:.5f} test_acc: {2:.5f}'.format(
                i, train_acc, test_acc))

print('################### start lightgbm training #######################')

# w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
# b = tf.Variable(0,dtype=tf.float32,name='bias')

lgbm = lgb.LGBMClassifier(n_estimators=1000)
lgbm.fit(X_train, y_train)

y_pred = lgbm.predict(X_test)
test_acc = np.mean(y_pred==y_test)
y_pred = lgbm.predict(X_train)
train_acc = np.mean(y_pred==y_train)

print('lgbm accuracy is test_acc ', test_acc, 'train_acc', train_acc)

# lgb_train = lgb.Dataset(X_train, y_train)
# #    lgb_eval = lgb.Dataset(val_X_split, val_y_split, reference=lgb_train)
# lgb_eval = lgb.Dataset(X_train, y_train)
#
# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': {'binary_logloss'},
#     'num_leaves': 31,
#     'learning_rate': 0.08,
#     'feature_fraction': 0.75,
#     'bagging_fraction': 0.33,
#     'bagging_freq': 3,
#     'seed': 42,
# }