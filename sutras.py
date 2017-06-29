# -*- coding: utf-8 -*-

import tensorflow as tf

import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

def main():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]), name='weight')
        b = tf.Variable(tf.zeros([10]), name='bias')
        y = tf.nn.softmax(tf.matmul(x,W) + b)
        y_ = tf.placeholder(tf.float32, [None, 10])
        train_op = training(loss(y, y_), FLAGS.learning_rate)
        acc = accuracy(y, y_)
        saver = tf.train.Saver()

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('log', sess.graph)
            for i in range(1000):
                batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})
                train_accuracy = sess.run(acc, feed_dict={x: mnist.train.images, y_: mnist.train.labels})

                print("step %d, training accuracy %g" % (i, train_accuracy))

                summary_str = sess.run(summary_op, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
                summary_writer.add_summary(summary_str, i)

            acc = sess.run(acc, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print("accuracy", acc)

def accuracy(logits, labels):
    """ 正解率(accuracy)を計算する関数
    引数: 
      logits: inference()の結果
      labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]
    返り値:
      accuracy: 正解率(float)
    """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy

def training(loss, learning_rate):
    """ 訓練のOpを定義する関数
    引数:
      loss: 損失のtensor, loss()の結果
      learning_rate: 学習係数
    返り値:
      train_step: 訓練のOp
    """
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def loss(logits, labels):
    """ lossを計算する関数
    引数:
      logits: ロジットのtensor, float - [batch_size, NUM_CLASSES]
      labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]
    返り値:
      cross_entropy: 交差エントロピーのtensor, float
    """

    # 交差エントロピーの計算
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
    # TensorBoardで表示するよう指定
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy


if __name__ == '__main__':
    main()