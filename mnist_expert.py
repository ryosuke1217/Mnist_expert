# -*- coding: utf-8 -*-

import tensorflow as tf

import input_data

tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
tf.app.flags.DEFINE_string('log_dir', 'log', 'log保存先')
tf.app.flags.DEFINE_integer('max_step', 2000, '訓練ステップ数')

FLAGS = tf.app.flags.FLAGS

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def loss(logits, labels):
    # 交差エントロピーの計算
    loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits + 1e-10), reduction_indices=[1]))
    tf.summary.scalar("loss", loss)
    return loss

def convolution(images):
    images = tf.reshape(images, [-1, 28, 28, 1])

    output = tf.layers.conv2d(images, filters=32, kernel_size=[5,5], strides=[2,2], padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output, pool_size=[2,2], strides=[2,2], padding='SAME')

    output = tf.layers.conv2d(images, filters=64, kernel_size=[5,5], strides=[2,2], padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output, pool_size=[2,2], strides=[2,2], padding='SAME')

    output = tf.contrib.layers.flatten(output)

    output = tf.layers.dense(output, 1024)
    output = tf.nn.relu(output)

    output = tf.layers.dense(output, 256)
    output = tf.nn.relu(output)

    output = tf.layers.dense(output, 10)
    output = tf.nn.softmax(output)
    
    return output

def main():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    with tf.Graph().as_default():

        # x:入力値 y_:出力値のplaceholderセット
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        
        with tf.Session() as sess:
            output = convolution(x)
            loss_op = loss(output, y_)
            train_op = training(loss_op, FLAGS.learning_rate)
            accuracy_op = accuracy(output, y_)
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            summary_op = tf.summary.merge_all()

            sess.run(tf.global_variables_initializer())

            for i in range(FLAGS.max_step):
                batch = mnist.train.next_batch(50)
                sess.run(train_op, feed_dict={x: batch[0], y_: batch[1]})
                if i % 100 == 0:
                    _accuracy, _loss = sess.run([accuracy_op, loss_op], feed_dict={x: batch[0], y_: batch[1]})
                    print("step %d, training_accuracy %g, train_loss %g" % (i, _accuracy, _loss))
                if i % 100 == 0 or i == FLAGS.max_step - 1:
                    summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1]})
            
            test_accuracy = sess.run(accuracy_op, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

            print("test_accuracy %g" % (test_accuracy))

if __name__ == '__main__':
    main()