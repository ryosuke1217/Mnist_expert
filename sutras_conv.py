# -*- coding: utf-8 -*-

import tensorflow as tf

import input_data

tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

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
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

# 重みを標準偏差0.1の正規分布で初期化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# バイアスを標準偏差0.1の正規分布で初期化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def main():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    with tf.Graph().as_default():

        # x:入力値 y_:出力値のplaceholderセット
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        # 1次元でデータを返すので、28×28×1にreshape
        #[バッチ数。縦、横、チャネル数]
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        # 畳み込み処理(1)→フィルター層は32個
        W_conv1 = weight_variable([5, 5, 1, 32])
        #32個のバイアスをセット
        b_conv1 = bias_variable([32])
        #畳み込み演算後に、Relu関数適用
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1))

        #プーリング処理(1)
        # 2×2のMAXプーリングをすると縦横半分の大きさになる
        h_pool1 = max_pool_2x2(h_conv1)

        # 畳み込み処理(2)→フィルターは64個
        #チャンネル数が32なのは、畳み込み層１のフィルター数が32だから
        #32個フィルターがあると、出力結果が[-1, 28, 28, 32]というshapeになる
        #入力のチャンネル数と重みのチャンネル数を合わせる。
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        #プーリング処理(2)
        h_pool2 = max_pool_2x2(h_conv2)

        #全結合処理（ノードの数は1024個）
        #2x2MAXプーリングを2回やってるので、この時点で縦横が、28/(2*2)の7になっている。
        #h_pool2のshapeは、[-1, 7, 7, 64]となっているので、7*7*64を入力ノード数とみなす。  
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        #全結合層の入力仕様に合わせて、2次元にreshape
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        print('h_pool2_flat',h_pool2_flat)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        print('h_fc1',h_fc1)

        #ドロップアウト
        #keep_probは、ドロップアウトさせない率
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        #出力層
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
        print('y_conv',y_conv)

        #評価処理
        train_op = training(loss(y_conv, y_), FLAGS.learning_rate)
        acc = accuracy(y_conv, y_)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('log', sess.graph)
            for i in range(1000):
                batch = mnist.train.next_batch(50)
                if i % 100 == 0:
                    train_accuracy = acc.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                    print("step %d, training_accuracy %g" % (i, train_accuracy))
                sess.run(train_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

                # summary_str = sess.run(summary_op, feed_dict={x: mnist.train.images, y_: mnist.train.labels, keep_prob: 1.0})
                # summary_writer.add_summary(summary_str, i)

            
            print("test_accuracy %g" % acc.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
    main()