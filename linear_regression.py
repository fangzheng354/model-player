#!/usr/bin/env python

import tensorflow as tf
import numpy as np

flags = tf.app.flags
flags.DEFINE_string("mode", "train", "Mode to run, options: train or predict")
FLAGS = flags.FLAGS

train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")

global_step = tf.Variable(0, name='global_step', trainable=False)
loss = tf.square(Y - tf.mul(X, w) - b)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)
predict_op = tf.mul(X, w) + b
checkpoint_period = 10000
saver = tf.train.Saver(sharded=True)
import json
tf.add_to_collection("inputs", json.dumps({'X': X.name, 'Y': Y.name}))
tf.add_to_collection("outputs", predict_op.name)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    tf.train.write_graph(sess.graph.as_graph_def(), "./", "linear_graph.pb", as_text=False)

    if FLAGS.mode == "train":
        for i in range(1000):
            for (x, y) in zip(train_X, train_Y):
                sess.run(train_op, feed_dict={X: x, Y: y})

            if i % checkpoint_period == 0:
                saver.save(sess, "./model/linear_model.ckpt", global_step=global_step)
        
    elif FLAGS.mode == "predict":
        ckpt = tf.train.get_checkpoint_state("./model/")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            predict_x = 10
            predict_y = 20
            predict_result = sess.run(predict_op, feed_dict={X: predict_x, Y: predict_y})
            print("x: {}, y: {}, predict result: {}".format(predict_x, predict_y, predict_result))
        else:
            print("No model found, exit")
    else:
        print("Invalide mode, exit")

    print(sess.run(w))
    print(sess.run(b))
