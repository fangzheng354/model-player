import tensorflow as tf

var1 = tf.Variable(0.2)
var2 = tf.Variable(0.3)

init_op = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "/tmp/tensorflow_checkpoint.ckpt")
    print(sess.run(var1))
