import tensorflow as tf

var1 = tf.Variable(0.1)
var2 = tf.Variable(0.2)

init_op = tf.initialize_all_variables()

saver = tf.train.Saver()
#saver = tf.train.Saver({"var1": var1, "var2": var2})

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "/tmp/tensorflow_checkpoint.ckpt")
    print(sess.run(var1))
