#!/usr/bin/env python

import tensorflow as tf
import json

# Get graph and variables file to setup service
checkpoint_dir = "./model/"
meta_graph_file = "./model/linear_model.ckpt-100.meta"

# Get request data to predict
#request_data = json.dumps({'X': 10.0, 'Y': 20.0})
request_data = {'X': 10.0, 'Y': 20.0}

with tf.Session() as sess:
    # Load model to serve
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.import_meta_graph(meta_graph_file)
        saver.restore(sess, ckpt.model_checkpoint_path)

        inputs = tf.get_collection('inputs')[0]
        outputs = tf.get_collection('outputs')[0]

        # Process predict request
        #response = sess.run(outputs, feed_dict={X: predict_x, Y: predict_y})
        #response = sess.run(outputs, feed_dict=request_data)
        response = sess.run(outputs, feed_dict={json.loads(inputs)['X']: 10, json.loads(inputs)['Y']: 20})
        print("Response: {}".format(response))
        
    else:
        print("No model found, exit")

                


    

