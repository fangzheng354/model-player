#!/usr/bin/env python

import tensorflow as tf
import json

def process_request(sess, inputs, outputs, request_data):
    # request_data = {'key': 1, 'X': 10.0, 'Y': 20.0}
    # inputs = {'key_placeholder': placeholder1, 'X': placeholder2, 'Y': placeholder3}
    # outputs = {'key': identity_op, 'predict_op1': predict_op1, 'predict_op2': predict_op2}
    print("Request: {}".format(request_data))
    feed_dict = {}
    for key in inputs.keys():
        feed_dict[inputs[key]] = request_data[key]
    outputs_values = outputs.values()
    response = sess.run(outputs, feed_dict=feed_dict)
    print("Response: {}".format(response))

def main():
    # Get graph and variables file to setup service
    checkpoint_dir = "./model/"
    meta_graph_file = "./model/linear_model.ckpt-100.meta"

    sess =  tf.Session()

    # Load model to serve
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.import_meta_graph(meta_graph_file)
        saver.restore(sess, ckpt.model_checkpoint_path)

        inputs = json.loads(tf.get_collection('inputs')[0])
        outputs = json.loads(tf.get_collection('outputs')[0])

        # Get request data to predict
        with open('./predict_sample.tensor.json') as f:
            for line in f.readlines():
                # line = {'key': 1, 'X': 10.0, 'Y': 20.0}
                sample = json.loads(line)
                process_request(sess, inputs, outputs, sample)
                
    else:
        print("No model found, exit")

    sess.close()

if __name__ == "__main__":
    main()
