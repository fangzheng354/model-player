#!/usr/bin/env python

import tensorflow as tf
import json

def process_request(sess, inputs, outputs, request_data):
    # request_data = {'X': 10.0, 'Y': 20.0}
    # inputs = {'X': placeholder1, 'Y': placeholder2}
    # outputs = {'predict_op1': predict_op1, 'predict_op2': predict_op2}
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

    # Get request data to predict
    request_data = {'X': 10.0, 'Y': 20.0}

    sess =  tf.Session()

    # Load model to serve
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.import_meta_graph(meta_graph_file)
        saver.restore(sess, ckpt.model_checkpoint_path)

        inputs = json.loads(tf.get_collection('inputs')[0])
        outputs = json.loads(tf.get_collection('outputs')[0])

        # Process predict request
        process_request(sess, inputs, outputs, request_data)
    else:
        print("No model found, exit")

    sess.close()

if __name__ == "__main__":
    main()
