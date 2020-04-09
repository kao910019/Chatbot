# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.python.client import timeline

from hparams import HParams

# python server.py --job_name=ps --task_index=0 > ps0.log &
# python server.py --job_name=wk --task_index=0 > worker0.log &
# python server.py --job_name=wk --task_index=1 > worker1.log &
# Use 'chrome://tracing/' to load timeline_client.json file

def main():
    hparams = HParams().hparams
    params_server = hparams.ps_hosts.split(",")
    worker_server = hparams.worker_hosts.split(",")
    # create cluster
    cluster = tf.train.ClusterSpec({"ps": params_server, "worker": worker_server})
    # create the server
    server = tf.train.Server(cluster, job_name=hparams.job_name, task_index=hparams.task_index)
    server.join()

    with tf.device('/job:ps/task:0/cpu:0'):
        input_data = tf.Variable(
                [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]],
                name="input_data")
        b = tf.Variable([[1.], [1.], [2.]], name="w")

    inputs = tf.split(input_data, 2)
    outputs = []

    # Track statistics of the run using Timeline
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    if hparams.job_name == 'ps':
        server.join()
    else:
        # 图内复制，只在worker0上创建client
        with tf.Session("grpc://localhost:2223") as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            for i in range(2):
                with tf.device("/job:worker/task:%d/gpu:0" % i):
                    print("now is worker %d: " % i)
                    print(sess.run(inputs[i]))
                    outputs.append(tf.matmul(inputs[i], b))
            with tf.device('/job:ps/task:0/cpu:0'):
                output = tf.concat(outputs, axis=0)
                print(sess.run(output, options=run_options, run_metadata=run_metadata))

            # for tensorboard
            tf.summary.FileWriter("logs/", sess.graph)

            # Create timeline and write it to a json file
            tl = timeline.Timeline(step_stats=run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline_client.json', 'w') as f:
                f.write(ctf)
                
if __name__ == "__main__":
    main()