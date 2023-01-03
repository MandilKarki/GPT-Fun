"""
-- Created by: Ashok Kumar Pant
-- Created on: 9/29/21
"""
if __name__ == '__main__':
    import tensorflow as tf
    import os

    tpu_address = 'grpc://10.128.0.6:8470'
    # tpu_address = 'treeleaf-tpu-1'
    # tpu_address = 'grpc://treeleaf-tpu-1'
    # tpu_address = 'grpc://35.238.249.133'

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)

    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
