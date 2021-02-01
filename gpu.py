import re

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.python.client import device_lib


def init_gpus():
    # limit memory used by gpu to 4GB
    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        config = tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4*1024)

        tf.config.experimental.set_virtual_device_configuration(gpu, [config])
        tf.config.experimental.list_logical_devices('GPU')


def enable_mixed_precision():
    return

    gpus = device_lib.list_local_devices()
    gpus = [g for g in gpus if g.device_type == 'GPU']

    for gpu in gpus:
        description = gpu.physical_device_desc.split('name: ')[-1]
        description = description.split(',')[0]

        compute = gpu.physical_device_desc.split('compute capability: ')[-1]

        if float(compute) >= 7:
            print('%s has tensor cores, enabling mixed precision' % description)

            _enable_mixed_precision()

            break
        else:
            print('%s doesn\'t have tensor cores, not enabling mixed precision' % description)


def _enable_mixed_precision():
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
