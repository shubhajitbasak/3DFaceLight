import os
from modules.models import PRNet_Model
from modules.utils import load_yaml, set_memory_growth

import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

from api import PRN

print('TensorFlow:', tf.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
set_memory_growth()

cfg = load_yaml('./configs/prnet.yaml')
model = PRNet_Model(cfg['input_size'], cfg['ch_size'])

# get trainable / non trainable params
print(model.summary())

forward_pass = tf.function(
    model.call,
    input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

graph_info = profile(forward_pass.get_concrete_function().graph,
                        options=ProfileOptionBuilder.float_operation())

# The //2 is necessary since `profile` counts multiply and accumulate
# as two flops, here we report the total number of multiply accumulate ops
flops = graph_info.total_float_ops // 2
print('Flops: {:,}'.format(flops))