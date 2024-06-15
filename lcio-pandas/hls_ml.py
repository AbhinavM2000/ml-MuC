import tensorflow as tf
import hls4ml
import os

# Set HLS PATH
os.environ['XILINX_VIVADO'] = '/home/vlsilab13/Xilinx/Vivado/2020.1'
os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']

# Create a function to load the Keras model from file
def load_keras_model(file_path):
    return tf.keras.models.load_model(file_path)

# Load the model from file
model_file_path = 'bestmodel.h5'
dummy_model = load_keras_model(model_file_path)

# Convert the TensorFlow model to hls4ml configuration
config = hls4ml.utils.config_from_keras_model(dummy_model, granularity='model')

# Set loop unroll factor in the HLS configuration
config['HLSConfig'] = {
    'IOType': 'io_stream',
    'Optimization': 'Performance',
    'UnrollFactor': 2
}

# Convert the model to HLS
hls_model = hls4ml.converters.convert_from_keras_model(
    dummy_model,
    hls_config=config,
    output_dir='test/test_1',
    part='xcu250-figd2104-2L-e'
)

# Visualize the model
hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True)

# Synthesize using HLS backend
hls_model.compile()
hls_model.build(csim=False)
