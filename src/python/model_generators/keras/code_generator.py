#!/usr/bin/python3
# code_generator.py

import tensorflow as tf


def add_layers(layers, template):
    template += '''model = tf.keras.Sequential()'''
    for layer in layers:
        if layer['type'] == 'dense':
            template += '''model.add(tf.keras.layers.Dense(units={units}, activation={activation}, 
            use_bias={use_bias}))''' \
                .format(units=layer['nodes'], activation=layer['activation_function'],
                        use_bias=layer['use_bias'])
        elif layer['type'] == 'input_layer':
            template += '''model.add(tf.keras.layers.InputLayer(input_shape={input_shape})''' \
                .format(input_shape=layer['input_shape'])
        elif layer['type'] == 'flatten':
            template += '''model.add(tf.keras.layers.Flatten())'''
        elif layer['type'] == 'embedding':
            template += '''model.add(tf.keras.layers.Embedding(input_dim={input_dim}, output_dim={output_dim}''' \
                .format(input_dim=layer['input_dim'], output_dim=layer['output_dim'])
        elif layer['type'] == 'dropout':
            template += '''model.add(tf.keras.layers.Dropout(rate={rate}''' \
                .format(rate=layer['rate'])
        elif layer['type'] == 'batch_normalization':
            template += '''model.add(tf.keras.layers.BatchNormalization(axis={axis}, momentum={momentum}, 
            epsilon={epsilon})''' \
                .format(axis=layer['axis'], momentum=layer['momentum'], epsilon=layer['epsilon'])
        elif layer['type'] == 'conv_2d':
            template += '''model.add(tf.keras.layers.Conv2D(filters={filters}, kernel_size={kernel_size}, 
                    strides={strides}, activation={activation}, use_bias={use_bias})''' \
                .format(filters=layer['filters'], kernel_size=layer['kernel_size'], strides=layer['strides'],
                        activation=layer['activation_function'], use_bias=layer['use_bias'])
        elif layer['type'] == 'max_pooling_2d':
            template += '''model.add(tf.keras.layers.MaxPool2D(pool_size={pool_size}, strides={strides})''' \
                .format(pool_size=layer['pool_size'], strides=layer['strides'], )
        elif layer['type'] == 'average_pooling_2d':
            template += '''model.add(tf.keras.layers.AveragePooling2D(pool_size={pool_size}, strides={strides})''' \
                .format(pool_size=layer['pool_size'], strides=layer['strides'], )
        elif layer['type'] == 'rnn':
            template += '''model.add(tf.keras.layers.SimpleRNN(units={units}, activation={activation}, 
            use_bias={use_bias}, dropout={dropout}, recurrent_dropout={recurrent_dropout})''' \
                .format(units=layer['nodes'], activation=layer['activation_function'], use_bias=layer['use_bias'],
                        dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout'])
        elif layer['type'] == 'lstm':
            template += '''model.add(tf.keras.layers.LSTM(units={units}, activation={activation}, 
            recurrent_activation={recurrent_activation}, use_bias={use_bias}, dropout={dropout}, 
            recurrent_dropout={recurrent_dropout})''' \
                .format(units=layer['nodes'], activation=layer['activation_function'],
                        recurrent_activation=layer['recurrent_activation_function'], use_bias=layer['use_bias'],
                        dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout'])
        elif layer['type'] == 'lstm':
            template += '''model.add(tf.keras.layers.LSTM(units={units}, activation={activation}, 
            recurrent_activation={recurrent_activation}, use_bias={use_bias}, dropout={dropout}, 
            recurrent_dropout={recurrent_dropout})''' \
                .format(units=layer['nodes'], activation=layer['activation_function'],
                        recurrent_activation=layer['recurrent_activation_function'], use_bias=layer['use_bias'],
                        dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout'])


def generate_template(name, layers, x_training_data=None, y_training_data=None, loss=None, optimizer=None,
                      learning_rate=None, epochs=None, batch_size=None):
    template = ''' All intro code '''
    add_layers(layers, template)
    template += ''' All outro code'''

    return template
