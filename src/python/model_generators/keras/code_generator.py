#!/usr/bin/python3
# code_generator.py

import tensorflow as tf

def add_layers(layers, template):
    template += '''model = tf.keras.Sequential()'''
    for layer in layers:
        if layer['type'] == 'dense':
            template += '''model.add(tf.keras.layers.Dense(units={units}, activation={activation_function}}))'''\
                .format(units=layer['nodes'], activation_function=layer['activation_function'])
        elif layer['type'] == 'flatten':
            template += '''model.add(tf.keras.layers.Flatten())'''
        elif layer['type'] == 'dropout':
            template += '''model.add(tf.keras.layers.Dropout(rate={rate}'''\
                .format(rate=layer['rate'])
        elif layer['type'] =='batch_normalization':
            template += '''model.add(tf.keras.layers.BatchNormalization(axis={axis}, momentum={momentum}, 
            epsilon={epsilon})'''\
                .format(axis=layer['axis'], momentum=layer['momentum'], epsilon=layer['epsilon'])
        elif layer['type'] == 'conv_2d':
            template += '''model.add(tf.keras.layers.Conv2D(axis={axis}, momentum={momentum}, 
                    epsilon={epsilon})''' \
                .format(axis=layer['axis'], momentum=layer['momentum'], epsilon=layer['epsilon'])



            'conv_2d',
            'max_pooling_2d',
            'average_pooling_2d',
            'rnn',
            'lstm',
            'gru'

def generate_template(name, layers, x_training_data=None, y_training_data=None, loss=None, optimizer=None,
                  learning_rate=None, epochs=None, batch_size=None):
    template = ''''''
    # Intro code
    add_layers(layers, template)
    # Outro code
