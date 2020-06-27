#!/usr/bin/python3
# code_generator.py

import tensorflow as tf

template = '''model = tf.keras.Sequential()'''

def add_layers(layers):
    for layer in layers:
        if layer['type'] == 'dense':
            template += '''model.add(tf.keras.layers.Dense(units=, activation=None))'''  # add the actual variables
        elif layer['type'] == 'flatten':
            template += '''model.add(tf.keras.layers.Flatten())'''
        elif layer['dropout'] == 'dropout':
            

def generate_template(name, layers, x_training_data=None, y_training_data=None, loss=None, optimizer=None,
                  learning_rate=None, epochs=None, batch_size=None):
    # generate keras code
