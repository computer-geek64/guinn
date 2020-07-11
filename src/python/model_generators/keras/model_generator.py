#!/usr/bin/python3
# model_generator.py

import tensorflow as tf


def add_layers(layers):
    model = tf.keras.Sequential()
    for layer in layers:
        if layer['type'] == 'dense':
            if layer['activation_function'] is not None:
                model.add(tf.keras.layers.Dense(units=layer['nodes'], activation=layer['activation_function'],
                                                use_bias=layer['use_bias']))
            else:
                model.add(tf.keras.layers.Dense(units=layer['nodes'], activation=None, use_bias=layer['use_bias']))
        elif layer['type'] == 'input_layer':
            model.add(tf.keras.layers.InputLayer(input_shape=layer['input_shape']))
        elif layer['type'] == 'flatten':
            model.add(tf.keras.layers.Flatten())
        elif layer['type'] == 'embedding':
            model.add(tf.keras.layers.Embedding(input_dim=layer['input_dim'], output_dim=layer['output_dim']))
        elif layer['type'] == 'dropout':
            model.add(tf.keras.layers.Dropout(rate=layer['rate']))
        elif layer['type'] == 'batch_normalization':
            model.add(tf.keras.layers.BatchNormalization(axis=layer['axis'], momentum=layer['momentum'],
                                                         epsilon=layer['epsilon']))
        elif layer['type'] == 'conv_2d':
            if layer['activation_function'] is not None:
                model.add(filters=layer['filters'], kernel_size=layer['kernel_size'], strides=layer['strides'],
                          activation=layer['activation_function'], use_bias=layer['use_bias'])
            else:
                model.add(filters=layer['filters'], kernel_size=layer['kernel_size'], strides=layer['strides'],
                          activation=None, use_bias=layer['use_bias'])
        elif layer['type'] == 'max_pooling_2d':
            model.add(tf.keras.layers.MaxPool2D(pool_size=layer['kernel_size'], strides=layer['strides']))
        elif layer['type'] == 'average_pooling_2d':
            model.add(tf.keras.layers.AveragePooling2D(pool_size=layer['kernel_size'], strides=layer['strides']))
        elif layer['type'] == 'rnn':
            if layer['activation_function'] is not None:
                model.add(tf.keras.layers.SimpleRNN(units=layer['nodes'], activation=layer['activation_function'],
                                                    use_bias=layer['use_bias'], dropout=layer['dropout'],
                                                    recurrent_dropout=layer['recurrent_dropout']))
            else:
                model.add(tf.keras.layers.SimpleRNN(units=layer['nodes'], activation=None,
                                                    use_bias=layer['use_bias'], dropout=layer['dropout'],
                                                    recurrent_dropout=layer['recurrent_dropout']))
        elif layer['type'] == 'lstm':
            if layer['activation_function'] is None and layer['recurrent_activation_function'] is not None:
                model.add(tf.keras.layers.LSTM(units=layer['nodes'], activation=None,
                                               recurrent_activation=layer['recurrent_activation_function'],
                                               use_bias=layer['use_bias'], dropout=layer['dropout'],
                                               recurrent_dropout=layer['recurrent_dropout']))
            elif layer['activation_function'] is not None and layer['recurrent_activation_function'] is None:
                model.add(tf.keras.layers.LSTM(units=layer['nodes'], activation=layer['activation_function'],
                                               recurrent_activation=None,
                                               use_bias=layer['use_bias'],
                                               dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout']))
            elif layer['activation_function'] is None and layer['recurrent_activation_function'] is None:
                model.add(tf.keras.layers.LSTM(units=layer['nodes'], activation=None,
                                               recurrent_activation=None,
                                               use_bias=layer['use_bias'],
                                               dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout']))
            else:
                model.add(tf.keras.layers.LSTM(units=layer['nodes'], activation=layer['activation_function'],
                                               recurrent_activation=layer['recurrent_activation_function'],
                                               use_bias=layer['use_bias'],
                                               dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout']))
        elif layer['type'] == 'gru':
            if layer['activation_function'] is None and layer['recurrent_activation_function'] is not None:
                model.add(tf.keras.layers.GRU(units=layer['nodes'], activation=None,
                                              recurrent_activation=layer['recurrent_activation_function'],
                                              use_bias=layer['use_bias'], dropout=layer['dropout'],
                                              recurrent_dropout=layer['recurrent_dropout']))
            elif layer['activation_function'] is not None and layer['recurrent_activation_function'] is None:
                model.add(tf.keras.layers.GRU(units=layer['nodes'], activation=layer['activation_function'],
                                              recurrent_activation=None, use_bias=layer['use_bias'],
                                              dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout']))
            elif layer['activation_function'] is None and layer['recurrent_activation_function'] is None:
                model.add(tf.keras.layers.GRU(units=layer['nodes'], activation=None, recurrent_activation=None,
                                              use_bias=layer['use_bias'], dropout=layer['dropout'],
                                              recurrent_dropout=layer['recurrent_dropout']))
            else:
                model.add(tf.keras.layers.GRU(units=layer['nodes'], activation=layer['activation_function'],
                                              recurrent_activation=layer['recurrent_activation_function'],
                                              use_bias=layer['use_bias'], dropout=layer['dropout'],
                                              recurrent_dropout=layer['recurrent_dropout']))
    return model


def check_shape_compatibility(model):
    layers = model.layers
    errors = []
    for index in range(0, len(layers) - 1):
        if layers[index].output_shape != layers[index + 1].input_shape:
            error = 'Shape incompatibility between layers {this_index} and {next_index}'.format(this_index=index,
                                                                                                next_index=index + 1)
            errors.append(error)
    return errors


def compile_fit(model, optimizer, x_train, y_train, loss, batch_size, epochs):
    if optimizer is None and loss is not None:
        model.compile(optimizer=None, loss=loss)
    elif optimizer is not None and loss is None:
        model.compile(optimizer=optimizer, loss=None)
    elif optimizer is None and loss is None:
        model.compile(optimizer=None, loss=None)
    else:
        model.compile(optimizer=optimizer, loss=loss)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)


def generate_model(layers, loss=None, optimizer='rmsprop', x_train=None, y_train=None, batch_size=32, epochs=10):
    model = add_layers(layers)
    errors = check_shape_compatibility(model)
    if len(errors) != 0:
        return errors
    compile_fit(model, optimizer, x_train, y_train, loss, batch_size, epochs)
    return model.save(model)

# TEST
'''layers = [{'type': 'dense',
           'nodes': 10,
           'activation_function': None,
           'use_bias': True},
          {'type': 'dense',
           'nodes': 5,
           'activation_function': 'relu',
           'use_bias': True}]
print(generate_template(layers=layers))'''
