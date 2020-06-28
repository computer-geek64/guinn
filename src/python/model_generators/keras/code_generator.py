#!/usr/bin/python3
# code_generator.py


def add_layers(layers):
    return_string = '''model = tf.keras.Sequential()\n'''
    for layer in layers:
        if layer['type'] == 'dense':
            if layer['activation_function'] is not None:
                return_string += '''model.add(tf.keras.layers.Dense(units={units}, activation=\'{activation}\', 
                use_bias={use_bias}))\n''' \
                    .format(units=layer['nodes'], activation=layer['activation_function'], use_bias=layer['use_bias'])
            else:
                return_string += '''model.add(tf.keras.layers.Dense(units={units}, activation={activation}, 
                use_bias={use_bias}))\n''' \
                    .format(units=layer['nodes'], activation=layer['activation_function'], use_bias=layer['use_bias'])
        elif layer['type'] == 'input_layer':
            return_string += '''model.add(tf.keras.layers.InputLayer(input_shape={input_shape})\n''' \
                .format(input_shape=layer['input_shape'])
        elif layer['type'] == 'flatten':
            return_string += '''model.add(tf.keras.layers.Flatten())\n'''
        elif layer['type'] == 'embedding':
            return_string += '''model.add(tf.keras.layers.Embedding(input_dim={input_dim}, output_dim={output_dim}\n''' \
                .format(input_dim=layer['input_dim'], output_dim=layer['output_dim'])
        elif layer['type'] == 'dropout':
            return_string += '''model.add(tf.keras.layers.Dropout(rate={rate}\n''' \
                .format(rate=layer['rate'])
        elif layer['type'] == 'batch_normalization':
            return_string += '''model.add(tf.keras.layers.BatchNormalization(axis={axis}, momentum={momentum}, 
            epsilon={epsilon})\n''' \
                .format(axis=layer['axis'], momentum=layer['momentum'], epsilon=layer['epsilon'])
        elif layer['type'] == 'conv_2d':
            if layer['activation_function'] is not None:
                return_string += '''model.add(tf.keras.layers.Conv2D(filters={filters}, kernel_size={kernel_size}, 
                strides={strides}, activation=\'{activation}\', use_bias={use_bias})\n''' \
                    .format(filters=layer['filters'], kernel_size=layer['kernel_size'], strides=layer['strides'],
                            activation=layer['activation_function'], use_bias=layer['use_bias'])
            else:
                return_string += '''model.add(tf.keras.layers.Conv2D(filters={filters}, kernel_size={kernel_size}, 
                strides={strides}, activation={activation}, use_bias={use_bias})\n''' \
                    .format(filters=layer['filters'], kernel_size=layer['kernel_size'], strides=layer['strides'],
                            activation=layer['activation_function'], use_bias=layer['use_bias'])
        elif layer['type'] == 'max_pooling_2d':
            return_string += '''model.add(tf.keras.layers.MaxPool2D(pool_size={pool_size}, strides={strides})\n''' \
                .format(pool_size=layer['kernel_size'], strides=layer['strides'])
        elif layer['type'] == 'average_pooling_2d':
            return_string += '''model.add(tf.keras.layers.AveragePooling2D(pool_size={pool_size}, strides={strides})\n''' \
                .format(pool_size=layer['kernel_size'], strides=layer['strides'])
        elif layer['type'] == 'rnn':
            if layer['activation_function'] is not None:
                return_string += '''model.add(tf.keras.layers.SimpleRNN(units={units}, activation=\'{activation}\', 
                use_bias={use_bias}, dropout={dropout}, recurrent_dropout={recurrent_dropout})\n''' \
                    .format(units=layer['nodes'], activation=layer['activation_function'], use_bias=layer['use_bias'],
                            dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout'])
            else:
                return_string += '''model.add(tf.keras.layers.SimpleRNN(units={units}, activation={activation}, 
                use_bias={use_bias}, dropout={dropout}, recurrent_dropout={recurrent_dropout})\n''' \
                    .format(units=layer['nodes'], activation=layer['activation_function'], use_bias=layer['use_bias'],
                            dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout'])
        elif layer['type'] == 'lstm':
            if layer['activation_function'] is None and layer['recurrent_activation'] is not None:
                return_string += '''model.add(tf.keras.layers.LSTM(units={units}, activation={activation}, 
                recurrent_activation=\'{recurrent_activation}\', use_bias={use_bias}, dropout={dropout}, 
                recurrent_dropout={recurrent_dropout})\n''' \
                    .format(units=layer['nodes'], activation=layer['activation_function'],
                            recurrent_activation=layer['recurrent_activation_function'], use_bias=layer['use_bias'],
                            dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout'])
            elif layer['activation_function'] is not None and layer['recurrent_activation'] is None:
                return_string += '''model.add(tf.keras.layers.LSTM(units={units}, activation=\'{activation}\', 
                recurrent_activation={recurrent_activation}, use_bias={use_bias}, dropout={dropout}, 
                recurrent_dropout={recurrent_dropout})\n''' \
                    .format(units=layer['nodes'], activation=layer['activation_function'],
                            recurrent_activation=layer['recurrent_activation_function'], use_bias=layer['use_bias'],
                            dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout'])
            elif layer['activation_function'] is None and layer['recurrent_activation'] is None:
                return_string += '''model.add(tf.keras.layers.LSTM(units={units}, activation={activation}, 
                recurrent_activation={recurrent_activation}, use_bias={use_bias}, dropout={dropout}, 
                recurrent_dropout={recurrent_dropout})\n''' \
                    .format(units=layer['nodes'], activation=layer['activation_function'],
                            recurrent_activation=layer['recurrent_activation_function'], use_bias=layer['use_bias'],
                            dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout'])
            else:
                return_string += '''model.add(tf.keras.layers.LSTM(units={units}, activation=\'{activation}\', 
                recurrent_activation=\'{recurrent_activation}\', use_bias={use_bias}, dropout={dropout}, 
                recurrent_dropout={recurrent_dropout})\n''' \
                    .format(units=layer['nodes'], activation=layer['activation_function'],
                            recurrent_activation=layer['recurrent_activation_function'], use_bias=layer['use_bias'],
                            dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout'])
        elif layer['type'] == 'gru':
            if layer['activation_function'] is None and layer['recurrent_activation'] is not None:
                return_string += '''model.add(tf.keras.layers.GRU(units={units}, activation={activation}, 
                recurrent_activation=\'{recurrent_activation}\', use_bias={use_bias}, dropout={dropout}, 
                recurrent_dropout={recurrent_dropout})\n''' \
                    .format(units=layer['nodes'], activation=layer['activation_function'],
                            recurrent_activation=layer['recurrent_activation_function'], use_bias=layer['use_bias'],
                            dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout'])
            elif layer['activation_function'] is not None and layer['recurrent_activation'] is None:
                return_string += '''model.add(tf.keras.layers.GRU(units={units}, activation=\'{activation}\', 
                recurrent_activation={recurrent_activation}, use_bias={use_bias}, dropout={dropout}, 
                recurrent_dropout={recurrent_dropout})\n''' \
                    .format(units=layer['nodes'], activation=layer['activation_function'],
                            recurrent_activation=layer['recurrent_activation_function'], use_bias=layer['use_bias'],
                            dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout'])
            elif layer['activation_function'] is None and layer['recurrent_activation'] is None:
                return_string += '''model.add(tf.keras.layers.GRU(units={units}, activation={activation}, 
                recurrent_activation={recurrent_activation}, use_bias={use_bias}, dropout={dropout}, 
                recurrent_dropout={recurrent_dropout})\n''' \
                    .format(units=layer['nodes'], activation=layer['activation_function'],
                            recurrent_activation=layer['recurrent_activation_function'], use_bias=layer['use_bias'],
                            dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout'])
            else:
                return_string += '''model.add(tf.keras.layers.GRU(units={units}, activation=\'{activation}\', 
                recurrent_activation=\'{recurrent_activation}\', use_bias={use_bias}, dropout={dropout}, 
                recurrent_dropout={recurrent_dropout})\n''' \
                    .format(units=layer['nodes'], activation=layer['activation_function'],
                            recurrent_activation=layer['recurrent_activation_function'], use_bias=layer['use_bias'],
                            dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout'])
    return return_string


def add_compile_fit(optimizer, x_train, y_train, loss, batch_size, epochs):
    if optimizer is None and loss is not None:
        return_string = '''model.compile(optimizer={optimizer}, loss=\'{loss}\')\n''' \
            .format(optimizer=optimizer, loss=loss)
    elif optimizer is not None and loss is None:
        return_string = '''model.compile(optimizer=\'{optimizer}\', loss={loss})\n''' \
            .format(optimizer=optimizer, loss=loss)
    elif optimizer is None and loss is None:
        return_string = '''model.compile(optimizer={optimizer}, loss={loss})\n''' \
            .format(optimizer=optimizer, loss=loss)
    else:
        return_string = '''model.compile(optimizer=\'{optimizer}\', loss={loss})\n''' \
            .format(optimizer=optimizer, loss=loss)

    if x_train is None or y_train is None:
        return_string += '''model.fit( 
            # x_train,
            # y_train,
            batch_size={batch_size}, epochs={epochs}
         )\n''' \
            .format(batch_size=batch_size, epochs=epochs)
    else:
        return_string += '''model.fit(x_train, y_train, batch_size={batch_size}, epochs={epochs})\n''' \
            .format(batch_size=batch_size, epochs=epochs)
    return return_string


def generate_template(layers, loss='mse', optimizer='sgd', x_train=None, y_train=None, batch_size=32, epochs=10):
    template = '''import tensorflow as tf\n'''
    template += add_layers(layers)
    template += add_compile_fit(optimizer, x_train, y_train, loss, batch_size, epochs)
    return template

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