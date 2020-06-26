#!/usr/bin/python3
# code_generator.py


def generate_code(name, layers, x_train_shape, y_train_shape):
    # Generate code
    name = name.lower().replace(' ', '_')

    template = '''#!/usr/bin/python3
# {name}.py

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


x_train = tf.placeholder(dtype=tf.float32, shape={x_train_shape}, name='x_train')
y_train = tf.placeholder(dtype=tf.float32, shape={y_train_shape}, name='y_train')

layer = x_train
'''.format(name=name, x_train_shape=x_train_shape, y_train_shape=y_train_shape)
    for layer in layers:
        if layer['activation_function'] == 'relu':
            layer['activation_function'] = 'tf.nn.relu'
        elif layer['activation_function'] == 'leaky_relu':
            layer['activation_function'] = 'tf.nn.leaky_relu'
        elif layer['activation_function'] == 'sigmoid':
            layer['activation_function'] = 'tf.nn.sigmoid_cross_entropy_with_logits'
        elif layer['activation_function'] == 'softmax':
            layer['activation_function'] = 'tf.nn.softmax'
        else:
            layer['activation_function'] = 'None'
        template += 'layer = tf.layers.Dense({nodes}, {activation_function}, name={name})(layer)\n'.format(**layer)
    print(template)


generate_code('Neural Network', [{'name': 'layer1', 'nodes': 10, 'activation_function': 'relu'}, {'name': 'layer2', 'nodes': 16, 'activation_function': 'softmax'}], (None, 2), (None,))