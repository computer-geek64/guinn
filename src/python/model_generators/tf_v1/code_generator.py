#!/usr/bin/python3
# code_generator.py

import numpy as np


def to_str_list(data):
    return str(np.array(data).tolist())


def generate_code(name, layers, x_training_data=None, y_training_data=None, loss=None, optimizer=None, learning_rate=None, epochs=None, batch_size=None):
    # Generate code
    name = name.lower().replace(' ', '_')

    if x_training_data and y_training_data:
        x_training_data = to_str_list(x_training_data)
        y_training_data = to_str_list(y_training_data)
    else:
        x_training_data = 'None  # Add x training data'
        y_training_data = 'None  # Add y training data'

    # Create basic template structure with training data
    template = '''#!/usr/bin/python3
# {name}.py

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Training data
x_training_data = {x_training_data}
y_training_data = {y_training_data}

# Training data placeholders
x_train = tf.placeholder(dtype=tf.float32, shape=(None,) + np.shape(x_training_data)[1:], name='x_train')
y_train = tf.placeholder(dtype=tf.float32, shape=(None,) + np.shape(y_training_data)[1:], name='y_train')

# Neural network layers
layer = x_train
'''.format(name=name, x_training_data=x_training_data, y_training_data=y_training_data)

    # Add layers
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
        template += 'layer = tf.layers.Dense({nodes}, {activation_function}, name=\'{name}\')(layer)\n'.format(**layer)

    # Add loss function
    if not loss:
        loss = 'tf.losses.mean_squared_error(x_train, y_train)  # Change loss function'

    template += '''
# Loss function
loss = {loss}
'''.format(loss=loss)

    # Add optimizer
    optimizer_learning_rate_comment = ''
    if not optimizer:
        optimizer = 'tf.train.GradientDescentOptimizer'
        optimizer_learning_rate_comment = '  # Change optimizer function'
    if not learning_rate:
        learning_rate = '0.01'
        optimizer_learning_rate_comment = '  # Change learning rate'
    if not optimizer and not learning_rate:
        optimizer_learning_rate_comment = '  # Change optimizer function and learning rate'

    template += '''
# Optimizer
optimizer = {optimizer}(learning_rate={learning_rate}, name=\'optimizer\'){optimizer_learning_rate_comment}
training = optimizer.minimize(loss)
'''.format(optimizer=optimizer, learning_rate=learning_rate, optimizer_learning_rate_comment=optimizer_learning_rate_comment)

    # Create Session
    template += '''
# Session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
'''

    # Add training
    if not epochs:
        epochs = '100  # Change number of epochs (number of complete passes on the training dataset)'

    if not batch_size:
        batch_size = '10  # Change batch size (number of training samples in one iteration)'

    template += '''
# Training
epochs = {epochs}
batch_size = {batch_size}

for epoch in range(epochs):
    for iteration in range(0, min(len(x_training_data), len(y_training_data)), batch_size):
        sess.run(training, feed_dict={{x_train: x_training_data[iteration:iteration + batch_size], y_train: y_training_data[iteration:iteration + batch_size]}})
training_loss = sess.run(loss, feed_dict={{x_train: x_training_data[iteration:iteration + batch_size], y_train: y_training_data[iteration:iteration + batch_size]}})
'''.format(epochs=epochs, batch_size=batch_size)
    return template


generate_code('Neural Network', [{'name': 'layer1', 'nodes': 10, 'activation_function': 'relu'}, {'name': 'layer2', 'nodes': 16, 'activation_function': 'softmax'}], x_training_data=[[1, 2, 3], [4, 5, 6]], y_training_data=[2, 3, 4, 5, 6, 7])