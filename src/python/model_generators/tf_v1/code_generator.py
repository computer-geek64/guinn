#!/usr/bin/python3
# code_generator.py

import numpy as np


def to_str_list(data):
    return 'np.array(' + str(np.array(data).tolist()) + ')'


def generate_code(name, layers, x_training_data=None, y_training_data=None, loss=None, optimizer=None, learning_rate=None, epochs=None, batch_size=None):
    # Generate code
    name = name.lower().replace(' ', '_')

    if x_training_data and y_training_data:
        x_training_data = to_str_list(x_training_data)
        y_training_data = to_str_list(y_training_data)
    else:
        x_training_data = 'None  # Add x training data'
        y_training_data = 'None  # Add y training data'

    # Create initial template class structure with training data
    template = '''#!/usr/bin/python3
# {name}.py

import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Training data
x_training_data = {x_training_data}
y_training_data = {y_training_data}

# Checks
if len(x_training_data.shape) == 1:
    x_training_data = np.reshape(x_training_data, (x_training_data.size, 1))
y_training_data = np.reshape(y_training_data, (y_training_data.size, {output_nodes}))


class NeuralNetwork:
    def __init__(self):
        # Training data placeholders
        self.x_train = tf.placeholder(dtype=tf.float32, shape=(None,) + np.shape(x_training_data)[1:], name='x_train')
        self.y_train = tf.placeholder(dtype=tf.float32, shape=(None,) + np.shape(y_training_data)[1:], name='y_train')

        # Neural network layers
        self.layer = self.x_train
'''.format(name=name, output_nodes=layers[-1]['nodes'], x_training_data=x_training_data, y_training_data=y_training_data)

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
        template += '        self.layer = tf.layers.Dense({nodes}, activation={activation_function}, name=\'{name}\')(self.layer)\n'.format(**layer)

    # Add loss function
    if not loss:
        loss = 'tf.losses.mean_squared_error(self.y_train, self.layer)  # Change loss function'

    template += '''
        # Loss function
        self.loss = {loss}
'''.format(loss=loss)

    # Add optimizer
    optimizer_learning_rate_comment = ''
    if not optimizer:
        optimizer = 'tf.train.GradientDescentOptimizer'
        optimizer_learning_rate_comment = '  # Change optimizer function'
    if not learning_rate:
        learning_rate = '0.001'
        optimizer_learning_rate_comment = '  # Change learning rate'
    if not optimizer and not learning_rate:
        optimizer_learning_rate_comment = '  # Change optimizer function and learning rate'

    template += '''
        # Optimizer
        optimizer = {optimizer}(learning_rate={learning_rate}, name=\'optimizer\'){optimizer_learning_rate_comment}
        self.training = optimizer.minimize(self.loss)
'''.format(optimizer=optimizer, learning_rate=learning_rate, optimizer_learning_rate_comment=optimizer_learning_rate_comment)

    # Create Session
    template += '''
        # Session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
'''

    # Add training
    if not epochs:
        epochs = '100  # Change number of epochs (number of complete passes on the training dataset)'

    if not batch_size:
        batch_size = '10  # Change batch size (number of training samples in one iteration)'

    template += '''
    def train(self, x_data, y_data):
        # Training
        epochs = {epochs}
        batch_size = {batch_size}

        for epoch in range(epochs):
            for iteration in range(0, min(len(x_data), len(y_data)), batch_size):
                self.sess.run(self.training, feed_dict={{self.x_train: x_data[iteration:iteration + batch_size], self.y_train: y_data[iteration:iteration + batch_size]}})
        return self.sess.run(self.loss, feed_dict={{self.x_train: x_data, self.y_train: y_data}})
'''.format(epochs=epochs, batch_size=batch_size)

    # Add testing
    template += '''
    def test(self, x_data, y_data):
        test_loss = {loss}
        return self.sess.run(test_loss, feed_dict={{self.x_train: x_data}})
'''.format(loss=loss.split('(')[0] + '(tf.constant(y_data, dtype=tf.float32), self.layer)')

    # Add prediction
    template += '''
    def predict(self, x_data):
        return self.sess.run(self.layer, feed_dict={self.x_train: x_data})
'''

    # Add saving
    template += '''
    def save(self, filepath_prefix=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'model')):
        if not os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')):
            os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
        saver = tf.train.Saver()
        return saver.save(self.sess, filepath_prefix)
'''

    # Add restoring
    template += '''
    def restore(self, filepath_prefix=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'model')):
        saver = tf.train.Saver()
        saver.restore(self.sess, filepath_prefix)
        return filepath_prefix
'''

    # Add final generation code
    template += '''

nn = NeuralNetwork()
print(nn.train(x_training_data, y_training_data))
print(nn.predict([[9]]))
'''
    return template

with open('neural_network.py', 'w') as file:
    file.write(generate_code('Neural Network', [{'name': 'layer1', 'nodes': 10, 'activation_function': None}, {'name': 'layer2', 'nodes': 1, 'activation_function': None}], x_training_data=[[1], [2], [3], [4], [5], [6]], y_training_data=[[2], [3], [4], [5], [6], [7]]))