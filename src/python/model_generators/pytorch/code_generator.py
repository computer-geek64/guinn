import torch
import torch.nn as nn


type_of_layer = {'dense': 'Linear','flatten': "Flatten",'dropout': "Dropout2D",'batch_normalization': "BatchNorm1D",'conv_2d':"Conv2D",'max_pooling_2d': "MaxPool2D",'average_pooling_2d': "AvgPool2D",'rnn': "RNN",'lstm': "LSTM",'gru':"GRU"}
def generate_code(layers):
    code_template = ''

    code_template += 'layers_list = []'
    for i in range(len(layers)):
        if (layers[i]['name'] == 'dense'):
            code_template += 'layers_list.append(nn.Linear({previous_layer_nodes}, {layer_nodes}))'\
                .format(previous_layer_nodes=layers[i]['nodes'], layer_nodes=layers[i]['nodes'])
        elif (layers[i]['name'] == 'flatten'):
            code_template += 'layers_list.append(nn.Flatten())'
        elif (layers[i]['name'] == 'dropout'):
            code_template += 'layers_list.append(nn.Dropout2d(p={probability}))'\
                .format(probability=layers[i]['probability'])
        elif (layers[i]['name'] == 'batch_normalization'):
            code_template += 'layers_list.append(nn.BatchNorm1d(num_features={num_of_features}))'\
                .format(num_of_features=layers[i]['num_of_features'])
        elif(layers[i]['name'] == 'conv_2d'):
            code_template += 'layers_list.append(nn.Conv2d({input_channels}, {output_channels}, {kernel_size}, stride={stride}, padding={padding}))'\
                .format(input_channels=layers[i]['input'], output_channels=layers[i]['output'], kernel_size=layers[i]['kernel_size'],
                        stride = layers[i]['stride'], padding= layers[i]['padding'])
        elif(layers[i]['name'] == 'max_pooling_2d'):
            code_template += 'layers_list.append(nn.MaxPool2d(kernel_size = {kernel_size}))'\
                .format(kernel_size = layers[i]['kernel_size'])
        elif (layers[i]['name'] == 'average_pooling_2d'):
            code_template += 'layers_list.append(nn.AvgPool2d(kernel_size = {kernel_size}))' \
                .format(kernel_size=layers[i]['kernel_size'])
        code_template += '\n'
        if (layers[i]['activation_function'] == 'relu'):
            code_template += 'layers_list.append(nn.ReLU())'
        elif (layers[i]['activation_function'] == 'leaky_relu'):
            code_template += 'layers_list.append(nn.LeakyReLU())'
        elif (layers[i]['activation_function'] == 'sigmoid'):
            code_template += 'layers_list.append(nn.Sigmoid())'
        elif (layers[i]['activation_function'] == 'tanh'):
            code_template += 'layers_list.append(nn.Tanh())'
        elif (layers[i]['activation_function'] == 'softmax'):
            code_template += 'layers_list.append(nn.Softmax())'
        code_template += '\n\n'
    code_template += 'network = nn.Sequential(*layer_list)'

    return code_template


layers = [{'nodes': 3, 'activation_function': 'none', 'name': 'input_layer'}, {'nodes': 10, 'activation_function': 'relu', 'name': 'dense'}, {'nodes': 16, 'activation_function': 'sigmoid', 'name': 'dense'}]
print(generate_code(layers))