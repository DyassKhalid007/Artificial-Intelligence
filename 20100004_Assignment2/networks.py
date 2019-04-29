# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:00:51 2019

@author: Dyass Khalid
Implementation of N layer Neural Nutwork(:p) both for classification and regression both as feedforward and radial basis
YAYY
"""
import numpy as np


import matplotlib.pyplot as plt
plt.style.use('seaborn')


from tqdm import tqdm_notebook
import os
import cv2 as cv
from sklearn.model_selection import train_test_split



class NeuralNetwork():
    @staticmethod
    def mean_squared_error(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def cross_entropy_loss(y_pred, y_true):
        return -(y_true * np.log(y_pred)).sum()
    @staticmethod
    def RB(x):
        return np.exp((x-np.mean(x)**2)/np.var(x)+1)
    @staticmethod
    def accuracy(y_pred, y_true):
        max_pred = []
        index = 0
        for values in y_pred:
            m = values[0]
            index = 0
            for i in range(len(values)):
                if values[i]>m:
                    m = values[i]
                    index = i
            max_pred.append(index)
        max_true = []
        for values in y_true:
            index = 0
            m = values[0]
            for i in range(len(values)):
                if values[i]>m:
                    m = values[i]
                    index = i
            max_true.append(index)
        count = 0
        for i in range(len(max_true)):
            if max_pred[i]==max_true[i]:
                count+=1
        return count/len(y_pred)*100

    @staticmethod
    def softmax(x):
        expx = np.exp(x)
        return expx / expx.sum(axis=1, keepdims=True)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def __init__(self, nodes_per_layer, mode):
        '''Creates a Feed-Forward Neural Network.
        "nodes_per_layer" is a list containing number of nodes in each layer (including input layer)
        "mode" can be one of 'regression' or 'classification' and controls the output activation as well as training metric'''
        if len(nodes_per_layer) < 2:
            raise ValueError('Network must have atleast 2 layers (input and output).')
        if not (np.array(nodes_per_layer) > 0).all():
            raise ValueError('Number of nodes in all layers must be positive.')
        if mode not in ['classification','regression']:
            raise ValueError('Only "classification" and "regression" modes are supported.')

        self.num_layers = len(nodes_per_layer) # includes input layer
        self.nodes_per_layer = nodes_per_layer
        self.input_shape = nodes_per_layer[0]
        self.output_shape = nodes_per_layer[-1]
        self.mode = mode

        self.__init_weights(nodes_per_layer)

    def __init_weights(self, nodes_per_layer):
        '''Initializes all weights based on standard normal distribution and all biases to 0.'''
        self.weights_ = []
        self.biases_ = []
        for i,_ in enumerate(nodes_per_layer):
            if i == 0:
                # skip input layer, it does not have weights/bias
                continue

            weight_matrix = np.random.normal(size=(nodes_per_layer[i-1], nodes_per_layer[i]))
            self.weights_.append(weight_matrix)
            bias_vector = np.zeros(shape=(nodes_per_layer[i],))
            self.biases_.append(bias_vector)

    def fit(self, Xs, Ys, epochs, lr=1e-3):
        '''Trains the model on the given dataset for "epoch" number of itterations with step size="lr".
        Returns list containing loss for each epoch.'''
        history = []
        k = 0
        for epoch in tqdm_notebook(range(epochs)):
            num_samples = Xs.shape[0]
            print(num_samples)
            for i in range(num_samples):
                sample_input = Xs[i,:].reshape((1,self.input_shape))

                sample_target = Ys[i,:]


                activations = self.forward_pass(sample_input)
                deltas = self.backward_pass(sample_target, activations)

                layer_inputs = [sample_input] + activations[:-1]
                self.weight_update(deltas, layer_inputs, lr)

            preds = self.predict(Xs)
            if self.mode == 'regression':
                current_loss = self.mean_squared_error(preds, Ys)
            elif self.mode == 'classification':
                current_loss = self.cross_entropy_loss(preds, Ys)
            print(current_loss)
            history.append(current_loss)
        k+=1
        print(k)
        return history



    def forward_pass(self, input_data):
        '''Executes the feed forward algorithm.
        "input_data" is the input to the network in row-major form
        Returns "activations", which is a list of all layer outputs (excluding input layer of course)'''
        activations = []
        #print(self.weights_)
        for i in range(len(self.nodes_per_layer)-2):
            if i==0:
                hidden_layer = np.dot(input_data,self.weights_[i])+self.biases_[i]
            else:
                 hidden_layer = np.dot(activations[-1],self.weights_[i])+self.biases_[i]

            hidden_layer_activation = NeuralNetwork.RB(hidden_layer)

            hidden_layer_activation = NeuralNetwork.sigmoid(hidden_layer)

            activations.append(hidden_layer_activation[:])

        output_layer = np.dot(activations[-1],self.weights_[-1])+self.biases_[-1]

        if self.mode=='classification':
            output_layer_activation = NeuralNetwork.RB(output_layer)
            output_layer_activation = NeuralNetwork.softmax(output_layer)
        elif self.mode=='regression':
            output_layer_activation = NeuralNetwork.sigmoid(output_layer)
        activations.append(output_layer_activation)
        return activations

    def backward_pass(self, targets, layer_activations):
        '''Executes the backpropogation algorithm.
        "targets" is the ground truth/labels
        "layer_activations" are the return value of the forward pass step
        Returns "deltas", which is a list containing weight update values for all layers (excluding the input layer of course)'''
        deltas = []
        '''error_l = layer_activations[1]-targets
        dy_dz_o = (layer_activations[1])*(1-(layer_activations[1]))
        delta_w_h = error_l*dy_dz_o
        error_h = np.transpose(np.dot(self.weights_[1],np.transpose(delta_w_h)))
        dy_dz_h = (layer_activations[0])*(1-(layer_activations[0]))
        delta_wi = np.multiply(error_h,dy_dz_h)
        deltas = [delta_wi,delta_w_h]'''


        errorl = layer_activations[-1] - targets
        dy_dz_o = layer_activations[-1]
        dy_dz_o = layer_activations[-1]*(1-layer_activations[-1])
        delta_w_h = errorl*dy_dz_o
        error_h = np.transpose(np.dot(self.weights_[-1],np.transpose(delta_w_h)))
        deltas.append(delta_w_h)
        for i in range(self.num_layers-3,-1,-1):
            #print("i is:",i)
            dy_dz_h = layer_activations[i]
            dy_dz_h = (layer_activations[i])*(1-(layer_activations[i]))
            delta_wi = np.multiply(error_h,dy_dz_h)
            error_h = np.transpose(np.dot((self.weights_[i]),np.transpose(delta_wi)))
            deltas.append(delta_wi)


        #print("Total layers:",self.num_layers)
        deltas.reverse()
        return deltas




    def weight_update(self, deltas, layer_inputs, lr):
        '''Executes the gradient descent algorithm.
        "deltas" is return value of the backward pass step
        "layer_inputs" is a list containing the inputs for all layers (including the input layer)
        "lr" is the learning rate'''
        #print(self.weights_)
        #print("Delats are:",len(deltas))
        #print(deltas)
        for i in range(len(layer_inputs)-2,-1,-1):
            self.weights_[i] = self.weights_[i] - lr*np.transpose((np.transpose(deltas[i]))*layer_inputs[i])
            self.biases_[i] = self.biases_[i] - lr*deltas[i]

    def predict(self, Xs):
        '''Returns the model predictions (output of the last layer) for the given "Xs".'''
        predictions = []
        num_samples = Xs.shape[0]
        for i in range(num_samples):
            sample = Xs[i,:,:].reshape((1,self.input_shape))
            sample_prediction = self.forward_pass(sample)[-1]
            predictions.append(sample_prediction.reshape((self.output_shape,)))
        return np.array(predictions)

    def evaluate(self, Xs, Ys):
        '''Returns appropriate metrics for the task, calculated on the dataset passed to this method.'''
        pred = self.predict(Xs)
        if self.mode == 'regression':
            return self.mean_squared_error(pred, Ys)
        elif self.mode == 'classification':
            return self.cross_entropy_loss(pred, Ys), self.accuracy(pred.argmax(axis=1), Ys.argmax(axis=1))
    def save_weights(self):
        np.save('w_i',self.weights_[0])
        np.save('w_h',self.weights_[1])
    def load_weights(self):
        self.weights_[0] = np.load('w_i.npy')
        self.weights_[1] = np.load('w_h.npy')



temp_list = []
temp_list = np.array(temp_list)
def give_test_images():
    temp_list = []
    labels = []
    #numbers = ['\\train\\0']#,'\\train\\1','\\train\\2','\\train\\3','\\train\\4']
    numbers = ['\\train\\0','\\train\\1','\\train\\2','\\train\\3','\\train\\4','\\train\\5','\\train\\6','\\train\\7','\\train\\8','\\train\\9']
    for number in numbers:
        cwd = os.getcwd()
        cwd = cwd+number
        k = 0
        i = 0
        for images in os.listdir(cwd):
            k+=1
            i+=1
            im = (cv.imread(cwd+'\\'+images))
            im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)

            im = cv.resize(im,(784,1))
            im = im/255
            m = np.mean(im)
            v = np.var(im)
            im = np.subtract(im,m)
            im = np.divide(im,v)
            temp_list.append(im)



        print(i)
        labels.append(i)
    temp_list = np.array(temp_list)
    mean_list = np.mean(temp_list)
    var_list = np.var(temp_list)
    temp_list = np.subtract(temp_list,mean_list)
    temp_list = np.divide(temp_list,var_list)
    return temp_list,labels
temp_list,numbers = give_test_images()
def generate_labels(numbers):
    test_labels = []
    j = 0
    for number in numbers:
        for i in range(number):
            if j == 0:
                test_labels.append([1,0,0,0,0,0,0,0,0,0])
            elif j == 1:
                test_labels.append([0,1,0,0,0,0,0,0,0,0])
            elif j == 2:
                test_labels.append([0,0,1,0,0,0,0,0,0,0])
            elif j == 3:
                test_labels.append([0,0,0,1,0,0,0,0,0,0])
            elif j == 4:
                test_labels.append([0,0,0,0,1,0,0,0,0,0])
            elif j == 5:
                test_labels.append([0,0,0,0,0,1,0,0,0,0])
            elif j == 6:
                test_labels.append([0,0,0,0,0,0,1,0,0,0])
            elif j == 7:
                test_labels.append([0,0,0,0,0,0,0,1,0,0])
            elif j == 8:
                test_labels.append ([0,0,0,0,0,0,0,0,1,0])
            elif j == 9:
                test_labels.append([0,0,0,0,0,0,0,0,0,1])
        j+=1
    return np.array(test_labels)
train_labels = generate_labels(numbers)
train_labels = np.array(train_labels)
xTrain,xTest,yTrain,yTest = train_test_split(temp_list,train_labels,test_size = 0.10)
print(xTrain.shape)
nn = NeuralNetwork([784,50,10],mode="classification")
nn.load_weights()
history = nn.fit(xTrain,yTrain,2,lr=0.0001)
print(history)
y_predict = nn.predict(xTrain)
print(nn.accuracy(y_predict,yTrain))
y_predict = nn.predict(xTest)
print(nn.accuracy(y_predict,yTest))
