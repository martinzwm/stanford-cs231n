# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 08:27:21 2020

@author: Martin
"""

from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def loss(self, X, y=None, reg=0.0):
        # Unpack parameters
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        N, D = X.shape
        
        # Forward pass
        H = np.maximum(0, np.dot(X, W1) + b1)
        scores = np.dot(H, W2) + b2
        
        if y is None:
            return scores
        
        # Compute loss
        scores -= scores.max() # stable softmax
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        correct_logprobs = -np.log(probs[range(N), y])
        data_loss = np.sum(correct_logprobs)/N
        
        reg_loss = reg*(np.sum(W1*W1) + np.sum(W2*W2))
        
        loss = data_loss + reg_loss
        
        # Backprop
        dscores = probs
        dscores[range(N),y] -= 1
        dscores /= N
        
        dW2 = np.dot(H.T, dscores) + 2*reg*W2
        dH = np.dot(dscores, W2.T)
        dH[H<=0] = 0
        db2 = np.sum(dscores, axis=0)
        
        dW1 = np.dot(X.T, dH) + 2*reg*W1
        db1 = np.sum(dH, axis=0)
        
        grads = {}
        grads['W1'], grads['b1'] = dW1, db1
        grads['W2'], grads['b2'] = dW2, db2
        
        return loss, grads
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=200,
              learning_rate=1e-3, learning_rate_decay=0.95, reg=5e-6,
              num_iters=200, verbose=False):
        
        num_train = X_train.shape[0]
        iterations_per_epoch = max(1, num_train/batch_size)
        
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        for it in range(num_iters):
            if iterations_per_epoch == 1:
                X_batch = X_train
                y_batch = y_train
            else:
                index = np.random.choice(num_train, batch_size)
                X_batch = X_train[index]
                y_batch = y_train[index]
            
            loss, grads = self.loss(X=X_batch, y=y_batch, reg=reg)
            
            # Updates
            W1_new = self.params['W1'] - learning_rate*grads['W1']
            W2_new = self.params['W2'] - learning_rate*grads['W2']
            b1_new = self.params['b1'] - learning_rate*grads['b1']
            b2_new = self.params['b2'] - learning_rate*grads['b2']
            
            self.params['W1'] = 0.5 * (self.params['W1'] + W1_new)
            self.params['W2'] = 0.5 * (self.params['W2'] + W2_new)
            self.params['b1'] = 0.5 * (self.params['b1'] + b1_new)
            self.params['b2'] = 0.5 * (self.params['b2'] + b2_new)
            
            # Data logging for debugging
            loss_history.append(loss)
            
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' %(it, num_iters, loss))
            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                
                learning_rate *= learning_rate_decay
        
        logs = {}
        logs['loss_history'] = loss_history
        logs['train_acc_history'] = train_acc_history
        logs['val_acc_history'] = val_acc_history
        
        return logs
    
    def predict(self, X):
        scores = self.loss(X)
        y_predict = np.argmax(scores, axis=1)
        return y_predict
        