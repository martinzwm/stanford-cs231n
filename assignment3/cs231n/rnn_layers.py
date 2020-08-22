from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""
def rnn_step_forward(x, h_prev, Wx, Wh, b):
    z = np.dot(x, Wx) + np.dot(h_prev, Wh) + b
    out = np.tanh(z)
    
    cache = (x, h_prev, Wx, Wh, z)
    
    return out, cache

def rnn_step_backward(dnext_h, cache):
    x, prev_h, Wx, Wh, z = cache
    dz = dnext_h * (1-(np.tanh(z))*(np.tanh(z)))
    
    dx = np.dot(dz, Wx.T)
    dWx = np.dot(x.T, dz)
    
    dprev_h = np.dot(dz, Wh.T)
    dWh = np.dot(prev_h.T, dz)
    
    db = np.sum(dz, axis=0)
    
    return dx, dprev_h, dWx, dWh, db

def rnn_forward(x, h0, Wx, Wh, b):
    N,T,D = x.shape
    _, H = h0.shape
    
    h = np.zeros((N,T,H))
    cache = []
    
    prev_h = h0
    
    for i in range(T):
        h[:,i,:], cache_i = rnn_step_forward(x[:,i,:], prev_h, Wx, Wh, b)
        prev_h = h[:,i,:]
        cache.append(cache_i)
        
    return h, cache

def rnn_backward(dh, cache):
    N,T,H = dh.shape
    D = cache[0][0].shape[1]
    
    dx = np.zeros((N,T,D))
    
    dx[:,T-1,:], dh_i, dWx, dWh, db = rnn_step_backward(dh[:,T-1,:], cache[T-1])
    
    for i in range(T-2, -1, -1):
        dx_i, dh_i, dWx_i, dWh_i, db_i = rnn_step_backward(dh[:,i,:]+dh_i, cache[i])
        dx[:,i,:] = dx_i
        dWx += dWx_i
        dWh += dWh_i
        db += db_i
        
    dh0 = dh_i
    
    return dx, dh0, dWx, dWh, db

def word_embedding_forward(x, W):
    out = W[x]
    cache = (x,W)
    return out, cache

def word_embedding_backward(dout, cache):
    x, W = cache
    dW = np.zeros(W.shape)
    np.add.at(dW, x, dout)
    
    return dW


def temporal_affine_forward(x, W, b):
    N, T, D = x.shape
    D, M = W.shape
    x_2D = x.reshape(N*T, D)
    
    out = np.dot(x_2D, W) + b
    out = out.reshape(N, T, M)
    
    cache = (x, W, b, out)
    return out, cache

def temporal_affine_backward(dout, cache):
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db

def temporal_softmax_loss(scores, y, mask, verbose=False):
    N, T, V = scores.shape
    scores = scores.reshape(N*T, V)
    y = y.reshape(N*T)
    mask = mask.reshape(N*T)
    
    probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    
    loss = - 1/N * np.sum(np.log(probs[range(N*T), y]))
    
    dscores = probs
    dscores[range(N*T), y] -= 1
    dscores /= N
    
    if verbose: print('dscores: ', dscores.shape)
    
    dscores = dscores.reshape(N, T, V)
    
    return loss, dscores

def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    N, H = prev_h.shape
    
    a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    i = sigmoid(a[:,:H])
    f = sigmoid(a[:,H:2*H])
    o = sigmoid(a[:,2*H:3*H])
    g = np.tanh(a[:,3*H:4*H])
    
    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)
    
    cache = (x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c)
    
    return next_h, next_c, cache
    

def lstm_step_backward(dnext_h, dnext_c, cache):
    x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c = cache
    
    N, H = dnext_h.shape
    
    do = dnext_h * np.tanh(next_c)
    dnext_c += dnext_h * o * (1-np.tanh(next_c)*np.tanh(next_c))
    
    dprev_c = dnext_c * f
    df = dnext_c * prev_c
    di = dnext_c * g
    dg = dnext_c * i
    
    da = np.zeros((N,4*H))
    da[:,:H] = di * i * (1-i)
    da[:,H:2*H] = df * f * (1-f)
    da[:,2*H:3*H] = do * o * (1-o)
    da[:,3*H:4*H] = dg * (1-g*g)
    
    dx, dWx = np.dot(da, Wx.T), np.dot(x.T, da)
    dprev_h, dWh = np.dot(da, Wh.T), np.dot(prev_h.T, da)
    db = np.sum(da, axis=0)
    
    return dx, dprev_h, dprev_c, dWx, dWh, db

def lstm_forward(x, h0, Wx, Wh, b):
    N, T, D = x.shape
    _, H = h0.shape
    
    h, cache = np.zeros((N, T, H)), []
    
    prev_h, prev_c = h0, np.zeros((N,H))
    
    for i in range(T):
        next_h, next_c, cache_i = lstm_step_forward(x[:,i,:], prev_h, prev_c, Wx, Wh, b)
        h[:,i,:] = next_h
        cache.append(cache_i)
        
        prev_h, prev_c = next_h, next_c
    
    return h, cache

def lstm_backward(dh, cache):
    N, T, H = dh.shape
    dx_i, dprev_h, dprev_c, dWx, dWh, db = lstm_step_backward(dh[:,-1,:], np.zeros((N,H)), cache[-1])
    
    N, D = dx_i.shape
    dx = np.zeros((N, T, D))
    dx[:,-1,:] = dx_i
    
    dnext_h, dnext_c = dprev_h, dprev_c
    
    for i in range(T-2, -1, -1):
        dx[:,i,:], dprev_h, dprev_c, dWx_i, dWh_i, db_i = lstm_step_backward(
                dh[:,i,:]+dnext_h, dnext_c, cache[i])
        dnext_h, dnext_c = dprev_h, dprev_c
        dWx += dWx_i
        dWh += dWh_i
        db += db_i
        
    dh0 = dprev_h
    
    return dx, dh0, dWx, dWh, db
    
    