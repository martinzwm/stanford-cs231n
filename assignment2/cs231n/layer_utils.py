# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

pass

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
from .layers import *
from .fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

# The following functions are generalized to become affine_norm_relu_forward()
# and affine_norm_relu_backward()
'''
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_params, normalization):
    a, fc_cache = affine_forward(x, w, b)
    
    bn_cache = None
    if normalization != None:
        a, bn_cache = batchnorm_forward(a, gamma, beta, bn_params)
        
    out, relu_cache = relu_forward(a)
    
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_bn_relu_backward(dout, cache, normalization):
    fc_cache, bn_cache, relu_cache = cache
    
    da = relu_backward(dout, relu_cache)
    
    dgamma, dbeta = None, None
    if normalization != None:
        da, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta
'''

# The following functions are generalized to become affine_norm_relu_do_forward()
# and affine_norm_relu_do_backward()
'''
def affine_norm_relu_forward(x, w, b, gamma, beta, bn_params, normalization):
    a, fc_cache = affine_forward(x, w, b)
    
    bn_cache = None
    if normalization == 'batchnorm':
        a, bn_cache = batchnorm_forward(a, gamma, beta, bn_params)
    elif normalization == 'layernorm':
        a, bn_cache = layernorm_forward(a, gamma, beta, bn_params) 
        
    out, relu_cache = relu_forward(a)
    
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_norm_relu_backward(dout, cache, normalization):
    fc_cache, bn_cache, relu_cache = cache
    
    da = relu_backward(dout, relu_cache)
    
    dgamma, dbeta = None, None
    if normalization == 'batchnorm':
        da, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    elif normalization == 'layernorm':
        a, bn_cache = layernorm_backward(da, bn_cache) 
    
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta
'''


def affine_norm_relu_do_forward(x, w, b, gamma, beta, bn_params,
                                normalization, dropout, do_params):
# affine -> normalization (batchnorm or layernorm) -> ReLU -> dropout
    
    # affine
    out, fc_cache = affine_forward(x, w, b)
    
    # norm
    bn_cache = None
    if normalization == 'batchnorm':
        out, bn_cache = batchnorm_forward(out, gamma, beta, bn_params)
    elif normalization == 'layernorm':
        out, bn_cache = layernorm_forward(out, gamma, beta, bn_params) 
    
    # ReLU
    out, relu_cache = relu_forward(out)
    
    # dropout
    do_cache = None
    if dropout:
        out, do_cache = dropout_forward(out, do_params)
    
    cache = (fc_cache, bn_cache, relu_cache, do_cache, normalization, dropout)
    return out, cache

def affine_norm_relu_do_backward(dout, cache):
    fc_cache, bn_cache, relu_cache, do_cache, normalization, dropout = cache
    
    # dropout
    if dropout:
        dout = dropout_backward(dout, do_cache)
    
    # ReLU
    dout = relu_backward(dout, relu_cache)
    
    # norm
    dgamma, dbeta = None, None
    if normalization == 'batchnorm':
        dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
    elif normalization == 'layernorm':
        dout, bn_cache = layernorm_backward(dout, bn_cache) 
    
    # affine
    dx, dw, db = affine_backward(dout, fc_cache)
    
    return dx, dw, db, dgamma, dbeta


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
