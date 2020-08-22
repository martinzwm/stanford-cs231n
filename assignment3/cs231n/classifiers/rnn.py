from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *

class CaptioningRNN(object):
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128, hidden_dim=128, cell_type='rnn',
                 dtype=np.float32):
        
        D, W, H = input_dim, wordvec_dim, hidden_dim
        
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type %s' % cell_type)
        
        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {v: k for k, v in word_to_idx.items()}
        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)
        
        V = len(word_to_idx) # number of words in the dictionary
        
        self.params = {}
        self.params['W_embed'] = np.random.randn(V, W) / 100
        
        self.params['W_proj'] = np.random.randn(D, H) / np.sqrt(D)
        self.params['b_proj'] = np.zeros(H)
        
        dim_mul = 1
        if cell_type == 'lstm': dim_mul = 4
        self.params['Wx'] = np.random.randn(W, H*dim_mul) / np.sqrt(W)
        self.params['Wh'] = np.random.randn(H, H*dim_mul) / np.sqrt(H)
        self.params['b'] = np.zeros(H*dim_mul)
        
        self.params['W_vocab'] = np.random.randn(H, V) / np.sqrt(H)
        self.params['b_vocab'] = np.zeros(V)
        
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        
    
    def loss(self, features, captions):
        captions_in, captions_out = captions[:, :-1], captions[:, 1:]
        mask = (captions_out != self._null)
        
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
        
        h0 = np.dot(features, W_proj) + b_proj
        
        x, x_cache = word_embedding_forward(captions_in, W_embed)
        
        if self.cell_type == 'rnn':
            h, h_cache = rnn_forward(x, h0, Wx, Wh, b)
        elif self.cell_type == 'lstm':
            h, h_cache = lstm_forward(x, h0, Wx, Wh, b)
        
        scores, scores_cache = temporal_affine_forward(h, W_vocab, b_vocab)
        
        loss, dscores = temporal_softmax_loss(scores, captions_out, mask)
        
        grads = {}
        dh, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dscores, scores_cache)
        
        if self.cell_type == 'rnn':
            dx, dh0, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dh, h_cache)
        elif self.cell_type == 'lstm':
            dx, dh0, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(dh, h_cache)
        
        grads['W_embed'] = word_embedding_backward(dx, x_cache)
        
        grads['W_proj'] = np.dot(features.T, dh0)
        grads['b_proj'] = np.sum(dh0, axis=0)
        
        return loss, grads
    
    def sample(self, features, max_length=30):
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
        
        
        captions = self._null * np.ones((N, max_length), dtype=np.int32)
        
        prev_word = self._start * np.ones((N, 1), dtype=np.int32)
        h0 = np.dot(features, W_proj) + b_proj
        h_prev = h0
        
        _, H = h0.shape
        if self.cell_type == 'lstm': c_prev = np.zeros((N,H))
        
        
        
        for i in range(max_length-1):
            x, _ = word_embedding_forward(prev_word, W_embed)
            
            if self.cell_type == 'rnn':    
                h, _ = rnn_step_forward(x, h_prev, Wx, Wh, b)
            elif self.cell_type == 'lstm':
                h, c, _ = lstm_step_forward(x, h_prev, c_prev, Wx, Wh, b)
                c_prev = c
            
            scores, _ = np.dot(h, W_vocab) + b_vocab
            captions[:,i] = np.argmax(scores, axis=1)
            prev_word = captions[:,i]
            
            h_prev = h
        
        
        return captions
    