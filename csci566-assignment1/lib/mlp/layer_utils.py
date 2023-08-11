from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v
    
    def apply_l1_regularization(self, lam):
        """
        Gather gradients for L1 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                ######## TODO ########
                # L1 gradient boils down to lam*sign(w)
                # https://math.stackexchange.com/questions/2692200/how-does-l-1-regularization-present-itself-in-gradient-descent
                params = self.params[n]
                self.grads[n] += (lam*np.sign(params))
                #self.grads[n] = v - lam*np.sign(params)
                ######## END  ########
    
    def apply_l2_regularization(self, lam):
        """
        Gather gradients for L2 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                ######## TODO ########
                # L2 gradient just becomes 2*lam*w 
                # https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261
                params = self.params[n]
                self.grads[n] += (2*lam*params)
                ######## END  ########


    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))


class flatten(object):
    def __init__(self, name="flatten"):
        """
        - name: the name of current layer
        - meta: to store the forward pass activations for computing backpropagation
        Note: params and grads should be just empty dicts here, do not update them
        """
        self.name = name
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, feat):
        output = None
        #############################################################################
        # TODO: Implement the forward pass of a flatten layer.                      #
        # You need to reshape (flatten) the input features.                         #
        # Store the results in the variable self.meta provided above.               #
        #############################################################################
        # flattens to (batchsize, total input size)
        output = np.reshape(feat,(feat.shape[0],np.prod(feat.shape[1:])))
        #output=feat    
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        dfeat = None
        #############################################################################
        # TODO: Implement the backward pass of a flatten layer.                     #
        # You need to reshape (flatten) the input gradients and return.             #
        # Store the results in the variable dfeat provided above.                   #
        #############################################################################
        dfeat = np.reshape(dprev,feat.shape)
        #pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        return dfeat


class fc(object):
    def __init__(self, input_dim, output_dim, init_scale=0.002, name="fc"):
        """
        In forward pass, please use self.params for the weights and biases for this layer
        In backward pass, store the computed gradients to self.grads
        - name: the name of current layer
        - input_dim: input dimension
        - output_dim: output dimension
        - meta: to store the forward pass activations for computing backpropagation
        """
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(input_dim, output_dim)
        self.params[self.b_name] = np.zeros(output_dim)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None

    def forward(self, feat):
        output = None
        assert len(feat.shape) == 2 and feat.shape[-1] == self.input_dim, \
            "But got {} and {}".format(feat.shape, self.input_dim)
        #############################################################################
        # TODO: Implement the forward pass of a single fully connected layer.       #
        # Store the results in the variable output provided above.                  #
        #############################################################################
        # wx + b 
        
        # bias needs to be stacked into a (batch_size,b) matrix for batch 
        batch_size = feat.shape[0]
        batch_b = np.repeat([self.params[self.b_name]],batch_size,axis=0)
        forward = np.dot(feat,self.params[self.w_name]) + batch_b
        output = forward
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        dfeat, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        assert len(feat.shape) == 2 and feat.shape[-1] == self.input_dim, \
            "But got {} and {}".format(feat.shape, self.input_dim)
        assert len(dprev.shape) == 2 and dprev.shape[-1] == self.output_dim, \
            "But got {} and {}".format(dprev.shape, self.output_dim)
        #############################################################################
        # TODO: Implement the backward pass of a single fully connected layer.      #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dfeat provided above.          #
        #############################################################################
        # o = wx + b
        # dL/do = dprev
        batch_size = dprev.shape[0]
        # do/dw = x
        # dL/dw = dL/do * do/dw = dprev * feat
        '''
        update = np.zeros(self.params[self.w_name].shape)
        for b in range(0,batch_size):
            update += np.outer(feat[b],dprev[b])

        self.grads[self.w_name] = update
        '''     
              
        #update = update/batch_size
        #self.grads[self.w_name] = update 
        #print(feat.T.shape)
        #print(dprev.shape)
        self.grads[self.w_name] = np.dot(feat.T,dprev)
        #self.grads[self.w_name] = np.sum(np.matmul(dprev,feat))
        #self.grads[self.w_name]= np.dot(feat,dprev.T)
        #self.grads[self.w_name] = np.sum(np.dot(feat.T,dprev))
        # do/db = 1
        # dL/db = dL/do * do/db = dprev * 1
        '''
        update = np.zeros(self.params[self.b_name].shape)
        for b in range(0,batch_size):
            update += dprev[b]
        update = update/batch_size
        '''        
        
        #self.grads[self.b_name] = dprev[0]
        #self.grads[self.b_name] = dprev
        self.grads[self.b_name] = np.sum(dprev,axis=0)

        # do/dx = w
        # dL/dx = dL/do * do/dx = dprev * w
        dfeat = np.dot(dprev,self.params[self.w_name].T)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        return dfeat

class gelu(object):
    def __init__(self, name="gelu"):
        """
        - name: the name of current layer
        - meta:  to store the forward pass activations for computing backpropagation
        Notes: params and grads should be just empty dicts here, do not update them
        """
        self.name = name 
        self.params = {}
        self.grads = {}
        self.meta = None 
    
    def forward(self, feat):
        output = None
        #############################################################################
        # TODO: Implement the forward pass of GeLU                                  #
        # Store the results in the variable output provided above.                  #
        #############################################################################
        #feat_3 = np.dot(np.dot(feat,feat.T),feat)
        feat_3 = feat**3
        inside_tanh = np.sqrt(2/np.pi)* (feat + 0.044715*feat_3)
        tanh_term = 1+np.tanh(inside_tanh)
        output = 0.5*feat*(tanh_term)
        #pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = feat
        return output
    
    def backward(self, dprev):
        """ You can use the approximate gradient for GeLU activations """
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        dfeat = None
        #############################################################################
        # TODO: Implement the backward pass of GeLU                                 #
        # Store the output gradients in the variable dfeat provided above.          #
        #############################################################################
        #o = GELU(x) = f(x)
        #dL/do = dprev
        #do/dx = f`(x)
        
        # backprop derivative: https://arxiv.org/pdf/2104.02523.pdf (page 3)
        feat_3 = feat**3
        term1 = 0.5*(np.tanh(0.0356774*feat_3 + 0.797885*feat))
        term2 = (0.0535161*feat_3 + 0.398942*feat)
        term3 = np.cosh(0.0356774*feat_3 + 0.797885*feat)**-2
        f_prime = term1 + 0.5 + (term2 * term3)
        # dL/dx = dprev * f_prime
        dfeat = dprev * f_prime
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        return dfeat



class dropout(object):
    def __init__(self, keep_prob, seed=None, name="dropout"):
        """
        - name: the name of current layer
        - keep_prob: probability that each element is kept.
        - meta: to store the forward pass activations for computing backpropagation
        - kept: the mask for dropping out the neurons
        - is_training: dropout behaves differently during training and testing, use
                       this to indicate which phase is the current one
        - rng: numpy random number generator using the given seed
        Note: params and grads should be just empty dicts here, do not update them
        """
        self.name = name
        self.params = {}
        self.grads = {}
        self.keep_prob = keep_prob
        self.meta = None
        self.kept = None
        self.is_training = False
        self.rng = np.random.RandomState(seed)
        assert keep_prob >= 0 and keep_prob <= 1, "Keep Prob = {} is not within [0, 1]".format(keep_prob)

    def forward(self, feat, is_training=True, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        kept = None
        output = None
        #############################################################################
        # TODO: Implement the forward pass of Dropout.                              #
        # Remember if the keep_prob = 0, there is no dropout.                       #
        # Use self.rng to generate random numbers.                                  #
        # During training, need to scale values with (1 / keep_prob).               #
        # Store the mask in the variable kept provided above.                       #
        # Store the results in the variable output provided above.                  #
        #############################################################################
        if is_training:
            # draws a random number, keep it if is less than keep prob
            mask = None
            scale = 1
            if self.keep_prob == 0:
                mask = np.ones(shape=feat.shape)
                scale = 1
            else:
                mask = self.rng.uniform(low=0,high=1,size=feat.shape) < self.keep_prob
                scale = 1/self.keep_prob

            # kept is a matrix of 1s and 0s
            kept = mask
            # scaled_mask is a matrix of scale and 0s
            scaled_mask = mask*scale
            output = scaled_mask * feat

        else:
            kept = np.ones(shape=feat.shape)
            output = feat
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.kept = kept
        self.is_training = is_training
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        dfeat = None
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        #############################################################################
        # TODO: Implement the backward pass of Dropout                              #
        # Select gradients only from selected activations.                          #
        # Store the output gradients in the variable dfeat provided above.          #
        #############################################################################
        # dL/do = dprev
        # o = dropout(x) = {scale * x if kept, 0 otherwise}
        # do/dx = {scale if kept, 0 otherwise}
        # dL/dx = {scale * dprev if kept, 0 otherwise} = dL/do * do/dx
        if self.is_training:
            if self.keep_prob == 0:
                keep_prob = 1
            else:
                keep_prob = self.keep_prob
            dfeat = (self.kept*(1/keep_prob)) * dprev
        else:
            dfeat = dprev
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.is_training = False
        self.meta = None
        return dfeat


def onehot(labels, max_label):
    onehot = np.eye(max_label)[labels.astype(int)]
    return onehot

class cross_entropy(object):
    def __init__(self, size_average=True):
        """
        - size_average: if dividing by the batch size or not
        - logit: intermediate variables to store the scores
        - label: Ground truth label for classification task
        """
        self.size_average = size_average
        self.logit = None
        self.label = None

    def forward(self, feat, label):
        logit = softmax(feat)
        loss = None
        #############################################################################
        # TODO: Implement the forward pass of an CE Loss                            #
        # Store the loss in the variable loss provided above.                       #
        #############################################################################
        max_label = logit.shape[1]
        batch_size = logit.shape[0]
        label_onehot = onehot(label,max_label)
        masked = logit*label_onehot
        true_prob = np.sum(masked,axis=1)
        loss = -np.log(true_prob)
        #use onehot label as mask for logits
        #loss = -np.sum(np.log(logit*label_onehot))
        if self.size_average:
            # For some reason we sum over the batch (from office hours)
            loss = np.sum(loss)/batch_size
        else:
            loss = np.sum(loss)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.logit = logit
        self.label = label
        return loss

    def backward(self):
        logit = self.logit
        label = self.label
        if logit is None:
            raise ValueError("No forward function called before for this module!")
        dlogit = None
        #############################################################################
        # TODO: Implement the backward pass of an CE Loss                           #
        # Store the output gradients in the variable dlogit provided above.         #
        #############################################################################
        # derivative simplifies to p-y:
        # https://deepnotes.io/softmax-crossentropy#derivative-of-cross-entropy-loss-with-softmax
        max_label = logit.shape[1]
        batch_size = logit.shape[0]
        label_onehot = onehot(label,max_label)
        dlogit = logit - label_onehot
        if self.size_average:
            dlogit = dlogit/batch_size
        else:
            dlogit = dlogit
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.logit = None
        self.label = None
        return dlogit


def softmax(feat):
    scores = None

    #############################################################################
    # TODO: Implement the forward pass of a softmax function                    #
    # Return softmax values over the last dimension of feat.                    #
    #############################################################################
    not_zero = 1e-6 # small value in softmax to prevent log(0) causing training collapse
    scores = np.exp(feat.T+not_zero)/np.sum(np.exp(feat.T+not_zero),axis=0)
    scores = scores.T
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return scores

def reset_seed(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
