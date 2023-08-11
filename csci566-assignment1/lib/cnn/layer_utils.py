from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from textwrap import indent

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
                ########### TODO #############
                pass
                ########### END  #############

    def apply_l2_regularization(self, lam):
        """
        Gather gradients for L2 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                ########### TODO #############
                pass
                ########### END  #############


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

class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0, init_scale=.02, name="conv"):
        
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size, 
                                                                input_channels, number_filters)
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
    
    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        batch_size,in_height, in_width, in_channels= input_size[0],input_size[1],input_size[2],input_size[3]
        output_height = int(((in_height+(2*self.padding)-self.kernel_size)/self.stride) + 1)
        output_width = int(((in_width+(2*self.padding)-self.kernel_size)/self.stride) + 1)
        output_channels = self.number_filters
        output_shape = (batch_size,output_height,output_width,output_channels)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _ , input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # TODO: Implement the forward pass of a single convolutional layer.         #
        # Store the results in the variable "output" provided above.                #
        #############################################################################
        # Only pad middle two indexes
        padded_in = np.pad(img,((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        _,pad_h,pad_w,_ = padded_in.shape
        
        temp_out = np.zeros(shape=output_shape)
        kernel_index = self.kernel_size


        for i in range(0,output_height):
            for j in range(0,output_width):
                # i,j contain top left of window
                # i + k_size, j + k_size contain bottom right of window
                '''
                print(str(i) + ' , ' + str(j))
                print(str(i+kernel_index) + ' , ' + str(j+kernel_index))
                print("===")
                print(padded_in.shape)
                '''
                # newaxis is added so that batch size does not mess with the broadcasting 
                # changes a (2,2) shape to a (2,2,1) and broadcasting figures out batch/filters during multiplication
                
                # Slice of image that corresponds to output (i,j)
                # shape is (batch, kernel, kernel, filter_in) and 1 for broadcasting
                indexed_img = padded_in[:,i*self.stride:i*self.stride+kernel_index,j*self.stride:j*self.stride+kernel_index,:,np.newaxis]
                
                # Swap axes to match kernel shapes
                #print(indexed_img.shape)
                #indexed_img = np.swapaxes(indexed_img,0,3)
                '''
                indexed_img = np.swapaxes(indexed_img,0,2)
                indexed_img = np.swapaxes(indexed_img,0,1)
                indexed_img = np.swapaxes(indexed_img,2,3)
                '''
                '''
                print(indexed_img.shape)
                for batch_img in indexed_img:
                    local_dot = batch_img * self.params[self.w_name]
                    temp_out[:,i,j] += np.sum(local_dot,axis=(0,1,2))
                    #print(batch_img.shape)
                '''
                #print(padded_in[:,i:i+kernel_index,j:j+kernel_index,:])
                # Multiplies matching shapes in columns together, and ignores 1 axes looks like:
                # (batch, kernel, kernel, filter_in, 1) * (1, kernel, kernel, filter_in, filter_out)
                local_dot = indexed_img * self.params[self.w_name][np.newaxis,:,:,:]
                # result is (batch, kernel, kernel, filter_in, filter_out)
                temp_out[:,i,j] = np.sum(local_dot,axis=(1,2,3))
                # (batch, index_i, index_j) = (batch, filter_out)

                
        output = temp_out + self.params[self.b_name]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img
        
        return output


    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        
        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        padded_in = np.pad(img,((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        batch_size,pad_h,pad_w,_ = padded_in.shape
        
        # Resulting gradient update of dw must match the shape of params it is updating
        temp_grad = np.zeros(shape=self.params[self.w_name].shape)

        kernel_index = self.kernel_size
        
        # Resulting output should match shape of original image without padding
        # temp_out = np.zeros(shape=padded_in.shape)
        
        # (batch, padded_img_size, padded_img_size, filter_in)
        temp_dimg = np.zeros(shape=padded_in.shape)
        '''
        output_shape = temp_out.shape
        _, output_height, output_width, _ = output_shape
        '''
        output_shape = self.get_output_size(img.shape)
        _, output_height, output_width, _ = output_shape

        kernel_index = self.kernel_size

        # For each local gradient, it is just that particular slice of the img
        for i in range(0,output_height):
            for j in range(0,output_width):
                
                # For each step in the convolution, output = local_x * kernel + b
                # do/dx = kernel
                # do/dw = local_x
                # dL/do = dprev
                
                # dL/db = 1 * dprev
                # dL/dw = local_x * dprev
                # dL/dx = kernel * dprev
                
                # this is the local image in the convolution slide (local_x)
                # and input to the convolution to get output at (i,j)
                indexed_img = padded_in[:,i*self.stride:i*self.stride+kernel_index,j*self.stride:j*self.stride+kernel_index,:,np.newaxis]
                
                # backprop gradient of the (i,j) output
                indexed_dprev = dprev[:,i:i+1,j:j+1,np.newaxis,:]
                
                # (batch, kernel, kernel, filter_in, 1) * (batch,1,1,1,filter_out)
                dL_dw = np.sum(indexed_img * indexed_dprev,axis=0)
                # (kernel kernel, filter_in, filter_out)
                temp_grad += dL_dw
                
                ########
                
                # (4, 4, 3, 12) -> (1, 4, 4, 3, 12)
                # kernel used to get output (i,j)
                match_w_shape = self.params[self.w_name][np.newaxis,:,:,:,:]
                #match_w_shape = np.swapaxes(match_w_shape)
                # (15, 1, 1, 1, 12) * (1, 4, 4, 3, 12) -> (15,4,4,3,12) 
                # (1,kernel,kernel,filter_in,filter_out) * (batch,1,1,1,filter_out)
                dL_dx = match_w_shape * indexed_dprev
                # (15,4,4,3,12) -> (15,4,4,3,1)
                dL_dx = np.sum(dL_dx, axis=4)
                # (batch,kernel,kernel,filter_in)

                temp_dimg[:,i*self.stride:i*self.stride+kernel_index,j*self.stride:j*self.stride+kernel_index,:] += dL_dx
                
        # Now, trucate temp_out to deal with the padding since
        # temp_out gradients were found including padding 
        # (batch, padded_img_size, padded_img_size, filter_in) -> (batch, img_size, img_size, filter_in))
        temp_dimg = temp_dimg[:,self.padding:-self.padding,self.padding:-self.padding,:]

        db_shape = np.zeros(shape=(1,self.number_filters))
        db_shape += np.sum(dprev,axis=(0,1,2))

        self.grads[self.w_name] = temp_grad
        self.grads[self.b_name] = db_shape
        dimg = temp_dimg
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)
        
        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        input_size = img.shape
        batch_size,in_height, in_width, in_channels= input_size[0],input_size[1],input_size[2],input_size[3]

        output_height = int(((in_height - self.pool_size)/self.stride) + 1)
        output_width = int(((in_width - self.pool_size)/self.stride) + 1)
        
        temp_out = np.zeros(shape=(batch_size,output_height,output_width,in_channels))
        
        for i in range(0,output_height):
            for j in range(0,output_width):
                # Sliding window on img corresponding to output (i,j) 
                indexed_img = img[:,i*self.stride:i*self.stride+self.pool_size,j*self.stride:j*self.stride+self.pool_size,:]

                # Output at (batch,i,j) is max pool
                temp_out[:,i,j] = np.max(indexed_img,axis=(1,2))
        
        output = temp_out
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img
        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        _, h_out, w_out, _ = dprev.shape
        h_pool, w_pool = self.pool_size,self.pool_size

        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients in self.grads with corresponding name.       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        input_size = img.shape
        batch_size,in_height, in_width, in_channels= input_size[0],input_size[1],input_size[2],input_size[3]
        
        #temp_out = np.zeros(shape=(batch_size,output_height,output_width,in_channels))
        temp_dimg = np.zeros(shape=img.shape)
        # Maxpool backward pass just passes the gradient to where the max was found 
        # in each window, 0 everywhere else (argmax instead of max)
        # output loops
        for i in range(0,h_out):
            for j in range(0,w_out):
                
                indexed_img = img[:,i*self.stride:i*self.stride+h_pool,j*self.stride:j*self.stride+w_pool,:]
                #max_index = np.where(indexed_img == np.max(indexed_img))
                '''
                max_index = np.argmax(indexed_img,axis=1)
                print(max_index.shape)
                max_index = np.argmax(max_index,axis=2)
                '''
                # batch loop
                for b in range(0,indexed_img.shape[0]):
                    # filter loop
                    for f in range(0,indexed_img.shape[3]):
                        max_index = np.unravel_index(indexed_img[b,:,:,f].argmax(),indexed_img[b,:,:,f].shape)

                        # create a 0 filter
                        zero_mask = np.zeros(shape=(h_pool,w_pool))
                        # set max_index from x to be dprev output
                        zero_mask[max_index[0],max_index[1]] = dprev[b,i,j,f]
                        # insert the mask into dimg
                        temp_dimg[b,i*self.stride:i*self.stride+h_pool,j*self.stride:j*self.stride+w_pool,f] += zero_mask

                #temp_dimg[:,i*self.stride:i*self.stride+h_pool,j*self.stride:j*self.stride+w_pool,:] = dprev[max_index][0]
        
        dimg = temp_dimg
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
