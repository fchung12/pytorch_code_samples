from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            # 8x8 image -> 6x6 output after conv
            ConvLayer2D(name='conv1',input_channels=3,kernel_size=3,number_filters=3,init_scale=0.022),
            # 6x6 -> 3x3 after maxpool
            # needs pool_size 2 stride 2 for (3x3)x3 input to fully conn
            MaxPoolingLayer(name='maxpool1',stride=2,pool_size=2),
            flatten(name='flatten'),
            fc(27,5,name='fc1',init_scale=0.022),
            
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        

        '''
        # submission 1
        self.net = sequential(
            # 32x32 -> 28x28
            ConvLayer2D(name='conv1',input_channels=3,kernel_size=5,number_filters=3),
            # 28x28 -> 13x13
            MaxPoolingLayer(name='maxpool1',stride=2,pool_size=4),
            #gelu(name='gelu1'),
            # 13x13 -> 9x9
            ConvLayer2D(name='conv2',input_channels=3,kernel_size=5,number_filters=2),
            # 9x9 -> 7x7
            MaxPoolingLayer(name='maxpool2',stride=1,pool_size=3),
            gelu(name='gelu2'),
            flatten(name='flatten'),
            fc(7*7*2,20,name='fc1')
        )
        '''
        
        self.net = sequential(
            ########## TODO: ##########
            # 32x32 -> 28x28
            ConvLayer2D(name='conv1',input_channels=3,kernel_size=5,number_filters=30),
            # 28x28 -> 13x13
            MaxPoolingLayer(name='maxpool1',stride=2,pool_size=4),
            # 13x13 -> 9x9
            ConvLayer2D(name='conv2',input_channels=30,kernel_size=5,number_filters=20),
            flatten(name='flatten'),
            gelu('gelu'),
            fc(81*20,20,name='fc1'),
            ########### END ###########
        )
        
        
        '''
        self.net = sequential(
            #32x32 -> 18x18
            ConvLayer2D(name='conv1',input_channels=3,kernel_size=15,number_filters=1),
            MaxPoolingLayer(name='maxpool1',stride=1,pool_size=3),
            flatten(name='flatten'),
            gelu(name='gelu'),
            fc(16*16,20,name='fc1'),   
        )
        '''
        
        '''        self.net = sequential(
            # 32x32 -> 13x13
            ConvLayer2D(name='conv1',input_channels=3,kernel_size=20,number_filters=3),
            # 13x13 -> 11x11
            MaxPoolingLayer(name='maxpool1',stride=1,pool_size=3),
            # 11x11 -> 7x7
            ConvLayer2D(name='conv2',input_channels=3,kernel_size=5,number_filters=3),
            gelu(name='gelu'),
            flatten(name='flatten'),
            fc(7*7*3,20, name='fc1')
        )
        '''
        
        