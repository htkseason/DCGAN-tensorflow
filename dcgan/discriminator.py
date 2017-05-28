
import tensorflow as tf
import numpy
import math
from dcgan.tfobjs import *

class Dnet:
    
    def __init__(self, input_tensor, is_training):
     
        # ===================
        with tf.variable_scope('layer1'):
            self.layer1 = ConvObj()
            self.layer1.set_input(input_tensor)
            self.layer1.batch_norm(self.layer1.conv2d([5, 5], 64, [1, 2, 2, 1]), is_training=is_training)
            self.layer1.set_output(leaky_relu(self.layer1.bn, 0.2))
        # ===================

        with tf.variable_scope('layer2'):
            self.layer2 = ConvObj()
            self.layer2.set_input(self.layer1.output)
            self.layer2.batch_norm(self.layer2.conv2d([5, 5], 128, [1, 2, 2, 1]), is_training=is_training)
            self.layer2.set_output(leaky_relu(self.layer2.bn, 0.2))
        # ===================

        with tf.variable_scope('layer3'):
            self.layer3 = ConvObj()
            self.layer3.set_input(self.layer2.output)
            self.layer3.batch_norm(self.layer3.conv2d([5, 5], 256, [1, 2, 2, 1]), is_training=is_training)
            self.layer3.set_output(leaky_relu(self.layer3.bn, 0.2))
            # ===================
    

        with tf.variable_scope('layer4'):
            self.layer4 = ConvObj()
            self.layer4.set_input(self.layer3.output)
            self.layer4.batch_norm(self.layer4.conv2d([5, 5], 512, [1, 2, 2, 1]), is_training=is_training)
            self.layer4.set_output(leaky_relu(self.layer4.bn, 0.2))
            # ===================
    

        with tf.variable_scope('layer5'):
            self.layer5 = FcObj()
            self.layer5.set_input(self.layer4.output)
            self.layer5.activation = tf.nn.sigmoid(self.layer5.fc(1))
            self.layer5.set_output(self.layer5.activation)
            # ===================
