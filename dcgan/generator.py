
import tensorflow as tf
import numpy
import math
from dcgan.tfobjs import *

class Gnet:

    def __init__(self, input_tensor, is_training):
         
        # ===================
        with tf.variable_scope('layer1'):
            self.layer1 = FcObj()
            self.layer1.set_input(input_tensor)
            self.layer1.fc(4 * 4 * 1024)
            self.layer1.set_output(self.layer1.logit)
        # ===================
        with tf.variable_scope('layer1_conv'):
            self.layer1_1 = ConvObj()
            self.layer1_1.set_input(tf.reshape(self.layer1.output, [-1, 4, 4, 1024]))
            self.layer1_1.set_output(tf.nn.relu(self.layer1_1.batch_norm(self.layer1_1.input, is_training=is_training)))
        # ===================

        with tf.variable_scope('layer2'):
            self.layer2 = ConvObj()
            self.layer2.set_input(self.layer1_1.output)
            self.layer2.batch_norm(self.layer2.deconv2d([5, 5], [8, 8, 512], strides=[1, 2, 2, 1]), is_training=is_training)
            self.layer2.set_output(tf.nn.relu(self.layer2.bn))
        # ===================

        with tf.variable_scope('layer3'):
            self.layer3 = ConvObj()
            self.layer3.set_input(self.layer2.output)
            self.layer3.batch_norm(self.layer3.deconv2d([5, 5], [16, 16, 256], strides=[1, 2, 2, 1]), is_training=is_training)
            self.layer3.set_output(tf.nn.relu(self.layer3.bn))
            # ===================
    
        with tf.variable_scope('layer4'):
            self.layer4 = ConvObj()
            self.layer4.set_input(self.layer3.output)
            self.layer4.batch_norm(self.layer4.deconv2d([5, 5], [32, 32, 128], strides=[1, 2, 2, 1]), is_training=is_training)
            self.layer4.set_output(tf.nn.relu(self.layer4.bn))
            # ===================
    
        with tf.variable_scope('layer5'):
            self.layer5 = ConvObj()
            self.layer5.set_input(self.layer4.output)
            self.layer5.activation = tf.nn.tanh(self.layer5.deconv2d([5, 5], [64, 64, 3], strides=[1, 2, 2, 1]))
            self.layer5.set_output(self.layer5.activation)
            # ===================

