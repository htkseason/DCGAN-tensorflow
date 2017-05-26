import tensorflow as tf
from tensorflow.python.framework import ops  
import matplotlib.pyplot as plt
import numpy as np
import celeba_input
import matplotlib
from PIL import Image


image = celeba_input.inputs(128)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners()

for i in range(100000):
    img = image.eval()
    plt.imshow( (img[0]+1)/2.0 )
    plt.show()
    print(i)