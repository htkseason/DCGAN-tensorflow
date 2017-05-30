import os
import tensorflow as tf
from celeba import celeba_input
import numpy
import matplotlib.pyplot as plt
from celeba import celeba_input
from tensorflow.python.platform import gfile

record_log_dir = '../log/xavier-dcgan/'

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True


sess = tf.InteractiveSession(config=config)

#===restore
saver = tf.train.import_meta_graph(tf.train.get_checkpoint_state(record_log_dir).model_checkpoint_path + '.meta')
saver.restore(sess, tf.train.get_checkpoint_state(record_log_dir).model_checkpoint_path)

gnet_output = tf.get_default_graph().get_tensor_by_name('gnet/layer5/Tanh:0')
gnet_output_grade = tf.get_default_graph().get_tensor_by_name('dnet_1/layer5/output:0')
is_training = tf.get_default_graph().get_tensor_by_name('Placeholder:0')


#====testing


[imgs, img_grades] = sess.run([gnet_output, gnet_output_grade], feed_dict={is_training : False})
print(imgs.shape)
print(img_grades.shape)

for i in range(imgs.shape[0]):
    if (img_grades[i][0] > 0.0):
        print(img_grades[i])
        if imgs.shape[-1] == 0:
            plt.imshow((imgs[i, :, :, 0] + 1.0) / 2.0, cmap='gray')
        else:
            plt.imshow((imgs[i] + 1.0) / 2.0)
        plt.show()
        
