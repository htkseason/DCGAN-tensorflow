import time
import tensorflow as tf
import numpy
import celeba_input
import os
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import GraphKeys
from tensorflow.python.framework.ops import convert_to_tensor
from dcgan.discriminator import Dnet
from dcgan.generator import Gnet


record_log = False
record_log_dir = './log/dcgan/'


iters = int(-1)

is_training = tf.placeholder(tf.bool)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.0002, global_step, 500, 1, staircase=False)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True



image64 = celeba_input.inputs(64)
ones64 = tf.ones([64, 1], dtype=tf.int32)
random64 = tf.random_uniform([64, 100], -1, 1, tf.float32, name='input_noise')
zeros64 = tf.zeros([64, 1], dtype=tf.int32)


# ========================


with tf.variable_scope('gnet'):
    gnet = Gnet(random64, is_training)
with tf.variable_scope('dnet'):
    dnet = Dnet(image64, is_training)
with tf.variable_scope('dnet', reuse=True):
    dgnet = Dnet(gnet.layer5.output, is_training)



with tf.variable_scope('loss'):
    weight_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    cross_entropy_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dnet.layer5.logit, labels=tf.cast(ones64, tf.float32))) + \
                    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dgnet.layer5.logit, labels=tf.cast(zeros64, tf.float32)))
    cross_entropy_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dgnet.layer5.logit, labels=tf.cast(ones64, tf.float32)))
    accuarcy = tf.reduce_mean(tf.cast(tf.concat([tf.greater(dnet.layer5.output , 0.5) , tf.less(dgnet.layer5.output , 0.5)] , axis=0), tf.float32))
    
    summary_losses = [tf.summary.scalar('cross_entropy_d', cross_entropy_d), tf.summary.scalar('cross_entropy_g', cross_entropy_g)]

# ===================

train_step_d = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(cross_entropy_d,
                                                                        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="dnet"))
train_step_g = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(cross_entropy_g,
                                                                        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gnet"), global_step=global_step)
# ===================

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners()

# ===================
merged = tf.summary.merge([summary_losses,tf.summary.image('gnet_image', gnet.layer5.output, 32)])


if record_log:
    log_writer = tf.summary.FileWriter(record_log_dir, sess.graph)
    

saver = tf.train.Saver()

# ========restore===============
#saver.restore(sess, tf.train.get_checkpoint_state(record_log_dir).model_checkpoint_path)
# tf.train.write_graph(sess.graph_def, "./log/", "graph.pb", as_text=True);

# builder = tf.saved_model.builder.SavedModelBuilder("./saved_model")
# builder.add_meta_graph_and_variables(sess, ["tf-muct"])
# builder.save()
# =============================

start_time = time.time()

while True:
    
    [ _ ] = sess.run([ train_step_d], feed_dict={ is_training: True})
    [ _ ] = sess.run([ train_step_g], feed_dict={ is_training: True})

    if global_step.eval() % 100 == 0 :
        print('step = %d, lr = %g, time = %g min' % (global_step.eval(), learning_rate.eval(), (time.time() - start_time) / 60.0))
        [ acc , loss_d, loss_g , summ] = sess.run([ accuarcy, cross_entropy_d, cross_entropy_g, merged], feed_dict={is_training: False})
        print(acc)
        print(loss_d)
        print(loss_g)
        if record_log:
            log_writer.add_summary(summ, global_step.eval())
        print("==================")
        
    if global_step.eval() % 500 == 499 :
        # [gene_imgs] = sess.run([ gnet.layer5.output], feed_dict={ is_training: False})
        # plt.imshow((gene_imgs[0] + 1.0) / 2.0)
        # plt.show()
        saver.save(sess, os.path.join(record_log_dir, 'model.ckpt'), global_step.eval())



print('total time = ', time.time() - start_time, 's')

if record_log:
    log_writer.close();
    pass


