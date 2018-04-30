

import tensorflow as tf
import numpy as np
from models import *
import h5py
import os
from sklearn.utils import shuffle
from util import *

os.environ['CUDA_VISIBLE_DEVICES']='0'

input_width = 64
input_height = 64

output_width = 64
output_height = 64

batch_size = 128

beta1 = 0.9
beta2 = 0.999

Mode = '3D'

###################################################

Methodname='CPCE_3D'
lambda_p = 0.1
in_depth = 9
Networkfolder = Methodname
is_transfer_learning = False

###################################################
   
# Generator
X = tf.placeholder(dtype=tf.float32, shape=[batch_size, in_depth, input_width, input_height, 1])
with tf.variable_scope('generator_model') as scope:
    Y_ = CPCE_3D(X, padding='valid')

real_data = tf.placeholder(dtype=tf.float32, shape=[batch_size, output_width, output_height, 1])

alpha = tf.random_uniform(shape=[batch_size,1], 
                            minval=0.,
                            maxval=1.)

# Discriminator
with tf.variable_scope('discriminator_model') as scope:
    disc_real = discriminator_model(real_data)
    scope.reuse_variables()
    disc_fake = discriminator_model(Y_)
    interpolates = alpha*tf.reshape(real_data, [batch_size, -1]) + (1-alpha)*tf.reshape(Y_, [batch_size, -1])
    interpolates = tf.reshape(interpolates, [batch_size, output_width, output_height, 1])
    gradients = tf.gradients(discriminator_model(interpolates), [interpolates])[0]



# Gradient penalty
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)

# Discriminator loss   
difference = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

# Generator loss
gen_cost = -tf.reduce_mean(disc_fake) 

# Final disc objective function
disc_loss = difference + 10 * gradient_penalty   # add gradient constraint to discriminator loss

gen_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator_model')
disc_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator_model')


# vgg network
with tf.variable_scope('vgg16') as scope:
    vgg_real = vgg_model(real_data)
    scope.reuse_variables()
    vgg_fake = vgg_model(Y_)


vgg_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg16')


mse_cost = tf.reduce_sum(tf.squared_difference(Y_, real_data)) / (batch_size * 64 * 64)
vgg_cost = tf.reduce_sum(tf.squared_difference(vgg_real, vgg_fake)) / (batch_size * 4 * 4 * 512)
gen_loss = gen_cost +  lambda_p * vgg_cost 

# optimizer
lr = tf.placeholder(tf.float32, shape=[])
gen_train_op = tf.train.AdamOptimizer(learning_rate=lr, 
                                        beta1=beta1,
                                        beta2=beta2
                                    ).minimize(gen_loss, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=lr, 
                                        beta1=beta1,
                                        beta2=beta2
                                    ).minimize(disc_loss, var_list=disc_params)



# training
sess = tf.Session()

sess.run(tf.global_variables_initializer())

# load vgg weights
print "Initialize VGG network ... "
weights = np.load('/Your/path/to/vgg19.npy', encoding='latin1').item()
keys = sorted(weights.keys())
layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
        'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
        'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
        'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']

for i, k in enumerate(layers):
    print i, k, weights[k][0].shape, weights[k][1].shape
    sess.run(vgg_params[2*i].assign(weights[k][0]))
    sess.run(vgg_params[2*i+1].assign(weights[k][1]))

num_epoch = 40
disc_iters = 4 
start_epoch = 0
learning_rate = 1e-4

saver = tf.train.Saver()


if is_transfer_learning:
    print "Making 3D weights from 2D"
    start_epoch = 10
    num_epoch += start_epoch 
    learning_rate = 5e-4 
    model_2D = "/Networks/CPCE_2D/CPCE_2D_" + str(start_epoch-1) +  ".ckpt" # 2D model at epoch 10.
    generate_weight = ckpt_to_numpy(model_2D, save_name='weights/'+Networkfolder,depth=in_depth)

    print "Loading 3D weights"

    g_weights = pickle.load(open(generate_weight, 'r'))
    v_names = ['conv1', 'conv2', 'conv3', 'conv4', 'deconv5', 'transpose1', 'deconv6', 'transpose2', 'deconv7', 'transpose3', 'deconv8']
    for i in xrange(len(g_weights)):
        sess.run(gen_params[i].assign(g_weights[v_names[i]]))


print "Loading data"
f = h5py.File('/Your/path/to/training_data.h5', 'r')
data, label = np.array(f['data']), np.array(f['label'])
f.close()

print "Start training ... "

for iteration in xrange(start_epoch, num_epoch):

    val_lr = learning_rate / (iteration + 1)
    data, label = shuffle(data, label)
    num_batches = data.shape[0] // batch_size
    
    i = 0
    for i in xrange(num_batches):
        
        # discriminator
        for j in xrange(disc_iters):
            idx = np.random.permutation(data.shape[0])
            batch_data = data[idx[:batch_size]]   
            batch_label = label[idx[:batch_size]] 
            sess.run([disc_train_op], feed_dict={real_data: normalize_y(batch_label),
                                                         X: normalize_x(batch_data, in_depth=in_depth),
                                                        lr: val_lr})

        batch_data = data[i*batch_size : (i+1)*batch_size]
        batch_label = label[i*batch_size : (i+1)*batch_size]

        # generator
        _disc_loss, _vgg_cost, _mse_cost, _gen_loss, _gen_cost, _ = sess.run([disc_loss, vgg_cost, mse_cost, 
                                                         gen_loss, gen_cost, gen_train_op], 
                                                        feed_dict={real_data: normalize_y(batch_label),
                                                                           X: normalize_x(batch_data, in_depth),
                                                                          lr: val_lr})
                                                        
        print('Epoch: %d  - %d - disc_loss: %.6f - gen_loss: %.6f - vgg_loss: %.6f  - mse_loss: %.6f'%(
        iteration, i, _disc_loss , _gen_loss, _vgg_cost, _mse_cost )) 

    saver.save(sess, './Networks/'+ Networkfolder +'/CPCE-3D'+ repr(iteration) + '.ckpt')
                
sess.close()





