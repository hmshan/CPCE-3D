
import tensorflow as tf
import numpy as np
import cPickle as pickle

# Convert 2D ckpt model to 3D numpy
def ckpt_to_numpy(checkpoint_dir, save_name = 'tmp_generate_weight', depth=3):
    v_names = ['conv1', 'conv2', 'conv3', 'conv4', 'deconv5', 'transpose1', 'deconv6', 'transpose2', 'deconv7', 'transpose3', 'deconv8']

    weights = dict()
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            var_name_list = var_name.split('/')
            if len(var_name_list) == 3 and var_name_list[0] == 'generator_model':
                weights[var_name_list[1]] = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

    in_depths = range(depth, 1, -2)

    for i in xrange(len(in_depths)):
        a = weights[v_names[i]]
        b = np.zeros(a.shape, dtype=np.float32)
        d = np.array([b,a,b])
        weights[v_names[i]] = d
    for i in v_names:
        print i, weights[i].shape
    pickle.dump(weights, open(save_name, 'w'))
    return save_name

def normalize_x(x, in_depth = 9, lower = -300.0, upper = 300.0):
    x = (x - 1024.0 - lower) / (upper - lower)
    x[x<0.0] = 0.0
    x[x>1.0] = 1.0
    x = np.expand_dims(x,4)
    interval = in_depth // 2
    center = x.shape[1] // 2
    return x[:,center-interval : center+1+interval,:,:,:]

def normalize_y(x, lower = -300.0, upper = 300.0):
    x = np.squeeze(x) # remove depth
    x = (x - 1024.0 - lower) / (upper - lower)
    x[x<0.0] = 0.0
    x[x>1.0] = 1.0
    x = np.expand_dims(x,3)
    return x






