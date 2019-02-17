import tensorflow as tf


def leaky_relu(inputs, alpha):
    return 0.5 * (1 + alpha) * inputs + 0.5 * (1-alpha) * tf.abs(inputs)


def discriminator_model(inputs):
    outputs = tf.layers.conv2d(inputs, 64, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 64, 3, padding='same', strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 128, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 128, 3, padding='same', strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv6')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.contrib.layers.flatten(outputs)
    outputs = tf.layers.dense(outputs, units=1024, name='dense1')
    outputs = leaky_relu(outputs, alpha=0.2)
    outputs = tf.layers.dense(outputs, units=1, name='dense2')

    return outputs


def CPCE_3D(inputs, padding='valid'):
    '''
    Input : [batch, in_depth, in_height, in_width, in_channels]
    '''

    if len(inputs.shape) == 5 and inputs.shape[1] > 1:
        outputs1 = tf.layers.conv3d(inputs, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1', use_bias=False)
    else:
        if len(inputs.shape) == 5:
            inputs = tf.squeeze(inputs, 1)
        outputs1 = tf.layers.conv2d(inputs, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1', use_bias=False)
    outputs2 = tf.nn.relu(outputs1)

    if len(outputs2.shape) == 5 and outputs2.shape[1] > 1:
        outputs2 = tf.layers.conv3d(outputs2, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2', use_bias=False)
    else:
        if len(outputs2.shape) == 5:
            outputs2 = tf.squeeze(outputs2, 1)
        outputs2 = tf.layers.conv2d(outputs2, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2', use_bias=False)
    outputs3 = tf.nn.relu(outputs2)

    if len(outputs3.shape) == 5 and outputs3.shape[1] > 1:
        outputs3 = tf.layers.conv3d(outputs3, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3', use_bias=False)
    else:
        if len(outputs3.shape) == 5:
            outputs3 = tf.squeeze(outputs3, 1)
        outputs3 = tf.layers.conv2d(outputs3, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3', use_bias=False)
    outputs4 = tf.nn.relu(outputs3)

    if len(outputs4.shape) == 5 and outputs4.shape[1] > 1:
        outputs4 = tf.layers.conv3d(outputs4, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4', use_bias=False)
    else:
        if len(outputs4.shape) == 5:
            outputs4 = tf.squeeze(outputs4, 1)
        outputs4 = tf.layers.conv2d(outputs4, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4', use_bias=False)
    outputs5 = tf.nn.relu(outputs4)

    if len(outputs5.shape) == 5:
        outputs5 = tf.squeeze(outputs5, 1)
    outputs5 = tf.layers.conv2d_transpose(outputs5, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv5', use_bias=False)

    if len(outputs3.shape) == 5:
        outputs5 = tf.concat([outputs3[:, outputs3.shape[1]/2, :, :, :], outputs5], 3)
    else:
        outputs5 = tf.concat([outputs3, outputs5], 3)
    outputs5 = tf.nn.relu(outputs5)

    outputs5 = tf.layers.conv2d(outputs5, 32, 1, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='transpose1', use_bias=False)
    outputs6 = tf.nn.relu(outputs5)

    outputs6 = tf.layers.conv2d_transpose(outputs6, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv6', use_bias=False)
    if len(outputs2.shape) == 5:
        outputs6 = tf.concat([outputs2[:, outputs2.shape[1]/2, :, :, :], outputs6], 3)
    else:
        outputs6 = tf.concat([outputs2, outputs6], 3)
    outputs6 = tf.nn.relu(outputs6)

    outputs6 = tf.layers.conv2d(outputs6, 32, 1, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='transpose2', use_bias=False)
    outputs7 = tf.nn.relu(outputs6)

    outputs7 = tf.layers.conv2d_transpose(outputs7, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv7', use_bias=False)
    if len(outputs1.shape) == 5:
        outputs7 = tf.concat([outputs1[:, outputs1.shape[1]/2, :, :, :], outputs7], 3)
    else:
        outputs7 = tf.concat([outputs1, outputs7], 3)
    outputs7 = tf.nn.relu(outputs7)

    outputs7= tf.layers.conv2d(outputs7, 32, 1, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='transpose3', use_bias=False)
    outputs8 = tf.nn.relu(outputs7)

    outputs8 = tf.layers.conv2d_transpose(outputs8, 1, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv8', use_bias=False)
    outputs = tf.nn.relu(outputs8)

    return outputs


def CPCE_2D(inputs, padding='valid'):
    # inputs = tf.placeholder(dtype=tf.float32, shape=[None, input_width, input_height, 1])
    outputs1 = tf.layers.conv2d(inputs, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1', use_bias=False)
    outputs2 = tf.nn.relu(outputs1)

    outputs2 = tf.layers.conv2d(outputs2, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2', use_bias=False)
    outputs3 = tf.nn.relu(outputs2)

    outputs3 = tf.layers.conv2d(outputs3, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3', use_bias=False)
    outputs4 = tf.nn.relu(outputs3)

    outputs4 = tf.layers.conv2d(outputs4, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4', use_bias=False)
    outputs5 = tf.nn.relu(outputs4)

    outputs5 = tf.layers.conv2d_transpose(outputs5, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv5', use_bias=False)
    outputs5 = tf.concat([outputs3, outputs5], 3)
    outputs5 = tf.nn.relu(outputs5)

    outputs5 = tf.layers.conv2d(outputs5, 32, 1, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='transpose1', use_bias=False)
    outputs6 = tf.nn.relu(outputs5)

    outputs6 = tf.layers.conv2d_transpose(outputs6, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv6', use_bias=False)
    outputs6 = tf.concat([outputs2, outputs6], 3)
    outputs6 = tf.nn.relu(outputs6)

    outputs6 = tf.layers.conv2d(outputs6, 32, 1, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='transpose2', use_bias=False)
    outputs7 = tf.nn.relu(outputs6)

    outputs7 = tf.layers.conv2d_transpose(outputs7, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv7', use_bias=False)
    outputs7 = tf.concat([outputs1, outputs7], 3)
    outputs7 = tf.nn.relu(outputs7)

    outputs7 = tf.layers.conv2d(outputs7, 32, 1, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='transpose3', use_bias=False)
    outputs8 = tf.nn.relu(outputs7)

    outputs8 = tf.layers.conv2d_transpose(outputs8, 1, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv8', use_bias=False)
    outputs = tf.nn.relu(outputs8)

    return outputs


def CPCE_2D_shortcut(inputs, padding='valid'):
    # inputs = tf.placeholder(dtype=tf.float32, shape=[None, input_width, input_height, 1])
    # adding shortcut connection from input to output may speed up training.
    #
    outputs1 = tf.layers.conv2d(inputs, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1', use_bias=False)
    outputs2 = tf.nn.relu(outputs1)

    outputs2 = tf.layers.conv2d(outputs2, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2', use_bias=False)
    outputs3 = tf.nn.relu(outputs2)

    outputs3 = tf.layers.conv2d(outputs3, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3', use_bias=False)
    outputs4 = tf.nn.relu(outputs3)

    outputs4 = tf.layers.conv2d(outputs4, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4', use_bias=False)
    outputs5 = tf.nn.relu(outputs4)

    outputs5 = tf.layers.conv2d_transpose(outputs5, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv5', use_bias=False)
    outputs5 = tf.concat([outputs3, outputs5], 3)
    outputs5 = tf.nn.relu(outputs5)

    outputs5 = tf.layers.conv2d(outputs5, 32, 1, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='transpose1', use_bias=False)
    outputs6 = tf.nn.relu(outputs5)

    outputs6 = tf.layers.conv2d_transpose(outputs6, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv6', use_bias=False)
    outputs6 = tf.concat([outputs2, outputs6], 3)
    outputs6 = tf.nn.relu(outputs6)

    outputs6 = tf.layers.conv2d(outputs6, 32, 1, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='transpose2', use_bias=False)
    outputs7 = tf.nn.relu(outputs6)

    outputs7 = tf.layers.conv2d_transpose(outputs7, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv7', use_bias=False)
    outputs7 = tf.concat([outputs1, outputs7], 3)
    outputs7 = tf.nn.relu(outputs7)

    outputs7 = tf.layers.conv2d(outputs7, 32, 1, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='transpose3', use_bias=False)
    outputs8 = tf.nn.relu(outputs7)

    outputs8 = tf.layers.conv2d_transpose(outputs8, 1, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv8', use_bias=False)
    outputs = tf.nn.relu(outputs8 + inputs)

    return outputs

def vgg_model(inputs):

    outputs = tf.concat([inputs*255-103.939, inputs*255-116.779, inputs*255-123.68], 3)
    outputs = tf.layers.conv2d(outputs, 64, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv1_1')
    outputs = tf.layers.conv2d(outputs, 64, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv1_2')
    outputs = tf.layers.max_pooling2d(outputs, 2, strides=(2, 2), padding='same', name='pool1')

    outputs = tf.layers.conv2d(outputs, 128, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv2_1')
    outputs = tf.layers.conv2d(outputs, 128, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv2_2')
    outputs = tf.layers.max_pooling2d(outputs, 2, strides=(2, 2), padding='same', name='pool2')

    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv3_1')
    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv3_2')
    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv3_3')
    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv3_4')
    outputs = tf.layers.max_pooling2d(outputs, 2, strides=(2, 2), padding='same', name='pool3')

    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv4_1')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv4_2')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv4_3')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv4_4')
    outputs = tf.layers.max_pooling2d(outputs, 2, strides=(2, 2), padding='same', name='pool4')

    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv5_1')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv5_2')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv5_3')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv5_4')

    return outputs
