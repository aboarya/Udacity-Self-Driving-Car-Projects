# all imports
import time
import pickle
import csv
import operator
from itertools import groupby

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


_index_in_epoch = 0
num_classes = 43
BATCH_SIZE = 64
EPOCHS = 100

def shuffle(x, y):
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    return (x, y)

def train_test_split(X, Y):
    count = int(len(X)*.7)
    
    X, Y = shuffle(X, Y)

    X_train = X[:count]
    Y_train = Y[:count]

    X_val = X[count:]
    Y_val = Y[count:]

    return (X_train, Y_train, X_val, Y_val)

def group_classes(Y):
    return {key:len(list(group)) for key, group in groupby(Y)}

def group_classes_sorted(Y):
    data = group_classes(Y)
    return sorted(data.items(), key=lambda x:x[1], reverse=True)


def plot_frequency(xlabel, xs, ys, with_names=True):
    fig, ax = plt.subplots(figsize=(15, 12))
    bars = ax.barh(xs, ys, 1, color='g', alpha=0.3)
    for i,bar in enumerate(bars):
        height = bar.get_y()
        if with_names:
            ax.text(bars[-1].get_width()-(bars[0].get_width()*6), height,
                '{} - {}'.format(i, xlabel[i]),rotation=0,ha='left', va='center')
        ax.text(bars[i].get_x()+bars[i].get_width()+10, height+bars[i].get_height()/2,
                '({} - {})'.format(i, ys[i]),rotation=0,ha='left', va='center')

    plt.show()
    
    
def get_images_and_counts(X, Y, count_data):
    images, labels, counts = [], [], []
    for label, count in count_data:
        images.append(X[Y.index(label)])
        counts.append(count)
        labels.append(label)

    return images, labels, counts


def plot_axes(axes, images, labels, counts=None, is_count=False, pred_labels=None):    
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i], cmap='binary')
        # Show true and predicted classes.
        if list(counts):
            xlabel = "Count: {0}".format(counts[i])
            title = "Class: {0}".format(labels[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], pred_labels[i])

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])


def plot_signs(images, labels, counts=None, pred_labels=None):
    """Create figure but watch out for 43!"""
    count = len(images)
    fig, axes = plt.subplots(6, 7, figsize=(10, 10))
    fig.subplots_adjust(hspace=1, wspace=1)
    plot_axes(axes, images[:-1], labels[:-1], counts, is_count=True)


def transform_image(img, ang_range, shear_range, trans_range):
    """
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 
    
    A Random uniform distribution is used to generate different parameters for transformation
    
    Copied from confluence post
    https://carnd-udacity.atlassian.net/wiki/display/CAR/Project+2+%28unbalanced+data%29+Generating+additional+data+by+jittering+the+original+image
    """
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    return img


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def next_batch(data, labels, batch_size):
    """Return the next `batch_size` examples from this data set."""
    global _index_in_epoch
    start = _index_in_epoch
    _index_in_epoch += batch_size
    _num_examples = len(data)

    if _index_in_epoch > _num_examples:
        # Shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        data = data[perm]
        labels = labels[perm]
        # Start next epoch
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples

    end = _index_in_epoch
    return data[start:end], labels[start:end]

def inference(x, keep_prob):

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def variable_summaries(var):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    # convolution
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    # X2 pooling
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """
        Reusable code for making a simple neural net layer.

        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            w_conv1 = weight_variable(input_dim)
            b_conv1 = bias_variable(output_dim)
            conv = conv2d(input_tensor, w_conv1) + b_conv1
            with tf.name_scope(layer_name+'_activation'):
                activations = act(conv)
        tf.histogram_summary(layer_name+'_activation', activations)
        return activations
    
    x = tf.reshape(x, [-1, 32, 32, 1])
    
    conv1 = nn_layer(x, [5, 5, 1, 32], [32], 'conv1')
    
    with tf.name_scope('pool1') as scope:
        conv1 = max_pool_2x2(conv1)
        
    conv2 = nn_layer(conv1, [5, 5, 32, 64], [64], 'conv2')

    with tf.name_scope('pool2') as scope:
        conv2 = max_pool_2x2(conv2)

    # Flatten
    with tf.name_scope('fc1') as scope:
        fc1 = flatten(conv2)
        fc1_shape = (fc1.get_shape().as_list()[-1], 512)
        
        # (16 * 16 * 512, 120)
        fc1_W = weight_variable((fc1_shape))
        fc1_b = bias_variable([512])
        fc1 = tf.matmul(fc1, fc1_W) + fc1_b
        fc1 = tf.nn.relu(fc1)
    
    with tf.name_scope('dropout1'):
        fc1_drop = tf.nn.dropout(fc1, keep_prob)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
            
    
    # Flatten
    with tf.name_scope('fc2') as scope:
       
        fc2_W = weight_variable((512, 128))
        fc2_b = bias_variable([128])
        fc2 = tf.matmul(fc1_drop, fc2_W) + fc2_b
        fc2 = tf.nn.relu(fc2)
    with tf.name_scope('dropout2'):
        fc2_drop = tf.nn.dropout(fc2, keep_prob)

    #2nd fully connected
    with tf.name_scope('fc3') as scope:
        w_fc3 = weight_variable([128, 43])
        b_fc3 = bias_variable([43])

    # softmax output
    with tf.name_scope('softmax') as scope:
        y_conv = tf.matmul(fc2_drop, w_fc3) + b_fc3

    return y_conv

def loss(logits, labels):
    # cross entropy
    with tf.name_scope('cross_entropy') as scope:
        diff = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
    tf.scalar_summary("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            y_pred_cls = tf.argmax(labels, 1)
            correct_prediction = tf.equal(tf.argmax(logits, 1), y_pred_cls)
        with tf.name_scope('accuracy'):
             accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy, y_pred_cls
