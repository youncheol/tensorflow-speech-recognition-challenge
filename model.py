import tensorflow as tf
import numpy as np


class DenseNet:
    def batch_norm_relu(self, inputs, is_training, reuse, name):
        bn = tf.layers.batch_normalization(inputs, 
                                           training=is_training, 
                                           reuse=reuse, 
                                           name=name)
        outputs = tf.nn.relu(bn)
        
        return outputs
    
    
    def initial_conv(self, inputs, reuse=False):
        l = tf.layers.conv2d(inputs=inputs,
                             filters=16,
                             kernel_size=3,
                             strides=2,
                             padding='SAME',
                             name='init_conv',
                             reuse=reuse)
        
        return l

    
    def composite_layer(self, inputs, keep_prob, name, is_training=True, reuse=False):
        l = inputs
        l = self.batch_norm_relu(l, is_training, reuse, name=name+'_bn1')
        l = tf.layers.conv2d(l, 4 * 12, 1, 1, 
                             padding='SAME', name=name+'_conv1', reuse=reuse)
            
        l = self.batch_norm_relu(l, is_training, reuse, name=name+'_bn2')
        
        l = tf.layers.conv2d(l, 12, 3, 1, 
                             padding='SAME', name=name+'_conv2', reuse=reuse)
        
        l = tf.layers.dropout(l, keep_prob, training=is_training)
        
        return tf.concat([inputs, l], axis=3) 


    def transition_layer(self, inputs, name, is_training=True, reuse=False):
        shape = inputs.get_shape().as_list()
        n_filters = int(shape[3] * 0.5)
        
        l = self.batch_norm_relu(inputs, is_training, reuse, name=name + '_bn')
        l = tf.layers.conv2d(l, n_filters, 1, 1, padding='SAME', name=name + '_conv', reuse=reuse)
        l = tf.layers.average_pooling2d(l, 2, 2, name='pool')

        return l
    
    def dense_net(self, inputs, keep_prob=0.2, is_training=True, reuse=False):
        l = self.initial_conv(inputs=inputs, reuse=reuse)
        
        with tf.variable_scope('block1') as scope:
            for i in range(6):
                l = self.composite_layer(l, 
                                         keep_prob, 
                                         name='dense_layer{}'.format(i), 
                                         is_training=is_training,
                                         reuse=reuse)
                
            l = self.transition_layer(l, 
                                     name='transition1',
                                     is_training=is_training,
                                     reuse=reuse)
                
        with tf.variable_scope('block2') as scope:
            for i in range(12):
                l = self.composite_layer(l, 
                                         keep_prob, 
                                         name='dense_layer{}'.format(i),
                                         is_training=is_training,
                                         reuse=reuse)

            l = self.transition_layer(l, 
                                     name='transition2',
                                     is_training=is_training,
                                     reuse=reuse)

        with tf.variable_scope('block3') as scope:
            for i in range(24):
                l = self.composite_layer(l, 
                                         keep_prob, 
                                         name='dense_layer{}'.format(i),
                                         is_training=is_training,
                                         reuse=reuse)
                
            l = self.transition_layer(l, 
                                     name='transition3',
                                     is_training=is_training,
                                     reuse=reuse)

        with tf.variable_scope('block4') as scope:
            for i in range(16):
                l = self.composite_layer(l, 
                                         keep_prob, 
                                         name='dense_layer{}'.format(i),
                                         is_training=is_training,
                                         reuse=reuse)
                
        return l
    
    
    def build_graph(self, inputs, is_training=True, reuse=False):
        l = self.dense_net(inputs, keep_prob=0.2, is_training=is_training, reuse=reuse)
        
        outputs = self.batch_norm_relu(l, is_training, reuse, name='last_bn')

        shape = outputs.get_shape().as_list()
        
        pool_size = (shape[1], shape[2])
        outputs= tf.layers.average_pooling2d(outputs, pool_size=pool_size, strides=1, padding='VALID')
        
        outputs = tf.layers.flatten(outputs)
        outputs = tf.layers.dense(outputs, 12, name='final_dense', reuse=reuse)
        
        return outputs
    
    
class CnnLstm:
    def __init__(self):
        self.num_classes = 12
        self.num_filters = [8, 16, 32, 32]
        self.filter_sizes = [7, 3, 3, 3]
        self.pool_sizes = [2, 2, 1, 1]
        self.cnn_dropout_keep_prob = [0, 0.3, 0.4, 0.4]
        self.fc_hidden_units = [1028, 512, 256]
        self.fc_dropout_keep_prob = [0.2, 0.3, 0.35]
        self.lstm_n_hiddens = [512]
        self.lstm_dropout_keep_prob = [0.5]
        self.idx_convolutional_layers = range(1, len(self.filter_sizes) + 1)
        self.idx_fc_layers = range(1, len(self.fc_hidden_units) + 1)
        self.idx_lstm_layers = range(1, len(self.lstm_n_hiddens) + 1)
        
        
    def convolutional_layer(self, inputs, is_training=True, reuse=False):
        l = inputs
        
        for i, num_filter, filter_size, pool_size, keep_prob in zip(self.idx_convolutional_layers,
                                                                    self.num_filters,
                                                                    self.filter_sizes,
                                                                    self.pool_sizes,
                                                                    self.cnn_dropout_keep_prob):
            l = tf.layers.conv2d(l, 
                                 filters=num_filter, 
                                 kernel_size=filter_size, 
                                 strides=1, 
                                 padding="SAME", 
                                 name="conv"+str(i),
                                 reuse=reuse)
            
            l = tf.layers.batch_normalization(l, training=is_training, name="conv_bn"+str(i), reuse=reuse)
            l = tf.nn.relu(l, name="conv_relu"+str(i))
            l = tf.layers.dropout(l, rate=keep_prob, training=is_training, name="conv_dropout"+str(i))

            if pool_size != 1:
                l = tf.layers.max_pooling2d(l, pool_size=pool_size, strides=pool_size, padding="SAME")
                
        return l
        
    
    def fc_layer(self, inputs, is_training=True, reuse=False):
        l = inputs
        
        for i, units, keep_prob in zip(self.idx_fc_layers, self.fc_hidden_units, self.fc_dropout_keep_prob):
            l = tf.layers.dense(inputs, units=units, reuse=reuse, name="fc"+str(i))
            l = tf.layers.batch_normalization(l, training=is_training, name="fc_bn"+str(i), reuse=reuse)
            l = tf.nn.relu(l, name="fc_relu"+str(i))
            l = tf.layers.dropout(l, rate=keep_prob, training=is_training, name="fc_dropout"+str(i))
            
        return l
  

    def lstm_layer(self, inputs, is_training=True, reuse=False):
        if is_training:
            keep_probs = [0.5]
            
        else:
            keep_probs = [1]
            
        cell = tf.nn.rnn_cell.BasicLSTMCell(512, reuse=reuse)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_probs[0])
        
        outputs, states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = outputs[-1]
        
        return outputs
 

    def get_reshaped_cnn_to_rnn(self, inputs):
        shape = inputs.get_shape().as_list() 
        inputs = tf.transpose(inputs, [0, 2, 1, 3])
        reshaped_inputs = tf.reshape(inputs, [-1, shape[2], shape[1] * shape[3]])
        
        return reshaped_inputs
  

    def get_logits(self, inputs, is_training=True, reuse=False):
        with tf.variable_scope("conv_layers") as scope:
            l = inputs
            l = self.convolutional_layer(l, is_training, reuse)
            
        with tf.variable_scope("lstm_layers") as scope:
            reshaped_l = self.get_reshaped_cnn_to_rnn(l)
            
            l = self.lstm_layer(reshaped_l, is_training, reuse)
            
        with tf.variable_scope("fc_layers") as scope:
            l = tf.layers.flatten(l)
            l = self.fc_layer(l, is_training, reuse)
                
        output = tf.layers.dense(l, units=self.num_classes, reuse=reuse, name='out')
            
        return output
    

def train_parser(serialized_example):
    features = {
        "spectrum": tf.FixedLenFeature([12800], tf.float32),
        "label": tf.FixedLenFeature([12], tf.int64)
    }

    parsed_feature = tf.parse_single_example(serialized_example, features)

    spec = parsed_feature['spectrum']
    label = parsed_feature['label']

    return spec, label
        
    
def test_parser(serialized_example):
    features = {
        "spectrum": tf.FixedLenFeature([12800], tf.float32),
    }

    parsed_feature = tf.parse_single_example(serialized_example, features)

    spec = parsed_feature['spectrum']

    return spec
