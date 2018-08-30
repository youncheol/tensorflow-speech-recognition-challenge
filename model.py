import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import tensorflow as tf


class DenseNet:
    def __init__(self, learning_rate=0.001):
        self.X = tf.placeholder(tf.float32, [None, 128, 100, 1])
        self.Y = tf.placeholder(tf.float32, [None, 12])
        self.training = tf.placeholder_with_default(False, shape=(), name="is_training")

        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        self.loss = None
        self.train_op = None
        self.predict = None
        self.predict_proba = None
        self.accuracy = None
        self.merged = None

        self._build_graph()

    def _composite_layer(self, inputs, keep_prob, name, training):
        with tf.variable_scope(name):
            bn_1 = tf.layers.batch_normalization(inputs, training=training, name="bn1")
            relu_1 = tf.nn.relu(bn_1, name="relu1")

            conv_1 = tf.layers.conv2d(relu_1, 4 * 12, 1, 1,
                                      padding='SAME', name="conv1")

            bn_2 = tf.layers.batch_normalization(conv_1, training=training, name="bn2")
            relu_2 = tf.nn.relu(bn_2, name="relu2")

            conv_2 = tf.layers.conv2d(relu_2, 12, 3, 1,
                                      padding='SAME', name='conv2')

            dropout = tf.layers.dropout(conv_2, keep_prob, training=training, name="dropout")

            return tf.concat([inputs, dropout], axis=3)


    def _transition_layer(self, inputs, name, training):
        with tf.variable_scope(name):
            shape = inputs.get_shape().as_list()
            n_filters = int(shape[3] * 0.5)

            bn = tf.layers.batch_normalization(inputs, training=training, name="bn")
            relu = tf.nn.relu(bn, name="relu")
            conv = tf.layers.conv2d(relu, n_filters, 1, 1, padding='SAME', name="conv")

            return  tf.layers.average_pooling2d(conv, 2, 2, name="pool")

    def _densenet(self, inputs, keep_prob, training):
        with tf.name_scope("initial_convolution"):
            l = tf.layers.conv2d(inputs=inputs,
                                 filters=16,
                                 kernel_size=3,
                                 strides=2,
                                 padding="SAME",
                                 name="init_conv")
        
        with tf.name_scope('dense_block1') as scope:
            for i in range(6):
                l = self._composite_layer(l,
                                          keep_prob,
                                          name=scope+'dense_layer{}'.format(i),
                                          training=training)

        with tf.name_scope("transition_layer1") as scope:
            l = self._transition_layer(l,
                                       name=scope+'transition1',
                                       training=training)
                
        with tf.name_scope('dense_block2') as scope:
            for i in range(12):
                l = self._composite_layer(l,
                                          keep_prob,
                                          name=scope+'dense_layer{}'.format(i),
                                          training=training)

        with tf.name_scope("transition_layer2") as scope:
            l = self._transition_layer(l,
                                       name=scope+'transition2',
                                       training=training)

        with tf.name_scope('dense_block3') as scope:
            for i in range(24):
                l = self._composite_layer(l,
                                          keep_prob,
                                          name=scope+'dense_layer{}'.format(i),
                                          training=training)

        with tf.name_scope("transition_layer3") as scope:
            l = self._transition_layer(l,
                                       name=scope+'transition3',
                                       training=training)

        with tf.name_scope('dense_block4') as scope:
            for i in range(16):
                l = self._composite_layer(l,
                                          keep_prob,
                                          name=scope+'dense_layer{}'.format(i),
                                          training=training)
                
        return l
    
    def _classification(self, inputs, training):
        with tf.name_scope("classification_layer"):
            bn = tf.layers.batch_normalization(inputs, training=training, name="last_bn")
            relu = tf.nn.relu(bn, name="last_relu")

            shape = relu.get_shape().as_list()
            pool_size = (shape[1], shape[2])

            pooling = tf.layers.average_pooling2d(relu, pool_size=pool_size, strides=1, padding="VALID")
            flat = tf.layers.flatten(pooling)
            linear = tf.layers.dense(flat, 12, name="last_dense")

        return linear

    def _optimization(self, inputs, targets):
        with tf.name_scope("loss"):
            loss = tf.losses.softmax_cross_entropy(targets, inputs)
            tf.summary.scalar("loss", loss)

        with tf.name_scope("train"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=self.global_step)

        with tf.name_scope("predict"):
            predict_proba = tf.nn.softmax(inputs)

        with tf.name_scope("accuracy"):
            prediction = tf.argmax(predict_proba, 1)
            accuracy = tf.metrics.accuracy(tf.argmax(targets, 1), prediction)
            tf.summary.scalar("accuracy", accuracy[1])

        self.merged = tf.summary.merge_all()

        return loss, train_op, prediction, predict_proba, accuracy

    def _build_graph(self):
        output = self._densenet(self.X, keep_prob=0.2, training=self.training)
        logits = self._classification(output, training=self.training)

        self.loss, self.train_op, self.predict, self.predict_proba, self.accuracy = self._optimization(logits, self.Y)

    def train(self, session, X, Y):
        return session.run([self.train_op, self.loss, self.merged],
                           feed_dict={self.X: X, self.Y: Y, self.training: True})

    def predict(self, session, X, proba=False):
        if proba:
            return session.run(self.predict_proba, feed_dict={self.X: X, self.training: False})
        else:
            return session.run(self.predict, feed_dict={self.X: X, self.training: False})



class CnnLstm:
    def __init__(self, learning_rate=0.001):
        self.num_filters = [8, 16, 32, 32]
        self.filter_sizes = [7, 3, 3, 3]
        self.pool_sizes = [2, 2, 1, 1]
        self.cnn_dropout_keep_prob = [0, 0.3, 0.4, 0.4]

        self.lstm_n_hiddens = [512]
        self.lstm_dropout_keep_prob = [0.5]

        self.fc_hidden_units = [1028, 512, 256]
        self.fc_dropout_keep_prob = [0.2, 0.3, 0.35]

        self.idx_convolutional_layers = range(1, len(self.filter_sizes) + 1)
        self.idx_fc_layers = range(1, len(self.fc_hidden_units) + 1)
        self.idx_lstm_layers = range(1, len(self.lstm_n_hiddens) + 1)

        self.X = tf.placeholder(tf.float32, [None, 128, 100, 1])
        self.Y = tf.placeholder(tf.float32, [None, 12])
        self.training = tf.placeholder_with_default(False, shape=(), name="is_training")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.learning_rate = learning_rate

        self.loss = None
        self.train_op = None
        self.pred = None
        self.predict_proba = None
        self.accuracy = None
        self.merged = None

        self._build_graph()

    def _convolutional_layer(self, inputs, training):
        with tf.variable_scope("convolutional_layers"):
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
                                     name="conv"+str(i))

                l = tf.layers.batch_normalization(l, training=training, name="bn"+str(i))
                l = tf.nn.relu(l, name="relu"+str(i))
                l = tf.layers.dropout(l, rate=keep_prob, training=training, name="dropout"+str(i))

                if pool_size != 1:
                    l = tf.layers.max_pooling2d(l, pool_size=pool_size, strides=pool_size, padding="SAME")

            return l

    def _lstm_layer(self, inputs, training):
        with tf.variable_scope("lstm_layer"):
            if training == False:
                self.lstm_dropout_keep_prob = [1]

            cell = tf.nn.rnn_cell.BasicLSTMCell(512)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.lstm_dropout_keep_prob[0])

            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            outputs = tf.transpose(outputs, [1, 0, 2])
            outputs = outputs[-1]

        return outputs

    def _fc_layer(self, inputs, training):
        with tf.variable_scope("fully_connected_layers"):
            l = tf.layers.flatten(inputs)

            for i, units, keep_prob in zip(self.idx_fc_layers, self.fc_hidden_units, self.fc_dropout_keep_prob):
                l = tf.layers.dense(l, units=units, name="fc"+str(i))
                l = tf.layers.batch_normalization(l, name="bn"+str(i))
                l = tf.nn.relu(l, name="relu"+str(i))
                l = tf.layers.dropout(l, rate=keep_prob, training=training, name="dropout"+str(i))

            logits = tf.layers.dense(l, units=12, name="last_fc")
            
        return logits

    def _get_reshaped_cnn_to_rnn(self, inputs):
        with tf.name_scope("reshape"):
            shape = inputs.get_shape().as_list()
            inputs = tf.transpose(inputs, [0, 2, 1, 3])
            reshaped_inputs = tf.reshape(inputs, [-1, shape[2], shape[1] * shape[3]])
        
        return reshaped_inputs

    def _optimization(self, inputs, targets):
        with tf.name_scope("loss"):
            loss = tf.losses.softmax_cross_entropy(targets, inputs)
            tf.summary.scalar("loss", loss)

        with tf.name_scope("train"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=self.global_step)

        with tf.name_scope("predict"):
            predict_proba = tf.nn.softmax(inputs)
            pred = tf.argmax(predict_proba, 1)

        with tf.name_scope("accuracy"):
            accuracy = tf.metrics.accuracy(tf.argmax(targets, 1), pred)
            tf.summary.scalar("accuracy", accuracy[1])

        self.merged = tf.summary.merge_all()

        return loss, train_op, pred, predict_proba, accuracy

    def _build_graph(self):
        conv_output = self._convolutional_layer(self.X, self.training)
        reshaped_output = self._get_reshaped_cnn_to_rnn(conv_output)
        lstm_output = self._lstm_layer(reshaped_output, self.training)
        logits = self._fc_layer(lstm_output, self.training)
            
        self.loss, self.train_op, self.pred, self.predict_proba, self.accuracy = self._optimization(logits, self.Y)

    def train(self, session, X, Y):
        return session.run([self.train_op, self.loss, self.merged],
                           feed_dict={self.X: X, self.Y: Y, self.training: True})

    def predict(self, session, X, proba=False):
        if proba:
            return session.run(self.predict_proba, feed_dict={self.X: X, self.training: False})
        else:
            return session.run(self.pred, feed_dict={self.X: X, self.training: False})
