import tensorflow as tf
import numpy as np
from model import DenseNet, CnnLstm, train_parser


batch_size = 128

tf.reset_default_graph()

train_data_dir = "./tfrecords/train.tfrecord"

train_dataset = tf.data.TFRecordDataset(train_data_dir).map(train_parser)
train_dataset = train_dataset.shuffle(500000, seed=1, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(batch_size)

train_itr = tf.contrib.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

spec, label = train_itr.get_next()
spec = tf.reshape(spec, [-1, 128, 100, 1])
spec = tf.cast(spec, tf.float32)

train_init_op = train_itr.make_initializer(train_dataset)

model = DenseNet()
# model = CnnLstm()

height = 128
width = 100
num_classes = 12
learning_rate = 0.01
epochs = 5

model_path = "./densenet/"
model_file = "densenet"

with tf.device('/gpu:0'):
    X = tf.placeholder(tf.float32, [None, height, width, 1])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    global_step = tf.Variable(0, trainable=False, name='global_step')

    logits_train = model.get_logits(X)
    
    loss = tf.losses.softmax_cross_entropy(Y, logits_train)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):    
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
        
    logits_eval = model.get_logits(X, is_training=False, reuse=True)
    predict_proba_ = tf.nn.softmax(logits_eval)
    prediction = tf.argmax(predict_proba_, 1)
    accuracy = tf.metrics.accuracy(tf.argmax(Y, 1), prediction)
                
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy[1])
        
    merged = tf.summary.merge_all()
    

saver = tf.train.Saver(tf.global_variables())

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))


sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

writer = tf.summary.FileWriter(model_path, sess.graph)
        
for epoch in range(epochs):
    sess.run(train_init_op)
    acc = []
    
    while True:
        try:
            step = sess.run(global_step)
            
            _spec, _label = sess.run([spec, label])
                
            _, c, _summ = sess.run([optimizer, loss, merged], feed_dict = {X: _spec, Y: _label})
            acc_train = sess.run(accuracy, feed_dict = {X: _spec, Y: _label})
            
            acc.append(acc_train[1])
            
            writer.add_summary(_summ, step)
            
            if step % 500 == 0:
                print('step: {}, cost: {}'.format(step, c))
                
        except tf.errors.OutOfRangeError:
            break
            
    print('epoch: {}, cost : {}, train_acc: {}'.format(epoch, c, np.mean(acc)))


saver.save(sess, model_path + model_file + '.ckpt', global_step=sess.run(global_step))

print("Model is saved.")
