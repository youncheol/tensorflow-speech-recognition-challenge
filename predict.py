import tensorflow as tf
import numpy as np
import pandas as pd
import csv
from model import DenseNet, CnnLstm, test_parser


batch_size = 128

tf.reset_default_graph()

test_data_dir = "./tfrecords/test.tfrecord"

test_dataset = tf.data.TFRecordDataset(test_data_dir).map(test_parser)
test_dataset = test_dataset.batch(batch_size)

test_itr = tf.contrib.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)

test_spec = test_itr.get_next()

test_spec = tf.reshape(test_spec, [-1, 128, 100, 1])
test_spec = tf.cast(test_spec, tf.float32)

test_init_op = test_itr.make_initializer(test_dataset)

height = 128
width = 100

model = DenseNet()
# model = CnnLstm()

with tf.device('/gpu:0'):
    X = tf.placeholder(tf.float32, [None, height, width, 1])
    
    logits_test = model.get_logits(X, is_training=False, reuse=False)
    test_predict_proba_ = tf.nn.softmax(logits_test)
    test_prediction = tf.argmax(test_predict_proba_, 1)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


# Restore model
imp_model = tf.train.import_meta_graph('./densenet/densenet.ckpt-44810.meta')
imp_model.restore(sess, tf.train.latest_checkpoint('./densenet/'))


# Create submission file
sess.run(test_init_op)

test_spec_ = sess.run(test_spec)

predict = sess.run(test_prediction, feed_dict={X: test_spec_})

while True:
    try:
        test_spec_ = sess.run(test_spec)

        predict = np.hstack([predict, sess.run(test_prediction, feed_dict={X: test_spec_})])
        
    except tf.errors.OutOfRangeError:
        break

class_names = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'silence', 'stop', 'unknown', 'up', 'yes']

df = pd.read_csv("sample_submission.csv")
files = df['fname']

model_path = "./densenet/"
model_file = "densenet"

with open(model_path + 'sub_' + model_file + '.csv', 'w') as f:
    fieldnames = ['fname', 'label']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    writer.writeheader()
    
    for i in range(len(predict)):
        writer.writerow({'fname': files[i], 'label': class_names[predict[i]]})
        
print("Submission file is created.")


# Create predict proba file
sess.run(test_init_op)

test_spec_ = sess.run(test_spec)

predict_proba = sess.run(test_predict_proba_, feed_dict={X: test_spec_})

while True:
    try:
        test_spec_ = sess.run(test_spec)
        
        predict_proba = np.vstack([predict_proba, sess.run(test_predict_proba_, feed_dict={X: test_spec_})])
            
    except tf.errors.OutOfRangeError:
        break
        
predict_proba = np.array(predict_proba)
print(predict_proba.shape)

pp = pd.DataFrame(predict_proba, index=files)
pp.to_csv(model_path + 'proba_' + model_file + '.csv', index=False)

print("Proba file is created.")