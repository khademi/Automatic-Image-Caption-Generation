from __future__ import print_function
import os
import tensorflow as tf
import numpy
import pickle
from models import ICG_model, ICG_model_grid, ICG_model_att


tfrecord_dir = "/local-scratch"
NUM_EPOCHS = 10000
PRE_TRAINED = True
rng = numpy.random.RandomState(1234)

f = open('../data/coco_dict.pickle','rb')
coco_dict = pickle.load(f)
f.close()

# Setting hyperparameter values
hyperparameters = {'batch_size': 100, 'save_freq': 1000, 'vocab_size': len(coco_dict), 'model': 'att_icg',
                   'word_emb_size': 512, 'feat_dim': 2048, 'dropout_keep_rate': 0.5, 'learning_rate': 0.00001
                   ,'grid_feat_dim': 1024,'grid_size': 14, 'grid_emb_size': 512}
feat_dim = hyperparameters['feat_dim']
grid_feat_dim = hyperparameters['grid_feat_dim']
grid_size = hyperparameters['grid_size']


# Load features from the tf record file
filename = os.path.join(tfrecord_dir, "coco_train_val_480_514.tfrecords")  
filename_queue = tf.train.string_input_producer([filename], num_epochs=NUM_EPOCHS)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        'caption': tf.VarLenFeature(tf.int64),
        'target': tf.VarLenFeature(tf.int64),
        'data_type': tf.FixedLenFeature([1], tf.int64),
        'im_id': tf.FixedLenFeature([1], tf.int64),
        'visual_concept': tf.FixedLenFeature([1000], tf.float32),
        'grid_feat': tf.FixedLenFeature([grid_feat_dim * grid_size * grid_size], tf.float32),
        'feat': tf.FixedLenFeature([feat_dim], tf.float32)      
    })


# Loading the model
if hyperparameters['model'] == 'icg' :    
    model = ICG_model(features, hyperparameters, is_train=True)
    DIR = "../model/icg"

if hyperparameters['model'] == 'icg_deep' :    
    model = ICG_model(features, hyperparameters, is_train=True)
    DIR = "../model/icg_deep"
    
if hyperparameters['model'] == 'att_icg' :
    model = ICG_model_att(features, hyperparameters, is_train=True)
    DIR = "../model/att_icg"

if hyperparameters['model'] == 'grid_icg' :
    model = ICG_model_grid(features, hyperparameters, is_train=True)
    DIR = "../model/grid_icg"

train_step = model._train_step
info = model._info
cross_entropy = model._cross_entropy
logits= model._logits
emb = model._W_emb


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
init = tf.local_variables_initializer()
sess.run(init)


if PRE_TRAINED:
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(DIR))


# Coordinator
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=1000)

# Run the training session and save the results every save frequency time
try:
    step = 0
    total_cost = 0.0
    while not coord.should_stop():
        step += 1
        cross_entropy_cost, _ = sess.run([cross_entropy, train_step])
        total_cost += cross_entropy_cost
        if step % hyperparameters['save_freq'] == 0:
            print('%d steps.' % (step))
            print(total_cost / hyperparameters['save_freq'])
            total_cost = 0.0
            saver.save(sess, os.path.join(DIR, "model"), global_step=step)
except tf.errors.OutOfRangeError:
    print('Done training for %d epochs, %d steps.' % (NUM_EPOCHS, step))
finally:
    # When done, ask the threads to stop
    coord.request_stop()
# Wait for threads to finish
coord.join(threads)
sess.close()








