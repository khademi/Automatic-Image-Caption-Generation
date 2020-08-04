import tensorflow as tf
import os
import numpy
from models import ICG_model, ICG_model_grid, ICG_model_att
import scipy.io
import pickle
import math
import json

image_names = os.listdir('../data/images/val2014/')
f = open('../data/coco_dict.pickle','rb')
coco_dict = pickle.load(f)
f.close()

VISUAL_CONCEPT_SIZE = 1000
coco_dict_rev = {}
for key, value in coco_dict.items():
    coco_dict_rev[value] = key

hyperparameters = {'batch_size': 100, 'vocab_size': len(coco_dict), 
                   'model': 'att_icg', 'word_emb_size': 512, 'feat_dim': 2048,
                   'dropout_keep_rate': 0.5, 'grid_feat_dim': 1024, 
                   'grid_size': 14, 'grid_emb_size': 512}

feat_dim = hyperparameters['feat_dim']
batch_size = hyperparameters['batch_size']
grid_feat_dim = hyperparameters['grid_feat_dim']
grid_size = hyperparameters['grid_size']

if hyperparameters['model'] == 'icg':    
    features = {'feat': tf.placeholder(tf.float32, [None, feat_dim])}
    model = ICG_model(features, hyperparameters, is_train=False)
    DIR = "../model/icg"
    feat_batch = model._feat_batch
    feat_dir = '../data/res-feat/layer514/val2014/'
if hyperparameters['model'] == 'icg_deep':    
    features = {'feat': tf.placeholder(tf.float32, [None, feat_dim])}
    model = ICG_model(features, hyperparameters, is_train=False)
    DIR = "../model/icg_deep"
    feat_batch = model._feat_batch
    feat_dir = '../data/res-feat/layer514/val2014/'  
if hyperparameters['model'] == 'att_icg':
    features = {'grid_feat': tf.placeholder(
        tf.float32, [None, grid_feat_dim * grid_size * grid_size])}
    model = ICG_model_att(features, hyperparameters, is_train=False)
    DIR = "../model/att_icg"
    grid_feat_batch = model._grid_feat_batch
    feat_dir = '../data/res-feat/layer480/val2014/'    
    
if hyperparameters['model'] == 'grid_icg':
    features = {'grid_feat': tf.placeholder(
        tf.float32, [None, grid_feat_dim * grid_size * grid_size]),
                'pb': tf.placeholder(tf.float32, [None, VISUAL_CONCEPT_SIZE])}
    model = ICG_model_grid(features, hyperparameters, is_train=False)
    DIR = "../model/grid_icg"
    grid_feat_batch = model._grid_feat_batch
    visual_concept_batch = model._visual_concept_batch
    feat_dir = '../data/res-feat/layer480/val2014/'
    
ids = model._ids
with tf.Session() as sess:    
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(DIR))
    all_dict = []
    for k in range(math.ceil(len(image_names) / batch_size)):
        image_names_batch = image_names[k*batch_size: (k+1)*batch_size]
        feats = []
        probs = []
        for image_name in image_names_batch:
            imid = str(int(image_name[14:-4]))
            feat = scipy.io.loadmat(feat_dir + 'res_' + imid + '.mat')
            feat = feat['feat'].astype(float)
            feat = numpy.reshape(feat, (-1), order = 'C')
            feats.append(feat)           
            prob = scipy.io.loadmat('/local-scratch/coco-vc/val2014/vc_' +
                                    imid + '.mat')
            prob = prob['pb'].astype(float)
            prob = numpy.reshape(prob, (-1))
            probs.append(prob)           
        feats = numpy.asarray(feats, dtype=numpy.float32)
        probs = numpy.asarray(probs, dtype=numpy.float32)       
        if hyperparameters['model'] == 'grid_icg':
            ids_val= sess.run(ids, feed_dict = {grid_feat_batch: feats,
                                                visual_concept_batch: probs})
        if hyperparameters['model'] == 'att_icg':
            ids_val= sess.run(ids, feed_dict = {grid_feat_batch: feats})
        if hyperparameters['model'] == 'icg':
            ids_val= sess.run(ids, feed_dict = {feat_batch: feats})
        if hyperparameters['model'] == 'icg_deep':
            ids_val= sess.run(ids, feed_dict = {feat_batch: feats})
        for l in range(len(image_names_batch)):
            image_dic = {}
            sentence = []
            for j in range(ids_val[l, :, 0].shape[0]):
                sentence.append(coco_dict_rev[ids_val[l][j][0]])
            sentence_without_stop = []
            for value in sentence:
                if value != '<STOP>':
                    sentence_without_stop.append(value)
                else:
                    break
            image_dic['caption'] = ' '.join(sentence_without_stop)
            image_dic['image_id'] = int(image_names_batch[l][14:-4])
            print(image_dic)
            all_dict.append(image_dic)
    
with open('../data/results/captions_val2014_icg_results.json', 'w') as outfile:  
    json.dump(all_dict, outfile)

