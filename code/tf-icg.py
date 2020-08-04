from __future__ import print_function
import os
import tensorflow as tf
import scipy.io
import numpy
import pickle
import json
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer('\w+')

feat_dir = "../data/res-feat/"
f = open('../data/coco_dict.pickle','rb')
coco_dict = pickle.load(f)
f.close()

def make_tfrecord(data_type: str, writer: tf.io.TFRecordWriter):
    if data_type == 'train2014':
        flag = 0
    elif data_type == 'val2014':
        flag = 1
    with open('../data/annotations/captions_' + data_type + '.json') as file:  
        captions = json.load(file)
                    
    for i in range(len(captions['annotations'])):
        sentence = captions['annotations'][i]['caption']
        tokens = tokenizer.tokenize(sentence)
        tokens = [w.lower() for w in tokens]
        word_ids = [0]
        target_ids = []
        for w in tokens:
            if not w in coco_dict: 
                word_ids.append(2)
                target_ids.append(2)
            else:
                word_ids.append(coco_dict[w])
                target_ids.append(coco_dict[w])
        target_ids.append(1)
    
        image_id = str(captions['annotations'][i]['image_id'])        
        # Get the visual concepts:
        # https://github.com/s-gupta/visual-concepts
        prob = scipy.io.loadmat('/local-scratch/coco-vc/' + data_type + '/vc_' 
                                + image_id + '.mat')
        prob = prob['pb'].astype(float)
        prob = numpy.reshape(prob, (-1))               
        # Get the features of the image
        feat = scipy.io.loadmat(feat_dir + 'layer514/' + data_type + '/res_' +
                                image_id + '.mat')
        feat = feat['feat'].astype(float)
        feat = numpy.reshape(feat, (-1))
        
        grid_feat = scipy.io.loadmat(feat_dir + 'layer480/' + data_type +
                                     '/res_' + image_id + '.mat')
        grid_feat = grid_feat['feat'].astype(float)
        grid_feat = numpy.reshape(grid_feat, (-1))
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'caption': tf.train.Feature(int64_list=tf.train.Int64List(
                value= word_ids)),
            'target': tf.train.Feature(int64_list=tf.train.Int64List(
                value= target_ids)),
            'data_type': tf.train.Feature(int64_list=tf.train.Int64List(
                value=[flag])),
            'im_id': tf.train.Feature(int64_list=tf.train.Int64List(
                value = [int(image_id)])),
            'visual_concept':tf.train.Feature(float_list=tf.train.FloatList(
                value= prob)),
            'grid_feat': tf.train.Feature(float_list=tf.train.FloatList(
                value=grid_feat)),
            'feat': tf.train.Feature(float_list=tf.train.FloatList(
                value=feat))}))
        writer.write(example.SerializeToString())
    print('done set ', data_type)   
    return
    
def prepare_data():
    filename = os.path.join("../data", 'coco_train_val_480_514' + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)
    make_tfrecord('train2014', writer)
    make_tfrecord('val2014', writer)   
    writer.close()
    return

if __name__ == '__main__':
  prepare_data()

