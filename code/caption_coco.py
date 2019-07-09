import numpy
import os
import scipy.io
import json
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
import pickle


count = {}
with open('../data/annotations/captions_val2014.json', 'r') as f:
    data = json.load(f)
    data['type'] = 'captions'
with open('../data/annotations/captions_val2014.json', 'w') as f:
    json.dump(data, f)

def make_vocab(data_type):
      
    with open('../data/annotations/captions_' + data_type + '.json') as json_file:  
        captions = json.load(json_file)

    for i in range(len(captions['annotations'])):
        sentence = captions['annotations'][i]['caption']
        temp_tokens = []
        tokens = tokenizer.tokenize(sentence)
        tokens = [w.lower() for w in tokens]
        temp_tokens.append(tokens)
        for w in tokens:
            if not w in count:   
                count[w] = 1
            else:
                count[w] = count[w] + 1
    return


make_vocab('train2014')
make_vocab('val2014')


coco_dict = {'<START>': 0, '<STOP>': 1, '<UNKNOWN>': 2}
#IMAGE = 0, START = 0, STOP = 1, unknown = 2
word_id = 3
for w in count:
    if count[w] >= 8:
          coco_dict[w] = word_id
          word_id += 1

print(len(coco_dict))


f = open('../data/coco_dict.pickle', 'wb')
pickle.dump(coco_dict, f)
f.close()


region_dict = {}
def region_dictionary(result):
    region_dir = '../data/densecap/'
    with open(region_dir + result + '.json') as json_file:
        all_captions = json.load(json_file)
    for result in range(len(all_captions['results'])):
        image_data = all_captions['results'][result]
        image_name = image_data['img_name']
        image_id = str(int(image_name[image_name.rfind('_')+1: -4]))
        captions = image_data['captions']
        scores = image_data['scores']
        boxes = image_data['boxes']
        region_dict[image_id] = {}
        captions_matrix = numpy.zeros((10,20), dtype= numpy.int64)
        box_matrix = numpy.zeros((10,4), dtype= numpy.float64)
        counter = 0
        for i in range(len(captions)):
            index_dict = {}       
            if scores[i] > 1.0:
                if counter == 10:
                    break
                tokens = tokenizer.tokenize(captions[i])
                tokens = [w.lower() for w in tokens]
                word_ids = []
                for w in tokens:
                    if not w in coco_dict:   
                        word_ids.append(2)
                    else:
                        word_ids.append(coco_dict[w])
                for i in range(len(word_ids)):
                    captions_matrix[counter][i] = word_ids[i]
                box_matrix[counter] = boxes[i]
                counter = counter + 1
        index_dict['box'] = box_matrix
        index_dict['caption'] = captions_matrix
        region_dict[image_id]= index_dict

        

#https://cs.stanford.edu/people/karpathy/densecap/
region_dictionary('results_COCOtrain2014_1')
region_dictionary('results_COCOtrain2014_2')
region_dictionary('results_COCOval2014')


f = open('../data/all_regions_dict.pickle', 'wb')
pickle.dump(region_dict, f)
f.close()





