import json
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer('\w+')
# A ``RegexpTokenizer`` splits a string into substrings using a regular expression.
""" To preprocess the captions with basic tokenization by converting all sentences 
to lower case, throwing away non-alphanumeric characters we follow the following paper: 
    
A. Karpathy and L. Fei-Fei. Deep visual-semantic alignments for generating image
descriptions. In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 3128-3137, 2015.
"""
import pickle

def make_count_dict(data_type: str, count_dict: dict) -> dict:    
    with open('../data/annotations/captions_' + 
              data_type + '.json') as json_file:  
        captions = json.load(json_file)
    for i in range(len(captions['annotations'])):
        sentence = captions['annotations'][i]['caption']
        temp_tokens = []
        tokens = tokenizer.tokenize(sentence)
        tokens = [w.lower() for w in tokens]
        temp_tokens.append(tokens)
        for w in tokens:
            if not w in count_dict:   
                count_dict[w] = 1
            else:
                count_dict[w] += 1
    return count_dict

def make_vocab() -> dict:
    count_dict = {}
    count_dict = make_count_dict('train2014', count_dict)
    count_dict = make_count_dict('val2014', count_dict)  
    coco_dict = {'<START>': 0, '<STOP>': 1, '<UNKNOWN>': 2}
    word_id = 3
    # filtering the words to those that occur at least 5 times in the training set:
    for w in count_dict:
        if count_dict[w] >= 5:
              coco_dict[w] = word_id
              word_id += 1   
    print("The size of the dictionary is:", len(coco_dict))   
    f = open('../data/coco_dict.pickle', 'wb')
    pickle.dump(coco_dict, f)
    f.close()
    return coco_dict

if __name__ == '__main__':
  _ = make_vocab()
