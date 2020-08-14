# Automatic-Image-Caption-Generation


**Image Caption Generation With Hierarchical Contextual Visual Spatial Attention**

  This Python implementation contains different ICG models mentioned in the following [paper](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w39/html/Khademi_Image_Caption_Generation_CVPR_2018_paper.html):

Image Caption Generation with Hierarchical Contextual Visual Spatial Attention
  By Mahmoud Khademi, Oliver Oschulte; The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2018, pp. 1943-1951



**License**

Automatic-Image-Caption-Generation is released under the MIT license.




**Citing Image Caption Generation**

If you find "Image Caption Generation With Hierarchical Contextual Visual Spatial Attention" useful in your research, please consider citing:

      @inproceedings{khademi2018image,
        title={Image Caption Generation with Hierarchical Contextual Visual Spatial Attention},
        author={Khademi, Mahmoud and Schulte, Oliver},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
        pages={1943--1951},
        year={2018}
      }


**Requirements**

You must have tensorflow and nltk packages installed.
All the codes are written in Python 3.

**Installation**

1- clone the Automatic-Image-Caption-Generation repository

      git clone --recursive https://github.com/khademi/Automatic-Image-Caption-Generation.git

2- Download the training, validation,test data annotations and images from:

      http://cocodataset.org/#download

3- Extract all of the annotation zip files into a directory named annotations put in to the data folder.

4- Extract all of the image zip files into a directory named images and put in to the data folder.

5- from http://www.vlfeat.org/matconvnet/ download the folder to the code folder and move res-feat.m to this folder. 

**Prepration**

1- Go to the code directory

2- Run create_vocab.py

      python create_vocab.py
  * This will create a dictionary of all the words in the data folder.

3- Go to the matlab directory and run res-feat.m

4- Run create_tfrecord.py

      python create_tfrecord.py
  * This will create a tf_record file.

**Training**

Run tf-icg-train.py

      python tf-icg-train.py

  * You can choose which models you want to train your data with in the hyperparameters section of the code. Currently there are three different models which you can pick:
    <p> icg_model</p>
    <p> att_icg</p>
    <p> grid_icg</p>

  * This step trains the model and saves the model attributes into the model folder.


**Testing**

  Run tf-icg-test.py

      python tf-icg-test.py

  * Similiar to training part, you can select the model in the hyperparameters.
  * This code outputs a caption for each image in the test images and saves them as a json file in /data/results.
  * For the att_icg and grid_icg models, you can run icg-vis.py and visualize where the attentions at each step are in the actual image for each image number.  
  
 Another way to evaluate your model is to download codes from [coco-caption]( https://github.com/tylin/cococaption/tree/master/pycocoevalcap) and replace the eval.py with the eval.py provides in the code folder
       
  
  
**Directory Tree**  
.  
├── code  
│   ├── create_vocab.py  
│   ├── coco-caption-master  
│   │   └── eval.py  
│   ├── icg-vis.ipynb  
│   ├── icg-vis.py  
│   ├── models.py  
│   ├── README.md  
│   ├── region-grounded.py  
│   ├── resnet_feat.m  
│   ├── create_tfrecord.py  
│   ├── tf-icg-test.py  
│   ├── tf-icg-train.py  
├── data  
│   ├── all_regions_dict.pickle  
│   ├── annotations  
│   ├── coco_dict.pickle  
│   ├── coco_train_val_480_514.tfrecords  
│   ├── coco-vc  
│   ├── densecap  
│   ├── images  
│   ├── res-feat  
│   └── results   
└── model  
    ├── att_icg  
    ├── grid_icg  
    ├── icg  
    └── icg_deep  

13 directories, 26 files

