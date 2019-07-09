clc; clear all; close all; format long;
%cd /coco/code/matlab/matconvnet-1.0-beta23
%run matlab/vl_compilenn ;
addpath matlab;
run  matlab/vl_setupnn

% load the pre-trained CNN
%net = dagnn.DagNN.loadobj(load('imagenet-googlenet-dag.mat')) ;
net = dagnn.DagNN.loadobj(load('imagenet-resnet-152-dag.mat')) ;
net.conserveMemory = 0;
net.mode = 'test' ;
dataDir = '../../../data/images/';
saveDir = '../../../data/res-feat';

layer = 514; 

%res_feat(net, 'train2014', dataDir, saveDir, layer);
%res_feat(net, 'val2014', dataDir, saveDir, layer);
%res_feat(net, 'test2014', dataDir, saveDir, layer);
res_feat(net, 'test2015', dataDir, saveDir, layer);

function res_feat(net, dataType, dataDir, saveDir, layer)
    img_names = dir([dataDir dataType  '/*.jpg']);
    for i = 1 : length(img_names)
        imgId = img_names(i).name(15:end-4);
        im = imread([dataDir  dataType '/' img_names(i).name]);
        im_ = single(im) ; % note: 0-255 range
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
        im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;
        net.eval({'data', im_});
        feat = net.vars(layer).value;
        save([saveDir '/layer' num2str(layer) '/' dataType '/res_' imgId ...
            '.mat'], 'feat','-v6');
    end
end

