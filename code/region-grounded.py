import numpy
import json
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
from PIL import Image, ImageDraw, ImageFont
with open('/local-scratch/densecap/results_COCOtrain2014_1.json') as json_file:
    all_captions = json.load(json_file)
image_data = all_captions['results'][6000]
img_name = image_data['img_name']
captions = image_data['captions']
print(len(captions))
scores = image_data['scores']
boxes = image_data['boxes']
z = 0
images_folder = '/local-scratch/MSCOCO/images/train2014/'
im = cv2.imread(images_folder + img_name)
for i in range(len(captions)):
    if scores[i] > 1.0:
        z = z + 1
        box = boxes[i]
        cv2.rectangle(im,(int(box[0]), int(box[1])),(int(box[0] + box[2]), int(box[1] + box[3])),(0,0,0),2)
        print(((int(box[0]), int(box[1])),(int(box[0] + box[2]), int(box[1] + box[3]))))
        cv2.putText(im, captions[i],(int(box[0]), int(box[1])), font, 0.5,(255,255,255),1)
        print(captions[i])
print(z)
cv2.imshow('Features', im)
cv2.imwrite('example.png',im)
cv2.waitKey(10)
cv2.destroyAllWindows()

