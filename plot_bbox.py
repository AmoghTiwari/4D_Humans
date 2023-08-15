import os
import cv2
import numpy as np

input_image_folder =  "example_data/images2"
output_image_folder = "outputs2/example_data/images2/bboxes_vis/default"
bboxes_folder = "outputs2/example_data/images2/bboxes/default"

for idx,img_fn in enumerate(sorted(os.listdir(input_image_folder))):
    img_fp = os.path.join(input_image_folder, img_fn)
    img_bn = "_".join(img_fn.split(".")[:-1])
    img = cv2.imread(img_fp)
    img_annot = img.copy()
    detections = np.load(os.path.join(bboxes_folder, img_bn+".npy"))
    for detection in detections:
        x,y,w,h = np.round(detection).astype('int')
        img_annot = cv2.rectangle(img_annot, (x,y,x+w,y+h), (0,0,255),2)
    cv2.imwrite(os.path.join(output_image_folder, img_fn), img_annot)
    # np.save(os.path.join(output_image_folder, img_bn+".npy"), detections[idx])
