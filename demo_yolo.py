""" REFERENCES: 
https://github.com/mkocabas/PARE
Parts of this script are borrowed from https://github.com/mkocabas/PARE/blob/master/scripts/demo.py
"""

import cv2
import os
import os.path as osp
import numpy as np
import json
import subprocess
import shutil
from easydict import EasyDict as edict
import torch

from yolov3.yolo import YOLOv3
from multi_person_tracker import MPT

MIN_NUM_FRAMES = 0

def run_openpose(
        video_file,
        output_folder,
        staf_folder,
        vis=False,
):
    pwd = os.getcwd()

    os.chdir(staf_folder)

    render = 1 if vis else 0
    display = 2 if vis else 0
    cmd = [
        'build/examples/openpose/openpose.bin',
        '--model_pose', 'BODY_21A',
        '--tracking', '1',
        '--render_pose', str(render),
        '--video', video_file,
        '--write_json', output_folder,
        '--display', str(display)
    ]

    print('Executing', ' '.join(cmd))
    subprocess.call(cmd)
    os.chdir(pwd)

def read_posetrack_keypoints(output_folder):

    people = dict()

    for idx, result_file in enumerate(sorted(os.listdir(output_folder))):
        json_file = osp.join(output_folder, result_file)
        data = json.load(open(json_file))
        # print(idx, data)
        for person in data['people']:
            person_id = person['person_id'][0]
            joints2d  = person['pose_keypoints_2d']
            if person_id in people.keys():
                people[person_id]['joints2d'].append(joints2d)
                people[person_id]['frames'].append(idx)
            else:
                people[person_id] = {
                    'joints2d': [],
                    'frames': [],
                }
                people[person_id]['joints2d'].append(joints2d)
                people[person_id]['frames'].append(idx)

    for k in people.keys():
        people[k]['joints2d'] = np.array(people[k]['joints2d']).reshape((len(people[k]['joints2d']), -1, 3))
        people[k]['frames'] = np.array(people[k]['frames'])

    return people

def run_posetracker(video_file, staf_folder, posetrack_output_folder='/tmp', display=False):
    posetrack_output_folder = os.path.join(
        posetrack_output_folder,
        f'{os.path.basename(video_file)}_posetrack'
    )

    # run posetrack on video
    run_openpose(
        video_file,
        posetrack_output_folder,
        vis=display,
        staf_folder=staf_folder
    )

    people_dict = read_posetrack_keypoints(posetrack_output_folder)

    shutil.rmtree(posetrack_output_folder)

    return people_dict

class Tracker:
    def __init__(self) -> None:
        self.args = edict({'tracking_method': None, 'tracker_batch_size': None, 'detector': None, 'yolo_img_size': None, 'display': None})
        self.args.tracking_method = 'bbox'
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.args.tracker_batch_size = 12
        self.args.detector = 'yolo'
        self.args.yolo_img_size = 416
        self.args.display = False
        pass

    def run_tracking(self, video_file, image_folder):
        # ========= Run tracking ========= #
        if self.args.tracking_method == 'pose':
            if not os.path.isabs(video_file):
                video_file = os.path.join(os.getcwd(), video_file)
            tracking_results = run_posetracker(video_file, staf_folder=self.args.staf_dir, display=self.args.display)
        else:
            # run multi object tracker
            mot = MPT(
                device=self.device,
                batch_size=self.args.tracker_batch_size,
                display=self.args.display,
                detector_type=self.args.detector,
                output_format='dict',
                yolo_img_size=self.args.yolo_img_size,
            )
            tracking_results = mot(image_folder)

        # remove tracklets if num_frames is less than MIN_NUM_FRAMES
        for person_id in list(tracking_results.keys()):
            if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
                del tracking_results[person_id]

        return tracking_results


    def run_detector(self, image_folder):
        # run multi object tracker
        mot = MPT(
            device=self.device,
            batch_size=self.args.tracker_batch_size,
            display=self.args.display,
            detector_type=self.args.detector,
            output_format='dict',
            yolo_img_size=self.args.yolo_img_size,
        )
        bboxes = mot.detect(image_folder)
        return bboxes

input_image_folder =  "example_data/sample_images"

tracker = Tracker()
detections = tracker.run_detector(image_folder=input_image_folder)

for idx,img_path in enumerate(sorted(os.listdir(input_image_folder))):
    img_fp = os.path.join(input_image_folder, img_path)
    img = cv2.imread(img_fp)
    img_annot = img.copy()

    for detection in detections[idx]:
        x,y,w,h = np.round(detection).astype('int')
        img_annot = cv2.rectangle(img_annot, (x,y,x+w,y+h), (0,0,255),2)
    cv2.imwrite("bbox_img.png", img_annot)
